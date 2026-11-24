"""Generate candidate evaluation questions from test sheets with an LLM.

Without `--call`, the script only counts calls and estimates tokens (dry run). With `--call`,
it queries the API for the prompts that are not cached yet, then rebuilds
`eval/questions_candidates.jsonl` from the cache.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, results_dir
from src.env import load_env
from src.evaluation.gemini import GeminiClient, QuotaExhausted, response_text
from src.evaluation.split import CHUNKS, SPLIT, normalize_body
from src.ingest.explore import describe

CONFIG = Path("configs/questions.yaml")
CANDIDATES = Path("eval/questions_candidates.jsonl")
EXCLUDED = Path("eval/excluded_source_chunks.json")
CACHE_DIR = Path("cache/llm")
CHARS_PER_TOKEN = 4  # rough rule of thumb, used for the dry-run estimate only
OUTPUT_TOKENS_PER_QUESTION = 80

PROMPT = """Tu prépares un jeu d'évaluation pour un moteur de recherche sur les démarches \
administratives françaises. Voici des extraits d'une fiche de Service-Public.fr, chacun précédé \
de son identifiant entre crochets.

Écris {n} questions qu'un particulier pourrait poser, en respectant ces règles :
- chaque question porte sur un seul extrait, dont tu donnes l'identifiant ;
- la réponse se trouve entièrement dans cet extrait ;
- formule la question avec tes propres mots, comme un usager qui ne connaît pas le vocabulaire \
administratif ; ne recopie pas les phrases de l'extrait ;
- la question doit se comprendre seule, sans voir la fiche : nomme la démarche ou la situation ;
- utilise des extraits différents quand c'est possible.

Réponds uniquement par un tableau JSON de la forme :
[{{"chunk_id": "...", "question": "...", "answer": "réponse courte tirée de l'extrait"}}]

Fiche : {title}

Extraits :
{excerpts}"""


def sample_fiches(test_ids: list[str], n: int, seed: int) -> list[str]:
    """Pick `n` test sheets at random, in a reproducible order."""
    ids = sorted(test_ids)
    rng = np.random.default_rng(seed)
    return [ids[i] for i in rng.permutation(len(ids))[:n]]


def pick_chunks(fiche_chunks: pd.DataFrame, k: int, min_words: int, seed: int) -> pd.DataFrame:
    """Up to `k` chunks of one sheet, long enough and with distinct bodies, in document order."""
    eligible = fiche_chunks[fiche_chunks["n_words"] >= min_words]
    eligible = eligible[~eligible["body"].map(normalize_body).duplicated()]
    rng = np.random.default_rng(seed)
    keep = sorted(rng.permutation(len(eligible))[:k])
    return eligible.iloc[keep]


def build_prompt(title: str, chunks: pd.DataFrame, n_questions: int) -> str:
    """Fill the prompt template with the sheet title and its selected chunks."""
    excerpts = "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in chunks.itertuples())
    return PROMPT.format(n=n_questions, title=title, excerpts=excerpts)


def parse_questions(text: str, allowed_ids: set[str]) -> tuple[list[dict], Counter]:
    """Keep well-formed items that point to one of the provided chunks; count the others."""
    dropped: Counter = Counter()
    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        dropped["invalid_json"] += 1
        return [], dropped
    if not isinstance(items, list):
        dropped["not_a_list"] += 1
        return [], dropped
    kept = []
    for item in items:
        if not isinstance(item, dict) or not str(item.get("question", "")).strip():
            dropped["malformed_item"] += 1
        elif item.get("chunk_id") not in allowed_ids:
            dropped["unknown_chunk_id"] += 1
        else:
            kept.append(
                {
                    "chunk_id": item["chunk_id"],
                    "question": str(item["question"]).strip(),
                    "answer": str(item.get("answer", "")).strip(),
                }
            )
    return kept, dropped


def content_words(text: str) -> list[str]:
    """Lowercased words of at least 4 letters."""
    return [w for w in re.findall(r"\w+", text.lower()) if len(w) >= 4]


def word_overlap(question: str, body: str) -> float:
    """Share of the question's content words that also appear in the chunk body."""
    words = content_words(question)
    if not words:
        return 0.0
    body_words = set(content_words(body))
    return sum(w in body_words for w in words) / len(words)


def relevant_index(chunks: pd.DataFrame) -> dict[str, list[str]]:
    """Map each chunk id to every chunk id of the corpus with the same normalised body."""
    bodies = chunks["body"].map(normalize_body)
    by_body = chunks.groupby(bodies)["chunk_id"].apply(sorted).to_dict()
    return dict(zip(chunks["chunk_id"], bodies.map(by_body), strict=True))


def plan_prompts(
    chunks: pd.DataFrame, test_ids: list[str], config: dict, excluded: frozenset[str] = frozenset()
) -> list[dict]:
    """One prompt per sampled test sheet that has at least one eligible, non-excluded chunk."""
    plans = []
    sources = chunks[~chunks["chunk_id"].isin(excluded)]
    by_fiche = dict(tuple(sources.groupby("fiche_id")))
    for i, fid in enumerate(sample_fiches(test_ids, config["n_fiches"], config["seed"])):
        if fid not in by_fiche:
            continue
        picked = pick_chunks(
            by_fiche[fid], config["chunks_per_fiche"], config["min_chunk_words"], config["seed"] + i
        )
        if picked.empty:
            continue
        title = picked["title"].iloc[0]
        prompt = build_prompt(title, picked, config["questions_per_fiche"])
        plans.append({"fiche_id": fid, "chunk_ids": list(picked["chunk_id"]), "prompt": prompt})
    return plans


def generation_config(config: dict) -> dict:
    return {
        "temperature": config["temperature"],
        "seed": config["seed"],
        "responseMimeType": "application/json",
    }


def call_budget(config: dict) -> int:
    """Calls allowed in one run: the per-run cap, never above the free-tier daily quota."""
    return min(config["max_calls_per_run"], config["requests_per_day"])


def estimate(plans: list[dict], client: GeminiClient, config: dict) -> dict:
    """Calls still to make and a rough token estimate, before spending any quota."""
    gen = generation_config(config)
    todo = [p for p in plans if not client.is_cached(p["prompt"], gen)]
    in_tokens = sum(len(p["prompt"]) for p in todo) // CHARS_PER_TOKEN
    out_tokens = len(todo) * config["questions_per_fiche"] * OUTPUT_TOKENS_PER_QUESTION
    return {
        "model": config["model"],
        "prompts": len(plans),
        "cached": len(plans) - len(todo),
        "calls_to_make": len(todo),
        "calls_this_run": min(len(todo), call_budget(config)),
        "requests_per_minute": config["requests_per_minute"],
        "requests_per_day": config["requests_per_day"],
        "estimated_input_tokens": in_tokens,
        "estimated_output_tokens": out_tokens,
        "estimated_minutes_at_rate_limit": round(len(todo) / config["requests_per_minute"], 1),
        "estimated_tokens_per_minute": round(
            (in_tokens + out_tokens) / max(len(todo), 1) * config["requests_per_minute"]
        ),
    }


def build_candidates(
    plans: list[dict], client: GeminiClient, chunks: pd.DataFrame, config: dict
) -> tuple[list[dict], dict]:
    """Turn cached responses into candidate records; sheets without a response are skipped."""
    gen = generation_config(config)
    info = chunks.set_index("chunk_id")
    relevant = relevant_index(chunks)
    candidates, dropped, usage = [], Counter(), Counter()
    missing = 0
    for plan in plans:
        if not client.is_cached(plan["prompt"], gen):
            missing += 1
            continue
        record = client.generate(plan["prompt"], gen)
        usage.update(record["response"].get("usageMetadata", {}))
        items, bad = parse_questions(response_text(record["response"]), set(plan["chunk_ids"]))
        dropped.update(bad)
        for k, item in enumerate(items):
            chunk = info.loc[item["chunk_id"]]
            overlap = word_overlap(item["question"], chunk["body"])
            candidates.append(
                {
                    "id": f"{plan['fiche_id']}-q{k}",
                    "fiche_id": plan["fiche_id"],
                    "theme": chunk["theme"],
                    "source_chunk_id": item["chunk_id"],
                    "relevant_chunk_ids": relevant[item["chunk_id"]],
                    "question": item["question"],
                    "answer": item["answer"],
                    "word_overlap": round(overlap, 3),
                    "flag_overlap": overlap >= config["overlap_flag"],
                    "model": config["model"],
                    "prompt_version": config["prompt_version"],
                }
            )
    stats = {
        "sheets_planned": len(plans),
        "sheets_without_response": missing,
        "candidates": len(candidates),
        "dropped_items": dict(dropped),
        "multi_relevant": sum(len(c["relevant_chunk_ids"]) > 1 for c in candidates),
        "flag_overlap": sum(c["flag_overlap"] for c in candidates),
        "word_overlap": describe([c["word_overlap"] for c in candidates]) if candidates else {},
        "candidates_by_theme": dict(Counter(c["theme"] for c in candidates).most_common()),
        "usage_tokens": {k: v for k, v in usage.items() if isinstance(v, int)},
    }
    return candidates, stats


def call_missing(plans: list[dict], client: GeminiClient, config: dict) -> None:
    """Query the API for uncached prompts, up to the call budget, stopping on quota errors."""
    gen = generation_config(config)
    todo = [p for p in plans if not client.is_cached(p["prompt"], gen)]
    budget = call_budget(config)
    for n, plan in enumerate(todo[:budget], start=1):
        try:
            client.generate(plan["prompt"], gen)
        except QuotaExhausted:
            print(f"Quota reached after {n - 1} new responses; rerun later to resume")
            return
        print(f"[{n}/{min(len(todo), budget)}] {plan['fiche_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--call", action="store_true", help="actually call the API")
    args = parser.parse_args()
    config = load_config(args.config)
    load_env()
    chunks = pd.read_parquet(CHUNKS)
    test_ids = json.loads(SPLIT.read_text())["test"]
    excluded = frozenset(json.loads(EXCLUDED.read_text())["chunk_ids"])
    plans = plan_prompts(chunks, test_ids, config, excluded)
    api_key = os.environ.get("GEMINI_API_KEY", "")
    client = GeminiClient(config["model"], api_key, CACHE_DIR, config["requests_per_minute"])
    out = results_dir(config, args.config)

    est = estimate(plans, client, config)
    in_test = chunks["fiche_id"].isin(set(test_ids))
    est["excluded_source_chunks_in_test"] = int(
        chunks.loc[in_test, "chunk_id"].isin(excluded).sum()
    )
    (out / "estimate.json").write_text(json.dumps(est, indent=2) + "\n")
    print(json.dumps(est, indent=2))
    if args.call:
        if not api_key:
            raise SystemExit("GEMINI_API_KEY is missing from the environment or .env")
        call_missing(plans, client, config)

    candidates, stats = build_candidates(plans, client, chunks, config)
    if not candidates:
        print("No cached response yet; nothing written")
        return
    lines = [json.dumps(c, ensure_ascii=False) for c in candidates]
    CANDIDATES.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "generation_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"Wrote {CANDIDATES} ({len(candidates)} candidates)")


if __name__ == "__main__":
    main()
