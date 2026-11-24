"""Terminal helper to review candidate questions one by one.

Each decision is appended to `eval/review_decisions.jsonl` (the last decision for an id wins),
so a review can be stopped and resumed at any time. The validated set itself is written by
hand from these decisions. `--stats` writes acceptance and rejection counts to `results/`.

With `--sample N --seed S`, only a random sample of the candidates is reviewed; the sample is
frozen in `eval/review_sample.json`. `--accept-rest` then marks every candidate without a
decision as "auto_accepted", so that reviewed and unreviewed questions stay distinguishable.
The rejection rate is computed on the human decisions of the sample only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.generate_questions import CANDIDATES
from src.evaluation.split import CHUNKS

DECISIONS = Path("eval/review_decisions.jsonl")
STATS = Path("results/review/review_stats.json")
SAMPLE = Path("eval/review_sample.json")
HUMAN_STATUSES = {"accepted", "edited", "rejected"}
AUTO_ACCEPTED = "auto_accepted"
REJECT_REASONS = {
    "c": "copied_wording",
    "w": "answer_not_in_chunk",
    "v": "vague_or_not_standalone",
    "d": "duplicate",
    "o": "other",
}
HELP = "[a] accept  [e] edit question  [k] narrow relevant chunks  [r] reject  [s] skip  [q] quit"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def latest_decisions(decisions: list[dict]) -> dict[str, dict]:
    """Last decision recorded for each candidate id."""
    return {d["id"]: d for d in decisions}


def make_decision(
    candidate: dict,
    status: str,
    question: str | None = None,
    relevant: list[str] | None = None,
    reason: str | None = None,
) -> dict:
    """One review record: 'accepted', 'edited' or 'rejected' by hand, or 'auto_accepted'."""
    return {
        "id": candidate["id"],
        "status": status,
        "question": question or candidate["question"],
        "relevant_chunk_ids": relevant or candidate["relevant_chunk_ids"],
        "reason": reason,
        "date": date.today().isoformat(),
    }


def draw_sample(candidates: list[dict], n: int, seed: int) -> list[str]:
    """Ids of `n` candidates drawn at random, returned in file order."""
    ids = sorted(c["id"] for c in candidates)
    picked = {ids[i] for i in np.random.default_rng(seed).permutation(len(ids))[:n]}
    return [c["id"] for c in candidates if c["id"] in picked]


def frozen_sample(candidates: list[dict], n: int, seed: int, path: Path = SAMPLE) -> list[str]:
    """Draw the sample once and store it; refuse a different sample later."""
    record = {"n": n, "seed": seed, "ids": draw_sample(candidates, n, seed)}
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise SystemExit(f"{path} holds a different sample; not touching it")
    else:
        path.write_text(json.dumps(record, indent=1) + "\n")
    return record["ids"]


def human_counts(decisions: list[dict], flagged: set[str]) -> dict:
    """Status, rejection rate and reasons over a list of human decisions."""
    status = Counter(d["status"] for d in decisions)
    return {
        "reviewed": len(decisions),
        "by_status": dict(status),
        "rejection_rate": round(status["rejected"] / len(decisions), 3) if decisions else None,
        "reject_reasons": dict(Counter(d["reason"] for d in decisions if d["reason"])),
        "overlap_flagged_reviewed": sum(d["id"] in flagged for d in decisions),
        "rejected_among_overlap_flagged": sum(
            d["status"] == "rejected" for d in decisions if d["id"] in flagged
        ),
    }


def review_stats(
    candidates: list[dict], decisions: dict[str, dict], sample: dict | None = None
) -> dict:
    """Overall status counts; rejection statistics on the sample when there is one."""
    flagged = {c["id"] for c in candidates if c.get("flag_overlap")}
    decided = [decisions[c["id"]] for c in candidates if c["id"] in decisions]
    human = [d for d in decided if d["status"] in HUMAN_STATUSES]
    stats = {
        "candidates": len(candidates),
        "by_status": dict(Counter(d["status"] for d in decided)),
        "undecided": len(candidates) - len(decided),
    }
    if sample is None:
        return {**stats, "human_review": human_counts(human, flagged)}
    in_sample = set(sample["ids"])
    sampled = [d for d in human if d["id"] in in_sample]
    return {
        **stats,
        "sample": {
            "size": len(in_sample),
            "seed": sample["seed"],
            "complete": len(sampled) == len(in_sample),
            **human_counts(sampled, flagged),
        },
        "human_reviewed_outside_sample": len(human) - len(sampled),
    }


def auto_accept_rest(
    candidates: list[dict], decisions: dict[str, dict], sample_ids: list[str]
) -> list[dict]:
    """'auto_accepted' records for undecided candidates, once the sample is fully reviewed."""
    missing = [i for i in sample_ids if decisions.get(i, {}).get("status") not in HUMAN_STATUSES]
    if missing:
        raise SystemExit(f"{len(missing)} sampled candidates are not reviewed yet")
    return [make_decision(c, AUTO_ACCEPTED) for c in candidates if c["id"] not in decisions]


def show(candidate: dict, chunks: pd.DataFrame, done: int, total: int) -> None:
    """Print a candidate with its source chunk and the list of relevant chunks."""
    source = chunks.loc[candidate["source_chunk_id"]]
    flag = "  [OVERLAP]" if candidate["flag_overlap"] else ""
    print("\n" + "=" * 80)
    print(f"{done + 1}/{total}  {candidate['id']}  {candidate['theme']}{flag}")
    print(f"overlap={candidate['word_overlap']}")
    print(f"\nQ: {candidate['question']}\nA: {candidate['answer']}\n")
    print(f"--- {candidate['source_chunk_id']}\n{source['text']}\n")
    print("Relevant chunks:")
    for i, cid in enumerate(candidate["relevant_chunk_ids"]):
        print(f"  {i}. {cid}  {chunks.loc[cid, 'header']}")


def ask(candidate: dict, read: Callable[[str], str]) -> dict | None:
    """Prompt for one decision; return None to skip, raise StopIteration to quit."""
    while True:
        cmd = read(f"{HELP}\n> ").strip().lower()
        if cmd == "a":
            return make_decision(candidate, "accepted")
        if cmd == "e":
            new = read("New question: ").strip()
            if new:
                return make_decision(candidate, "edited", question=new)
        elif cmd == "k":
            picks = read("Indices to keep (e.g. 0,2): ").replace(" ", "").split(",")
            ids = candidate["relevant_chunk_ids"]
            keep = [ids[int(i)] for i in picks if i.isdigit() and int(i) < len(ids)]
            if keep:
                return make_decision(candidate, "edited", relevant=keep)
        elif cmd == "r":
            codes = " ".join(f"[{k}] {v}" for k, v in REJECT_REASONS.items())
            code = read(f"{codes}\n> ").strip().lower()
            if code in REJECT_REASONS:
                return make_decision(candidate, "rejected", reason=REJECT_REASONS[code])
        elif cmd == "s":
            return None
        elif cmd == "q":
            raise StopIteration


def append_decisions(decisions: list[dict], path: Path = DECISIONS) -> None:
    with path.open("a", encoding="utf-8") as f:
        for decision in decisions:
            f.write(json.dumps(decision, ensure_ascii=False) + "\n")


def run_review(candidates: list[dict], chunks: pd.DataFrame) -> None:
    """Review every given candidate that has no decision yet, appending each decision to disk."""
    decided = latest_decisions(read_jsonl(DECISIONS))
    todo = [c for c in candidates if c["id"] not in decided]
    done = len(candidates) - len(todo)
    print(f"{done} already reviewed, {len(todo)} to go")
    for i, candidate in enumerate(todo):
        show(candidate, chunks, done + i, len(candidates))
        try:
            decision = ask(candidate, input)
        except (StopIteration, EOFError, KeyboardInterrupt):
            print("\nStopped; decisions so far are saved")
            return
        if decision:
            append_decisions([decision])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", action="store_true", help="write review statistics and exit")
    parser.add_argument("--sample", type=int, help="review only a random sample of this size")
    parser.add_argument("--seed", type=int, default=42, help="seed of the sample")
    parser.add_argument(
        "--accept-rest", action="store_true", help="mark undecided candidates as auto_accepted"
    )
    args = parser.parse_args()
    candidates = read_jsonl(CANDIDATES)
    if not candidates:
        raise SystemExit(f"No candidates in {CANDIDATES}")
    decisions = latest_decisions(read_jsonl(DECISIONS))
    sample = json.loads(SAMPLE.read_text()) if SAMPLE.exists() else None
    if args.stats:
        stats = review_stats(candidates, decisions, sample)
        STATS.parent.mkdir(parents=True, exist_ok=True)
        STATS.write_text(json.dumps(stats, indent=2) + "\n")
        print(json.dumps(stats, indent=2))
        return
    if args.accept_rest:
        if sample is None:
            raise SystemExit(f"No sample in {SAMPLE}; review one with --sample first")
        records = auto_accept_rest(candidates, decisions, sample["ids"])
        append_decisions(records)
        print(f"Marked {len(records)} candidates as {AUTO_ACCEPTED}")
        return
    if args.sample is not None:
        sample_ids = set(frozen_sample(candidates, args.sample, args.seed))
        candidates = [c for c in candidates if c["id"] in sample_ids]
    chunks = pd.read_parquet(CHUNKS, columns=["chunk_id", "header", "text"]).set_index("chunk_id")
    run_review(candidates, chunks)


if __name__ == "__main__":
    main()
