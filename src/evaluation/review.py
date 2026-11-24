"""Terminal helper to review candidate questions one by one.

Each decision is appended to `eval/review_decisions.jsonl` (the last decision for an id wins),
so a review can be stopped and resumed at any time. The validated set itself is written by
hand from these decisions. `--stats` writes acceptance and rejection counts to `results/`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pandas as pd

from src.evaluation.generate_questions import CANDIDATES
from src.evaluation.split import CHUNKS

DECISIONS = Path("eval/review_decisions.jsonl")
STATS = Path("results/review/review_stats.json")
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
    """One review record: status is 'accepted', 'edited' or 'rejected'."""
    return {
        "id": candidate["id"],
        "status": status,
        "question": question or candidate["question"],
        "relevant_chunk_ids": relevant or candidate["relevant_chunk_ids"],
        "reason": reason,
        "date": date.today().isoformat(),
    }


def review_stats(candidates: list[dict], decisions: dict[str, dict]) -> dict:
    """Counts per status and rejection reason, and the rejection rate over reviewed items."""
    reviewed = [decisions[c["id"]] for c in candidates if c["id"] in decisions]
    status = Counter(d["status"] for d in reviewed)
    flagged = {c["id"] for c in candidates if c.get("flag_overlap")}
    return {
        "candidates": len(candidates),
        "reviewed": len(reviewed),
        "by_status": dict(status),
        "rejection_rate": round(status["rejected"] / len(reviewed), 3) if reviewed else None,
        "reject_reasons": dict(Counter(d["reason"] for d in reviewed if d["reason"])),
        "rejected_among_overlap_flagged": sum(
            d["status"] == "rejected" for d in reviewed if d["id"] in flagged
        ),
        "overlap_flagged_reviewed": sum(d["id"] in flagged for d in reviewed),
    }


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


def run_review(candidates: list[dict], chunks: pd.DataFrame) -> None:
    """Review every candidate that has no decision yet, appending each decision to disk."""
    decided = latest_decisions(read_jsonl(DECISIONS))
    todo = [c for c in candidates if c["id"] not in decided]
    print(f"{len(decided)} already reviewed, {len(todo)} to go")
    for i, candidate in enumerate(todo):
        show(candidate, chunks, len(decided) + i, len(candidates))
        try:
            decision = ask(candidate, input)
        except (StopIteration, EOFError, KeyboardInterrupt):
            print("\nStopped; decisions so far are saved")
            return
        if decision:
            with DECISIONS.open("a", encoding="utf-8") as f:
                f.write(json.dumps(decision, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", action="store_true", help="write review statistics and exit")
    args = parser.parse_args()
    candidates = read_jsonl(CANDIDATES)
    if not candidates:
        raise SystemExit(f"No candidates in {CANDIDATES}")
    if args.stats:
        stats = review_stats(candidates, latest_decisions(read_jsonl(DECISIONS)))
        STATS.parent.mkdir(parents=True, exist_ok=True)
        STATS.write_text(json.dumps(stats, indent=2) + "\n")
        print(json.dumps(stats, indent=2))
        return
    chunks = pd.read_parquet(CHUNKS, columns=["chunk_id", "header", "text"]).set_index("chunk_id")
    run_review(candidates, chunks)


if __name__ == "__main__":
    main()
