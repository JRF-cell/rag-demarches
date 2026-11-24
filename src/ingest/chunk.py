"""Split parsed sheets into retrieval chunks: one section, cut at line boundaries to a max size."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src.ingest.explore import describe

# Chunking parameters, see docs/decisions.md (D2).
MAX_WORDS = 250
HEADER_SEP = " > "

FICHES = Path("data/processed/fiches.parquet")
CHUNKS = Path("data/processed/chunks.parquet")
STATS = Path("results/corpus_stats.json")


def split_long_line(line: str, max_words: int) -> list[str]:
    """Cut a line longer than `max_words` into consecutive slices of `max_words` words."""
    words = line.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def split_section(text: str, max_words: int = MAX_WORDS) -> list[str]:
    """Greedily pack whole lines into pieces of at most `max_words` words, without overlap."""
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.split("\n"):
        for part in split_long_line(line, max_words):
            n = len(part.split())
            if current and size + n > max_words:
                pieces.append("\n".join(current))
                current, size = [], 0
            current.append(part)
            size += n
    if current:
        pieces.append("\n".join(current))
    return pieces


def make_header(title: str, path: list[str]) -> str:
    """Context header 'sheet title > section path' prepended to every chunk."""
    return HEADER_SEP.join([title, *path])


def fiche_chunks(fiche: dict, max_words: int = MAX_WORDS) -> list[dict[str, object]]:
    """All chunks of one parsed sheet, numbered in document order."""
    chunks = []
    for section_index, sec in enumerate(fiche["sections"]):
        path = list(sec["path"])
        header = make_header(fiche["title"], path)
        for body in split_section(sec["text"], max_words):
            chunks.append(
                {
                    "chunk_id": f"{fiche['id']}-{len(chunks):03d}",
                    "fiche_id": fiche["id"],
                    "section_index": section_index,
                    "path": path,
                    "header": header,
                    "body": body,
                    "text": f"{header}\n{body}",
                    "n_words": len(body.split()),
                    "title": fiche["title"],
                    "theme": fiche["theme"],
                    "url": fiche["url"],
                    "modified": fiche["modified"],
                }
            )
    return chunks


def build_chunks(fiches: pd.DataFrame, max_words: int = MAX_WORDS) -> pd.DataFrame:
    """Chunk every sheet of the table."""
    rows = [c for fiche in fiches.to_dict("records") for c in fiche_chunks(fiche, max_words)]
    return pd.DataFrame(rows)


def corpus_stats(fiches: pd.DataFrame, chunks: pd.DataFrame) -> dict[str, object]:
    """Corpus-level statistics of the final chunk table."""
    per_fiche = chunks.groupby("fiche_id").size()
    with_header = chunks["text"].str.split().str.len()
    return {
        "max_words": MAX_WORDS,
        "fiches": len(fiches),
        "sections": int(fiches["sections"].map(len).sum()),
        "chunks": len(chunks),
        "chunk_body_words": describe(chunks["n_words"].tolist()),
        "chunk_words_with_header": describe(with_header.tolist()),
        "chunks_per_fiche": describe(per_fiche.tolist()),
        "fiches_by_theme": dict(Counter(fiches["theme"]).most_common()),
        "chunks_by_theme": dict(Counter(chunks["theme"]).most_common()),
        "fiches_by_type": dict(Counter(fiches["type"]).most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fiches", type=Path, default=FICHES)
    parser.add_argument("--output", type=Path, default=CHUNKS)
    args = parser.parse_args()
    fiches = pd.read_parquet(args.fiches)
    chunks = build_chunks(fiches)
    chunks.to_parquet(args.output, index=False)
    STATS.write_text(json.dumps(corpus_stats(fiches, chunks), indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {args.output} ({len(chunks)} chunks) and {STATS}")


if __name__ == "__main__":
    main()
