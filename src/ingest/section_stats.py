"""Section length statistics and chunk counts under candidate chunking strategies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.ingest.chunk import split_section
from src.ingest.explore import describe

FICHES = Path("data/processed/fiches.parquet")
OUTPUT = Path("results/section_stats.json")
MAX_WORDS = (150, 250, 400)
WINDOW, STRIDE = 200, 150


def pack_lines(text: str, max_words: int) -> list[int]:
    """Word count of each piece produced by the section splitter."""
    return [len(piece.split()) for piece in split_section(text, max_words)]


def sliding_windows(n_words: int, window: int = WINDOW, stride: int = STRIDE) -> list[int]:
    """Word counts of fixed windows over a whole sheet, ignoring its structure."""
    if n_words <= window:
        return [n_words]
    starts = range(0, n_words - window + stride, stride)
    return [min(window, n_words - s) for s in starts if s < n_words]


def header_words(title: str, path: list[str]) -> int:
    """Words of the 'title > path' context header a chunk would carry."""
    return len(" ".join([title, *path]).split())


def compute(fiches: pd.DataFrame) -> dict[str, object]:
    """Section statistics plus the chunk distribution of each candidate strategy."""
    section_words, headers, sheet_words = [], [], []
    texts = []
    for row in fiches.itertuples():
        total = 0
        for sec in row.sections:
            n = len(sec["text"].split())
            section_words.append(n)
            headers.append(header_words(row.title, list(sec["path"])))
            texts.append(sec["text"])
            total += n
        sheet_words.append(total)

    report: dict[str, object] = {
        "fiches": len(fiches),
        "sections": len(section_words),
        "section_words": describe(section_words),
        "header_words": describe(headers),
        "sections_below_20_words": sum(n < 20 for n in section_words),
        "sections_above": {str(m): sum(n > m for n in section_words) for m in MAX_WORDS},
        "strategies": {},
    }
    report["strategies"]["one_chunk_per_section"] = describe(section_words)
    for m in MAX_WORDS:
        pieces = [p for text in texts for p in pack_lines(text, m)]
        report["strategies"][f"section_split_max_{m}"] = describe(pieces)
    windows = [w for n in sheet_words for w in sliding_windows(n)]
    report["strategies"][f"fixed_window_{WINDOW}_stride_{STRIDE}"] = describe(windows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fiches", type=Path, default=FICHES)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = compute(pd.read_parquet(args.fiches))
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
