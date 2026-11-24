"""Parse the in-scope sheets of a snapshot into data/processed/fiches.parquet."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd
from lxml import etree

from src.ingest.explore import latest_snapshot
from src.ingest.parse_xml import parse_fiche

# Corpus scope, see docs/decisions.md (D1).
KEPT_TYPES = ("Fiche d'information conditionnée", "Fiche Question-réponse conditionnée")
OUTPUT = Path("data/processed/fiches.parquet")
REPORT = Path("results/parse_report.json")


def sheet_type(path: Path) -> str | None:
    """Read the `type` attribute of <Publication> without parsing the whole body."""
    try:
        for _, el in etree.iterparse(str(path), events=("start",), tag="Publication"):
            return el.get("type")
    except etree.XMLSyntaxError:
        return None
    return None


def build_corpus(snapshot: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Parse every in-scope F sheet; return the table and a report of what was dropped."""
    files = sorted((snapshot / "xml").glob("F*.xml"))
    out_of_scope: Counter[str] = Counter()
    failed: list[str] = []
    rows = []
    for path in files:
        kind = sheet_type(path)
        if kind not in KEPT_TYPES:
            out_of_scope[str(kind)] += 1
            continue
        fiche = parse_fiche(path)
        if fiche is None:
            failed.append(path.stem)
        else:
            rows.append(fiche)
    table = pd.DataFrame(rows)
    report = {
        "snapshot": snapshot.name,
        "kept_types": list(KEPT_TYPES),
        "f_files": len(files),
        "out_of_scope_by_type": dict(out_of_scope),
        "parse_failed_or_empty": failed,
        "fiches_kept": len(table),
        "sections_kept": int(table["sections"].map(len).sum()) if len(table) else 0,
    }
    return table, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", nargs="?", type=Path, help="Snapshot folder (default: latest)")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    table, report = build_corpus(args.snapshot or latest_snapshot())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output, index=False)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {args.output} ({report['fiches_kept']} sheets) and {REPORT}")


if __name__ == "__main__":
    main()
