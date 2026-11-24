"""Descriptive statistics of a raw snapshot, used to choose the corpus subset."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from lxml import etree

RAW_DIR = Path("data/raw")
OUTPUT = Path("results/corpus_stats_raw.json")
BODY_TAGS = ("Introduction", "Texte", "ListeSituations", "Conclusion")
NO_THEME = "(no theme)"


def latest_snapshot(raw_dir: Path = RAW_DIR) -> Path:
    """Return the most recent dated snapshot folder."""
    snapshots = sorted(p for p in raw_dir.iterdir() if (p / "manifest.json").exists())
    if not snapshots:
        raise FileNotFoundError(f"No snapshot in {raw_dir}; run `make data` first")
    return snapshots[-1]


def file_kind(path: Path) -> str:
    """Classify a file by its ID prefix: F (sheet), N (folder/theme), R (resource) or other."""
    match = re.match(r"([FNR])\d+\.xml$", path.name)
    return match.group(1) if match else "other"


def body_blocks(root: etree._Element) -> list[etree._Element]:
    """Return the top-level elements that hold the readable body of a sheet."""
    return [block for tag in BODY_TAGS for block in root.findall(tag)]


def body_words(root: etree._Element) -> int:
    """Count whitespace-separated words in the body blocks of a sheet."""
    return sum(len(" ".join(block.itertext()).split()) for block in body_blocks(root))


def body_tag_counts(root: etree._Element) -> Counter[str]:
    """Count element names found inside the body blocks."""
    counts: Counter[str] = Counter()
    for block in body_blocks(root):
        counts.update(etree.QName(el).localname for el in block.iter() if isinstance(el.tag, str))
    return counts


def sheet_record(root: etree._Element) -> dict[str, object]:
    """Extract the per-sheet fields needed for the statistics."""
    theme = root.findtext("Theme/Titre") or NO_THEME
    return {
        "id": root.get("ID"),
        "type": root.get("type"),
        "theme": theme.strip(),
        "words": body_words(root),
        "n_chapitre": sum(len(b.findall(".//Chapitre")) for b in body_blocks(root)),
        "n_situation": len(root.findall("ListeSituations/Situation")),
        "schema": root.get("{http://www.w3.org/2001/XMLSchema-instance}noNamespaceSchemaLocation"),
    }


def describe(values: list[int]) -> dict[str, float | int]:
    """Summary statistics of an integer distribution."""
    arr = np.asarray(values)
    return {
        "n": int(arr.size),
        "sum": int(arr.sum()),
        "min": int(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "median": float(np.median(arr)),
        "mean": round(float(arr.mean()), 1),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(arr.max()),
    }


def collect(snapshot: Path) -> dict[str, object]:
    """Parse every file of a snapshot and aggregate the statistics."""
    files = sorted((snapshot / "xml").glob("*.xml"))
    kinds = Counter(file_kind(p) for p in files)
    sheets, unreadable, tags = [], [], Counter()
    schemas: Counter[str] = Counter()
    for path in files:
        if file_kind(path) != "F":
            continue
        try:
            root = etree.parse(str(path)).getroot()
        except etree.XMLSyntaxError as err:
            unreadable.append({"file": path.name, "error": str(err)})
            continue
        record = sheet_record(root)
        schemas[str(record.pop("schema"))] += 1
        sheets.append(record)
        tags.update(body_tag_counts(root))
    return {
        "kinds": kinds,
        "sheets": sheets,
        "unreadable": unreadable,
        "tags": tags,
        "schemas": schemas,
    }


def summarize(snapshot: Path, data: dict) -> dict[str, object]:
    """Build the JSON report from collected data."""
    sheets = data["sheets"]
    by_theme: dict[str, list[int]] = defaultdict(list)
    by_type: dict[str, list[int]] = defaultdict(list)
    for s in sheets:
        by_theme[s["theme"]].append(s["words"])
        by_type[s["type"]].append(s["words"])
    themes = sorted(by_theme.items(), key=lambda kv: -len(kv[1]))
    return {
        "snapshot": snapshot.name,
        "manifest": json.loads((snapshot / "manifest.json").read_text()),
        "schema_location_in_sheets": dict(data["schemas"]),
        "files_by_kind": dict(data["kinds"]),
        "sheets_parsed": len(sheets),
        "sheets_unreadable": data["unreadable"],
        "sheets_without_body_text": [s["id"] for s in sheets if s["words"] == 0],
        "words_per_sheet": describe([s["words"] for s in sheets]),
        "chapitres_per_sheet": describe([s["n_chapitre"] for s in sheets]),
        "situations_per_sheet": describe([s["n_situation"] for s in sheets]),
        "sheets_by_type": {t: describe(w) for t, w in by_type.items()},
        "sheets_by_theme": {t: describe(w) for t, w in themes},
        "sheets_by_theme_and_type": {
            theme: dict(Counter(s["type"] for s in sheets if s["theme"] == theme))
            for theme, _ in themes
        },
        "body_tag_inventory": dict(data["tags"].most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", nargs="?", type=Path, help="Snapshot folder (default: latest)")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    snapshot = args.snapshot or latest_snapshot()
    report = summarize(snapshot, collect(snapshot))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
