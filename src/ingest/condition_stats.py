"""Measure how conditional content (<Condition>) is handled by the parser.

A <Condition> holds no text: it references questionnaire variables (`<estVrai var="T..."/>`),
whose labels live in the <Questionnaire> of some sheets. The parser skips both, so the text of
a conditioned element is kept but loses the condition under which it applies. This script
counts conditioned elements per parent tag, whether their variables can be resolved to a label,
whether their text reaches the chunks, and which chunks hold text of an untitled
conditioned element (where the condition is the only context of the text). Those chunks stay
in the index but are never used as the source of an evaluation question.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from lxml import etree

from src.ingest.explore import latest_snapshot
from src.ingest.parse_xml import clean, inline_text, render_children

FICHES = Path("data/processed/fiches.parquet")
CHUNKS = Path("data/processed/chunks.parquet")
OUTPUT = Path("results/condition_stats.json")
EXCLUDED = Path("eval/excluded_source_chunks.json")
# Parents whose title already states the case (rendered as a "<title> :" line or a path).
TITLED = {"Cas", "Situation", "Chapitre", "SousChapitre"}
MIN_LINE_WORDS = 5


def variable_labels(root: etree._Element) -> dict[str, list[str]]:
    """Map each variable set to true by a questionnaire choice to the choice titles."""
    labels: dict[str, list[str]] = defaultdict(list)
    for choice in root.iter("Choix"):
        title = choice.find("Titre")
        text = inline_text(title) if title is not None else ""
        for assign in choice.iter("affecteVrai"):
            labels[assign.get("var")].append(text)
    return labels


def condition_vars(condition: etree._Element) -> list[str]:
    return [e.get("var") for e in condition.iter() if e.get("var")]


def corpus_labels(xml_dir: Path) -> dict[str, list[str]]:
    """Variable labels collected from every questionnaire of the snapshot (all file types)."""
    labels: dict[str, list[str]] = defaultdict(list)
    for path in sorted(xml_dir.glob("*.xml")):
        root = etree.parse(str(path)).getroot()
        for var, texts in variable_labels(root).items():
            labels[var].extend(texts)
    return labels


def never_negated_vars(xml_dir: Path) -> set[str]:
    """Variables only ever tested with <estVrai>, never assigned by a questionnaire."""
    tested: dict[str, set[str]] = defaultdict(set)
    assigned: set[str] = set()
    for path in sorted(xml_dir.glob("*.xml")):
        root = etree.parse(str(path)).getroot()
        for el in root.iter("estVrai", "estFaux", "estIndefini"):
            tested[el.get("var")].add(el.tag)
        assigned.update(el.get("var") for el in root.iter("affecteVrai", "affecteFaux"))
    return {v for v, tags in tested.items() if tags == {"estVrai"} and v not in assigned}


def condition_stats(
    xml_dir: Path, fiches: pd.DataFrame, chunks: pd.DataFrame
) -> tuple[dict, list[str]]:
    """Counts per parent tag, label resolution and coverage; plus the ids of affected chunks."""
    global_labels = corpus_labels(xml_dir)
    bodies = {
        fid: list(zip(g["chunk_id"], g["body"], strict=True))
        for fid, g in chunks.groupby("fiche_id")
    }
    per_tag: dict[str, Counter] = defaultdict(Counter)
    sheets_per_tag: dict[str, set] = defaultdict(set)
    affected_chunks: set[str] = set()
    resolvable_chunks: set[str] = set()
    flag_only_chunks: set[str] = set()
    never_negated = never_negated_vars(xml_dir)
    affected_sheets: set[str] = set()
    sheets_with_questionnaire = 0
    untitled_vars: Counter = Counter()
    for fid, sections in zip(fiches["id"], fiches["sections"], strict=True):
        root = etree.parse(str(xml_dir / f"{fid}.xml")).getroot()
        sheets_with_questionnaire += root.find("Questionnaire") is not None
        local_labels = variable_labels(root)
        section_text = " ".join(clean(s["text"]) for s in sections)
        sheet_bodies = bodies.get(fid, [])
        for condition in root.iter("Condition"):
            parent = condition.getparent()
            tag = parent.tag
            variables = condition_vars(condition)
            lines = [clean(line) for line in render_children(parent)]
            words = sum(len(line.split()) for line in lines)
            counts = per_tag[tag]
            sheets_per_tag[tag].add(fid)
            counts["conditions"] += 1
            counts["words"] += words
            counts["resolved_in_sheet"] += all(v in local_labels for v in variables)
            counts["resolved_in_snapshot"] += all(v in global_labels for v in variables)
            counts["words_missing_from_sections"] += sum(
                len(line.split()) for line in lines if line and line not in section_text
            )
            if tag in TITLED:
                continue
            untitled_vars.update(v for v in variables if v in never_negated)
            long_lines = [line for line in lines if len(line.split()) >= MIN_LINE_WORDS]
            for chunk_id, body in sheet_bodies:
                flat = clean(body)
                if any(line in flat for line in long_lines):
                    affected_chunks.add(chunk_id)
                    affected_sheets.add(fid)
                    if all(v in global_labels for v in variables):
                        resolvable_chunks.add(chunk_id)
                    if all(v in never_negated for v in variables):
                        flag_only_chunks.add(chunk_id)
    by_tag = {
        tag: {**dict(c), "sheets": len(sheets_per_tag[tag])}
        for tag, c in sorted(per_tag.items(), key=lambda kv: -kv[1]["conditions"])
    }
    stats = {
        "sheets": len(fiches),
        "chunks": len(chunks),
        "sheets_with_questionnaire": int(sheets_with_questionnaire),
        "titled_parents": sorted(TITLED),
        "by_parent_tag": by_tag,
        "untitled_conditioned": {
            "min_line_words_for_matching": MIN_LINE_WORDS,
            "chunks": len(affected_chunks),
            "sheets": len(affected_sheets),
            "chunks_with_resolvable_condition": len(resolvable_chunks),
            "chunks_with_never_negated_condition_only": len(flag_only_chunks),
        },
        "never_negated_variables_in_untitled": dict(untitled_vars.most_common(5)),
    }
    return stats, sorted(affected_chunks)


def main() -> None:
    xml_dir = latest_snapshot() / "xml"
    fiches = pd.read_parquet(FICHES)
    chunks = pd.read_parquet(CHUNKS, columns=["chunk_id", "fiche_id", "body"])
    stats, excluded = condition_stats(xml_dir, fiches, chunks)
    OUTPUT.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    record = {"reason": "text of an untitled conditioned element", "chunk_ids": excluded}
    EXCLUDED.write_text(json.dumps(record, indent=1) + "\n")
    print(f"Wrote {EXCLUDED} ({len(excluded)} chunk ids)")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
