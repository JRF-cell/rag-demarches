"""Train/test split by sheet, keeping sheets that share chunk text on the same side.

Sheets are linked when they contain a chunk with the same normalised body (of at least
`link_min_words` words). Linked sheets form groups (connected components), and whole groups
are assigned to train or test, theme by theme, until each theme reaches `test_fraction`.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, results_dir

FICHES = Path("data/processed/fiches.parquet")
CHUNKS = Path("data/processed/chunks.parquet")
SPLIT = Path("eval/split.json")
CONFIG = Path("configs/split.yaml")


def normalize_body(text: str) -> str:
    """Lowercase and collapse whitespace, so that identical texts compare equal."""
    return re.sub(r"\s+", " ", text).strip().lower()


def link_groups(chunks: pd.DataFrame, fiche_ids: list[str], min_words: int) -> dict[str, str]:
    """Map each sheet to a group id; sheets sharing a chunk body are in the same group."""
    parent = {fid: fid for fid in fiche_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    long_chunks = chunks[chunks["n_words"] >= min_words]
    bodies = long_chunks["body"].map(normalize_body)
    for ids in long_chunks.groupby(bodies)["fiche_id"].unique():
        root = find(ids[0])
        for other in ids[1:]:
            parent[find(other)] = root
    return {fid: find(fid) for fid in fiche_ids}


def group_themes(groups: dict[str, str], themes: dict[str, str]) -> dict[str, str]:
    """Majority theme of each group (ties broken alphabetically)."""
    counts: dict[str, Counter] = {}
    for fid, gid in groups.items():
        counts.setdefault(gid, Counter())[themes[fid]] += 1
    return {gid: min(c, key=lambda t: (-c[t], t)) for gid, c in counts.items()}


def split_groups(
    groups: dict[str, str], themes: dict[str, str], test_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Assign whole groups to test, per theme, until the theme reaches `test_fraction`."""
    rng = np.random.default_rng(seed)
    members: dict[str, list[str]] = {}
    for fid, gid in groups.items():
        members.setdefault(gid, []).append(fid)
    theme_of_group = group_themes(groups, themes)
    test: set[str] = set()
    for theme in sorted(set(theme_of_group.values())):
        gids = sorted(g for g, t in theme_of_group.items() if t == theme)
        target = round(test_fraction * sum(len(members[g]) for g in gids))
        taken = 0
        for i in rng.permutation(len(gids)):
            if taken >= target:
                break
            test.update(members[gids[i]])
            taken += len(members[gids[i]])
    train = sorted(set(groups) - test)
    return train, sorted(test)


def shared_body_leak(chunks: pd.DataFrame, test_ids: set[str], min_words: int) -> dict:
    """Count test chunks whose normalised body also appears in a train sheet."""
    sub = chunks[chunks["n_words"] >= min_words]
    bodies = sub["body"].map(normalize_body)
    is_test = sub["fiche_id"].isin(test_ids)
    train_bodies = set(bodies[~is_test])
    leaked = bodies[is_test].isin(train_bodies)
    return {"test_chunks": int(is_test.sum()), "leaked": int(leaked.sum())}


def random_split(fiche_ids: list[str], test_fraction: float, seed: int) -> set[str]:
    """Plain random split by sheet, used only as a baseline for the leak count."""
    rng = np.random.default_rng(seed)
    ids = sorted(fiche_ids)
    n_test = round(test_fraction * len(ids))
    return {ids[i] for i in rng.permutation(len(ids))[:n_test]}


def split_stats(
    fiches: pd.DataFrame,
    chunks: pd.DataFrame,
    groups: dict[str, str],
    test: list[str],
    config: dict,
) -> dict:
    """Sizes per side and theme, group sizes, and leak counts against a random baseline."""
    test_ids = set(test)
    side = fiches["id"].isin(test_ids).map({True: "test", False: "train"})
    chunk_side = chunks["fiche_id"].isin(test_ids).map({True: "test", False: "train"})
    sizes = Counter(groups.values())
    min_words = config["link_min_words"]
    baseline = random_split(list(fiches["id"]), config["test_fraction"], config["seed"])
    return {
        "fiches": dict(Counter(side)),
        "chunks": dict(Counter(chunk_side)),
        "groups": len(sizes),
        "single_sheet_groups": sum(1 for n in sizes.values() if n == 1),
        "largest_groups": sorted(sizes.values(), reverse=True)[:10],
        "test_fiches_by_theme": {
            theme: {"test": int(sum(g == "test")), "total": len(g)}
            for theme, g in side.groupby(fiches["theme"])
        },
        "test_fiches_by_type": {t: int(sum(g == "test")) for t, g in side.groupby(fiches["type"])},
        "leak_min_words": min_words,
        "leak_group_split": shared_body_leak(chunks, test_ids, min_words),
        "leak_group_split_all_lengths": shared_body_leak(chunks, test_ids, 0),
        "leak_random_split_baseline": shared_body_leak(chunks, baseline, min_words),
    }


def build_split(fiches: pd.DataFrame, chunks: pd.DataFrame, config: dict) -> dict:
    """Compute the split record written to eval/split.json."""
    fiche_ids = list(fiches["id"])
    groups = link_groups(chunks, fiche_ids, config["link_min_words"])
    themes = dict(zip(fiches["id"], fiches["theme"], strict=True))
    train, test = split_groups(groups, themes, config["test_fraction"], config["seed"])
    return {
        "seed": config["seed"],
        "test_fraction": config["test_fraction"],
        "link_min_words": config["link_min_words"],
        "train": train,
        "test": test,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)
    fiches = pd.read_parquet(FICHES)
    chunks = pd.read_parquet(CHUNKS)
    split = build_split(fiches, chunks, config)
    if SPLIT.exists():
        frozen = json.loads(SPLIT.read_text())
        if frozen != split:
            raise SystemExit(
                f"{SPLIT} exists and differs from the recomputed split; not touching it"
            )
        print(f"{SPLIT} already exists and matches the recomputed split")
    else:
        SPLIT.parent.mkdir(parents=True, exist_ok=True)
        SPLIT.write_text(json.dumps(split, indent=1) + "\n")
        print(f"Wrote {SPLIT}")
    groups = link_groups(chunks, list(fiches["id"]), config["link_min_words"])
    stats = split_stats(fiches, chunks, groups, split["test"], config)
    out = results_dir(config, args.config)
    (out / "split_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {out / 'split_stats.json'}")


if __name__ == "__main__":
    main()
