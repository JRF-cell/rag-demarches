import json
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.split import (
    CHUNKS,
    FICHES,
    SPLIT,
    link_groups,
    normalize_body,
    shared_body_leak,
    split_groups,
)

LONG = "word " * 30


def make_chunks(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame([{"fiche_id": f, "body": b, "n_words": len(b.split())} for f, b in rows])


def test_normalize_body_ignores_case_and_spacing():
    assert normalize_body("  Vous  devez\nPAYER ") == "vous devez payer"


def test_link_groups_joins_sheets_sharing_a_long_body_transitively():
    chunks = make_chunks(
        [("A", LONG + "x"), ("B", LONG.upper() + "x"), ("B", LONG + "y"), ("C", LONG + "y")]
        + [("D", LONG + "z")]
    )
    groups = link_groups(chunks, ["A", "B", "C", "D"], min_words=20)
    assert groups["A"] == groups["B"] == groups["C"]
    assert groups["D"] != groups["A"]


def test_link_groups_ignores_short_shared_bodies():
    chunks = make_chunks([("A", "Oui."), ("B", "Oui.")])
    groups = link_groups(chunks, ["A", "B"], min_words=20)
    assert groups["A"] != groups["B"]


def test_split_groups_is_disjoint_complete_deterministic_and_keeps_groups_whole():
    ids = [f"F{i}" for i in range(100)]
    groups = {fid: fid for fid in ids}
    groups.update({"F1": "F0", "F2": "F0", "F3": "F0"})
    themes = {fid: ("t1" if i < 50 else "t2") for i, fid in enumerate(ids)}
    train, test = split_groups(groups, themes, test_fraction=0.2, seed=0)
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(ids)
    assert (train, test) == split_groups(groups, themes, test_fraction=0.2, seed=0)
    group_sides = {fid in test for fid in ("F0", "F1", "F2", "F3")}
    assert len(group_sides) == 1
    for theme in ("t1", "t2"):
        n_test = sum(themes[f] == theme for f in test)
        assert 10 <= n_test <= 13


def test_shared_body_leak_counts_test_chunks_seen_in_train():
    chunks = make_chunks([("A", LONG), ("B", LONG), ("C", LONG + "c")])
    assert shared_body_leak(chunks, {"B", "C"}, min_words=20) == {"test_chunks": 2, "leaked": 1}


@pytest.mark.skipif(not SPLIT.exists(), reason="eval/split.json not built")
def test_frozen_split_is_disjoint():
    split = json.loads(SPLIT.read_text())
    assert split["train"] and split["test"]
    assert set(split["train"]).isdisjoint(split["test"])
    assert len(set(split["train"])) == len(split["train"])
    assert len(set(split["test"])) == len(split["test"])


@pytest.mark.skipif(
    not (SPLIT.exists() and Path(FICHES).exists()), reason="split or corpus not built"
)
def test_frozen_split_covers_the_corpus_without_shared_long_bodies():
    split = json.loads(SPLIT.read_text())
    fiches = pd.read_parquet(FICHES, columns=["id"])
    chunks = pd.read_parquet(CHUNKS, columns=["fiche_id", "body", "n_words"])
    assert set(split["train"]) | set(split["test"]) == set(fiches["id"])
    leak = shared_body_leak(chunks, set(split["test"]), split["link_min_words"])
    assert leak["leaked"] == 0
