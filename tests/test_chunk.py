from pathlib import Path

import pandas as pd

from src.ingest.chunk import build_chunks, fiche_chunks, make_header, split_section
from src.ingest.parse_xml import parse_fiche

FIXTURES = Path(__file__).parent / "fixtures"


def test_short_section_is_a_single_chunk():
    assert split_section("one two three", max_words=5) == ["one two three"]


def test_whole_lines_are_packed_up_to_the_limit():
    # Lines of 2, 2 and 3 words with a limit of 5: [2+2] then [3].
    text = "a b\nc d\ne f g"
    assert split_section(text, max_words=5) == ["a b\nc d", "e f g"]


def test_line_exactly_at_limit_is_not_split():
    assert split_section("a b c\nd e", max_words=5) == ["a b c\nd e"]


def test_line_longer_than_limit_is_cut_into_slices():
    # 7 words, limit 3: [a b c] [d e f] then the remainder [g] packed with the next line [h].
    text = "a b c d e f g\nh"
    assert split_section(text, max_words=3) == ["a b c", "d e f", "g\nh"]


def test_no_overlap_and_no_lost_words():
    text = "a b c d e f g\nh i\nj k l m"
    pieces = split_section(text, max_words=3)
    assert " ".join(pieces).split() == text.split()


def test_header_joins_title_and_path():
    assert make_header("Titre", ["Chapitre", "Sous-chapitre"]) == "Titre > Chapitre > Sous-chapitre"
    assert make_header("Titre", []) == "Titre"


def test_chunks_of_a_real_sheet():
    fiche = parse_fiche(FIXTURES / "F1357.xml")
    chunks = fiche_chunks(fiche, max_words=250)
    assert [c["chunk_id"] for c in chunks] == ["F1357-000", "F1357-001", "F1357-002", "F1357-003"]
    assert chunks[1]["header"] == f"{fiche['title']} > En métropole"
    assert chunks[1]["text"] == f"{chunks[1]['header']}\n{chunks[1]['body']}"
    assert all(c["url"] == fiche["url"] and c["fiche_id"] == "F1357" for c in chunks)


def test_all_fixture_chunks_respect_the_limit():
    fiches = [parse_fiche(p) for p in sorted(FIXTURES.glob("F*.xml"))]
    table = build_chunks(pd.DataFrame([f for f in fiches if f]), max_words=40)
    assert table["n_words"].between(1, 40).all()
    assert table["chunk_id"].is_unique
