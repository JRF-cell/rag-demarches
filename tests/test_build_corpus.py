import shutil
from pathlib import Path

from src.ingest.build_corpus import build_corpus

FIXTURES = Path(__file__).parent / "fixtures"


def test_build_corpus_keeps_only_in_scope_types(tmp_path):
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    for path in FIXTURES.glob("F*.xml"):
        shutil.copy(path, xml_dir)
    table, report = build_corpus(tmp_path)
    assert sorted(table["id"]) == sorted(["F1", "F1357", "F1887", "F2408", "F1719", "F116"])
    assert report["out_of_scope_by_type"] == {"Recherche guidée": 1}
    assert report["parse_failed_or_empty"] == []
    assert report["sections_kept"] == int(table["sections"].map(len).sum())
