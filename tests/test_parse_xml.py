from pathlib import Path

import pytest
from lxml import etree

from src.ingest.parse_xml import inline_text, parse_fiche

FIXTURES = Path(__file__).parent / "fixtures"
PARSED_IDS = ["F1", "F1357", "F1887", "F2408", "F1719", "F116"]


def parse(fiche_id: str) -> dict:
    fiche = parse_fiche(FIXTURES / f"{fiche_id}.xml")
    assert fiche is not None
    return fiche


def all_text(fiche: dict) -> str:
    return "\n".join(s["text"] for s in fiche["sections"])


def test_metadata_are_extracted():
    fiche = parse("F1")
    assert fiche["id"] == "F1"
    assert fiche["type"] == "Fiche d'information conditionnée"
    assert fiche["title"] == (
        "Impôt sur le revenu - Versement de cotisations syndicales (crédit d'impôt)"
    )
    assert fiche["theme"] == "Argent - Impôts - Consommation"
    assert fiche["modified"] == "2026-04-15"
    assert fiche["important_change"] == "2025-04-10T00:00:00"
    assert fiche["url"] == "https://www.service-public.gouv.fr/particuliers/vosdroits/F1"
    assert fiche["breadcrumb"][0] == "Accueil particuliers"
    assert fiche["breadcrumb"][-1] == fiche["title"]


def test_missing_important_change_date_is_none():
    assert parse("F1887")["important_change"] is None


def test_chapters_and_subchapters_become_section_paths():
    sections = parse("F1")["sections"]
    assert len(sections) == 6
    assert sections[0]["path"] == []
    conditions = (
        "Quelles sont les conditions à remplir pour le crédit d'impôt"
        " pour versement de cotisations syndicales ?"
    )
    assert sections[1]["path"] == [conditions, "Personnes concernées"]
    assert sections[2]["path"] == [conditions, "Organisations syndicales concernées"]
    assert "66 % des cotisations annuelles" in sections[3]["text"]


def test_lists_are_rendered_as_bullets():
    text = parse("F1")["sections"][1]["text"]
    assert "\n- Retraité\n" in text


def test_callout_title_repeating_label_is_dropped():
    text = all_text(parse("F1"))
    assert "À noter :" in text
    assert "À noter : À noter" not in text
    assert "Exemple : Exemple" not in text


def test_situations_prefix_paths_and_merge_intro_with_text():
    sections = parse("F1357")["sections"]
    assert [s["path"] for s in sections] == [[], ["En métropole"], ["Outre-mer"], ["À l'étranger"]]
    outre_mer = sections[2]["text"]
    assert outre_mer.startswith("Tout dépend du lieu de la demande")
    assert "\nEn Guyane :\n" in outre_mer


def test_cases_are_rendered_with_their_title():
    text = all_text(parse("F1887"))
    assert "\nPour un majeur :\n" in text
    assert "\nPour un mineur :\n" in text


def test_tables_are_linearised_row_by_row():
    text = all_text(parse("F2408"))
    assert "Différences entre frais professionnels et avantages en nature" in text
    assert "\nFrais professionnels | Non | Non\n" in text


def test_short_sheet_has_single_root_section():
    sections = parse("F1719")["sections"]
    assert len(sections) == 1
    assert sections[0]["path"] == []
    assert sections[0]["text"].startswith("Non, l’aide au permis B France Travail est supprimée")


def test_conditional_fragments_kept_and_references_reduced_to_titles():
    text = all_text(parse("F116"))
    assert text.startswith("Non, vous ne pouvez pas circuler")
    assert "Où s'adresser : Ambassade ou consulat étranger en France" in text
    assert "Ministère chargé de l'Europe" not in text


@pytest.mark.parametrize("fiche_id", PARSED_IDS)
def test_text_is_clean(fiche_id):
    for sec in parse(fiche_id)["sections"]:
        text = sec["text"]
        assert text
        assert "<" not in text and ">" not in text
        assert "  " not in text.replace("\n  ", "\n")
        assert all(line.strip() for line in text.split("\n"))


def test_sheet_without_body_text_returns_none():
    assert parse_fiche(FIXTURES / "F35490.xml") is None


def test_truncated_xml_returns_none(tmp_path):
    raw = (FIXTURES / "F1.xml").read_bytes()
    broken = tmp_path / "F1.xml"
    broken.write_bytes(raw[: len(raw) // 2])
    assert parse_fiche(broken) is None


def test_non_publication_root_returns_none(tmp_path):
    other = tmp_path / "F0.xml"
    other.write_text('<?xml version="1.0" encoding="UTF-8"?><Arborescence ID="F0"/>')
    assert parse_fiche(other) is None


def test_missing_file_returns_none(tmp_path):
    assert parse_fiche(tmp_path / "absent.xml") is None


def test_escaped_html_in_text_is_removed():
    el = etree.fromstring(
        '<Paragraphe>arabe - &lt;span lang="ar" dir="rtl"&gt;العربية&lt;/span&gt;,'
        " moins de &lt; 5 ans</Paragraphe>"
    )
    assert inline_text(el) == "arabe - العربية, moins de < 5 ans"
