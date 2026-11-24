"""Parse Service-Public.gouv.fr sheet XML (feed 3.5) into titled plain-text sections."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from pathlib import Path

from lxml import etree

logger = logging.getLogger(__name__)

DC = "{http://purl.org/dc/elements/1.1/}"

# Elements whose content is not readable text for a user (metadata, links, conditions).
SKIPPED = {
    "Condition",
    "Source",
    "PivotLocal",
    "RessourceWeb",
    "Colonne",
    "TitreRiche",
    "Infographie",
    "Image",
    "NoticeLiee",
    "PourEnSavoirPlus",
    "Questionnaire",
}

# Callout blocks rendered as "<label> : <title>" followed by their content.
CALLOUTS = {
    "ASavoir": "À savoir",
    "ANoter": "À noter",
    "Attention": "Attention",
    "Exemple": "Exemple",
    "Complement": "Complément",
}


def clean(text: str) -> str:
    """Collapse runs of whitespace into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


# A few sheets carry escaped HTML inside their text (e.g. "&lt;span lang=...&gt;").
ESCAPED_HTML_TAG = re.compile(r"</?[A-Za-z][^<>]*>")


def inline_text(el: etree._Element) -> str:
    """Text of an element and all its inline descendants, whitespace-normalised."""
    return clean(ESCAPED_HTML_TAG.sub("", "".join(el.itertext())))


def title_of(el: etree._Element) -> str:
    """Text of the direct <Titre> child, or an empty string."""
    titre = el.find("Titre")
    return inline_text(titre) if titre is not None else ""


def render_children(el: etree._Element, depth: int = 0) -> list[str]:
    """Render every element child of `el`, skipping its <Titre>."""
    lines: list[str] = []
    for child in el:
        if isinstance(child.tag, str) and child.tag != "Titre":
            lines.extend(render_block(child, depth))
    return lines


def render_list(el: etree._Element, depth: int) -> list[str]:
    """Render <Liste>/<Item> as dash bullets, indented by nesting depth."""
    lines: list[str] = []
    for item in el.findall("Item"):
        item_lines = render_children(item, depth + 1)
        if item_lines:
            lines.append("  " * depth + "- " + item_lines[0])
            lines.extend(item_lines[1:])
    return lines


def render_table(el: etree._Element) -> list[str]:
    """Render a <Tableau> as its caption followed by one ' | '-separated line per row."""
    caption = el.find("TitreRiche")
    lines = [inline_text(caption)] if caption is not None else []
    for row in el.findall("Rangée"):
        cells = [" ".join(render_children(cell)) for cell in row.findall("Cellule")]
        lines.append(" | ".join(cells))
    return lines


def render_online_service(el: etree._Element) -> list[str]:
    """Render a <ServiceEnLigne> as a one-line reference to the form or online service."""
    cerfa = el.get("numerocerfa")
    if cerfa:
        return [f"Formulaire : {title_of(el)} (cerfa n° {cerfa})"]
    return [f"Service en ligne : {title_of(el)}"]


def render_block(el: etree._Element, depth: int = 0) -> list[str]:
    """Render one body element into text lines."""
    tag = el.tag
    if not isinstance(tag, str) or tag in SKIPPED:
        return []
    if tag in ("Paragraphe", "TitreFlottant"):
        text = inline_text(el)
        return [text] if text else []
    if tag == "Liste":
        return render_list(el, depth)
    if tag == "Tableau":
        return render_table(el)
    if tag in CALLOUTS:
        label, title = CALLOUTS[tag], title_of(el)
        # The title often just repeats the label ("À noter"); keep it only when it adds something.
        header = f"{label} : {title}" if title and title.lower() != label.lower() else f"{label} :"
        return [header, *render_children(el, depth)]
    if tag == "BlocCas":
        return [line for cas in el.findall("Cas") for line in render_case(cas, depth)]
    if tag == "OuSAdresser":
        return [f"Où s'adresser : {title_of(el)}", *render_children(el, depth)]
    if tag == "ServiceEnLigne":
        return render_online_service(el)
    # Containers (Introduction, Texte, Conclusion, FragmentConditionne, ...) and unknown tags.
    return render_children(el, depth)


def render_case(cas: etree._Element, depth: int) -> list[str]:
    """Render one <Cas> of a <BlocCas> as '<case title> :' followed by its content."""
    return [f"{title_of(cas)} :", *render_children(cas, depth)]


def section(path: list[str], lines: list[str]) -> dict[str, object] | None:
    """Build a section dict, or None when it has no text."""
    text = "\n".join(line for line in lines if line)
    return {"path": list(path), "text": text} if text else None


def texte_sections(texte: etree._Element, path: list[str]) -> Iterator[dict[str, object]]:
    """Split a <Texte> into sections at <Chapitre> and <SousChapitre> boundaries."""
    buffer: list[str] = []
    for child in texte:
        if not isinstance(child.tag, str):
            continue
        if child.tag in ("Chapitre", "SousChapitre"):
            if sec := section(path, buffer):
                yield sec
            buffer = []
            yield from texte_sections(child, [*path, title_of(child)])
        elif child.tag != "Titre":
            buffer.extend(render_block(child))
    if sec := section(path, buffer):
        yield sec


def situation_sections(situation: etree._Element) -> Iterator[dict[str, object]]:
    """Sections of one <Situation>, all prefixed by the situation title."""
    path = [title_of(situation)]
    for child in situation:
        if child.tag == "Texte":
            yield from texte_sections(child, path)
        elif child.tag in ("Introduction", "Conclusion") and (
            sec := section(path, render_block(child))
        ):
            yield sec


def body_sections(root: etree._Element) -> list[dict[str, object]]:
    """All sections of a sheet, in document order."""
    sections: list[dict[str, object]] = []
    for child in root:
        if child.tag == "Texte":
            sections.extend(texte_sections(child, []))
        elif child.tag == "ListeSituations":
            for sub in child:
                if sub.tag == "Situation":
                    sections.extend(situation_sections(sub))
                elif sub.tag == "Conclusion" and (sec := section([], render_block(sub))):
                    sections.append(sec)
        elif child.tag in ("Introduction", "Conclusion") and (
            sec := section([], render_block(child))
        ):
            sections.append(sec)
    return merge_same_path(sections)


def merge_same_path(sections: list[dict[str, object]]) -> list[dict[str, object]]:
    """Merge consecutive sections that share a path (e.g. a situation's intro and its text)."""
    merged: list[dict[str, object]] = []
    for sec in sections:
        if merged and merged[-1]["path"] == sec["path"]:
            merged[-1]["text"] = f"{merged[-1]['text']}\n{sec['text']}"
        else:
            merged.append(sec)
    return merged


def modified_date(root: etree._Element) -> str | None:
    """ISO date from '<dc:date>modified YYYY-MM-DD</dc:date>'."""
    match = re.search(r"\d{4}-\d{2}-\d{2}", root.findtext(f"{DC}date") or "")
    return match.group(0) if match else None


def parse_fiche(path: Path) -> dict[str, object] | None:
    """Parse one sheet file; return None (and log why) if it is unreadable or empty."""
    try:
        root = etree.parse(str(path)).getroot()
    except (etree.XMLSyntaxError, OSError) as err:
        logger.warning("Unreadable XML %s: %s", path.name, err)
        return None
    fiche_id, title = root.get("ID"), clean(root.findtext(f"{DC}title") or "")
    if root.tag != "Publication" or not fiche_id or not title:
        logger.warning("Missing Publication ID or title in %s", path.name)
        return None
    sections = body_sections(root)
    if not sections:
        logger.info("No body text in %s", path.name)
        return None
    return {
        "id": fiche_id,
        "type": root.get("type"),
        "title": title,
        "description": clean(root.findtext(f"{DC}description") or ""),
        "theme": clean(root.findtext("Theme/Titre") or "") or None,
        "breadcrumb": [clean(n.text or "") for n in root.findall("FilDAriane/Niveau")],
        "modified": modified_date(root),
        "important_change": root.get("dateDerniereModificationImportante"),
        "url": root.get("spUrl"),
        "sections": sections,
    }
