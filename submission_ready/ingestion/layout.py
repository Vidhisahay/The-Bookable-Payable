"""Small, optional spatial-text representation and conservative party-role resolver."""

from __future__ import annotations

import re
from typing import Any


def normalized_words(words: list[dict[str, Any]], width: float, height: float, *, source: str) -> dict[str, Any]:
    """Return normalized page coordinates (0..1) while retaining provenance."""
    if width <= 0 or height <= 0:
        return {"coordinate_system": "normalized_page", "words": []}
    result = []
    for word in words:
        box = word.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x0, y0, x1, y1 = (float(value) for value in box)
        result.append({
            "text": str(word.get("text", "")),
            "bbox": [max(0.0, min(1.0, x0 / width)), max(0.0, min(1.0, y0 / height)),
                     max(0.0, min(1.0, x1 / width)), max(0.0, min(1.0, y1 / height))],
            "source": source,
            "confidence": word.get("confidence"),
            "block": word.get("block"),
            "line": word.get("line"),
        })
    return {"coordinate_system": "normalized_page", "words": result}


def layout_lines(layout: dict[str, Any]) -> list[dict[str, Any]]:
    """Group nearby words into visual lines, independent of OCR reading order."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for index, word in enumerate(layout.get("words", [])):
        box = word.get("bbox")
        if not isinstance(box, list) or len(box) != 4 or not word.get("text", "").strip():
            continue
        # Native PDF and TSV both provide line ids. Fallback is geometric.
        key = (word.get("source"), word.get("block"), word.get("line")) if word.get("line") is not None else ("y", round((box[1] + box[3]) / 2, 2))
        groups.setdefault(key, []).append(word)
    lines = []
    for group in groups.values():
        group.sort(key=lambda item: item["bbox"][0])
        lines.append({"text": " ".join(item["text"] for item in group),
                      "bbox": [min(w["bbox"][0] for w in group), min(w["bbox"][1] for w in group),
                               max(w["bbox"][2] for w in group), max(w["bbox"][3] for w in group)],
                      "words": group})
    return sorted(lines, key=lambda line: (line["bbox"][1], line["bbox"][0]))


def layout_regions(layout: dict[str, Any]) -> list[dict[str, Any]]:
    """Summarize spatial text blocks/regions without flattening their columns."""
    regions: dict[Any, list[dict[str, Any]]] = {}
    ungrouped: list[dict[str, Any]] = []
    for line in layout_lines(layout):
        blocks = {word.get("block") for word in line["words"] if word.get("block") is not None}
        if len(blocks) == 1:
            regions.setdefault(next(iter(blocks)), []).append(line)
        else:
            ungrouped.append(line)
    result = []
    for block, lines in regions.items():
        result.append({"block": block, "text": "\n".join(line["text"] for line in lines),
                       "bbox": [min(line["bbox"][0] for line in lines), min(line["bbox"][1] for line in lines),
                                max(line["bbox"][2] for line in lines), max(line["bbox"][3] for line in lines)]})
    for index, line in enumerate(ungrouped):
        result.append({"block": None, "text": line["text"], "bbox": line["bbox"]})
    return sorted(result, key=lambda region: (region["bbox"][1], region["bbox"][0]))


_ENTITY = re.compile(r"\b(?:GmbH|Limited|Ltd\.?|LLC|PLC|Inc\.?|Corp\.?|Pte\.?|OÜ|AS|BV|B\.V\.|AG|S\.r\.l\.)\b", re.I)
_LABELS = {
    "supplier": re.compile(r"(?:^\s*from\b|\b(?:supplier|vendor|seller|issued\s+by)\b)", re.I),
    "buyer": re.compile(r"\b(?:bill\s+to|buyer|customer|sold\s+to|invoice\s+to)\b", re.I),
    "ship_to": re.compile(r"\b(?:ship\s+to|deliver\s+to|delivery\s+address)\b", re.I),
    "payee": re.compile(r"\b(?:payee|remit\s+to|remit\s+payment\s+to|payment\s+to)\b", re.I),
}


def resolve_party_roles(layout: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Resolve only explicit labelled parties with nearby legal-entity evidence.

    Proximity is measured in normalized page space. Column position itself never
    determines the role. Duplicate/conflicting role labels remain ambiguous.
    """
    lines = layout_lines(layout)
    results: dict[str, dict[str, Any]] = {}
    for role, pattern in _LABELS.items():
        labels = [line for line in lines if pattern.search(line["text"])]
        candidates: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        for label in labels:
            same_line_value = pattern.sub("", label["text"], count=1).strip(" :#-|\t")
            if same_line_value and _ENTITY.search(same_line_value):
                candidates.append((same_line_value, label, label))
            for entity in lines:
                if entity is label or not _ENTITY.search(entity["text"]):
                    continue
                if (re.search(r"\b(?:invoice\s+date|invoice\s+number|due\s+date)\b", entity["text"], re.I)
                        and not pattern.search(entity["text"])):
                    continue
                # Entity block must be near the role label. Allow either OCR
                # ordering and same-line labels, but avoid across-column jumps.
                lx0, ly0, lx1, ly1 = label["bbox"]
                ex0, ey0, ex1, ey1 = entity["bbox"]
                vertical = max(0.0, max(ey0 - ly1, ly0 - ey1))
                horizontal = max(0.0, max(ex0 - lx1, lx0 - ex1))
                same_line = vertical <= 0.012 and horizontal <= 0.12
                if vertical > 0.075 or (not same_line and horizontal > 0.12):
                    continue
                # A label on the same line can precede or follow its value.
                value = entity["text"].strip()
                # Avoid returning the label text as part of the company value.
                if pattern.search(value):
                    value = pattern.sub("", value).strip(" :#-|\t")
                # Preserve a name broken across adjacent visual lines when
                # those lines occupy the same column and only the latter has
                # the legal suffix.
                prior = [other for other in lines if other is not label and other is not entity
                         and other["bbox"][3] <= entity["bbox"][1]
                         and entity["bbox"][1] - other["bbox"][3] <= 0.025
                         and abs(other["bbox"][0] - entity["bbox"][0]) <= 0.08
                         and not any(rx.search(other["text"]) for rx in _LABELS.values())]
                if prior:
                    closest = max(prior, key=lambda other: other["bbox"][3])
                    value = closest["text"].strip() + " " + value
                if _ENTITY.search(value):
                    candidates.append((value, label, entity))
        unique = {re.sub(r"\W+", " ", value.casefold()).strip(): (value, label, entity)
                  for value, label, entity in candidates}
        if len(unique) == 1:
            value, label, entity = next(iter(unique.values()))
            results[role] = {"status": "found", "value": value,
                             "evidence": {"label": label["text"], "party": entity["text"],
                                          "source": entity["words"][0].get("source") if entity["words"] else None}}
        elif unique or len(labels) > 1:
            results[role] = {"status": "ambiguous", "value": None,
                             "evidence": {"labels": [line["text"] for line in labels],
                                          "candidates": sorted(unique)}}
        else:
            results[role] = {"status": "missing", "value": None, "evidence": {}}
    # Some invoice designs do not print “Supplier” or “Bill To”. A narrow
    # structural cue is still available when a legal entity shares the visual
    # baseline with the explicit Invoice Date label, to its right, on a page
    # headed “Tax Invoice”. This is metadata adjacency, not a corner/position
    # heuristic; competing entities or conflicting explicit role labels stop it.
    if results["supplier"]["status"] == "missing":
        header_entities: list[tuple[str, dict[str, Any]]] = []
        for line in lines:
            words = line["words"]
            date_word = next((word for word in words if word["text"].casefold() == "date"
                              and any(prior["text"].casefold() == "invoice" and prior["bbox"][2] <= word["bbox"][0]
                                      for prior in words)), None)
            if date_word is None:
                continue
            date_y = (date_word["bbox"][1] + date_word["bbox"][3]) / 2
            suffixes = [word for word in words if _ENTITY.search(word["text"])
                        and word["bbox"][0] >= date_word["bbox"][2]
                        and abs((word["bbox"][1] + word["bbox"][3]) / 2 - date_y) <= 0.015]
            if len(suffixes) != 1:
                continue
            suffix = suffixes[0]
            after_date = [word for word in words if word["bbox"][0] >= date_word["bbox"][2]
                          and word["bbox"][2] <= suffix["bbox"][2]]
            # Keep only the visual text between the date label and legal suffix;
            # this excludes metadata/date tokens elsewhere on the same row.
            name_words = [word["text"] for word in after_date
                          if not re.fullmatch(r"(?:\d{1,4}[./-]\d{1,2}[./-]\d{1,4}|\d{1,4}|[:#])", word["text"])]
            value = " ".join(name_words).strip()
            if len(value.split()) >= 2 and _ENTITY.search(value):
                header_entities.append((value, line))
        distinct = {re.sub(r"\W+", " ", value.casefold()).strip(): (value, line)
                    for value, line in header_entities}
        if len(distinct) == 1:
            value, line = next(iter(distinct.values()))
            value_key = re.sub(r"\W+", " ", value.casefold()).strip()
            explicitly_other_party = any(
                results[role]["status"] == "found"
                and re.sub(r"\W+", " ", str(results[role]["value"]).casefold()).strip() == value_key
                for role in ("buyer", "ship_to", "payee")
            )
            if not explicitly_other_party:
                results["supplier"] = {"status": "found", "value": value,
                                       "evidence": {"label": "Invoice Date adjacency",
                                                    "party": line["text"], "source": "layout_structure"}}
        elif len(distinct) > 1:
            results["supplier"] = {"status": "ambiguous", "value": None,
                                   "evidence": {"candidates": sorted(distinct), "basis": "invoice_header_adjacency"}}
    return results


def resolve_party_vat(layout: dict[str, Any], party_name: str) -> dict[str, Any]:
    """Associate a VAT label/value only with a nearby party block in the same region."""
    lines = layout_lines(layout)
    party_key = re.sub(r"\W+", " ", party_name.casefold()).strip()
    party_lines = [line for line in lines
                   if party_key and party_key in re.sub(r"\W+", " ", line["text"].casefold())]
    if not party_lines:
        return {"status": "missing", "value": None, "evidence": {}}
    party = party_lines[0]
    vat_label = re.compile(r"\b(?:vat|gst)\s*(?:id|no\.?|number|registration(?:\s+number|\s+no\.?)?)\b", re.I)
    id_pattern = re.compile(r"\b(?:[A-Z]{2}\s*)?[A-Z0-9][A-Z0-9 -]{5,}[A-Z0-9]\b", re.I)
    possibilities: list[tuple[str, float, dict[str, Any], str]] = []
    px = (party["bbox"][0] + party["bbox"][2]) / 2
    py = (party["bbox"][1] + party["bbox"][3]) / 2
    for index, line in enumerate(lines):
        if not vat_label.search(line["text"]):
            continue
        values = [line]
        values.extend(other for other in lines if other is not line
                      and abs(other["bbox"][1] - line["bbox"][3]) <= 0.035
                      and abs(other["bbox"][0] - line["bbox"][0]) <= 0.08)
        for value_line in values:
            match = id_pattern.search(vat_label.sub("", value_line["text"]))
            if not match:
                continue
            value = re.sub(r"[\s-]", "", match.group(0)).upper()
            # Avoid interpreting a short year or account-like token as an ID.
            if sum(character.isdigit() for character in value) < 6:
                continue
            vx = (line["bbox"][0] + line["bbox"][2]) / 2
            vy = (line["bbox"][1] + line["bbox"][3]) / 2
            dx, dy = abs(px - vx), abs(py - vy)
            if dx <= 0.18 and dy <= 0.25:
                possibilities.append((value, dx + dy, line, value_line["text"]))
    if not possibilities:
        return {"status": "missing", "value": None, "evidence": {}}
    possibilities.sort(key=lambda item: item[1])
    best = possibilities[0]
    if len(possibilities) > 1 and abs(possibilities[1][1] - best[1]) < 0.035:
        return {"status": "ambiguous", "value": None,
                "evidence": {"candidate_values": sorted({item[0] for item in possibilities})}}
    return {"status": "found", "value": best[0],
            "evidence": {"label": best[2]["text"], "value": best[3], "party": party["text"]}}
