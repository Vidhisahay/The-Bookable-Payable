"""Deterministic, conservative payable-field extraction from classified pages."""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Callable

from ..models import DocumentRecord, PageRecord
from ..layout import resolve_party_roles, resolve_party_vat
from .models import (
    ChargeItem,
    Evidence,
    ExtractedField,
    ExtractionOutcome,
    LineItem,
    PayableCandidate,
    TaxItem,
)
from .numbers import parse_localized_decimal
from .line_type import classify_line_type


_DATE = r"(?P<date>\d{1,4}[./-]\d{1,2}[./-]\d{1,4}|\d{1,2}(?:-|\s+)[A-Za-z]{3,9}(?:-|\s+)\d{4})"
_NUMBER = r"[+-]?(?:\d{1,3}(?:[ .,\u00a0\u202f]\d{3})+|\d+)(?:[,.]\d{1,6})?"
_MONEY = re.compile(rf"(?<![\w/])(?P<number>{_NUMBER})(?!\w)")
_TABLE_MONEY = re.compile(r"(?<![\w/])(?P<number>[+-]?(?:\d{1,3}(?:[.,]\d{3})+|\d+)(?:[,.]\d{1,6})?)(?!\w)")
_CURRENCY_CODES = {
    "EUR": "EUR", "EURO": "EUR", "USD": "USD", "GBP": "GBP", "SGD": "SGD",
    "THB": "THB", "BAHT": "THB", "INR": "INR", "CAD": "CAD", "AUD": "AUD",
    "JPY": "JPY", "SEK": "SEK", "NOK": "NOK", "DKK": "DKK", "CHF": "CHF",
    "CNY": "CNY", "RMB": "CNY", "BRL": "BRL", "MXN": "MXN", "PLN": "PLN",
    "CZK": "CZK", "HUF": "HUF", "ILS": "ILS", "ZAR": "ZAR", "KES": "KES",
}


def _norm(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _evidence(page: PageRecord, text: str) -> Evidence:
    return Evidence(page.filename, page.page_number, text.strip())


def _readability(pages: list[PageRecord]) -> str:
    if not any(page.text.strip() for page in pages) or any(
        not page.text.strip() or (page.error and page.extraction_method in {"ocr_failed", "page_failed"})
        for page in pages
    ):
        return "unreadable"
    return "missing"


def _field(
    pages: list[PageRecord],
    matcher: Callable[[str], list[tuple[str, str]]],
    *,
    convert: Callable[[str], Any] = lambda value: value,
    normalize: Callable[[Any], Any] = lambda value: value,
    multiple_status: str = "conflicting",
) -> ExtractedField:
    found: list[tuple[Any, Evidence]] = []
    for page in pages:
        for raw_value, source_line in matcher(page.text):
            try:
                value = convert(raw_value)
            except (ValueError, TypeError):
                continue
            found.append((value, _evidence(page, source_line)))
    if not found:
        error = next((page.error for page in pages if page.error and page.extraction_method in {"ocr_failed", "page_failed"}), None)
        status = "unreadable" if _readability(pages) == "unreadable" or error else "missing"
        return ExtractedField(status=status, note=error if status == "unreadable" else None)

    distinct: list[Any] = []
    for value, _source in found:
        if not any(normalize(value) == normalize(existing) for existing in distinct):
            distinct.append(value)
    sources = [source for _value, source in found]
    if len(distinct) == 1:
        return ExtractedField(distinct[0], "found", sources)
    return ExtractedField(None, multiple_status, sources, distinct, "More than one distinct value is supported by the document text.")


def _label_values(pattern: str, text: str) -> list[tuple[str, str]]:
    return [(match.group("value").strip(" :#-").rstrip(".,;"), line) for line in text.splitlines()
            if (match := re.search(pattern, line, re.I))]


def _remittance_supplier(pages: list[PageRecord]) -> tuple[ExtractedField, ExtractedField]:
    """Use a legal-name/address block tied to bank details as supplier evidence.

    OCR often preserves a remittance block while dropping its visual columns.
    The name is accepted only when exactly one legal-entity line occurs shortly
    before a bank/IBAN line; this avoids guessing between invoice parties.
    """
    legal_suffix = re.compile(
        r"(?:\bO[ÜU]\b|\bGmbH\b|\b(?:Pte\.?\s*)?Ltd\.?\b|\bLimited\b|\bLLC\b|\bPLC\b|"
        r"\bPty\.?\s*Ltd\.?\b|\bSdn\.?\s*Bhd\.?\b|\bLda\.?\b|\bInc\.?\b|\bCorp\.?\b|"
        r"\bS\.?A\.?\b|\bB\.?V\.?\b|\bAG\b|\bS\.?r\.?l\.?\b)", re.I,
    )
    bank_marker = re.compile(r"\b(?:IBAN|bank\s+account|bank\s+details|bank\s+transfer)\b", re.I)
    name_exclusions = re.compile(r"\b(?:invoice|account|payment|bank|customer|buyer|bill\s+to|page)\b", re.I)
    for page in pages:
        lines = page.text.splitlines()
        for bank_index, bank_line in enumerate(lines):
            if not bank_marker.search(bank_line):
                continue
            start = max(0, bank_index - 5)
            block = lines[start:bank_index + 1]
            names = [(idx, line.strip()) for idx, line in enumerate(block)
                     if legal_suffix.search(line) and not name_exclusions.search(line)]
            unique = list(dict.fromkeys(_norm(name) for _, name in names))
            if len(unique) != 1:
                continue
            name_index, name = names[0]
            evidence_text = " ".join(line.strip() for line in block[name_index:] if line.strip())
            name_field = ExtractedField(name, "found", [_evidence(page, evidence_text)])
            address_lines = [line.strip() for line in block[name_index + 1:]
                             if line.strip() and re.search(r"\d", line)
                             and not bank_marker.search(line)]
            if address_lines:
                address = ", ".join(address_lines)
                address_field = ExtractedField(address, "found", [_evidence(page, address)])
            else:
                address_field = ExtractedField(status="missing")
            return name_field, address_field
    return ExtractedField(status="missing"), ExtractedField(status="missing")


def _header_supplier(pages: list[PageRecord]) -> tuple[ExtractedField, ExtractedField]:
    """Read a sole legal entity in the invoice header, or leave role unresolved.

    Flattened OCR order is not spatial order. In particular, taking the first
    legal suffix before a "Bill To" or invoice metadata label can select the
    buyer from the left column. This fallback therefore accepts a header name
    only when the header has exactly one distinct legal entity; explicit role
    labels and VAT-linked evidence are handled by their own parsers first.
    """
    legal_suffix = re.compile(
        r"(?:\bO[ÃœU]\b|\bGmbH\b|\b(?:Pte\.?\s*)?Ltd\.?\b|\bLimited\b|\bLLC\b|\bPLC\b|"
        r"\bPty\.?\s*Ltd\.?\b|\bSdn\.?\s*Bhd\.?\b|\bLda\.?\b|\bInc\.?\b|\bCorp\.?\b|"
        r"\bS\.?A\.?\b|\bB\.?V\.?\b|\bAG\b|\bS\.?r\.?l\.?\b)", re.I,
    )
    recipient = re.compile(r"\b(?:bill\s*to|sold\s*to|ship\s*to|customer|client)\b", re.I)
    structural = re.compile(r"\b(?:bill to|ship to|sold to|customer|payment instructions)\b", re.I)
    for page in pages:
        lines = page.text.splitlines()
        table_start = next((i for i, line in enumerate(lines)
                            if re.search(r"description|item\s+description", line, re.I)
                            and re.search(r"quantity|qty|unit\s+price|amount|total", line, re.I)), len(lines))
        limit = min(table_start, 40)
        entities: list[tuple[int, str]] = []
        for index, line in enumerate(lines[:limit]):
            suffix = legal_suffix.search(line)
            if not suffix or structural.search(line) or len(legal_suffix.findall(line)) != 1:
                continue
            # A company explicitly inside a recipient block is not an issuer.
            previous = " ".join(lines[max(0, index - 2):index + 1])
            if recipient.search(previous):
                continue
            entity = line[:suffix.end()].strip(" :|-\t")
            if line[suffix.end():].startswith(".") and not entity.endswith("."):
                entity += "."
            entity = re.sub(r"^(?:invoice\s+date|invoice\s+number|invoice|tax\s+invoice)\s+", "", entity, flags=re.I)
            entities.append((index, entity))
        if len({_norm(name) for _, name in entities}) != 1:
            continue
        index, name = entities[0]
        # A sole legal entity in the upper left can be the addressee. A
        # following personal salutation is positive recipient evidence, so
        # it must not be promoted to the invoice issuer by page position.
        if any(re.match(r"^\s*(?:Herrn?|Frau|Dear|Attn\.?|Attention\s*:|To\s*:)", line, re.I)
               for line in lines[index + 1:index + 3]):
            continue
        address_lines: list[str] = []
        contaminated_address = False
        for line in lines[index + 1:limit]:
            value = line.strip()
            if not value:
                continue
            if structural.search(value) or recipient.search(value):
                break
            # Flattened OCR can merge a reference printed in another column
            # into an address line. Withhold the address rather than emit a
            # mixed address/reference value.
            if re.search(r"\b\d{4,}[A-Z]{2,}\d+\b", value, re.I):
                contaminated_address = True
                break
            if not re.search(r"\d", value) and not re.search(r"\b(?:United States|USA|Germany|Estonia|Portugal|Switzerland)\b", value, re.I):
                break
            address_lines.append(value)
            if len(address_lines) >= 4:
                break
        name_field = ExtractedField(name, "found", [_evidence(page, name)])
        address = "" if contaminated_address else ", ".join(address_lines)
        address_field = ExtractedField(address, "found", [_evidence(page, address)]) if address else ExtractedField(status="missing")
        return name_field, address_field
    return ExtractedField(status="missing"), ExtractedField(status="missing")


def _footer_supplier_from_tax_block(pages: list[PageRecord]) -> tuple[ExtractedField, ExtractedField]:
    """Find a legal supplier entity adjacent to its own VAT registration block."""
    legal_suffix = re.compile(
        r"(?:\bO[ÃœU]\b|\bGmbH\b|\b(?:Pte\.?\s*)?Ltd\.?\b|\bLimited\b|\bLLC\b|\bPLC\b|"
        r"\bPty\.?\s*Ltd\.?\b|\bSdn\.?\s*Bhd\.?\b|\bLda\.?\b|\bInc\.?\b|\bCorp\.?\b|"
        r"\bS\.?A\.?\b|\bB\.?V\.?\b|\bAG\b|\bS\.?r\.?l\.?\b)", re.I,
    )
    vat_label = re.compile(r"\bvat\s*(?:id|no\.?|number|registration\s+no\.?|[:#])", re.I)
    buyer_context = re.compile(r"\b(?:bill\s*to|sold\s*to|ship\s*to|delivery\s+address|customer|buyer)\b", re.I)
    found: dict[str, tuple[str, Evidence]] = {}
    for page in pages:
        lines = page.text.splitlines()
        table_start = next((i for i, line in enumerate(lines)
                            if re.search(r"description|item\s+description", line, re.I)
                            and re.search(r"quantity|qty|unit\s+price|amount|total", line, re.I)), len(lines))
        header_entities = {
            _norm(line.strip()) for line in lines[:min(table_start, 40)]
            if legal_suffix.search(line) and len(legal_suffix.findall(line)) == 1
        }
        last_total = max((index for index, line in enumerate(lines)
                          if re.search(r"\b(?:grand\s+total|total\s+amount|invoice\s+total|amount\s+due)\b", line, re.I)),
                         default=-1)
        # Flattened two-column OCR can put a buyer's legal name inside the
        # seven-line neighborhood of the issuer's VAT label. In that case the
        # VAT proximity is not enough to assign the role.
        for vat_index, _line in enumerate(lines):
            if not vat_label.search(lines[vat_index]):
                continue
            # Multiple parties in the header make a header VAT label unsafe.
            # A separate legal entity and VAT block after the payable total
            # can still identify the issuer in a registered-office footer.
            if len(header_entities) > 1 and vat_index <= last_total:
                continue
            start = max(0, vat_index - 7)
            block = lines[start:vat_index + 1]
            if any(buyer_context.search(value) for value in block):
                continue
            entities: list[str] = []
            for candidate_line in lines[start:vat_index + 1]:
                value = candidate_line.strip()
                match = legal_suffix.search(value)
                if match and len(legal_suffix.findall(value)) == 1:
                    name = value[:match.end()].strip(" :|-")
                    if len(name.split()) >= 2:
                        entities.append(name)
            if len({_norm(name) for name in entities}) != 1:
                continue
            name = entities[0]
            found[_norm(name)] = (name, _evidence(page, " ".join(v.strip() for v in block if v.strip())))
    if len(found) != 1:
        return ExtractedField(status="missing"), ExtractedField(status="missing")
    name, evidence = next(iter(found.values()))
    return ExtractedField(name, "found", [evidence]), ExtractedField(status="missing")


def _vat_field_near_supplier(pages: list[PageRecord], supplier_name: str) -> ExtractedField:
    """Prefer the VAT ID located in the supplier's local legal-entity block."""
    normalized_name = _norm(supplier_name)
    labeled_vat = re.compile(
        r"(?:vat\s*(?:id|no\.?|number|registration\s+no\.?)|ust-idnr\.?|ustid|kmkr\s*(?:nr\.?|no\.?))"
        r"\s*[:#-]?\s*(?P<value>[A-Z]{0,3}[A-Z0-9][A-Z0-9 -]{4,})", re.I,
    )
    # A VAT number may be followed by postal or account-number columns on the
    # same OCR line. Keep the digit run contiguous so those later columns are
    # not silently absorbed into the identifier.
    bare_vat = re.compile(r"\bVAT\s*[:#]\s*(?P<value>[A-Z]{2}\s?\d{7,14})\b", re.I)
    found: list[tuple[str, Evidence, int]] = []
    for page in pages:
        lines = page.text.splitlines()
        supplier_indexes = [index for index, line in enumerate(lines) if normalized_name in _norm(line)]
        for supplier_index in supplier_indexes:
            for index in range(max(0, supplier_index - 2), min(len(lines), supplier_index + 10)):
                # A country-prefixed VAT token has a clear end. OCR often
                # appends unrelated postal/account columns after it.
                match = bare_vat.search(lines[index]) or labeled_vat.search(lines[index])
                if match:
                    value = re.sub(r"\s", "", match.group("value")).rstrip(".,;:")
                    found.append((value, _evidence(page, lines[index]), abs(index - supplier_index)))
    if not found:
        return ExtractedField(status="missing")
    nearest = min(distance for _, _, distance in found)
    found = [item for item in found if item[2] == nearest]
    values = list(dict.fromkeys(value.casefold() for value, _, _ in found))
    if len(values) != 1:
        return ExtractedField(None, "ambiguous", [ev for _, ev, _ in found], [v for v, _, _ in found],
                              "Multiple VAT identifiers occur near the supplier block.")
    return ExtractedField(found[0][0], "found", [ev for _, ev, _ in found])


def _ocr_header_address(pages: list[PageRecord], supplier_name: str) -> ExtractedField:
    """Recover the address adjacent to an already identified OCR header name."""
    country = re.compile(r"\b(?:United States|USA|Germany|Estonia|Portugal|Switzerland)\b", re.I)
    metadata = re.compile(r"\b(?:Federal Tax ID|Invoice Number|Invoice Date|Due Date|Payment Terms|Your P\.?O\.?)\b", re.I)
    for page in pages:
        table_ocr = page.evidence.get("table_ocr", {}) if isinstance(page.evidence, dict) else {}
        text = table_ocr.get("text", "") if isinstance(table_ocr, dict) else ""
        if not text:
            continue
        lines = text.splitlines()
        for index, line in enumerate(lines):
            if _norm(supplier_name) not in _norm(line):
                continue
            addresses: list[str] = []
            for following in lines[index + 1:index + 5]:
                value = metadata.split(following, maxsplit=1)[0].strip(" :|-\t")
                value = re.sub(r"\b\d{1,2}-[A-Za-z]{3,9}-\d{4}\b.*$", "", value).strip(" :|-\t")
                if not value:
                    continue
                if re.search(r"\b(?:bill\s*to|ship\s*to|payment instructions)\b", value, re.I):
                    break
                if re.search(r"\d", value) or country.search(value):
                    addresses.append(value)
                elif addresses:
                    break
            if addresses:
                value = ", ".join(addresses)
                return ExtractedField(value, "found", [_evidence(page, value)])
    return ExtractedField(status="missing")


def _date_value(raw: str) -> date:
    value = raw.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y", "%d-%b-%Y", "%d-%B-%Y", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"unrecognized date: {value}")


def _page_uses_day_first(text: str) -> bool:
    """Use an unambiguous numeric date on the same page to resolve its peers."""
    dates = [re.split(r"[./-]", raw) for raw in re.findall(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{4}\b", text)]
    dmy = any(int(parts[0]) > 12 and int(parts[1]) <= 12 for parts in dates)
    mdy = any(int(parts[1]) > 12 and int(parts[0]) <= 12 for parts in dates)
    return dmy and not mdy


def _date_field(pages: list[PageRecord], pattern: str, *, exclude_pattern: str | None = None) -> ExtractedField:
    ambiguous: list[tuple[str, Evidence]] = []
    for page in pages:
        lines = page.text.splitlines()
        for index, line in enumerate(lines):
            if exclude_pattern and re.search(exclude_pattern, line, re.I):
                continue
            if re.search(r"invoice", pattern, re.I) and re.search(r"\bdue\s+date\b", line, re.I):
                continue
            if re.search(r"\b(?:bill|billing|statement|delivery)\s+date\b", line, re.I):
                continue
            match = re.search(pattern + r"\s*[:#-]?\s*" + _DATE, line, re.I)
            if not match:
                label = re.search(pattern, line, re.I)
                neighboring = lines[index + 1:index + 2] if label else []
                candidates = [(date_match.group("date"), source_line.strip())
                              for source_line in neighboring
                              for date_match in re.finditer(_DATE, source_line, re.I)]
                distinct_dates = list(dict.fromkeys(value for value, _source in candidates))
                if len(distinct_dates) != 1:
                    continue
                raw = distinct_dates[0]
                date_source = next(source for value, source in candidates if value == raw)
                source = f"{line.strip()} {date_source}"
            else:
                raw = match.group("date")
                source = line
            parts = re.split(r"[./-]", raw)
            locale_dmy = bool(re.search(r"rechnungsdatum|\brechnung\b|\barve\b|makset[aä]htaeg|fatura|factura|mwst|ใบวางบิล", _norm(page.text)))
            if (len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit()
                    and len(parts[0]) <= 2 and len(parts[2]) == 4 and 1 <= int(parts[0]) <= 12
                    and 1 <= int(parts[1]) <= 12 and parts[0] != parts[1]
                    and not (locale_dmy or _page_uses_day_first(page.text))):
                ambiguous.append((raw, _evidence(page, source)))
                continue
            try:
                ambiguous.append((str(_date_value(raw)), _evidence(page, source)))
            except ValueError:
                continue
    if not ambiguous:
        status = "unreadable" if _readability(pages) == "unreadable" or any(p.error for p in pages) else "missing"
        return ExtractedField(status=status)
    values = list(dict.fromkeys(value for value, _ in ambiguous))
    evidence = [source for _, source in ambiguous]
    if len(values) > 1:
        raw_ambiguous = any("/" in item.text and re.search(r"\b\d{1,2}/\d{1,2}/\d{4}\b", item.text) for item in evidence)
        return ExtractedField(None, "ambiguous" if raw_ambiguous else "conflicting", evidence, values)
    raw = values[0]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return ExtractedField(None, "ambiguous", evidence, [raw], "Day/month order cannot be determined.")
    return ExtractedField(date.fromisoformat(raw), "found", evidence)


def _due_date_field(pages: list[PageRecord]) -> ExtractedField:
    due_pattern = r"(?:due\s+date|payment\s+due|total\s+amount\s+due\s+on|payable\s+by|bis\s+zum|f[aä]llig(?:keit)?(?:sdatum)?|zahlbar\s+bis|makset[aä]htaeg|fecha\s+de\s+vencimiento|data\s+de\s+vencimento)"
    result = _date_field(pages, due_pattern)
    if result.status != "found":
        return result
    for page in pages:
        lines = page.text.splitlines()
        for index, line in enumerate(lines):
            if not re.fullmatch(r"\s*Due\s+Date\s*:?\s*", line, re.I):
                continue
            next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
            if re.fullmatch(_DATE, next_line):
                continue
            return ExtractedField(None, "ambiguous", [*result.evidence, _evidence(page, line)],
                                  [result.value], "A separate Due Date label has no reliably associated value and may conflict with the parsed due date.")
    return result


def _amounts(line: str, *, table_row: bool = False) -> list[tuple[Decimal, re.Match[str]]]:
    found: list[tuple[Decimal, re.Match[str]]] = []
    for match in (_TABLE_MONEY if table_row else _MONEY).finditer(line):
        raw = match.group("number")
        # OCR can place date fragments beside a printed-total label.
        if any(m.start() <= match.start() and match.end() <= m.end()
               for m in re.finditer(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", line)):
            continue
        # A whole-number quantity followed by a unit is not a currency amount.
        if not re.search(r"[,.]\d{2,6}\b", raw) and re.match(
            r"\s+[A-Za-z\u00c0-\u024f]{1,8}\b", line[match.end():]
        ):
            continue
        # Integer quantities and identifiers are not money unless currency or a decimal mark supports it.
        context = line[max(0, match.start() - 10): min(len(line), match.end() + 10)]
        if not re.search(r"[,.]\d{2,6}\b", raw) and not re.search(r"€|£|\b(?:EUR|USD|GBP|SGD|THB|BAHT|INR|CAD|AUD|JPY|CHF)\b", context, re.I):
            continue
        parsed = parse_localized_decimal(raw)
        if parsed.value is not None:
            found.append((parsed.value, match))
    return found


def _line_items(pages: list[PageRecord]) -> list[LineItem]:
    rows: list[LineItem] = []
    header = re.compile(r"description|beschreibung|descri[cç][aã]o|descripci[oó]n|designation|quantity|menge|quantit|quantidade|unit price|einzelpreis|pre[cç]o unit[aá]rio|precio unitario|product reference|pos\.?\s+|toode/teenus|item\b|hind\s+kokku\s+käibemaksuta", re.I)
    line_pages: list[PageRecord] = []
    for page in pages:
        table_ocr = page.evidence.get("table_ocr", {}) if isinstance(page.evidence, dict) else {}
        if not (isinstance(table_ocr, dict) and table_ocr.get("status") == "found" and table_ocr.get("text")):
            table_ocr = page.evidence.get("activity_table_ocr", {}) if isinstance(page.evidence, dict) else {}
        if isinstance(table_ocr, dict) and table_ocr.get("status") == "found" and table_ocr.get("text"):
            line_pages.append(PageRecord(page.filename, page.page_number, table_ocr["text"], "table_ocr",
                                         {"native_text": page.text, "psm": table_ocr.get("psm")}))
        else:
            line_pages.append(page)
    for page in line_pages:
        in_table = False
        table_header = ""
        end = re.compile(r"subtotal|vahesumma|summa\s+(?:ilma\s+käibemaksuta|koos\s+käibemaksuga)|total amount|gross total|grand total|endbetrag|vat\s+\d|mwst|total due|amount due|balance|terms of payment|payment due|arve kokku|total current charges", re.I)
        lines = page.text.splitlines()
        activity_header_index = next((index for index, value in enumerate(lines)
                                      if re.search(r"\bactivity\b", value, re.I)
                                      and re.search(r"\btax\b", value, re.I)
                                      and re.search(r"\bqty\b|quantity|\batv\b", value, re.I)
                                      and re.search(r"\brate\b", value, re.I)
                                      and re.search(r"\bamount\b", value, re.I)), None)
        if activity_header_index is not None:
            row_pattern = re.compile(
                rf"^\s*(?P<description>.+?)\s+(?P<tax_class>[A-Za-z]{{2,8}})\s+"
                rf"(?P<quantity>{_NUMBER})\s+(?P<price>{_NUMBER})\s+(?P<total>{_NUMBER})\s*$", re.I,
            )
            activity_rows: list[LineItem] = []
            for row in lines[activity_header_index + 1:]:
                if re.search(r"\b(?:subtotal|tax\s+summary|total\s+due)\b", row, re.I):
                    break
                match = row_pattern.match(row)
                if not match:
                    continue
                parsed = [parse_localized_decimal(match.group(key)) for key in ("quantity", "price", "total")]
                if any(value.value is None for value in parsed):
                    continue
                description = match.group("description").strip()
                evidence = [_evidence(page, row.strip())]
                item_type = classify_line_type(description)
                activity_rows.append(LineItem(
                    ExtractedField(description, "found", evidence),
                    ExtractedField(item_type, "found", evidence) if item_type else ExtractedField(status="missing"),
                    ExtractedField(parsed[0].value, "found", evidence), ExtractedField(status="missing"),
                    ExtractedField(parsed[1].value, "found", evidence),
                    ExtractedField(status="missing", note="Net basis requires agreement between the printed line extensions, subtotal, header tax, and gross."),
                    ExtractedField(parsed[2].value, "found", evidence),
                ))
            if activity_rows:
                rows.extend(activity_rows)
                continue
        product_header_index = next((index for index, value in enumerate(lines)
                                     if re.search(r"product", value, re.I)
                                     and re.search(r"reference", value, re.I)
                                     and re.search(r"quantity|auantity", value, re.I)
                                     and re.search(r"unit\s+price", value, re.I)
                                     and re.search(r"net\s+amount", value, re.I)
                                     and re.search(r"\bvat\b", value, re.I)), None)
        if product_header_index is not None:
            row_pattern = re.compile(
                rf"^\s*(?P<item>\d{{4,}})\s+(?P<ref>[A-Z0-9-]{{4,}})\s+"
                rf"(?P<qty>\d+(?:[.,]\d+)?)\s+(?P<rate>\d+(?:[.,]\d+)?)\s*%\s+"
                rf"(?P<price>{_NUMBER})\s+(?P<total>{_NUMBER})\s*$", re.I,
            )
            table_rows: list[LineItem] = []
            for row_index in range(product_header_index + 1, len(lines)):
                match = row_pattern.match(lines[row_index])
                if not match:
                    continue
                description_index = row_index + 1
                description = lines[description_index].strip() if description_index < len(lines) else ""
                if not description or row_pattern.match(description):
                    continue
                quantity = parse_localized_decimal(match.group("qty"))
                rate = parse_localized_decimal(match.group("rate"))
                price = parse_localized_decimal(match.group("price"))
                total = parse_localized_decimal(match.group("total"))
                if any(value.value is None for value in (quantity, rate, price, total)):
                    continue
                source = f"{lines[row_index].strip()} {description}"
                evidence = [_evidence(page, source)]
                item_type = classify_line_type(description)
                table_rows.append(LineItem(
                    ExtractedField(description, "found", evidence),
                    ExtractedField(item_type, "found", evidence) if item_type else ExtractedField(status="missing"),
                    ExtractedField(quantity.value, "found", evidence),
                    ExtractedField(status="missing"),
                    ExtractedField(price.value, "found", evidence),
                    ExtractedField("net", "found", [_evidence(page, lines[product_header_index])]),
                    ExtractedField(total.value, "found", evidence),
                    taxes=[TaxItem(
                        "line", "tax",
                        ExtractedField("VAT", "found", [_evidence(page, lines[product_header_index])]),
                        ExtractedField(rate.value, "found", evidence),
                        ExtractedField(status="missing"),
                    )],
                ))
            if table_rows:
                rows.extend(table_rows)
                continue
        # Label-driven column table parsing. Unlike the legacy generic path
        # below, this does not require an ordinal at the start of each row;
        # it uses the table's own column labels and accepts only a complete
        # numeric suffix that fits those labels.
        column_rows = _column_table_rows(page, lines, end)
        if column_rows:
            rows.extend(column_rows)
            continue
        for line_index, line in enumerate(lines):
            if header.search(line):
                in_table = True
                native_context = str(page.evidence.get("native_text", "")) if isinstance(page.evidence, dict) else ""
                table_header = line + " " + native_context
                continue
            if end.search(line):
                in_table = False
            if not in_table:
                continue
            row = re.match(
                r"^\s*(?:(\d{1,3})[.)]?\s+(.+)|(\d+(?:[.,]\d+)?)\s*x\s+(.+))$",
                line,
                re.I,
            )
            if not row:
                continue
            values = _amounts(line, table_row=True)
            is_quantity_x_row = row.group(3) is not None
            description_start = row.start(4) if is_quantity_x_row else row.start(2)
            if is_quantity_x_row and len(values) < 2:
                # Kit component rows have unit prices but no printed extensions.
                continue
            # Some PDF text layers split a labelled amount column onto the next
            # line. Join only a numeric-only adjacent row after an item row.
            source_line = line
            next_line = lines[line_index + 1].strip() if line_index + 1 < len(lines) else ""
            if len(values) == 1 and re.search(r"amount|total|gesamtpreis|summa", table_header, re.I) and re.fullmatch(r"[€$£]?\s*\d[\d., ]*\s*(?:€|EUR|USD|GBP)?", next_line, re.I):
                adjacent_values = _amounts(next_line, table_row=True)
                if len(adjacent_values) == 1:
                    source_line = f"{line} {next_line}"
                    values = _amounts(source_line, table_row=True)
                    line_index += 1
            first_amount = values[0][1] if values else None
            description = line[description_start:(first_amount.start() if first_amount else len(line))].strip(" :|-")
            description = re.sub(
                r"\s+(?:\d+(?:[.,]\d+)?\s+(?:tundi|päev)|fikseeritud)\s*[€$£]?\s*$",
                "",
                description,
                flags=re.I,
            )
            if page.extraction_method == "table_ocr":
                continuations: list[str] = []
                for continuation in lines[line_index + 1:]:
                    if end.search(continuation) or re.match(r"^\s*\d{1,3}[.)]?\s+\S", continuation):
                        break
                    if _amounts(continuation, table_row=True):
                        break
                    if re.search(r"[A-Za-z]", continuation):
                        continuations.append(continuation.strip())
                if continuations:
                    description = " ".join([description, *continuations]).strip()
                    source_line = " ".join([source_line, *continuations])
            if not description or not values:
                continue
            between = line[first_amount.end():values[-1][1].start()] if len(values) > 1 else ""
            quantity_label = re.search(r"\bqty\b|quantity|menge|quantit|anzahl|quantidade|cant\.?", table_header, re.I)
            price_label = re.search(r"unit\s+price|einzelpreis|price|preis|pre[cç]o|precio", table_header, re.I)
            quantity_before_price = bool(quantity_label and price_label and quantity_label.start() < price_label.start())
            quantity_match = None if quantity_before_price else re.search(r"(?<!\w)(\d+(?:[.,]\d+)?)\s*([\w.]+)?", between)
            quantity = ExtractedField(status="missing")
            unit = ExtractedField(status="missing")
            if is_quantity_x_row:
                parsed_qty = parse_localized_decimal(row.group(3))
                if parsed_qty.value is not None:
                    quantity = ExtractedField(parsed_qty.value, "found", [_evidence(page, line)])
            if quantity_match:
                parsed_quantity = parse_localized_decimal(quantity_match.group(1))
                if parsed_quantity.value is not None:
                    ev = [_evidence(page, line)]
                    quantity = ExtractedField(parsed_quantity.value, "found", ev)
                    if quantity_match.group(2):
                        unit = ExtractedField(quantity_match.group(2).rstrip("."), "found", ev)
            if quantity.status == "missing" and re.search(r"quantity|menge|quantit|anzahl|quantidade|cant\.?", table_header, re.I):
                prefix = line[description_start:first_amount.start()]
                if is_quantity_x_row:
                    parsed_qty = parse_localized_decimal(row.group(3))
                    if parsed_qty.value is not None:
                        quantity = ExtractedField(parsed_qty.value, "found", [_evidence(page, line)])
                prefix_numbers = list(re.finditer(r"(?<!\w)(\d+(?:[.,]\d+)?)(?:\s+([\w.]+))?\s*$", prefix))
                digit_tokens = re.findall(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)", prefix)
                prefix_has_numeric_description = bool(prefix_numbers and re.search(r"\d", prefix[:prefix_numbers[-1].start()]))
                if prefix_numbers and len(digit_tokens) == 1 and not prefix_has_numeric_description:
                    quantity_match = prefix_numbers[-1]
                    parsed_quantity = parse_localized_decimal(quantity_match.group(1))
                    if parsed_quantity.value is not None:
                        quantity = ExtractedField(parsed_quantity.value, "found", [_evidence(page, line)])
                        if quantity_match.group(2):
                            unit = ExtractedField(quantity_match.group(2).rstrip("."), "found", [_evidence(page, line)])
                elif prefix_numbers and (len(digit_tokens) > 1 or prefix_has_numeric_description):
                    quantity = ExtractedField(None, "ambiguous", [_evidence(page, line)], digit_tokens,
                                              "Several numeric tokens precede the unit price; their quantity role is unclear.")
            no_tax_cell = None
            if first_amount:
                prefix = line[description_start:first_amount.start()]
                # In tables that label the quantity and tax-status columns,
                # the quantity immediately before an N tax marker is not part
                # of the description (which may itself contain dates).
                no_tax_cell = re.search(r"(?P<qty>\d+(?:[.,]\d+)?)\s+N\s*$", prefix, re.I)
                if (no_tax_cell and re.search(r"\bqty\b|quantity|no\.?\s+of\s+users", table_header, re.I)
                        and re.search(r"\btax\b", table_header, re.I)
                        and re.search(r"extended\s+amount", table_header, re.I)):
                    parsed_qty = parse_localized_decimal(no_tax_cell.group("qty"))
                    if parsed_qty.value is not None:
                        quantity = ExtractedField(parsed_qty.value, "found", [_evidence(page, line)])
                        if re.search(r"no\.?\s+of\s+users", table_header, re.I):
                            unit = ExtractedField("users", "found", [_evidence(page, "QTY/ NO. OF USERS")])
                        description = re.sub(r"\s+\d+(?:[.,]\d+)?\s+N\b", "", description, flags=re.I)
                leading_qty = re.search(
                    r"(?<!\w)(?P<qty>\d+(?:[.,]\d+)?)\s+(?P<unit>[A-Za-z]{1,8})\s+(?:USD|EUR|GBP|SGD|THB|INR|CAD|AUD|JPY|CHF)\s*$",
                    prefix, re.I,
                )
                if leading_qty:
                    raw_qty = leading_qty.group("qty")
                    parsed_qty = parse_localized_decimal(raw_qty)
                    quantity_value = parsed_qty.value
                    # In English-style tables a dot-decimal unit price establishes
                    # the decimal convention for a three-place quantity such as
                    # 567.000, which is otherwise ambiguous in isolation.
                    if parsed_qty.ambiguous and "." in raw_qty and "," not in raw_qty and re.search(r"\d\.\d{2}\b", line):
                        quantity_value = Decimal(raw_qty)
                    if quantity_value is not None:
                        evidence = [_evidence(page, line)]
                        quantity = ExtractedField(quantity_value, "found", evidence)
                        unit = ExtractedField(leading_qty.group("unit"), "found", evidence)
                        description = prefix[:leading_qty.start()].strip(" :|-")
            item_type = classify_line_type(description)
            evidence = [_evidence(page, source_line)]
            if re.search(r"gross\s+price|incl(?:uding)?\.?\s+(?:vat|tax)|tax.inclusive|brutto|com\s+iva|con\s+iva", table_header, re.I):
                price_basis = ExtractedField("tax_inclusive", "found", [_evidence(page, table_header)])
            elif re.search(r"net\s+price|excl(?:uding)?\.?\s+(?:vat|tax)|netto|sem\s+iva|sin\s+iva|käibemaksuta", table_header, re.I):
                price_basis = ExtractedField("net", "found", [_evidence(page, table_header)])
            elif (add_tax := re.search(r"\b(?:zzgl\.?|plus)\s*\d{1,2}(?:[,.]\d+)?\s*%\s*(?:mwst|ust|vat|gst|iva|tva)\b", page.text, re.I)):
                price_basis = ExtractedField("net", "found", [_evidence(page, add_tax.group(0))])
            else:
                price_basis = ExtractedField(status="missing", note="The table does not state whether prices include tax.")
            if no_tax_cell and re.search(r"\btax\b", table_header, re.I):
                price_basis = ExtractedField("net", "found", [_evidence(page, f"{table_header}; tax marker N")])
            line_taxes: list[TaxItem] = []
            tax_label = re.search(r"vat|mwst|ust\b|gst|iva|tva|moms|km\b", line, re.I) or re.search(
                r"vat|mwst|ust\b|gst|iva|tva|moms|km\b", table_header, re.I,
            )
            tax_rate = re.search(r"(?P<rate>\d{1,2}(?:[,.]\d+)?)\s*%", line)
            if tax_label and tax_rate:
                explicit_tax_amount_column = re.search(r"(?:vat|tax|gst|mwst).{0,20}amount|amount.{0,20}(?:vat|tax|gst|mwst)", table_header, re.I)
                amounts_after_rate = [value for value in values if value[1].start() > tax_rate.end()]
                if explicit_tax_amount_column and len(amounts_after_rate) == 1:
                    line_tax_amount = ExtractedField(amounts_after_rate[0][0], "found", evidence)
                elif explicit_tax_amount_column and len(amounts_after_rate) > 1:
                    line_tax_amount = ExtractedField(None, "ambiguous", evidence, [value[0] for value in amounts_after_rate])
                else:
                    line_tax_amount = ExtractedField(status="missing")
                line_taxes.append(TaxItem(
                    "line", "tax",
                    ExtractedField(tax_label.group(0), "found", evidence),
                    ExtractedField(parse_localized_decimal(tax_rate.group("rate")).value, "found", evidence),
                    line_tax_amount,
                ))
            unit_price = ExtractedField(values[0][0], "found", evidence) if len(values) >= 2 else ExtractedField(status="missing")
            printed_total = ExtractedField(values[-1][0], "found", evidence) if len(values) >= 2 or re.search(r"amount|total|gesamtpreis|summa", table_header, re.I) else ExtractedField(status="missing")
            if len(values) == 1:
                if printed_total.status != "found":
                    uncertain_amount = ExtractedField(None, "ambiguous", evidence, [values[0][0]],
                                                       "A single amount appears on the item row; its field role cannot be identified.")
                    unit_price = uncertain_amount
                    printed_total = uncertain_amount
            rows.append(LineItem(
                ExtractedField(description, "found", evidence),
                ExtractedField(item_type, "found", evidence) if item_type else ExtractedField(status="missing"),
                quantity,
                unit,
                unit_price,
                price_basis,
                printed_total,
                line_taxes,
            ))
    return rows


def _column_table_rows(page: PageRecord, lines: list[str], end: re.Pattern[str]) -> list[LineItem]:
    """Extract rows from common description/quantity/price/tax/amount tables.

    Table labels determine the roles of trailing values. Rows without enough
    printed values are retained with missing fields, never completed by math.
    """
    header_index = next((i for i, line in enumerate(lines)
                         if re.search(r"description|item|service|product", line, re.I)
                         and re.search(r"amount|total|value", line, re.I)), None)
    if header_index is None:
        return []
    table_header = lines[header_index]
    has_quantity = bool(re.search(r"quantity|\bqty\b", table_header, re.I))
    has_price = bool(re.search(r"unit\s*price|price\s+per\s+unit", table_header, re.I))
    has_tax_column = bool(re.search(r"\bvat\b|\btax\b|\bgst\b", table_header, re.I))
    rows: list[LineItem] = []
    summary_end = re.compile(r"\b(?:subtotal|grand\s+total|invoice\s+total|amount\s+due|balance\s+due|tax\s+summary|total\s+(?:vat|tax|gst))\b", re.I)
    row_pattern = re.compile(
        rf"^\s*(?P<description>.+?)\s+(?P<quantity>{_NUMBER})\s+"
        rf"(?P<price>{_NUMBER})(?:\s+(?P<tax>(?:\d+(?:[.,]\d+)?\s*%|no\s+(?:vat|tax)|exempt)))?\s+"
        rf"(?P<amount>{_NUMBER})\s*$", re.I,
    )
    amount_only = re.compile(rf"^\s*(?P<description>.+?)\s+(?P<amount>{_NUMBER})\s*$", re.I)
    pending_description: list[str] = []
    for line in lines[header_index + 1:]:
        if summary_end.search(line):
            break
        if not line.strip() or re.search(r"\b(?:total\s+vat|tax\s+summary|subtotal)\b", line, re.I):
            continue
        # Numbered rows are handled by the established parser below, which
        # retains its ambiguity checks for numeric descriptions and quantities.
        if re.match(r"^\s*\d{1,3}[.)]?\s+", line):
            pending_description.clear()
            continue
        match = row_pattern.match(line) if has_quantity and has_price else None
        evidence = [_evidence(page, line.strip())]
        if match:
            quantity = parse_localized_decimal(match.group("quantity"))
            price = parse_localized_decimal(match.group("price"))
            amount = parse_localized_decimal(match.group("amount"))
            if any(value.value is None for value in (quantity, price, amount)):
                continue
            description = " ".join([*pending_description, match.group("description").strip()]).strip()
            if pending_description:
                evidence = [_evidence(page, " ".join([*pending_description, line.strip()]))]
            pending_description.clear()
            qty_field = ExtractedField(quantity.value, "found", evidence)
            price_field = ExtractedField(price.value, "found", evidence)
            amount_field = ExtractedField(amount.value, "found", evidence)
            tax_cell = (match.group("tax") or "").strip()
            line_taxes: list[TaxItem] = []
            if has_tax_column and re.search(r"\d", tax_cell):
                rate_match = re.search(r"\d+(?:[.,]\d+)?", tax_cell)
                parsed_rate = parse_localized_decimal(rate_match.group()) if rate_match else None
                if parsed_rate and parsed_rate.value is not None:
                    tax_name = re.search(r"\b(?:VAT|GST|tax)\b", table_header, re.I)
                    line_taxes.append(TaxItem(
                        "line", "tax",
                        ExtractedField(tax_name.group(0) if tax_name else "Tax", "found", [_evidence(page, table_header)]),
                        ExtractedField(parsed_rate.value, "found", evidence),
                        ExtractedField(status="missing"),
                    ))
            elif has_tax_column and re.search(r"no\s+(?:vat|tax)|exempt", tax_cell, re.I):
                # Explicit zero-tax status is retained as a zero-rate line tax.
                tax_name = re.search(r"\b(?:VAT|GST|tax)\b", table_header, re.I)
                line_taxes.append(TaxItem(
                    "line", "tax",
                    ExtractedField(tax_name.group(0) if tax_name else "Tax", "found", [_evidence(page, table_header)]),
                    ExtractedField(Decimal("0"), "found", evidence),
                    ExtractedField(Decimal("0"), "found", evidence),
                ))
            item_type = classify_line_type(description)
            basis = "net" if re.search(r"\bnet\b|excl(?:uding)?\.?\s+(?:tax|vat)|excluding\s+(?:tax|vat)", table_header, re.I) else None
            if not basis and re.search(r"gross|including\s+(?:tax|vat)", table_header, re.I):
                basis = "tax_inclusive"
            rows.append(LineItem(
                ExtractedField(description, "found", evidence),
                ExtractedField(item_type, "found", evidence) if item_type else ExtractedField(status="missing"),
                qty_field, ExtractedField(status="missing"), price_field,
                ExtractedField(basis, "found", [_evidence(page, table_header)]) if basis else ExtractedField(status="missing"),
                amount_field, line_taxes,
            ))
            continue
        # A prose-only line directly before a complete numeric row can be a
        # wrapped description. Keep it only inside a labelled table, and do
        # not turn a value-free row into an accounting component.
        if has_quantity and has_price and re.search(r"[A-Za-z]", line) and not _amounts(line, table_row=True):
            pending_description.append(line.strip())
            if len(pending_description) > 2:
                pending_description.pop(0)
        else:
            pending_description.clear()
        if not has_quantity and not has_price:
            match = amount_only.match(line)
            if match:
                amount = parse_localized_decimal(match.group("amount"))
                if amount.value is None:
                    continue
                description = match.group("description").strip()
                # Ignore row labels that are totals or tax summaries.
                if re.search(r"\b(?:subtotal|total|vat|tax|balance|amount due|discount|freight|shipping)\b", description, re.I):
                    continue
                rows.append(LineItem(
                    ExtractedField(description, "found", evidence), ExtractedField(status="missing"),
                    ExtractedField(status="missing"), ExtractedField(status="missing"),
                    ExtractedField(status="missing", note="The table provides an amount only."),
                    ExtractedField(status="missing"), ExtractedField(amount.value, "found", evidence),
                ))
    return rows


def _establish_net_basis_from_document_totals(
    lines: list[LineItem], subtotal: ExtractedField, tax_total: ExtractedField,
    gross_total: ExtractedField, taxes: list[TaxItem], pages: list[PageRecord], charges: list[ChargeItem],
) -> None:
    """Use a fully itemized subtotal/tax/gross chain to establish net line amounts."""
    if (not lines or charges or subtotal.status != "found" or tax_total.status != "found"
            or gross_total.status != "found" or not taxes):
        return
    if any(line.price_basis.status == "found" for line in lines):
        return
    if any(field.status != "found" for line in lines
           for field in (line.quantity, line.unit_price, line.printed_total)):
        return
    tax_records = [tax for tax in taxes if tax.scope == "header" and tax.kind == "tax"]
    if (len(tax_records) != 1 or tax_records[0].rate.status != "found"
            or tax_records[0].amount.status != "found"
            or Decimal(str(tax_records[0].amount.value)) != Decimal(str(tax_total.value))):
        return
    line_sum = sum((Decimal(str(line.printed_total.value)) for line in lines), Decimal("0"))
    if (line_sum != Decimal(str(subtotal.value))
            or line_sum + Decimal(str(tax_total.value)) != Decimal(str(gross_total.value))):
        return
    total_evidence = [evidence for field in (subtotal, tax_total, gross_total)
                      for evidence in field.evidence]
    for line in lines:
        line.price_basis = ExtractedField(
            "net", "found", list(dict.fromkeys([*line.price_basis.evidence, *total_evidence])),
            note="The source explicitly itemizes line extensions summing to its subtotal, then states header tax and gross equal subtotal plus tax.",
        )


def _tax_items(pages: list[PageRecord]) -> list[TaxItem]:
    result: list[TaxItem] = []
    tax_cue = re.compile(r"\bvat\b|\bmwst\b|\bust\b|\bgst\b|\biva\b|\btva\b|\bmoms\b|k[aä]ibemaks|imposto|impuesto|\bkm\b|withholding|reverse charge|reten|kinnipid", re.I)
    rate_re = re.compile(r"(?P<rate>\d{1,2}(?:[,.]\d+)?)\s*%")
    for page in pages:
        in_table = False
        tax_summary_columns = False
        for line in page.text.splitlines():
            if re.search(r"\brate\b.*\btax\b.*\bnet\b", line, re.I):
                tax_summary_columns = True
                continue
            if tax_summary_columns and line.strip() and not tax_cue.search(line):
                tax_summary_columns = False
            if re.search(r"description|beschreibung|designation|quantity|menge|unit price|einzelpreis|product reference|toode/teenus|item\b", line, re.I):
                in_table = True
                continue
            if re.search(r"subtotal|vahesumma|summa\s+(?:ilma\s+käibemaksuta|koos\s+käibemaksuga)|allahindlus|tasumata|total amount|gross total|grand total|endbetrag|total\s+(?:vat|tax|gst)|(?<!no )vat\s*@?\s*\d|mwst\s+\d|km\s+\d|total due|payment due|arve kokku", line, re.I):
                in_table = False
            if re.search(r"vahesumma|summa\s+(?:ilma\s+käibemaksuta|koos\s+käibemaksuga)|allahindlus|tasumata|käibemaksuta|^\s*hind\b", line, re.I):
                continue
            if in_table:
                continue
            if re.search(r"\bkm\s*-\s*ta\b", line, re.I):
                continue
            if re.search(r"\b(?:no|without)\s+(?:vat|tax)\b", line, re.I):
                # This is an explicit zero-tax bucket/status, not a separate
                # payable-level tax component. Line scope is retained by the
                # table parser when the source marks individual rows.
                continue
            if not tax_cue.search(line):
                continue
            if re.search(r"\bvat\s*(?:id|no\.?|number|registration)|\bustid\b|ust-idnr", line, re.I):
                continue
            if re.search(r"\bvat\s*[:#]\s*[A-Z]{2}\s*\d[\d ]{6,}\b", line, re.I):
                continue
            if re.search(r"grand\s+total|gross\s+total|total\s+amount|invoice\s+total", line, re.I):
                continue
            rate_match = rate_re.search(line)
            amounts = _amounts(line)
            is_withholding = bool(re.search(r"withholding|reten|kinnipid", line, re.I))
            # "Reverse charge" by itself describes the tax treatment; without a
            # rate or amount it is not a separate tax component for the schema.
            if re.search(r"\breverse\s+charge\b", line, re.I) and not rate_match and not amounts:
                continue
            tax_name = re.search(r"withholding\s+tax|reverse\s+charge", line, re.I) or tax_cue.search(line)
            result.append(TaxItem(
                "header",
                "withholding" if is_withholding else "unknown" if "reverse charge" in line.casefold() and not amounts else "tax",
                ExtractedField(re.sub(r"[^\w -]", "", tax_name.group(0)), "found", [_evidence(page, line)]),
                ExtractedField(parse_localized_decimal(rate_match.group("rate")).value, "found", [_evidence(page, line)]) if rate_match else ExtractedField(status="missing"),
                ExtractedField((amounts[0] if tax_summary_columns and len(amounts) > 1 else amounts[-1])[0], "found", [_evidence(page, line)]) if amounts else ExtractedField(status="missing"),
            ))
    unique: dict[tuple[Any, ...], TaxItem] = {}
    for item in result:
        key = (item.scope, item.kind, item.name.status, item.name.value,
               item.rate.status, item.rate.value, item.amount.status, item.amount.value)
        prior = unique.get(key)
        if prior is None:
            unique[key] = item
            continue
        for target, source in ((prior.name, item.name), (prior.rate, item.rate), (prior.amount, item.amount)):
            target.evidence.extend(evidence for evidence in source.evidence if evidence not in target.evidence)
    return list(unique.values())


def _remove_line_tax_summaries(taxes: list[TaxItem], lines: list[LineItem], tax_total: ExtractedField) -> list[TaxItem]:
    """Avoid adding a tax summary again when the source itemizes that tax by line."""
    if tax_total.status != "found" or not lines or any(not line.taxes for line in lines):
        return taxes
    if any(line.printed_total.status != "found" or len(line.taxes) != 1
           or line.taxes[0].rate.status != "found" for line in lines):
        return taxes
    calculated = Decimal("0")
    for line in lines:
        tax = line.taxes[0]
        if tax.amount.status == "found":
            calculated += Decimal(str(tax.amount.value))
        else:
            base = Decimal(str(line.printed_total.value))
            rate = Decimal(str(tax.rate.value))
            calculated += (base * rate / Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if calculated != Decimal(str(tax_total.value)):
        return taxes
    result: list[TaxItem] = []
    for tax in taxes:
        same_amount = (tax.amount.status == "found"
                       and Decimal(str(tax.amount.value)) == Decimal(str(tax_total.value)))
        if tax.scope == "header" and tax.kind == "tax" and same_amount:
            continue
        result.append(tax)
    return result


def _charges(pages: list[PageRecord]) -> list[ChargeItem]:
    cues = {
        "discount": r"discount|allahindlus|rabatt|skonto|descuento|desconto",
        "freight": r"freight|shipping|delivery charge|transport|porto|fracht|frete",
        "insurance": r"insurance|assurance|versicherung|seguro",
        "excise_duty": r"excise duty|excise tax|accise|impuesto especial",
        "extra_charge": r"surcharge|service charge|management fee|extra charge|handling fee",
    }
    results: list[ChargeItem] = []
    for page in pages:
        in_item_table = False
        for line in page.text.splitlines():
            if (re.search(r"\bactivity\b", line, re.I) and re.search(r"\bqty\b|quantity|\batv\b", line, re.I)
                    and re.search(r"\bamount\b", line, re.I)) or (
                    re.search(r"description", line, re.I) and re.search(r"quantity|qty", line, re.I)
                    and re.search(r"amount|total", line, re.I)):
                in_item_table = True
                continue
            if in_item_table and re.search(r"\b(?:subtotal|tax\s+summary|grand\s+total|total\s+due|amount\s+due)\b", line, re.I):
                in_item_table = False
            if in_item_table:
                continue
            for kind, cue in cues.items():
                if re.search(cue, line, re.I):
                    if kind == "freight" and re.match(r"\s*\d+(?:[.,]\d+)?\s*x\s+.*transport\b", line, re.I):
                        continue
                    amounts = _amounts(line)
                    results.append(ChargeItem(
                        kind,
                        ExtractedField(line.strip(), "found", [_evidence(page, line)]),
                        ExtractedField(amounts[-1][0], "found", [_evidence(page, line)]) if amounts else ExtractedField(status="missing"),
                    ))
                    break
    return results


def _currency_field(pages: list[PageRecord]) -> ExtractedField:
    symbols = {"€": "EUR", "£": "GBP"}
    patterns: list[tuple[str, Evidence]] = []
    for page in pages:
        for line in page.text.splitlines():
            for token, code in _CURRENCY_CODES.items():
                if re.search(rf"(?<![A-Z]){re.escape(token)}(?![A-Z])", line, re.I):
                    patterns.append((code, _evidence(page, line)))
            for token, code in symbols.items():
                if token in line:
                    patterns.append((code, _evidence(page, line)))
    explicit_code = bool(patterns)
    if not explicit_code:
        dollar_evidence = [_evidence(page, line) for page in pages for line in page.text.splitlines() if "$" in line]
        if dollar_evidence:
            return ExtractedField(None, "ambiguous", dollar_evidence, ["USD", "SGD", "CAD", "AUD"],
                                  "The dollar symbol does not identify a currency without a code or locale evidence.")
    if not patterns:
        return ExtractedField(status=_readability(pages))
    # A printed payment-currency statement is stronger than an incidental
    # converted currency used for tax reporting elsewhere on the invoice.
    payment_codes = [(match.group(1).upper(), _evidence(page, line))
                     for page in pages for line in page.text.splitlines()
                     if (match := re.search(r"\bpayment\s+in\s+(USD|EUR|GBP|SGD|THB|INR|CAD|AUD|JPY|CHF)\b", line, re.I))]
    for page in pages:
        lines = page.text.splitlines()
        for index, line in enumerate(lines):
            if not re.search(r"please\s+pay|amount\s+due|total\s+due|amount\s+payable", line, re.I):
                continue
            for source_line in lines[index:index + 7]:
                match = re.search(r"\b(EUR|USD|GBP|SGD|THB|INR|CAD|AUD|JPY|CHF|KES)\b\s*\d[\d.,]*", source_line, re.I)
                if match:
                    payment_codes.append((match.group(1).upper(), _evidence(page, source_line)))
    payment_values = list(dict.fromkeys(code for code, _ in payment_codes))
    if len(payment_values) == 1:
        return ExtractedField(payment_values[0], "found", [ev for _code, ev in payment_codes])
    values = list(dict.fromkeys(code for code, _ in patterns))
    evidence = [source for _code, source in patterns]
    if len(values) > 1:
        return ExtractedField(None, "ambiguous", evidence, values)
    return ExtractedField(values[0], "found", evidence)


def _candidate(document: DocumentRecord, pages: list[PageRecord], multi: bool) -> PayableCandidate:
    full_text = "\n".join(page.text for page in pages)
    credit_cue = re.search(
        r"credit\s+(?:note|memo|invoice)|gutschrift|avoir|nota de cr[eé]dito|nota di credito|kreeditarve",
        full_text, re.I,
    )
    explicit_credit_title = re.search(
        r"(?:credit\s+(?:note|memo|invoice)\s*(?:no\.?|number|nr\.?)|gutschrift|avoir|nota de cr[eé]dito|nota di credito|kreeditarve)",
        "\n".join(page.text for page in pages[:2]), re.I,
    )
    negative_total = re.search(r"(?:grand\s+total|gross\s+total|total\s+amount|invoice\s+total|endbetrag|arve\s+kokku)[^\r\n-]*-\s*\d", full_text, re.I)
    is_credit = document.classification.category == "credit_memo" or bool(explicit_credit_title or (credit_cue and negative_total))
    number_pattern = (
        r"(?:credit\s+(?:note|memo)\s*(?:no\.?|number|nr\.?)|kreeditarve\s+(?:arvele\s*)?(?:nr\.?|no\.?)|(?:invoice|tax invoice|rechnung|arve|factura|fatura|faktura|fattura)\s*(?:number|no\.?|nr\.?|#))"
        r"\s*[:#-]?\s*(?P<value>[A-Z0-9][A-Z0-9./-]*)(?![A-Z0-9])(?!\s+(?:mnt|street|strasse|straat|road|rd\.?|avenue|ave\.?|boulevard|blvd\.?|lane|floor|suite|unit)\b)"
    )
    def invoice_number_values(text: str) -> list[tuple[str, str]]:
        # Flat PDF text can put a date beneath an Invoice Number label.
        return [(value, source) for value, source in _label_values(number_pattern, text)
                if not re.fullmatch(_DATE, value)]

    number_field = _field(pages, invoice_number_values, normalize=lambda v: str(v).casefold(), multiple_status="ambiguous")
    if number_field.status == "ambiguous":
        valid = [(value, source) for value, source in zip(number_field.alternatives, number_field.evidence)
                 if not re.search(r"(?<!\d)\d{1,2}[./-]\d{1,2}[./-]\d{4}(?!\d)", source.text)]
        if len(valid) == 1:
            number_field = ExtractedField(valid[0][0], "found", [valid[0][1]])
    if number_field.status == "missing":
        adjacent_numbers: list[tuple[str, Evidence]] = []
        number_label = r"(?:credit\s+(?:note|memo)\s*(?:no\.?|number|nr\.?)|kreeditarve\s+(?:arvele\s*)?(?:nr\.?|no\.?)|(?:invoice|tax\s+invoice|rechnung|arve|factura|fatura|faktura|fattura)\s*(?:number|no\.?|nr\.?|#))\s*[:#-]?\s*$"
        for page in pages:
            lines = page.text.splitlines()
            for index, line in enumerate(lines):
                label = re.search(number_label, line, re.I) or re.search(
                    r"(?:invoice|tax\s+invoice|credit\s+(?:note|memo))\s+(?:number|no\.?|nr\.?|#)", line, re.I)
                if not label:
                    continue
                window = " ".join(lines[index:min(len(lines), index + 3)])
                token_matches = list(re.finditer(r"(?<![A-Z0-9])(?:[A-Z]{1,10}[-/.]?\d[A-Z0-9./-]*|(?<![-/])[0-9]{4,15})(?![A-Z0-9])", window, re.I))
                date_spans = []
                for date_match in re.finditer(_DATE, window, re.I):
                    try:
                        _date_value(date_match.group("date"))
                    except ValueError:
                        continue
                    date_spans.append(date_match.span())
                tokens = [match.group(0) for match in token_matches
                          if not any(start < match.end() and match.start() < stop for start, stop in date_spans)]
                unique_tokens = list(dict.fromkeys(token.casefold() for token in tokens))
                if len(unique_tokens) == 1:
                    value = tokens[0]
                    adjacent_numbers.append((value, _evidence(page, window)))
        values = list(dict.fromkeys(value.casefold() for value, _ in adjacent_numbers))
        if len(values) == 1:
            number_field = ExtractedField(adjacent_numbers[0][0], "found", [ev for _value, ev in adjacent_numbers])
        elif len(values) > 1:
            number_field = ExtractedField(None, "ambiguous", [ev for _value, ev in adjacent_numbers], [value for value, _ in adjacent_numbers])
    spatial_supplier_values: list[tuple[str, Evidence]] = []
    spatial_supplier_ambiguous = False
    for page in pages:
        layout = page.evidence.get("text_layout") if isinstance(page.evidence, dict) else None
        if not isinstance(layout, dict):
            continue
        role = resolve_party_roles(layout).get("supplier", {})
        if role.get("status") == "found" and role.get("value"):
            source = role.get("evidence", {})
            spatial_supplier_values.append((str(role["value"]), _evidence(page, f"{source.get('label', '')} | {source.get('party', '')}")))
        elif role.get("status") == "ambiguous":
            spatial_supplier_ambiguous = True
    spatial_names = {_norm(value) for value, _ in spatial_supplier_values}
    if spatial_supplier_ambiguous or len(spatial_names) > 1:
        supplier_field = ExtractedField(None, "ambiguous", [evidence for _, evidence in spatial_supplier_values],
                                        [value for value, _ in spatial_supplier_values],
                                        "Spatial supplier role labels conflict or identify multiple parties.")
    elif spatial_supplier_values:
        supplier_field = ExtractedField(spatial_supplier_values[0][0], "found", [e for _, e in spatial_supplier_values])
    else:
        supplier_pattern = r"(?:supplier|vendor|seller|from|lieferant|fournisseur|fornecedor|proveedor|remit\s+payment\s+to)\s*(?:name|nome|nombre)?\s*[:#-]\s*(?P<value>[^\r\n]+)"
        supplier_field = _field(pages, lambda text: _label_values(supplier_pattern, text), normalize=lambda v: _norm(str(v)))
    if supplier_field.status == "missing":
        for page in pages:
            lines = page.text.splitlines()
            pairs = [(lines[index + 1].strip(), _evidence(page, f"{line.strip()} {lines[index + 1].strip()}"))
                     for index, line in enumerate(lines[:-1])
                     if re.search(r"(?:supplier|vendor|seller|remit\s+payment\s+to)\s*(?:name)?\s*[:#-]?\s*$", line, re.I)
                     and lines[index + 1].strip()
                     and not re.search(r"^(?:invoice|date|amount|due|account)\b", lines[index + 1], re.I)]
            # Some PDF text layers place the entity immediately before its
            # visual label (for example, Remit payment to:).
            pairs.extend((lines[index - 1].strip(), _evidence(page, f"{lines[index - 1].strip()} {line.strip()}"))
                         for index, line in enumerate(lines[1:], start=1)
                         if re.search(r"remit\s+payment\s+to\s*:?\s*$", line, re.I)
                         and re.match(r"\s*(?:invoice|amount|due|account|billing|customer)\b", lines[index + 1] if index + 1 < len(lines) else "", re.I)
                         and lines[index - 1].strip()
                         and not re.search(r"^(?:invoice|date|amount|due|account|remit)\b", lines[index - 1], re.I))
            values = list(dict.fromkeys(_norm(value) for value, _ in pairs))
            if len(values) == 1:
                supplier_field = ExtractedField(pairs[0][0], "found", [ev for _value, ev in pairs])
                break
            if len(values) > 1:
                supplier_field = ExtractedField(None, "ambiguous", [ev for _value, ev in pairs], [value for value, _ in pairs])
                break
    remittance_supplier, remittance_address = _remittance_supplier(pages)
    header_address = ExtractedField(status="missing")
    if supplier_field.status == "missing":
        header_supplier, header_address = _header_supplier(pages)
        if header_supplier.status == "found":
            supplier_field = header_supplier
    if supplier_field.status == "missing":
        footer_supplier, footer_address = _footer_supplier_from_tax_block(pages)
        if footer_supplier.status == "found":
            supplier_field = footer_supplier
            if remittance_address.status != "found":
                remittance_address = footer_address
    # A remittance beneficiary is weaker than an explicitly labelled party,
    # a sole header legal entity, or a VAT-linked issuer candidate.
    if supplier_field.status == "missing" and remittance_supplier.status == "found":
        supplier_field = remittance_supplier
    address_pattern = r"(?:supplier|vendor|seller|lieferant|fournisseur|fornecedor|proveedor)\s+address\s*[:#-]\s*(?P<value>[^\r\n]+)"
    supplier_address_field = _field(pages, lambda text: _label_values(address_pattern, text))
    if supplier_address_field.status == "missing" and remittance_address.status == "found":
        supplier_address_field = remittance_address
    if supplier_address_field.status == "missing" and header_address.status == "found":
        supplier_address_field = header_address
    if supplier_address_field.status == "missing" and supplier_field.status == "found":
        supplier_address_field = _ocr_header_address(pages, str(supplier_field.value))
    vat_pattern = r"(?:vat\s*(?:id|no\.?|number|registration\s+no\.?)|ust-idnr\.?|ustid|kmkr\s*(?:nr\.?|no\.?)|tax\s+id)\s*[:#-]?\s*(?P<value>[A-Z]{0,3}[A-Z0-9][A-Z0-9 -]{4,})"
    po_pattern = r"(?:purchase\s+order(?:\s*(?:no\.?|number))?|po\s*(?:#|no\.?|number)|bestellnummer|order\s+number|n[uú]mero\s+de\s+pedido|n[uú]mero\s+de\s+ordem)\s*[:#-]?\s*(?P<value>[A-Z0-9][A-Z0-9/-]*)"
    gross_pattern = r"\b(?:grand\s+total|gross\s+total|total\s+amount|amount\s+due|please\s+pay|invoice\s+total|total\s+payable|total\s+a\s+pagar|kogusumma|endbetrag|arve\s+kokku|summa\s+koos\s+käibemaksuga|gesamtsumme|total\s*(?:\(\s*including\s+vat\s*\)|incl(?:uding)?\.?\s+(?:vat|tax|sst)|\(\s*(?:USD|EUR|GBP|SGD|THB|INR|CAD|AUD|JPY|CHF)\s*\))|total\s+(?:da\s+fatura|de\s+la\s+factura|ttc|(?:USD|EUR|GBP|SGD|THB|INR|CAD|AUD|JPY|CHF))|valor\s+total|importe\s+total|total\s+geral)(?:\b|(?=\s|$))"
    subtotal_pattern = r"\b(?:sub[\s-]?total|vahesumma)\b"
    net_subtotal_pattern = r"(?:net\s+amount|summa\s+(?:ilma\s+käibemaksuta|km-ta)|summe\s+netto|amount\s+excluding\s+(?:tax|vat)|net\s+total|total\s+before\s+tax|base\s+imponible|base\s+tribut[aá]vel)"
    tax_total_pattern = r"(?:total\s+(?:tax|vat|gst)|(?:tax|vat|gst)\s+total|summe\s+steuer)"

    def amount_field(pattern: str) -> ExtractedField:
        found: list[tuple[Decimal, Evidence]] = []
        ambiguous: list[tuple[str, Evidence]] = []
        for page in pages:
            lines = page.text.splitlines()
            conversion_started = False
            for index, line in enumerate(lines):
                if re.search(r"conversion\s+to\s+\w+\s+for\s+tax\s+purpose\s+only", line, re.I):
                    conversion_started = True
                if pattern == gross_pattern and conversion_started:
                    continue
                if not re.search(pattern, line, re.I):
                    continue
                amounts = _amounts(line)
                if amounts:
                    found.append((amounts[-1][0], _evidence(page, line)))
                    continue
                if pattern == gross_pattern and re.search(r"please\s+pay", line, re.I):
                    payment_amounts = [
                        (amount, candidate_line) for candidate_line in lines[index + 1:index + 7]
                        if re.search(r"\b(?:EUR|USD|GBP|SGD|THB|INR|CAD|AUD|JPY|CHF|KES)\b\s*\d", candidate_line, re.I)
                        for amount, _match in _amounts(candidate_line)
                    ]
                    if len(payment_amounts) == 1:
                        amount, source_line = payment_amounts[0]
                        found.append((amount, _evidence(page, f"{line.strip()} {source_line.strip()}")))
                        continue
                next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
                adjacent = _amounts(next_line)
                explicit_currency = re.search(r"\b(?:EUR|USD|GBP|SGD|THB|INR|CAD|AUD|JPY|CHF|KES)\b", next_line, re.I)
                if len(adjacent) == 1 and explicit_currency:
                    found.append((adjacent[0][0], _evidence(page, f"{line.strip()} {next_line}")))
                    continue
                if re.fullmatch(r"[€$£]?\s*\d[\d., ]*\s*(?:€|EUR|USD|GBP|THB|SGD|KES)?", next_line, re.I):
                    adjacent = _amounts(next_line)
                    if adjacent:
                        found.append((adjacent[-1][0], _evidence(page, f"{line.strip()} {next_line}")))
                        continue
                for match in _MONEY.finditer(line):
                    parsed = parse_localized_decimal(match.group("number"))
                    if parsed.ambiguous:
                        ambiguous.append((match.group("number"), _evidence(page, line)))
            if not found and pattern in {subtotal_pattern, tax_total_pattern}:
                # Some native table layouts place the three total labels on
                # separate lines, followed by their values in column order.
                for index, line in enumerate(lines[:-5]):
                    if (re.fullmatch(r"\s*SUBTOTAL\s*", line, re.I)
                            and re.fullmatch(r"\s*TAX\s*", lines[index + 1], re.I)
                            and re.search(r"TOTAL\s*\(\s*USD\s*\)", lines[index + 2], re.I)):
                        value_index = index + (3 if pattern == subtotal_pattern else 4)
                        values_here = _amounts(lines[value_index])
                        if values_here:
                            label = "SUBTOTAL" if pattern == subtotal_pattern else "TAX"
                            found.append((values_here[-1][0], _evidence(page, f"{label} {lines[value_index].strip()}")))
                        break
        if ambiguous:
            alternatives = list(dict.fromkeys(value for value, _ in ambiguous))
            evidence = [source for _value, source in ambiguous] + [source for _value, source in found]
            return ExtractedField(None, "ambiguous", evidence, alternatives,
                                  "A printed amount has a separator pattern that can mean either a decimal or a grouping mark.")
        if not found:
            return ExtractedField(status=_readability(pages))
        values = list(dict.fromkeys(value for value, _ in found))
        evidence = [source for _value, source in found]
        if len(values) > 1:
            return ExtractedField(None, "conflicting", evidence, values,
                                  "More than one distinct printed amount matches this total label.")
        return ExtractedField(values[0], "found", evidence)

    warnings: list[str] = []
    if multi:
        warnings.append("Multiple invoice-like sections were indicated by page classification; review candidate boundaries.")
    unreadable_pages = [page.page_number for page in pages if not page.text.strip() or page.error]
    if unreadable_pages:
        warnings.append(f"Some pages have missing or failed text extraction: {unreadable_pages}.")

    taxes = _tax_items(pages)
    tax_total = amount_field(tax_total_pattern)
    if tax_total.status == "missing":
        header_tax_amounts = [tax.amount for tax in taxes if tax.kind == "tax" and tax.amount.status == "found"]
        distinct = list(dict.fromkeys(item.value for item in header_tax_amounts))
        if len(distinct) == 1:
            tax_total = ExtractedField(distinct[0], "found", [ev for item in header_tax_amounts for ev in item.evidence], note="Single printed header tax amount; no tax total label was present.")
        elif len(distinct) > 1:
            tax_total = ExtractedField(None, "ambiguous", [ev for item in header_tax_amounts for ev in item.evidence], distinct, "Multiple printed tax amounts; no total was explicitly identified.")
    printed_subtotal = amount_field(subtotal_pattern)
    if printed_subtotal.status == "missing":
        printed_subtotal = amount_field(net_subtotal_pattern)
    printed_gross = amount_field(gross_pattern)
    extracted_lines = _line_items(pages)
    extracted_charges = _charges(pages)
    _establish_net_basis_from_document_totals(
        extracted_lines, printed_subtotal, tax_total, printed_gross, taxes, pages, extracted_charges,
    )
    taxes = _remove_line_tax_summaries(taxes, extracted_lines, tax_total)

    po_field = _field(pages, lambda text: _label_values(po_pattern, text), normalize=lambda v: str(v).casefold())
    if po_field.status == "missing":
        po_field = _field(pages, lambda text: _label_values(
            r"your\s+p\.?\s*o\.?\s+(?:no\.?|number)\s*[:#-]?\s*(?P<value>[A-Z0-9][A-Z0-9/-]*)", text),
            normalize=lambda v: str(v).casefold())
    if po_field.status == "missing":
        adjacent_pos: list[tuple[str, Evidence]] = []
        po_label = re.compile(r"(?:your\s+)?(?:purchase\s+order|p\.?\s*o\.?|po)\s*(?:no\.?|number|#)?\s*[:#-]?\s*$", re.I)
        for page in pages:
            lines = page.text.splitlines()
            for index, line in enumerate(lines[:-1]):
                value = lines[index + 1].strip()
                if po_label.search(line) and re.fullmatch(r"[A-Z0-9][A-Z0-9/-]{1,30}", value, re.I):
                    adjacent_pos.append((value, _evidence(page, f"{line.strip()} {value}")))
        distinct_pos = list(dict.fromkeys(value.casefold() for value, _ in adjacent_pos))
        if len(distinct_pos) == 1:
            po_field = ExtractedField(adjacent_pos[0][0], "found", [evidence for _, evidence in adjacent_pos])
        elif len(distinct_pos) > 1:
            po_field = ExtractedField(None, "ambiguous", [evidence for _, evidence in adjacent_pos], [value for value, _ in adjacent_pos])
    def unassigned_vat_values(text: str, *, footer_only: bool = False) -> list[tuple[str, str]]:
        """Only use an unassigned VAT label while it is in the invoice identity area."""
        lines = text.splitlines()
        buyer_label = re.compile(r"\b(?:bill\s*to|sold\s*to|ship\s*to|delivery\s+address|customer|buyer)\b", re.I)
        issuer_vat_label = re.compile(r"\b(?:supplier|vendor|seller|issuer)\s+vat\b", re.I)
        values: list[tuple[str, str]] = []
        for index, line in enumerate(lines):
            match = re.search(vat_pattern, line, re.I)
            if not match or any(buyer_label.search(previous) for previous in lines[max(0, index - 5):index + 1]):
                continue
            value = match.group("value").strip(" :#-").rstrip(".,;")
            # A bare VAT label after the printed amount can identify the buyer
            # (for example on a reverse-charge invoice). Its mere presence is
            # not evidence that the number belongs to the issuer.
            after_total = any(re.search(gross_pattern, previous, re.I) and _amounts(previous)
                              for previous in lines[:index])
            unscoped_footer = after_total and not issuer_vat_label.search(line)
            if unscoped_footer == footer_only:
                values.append((value, line))
        return values

    vat_field = _field(pages, unassigned_vat_values, normalize=lambda v: re.sub(r"\s", "", str(v)).casefold())
    if vat_field.status == "missing":
        footer_vat = _field(pages, lambda text: unassigned_vat_values(text, footer_only=True))
        if footer_vat.status == "found":
            vat_field = ExtractedField(None, "ambiguous", footer_vat.evidence, [footer_vat.value],
                                   "A VAT label after the payable total is not linked to the supplier.")
    if vat_field.status == "found" and vat_field.evidence and all(
        re.search(r"federal\s+tax\s+id", item.text, re.I) for item in vat_field.evidence
    ):
        vat_field = ExtractedField(status="missing", note="A Federal Tax ID is printed, but it is not labeled as a VAT ID.")
    if supplier_field.status == "found":
        supplier_vat = _vat_field_near_supplier(pages, str(supplier_field.value))
        if supplier_vat.status == "found":
            vat_field = supplier_vat
        spatial_vats: list[tuple[str, Evidence]] = []
        spatial_vat_ambiguous = False
        for page in pages:
            layout = page.evidence.get("text_layout") if isinstance(page.evidence, dict) else None
            if not isinstance(layout, dict):
                continue
            party_vat = resolve_party_vat(layout, str(supplier_field.value))
            if party_vat.get("status") == "found":
                evidence = party_vat.get("evidence", {})
                spatial_vats.append((str(party_vat["value"]), _evidence(
                    page, f"{evidence.get('party', '')} | {evidence.get('label', '')} | {evidence.get('value', '')}")))
            elif party_vat.get("status") == "ambiguous":
                spatial_vat_ambiguous = True
        spatial_vat_values = {value for value, _ in spatial_vats}
        if spatial_vat_ambiguous or len(spatial_vat_values) > 1:
            vat_field = ExtractedField(None, "ambiguous", [e for _, e in spatial_vats], sorted(spatial_vat_values),
                                       "Spatial VAT labels are competing for the resolved supplier block.")
        elif spatial_vats:
            value = spatial_vats[0][0]
            if vat_field.status in {"missing", "conflicting", "ambiguous"}:
                vat_field = ExtractedField(value, "found", [e for _, e in spatial_vats])
            elif re.sub(r"\s", "", str(vat_field.value)).casefold() != re.sub(r"\s", "", value).casefold():
                vat_field = ExtractedField(None, "conflicting", [*vat_field.evidence, *(e for _, e in spatial_vats)],
                                           [vat_field.value, value], "Flat and spatial supplier VAT evidence disagree.")

    candidate = PayableCandidate(
        document_type=ExtractedField("credit_memo" if is_credit else "invoice", "found", [_evidence(page, page.text[:240]) for page in pages if page.text.strip()]),
        invoice_number=number_field,
        invoice_date=_date_field(pages, r"(?:invoice\s+date|date\s+of\s+invoice|rechnungsdatum|datum\s+der\s+rechnung|kuup[aä]ev|fecha\s+(?:de\s+)?(?:factura|emisi[oó]n)|data\s+da\s+fatura|\bdate)"),
        due_date=_due_date_field(pages),
        supplier_name=supplier_field,
        supplier_address=supplier_address_field,
        supplier_vat_id=vat_field,
        currency=_currency_field(pages),
        printed_gross_total=printed_gross,
        printed_subtotal=printed_subtotal,
        printed_tax_total=tax_total,
        purchase_order_number=po_field,
        lines=extracted_lines,
        taxes=taxes,
        charges=extracted_charges,
        pages=[page.page_number for page in pages],
        review_required=multi or any(page.error or not page.text.strip() for page in pages),
        warnings=warnings,
    )
    critical_fields = (candidate.invoice_number, candidate.invoice_date, candidate.printed_gross_total)
    if any(item.status in {"missing", "ambiguous", "conflicting", "unreadable"} for item in critical_fields):
        candidate.review_required = True
    important_fields = (
        candidate.due_date, candidate.supplier_address, candidate.supplier_vat_id, candidate.currency,
        candidate.printed_subtotal, candidate.printed_tax_total, candidate.purchase_order_number,
    )
    has_field_conflict = any(item.status in {"ambiguous", "conflicting", "unreadable"} for item in important_fields)
    has_tax_conflict = any(
        field.status in {"ambiguous", "conflicting", "unreadable"}
        for tax in [*candidate.taxes, *(tax for line in candidate.lines for tax in line.taxes)]
        for field in (tax.name, tax.rate, tax.amount)
    )
    if has_field_conflict or has_tax_conflict:
        candidate.review_required = True
        candidate.warnings.append("One or more extracted fields have ambiguous, conflicting, or unreadable values.")
    if candidate.supplier_name.status == "missing":
        candidate.review_required = True
        candidate.warnings.append("Supplier name was not explicitly labeled; no name was inferred from the header.")
    if any(field.status in {"ambiguous", "conflicting", "unreadable"} for line in candidate.lines
           for field in (line.description, line.quantity, line.unit_price, line.printed_total)):
        candidate.review_required = True
        candidate.warnings.append("At least one line item contains ambiguous or unreadable values.")
    return candidate


def extract_payables(document: DocumentRecord) -> ExtractionOutcome:
    """Extract candidates only from classified invoices or credit memos.

    This function performs no master-data matching, tax calculation, ERP validation,
    or gross-total balancing. Payment reminders are explicitly excluded.
    """
    category = document.classification.category
    if category == "payment_reminder":
        return ExtractionOutcome(document.filename, excluded_reason="payment_reminder", warnings=[
            "A payment reminder can reference an existing invoice but does not establish a new payable."
        ])
    if category not in {"invoice", "credit_memo"}:
        return ExtractionOutcome(document.filename, excluded_reason=f"not_a_confirmed_payable_document:{category}", review_required=category == "uncertain")
    if not document.pages or not any(page.text.strip() for page in document.pages):
        return ExtractionOutcome(document.filename, excluded_reason="unreadable_document", review_required=True,
                                 warnings=[document.error] if document.error else ["No readable page text is available."])

    starts = sorted(set(page for page in document.classification.possible_document_starts if page > 1))
    groups: list[list[PageRecord]] = []
    boundaries = [1, *starts, max((page.page_number for page in document.pages), default=1) + 1]
    for low, high in zip(boundaries, boundaries[1:]):
        group = [page for page in document.pages if low <= page.page_number < high]
        if group:
            groups.append(group)
    if not groups:
        groups = [document.pages]
    multiple = len(groups) > 1 or document.classification.potential_multiple_documents
    candidates = [_candidate(document, group, multiple) for group in groups]
    if document.classification.uncertain:
        for candidate in candidates:
            candidate.review_required = True
            candidate.warnings.append("The Phase 2 document classification is uncertain.")
    warnings = [warning for candidate in candidates for warning in candidate.warnings]
    return ExtractionOutcome(document.filename, candidates, review_required=multiple or any(candidate.review_required for candidate in candidates), warnings=warnings)
