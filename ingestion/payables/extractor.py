"""Deterministic, conservative payable-field extraction from classified pages."""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Callable

from ..models import DocumentRecord, PageRecord
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


_DATE = r"(?P<date>\d{1,4}[./-]\d{1,2}[./-]\d{1,4})"
_NUMBER = r"[+-]?(?:\d{1,3}(?:[ .,\u00a0\u202f]\d{3})+|\d+)(?:[,.]\d{1,6})?"
_MONEY = re.compile(rf"(?<![\w/])(?P<number>{_NUMBER})(?!\w)")
_TABLE_MONEY = re.compile(r"(?<![\w/])(?P<number>[+-]?(?:\d{1,3}(?:[.,]\d{3})+|\d+)(?:[,.]\d{1,6})?)(?!\w)")
_CURRENCY_CODES = {
    "EUR": "EUR", "EURO": "EUR", "USD": "USD", "GBP": "GBP", "SGD": "SGD",
    "THB": "THB", "BAHT": "THB", "INR": "INR", "CAD": "CAD", "AUD": "AUD",
    "JPY": "JPY", "SEK": "SEK", "NOK": "NOK", "DKK": "DKK", "CHF": "CHF",
    "CNY": "CNY", "RMB": "CNY", "BRL": "BRL", "MXN": "MXN", "PLN": "PLN",
    "CZK": "CZK", "HUF": "HUF", "ILS": "ILS", "ZAR": "ZAR",
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


def _date_value(raw: str) -> date:
    value = raw.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"unrecognized date: {value}")


def _date_field(pages: list[PageRecord], pattern: str) -> ExtractedField:
    ambiguous: list[tuple[str, Evidence]] = []
    for page in pages:
        lines = page.text.splitlines()
        for index, line in enumerate(lines):
            if re.search(r"\b(?:bill|billing|statement|delivery)\s+date\b", line, re.I):
                continue
            match = re.search(pattern + r"\s*[:#-]?\s*" + _DATE, line, re.I)
            if not match:
                label = re.search(pattern + r"\s*[:#-]?\s*$", line, re.I)
                next_line = lines[index + 1].strip() if label and index + 1 < len(lines) else ""
                adjacent = re.fullmatch(_DATE, next_line)
                if not adjacent:
                    continue
                raw = adjacent.group("date")
                source = f"{line.strip()} {next_line}"
            else:
                raw = match.group("date")
                source = line
            parts = re.split(r"[./-]", raw)
            locale_dmy = bool(re.search(r"rechnungsdatum|\brechnung\b|\barve\b|makset[aä]htaeg|fatura|factura|mwst|ใบวางบิล", _norm(page.text)))
            if len(parts[0]) <= 2 and len(parts[2]) == 4 and 1 <= int(parts[0]) <= 12 and 1 <= int(parts[1]) <= 12 and parts[0] != parts[1] and not locale_dmy:
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
    header = re.compile(r"description|beschreibung|descri[cç][aã]o|descripci[oó]n|designation|quantity|menge|quantit|quantidade|unit price|einzelpreis|pre[cç]o unit[aá]rio|precio unitario|product reference|pos\.?\s+|toode/teenus|item\b", re.I)
    for page in pages:
        in_table = False
        table_header = ""
        end = re.compile(r"subtotal|total amount|gross total|grand total|endbetrag|vat\s+\d|mwst|total due|amount due|balance|terms of payment|payment due|arve kokku|total current charges", re.I)
        lines = page.text.splitlines()
        for line_index, line in enumerate(lines):
            if header.search(line):
                in_table = True
                table_header = line
                continue
            if end.search(line):
                in_table = False
            if not in_table:
                continue
            row = re.match(r"^\s*(\d{1,3})[.)]?\s+(.+)$", line)
            if not row:
                continue
            values = _amounts(line, table_row=True)
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
            description = line[row.start(2):(first_amount.start() if first_amount else len(line))].strip(" :|-")
            if not description or not values:
                continue
            between = line[first_amount.end():values[-1][1].start()] if len(values) > 1 else ""
            quantity_label = re.search(r"\bqty\b|quantity|menge|quantit|anzahl|quantidade|cant\.?", table_header, re.I)
            price_label = re.search(r"unit\s+price|einzelpreis|price|preis|pre[cç]o|precio", table_header, re.I)
            quantity_before_price = bool(quantity_label and price_label and quantity_label.start() < price_label.start())
            quantity_match = None if quantity_before_price else re.search(r"(?<!\w)(\d+(?:[.,]\d+)?)\s*([\w.]+)?", between)
            quantity = ExtractedField(status="missing")
            unit = ExtractedField(status="missing")
            if quantity_match:
                parsed_quantity = parse_localized_decimal(quantity_match.group(1))
                if parsed_quantity.value is not None:
                    ev = [_evidence(page, line)]
                    quantity = ExtractedField(parsed_quantity.value, "found", ev)
                    if quantity_match.group(2):
                        unit = ExtractedField(quantity_match.group(2).rstrip("."), "found", ev)
            if quantity.status == "missing" and re.search(r"quantity|menge|quantit|anzahl|quantidade|cant\.?", table_header, re.I):
                prefix = line[row.start(2):first_amount.start()]
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
            if first_amount:
                prefix = line[row.start(2):first_amount.start()]
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
            item_type = "service" if re.search(r"hour|hours|stunde|stunden|std\.?|service|management|arbeit|toitlust|servi[cç]o|servicio|horas", description, re.I) else None
            evidence = [_evidence(page, source_line)]
            if re.search(r"gross\s+price|incl(?:uding)?\.?\s+(?:vat|tax)|tax.inclusive|brutto|com\s+iva|con\s+iva", table_header, re.I):
                price_basis = ExtractedField("tax_inclusive", "found", [_evidence(page, table_header)])
            elif re.search(r"net\s+price|excl(?:uding)?\.?\s+(?:vat|tax)|netto|sem\s+iva|sin\s+iva", table_header, re.I):
                price_basis = ExtractedField("net", "found", [_evidence(page, table_header)])
            elif (add_tax := re.search(r"\b(?:zzgl\.?|plus)\s*\d{1,2}(?:[,.]\d+)?\s*%\s*(?:mwst|ust|vat|gst|iva|tva)\b", page.text, re.I)):
                price_basis = ExtractedField("net", "found", [_evidence(page, add_tax.group(0))])
            else:
                price_basis = ExtractedField(status="missing", note="The table does not state whether prices include tax.")
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


def _tax_items(pages: list[PageRecord]) -> list[TaxItem]:
    result: list[TaxItem] = []
    tax_cue = re.compile(r"vat|mwst|ust\b|gst|iva|tva|moms|k[aä]ibemaks|imposto|impuesto|km\b|withholding|reverse charge|reten|kinnipid", re.I)
    rate_re = re.compile(r"(?P<rate>\d{1,2}(?:[,.]\d+)?)\s*%")
    for page in pages:
        in_table = False
        for line in page.text.splitlines():
            if re.search(r"description|beschreibung|designation|quantity|menge|unit price|einzelpreis|product reference|toode/teenus|item\b", line, re.I):
                in_table = True
                continue
            if re.search(r"subtotal|total amount|gross total|grand total|endbetrag|vat\s+\d|mwst|km\s+\d|total due|payment due|arve kokku", line, re.I):
                in_table = False
            if in_table:
                continue
            if re.search(r"\bkm\s*-\s*ta\b", line, re.I):
                continue
            if not tax_cue.search(line):
                continue
            if re.search(r"\bvat\s*(?:id|no\.?|number|registration)|\bustid\b|ust-idnr", line, re.I):
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
                ExtractedField(amounts[-1][0], "found", [_evidence(page, line)]) if amounts else ExtractedField(status="missing"),
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


def _charges(pages: list[PageRecord]) -> list[ChargeItem]:
    cues = {
        "discount": r"discount|rabatt|skonto|descuento|desconto",
        "freight": r"freight|shipping|delivery charge|transport|porto|fracht|frete",
        "insurance": r"insurance|assurance|versicherung|seguro",
        "excise_duty": r"excise duty|excise tax|accise|impuesto especial",
        "extra_charge": r"surcharge|service charge|management fee|extra charge|handling fee",
    }
    results: list[ChargeItem] = []
    for page in pages:
        for line in page.text.splitlines():
            for kind, cue in cues.items():
                if re.search(cue, line, re.I):
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
        r"\s*[:#-]?\s*(?P<value>[A-Z0-9][A-Z0-9./-]*)"
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
            for index, line in enumerate(lines[:-1]):
                value = lines[index + 1].strip()
                if (re.search(number_label, line, re.I)
                        and re.fullmatch(r"[A-Z0-9][A-Z0-9./-]{0,30}", value, re.I)
                        and not re.fullmatch(_DATE, value)):
                    adjacent_numbers.append((value, _evidence(page, f"{line.strip()} {value}")))
        values = list(dict.fromkeys(value.casefold() for value, _ in adjacent_numbers))
        if len(values) == 1:
            number_field = ExtractedField(adjacent_numbers[0][0], "found", [ev for _value, ev in adjacent_numbers])
        elif len(values) > 1:
            number_field = ExtractedField(None, "ambiguous", [ev for _value, ev in adjacent_numbers], [value for value, _ in adjacent_numbers])
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
    address_pattern = r"(?:supplier|vendor|seller|lieferant|fournisseur|fornecedor|proveedor)\s+address\s*[:#-]\s*(?P<value>[^\r\n]+)"
    vat_pattern = r"(?:vat\s*(?:id|no\.?|number|registration\s+no\.?)|ust-idnr\.?|ustid|kmkr\s*(?:nr\.?|no\.?)|tax\s+id)\s*[:#-]?\s*(?P<value>[A-Z]{0,3}[A-Z0-9][A-Z0-9 -]{4,})"
    po_pattern = r"(?:purchase\s+order(?:\s*(?:no\.?|number))?|po\s*(?:#|no\.?|number)|bestellnummer|order\s+number|n[uú]mero\s+de\s+pedido|n[uú]mero\s+de\s+ordem)\s*[:#-]?\s*(?P<value>[A-Z0-9][A-Z0-9/-]*)"
    gross_pattern = r"(?:grand\s+total|gross\s+total|total\s+amount|amount\s+due|invoice\s+total|total\s+payable|endbetrag|arve\s+kokku|gesamtsumme|total\s*\(\s*including\s+vat\s*\)|total\s+(?:da\s+fatura|de\s+la\s+factura|ttc)|valor\s+total|importe\s+total|total\s+geral)"
    subtotal_pattern = r"(?:subtotal|net\s+amount|summa\s+km-ta|summe\s+netto|amount\s+excluding\s+(?:tax|vat)|net\s+total|total\s+before\s+tax|base\s+imponible|base\s+tribut[aá]vel)"
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
                next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
                if re.fullmatch(r"[€$£]?\s*\d[\d., ]*\s*(?:€|EUR|USD|GBP|THB|SGD)?", next_line, re.I):
                    adjacent = _amounts(next_line)
                    if adjacent:
                        found.append((adjacent[-1][0], _evidence(page, f"{line.strip()} {next_line}")))
                        continue
                for match in _MONEY.finditer(line):
                    parsed = parse_localized_decimal(match.group("number"))
                    if parsed.ambiguous:
                        ambiguous.append((match.group("number"), _evidence(page, line)))
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

    candidate = PayableCandidate(
        document_type=ExtractedField("credit_memo" if is_credit else "invoice", "found", [_evidence(page, page.text[:240]) for page in pages if page.text.strip()]),
        invoice_number=number_field,
        invoice_date=_date_field(pages, r"(?:invoice\s+date|date\s+of\s+invoice|rechnungsdatum|datum\s+der\s+rechnung|kuup[aä]ev|fecha\s+(?:de\s+)?(?:factura|emisi[oó]n)|data\s+da\s+fatura|\bdate)"),
        due_date=_due_date_field(pages),
        supplier_name=supplier_field,
        supplier_address=_field(pages, lambda text: _label_values(address_pattern, text)),
        supplier_vat_id=_field(pages, lambda text: _label_values(vat_pattern, text), normalize=lambda v: re.sub(r"\s", "", str(v)).casefold()),
        currency=_currency_field(pages),
        printed_gross_total=amount_field(gross_pattern),
        printed_subtotal=amount_field(subtotal_pattern),
        printed_tax_total=tax_total,
        purchase_order_number=_field(pages, lambda text: _label_values(po_pattern, text), normalize=lambda v: str(v).casefold()),
        lines=_line_items(pages),
        taxes=taxes,
        charges=_charges(pages),
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
