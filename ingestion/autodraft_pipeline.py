"""End-to-end document-to-autodraft orchestration with separate diagnostics.

The returned ``accounting`` object conforms to the assessment's per-file output
wrapper. OCR/page evidence and match/reconciliation explanations live separately
under ``diagnostics`` and are never mixed into payable schema objects.
"""
from __future__ import annotations

import re
from dataclasses import replace
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from .autodraft import MasterData, generate_autodraft
from .classifier import classify_pages
from .models import DocumentRecord
from .payables import extract_payables
from .payables.models import ExtractedField, PayableCandidate, field_to_dict
from .pipeline import IngestionConfig, IngestionPipeline


_NONPAYABLE_PAGE_CATEGORIES = {
    "payment_reminder", "delivery_note", "statement", "purchase_order", "receipt", "remittance",
    "quotation", "contract", "timesheet", "supporting_document",
}
_PAYABLE_PAGE_CATEGORIES = {"invoice", "credit_memo"}
_CONFIRMED_NONPAYABLE_CATEGORIES = {
    "payment_reminder", "delivery_note", "statement", "purchase_order", "receipt",
    "remittance", "quotation", "contract", "timesheet", "supporting_document", "debit_note",
}


def _json_value(value: Any) -> Any:
    if isinstance(value, (date, datetime, Decimal)):
        return value.isoformat() if isinstance(value, (date, datetime)) else str(value)
    return value


def _field(value: ExtractedField) -> dict[str, Any]:
    result = field_to_dict(value)
    result["value"] = _json_value(result["value"])
    result["alternatives"] = [_json_value(item) for item in result["alternatives"]]
    return result


def _candidate_diagnostics(candidate: PayableCandidate) -> dict[str, Any]:
    fields = ("document_type", "invoice_number", "invoice_date", "due_date", "supplier_name",
              "supplier_address", "supplier_vat_id", "currency", "printed_gross_total",
              "printed_subtotal", "printed_tax_total", "purchase_order_number")
    rendered = {name: _field(getattr(candidate, name)) for name in fields}
    rendered["lines"] = [{
        "description": _field(line.description), "item_type": _field(line.item_type), "quantity": _field(line.quantity),
        "unit": _field(line.unit), "unit_price": _field(line.unit_price), "price_basis": _field(line.price_basis),
        "printed_total": _field(line.printed_total), "taxes": [_tax_diagnostic(tax) for tax in line.taxes],
    } for line in candidate.lines]
    rendered["taxes"] = [_tax_diagnostic(tax) for tax in candidate.taxes]
    rendered["charges"] = [{"kind": charge.kind, "description": _field(charge.description), "amount": _field(charge.amount)} for charge in candidate.charges]
    rendered["pages"] = candidate.pages
    rendered["review_required"] = candidate.review_required
    rendered["warnings"] = candidate.warnings
    return rendered


def _tax_diagnostic(tax: Any) -> dict[str, Any]:
    return {"scope": tax.scope, "kind": tax.kind, "name": _field(tax.name), "rate": _field(tax.rate), "amount": _field(tax.amount)}


def _explicit_buyer_hints(document: DocumentRecord, masters: MasterData) -> tuple[dict[str, str], dict[str, Any]]:
    """Find exact master names/codes only inside explicit buyer/bill-to context."""
    context = re.compile(r"\b(?:bill\s*to|buyer|customer|invoice\s*to|sold\s*to)\b", re.I)
    hints: dict[str, str] = {}
    evidence: dict[str, Any] = {}
    text = "\n".join(page.text for page in document.pages)
    lines = text.splitlines()
    for index, line in enumerate(lines):
        marker = context.search(line)
        if not marker:
            continue
        # Bill-to blocks usually have a label followed by several address/name lines.
        block = " ".join(lines[index:index + 6]).casefold()
        for company in masters.companies:
            company_values = (company.get("company_code", ""), company.get("company_name", ""))
            unit_matches = [unit for unit in company.get("business_units", [])
                            if any(value and value.casefold() in block for value in (unit.get("business_unit_code", ""), unit.get("business_unit_name", "")))]
            company_found = any(value and value.casefold() in block for value in company_values)
            if company_found or len(unit_matches) == 1:
                hints["company_code"] = company["company_code"]
                evidence["company_code"] = {"value": company["company_code"], "source": line}
                if len(unit_matches) == 1:
                    unit = unit_matches[0]
                    hints["business_unit_code"] = unit["business_unit_code"]
                    evidence["business_unit_code"] = {"value": unit["business_unit_code"], "source": line}
                    locations = [loc for loc in unit.get("locations", [])
                                 if any(value and value.casefold() in block for value in (loc.get("location_code", ""), loc.get("location_name", ""), loc.get("invoice_to_address", "")))]
                    if len(locations) == 1:
                        hints["location_code"] = locations[0]["location_code"]
                        evidence["location_code"] = {"value": locations[0]["location_code"], "source": line}
                break
    return hints, evidence


def _explicit_payment_term(document: DocumentRecord, masters: MasterData) -> tuple[str | None, dict[str, Any]]:
    """Match a printed master alias; never derive term days from invoice/due dates."""
    text = "\n".join(page.text for page in document.pages)
    hits: list[tuple[str, str, int]] = []
    for term in masters.payment_terms:
        for alias in term.get("text_aliases", []):
            match = re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text, re.I)
            if match:
                hits.append((term["payment_term_id"], alias, match.start()))
    identities = {term_id for term_id, _alias, _pos in hits}
    if len(identities) != 1:
        return None, {"status": "ambiguous" if identities else "missing", "matches": sorted(identities),
                      "reason": "Multiple explicit master terms appear." if identities else "No exact payment-term alias was found; dates are not used to infer terms."}
    term_id = next(iter(identities))
    _matched_id, alias, pos = next(hit for hit in hits if hit[0] == term_id)
    excerpt = text[max(0, pos - 45):pos + len(alias) + 45].replace("\n", " ").strip()
    return alias, {"status": "found", "payment_term_id": term_id, "alias": alias, "evidence": excerpt}


def _scope_payable_pages(document: DocumentRecord) -> tuple[list[Any], list[dict[str, Any]], list[int]]:
    """Keep invoice pages and uncertain continuations, stopping at explicit other documents."""
    categories = {item.page_number: item.category for item in document.classification.page_classifications}
    if not any(category in _NONPAYABLE_PAGE_CATEGORIES for category in categories.values()):
        return document.pages, [{"page_number": page.page_number, "included": True, "reason": "No explicit non-payable page boundary was classified."} for page in document.pages], []
    scoped: list[Any] = []
    trace: list[dict[str, Any]] = []
    section_starts: list[int] = []
    in_payable_section = False
    for page in document.pages:
        category = categories.get(page.page_number, "uncertain")
        if category in _PAYABLE_PAGE_CATEGORIES:
            if scoped and not in_payable_section:
                section_starts.append(page.page_number)
            in_payable_section = True
            scoped.append(page)
            trace.append({"page_number": page.page_number, "included": True, "reason": f"Explicit {category} page."})
        elif category in _NONPAYABLE_PAGE_CATEGORIES:
            in_payable_section = False
            trace.append({"page_number": page.page_number, "included": False, "reason": f"Explicit {category} page ends the payable section."})
        elif in_payable_section:
            scoped.append(page)
            trace.append({"page_number": page.page_number, "included": True, "reason": "Uncertain page retained as a possible continuation before the next explicit document boundary."})
        else:
            trace.append({"page_number": page.page_number, "included": False, "reason": "Uncertain page outside an identified payable section."})
    return scoped, trace, section_starts


class AutodraftPipeline:
    def __init__(self, *, ingestion: IngestionPipeline | None = None, masters: MasterData | None = None,
                 master_directory: str | Path = "master_data"):
        self.ingestion = ingestion or IngestionPipeline()
        self.masters = masters or MasterData.load(master_directory)

    def process_file(self, path: Path, page_numbers: list[int] | None = None) -> dict[str, Any]:
        document = self.ingestion.process_file(path, page_numbers=page_numbers)
        return self.process_document(document)

    def process_document(self, document: DocumentRecord) -> dict[str, Any]:
        """Run classification scoping, field extraction, matching, and ERP checks
        for previously ingested page records (useful for validated OCR caches)."""
        payable_pages, scope_trace, section_starts = _scope_payable_pages(document)
        scoped_classification = classify_pages(payable_pages) if payable_pages else document.classification
        extraction_classification = replace(
            scoped_classification,
            possible_document_starts=sorted(set(document.classification.possible_document_starts + section_starts)),
            potential_multiple_documents=scoped_classification.potential_multiple_documents or bool(section_starts),
        )
        scoped_document = DocumentRecord(document.filename, payable_pages, extraction_classification, document.error)
        outcome = extract_payables(scoped_document)
        accounting_payables: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        unresolved_reasons: list[str] = []
        for candidate in outcome.candidates:
            scoped_pages = [page for page in document.pages if page.page_number in candidate.pages]
            scoped_document = DocumentRecord(document.filename, scoped_pages, document.classification, document.error)
            hints, buyer_evidence = _explicit_buyer_hints(scoped_document, self.masters)
            term_text, term_evidence = _explicit_payment_term(scoped_document, self.masters)
            payable, diagnostic = generate_autodraft(candidate, self.masters, buyer_hints=hints, payment_term_text=term_text)
            reconciliation = diagnostic.get("reconciliation", {})
            emitted = reconciliation.get("status") == "passed"
            if emitted:
                accounting_payables.append(payable)
            else:
                unresolved_reasons.append(reconciliation.get("reason", "The payable evidence is incomplete or ambiguous."))
            candidates.append({"fields": _candidate_diagnostics(candidate), "master_hints": {"buyer": buyer_evidence, "payment_term": term_evidence},
                               "output_status": "emitted_bookable_payable" if emitted else "unresolved_not_emitted",
                               **diagnostic})

        declined: list[dict[str, str]] = []
        has_payable_page_evidence = any(
            page.category in _PAYABLE_PAGE_CATEGORIES
            for page in document.classification.page_classifications
        )
        page_categories = [page.category for page in document.classification.page_classifications]
        complete_nonpayable_evidence = bool(page_categories) and all(
            category in _CONFIRMED_NONPAYABLE_CATEGORIES for category in page_categories
        ) and all(page.text.strip() and not page.error for page in document.pages)
        confirmed_nonpayable = (
            document.classification.category in _CONFIRMED_NONPAYABLE_CATEGORIES
            and complete_nonpayable_evidence
            and not has_payable_page_evidence
        )
        if outcome.excluded_reason and confirmed_nonpayable:
            declined.append({"doc_type": document.classification.category, "reason": "; ".join(outcome.warnings) or outcome.excluded_reason})

        if outcome.excluded_reason and not confirmed_nonpayable:
            unresolved_reasons.extend(outcome.warnings or [outcome.excluded_reason])
        if not outcome.candidates and not outcome.excluded_reason:
            unresolved_reasons.append("No payable candidate could be established from the available page evidence.")

        accounting = {"file": document.filename, "payables": accounting_payables, "declined": declined}
        diagnostics = {
            "document_error": document.error,
            "classification": document.classification.to_dict(),
            "pages": [page.to_dict() for page in document.pages],
            "extraction": {"excluded_reason": outcome.excluded_reason, "review_required": outcome.review_required,
                           "page_scope": scope_trace,
                           "warnings": outcome.warnings, "candidates": candidates,
                           "unresolved_reasons": list(dict.fromkeys(unresolved_reasons))},
        }
        return {"accounting": accounting, "diagnostics": diagnostics}
