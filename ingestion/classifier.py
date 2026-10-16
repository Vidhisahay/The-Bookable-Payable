"""Conservative classification from page text cues (never from filenames)."""

from __future__ import annotations

import re
from collections import Counter

from .models import DocumentClassification, PageClassification, PageRecord


_CUES: dict[str, tuple[tuple[str, ...], ...]] = {
    "credit_memo": (
        ("credit memo", "credit note", "credit invoice", "gutschrift", "avoir", "nota de credito", "nota de crédito", "nota di credito", "stornorechnung", "kreeditarve"),
        ("amount credited", "credit amount", "credited", "crédito", "crédit"),
    ),
    "debit_note": (
        ("debit note", "debit memo", "debit advice", "belastungsanzeige"),
        ("amount debited", "debit amount"),
    ),
    "invoice": (
        ("invoice", "tax invoice", "rechnung", "arve", "factura", "fatura", "faktura", "fattura", "invoicing", "bill to", "invoice no", "invoice number"),
        ("amount due", "total due", "total payable", "payment due", "subtotal", "vat", "gst", "iva"),
    ),
    "delivery_note": (
        ("delivery note", "delivery docket", "lieferschein", "proof of delivery", "packing slip", "dispatch note", "bon de livraison", "guia de remessa"),
    ),
    "statement": (
        ("account statement", "statement of account", "customer statement", "account summary", "saldo anterior", "balance brought forward"),
    ),
    "payment_reminder": (
        ("payment reminder", "dunning notice", "overdue notice", "mahnung", "zahlungserinnerung", "final notice"),
        ("zahlungsausgleich", "zahlungstermin", "payment overdue", "past due", "reminder fee"),
    ),
    "purchase_order": (
        ("purchase order", "order confirmation", "bestellung", "commande", "orden de compra", "purchase requisition"),
    ),
    "receipt": (
        ("receipt", "payment received", "paid in full", "proof of payment", "payment confirmation", "quittung"),
    ),
    "remittance": (
        ("remittance advice", "remittance notice", "payment advice", "advice of payment"),
    ),
    "quotation": (
        ("quotation", "quote", "pro forma offer", "estimate"),
    ),
    "contract": (
        ("service agreement", "contract agreement", "terms of service", "master services agreement"),
    ),
    "timesheet": (
        ("timesheet", "time sheet", "hours worked", "time report"),
    ),
    "supporting_document": (
        ("service report", "work report", "goods received note", "inspection report", "technical report", "certificate of completion",
         "shipment cartage advice", "cargo pg", "customs declaration", "customs permit", "in-non-payment"),
    ),
}


def _score(text: str) -> tuple[str, float, list[str]]:
    normalized = re.sub(r"\s+", " ", text.casefold())
    # A customs detail sheet may repeat the parent invoice number, date, and tax
    # labels. It supports the consolidated invoice but does not itself establish
    # a second payable. Treat the explicit document title as stronger evidence
    # than the generic word "invoice" in that same title.
    if re.search(r"\bcustoms\s+detailed\s+invoice\b", normalized):
        return "supporting_document", 0.95, ["customs detailed invoice"]
    if re.search(r"\bkreeditarve\b", normalized):
        return "credit_memo", 0.95, ["kreeditarve"]
    scores: Counter[str] = Counter()
    evidence: dict[str, list[str]] = {}
    for category, cue_groups in _CUES.items():
        found: list[str] = []
        for group_index, group in enumerate(cue_groups):
            for cue in group:
                if re.search(r"(?<!\w)" + re.escape(cue) + r"(?!\w)", normalized):
                    found.append(cue)
                    weight = 4 if category == "payment_reminder" and group_index == 0 else (3 if group_index == 0 else 1)
                    scores[category] += weight
        evidence[category] = found

    if not scores:
        return "uncertain", 0.0, []

    ordered = scores.most_common()
    best, best_score = ordered[0]
    next_score = ordered[1][1] if len(ordered) > 1 else 0
    if best_score < 3 or best_score == next_score:
        return "uncertain", 0.0, sorted({cue for values in evidence.values() for cue in values})

    # Multiple independent cue families give higher confidence than a single keyword.
    cue_count = len(evidence[best])
    confidence = min(0.95, 0.55 + 0.12 * max(0, cue_count - 1) + 0.05 * (best_score >= 4))
    return best, round(confidence, 2), evidence[best]


def _invoice_reference(text: str) -> str | None:
    match = re.search(
        r"\b(?:tax\s+invoice|invoice|credit\s+(?:memo|note)|rechnung|arve|factura|fatura|faktura|fattura)"
        r"\s*(?:number|no\.?|nr\.?|#)\s*[:#-]?\s*([A-Z0-9][A-Z0-9/-]{2,})",
        text, re.I,
    )
    return match.group(1).casefold() if match else None


def classify_pages(pages: list[PageRecord]) -> DocumentClassification:
    page_classifications: list[PageClassification] = []
    for page in pages:
        category, confidence, evidence = _score(page.text)
        page_classifications.append(PageClassification(page.page_number, category, confidence, evidence))

    labeled = [p for p in page_classifications if p.category != "uncertain"]
    counts = Counter(p.category for p in labeled)
    if not counts:
        category, confidence, uncertain, evidence = "uncertain", 0.0, True, []
    else:
        category, page_count = counts.most_common(1)[0]
        # Conflicting strong categories or an unresolved page prevent a confident
        # whole-file label; retain the majority candidate for downstream review.
        conflict = len(counts) > 1
        confidence = round(sum(p.confidence for p in labeled if p.category == category) / page_count, 2)
        uncertain = conflict or any(p.category == "uncertain" for p in page_classifications)
        evidence = sorted({cue for p in labeled if p.category == category for cue in p.evidence})
        if uncertain:
            confidence = min(confidence, 0.55)

    # Page-level starts are candidates only. They do not assert that each page
    # starts a payable; repeated headers can also cause these cues.
    starts: list[int] = []
    multiple_evidence: list[str] = []
    previous_invoice_reference: str | None = None
    previous_was_invoice = False
    for record, classification in zip(pages, page_classifications):
        is_invoice_page = classification.category in {"invoice", "credit_memo"}
        reference = _invoice_reference(record.text) if is_invoice_page else None
        if is_invoice_page and record.page_number > 1 and reference:
            if previous_was_invoice and previous_invoice_reference and reference != previous_invoice_reference:
                starts.append(record.page_number)
                multiple_evidence.append(f"invoice identifier changed to {reference} on page {record.page_number}")
            # Repeated invoice identifiers after supporting pages belong to the
            # same document unless a different identifier supplies independent
            # evidence of a new payable. A page boundary by itself is not one.
        previous_was_invoice = is_invoice_page
        previous_invoice_reference = reference if is_invoice_page else None
    repeated_headers = [
        p.page_number for p in pages
        if len(re.findall(r"\b(?:invoice|credit\s+(?:memo|note))\s*(?:number|no\.?|#)\s*[:#-]?\s*[A-Z0-9][A-Z0-9/-]*", p.text, re.I)) >= 2
    ]
    invoice_like = {p.category for p in page_classifications if p.category in {"invoice", "credit_memo"}}
    mixed = len(counts) > 1 or len(invoice_like) > 1
    multiple_evidence.extend(f"multiple invoice identifiers found on page {page}" for page in repeated_headers)
    return DocumentClassification(
        category=category,
        confidence=confidence,
        uncertain=uncertain,
        evidence=evidence,
        page_classifications=page_classifications,
        mixed_document=mixed,
        possible_document_starts=starts,
        potential_multiple_documents=bool(starts or repeated_headers),
        multiple_document_evidence=multiple_evidence,
    )
