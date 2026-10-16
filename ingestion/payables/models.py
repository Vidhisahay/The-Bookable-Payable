"""Typed extraction results with provenance and explicit uncertainty states."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal


FieldStatus = Literal["found", "missing", "ambiguous", "conflicting", "unreadable"]


@dataclass(frozen=True)
class Evidence:
    filename: str
    page_number: int
    text: str


@dataclass
class ExtractedField:
    value: Any = None
    status: FieldStatus = "missing"
    evidence: list[Evidence] = field(default_factory=list)
    alternatives: list[Any] = field(default_factory=list)
    note: str | None = None


@dataclass
class LineItem:
    description: ExtractedField
    item_type: ExtractedField
    quantity: ExtractedField
    unit: ExtractedField
    unit_price: ExtractedField
    price_basis: ExtractedField
    printed_total: ExtractedField
    taxes: list["TaxItem"] = field(default_factory=list)


@dataclass
class TaxItem:
    scope: Literal["line", "header"]
    kind: Literal["tax", "withholding", "unknown"]
    name: ExtractedField
    rate: ExtractedField
    amount: ExtractedField


@dataclass
class ChargeItem:
    kind: str
    description: ExtractedField
    amount: ExtractedField


@dataclass
class PayableCandidate:
    document_type: ExtractedField
    invoice_number: ExtractedField
    invoice_date: ExtractedField
    due_date: ExtractedField
    supplier_name: ExtractedField
    supplier_address: ExtractedField
    supplier_vat_id: ExtractedField
    currency: ExtractedField
    printed_gross_total: ExtractedField
    printed_subtotal: ExtractedField
    printed_tax_total: ExtractedField
    purchase_order_number: ExtractedField
    lines: list[LineItem]
    taxes: list[TaxItem]
    charges: list[ChargeItem]
    pages: list[int]
    review_required: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class ExtractionOutcome:
    filename: str
    candidates: list[PayableCandidate] = field(default_factory=list)
    excluded_reason: str | None = None
    review_required: bool = False
    warnings: list[str] = field(default_factory=list)


def field_to_dict(value: ExtractedField) -> dict[str, Any]:
    """Return a JSON-compatible view for diagnostics; Decimal values stay exact."""
    rendered = str(value.value) if isinstance(value.value, Decimal) else value.value
    alternatives = [str(item) if isinstance(item, Decimal) else item for item in value.alternatives]
    return {
        "value": rendered,
        "status": value.status,
        "evidence": [
            {"filename": item.filename, "page_number": item.page_number, "text": item.text}
            for item in value.evidence
        ],
        "alternatives": alternatives,
        "note": value.note,
    }
