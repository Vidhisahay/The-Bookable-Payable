"""Conservative master-data resolution, schema mapping, and ERP reconciliation.

This module intentionally keeps extraction candidates/provenance separate from the
assessment's exact payable JSON object. It does not write files.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from erp import erp_book
from .payables.models import ExtractedField, PayableCandidate, TaxItem


def _norm(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())


def _idnorm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", _norm(value))


def _found(field: ExtractedField) -> bool:
    return field.status == "found" and field.value is not None


def _text(field: ExtractedField) -> str:
    return str(field.value) if _found(field) else ""


def _document_number(candidate: PayableCandidate, credit: bool) -> str:
    if not _found(candidate.invoice_number):
        return ""
    # A credit document may cite the original invoice in a labelled reference field;
    # that is not evidence for the credit memo's own number.
    if credit:
        evidence = " ".join(item.text.casefold() for item in candidate.invoice_number.evidence)
        if re.search(r"credit\s+(?:note|memo).{0,35}\bfor\s+(?:invoice|bill)|kreeditarve\s+arvele|credit\s+against\s+(?:invoice|bill)", evidence):
            return ""
    return str(candidate.invoice_number.value)


def _amount(field: ExtractedField, credit: bool = False) -> str:
    if not _found(field):
        return ""
    value = Decimal(str(field.value))
    if credit:
        value = abs(value)
    return format(value, "f")


@dataclass(frozen=True)
class Match:
    code: str = ""
    status: str = "unmatched"  # matched | unmatched | ambiguous | insufficient_evidence
    reason: str = ""
    candidates: tuple[str, ...] = ()


@dataclass
class MasterData:
    suppliers: list[dict[str, Any]] = field(default_factory=list)
    companies: list[dict[str, Any]] = field(default_factory=list)
    taxes: list[dict[str, Any]] = field(default_factory=list)
    payment_terms: list[dict[str, Any]] = field(default_factory=list)
    purchase_orders: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, directory: str | Path = "master_data") -> "MasterData":
        root = Path(directory)
        def read(filename: str, key: str) -> list[dict[str, Any]]:
            with (root / filename).open(encoding="utf-8") as stream:
                return json.load(stream).get(key, [])
        return cls(read("suppliers.json", "suppliers"), read("chart_of_books.json", "companies"),
                   read("tax_master.json", "taxes"), read("payment_terms.json", "payment_terms"),
                   read("po_master.json", "purchase_orders"))


def match_supplier(candidate: PayableCandidate, masters: MasterData) -> Match:
    """Match VAT exactly; name-only matches also require corroborating exact address."""
    vat = _idnorm(candidate.supplier_vat_id.value) if _found(candidate.supplier_vat_id) else ""
    name = _norm(candidate.supplier_name.value) if _found(candidate.supplier_name) else ""
    address = _norm(candidate.supplier_address.value) if _found(candidate.supplier_address) else ""
    if vat:
        matches = [s for s in masters.suppliers if _idnorm(s.get("vat_id")) == vat and s.get("vat_id")]
        if len(matches) == 1:
            return Match(str(matches[0]["supplier_id"]), "matched", "Unique exact normalized VAT ID.")
        if len(matches) > 1:
            return Match(status="ambiguous", reason="VAT ID maps to multiple suppliers.", candidates=tuple(str(x["supplier_id"]) for x in matches))
    if name and address:
        matches = [s for s in masters.suppliers if _norm(s.get("name")) == name and _norm(s.get("address")) == address]
        if len(matches) == 1:
            return Match(str(matches[0]["supplier_id"]), "matched", "Unique exact normalized name and address.")
        if len(matches) > 1:
            return Match(status="ambiguous", reason="Name and address map to multiple suppliers.", candidates=tuple(str(x["supplier_id"]) for x in matches))
    if name and any(_norm(s.get("name")) == name for s in masters.suppliers):
        return Match(status="insufficient_evidence", reason="Supplier name matches, but identity lacks VAT or corroborating address.")
    return Match(reason="No defensible supplier match in master data.")


def match_buyer(hints: dict[str, str] | None, masters: MasterData) -> dict[str, str]:
    """Resolve only explicit exact normalized company, unit, and location hints."""
    result = {"company_code": "", "business_unit_code": "", "location_code": ""}
    hints = hints or {}
    companies = [c for c in masters.companies if _norm(c.get("company_code")) == _norm(hints.get("company_code"))] if hints.get("company_code") else []
    if len(companies) != 1:
        return result
    company = companies[0]
    result["company_code"] = company["company_code"]
    units = [u for u in company.get("business_units", []) if _norm(u.get("business_unit_code")) == _norm(hints.get("business_unit_code"))] if hints.get("business_unit_code") else []
    if len(units) == 1:
        result["business_unit_code"] = units[0]["business_unit_code"]
        locations = [l for l in units[0].get("locations", []) if _norm(l.get("location_code")) == _norm(hints.get("location_code"))] if hints.get("location_code") else []
        if len(locations) == 1:
            result["location_code"] = locations[0]["location_code"]
    return result


def match_payment_term(raw: str | None, masters: MasterData) -> Match:
    if not raw:
        return Match(status="insufficient_evidence", reason="Candidate model has no explicit payment-term text.")
    value = _norm(raw)
    hits = [term for term in masters.payment_terms if value in {_norm(a) for a in term.get("text_aliases", [])} or _norm(term.get("payment_term_id")) == value]
    if len(hits) == 1:
        return Match(hits[0]["payment_term_id"], "matched", "Exact normalized payment-term alias.")
    return Match(status="ambiguous" if hits else "unmatched", reason="Payment-term text did not uniquely match a master alias.", candidates=tuple(x["payment_term_id"] for x in hits))


def match_po(candidate: PayableCandidate, supplier: Match, masters: MasterData) -> Match:
    if not _found(candidate.purchase_order_number):
        return Match(status="insufficient_evidence", reason="No explicit PO number was extracted.")
    number = _idnorm(candidate.purchase_order_number.value)
    hits = [po for po in masters.purchase_orders if _idnorm(po.get("po_number")) == number]
    if supplier.status != "matched":
        return Match(status="insufficient_evidence", reason="PO cannot be linked to supplier until supplier identity is resolved.")
    hits = [po for po in hits if str(po.get("supplier_id")) == supplier.code]
    if len(hits) == 1:
        return Match(hits[0]["po_id"], "matched", "Exact normalized PO number and supplier relationship match.")
    return Match(status="ambiguous" if hits else "unmatched", reason="No unique PO for this number and supplier.", candidates=tuple(x["po_id"] for x in hits))


def match_tax(tax: TaxItem, candidate: PayableCandidate, supplier: Match, masters: MasterData) -> Match:
    rate = Decimal(str(tax.rate.value)) if _found(tax.rate) else None
    rawname = _norm(tax.name.value) if _found(tax.name) else ""
    country = ""
    if supplier.status == "matched":
        row = next((s for s in masters.suppliers if str(s.get("supplier_id")) == supplier.code), None)
        country = str((row or {}).get("country", "")).upper()
    type_aliases = {"vat": "VAT", "käibemaks": "VAT", "iva": "IVA", "gst": "GST", "sst": "SST", "nhil": "NHIL", "getfund": "GETFL", "getfl": "GETFL", "covid": "COVID", "cst": "CST", "wht": "WHT", "withholding tax": "WHT", "moms": "MOMS", "hst": "HST", "use tax": "USE", "np": "NP"}
    inferred_type = next((code for alias, code in type_aliases.items() if alias in rawname), "")
    hits = [row for row in masters.taxes if rate is not None and Decimal(str(row.get("rate"))) == rate
            and (not country or row.get("country", "").upper() == country)
            and (not inferred_type or row.get("tax_type") == inferred_type)]
    # If name contains an explicit master code or exact name, use that as additional evidence.
    exact_name = [row for row in hits if _norm(row.get("name")) == rawname or _norm(row.get("tax_type")) == rawname]
    if len(exact_name) == 1:
        hits = exact_name
    if len(hits) == 1:
        return Match(hits[0]["code"], "matched", "Unique master tax match by rate, supplier country, and tax type evidence.")
    return Match(status="ambiguous" if len(hits) > 1 else "unmatched", reason="Tax evidence does not uniquely identify a master code.", candidates=tuple(str(x["code"]) for x in hits))


def _tax_json(tax: TaxItem, candidate: PayableCandidate, supplier: Match, masters: MasterData, credit: bool) -> dict[str, str]:
    match = match_tax(tax, candidate, supplier, masters)
    rate = _amount(tax.rate, credit)
    amount = _amount(tax.amount, credit)
    if tax.kind == "withholding" and _found(tax.amount) and not credit:
        amount = format(-abs(Decimal(str(tax.amount.value))), "f")
    return {"tax_type": tax.kind.upper() if tax.kind == "withholding" else ("VAT" if not _found(tax.name) else _tax_type(tax.name.value)),
            "tax_name": _text(tax.name), "tax_rate": rate, "tax_amount": amount, "tax_type_code": match.code}


def _tax_type(name: Any) -> str:
    value = _norm(name)
    for alias, code in (("withholding", "WHT"), ("gst", "GST"), ("sst", "SST"), ("iva", "IVA"), ("nhil", "NHIL"), ("getfl", "GETFL"), ("covid", "COVID"), ("cst", "CST"), ("moms", "MOMS"), ("hst", "HST"), ("käibemaks", "VAT"), ("mwst", "VAT"), ("ust", "VAT"), ("vat", "VAT")):
        if alias in value:
            return code
    return ""


def _line_json(line: Any, candidate: PayableCandidate, supplier: Match, masters: MasterData, credit: bool) -> dict[str, Any]:
    price_known_net = _found(line.price_basis) and line.price_basis.value == "net"
    lines_taxes = [_tax_json(tax, candidate, supplier, masters, credit) for tax in line.taxes]
    return {
        "description": _text(line.description), "item_type": _text(line.item_type).upper(), "uom": _text(line.unit),
        "quantity": _amount(line.quantity, credit), "unit_price": _amount(line.unit_price, credit) if price_known_net else "",
        "total": _amount(line.printed_total, credit), "discount": "", "discount_percentage": "",
        "tax_rate": "", "tax_amount": "", "taxes": lines_taxes,
    }


def generate_autodraft(candidate: PayableCandidate, masters: MasterData, *, buyer_hints: dict[str, str] | None = None, payment_term_text: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return exact-schema payable plus sidecar match/check diagnostics (no disk writes)."""
    credit = _found(candidate.document_type) and str(candidate.document_type.value).casefold() == "credit_memo"
    supplier = match_supplier(candidate, masters)
    po = match_po(candidate, supplier, masters)
    term = match_payment_term(payment_term_text, masters)
    buyer = match_buyer(buyer_hints, masters)
    header_taxes = [tax for tax in candidate.taxes if tax.scope == "header"]
    line_tax_records = [tax for line in candidate.lines for tax in line.taxes]
    taxes = [_tax_json(tax, candidate, supplier, masters, credit) for tax in header_taxes]
    totals: dict[str, list[Decimal]] = {"discount": [], "freight": [], "insurance": [], "extra_charge": [], "excise_duty": []}
    for charge in candidate.charges:
        if charge.kind in totals and _found(charge.amount):
            amount = abs(Decimal(str(charge.amount.value))) if credit or charge.kind == "discount" else Decimal(str(charge.amount.value))
            totals[charge.kind].append(amount)
    def summed(key: str) -> str:
        return format(sum(totals[key], Decimal("0")), "f") if totals[key] else ""
    payable: dict[str, Any] = {
        "invoice_number": _document_number(candidate, credit),
        "invoice_date": candidate.invoice_date.value.isoformat() if _found(candidate.invoice_date) and isinstance(candidate.invoice_date.value, date) else "",
        "due_date": candidate.due_date.value.isoformat() if _found(candidate.due_date) and isinstance(candidate.due_date.value, date) else "",
        "invoice_type": "CREDIT_MEMO" if credit else "INVOICE", "currency": _text(candidate.currency),
        "supplier": {"name": _text(candidate.supplier_name), "supplier_id": supplier.code, "address": _text(candidate.supplier_address), "vat_id": _text(candidate.supplier_vat_id)},
        "buyer": buyer, "payment_term_id": term.code, "po_number": _text(candidate.purchase_order_number), "po_id": po.code,
        "gross_total": _amount(candidate.printed_gross_total, credit), "subtotal": _amount(candidate.printed_subtotal, credit),
        "total_tax_amount": _amount(candidate.printed_tax_total, credit), "discount_amount": summed("discount"),
        "freight_charges": summed("freight"), "insurance_charges": summed("insurance"), "extra_charges": summed("extra_charge"),
        "excise_duties": summed("excise_duty"), "taxes": taxes,
        "line_items": [_line_json(line, candidate, supplier, masters, credit) for line in candidate.lines],
    }
    diagnostics = {"matches": {"supplier": supplier.__dict__, "po": po.__dict__, "payment_term": term.__dict__, "buyer": buyer,
                               "taxes": [match_tax(t, candidate, supplier, masters).__dict__ for t in header_taxes + line_tax_records]},
                   "reconciliation": reconcile(payable, candidate)}
    return payable, diagnostics


def reconcile(payable: dict[str, Any], candidate: PayableCandidate) -> dict[str, Any]:
    """Compare ERP's computed gross to printed gross; never adjust source values."""
    if not _found(candidate.printed_gross_total):
        return {"status": "unresolved", "reason": "Printed gross is missing, ambiguous, conflicting, or unreadable.", "erp_gross": "", "printed_gross": ""}
    critical = [candidate.currency, candidate.printed_gross_total]
    critical.extend(x for line in candidate.lines for x in (line.quantity, line.unit_price, line.price_basis))
    if not candidate.lines:
        return {"status": "unresolved", "reason": "No supported line components were extracted; the ERP total alone cannot establish the payable structure.", "erp_gross": "", "printed_gross": _amount(candidate.printed_gross_total, True)}
    for tax in candidate.taxes + [tax for line in candidate.lines for tax in line.taxes]:
        if tax.kind != "unknown" and not _found(tax.amount) and not _found(tax.rate):
            critical.append(tax.amount)
    for charge in candidate.charges:
        critical.append(charge.amount)
    unknown = [f for f in critical if f.status != "found"]
    if any(_found(line.price_basis) and line.price_basis.value != "net" for line in candidate.lines):
        reason = "At least one unit price is not established as net, as required by the schema."
        return {"status": "unresolved", "reason": reason, "erp_gross": "", "printed_gross": _amount(candidate.printed_gross_total, True)}
    try:
        booked = Decimal(str(erp_book(payable)["will_book_gross"])).quantize(Decimal("0.01"))
    except Exception as exc:  # malformed candidate should be reported instead of crashing batch processing
        return {"status": "unresolved", "reason": f"ERP calculation could not run: {exc}", "erp_gross": "", "printed_gross": _amount(candidate.printed_gross_total, True)}
    printed = abs(Decimal(str(candidate.printed_gross_total.value))) if payable["invoice_type"] == "CREDIT_MEMO" else Decimal(str(candidate.printed_gross_total.value))
    result = {"erp_gross": format(booked, ".2f"), "printed_gross": format(printed, ".2f"), "difference": format(booked - printed, ".2f")}
    if unknown:
        statuses = {field.status for field in unknown}
        if "unreadable" in statuses:
            reason = "Unresolved because required source text is unreadable or OCR/page extraction failed."
        elif statuses.intersection({"ambiguous", "conflicting"}):
            reason = "Unresolved because one or more required source values are ambiguous or conflicting."
        elif any(line.price_basis.status != "found" for line in candidate.lines):
            reason = "Unresolved because a line price is not evidenced as tax-exclusive net price, as required by the schema."
        else:
            reason = "Unresolved because required source components are missing; equality alone does not establish correctness."
        return {**result, "status": "unresolved", "reason": reason}
    if booked == printed:
        return {**result, "status": "passed", "reason": "ERP gross equals the determinable printed gross to the cent."}
    return {**result, "status": "failed", "reason": "Arithmetic mismatch: complete supported net line components do not reconcile to printed gross."}
