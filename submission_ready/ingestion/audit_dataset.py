"""Audit generated per-PDF accounting outputs against schema and masters."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from erp import erp_book


PAYABLE_KEYS = {"invoice_number", "invoice_date", "due_date", "invoice_type", "currency", "supplier", "buyer",
                "payment_term_id", "po_number", "po_id", "gross_total", "subtotal", "total_tax_amount",
                "discount_amount", "freight_charges", "insurance_charges", "extra_charges", "excise_duties", "taxes", "line_items"}
TAX_KEYS = {"tax_type", "tax_name", "tax_rate", "tax_amount", "tax_type_code"}
LINE_KEYS = {"description", "item_type", "uom", "quantity", "unit_price", "total", "discount", "discount_percentage", "tax_rate", "tax_amount", "taxes"}
TAX_TYPES = {"VAT", "GST", "WHT", "SST", "IVA", "NHIL", "GETFL", "COVID", "CST", "MOMS", "HST"}
NONPAYABLE_TYPES = {"payment_reminder", "delivery_note", "statement", "purchase_order", "receipt", "remittance", "quotation", "contract", "timesheet", "supporting_document", "debit_note"}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(input_dir: Path, output_dir: Path, diagnostics_dir: Path, master_dir: Path) -> dict[str, Any]:
    inputs = {p.stem + ".json" for p in input_dir.rglob("*") if p.is_file() and p.suffix.casefold() == ".pdf"}
    output_paths = {p.name: p for p in output_dir.glob("*.json")}
    diag_paths = {p.name: p for p in diagnostics_dir.glob("*.json")}
    masters = {name: _read(master_dir / f"{name}.json") for name in ("suppliers", "chart_of_books", "payment_terms", "po_master", "tax_master")}
    suppliers = {str(x["supplier_id"]) for x in masters["suppliers"]["suppliers"]}
    companies = {str(x["company_code"]): x for x in masters["chart_of_books"]["companies"]}
    terms = {str(x["payment_term_id"]) for x in masters["payment_terms"]["payment_terms"]}
    pos = {str(x["po_id"]) for x in masters["po_master"]["purchase_orders"]}
    tax_codes = {str(x["code"]) for x in masters["tax_master"]["taxes"]}
    issues: list[str] = []
    counts: Counter[str] = Counter()
    outputs: dict[str, dict[str, Any]] = {}
    for filename in sorted(inputs):
        if filename not in output_paths:
            issues.append(f"missing output: {filename}")
            continue
        try:
            data = _read(output_paths[filename])
            outputs[filename] = data
        except Exception as exc:
            issues.append(f"invalid JSON {filename}: {exc}")
            continue
        if set(data) != {"file", "payables", "declined"} or data.get("file") != filename[:-5] + ".pdf":
            issues.append(f"top-level keys/file mismatch: {filename}")
        if not isinstance(data.get("payables"), list) or not isinstance(data.get("declined"), list):
            issues.append(f"payables/declined must be arrays: {filename}")
            continue
        counts["payables"] += len(data["payables"])
        counts["declined"] += len(data["declined"])
        for index, payable in enumerate(data["payables"]):
            label = f"{filename} payable[{index}]"
            if set(payable) != PAYABLE_KEYS:
                issues.append(f"schema keys mismatch: {label}")
                continue
            if payable["invoice_type"] not in {"INVOICE", "CREDIT_MEMO"}:
                issues.append(f"invalid invoice_type: {label}")
            if not isinstance(payable["taxes"], list) or not isinstance(payable["line_items"], list):
                issues.append(f"taxes/line_items must be arrays: {label}")
                continue
            for tax in payable["taxes"]:
                if set(tax) != TAX_KEYS or tax.get("tax_type") not in TAX_TYPES:
                    issues.append(f"invalid header tax structure/type: {label}")
                if tax.get("tax_type_code") and tax["tax_type_code"] not in tax_codes:
                    issues.append(f"unknown tax code {tax.get('tax_type_code')}: {label}")
            for line in payable["line_items"]:
                if set(line) != LINE_KEYS or line.get("item_type") not in {"GOODS", "SERVICE", "FREIGHT", "TAX"}:
                    issues.append(f"invalid line structure/type: {label}")
                if not isinstance(line.get("taxes"), list):
                    issues.append(f"line taxes must be an array: {label}")
                else:
                    for tax in line["taxes"]:
                        if set(tax) != TAX_KEYS or tax.get("tax_type") not in TAX_TYPES:
                            issues.append(f"invalid line tax structure/type: {label}")
                        if tax.get("tax_type_code") and tax["tax_type_code"] not in tax_codes:
                            issues.append(f"unknown line tax code {tax.get('tax_type_code')}: {label}")
            if payable["supplier"].get("supplier_id") and payable["supplier"]["supplier_id"] not in suppliers:
                issues.append(f"unknown supplier code: {label}")
            if payable["buyer"].get("company_code"):
                company = companies.get(payable["buyer"]["company_code"])
                if not company:
                    issues.append(f"unknown company code: {label}")
                else:
                    units = {x["business_unit_code"]: x for x in company.get("business_units", [])}
                    unit = units.get(payable["buyer"].get("business_unit_code"))
                    if payable["buyer"].get("business_unit_code") and not unit:
                        issues.append(f"unknown business-unit code: {label}")
                    elif unit and payable["buyer"].get("location_code") and payable["buyer"]["location_code"] not in {x["location_code"] for x in unit.get("locations", [])}:
                        issues.append(f"unknown location code: {label}")
            if payable["payment_term_id"] and payable["payment_term_id"] not in terms:
                issues.append(f"unknown payment-term code: {label}")
            if payable["po_id"] and payable["po_id"] not in pos:
                issues.append(f"unknown PO code: {label}")
            try:
                erp_result = erp_book(payable)
                counts["erp_calculated"] += 1
                if not erp_result.get("currency"):
                    issues.append(f"ERP returned empty currency: {label}")
            except Exception as exc:
                issues.append(f"ERP failed for {label}: {exc}")
        for entry in data["declined"]:
            if not isinstance(entry, dict) or not entry.get("reason") or entry.get("doc_type") not in NONPAYABLE_TYPES:
                issues.append(f"decline lacks positive non-payable type/reason: {filename}")
        if filename not in diag_paths:
            issues.append(f"missing diagnostics: {filename}")
    if output_paths.keys() - inputs:
        issues.extend(f"unexpected output: {name}" for name in sorted(output_paths.keys() - inputs))
    for filename in sorted(inputs & diag_paths.keys()):
        try:
            diag = _read(diag_paths[filename])
        except Exception as exc:
            issues.append(f"invalid diagnostics {filename}: {exc}")
            continue
        pages = diag.get("pages", [])
        counts["pages"] += len(pages)
        counts["ocr_timeout_pages"] += sum(page.get("extraction_method") == "ocr_timeout" for page in pages)
        counts["extraction_error_pages"] += sum(page.get("extraction_method") in {"ocr_failed", "page_failed", "ocr_empty"} for page in pages)
        for candidate in diag.get("extraction", {}).get("candidates", []):
            status = candidate.get("reconciliation", {}).get("status")
            if status in {"passed", "failed", "unresolved"}:
                counts[f"erp_{status}"] += 1
        data = outputs.get(filename, {})
        if not data.get("payables") and not data.get("declined"):
            counts["unresolved_documents"] += 1
    counts["input_pdfs"] = len(inputs)
    counts["output_files"] = len(output_paths)
    counts["diagnostics_files"] = len(diag_paths)
    counts["missing_outputs"] = len(inputs - output_paths.keys())
    counts["missing_diagnostics"] = len(inputs - diag_paths.keys())
    return {"counts": dict(counts), "issues": issues}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("documents"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--diagnostics-dir", type=Path, default=Path("output_diagnostics"))
    parser.add_argument("--master-dir", type=Path, default=Path("master_data"))
    args = parser.parse_args(argv)
    result = audit(args.input_dir, args.output_dir, args.diagnostics_dir, args.master_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
