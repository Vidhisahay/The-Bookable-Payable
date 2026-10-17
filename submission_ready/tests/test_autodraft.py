from __future__ import annotations

import unittest
from datetime import date
from decimal import Decimal

from ingestion.autodraft import MasterData, generate_autodraft, match_supplier
from ingestion.payables.models import ExtractedField, PayableCandidate, LineItem, TaxItem, ChargeItem, Evidence


def f(value=None, status="found"):
    return ExtractedField(value, status, [Evidence("fixture.pdf", 1, f"source: {value}")] if status == "found" else [])


def candidate(**overrides):
    values = dict(
        document_type=f("invoice"), invoice_number=f("INV-1"), invoice_date=f(date(2026, 1, 1)), due_date=f(date(2026, 1, 11)),
        supplier_name=f("Phocus Direct Communication GmbH"), supplier_address=f("Lina-Ammon-Strasse 19b, 90471 Nurnberg, DE"),
        supplier_vat_id=f("DE209177122"), currency=f("EUR"), printed_gross_total=f(Decimal("119.00")),
        printed_subtotal=f(Decimal("100.00")), printed_tax_total=f(Decimal("19.00")), purchase_order_number=f(""),
        lines=[LineItem(f("Service"), f("SERVICE"), f(Decimal("1")), f("EA"), f(Decimal("100")), f("net"), f(Decimal("100")),
                        [TaxItem("line", "tax", f("VAT"), f(Decimal("19")), f(Decimal("19")))])],
        taxes=[], charges=[], pages=[1])
    values.update(overrides)
    return PayableCandidate(**values)


class AutodraftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.masters = MasterData.load()

    def test_supplier_success_requires_identity_evidence(self):
        self.assertEqual(match_supplier(candidate(), self.masters).code, "2845695")

    def test_supplier_name_only_is_insufficient_and_nonmatch_is_blank(self):
        c = candidate(supplier_vat_id=f(None, "missing"), supplier_address=f(None, "missing"))
        self.assertEqual(match_supplier(c, self.masters).status, "insufficient_evidence")
        c.supplier_name = f("No Such Vendor")
        self.assertEqual(match_supplier(c, self.masters).code, "")

    def test_ambiguous_supplier_not_selected(self):
        masters = MasterData(suppliers=[{"supplier_id": "1", "name": "Same", "address": "A", "vat_id": ""},
                                       {"supplier_id": "2", "name": "Same", "address": "A", "vat_id": ""}])
        c = candidate(supplier_name=f("Same"), supplier_address=f("A"), supplier_vat_id=f(None, "missing"))
        self.assertEqual(match_supplier(c, masters).status, "ambiguous")
        self.assertEqual(match_supplier(c, masters).code, "")

    def test_schema_matching_and_tax_code(self):
        c = candidate(taxes=[TaxItem("header", "tax", f("VAT reverse charge"), f(Decimal("0")), f(Decimal("0")))])
        payable, diag = generate_autodraft(c, self.masters, buyer_hints={"company_code": "BOLTGROUP", "business_unit_code": "EE004", "location_code": "LOC_EE_001"}, payment_term_text="Net 10")
        self.assertEqual(payable["supplier"]["supplier_id"], "2845695")
        self.assertEqual(payable["buyer"], {"company_code": "BOLTGROUP", "business_unit_code": "EE004", "location_code": "LOC_EE_001"})
        self.assertEqual(payable["payment_term_id"], "Net_10")
        self.assertEqual(payable["taxes"][0]["tax_type_code"], "DE_000_RC")
        self.assertEqual(payable["line_items"][0]["taxes"][0]["tax_type_code"], "DE_190_VAT")
        self.assertEqual(set(payable), {"invoice_number", "invoice_date", "due_date", "invoice_type", "currency", "supplier", "buyer", "payment_term_id", "po_number", "po_id", "gross_total", "subtotal", "total_tax_amount", "discount_amount", "freight_charges", "insurance_charges", "extra_charges", "excise_duties", "taxes", "line_items"})
        self.assertEqual(diag["reconciliation"]["status"], "passed")

    def test_unresolved_codes_stay_empty_and_po_supplier_relationship_is_required(self):
        c = candidate(purchase_order_number=f("PO-EE-2026-0044"), supplier_name=f("unknown"), supplier_address=f(None, "missing"), supplier_vat_id=f(None, "missing"))
        payable, diag = generate_autodraft(c, self.masters)
        self.assertEqual(payable["po_number"], "PO-EE-2026-0044")
        self.assertEqual(payable["po_id"], "")
        self.assertEqual(payable["supplier"]["supplier_id"], "")
        self.assertEqual(diag["matches"]["po"]["status"], "insufficient_evidence")

    def test_po_match_requires_exact_number_and_supplier(self):
        c = candidate(purchase_order_number=f("PO-EE-2026-0044"), supplier_name=f("Ehast Koiduni OU"),
                      supplier_address=f("Parnu mnt 158, 11317 Tallinn, EE"), supplier_vat_id=f("EE101234567"))
        payable, _ = generate_autodraft(c, self.masters)
        self.assertEqual(payable["po_id"], "PO-EE-2026-0044")
        c.purchase_order_number = f("PO-EE-2026-004")
        payable, _ = generate_autodraft(c, self.masters)
        self.assertEqual(payable["po_id"], "")

    def test_tax_needs_unique_country_rate_type_match(self):
        c = candidate(taxes=[TaxItem("header", "tax", f("VAT"), f(Decimal("7")), f(Decimal("7")))])
        payable, _ = generate_autodraft(c, self.masters)
        self.assertEqual(payable["taxes"][0]["tax_type_code"], "DE_070_VAT")
        c.taxes = [TaxItem("header", "tax", f("VAT"), f(None, "missing"), f(Decimal("7")))]
        payable, _ = generate_autodraft(c, self.masters)
        self.assertEqual(payable["taxes"][0]["tax_type_code"], "")

    def test_credit_memo_positive_components(self):
        c = candidate(document_type=f("credit_memo"), printed_gross_total=f(Decimal("-119.00")),
                      printed_subtotal=f(Decimal("-100.00")), printed_tax_total=f(Decimal("-19.00")))
        payable, diag = generate_autodraft(c, self.masters)
        self.assertEqual(payable["invoice_type"], "CREDIT_MEMO")
        self.assertEqual(payable["gross_total"], "119.00")
        self.assertEqual(payable["line_items"][0]["total"], "100")
        self.assertEqual(diag["reconciliation"]["status"], "passed")

    def test_credit_reference_is_not_emitted_as_credit_memo_number(self):
        c = candidate(document_type=f("credit_memo"))
        c.invoice_number = ExtractedField("6265", "found", [Evidence("fixture.pdf", 1, "Kreeditarve arvele nr. 6265")])
        payable, _ = generate_autodraft(c, self.masters)
        self.assertEqual(payable["invoice_number"], "")

    def test_line_and_header_taxes_stay_at_source_scope(self):
        c = candidate(taxes=[TaxItem("header", "tax", f("VAT"), f(Decimal("19")), f(Decimal("19")))])
        payable, _ = generate_autodraft(c, self.masters)
        self.assertEqual(len(payable["taxes"]), 1)
        self.assertEqual(len(payable["line_items"][0]["taxes"]), 1)

    def test_charges_are_decomposed_and_not_used_to_force_match(self):
        c = candidate(printed_gross_total=f(Decimal("124.00")), charges=[
            ChargeItem("freight", f("Freight EUR 5.00"), f(Decimal("5.00"))),
            ChargeItem("discount", f("Discount EUR 2.00"), f(Decimal("2.00")))])
        payable, diag = generate_autodraft(c, self.masters)
        self.assertEqual(payable["freight_charges"], "5.00")
        self.assertEqual(payable["discount_amount"], "2.00")
        self.assertEqual(diag["reconciliation"]["status"], "failed")

    def test_missing_component_or_non_net_price_is_unresolved(self):
        c = candidate(lines=[])
        _, diag = generate_autodraft(c, self.masters)
        self.assertEqual(diag["reconciliation"]["status"], "unresolved")
        c = candidate(lines=[LineItem(f("service"), f("SERVICE"), f(Decimal("1")), f("EA"), f(Decimal("100")), f("tax_inclusive"), f(Decimal("100")), [])])
        _, diag = generate_autodraft(c, self.masters)
        self.assertEqual(diag["reconciliation"]["status"], "unresolved")


if __name__ == "__main__":
    unittest.main()
