"""Unseen-layout line tests: source terminology, wrapping, and ambiguity."""

from __future__ import annotations

import unittest
from decimal import Decimal

from erp import erp_book
from ingestion.autodraft import MasterData, generate_autodraft
from ingestion.models import DocumentClassification, DocumentRecord, PageRecord
from ingestion.payables import extract_payables
from ingestion.payables.line_type import classify_line_type


class LineItemGeneralizationTests(unittest.TestCase):
    def _candidate(self, rows: list[str], *, supplier: str = "New Example LLC", number: str = "Q-91"):
        text = "\n".join([
            f"Invoice Number: {number}", "Invoice Date: 2026-05-01",
            f"Supplier: {supplier}", "Currency: EUR",
            "Description Quantity Net Unit Price Amount EUR", *rows,
            "Subtotal EUR 30.00", "Invoice Total EUR 30.00",
        ])
        page = PageRecord("unseen.pdf", 1, text, "fixture")
        doc = DocumentRecord(page.filename, [page], DocumentClassification("invoice", 0.9, False))
        return extract_payables(doc).candidates[0]

    def test_explicit_terms_and_competing_signals(self):
        cases = [
            ("Consulting Service", "service"),
            ("Management Fee May", "service"),
            ("International Freight", "freight"),
            ("Industrial Mixer", "goods"),
            ("Business Rates", None),
            ("Utilities", None),
            ("Service Kit", None),
            ("Transport Equipment", None),
            ("Shipping Software", None),
        ]
        for description, expected in cases:
            with self.subTest(description=description):
                self.assertEqual(classify_line_type(description), expected)

    def test_wrapped_unnumbered_rows_retain_printed_components(self):
        for supplier, number, order in [
            ("Cedar Plain LLC", "Q-91", ["Consulting Service 1 10.00 10.00", "Industrial Mixer 2 10.00 20.00"]),
            ("Coastal Works Ltd", "Q-92", ["Industrial Mixer 2 10.00 20.00", "Consulting Service 1 10.00 10.00"]),
        ]:
            with self.subTest(supplier=supplier):
                wrapped = ["Annual on-site", order[0], *order[1:]]
                candidate = self._candidate(wrapped, supplier=supplier, number=number)
                self.assertEqual(len(candidate.lines), 2)
                self.assertEqual(candidate.lines[0].description.value, "Annual on-site " + order[0].rsplit(" ", 3)[0])
                self.assertEqual(sum(x.printed_total.value for x in candidate.lines), Decimal("30.00"))
                payable, diagnostics = generate_autodraft(candidate, MasterData())
                self.assertEqual(erp_book(payable)["will_book_gross"], 30.0)
                self.assertEqual(diagnostics["reconciliation"]["status"], "passed")

    def test_ambiguous_type_stays_missing_even_when_arithmetic_foots(self):
        candidate = self._candidate(["Business Rates 1 30.00 30.00"])
        self.assertEqual(candidate.lines[0].item_type.status, "missing")
        payable, diagnostics = generate_autodraft(candidate, MasterData())
        self.assertEqual(erp_book(payable)["will_book_gross"], 30.0)
        self.assertEqual(diagnostics["reconciliation"]["status"], "passed")
        self.assertEqual(payable["line_items"][0]["item_type"], "")

    def test_amount_only_row_keeps_missing_quantity_and_price(self):
        text = "\n".join(["Invoice Number: Q-93", "Invoice Date: 2026-05-01",
                          "Supplier: New Example LLC", "Currency: EUR", "Description Amount EUR",
                          "Platform access 30.00", "Invoice Total EUR 30.00"])
        page = PageRecord("unseen.pdf", 1, text, "fixture")
        candidate = extract_payables(DocumentRecord(page.filename, [page], DocumentClassification("invoice", 0.9, False))).candidates[0]
        self.assertEqual(candidate.lines[0].quantity.status, "missing")
        self.assertEqual(candidate.lines[0].unit_price.status, "missing")


if __name__ == "__main__":
    unittest.main()
