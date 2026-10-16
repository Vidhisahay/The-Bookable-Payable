from __future__ import annotations

import unittest
from decimal import Decimal

from ingestion.models import DocumentClassification, DocumentRecord, PageRecord
from ingestion.payables import extract_payables
from ingestion.payables.numbers import parse_localized_decimal


class PayableExtractionTests(unittest.TestCase):
    def document(self, text: str, category: str = "invoice", filename: str = "sample.pdf") -> DocumentRecord:
        page = PageRecord(filename, 1, text, "fixture_ocr", {"fixture": True})
        return DocumentRecord(filename, [page], DocumentClassification(category, 0.9, False))

    def test_german_invoice_fields_lines_taxes_and_evidence(self):
        text = """Rechnung Nr.: 852566
Rechnungsdatum: 02.02.2026
Bitte begleichen Sie den Betrag bis zum 12.02.2026.
Pos Beschreibung Einzelpreis Menge Gesamtpreis
1 Projektmanagement Nachberechnung aus November 73,00 EUR 4 Std. 292,00 EUR
2 Projektmanagement Nachberechnung aus Dezember 73,00 EUR 2 Std. 146,00 EUR
Gesamtsumme 438,00 EUR
zzgl. 0 % MwSt 0,00 EUR
Endbetrag 438,00 EUR
Vat ID: EE398766194
Reverse Charge"""
        outcome = extract_payables(self.document(text, filename="INV-01.pdf"))
        self.assertEqual(len(outcome.candidates), 1)
        candidate = outcome.candidates[0]
        self.assertEqual(candidate.invoice_number.value, "852566")
        self.assertEqual(candidate.invoice_date.value.isoformat(), "2026-02-02")
        self.assertEqual(candidate.due_date.value.isoformat(), "2026-02-12")
        self.assertEqual(candidate.supplier_vat_id.value, "EE398766194")
        self.assertEqual(candidate.currency.value, "EUR")
        self.assertEqual(candidate.printed_gross_total.value, Decimal("438.00"))
        self.assertEqual(candidate.printed_tax_total.value, Decimal("0.00"))
        self.assertEqual(len(candidate.lines), 2)
        self.assertEqual(candidate.lines[0].quantity.value, Decimal("4"))
        self.assertEqual(candidate.lines[0].unit.value, "Std")
        self.assertEqual(candidate.lines[0].unit_price.value, Decimal("73.00"))
        self.assertEqual(candidate.lines[0].printed_total.value, Decimal("292.00"))
        self.assertEqual(candidate.lines[0].price_basis.value, "net")
        self.assertIn("zzgl. 0 % MwSt", candidate.lines[0].price_basis.evidence[0].text)
        self.assertTrue(candidate.lines[0].description.evidence[0].text.startswith("1 Projektmanagement"))
        self.assertEqual(candidate.invoice_number.evidence[0].filename, "INV-01.pdf")
        self.assertEqual(candidate.invoice_number.evidence[0].page_number, 1)
        self.assertEqual(candidate.taxes[0].scope, "header")
        self.assertEqual(candidate.taxes[0].rate.value, Decimal("0"))
        self.assertEqual(candidate.taxes[0].amount.value, Decimal("0.00"))
        self.assertEqual(candidate.supplier_name.status, "missing")
        self.assertTrue(candidate.review_required)

    def test_payment_reminder_is_not_extracted_as_a_new_payable(self):
        text = "Mahnung Payment reminder. Rechnung VRB700A6126940 total EUR 48,94. Bitte zahlen."
        outcome = extract_payables(self.document(text, category="payment_reminder", filename="DU-08.pdf"))
        self.assertEqual(outcome.candidates, [])
        self.assertEqual(outcome.excluded_reason, "payment_reminder")
        self.assertIn("does not establish a new payable", outcome.warnings[0])

    def test_credit_memo_is_detected_from_document_evidence(self):
        text = "Credit Note No.: CN-2041\nDate: 15/04/2026\nTotal Amount GBP -125.50\n"
        outcome = extract_payables(self.document(text, filename="mixed-classifier.pdf"))
        candidate = outcome.candidates[0]
        self.assertEqual(candidate.document_type.value, "credit_memo")
        self.assertEqual(candidate.invoice_number.value, "CN-2041")
        self.assertEqual(candidate.printed_gross_total.value, Decimal("-125.50"))
        # Date-only context uses DMY when the day is unambiguous.
        self.assertEqual(candidate.invoice_date.value.isoformat(), "2026-04-15")

    def test_ambiguous_numeric_date_and_locale_amount_are_not_guessed(self):
        text = "Invoice No. AB-123\nDate: 03/04/2026\nGross total EUR 1.234\n"
        candidate = extract_payables(self.document(text)).candidates[0]
        self.assertEqual(candidate.invoice_date.status, "ambiguous")
        self.assertEqual(candidate.printed_gross_total.status, "ambiguous")
        parsed = parse_localized_decimal("1.234")
        self.assertTrue(parsed.ambiguous)
        self.assertIsNone(parsed.value)

        dollar_candidate = extract_payables(self.document(
            "Invoice No. INV-1\nInvoice date: 2026-03-01\nSupplier: Acme\nGross total $12.00"
        )).candidates[0]
        self.assertEqual(dollar_candidate.currency.status, "ambiguous")

    def test_multiple_printed_totals_are_conflicting(self):
        text = "Invoice number: INV-20\nInvoice date: 2026-03-01\nGrand total EUR 100.00\nTotal amount EUR 110.00"
        candidate = extract_payables(self.document(text)).candidates[0]
        self.assertEqual(candidate.printed_gross_total.status, "conflicting")
        self.assertEqual(candidate.printed_gross_total.alternatives, [Decimal("100.00"), Decimal("110.00")])

    def test_explicit_supplier_address_vat_and_purchase_order_are_preserved(self):
        text = """Invoice number INV-5
Invoice date: 2026-03-01
Supplier Name: Acme Services Ltd
Supplier Address: 10 Main Street, Dublin
VAT Registration No: IE1234567A
Purchase Order No: PO-7788
Gross total EUR 30.00"""
        candidate = extract_payables(self.document(text)).candidates[0]
        self.assertEqual(candidate.supplier_name.value, "Acme Services Ltd")
        self.assertEqual(candidate.supplier_address.value, "10 Main Street, Dublin")
        self.assertEqual(candidate.supplier_vat_id.value, "IE1234567A")
        self.assertEqual(candidate.purchase_order_number.value, "PO-7788")

    def test_price_basis_is_only_set_when_the_table_states_it(self):
        text = "Invoice No. AA-1\nInvoice date: 2026-03-01\nDescription Quantity Unit price excl VAT VAT amount\n1 Widget 2 10.00 7% 1.40\nGrand total EUR 21.40"
        candidate = extract_payables(self.document(text)).candidates[0]
        self.assertEqual(candidate.lines[0].price_basis.value, "net")
        self.assertEqual(candidate.lines[0].price_basis.status, "found")
        self.assertEqual(candidate.lines[0].quantity.value, Decimal("2"))
        self.assertEqual(candidate.lines[0].taxes[0].scope, "line")
        self.assertEqual(candidate.lines[0].taxes[0].rate.value, Decimal("7"))
        self.assertEqual(candidate.lines[0].taxes[0].amount.value, Decimal("1.40"))
        self.assertEqual(candidate.lines[0].taxes[0].amount.status, "found")

    def test_ocr_line_quantity_near_currency_is_not_mistaken_for_printed_total(self):
        text = """Rechnung Nr.: 852566
Pos Beschreibung Einzelpreis Menge Gesamtpreis
1 Projektmanagement 73,00 EUR 4 Std. 292,00 EUR
Gesamtsumme 438,00 EUR"""
        candidate = extract_payables(self.document(text)).candidates[0]
        self.assertEqual(candidate.lines[0].quantity.value, Decimal("4"))
        self.assertEqual(candidate.lines[0].unit.value, "Std")
        self.assertEqual(candidate.lines[0].unit_price.value, Decimal("73.00"))
        self.assertEqual(candidate.lines[0].printed_total.value, Decimal("292.00"))

    def test_date_fragment_on_total_label_is_not_treated_as_money(self):
        text = """Rechnung Nr.: 852566
Lieferdatum: Zeitraum von 01.01.2026 bis 31.01.2026 Gesamtsumme
438,00 EUR"""
        candidate = extract_payables(self.document(text)).candidates[0]
        self.assertEqual(candidate.printed_gross_total.value, Decimal("438.00"))
        self.assertEqual(candidate.printed_gross_total.status, "found")

    def test_native_pdf_label_and_value_on_adjacent_lines(self):
        text = """INVOICE
Invoice Number:
39015
Invoice Date:
6/1/2018
Bill Date: 05/23/2018
Due Date:
6/18/2018
Remit payment to:
100 Airport KPG III, LLC.
PO#16299
Amount Due:
17,657.53"""
        candidate = extract_payables(self.document(text)).candidates[0]
        self.assertEqual(candidate.invoice_number.value, "39015")
        self.assertEqual(candidate.invoice_date.status, "ambiguous")
        self.assertEqual(candidate.due_date.value.isoformat(), "2018-06-18")
        self.assertEqual(candidate.supplier_name.value, "100 Airport KPG III, LLC.")
        self.assertEqual(candidate.purchase_order_number.value, "16299")
        self.assertEqual(candidate.printed_gross_total.value, Decimal("17657.53"))

    def test_date_printed_beside_invoice_number_label_is_not_the_identifier(self):
        text = """INVOICE
Invoice Number: 6/1/2018
Invoice Number: 39015
Total Amount Due on 06/18/2018 $17,657.53"""
        candidate = extract_payables(self.document(text)).candidates[0]
        self.assertEqual(candidate.invoice_number.value, "39015")
        self.assertEqual(candidate.invoice_date.status, "missing")
        self.assertEqual(candidate.due_date.value.isoformat(), "2018-06-18")
        self.assertEqual(candidate.printed_gross_total.value, Decimal("17657.53"))

    def test_adjacent_date_value_under_invoice_number_label_is_filtered(self):
        text = """INVOICE
Invoice Number:
6/1/2018
Invoice Number:
39015
Amount Due: 17,657.53"""
        candidate = extract_payables(self.document(text)).candidates[0]
        self.assertEqual(candidate.invoice_number.value, "39015")

    def test_unpaired_due_date_label_prevents_selecting_a_later_conflicting_due_date(self):
        text = """INVOICE
Due Date:
Invoice For: AmeriHealth Caritas
Total Amount Due on 06/18/2018 $17,657.53"""
        candidate = extract_payables(self.document(text)).candidates[0]
        self.assertEqual(candidate.due_date.status, "ambiguous")
        self.assertIsNone(candidate.due_date.value)
        self.assertIn("Due Date:", [item.text for item in candidate.due_date.evidence])

    def test_supplier_preceding_remit_label_in_pdf_reading_order_is_preserved(self):
        text = """INVOICE
100 Airport KPG III, LLC.
Remit payment to:
Amount Due: 17,657.53"""
        candidate = extract_payables(self.document(text)).candidates[0]
        self.assertEqual(candidate.supplier_name.value, "100 Airport KPG III, LLC.")
        self.assertEqual(candidate.supplier_name.evidence[0].page_number, 1)

    def test_tax_conversion_amount_does_not_replace_invoice_amount_due(self):
        text = """TAX INVOICE
Total Amount Due
Amount Due USD 1,040.06
Conversion to SGD for Tax purpose only
Total Amount Due (SGD) 1,350.55
Payment in USD"""
        candidate = extract_payables(self.document(text)).candidates[0]
        self.assertEqual(candidate.printed_gross_total.value, Decimal("1040.06"))
        self.assertEqual(candidate.currency.value, "USD")

    def test_duplicate_header_tax_on_repeated_invoice_summary_is_merged_with_evidence(self):
        from ingestion.models import DocumentClassification, DocumentRecord, PageRecord
        text = "TAX INVOICE\nGST 0% 0.00\nAmount Due USD 10.00"
        pages = [PageRecord("repeat.pdf", number, text, "fixture") for number in (1, 2)]
        candidate = extract_payables(DocumentRecord("repeat.pdf", pages, DocumentClassification("invoice", 0.9, False))).candidates[0]
        self.assertEqual(len(candidate.taxes), 1)
        self.assertEqual([item.page_number for item in candidate.taxes[0].amount.evidence], [1, 2])

    def test_flattened_quantity_unit_price_exchange_rate_table(self):
        text = """TAX INVOICE
Qty UoM Curr Unit Price ROE Total(USD)
1 International Freight 567.000 KG USD 1.28 1.00000 725.76
Total Amount Due USD 725.76"""
        candidate = extract_payables(self.document(text)).candidates[0]
        self.assertEqual(len(candidate.lines), 1)
        line = candidate.lines[0]
        self.assertEqual(line.description.value, "International Freight")
        self.assertEqual(line.quantity.value, Decimal("567.000"))
        self.assertEqual(line.unit.value, "KG")
        self.assertEqual(line.unit_price.value, Decimal("1.28"))
        self.assertEqual(line.printed_total.value, Decimal("725.76"))

    def test_multiple_invoice_sections_become_separate_candidates(self):
        pages = [
            PageRecord("mixed.pdf", 1, "Invoice number INV-101\nInvoice date: 2026-01-01\nGrand total EUR 10.00", "fixture"),
            PageRecord("mixed.pdf", 2, "Invoice number INV-202\nInvoice date: 2026-02-01\nGrand total EUR 20.00", "fixture"),
        ]
        classification = DocumentClassification(
            "invoice", 0.55, True, possible_document_starts=[2], potential_multiple_documents=True,
        )
        outcome = extract_payables(DocumentRecord("mixed.pdf", pages, classification))
        self.assertEqual(len(outcome.candidates), 2)
        self.assertEqual([candidate.invoice_number.value for candidate in outcome.candidates], ["INV-101", "INV-202"])
        self.assertEqual([candidate.pages for candidate in outcome.candidates], [[1], [2]])
        self.assertTrue(all(candidate.review_required for candidate in outcome.candidates))

    def test_thai_invoice_preserves_printed_gross_and_uncertain_row_fields(self):
        text = """ใบวางบิล (INVOICE)
ITEM QUANTITY UNIT PRICE AMOUNT
1 Staff 2 Units X 6 Days 2 600.00 7,200.00
Vat 7 %
GRAND TOTAL (INCLUDING VAT) 8,397.36
WITHHOLDING TAX 3% 235.44
BAHT"""
        candidate = extract_payables(self.document(text, filename="HLD-01.pdf")).candidates[0]
        self.assertEqual(candidate.currency.value, "THB")
        self.assertEqual(candidate.printed_gross_total.value, Decimal("8397.36"))
        self.assertEqual(candidate.printed_tax_total.status, "missing")
        self.assertEqual(candidate.taxes[0].rate.value, Decimal("7"))
        withholding = next(tax for tax in candidate.taxes if tax.kind == "withholding")
        self.assertEqual(withholding.amount.value, Decimal("235.44"))
        self.assertEqual(candidate.lines[0].unit_price.value, Decimal("600.00"))
        self.assertEqual(candidate.lines[0].printed_total.value, Decimal("7200.00"))
        self.assertEqual(candidate.lines[0].quantity.status, "ambiguous")
        self.assertTrue(candidate.review_required)

    def test_estonian_credit_memo_preserves_negative_total_and_tax_lines(self):
        text = """Kuupäev 16.03.2025
Maksetähtaeg 16.03.2025
Arve kokku -400,00 EUR
Toode/Teenus Hind Kogus KM% Summa
Toitlustamine Tavolo 11.03.25 -327,868852 1 (tk) 22 -327,87
Summa km-ta 22% -327,87
KM 22% -72,13
Kreeditarve arvele nr. 6265."""
        candidate = extract_payables(self.document(text, filename="DU-11.pdf")).candidates[0]
        self.assertEqual(candidate.document_type.value, "credit_memo")
        self.assertEqual(candidate.invoice_number.value, "6265")
        self.assertEqual(candidate.invoice_date.value.isoformat(), "2025-03-16")
        self.assertEqual(candidate.due_date.value.isoformat(), "2025-03-16")
        self.assertEqual(candidate.currency.value, "EUR")
        self.assertEqual(candidate.printed_gross_total.value, Decimal("-400.00"))
        self.assertEqual(candidate.printed_subtotal.value, Decimal("-327.87"))
        self.assertEqual(candidate.printed_tax_total.value, Decimal("-72.13"))


if __name__ == "__main__":
    unittest.main()
