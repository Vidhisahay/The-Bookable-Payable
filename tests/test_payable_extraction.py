from __future__ import annotations

import unittest
from decimal import Decimal

from ingestion.models import DocumentClassification, DocumentRecord, PageRecord
from ingestion.payables import extract_payables
from ingestion.payables.numbers import parse_localized_decimal
from ingestion.autodraft import MasterData, generate_autodraft
from erp import erp_book


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
        self.assertEqual(candidate.supplier_vat_id.status, "ambiguous")
        self.assertIsNone(candidate.supplier_vat_id.value)
        self.assertIn("Vat ID: EE398766194", candidate.supplier_vat_id.evidence[0].text)
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

    def test_party_role_conflict_does_not_promote_first_legal_name(self):
        text = """Tax Invoice
Invoice Date: 2026-05-01  Silverbrook Media Limited
Northwind Services UK Ltd  Invoice Number: INV-1587
Bill To: Northwind Services UK Ltd
VAT Number: GB123456789
Description Quantity Unit Price VAT Amount GBP
Management Service June 1.00 100.00 20% 100.00
Subtotal 100.00
Total VAT 20% 20.00
Invoice Total GBP 120.00"""
        candidate = extract_payables(self.document(text)).candidates[0]
        # The explicit buyer block contradicts a naive "first legal name" rule;
        # without a role label or reliable VAT linkage, identity remains blank.
        self.assertEqual(candidate.supplier_name.status, "missing")

    def test_footer_issuer_vat_does_not_inherit_customer_vat(self):
        text = """Bill To: Cedar Customer Limited
Delivery Address: 20 Market Street
VAT Registration No.: GB111111111
Credit Note No.: 5900366703
Currency in: GBP
Product Reference No. Quantity Unit VAT Unit Price Net Amount
4582314 952-000091 1 20% 10.00 10.00
Touch Controller Device
Net Amount GBP 10.00
VAT 20% 2.00
Total Amount GBP 12.00
Blackpine Consulting Limited Registered in England
Registered Office: 50 Market Lane
VAT:GB222222222 Account Number: 12345678"""
        candidate = extract_payables(self.document(text, category="credit_memo")).candidates[0]
        self.assertEqual(candidate.supplier_name.value, "Blackpine Consulting Limited")
        self.assertEqual(candidate.supplier_vat_id.value, "GB222222222")
        self.assertIn("VAT:GB222222222", candidate.supplier_vat_id.evidence[0].text)

    def test_customer_vat_without_issuer_evidence_is_not_supplier_identity(self):
        text = """Bill To: Cedar Customer Limited
Delivery Address: 20 Market Street
VAT Registration No.: GB111111111
Invoice Number: Q-77
Currency: GBP
Description Quantity Net Unit Price Amount GBP
Consulting Service 1 10.00 10.00
Invoice Total GBP 10.00"""
        candidate = extract_payables(self.document(text)).candidates[0]
        self.assertEqual(candidate.supplier_name.status, "missing")
        self.assertEqual(candidate.supplier_vat_id.status, "missing")

    def test_addressee_in_header_is_not_supplier(self):
        text = """Northwind Operations OU
Herrn Alex Kask
Market Road 15
Invoice Number: R-830
Invoice Date: 2026-02-02
Dear Mr Kask,
Description Quantity Net Unit Price Amount EUR
Consulting Service 1 10.00 10.00
Invoice Total EUR 10.00
VAT ID: EE123456789"""
        candidate = extract_payables(self.document(text)).candidates[0]
        self.assertEqual(candidate.supplier_name.status, "missing")
        self.assertEqual(candidate.supplier_vat_id.status, "ambiguous")
        self.assertIsNone(candidate.supplier_vat_id.value)

    def test_flattened_reference_is_not_part_of_supplier_address(self):
        text = """Oakhaven Supply Ltd
Market Road 8
P.O. Box 14108, Nairobi 32917ALS55
00400 KEN
Invoice Number: Q-830
Bill To: Cedar Customer Limited
Description Quantity Net Unit Price Amount KES
Handling Service 1 10.00 10.00
Invoice Total KES 10.00"""
        candidate = extract_payables(self.document(text)).candidates[0]
        self.assertEqual(candidate.supplier_name.value, "Oakhaven Supply Ltd")
        self.assertEqual(candidate.supplier_address.status, "missing")

    def test_extended_amount_is_not_invented_as_unit_price_for_quantity_one(self):
        text = """INVOICE
Supplier: New Example LLC
Invoice Number: Q-831
Currency: USD
ITEM DESCRIPTION QTY/ NO. OF USERS TAX EXTENDED AMOUNT
1 Managed Cloud Service 1 N 91,580.50
SUBTOTAL USD 91,580.50
TOTAL(USD) 91,580.50"""
        candidate = extract_payables(self.document(text)).candidates[0]
        self.assertEqual(candidate.lines[0].quantity.value, Decimal("1"))
        self.assertEqual(candidate.lines[0].printed_total.value, Decimal("91580.50"))
        self.assertEqual(candidate.lines[0].unit_price.status, "missing")
        _payable, diagnostics = generate_autodraft(candidate, MasterData())
        self.assertEqual(diagnostics["reconciliation"]["status"], "unresolved")

    def test_spatial_role_evidence_resolves_supplier_when_party_columns_are_swapped(self):
        from ingestion.layout import normalized_words

        words = []
        # Buyer is on the left/top and seller on the right/bottom. Explicit
        # labels, not reading order or page position, establish each role.
        for block, line, text, box in [
            (1, 1, "Bill To", (20, 40, 90, 52)),
            (1, 1, "Northwind", (95, 40, 165, 52)),
            (1, 1, "Services", (170, 40, 220, 52)),
            (1, 1, "Ltd", (225, 40, 245, 52)),
            (2, 2, "Supplier", (340, 180, 410, 192)),
            (2, 2, "Silverbrook", (415, 180, 500, 192)),
            (2, 2, "Media", (505, 180, 545, 192)),
            (2, 2, "Limited", (550, 180, 600, 192)),
        ]:
            words.append({"text": text, "bbox": box, "block": block, "line": line, "confidence": 90})
        spatial = {"page_number": 1, "source": "native", "page_width": 600, "page_height": 800,
                   **normalized_words(words, 600, 800, source="native")}
        page = PageRecord("spatial-fixture.pdf", 1,
                          "Invoice number X-1\nInvoice date: 2026-01-01\nGross total EUR 10.00",
                          "native_text", {"text_layout": spatial})
        doc = DocumentRecord(page.filename, [page], DocumentClassification("invoice", 0.9, False))
        candidate = extract_payables(doc).candidates[0]
        self.assertEqual(candidate.supplier_name.value, "Silverbrook Media Limited")
        self.assertEqual(candidate.supplier_name.status, "found")
        self.assertEqual(candidate.printed_gross_total.value, Decimal("10.00"))

    def test_conflicting_spatial_supplier_labels_do_not_override_as_a_guess(self):
        from ingestion.layout import normalized_words

        words = []
        for line, row in enumerate([
            [("Supplier", 20), ("Asteria", 90), ("Limited", 145)],
            [("Supplier", 360), ("Beacon", 430), ("Trading", 485), ("LLC", 540)],
        ], 1):
            for text, x in row:
                words.append({"text": text, "bbox": [x, line * 80, x + 48, line * 80 + 12],
                              "block": line, "line": line, "confidence": 95})
        spatial = {"page_number": 1, "source": "native", "page_width": 600, "page_height": 800,
                   **normalized_words(words, 600, 800, source="native")}
        page = PageRecord("ambiguous-layout.pdf", 1,
                          "Invoice number X-2\nInvoice date: 2026-01-01\nGross total EUR 10.00",
                          "native_text", {"text_layout": spatial})
        doc = DocumentRecord(page.filename, [page], DocumentClassification("invoice", 0.9, False))
        candidate = extract_payables(doc).candidates[0]
        self.assertEqual(candidate.supplier_name.status, "ambiguous")
        self.assertIsNone(candidate.supplier_name.value)

    def test_labelled_column_table_extracts_rows_and_preserves_line_tax_scope(self):
        text = """Invoice Number: TEST-744
Invoice Date: 2026-05-01
Supplier: Example Services Ltd
Currency: EUR
Description Quantity Net Unit Price VAT Amount EUR
Consulting Service 1.00 10.00 10% 10.00
Support Service 2.00 10.00 10% 20.00
Subtotal EUR 30.00
Total VAT 10% EUR 3.00
Invoice Total EUR 33.00"""
        candidate = extract_payables(self.document(text)).candidates[0]
        self.assertEqual(len(candidate.lines), 2)
        self.assertEqual([line.description.value for line in candidate.lines],
                         ["Consulting Service", "Support Service"])
        self.assertEqual([line.quantity.value for line in candidate.lines],
                         [Decimal("1.00"), Decimal("2.00")])
        self.assertEqual([line.unit_price.value for line in candidate.lines],
                         [Decimal("10.00"), Decimal("10.00")])
        self.assertTrue(all(line.price_basis.value == "net" for line in candidate.lines))
        self.assertTrue(all(len(line.taxes) == 1 and line.taxes[0].scope == "line"
                            for line in candidate.lines))
        payable, diag = generate_autodraft(candidate, MasterData())
        self.assertEqual(erp_book(payable)["will_book_gross"], 33.0)
        self.assertEqual(diag["reconciliation"]["status"], "passed")

    def test_column_rows_are_invariant_to_row_order_and_spacing(self):
        rows = ["Consulting Service 1.00 10.00 10% 10.00",
                "Support Service 2.00 10.00 10% 20.00"]
        def extract_with(order):
            text = "\n".join([
                "Invoice Number: TEST-889", "Invoice Date: 2026-05-01",
                "Supplier: Example Services Ltd", "Currency: EUR",
                "Description Quantity Net Unit Price VAT Amount EUR", *order,
                "Subtotal EUR 30.00", "Total VAT 10% EUR 3.00", "Invoice Total EUR 33.00",
            ])
            return extract_payables(self.document(text)).candidates[0]
        normal = extract_with(rows)
        changed = extract_with(["   " + rows[1] + "   ", rows[0]])
        normalize = lambda c: sorted((str(x.description.value), str(x.quantity.value),
                                      str(x.unit_price.value), str(x.printed_total.value)) for x in c.lines)
        self.assertEqual(normalize(normal), normalize(changed))
        for candidate in (normal, changed):
            payable, diag = generate_autodraft(candidate, MasterData())
            self.assertEqual(erp_book(payable)["will_book_gross"], 33.0)
            self.assertEqual(diag["reconciliation"]["status"], "passed")

    def test_label_neighbor_fields_and_varied_synthetic_invoices(self):
        cases = [
            ("EUR", "Example Consulting Ltd", "AC-410", "10.00", "20.00", "10", "33.00"),
            ("GBP", "Different Supplier PLC", "BILL-88", "100.00", "50.00", "5", "157.50"),
        ]
        for currency, supplier, invoice, first, second, rate, expected in cases:
            with self.subTest(currency=currency, supplier=supplier, invoice=invoice):
                subtotal = Decimal(first) + Decimal(second)
                tax = (subtotal * Decimal(rate) / Decimal("100")).quantize(Decimal("0.01"))
                gross = subtotal + tax
                text = f"""Invoice Number Parnu mnt 117
Rua do Ouro 175 {invoice} Parnu mnt 26
Invoice Date {supplier}
TAX INVOICE 01 May 2026 Attention Finance
Due Date: 31 May 2026
Supplier Name: {supplier}
Currency: {currency}
Description Quantity Net Unit Price VAT Amount {currency}
Consulting Service 1.00 {first} {rate}% {first}
Support Service 1.00 {second} {rate}% {second}
Subtotal {currency} {subtotal:.2f}
Total VAT {rate}% {currency} {tax:.2f}
Invoice Total {currency} {gross:.2f}"""
                candidate = extract_payables(self.document(text)).candidates[0]
                self.assertEqual(candidate.invoice_number.value, invoice)
                self.assertEqual(candidate.invoice_date.value.isoformat(), "2026-05-01")
                self.assertEqual(candidate.supplier_name.value, supplier)
                self.assertEqual(candidate.currency.value, currency)
                self.assertEqual(len(candidate.lines), 2)
                payable, diagnostics = generate_autodraft(candidate, MasterData())
                self.assertEqual(erp_book(payable)["will_book_gross"], float(expected))
                self.assertEqual(diagnostics["reconciliation"]["status"], "passed")

    def test_amount_only_table_does_not_invent_quantity_or_unit_price(self):
        text = """Invoice Number: TEST-991
Invoice Date: 2026-05-01
Supplier: Example Ltd
Description Amount EUR
One-time platform access 125.00
Invoice Total EUR 125.00"""
        candidate = extract_payables(self.document(text)).candidates[0]
        self.assertEqual(len(candidate.lines), 1)
        line = candidate.lines[0]
        self.assertEqual(line.printed_total.value, Decimal("125.00"))
        self.assertEqual(line.quantity.status, "missing")
        self.assertEqual(line.unit_price.status, "missing")

    def test_supplier_name_and_address_from_unique_bank_remittance_block(self):
        text = """Invoice number INV-BANK
Invoice date: 2026-03-01
Description Quantity Unit price excl VAT Total
1 service 1 EA 100.00 100.00 EUR
Grand total EUR 100.00
Meridian Logistics OÜ
High Street 4, Tartumaa 50703
Oakhaven Bank IBAN: EE682389839721862212"""
        candidate = extract_payables(self.document(text)).candidates[0]
        self.assertEqual(candidate.supplier_name.value, "Meridian Logistics OÜ")
        self.assertEqual(candidate.supplier_address.value, "High Street 4, Tartumaa 50703")
        self.assertEqual(candidate.supplier_name.status, "found")
        self.assertEqual(candidate.supplier_name.evidence[0].page_number, 1)

    def test_ambiguous_remittance_bank_block_does_not_guess_supplier(self):
        text = """Invoice number INV-BANK-AMBIG
Vendor Services Ltd
Purchasing Holdings Ltd
Main Street 4
Bank IBAN: XX123"""
        candidate = extract_payables(self.document(text)).candidates[0]
        self.assertEqual(candidate.supplier_name.status, "missing")

    def test_common_terminal_total_labels_are_recognized_without_generic_total_guessing(self):
        cases = [
            ("TOTAL A PAGAR EUR 535,79", Decimal("535.79")),
            ("Kogusumma (EUR) 254,52", Decimal("254.52")),
            ("Total incl. VAT EUR 1,683.98", Decimal("1683.98")),
            ("TOTAL (USD) 17,657.53", Decimal("17657.53")),
        ]
        for label, expected in cases:
            with self.subTest(label=label):
                candidate = extract_payables(self.document(
                    f"Invoice number INV-TOTAL\nInvoice date: 2026-03-01\n{label}"
                )).candidates[0]
                self.assertEqual(candidate.printed_gross_total.status, "found")
                self.assertEqual(candidate.printed_gross_total.value, expected)
                self.assertEqual(candidate.printed_gross_total.evidence[0].page_number, 1)

    def test_pre_discount_subtotal_is_not_conflicted_with_after_discount_net_amount(self):
        text = """Invoice number INV-DISCOUNT
Vahesumma EUR 538.60
Allahindlus 15% EUR 48.09
Summa ilma käibemaksuta EUR 490.51
VAT 24% EUR 117.72
Grand total EUR 608.23"""
        candidate = extract_payables(self.document(text)).candidates[0]
        self.assertEqual(candidate.printed_subtotal.status, "found")
        self.assertEqual(candidate.printed_subtotal.value, Decimal("538.60"))
        self.assertEqual(candidate.printed_subtotal.evidence[0].text, "Vahesumma EUR 538.60")
        self.assertEqual(candidate.printed_gross_total.value, Decimal("608.23"))

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
