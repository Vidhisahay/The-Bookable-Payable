from __future__ import annotations

import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from ingestion.autodraft_pipeline import AutodraftPipeline
from ingestion.autodraft import MasterData
from ingestion.cli import write_json_line
from ingestion.pdf_reader import TesseractOCR
from ingestion.pipeline import IngestionConfig, IngestionPipeline
from test_ingestion import FakeOCR, FakePage, FakeReader


TOP_KEYS = {"invoice_number", "invoice_date", "due_date", "invoice_type", "currency", "supplier", "buyer",
            "payment_term_id", "po_number", "po_id", "gross_total", "subtotal", "total_tax_amount",
            "discount_amount", "freight_charges", "insurance_charges", "extra_charges", "excise_duties", "taxes", "line_items"}
LINE_KEYS = {"description", "item_type", "uom", "quantity", "unit_price", "total", "discount", "discount_percentage", "tax_rate", "tax_amount", "taxes"}

INVOICE = """TAX INVOICE
Invoice number: INV-700
Invoice date: 2026-01-01
Due date: 2026-01-11
Supplier Name: Phocus Direct Communication GmbH
Supplier Address: Lina-Ammon-Strasse 19b, 90471 Nurnberg, DE
VAT ID: DE209177122
Bill To
Bolt Holdings OU
Vana-Louna 15, 10134 Tallinn, Estonia
Terms of payment: Net 10
Description Quantity Unit price excl VAT Total
1 consulting service 1 EA 100.00 100.00 EUR
Subtotal EUR 100.00
VAT 19% 19.00 EUR
Grand total EUR 119.00
"""


class EndToEndTests(unittest.TestCase):
    def run_text(self, pages: list[str], *, ocr=None, timeout=5, masters=None):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.pdf"
            path.write_bytes(b"fake pdf")
            reader = FakeReader([FakePage(text) for text in pages])
            pipeline = AutodraftPipeline(ingestion=IngestionPipeline(
                IngestionConfig(ocr_timeout_seconds=timeout), reader, ocr or FakeOCR()), masters=masters)
            return pipeline.process_file(path)

    def test_native_invoice_full_mapping_and_exact_schema(self):
        result = self.run_text([INVOICE])
        self.assertEqual(result["diagnostics"]["pages"][0]["extraction_method"], "native_text")
        self.assertEqual(result["diagnostics"]["classification"]["category"], "invoice")
        self.assertEqual(len(result["accounting"]["payables"]), 1)
        payable = result["accounting"]["payables"][0]
        self.assertEqual(set(payable), TOP_KEYS)
        self.assertEqual(set(payable["supplier"]), {"name", "supplier_id", "address", "vat_id"})
        self.assertEqual(set(payable["buyer"]), {"company_code", "business_unit_code", "location_code"})
        self.assertEqual(set(payable["line_items"][0]), LINE_KEYS)
        self.assertEqual(payable["supplier"]["supplier_id"], "2845695")
        self.assertEqual(payable["buyer"]["business_unit_code"], "EE004")
        self.assertEqual(payable["payment_term_id"], "Net_10")
        self.assertEqual(result["diagnostics"]["extraction"]["candidates"][0]["reconciliation"]["status"], "passed")

    def test_activity_table_and_separate_tax_summary_reconcile_without_duplicate_charges(self):
        text = """TAX INVOICE
Supplier Name: Oakhaven Supply Ltd
BILL TO
DATE PLEASE PAY DUE DATE
Nairobi Kenya
03/06/2025 KES 70,654.30 03/07/2025
ACTIVITY TAX atv RATE AMOUNT
Handling Out (per CBM) Std 0.2744 500.00 137.20
Transport Std 1 3,000.00 3,000.00
Transport Std 1 1,500.00 1,500.00
Storage charges Std 40.53574 30.00 1,216.07
Storage charges Std 22.30872 30.00 669.26
Storage charges Sid 1,646.2116 30.00 49,386.35
Stock Management fee Std 1 5,000.00 5,000.00
SUBTOTAL 60,908.88
TAX SUMMARY
RATE TAX NET
VAT @ 16% 9,745.42 60,908.88
TOTAL DUE KES 70,654.30
United States Dollar (USD): Account Number 8672480023
Euro (EUR): Account Number 8672480039
"""
        result = self.run_text([text])
        self.assertEqual(len(result["accounting"]["payables"]), 1)
        payable = result["accounting"]["payables"][0]
        self.assertEqual(payable["currency"], "KES")
        self.assertEqual(payable["supplier"]["name"], "Oakhaven Supply Ltd")
        self.assertEqual(payable["supplier"]["supplier_id"], "")
        self.assertEqual(payable["invoice_number"], "")
        self.assertEqual(payable["invoice_date"], "")
        self.assertEqual(payable["gross_total"], "70654.30")
        self.assertEqual(payable["subtotal"], "60908.88")
        self.assertEqual(payable["total_tax_amount"], "9745.42")
        self.assertEqual(len(payable["line_items"]), 7)
        self.assertEqual(payable["taxes"][0]["tax_amount"], "9745.42")
        self.assertEqual(payable["taxes"][0]["tax_rate"], "16")
        self.assertEqual(payable["freight_charges"], "")
        self.assertEqual(payable["extra_charges"], "")
        self.assertEqual(result["diagnostics"]["extraction"]["candidates"][0]["reconciliation"]["erp_gross"], "70654.30")

    def test_page_ocr_cache_replay_keeps_accounting_payload_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "fixture.pdf"
            path.write_bytes(b"stable scanned fixture")
            config = IngestionConfig(ocr_timeout_seconds=5, ocr_page_segmentation_modes=(6,),
                                     ocr_cache_directory=root / "ocr-cache")
            generated = []
            for ocr in (FakeOCR(INVOICE), FakeOCR("cached text must be used")):
                pipeline = AutodraftPipeline(
                    ingestion=IngestionPipeline(config, FakeReader([FakePage("")]), ocr),
                    masters=MasterData.load("master_data"),
                )
                generated.append(pipeline.process_file(path))
            self.assertEqual(generated[0]["accounting"], generated[1]["accounting"])
            self.assertEqual(generated[0]["accounting"]["payables"][0]["gross_total"], "119.00")
            self.assertEqual(generated[1]["diagnostics"]["pages"][0]["evidence"]["ocr_cache"]["status"], "hit")
            self.assertEqual(generated[1]["diagnostics"]["extraction"]["candidates"][0]["reconciliation"]["status"], "passed")

    def test_native_column_table_uses_bounded_ocr_and_recovers_oracle_payable(self):
        native = """Oracle America, Inc.
500 Oracle Parkway
Redwood Shores, CA 94065
United States
INVOICE
Federal Tax ID: 94-2805249
INVOICE NUMBER
44273107
INVOICE DATE
18-Jun-2018
YOUR P.O. NUMBER
1000021496
PAYMENT TERMS
30 NET
DUE DATE
18-Jul-2018
BILL TO:
AmeriHealth Caritas Services
ITEM
NO.
DESCRIPTION
QTY/
NO. OF USERS
TAX
EXTENDED
AMOUNT
1
Managed Cloud Services - Computer and Administration Services -
Managed Cloud Services - Additional Services 18-JUL-2018 :
17-OCT-2018
11
N
91,580.50
SUBTOTAL
TAX
TOTAL(USD)
91,580.50
0.00
91,580.50
"""
        table_ocr = """Oracle America, Inc. INVOICE
500 Oracle Parkway Federal Tax ID: 94-2805249
Redwood Shores, CA 94065 INVOICE NUMBER 44273107
United States 18-Jun-2018
ITEM QTY/ EXTENDED
1 Managed Cloud Services - Computer and Administration Services - Managed Cloud Services - Additional Services 18-JUL-2018 : 17-OCT-2018 1 N 91,580.50
SUBTOTAL TOTAL(USD)
91,580.50 91,580.50
"""
        result = self.run_text([native], ocr=FakeOCR(table_ocr))
        payable = result["accounting"]["payables"][0]
        candidate = result["diagnostics"]["extraction"]["candidates"][0]
        self.assertEqual(payable["invoice_number"], "44273107")
        self.assertEqual(payable["invoice_date"], "2018-06-18")
        self.assertEqual(payable["due_date"], "2018-07-18")
        self.assertEqual(payable["currency"], "USD")
        self.assertEqual(payable["supplier"]["name"], "Oracle America, Inc.")
        self.assertEqual(payable["supplier"]["address"], "500 Oracle Parkway, Redwood Shores, CA 94065, United States")
        self.assertEqual(payable["supplier"]["vat_id"], "")
        self.assertEqual(payable["po_number"], "1000021496")
        self.assertEqual(payable["payment_term_id"], "Net_30")
        self.assertEqual(payable["subtotal"], "91580.50")
        self.assertEqual(payable["total_tax_amount"], "0.00")
        self.assertEqual(payable["line_items"][0]["quantity"], "1")
        self.assertIn("Managed Cloud Services - Additional Services", payable["line_items"][0]["description"])
        self.assertIn("17-OCT-2018", payable["line_items"][0]["description"])
        self.assertEqual(payable["line_items"][0]["unit_price"], "91580.50")
        self.assertEqual(payable["line_items"][0]["total"], "91580.50")
        self.assertEqual(candidate["reconciliation"]["erp_gross"], "91580.50")
        self.assertEqual(candidate["bookability"]["status"], "passed")
        self.assertEqual(result["diagnostics"]["pages"][0]["evidence"]["table_ocr"]["status"], "found")

    def test_product_reference_credit_table_keeps_line_vat_and_supplier_vat_scope(self):
        text = """Credit Note No.: 5900366703
Date: 15/04/2026
Customer: Northwind Services UK Limited
VAT Registration No.: G8726874417
Currency in: GBP
Product Reference No. Quantity Unit VAT Unit Price Net Amount
4582314 952-000091 6 20% 457.24 2,743.44
Logitech Tap Scheduler Touch Controller
4541892 939-001950 2 20% 743.35 1,486.70
Logitech Tap Touch Controller Cat5e Kit
Net Amount GBP 4,230.14
VAT 20 % 846.03
Total Amount GBP 5,076.17
Terms of Payment: Within 30 days of invoice date
Blackpine Consulting Limited Registered in England
Parmu mnt 50
VAT: GB447324473 14 5HP Account Number: 123456
"""
        result = self.run_text([text])
        payable = result["accounting"]["payables"][0]
        candidate = result["diagnostics"]["extraction"]["candidates"][0]
        self.assertEqual(payable["invoice_type"], "CREDIT_MEMO")
        self.assertEqual(payable["invoice_number"], "5900366703")
        self.assertEqual(payable["currency"], "GBP")
        self.assertEqual(payable["supplier"]["name"], "Blackpine Consulting Limited")
        self.assertEqual(payable["supplier"]["vat_id"], "GB447324473")
        self.assertEqual(payable["gross_total"], "5076.17")
        self.assertEqual(payable["subtotal"], "4230.14")
        self.assertEqual(payable["total_tax_amount"], "846.03")
        self.assertEqual(payable["taxes"], [])
        self.assertEqual(len(payable["line_items"]), 2)
        self.assertEqual(payable["line_items"][0]["quantity"], "6")
        self.assertEqual(payable["line_items"][0]["unit_price"], "457.24")
        self.assertEqual(payable["line_items"][0]["total"], "2743.44")
        self.assertEqual(payable["line_items"][0]["taxes"][0]["tax_rate"], "20")
        self.assertEqual(payable["line_items"][1]["quantity"], "2")
        self.assertEqual(candidate["reconciliation"]["erp_gross"], "5076.17")
        self.assertEqual(candidate["reconciliation"]["status"], "passed")
        self.assertEqual(candidate["bookability"]["status"], "passed")

    def test_unambiguous_invoice_date_resolves_day_first_due_date_on_same_page(self):
        text = INVOICE.replace("2026-01-01", "16.02.2026").replace("2026-01-11", "08.03.2026")
        result = self.run_text([text])
        fields = result["diagnostics"]["extraction"]["candidates"][0]["fields"]
        self.assertEqual(fields["invoice_date"]["value"], "2026-02-16")
        self.assertEqual(fields["due_date"]["value"], "2026-03-08")

    def test_explicit_german_plus_vat_wording_establishes_net_price(self):
        text = """Supplier Name: Example Supplier
Rechnung Nr.: INV-PLUS
Rechnungsdatum: 02.02.2026
Pos Beschreibung Einzelpreis Menge Gesamtpreis
1 Projektmanagement 73,00 EUR 4 Std. 292,00 EUR
zzgl. 0 % MwSt 0,00 EUR
Reverse Charge
Endbetrag 292,00 EUR"""
        result = self.run_text([text])
        candidate = result["diagnostics"]["extraction"]["candidates"][0]
        self.assertEqual(candidate["fields"]["lines"][0]["price_basis"]["value"], "net")
        self.assertEqual(candidate["reconciliation"]["status"], "passed")
        self.assertEqual(candidate["reconciliation"]["erp_gross"], "292.00")

    def test_reverse_charge_context_is_not_emitted_as_an_untyped_tax(self):
        text = """Supplier Name: Example Supplier
Rechnung Nr.: INV-PLUS
Rechnungsdatum: 02.02.2026
Pos Beschreibung Einzelpreis Menge Gesamtpreis
1 Projektmanagement 73,00 EUR 4 Std. 292,00 EUR
zzgl. 0 % MwSt 0,00 EUR
Reverse Charge
Endbetrag 292,00 EUR"""
        result = self.run_text([text])
        taxes = result["accounting"]["payables"][0]["taxes"]
        self.assertEqual(len(taxes), 1)
        self.assertEqual(taxes[0]["tax_type"], "VAT")
        self.assertEqual(taxes[0]["tax_name"], "MwSt")

    def test_estonian_total_labels_extract_printed_gross_and_subtotal(self):
        text = """Supplier Name: Example Supplier
Arve #9972-907
Kuupäev 18-03-2026
Summa ilma käibemaksuta €490,51
Käibemaks 24% €117,72
Summa koos käibemaksuga €608,23"""
        result = self.run_text([text])
        candidate = result["diagnostics"]["extraction"]["candidates"][0]
        self.assertEqual(candidate["fields"]["printed_subtotal"]["value"], "490.51")
        self.assertEqual(candidate["fields"]["printed_gross_total"]["value"], "608.23")

    def test_estonian_billed_rows_ignore_kit_breakdown_and_keep_transport_as_line(self):
        text = """Supplier Name: Example Supplier
Arve #9972-907
Kuupäev 18-03-2026
Hind Kokku käibemaksuta
9x Led Bar FOS Luminus BAR 24 tundi €6,00 €54,00
1x Aktiivkõlar Veritas 8000 komplekt 24 tundi €103,60 €103,60
2x Aktiivkõlar NOVA Veritas 12 24 tundi €20,00
2x Aktiivsubwoofer Nova Veritas 18S 24 tundi €28,00
1x Linnasisene transport Fikseeritud €50,00 €50,00
2x Tehnik 3 tundi €84,00 €168,00
Vahesumma €538,60
Allahindlus 15 % €48,09
Summa ilma käibemaksuta €490,51
Käibemaks 24% €117,72
Summa koos käibemaksuga €608,23"""
        result = self.run_text([text])
        candidate = result["diagnostics"]["extraction"]["candidates"][0]
        lines = candidate["fields"]["lines"]
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[1]["description"]["value"], "Aktiivkõlar Veritas 8000 komplekt")
        self.assertEqual(lines[1]["quantity"]["value"], "1")
        self.assertEqual(lines[1]["unit_price"]["value"], "103.60")
        self.assertEqual(lines[1]["printed_total"]["value"], "103.60")
        self.assertEqual(lines[2]["description"]["value"], "Linnasisene transport")
        self.assertEqual(lines[2]["quantity"]["value"], "1")
        self.assertEqual(lines[2]["unit_price"]["value"], "50.00")
        technician = lines[3]
        self.assertEqual(technician["quantity"]["value"], "2")
        self.assertEqual(technician["unit_price"]["value"], "84.00")
        self.assertEqual(technician["printed_total"]["value"], "168.00")
        self.assertFalse(any(charge["kind"] == "freight" for charge in candidate["fields"]["charges"]))
        discount = next(charge for charge in candidate["fields"]["charges"] if charge["kind"] == "discount")
        self.assertEqual(discount["amount"]["value"], "48.09")

    def test_ocr_dependent_invoice_retains_ocr_provenance(self):
        result = self.run_text([""], ocr=FakeOCR(INVOICE))
        self.assertEqual(result["diagnostics"]["pages"][0]["extraction_method"], "ocr")
        self.assertTrue(result["diagnostics"]["pages"][0]["evidence"]["ocr_attempted"])
        self.assertEqual(len(result["accounting"]["payables"]), 1)

    def test_reminder_is_declined_without_new_payable(self):
        result = self.run_text(["Payment reminder. Mahnung. Invoice 991 overdue. Please pay now."])
        self.assertEqual(result["diagnostics"]["classification"]["category"], "payment_reminder")
        self.assertEqual(result["accounting"]["payables"], [])
        self.assertEqual(result["accounting"]["declined"][0]["doc_type"], "payment_reminder")

    def test_multiple_invoice_sections_are_preserved_as_two_candidates(self):
        p1 = INVOICE.replace("INV-700", "INV-701").replace("119.00", "119.00")
        p2 = INVOICE.replace("INV-700", "INV-702").replace("119.00", "119.00")
        result = self.run_text([p1, p2])
        self.assertEqual(len(result["accounting"]["payables"]), 2)
        self.assertEqual([p["invoice_number"] for p in result["accounting"]["payables"]], ["INV-701", "INV-702"])

    def test_explicit_attachment_pages_do_not_contaminate_invoice_totals(self):
        second = INVOICE.replace("INV-700", "INV-702")
        result = self.run_text([INVOICE, "Shipment Cartage Advice Cargo PG 1 OF 2", second])
        self.assertEqual(len(result["accounting"]["payables"]), 2)
        scope = result["diagnostics"]["extraction"]["page_scope"]
        self.assertFalse(scope[1]["included"])
        self.assertEqual(result["accounting"]["payables"][0]["gross_total"], "119.00")

    def test_cargo_customs_attachment_is_scoped_out_of_usd_invoice(self):
        cargo = "CARGO PG 1 OF 2 TYPE: IN-NON-PAYMENT Customs declaration TOTAL GST AMT: S$ 722.76"
        result = self.run_text([INVOICE, cargo])
        self.assertEqual(len(result["accounting"]["payables"]), 1)
        self.assertEqual(result["accounting"]["payables"][0]["gross_total"], "119.00")
        scope = result["diagnostics"]["extraction"]["page_scope"]
        self.assertTrue(scope[0]["included"])
        self.assertFalse(scope[1]["included"])

    def test_attachment_majority_does_not_suppress_scoped_invoice(self):
        cargo = "CARGO PG 1 OF 2 TYPE: IN-NON-PAYMENT Customs declaration TOTAL GST AMT: S$ 722.76"
        texts = [INVOICE,
                 "Shipment Cartage Advice", cargo, "CARGO PG 2 OF 2 Customs declaration GST (S$) 722.76"]
        result = self.run_text(texts)
        self.assertEqual(result["diagnostics"]["classification"]["category"], "supporting_document")
        self.assertEqual(len(result["accounting"]["payables"]), 1)
        self.assertEqual(result["accounting"]["declined"], [])
        self.assertEqual(result["accounting"]["payables"][0]["gross_total"], "119.00")
        self.assertEqual(result["diagnostics"]["extraction"]["candidates"][0]["fields"]["pages"], [1])

    def test_uncertain_document_is_unresolved_not_declined(self):
        result = self.run_text([""])
        self.assertEqual(result["accounting"]["payables"], [])
        self.assertEqual(result["accounting"]["declined"], [])
        self.assertTrue(result["diagnostics"]["extraction"]["unresolved_reasons"])

    def test_supporting_majority_with_unreadable_page_is_not_declined(self):
        class FailOCR(FakeOCR):
            def recognize(self, *_args, **_kwargs):
                raise RuntimeError("fixture OCR failure")
        result = self.run_text(["Customs Detailed Invoice", ""], ocr=FailOCR())
        self.assertEqual(result["accounting"]["payables"], [])
        self.assertEqual(result["accounting"]["declined"], [])
        self.assertTrue(result["diagnostics"]["extraction"]["unresolved_reasons"])

    def test_repeated_invoice_identifier_on_supporting_pages_is_one_payable(self):
        detail = "Page 1 of 1\nInvoice Number: INV-700\nCustoms Detailed Invoice\nCustomer PO 12345\nGST S$ 722.76"
        result = self.run_text([INVOICE, "Page 2 of 2\nInvoice number INV-700", detail])
        self.assertEqual(len(result["accounting"]["payables"]), 1)
        self.assertEqual(result["accounting"]["payables"][0]["invoice_number"], "INV-700")
        self.assertEqual(result["accounting"]["payables"][0]["gross_total"], "119.00")
        self.assertFalse(result["diagnostics"]["classification"]["potential_multiple_documents"])
        self.assertFalse(result["diagnostics"]["extraction"]["page_scope"][2]["included"])

    def test_explicit_separator_starts_another_invoice_even_without_number(self):
        invoice = INVOICE.replace("Invoice number: INV-700\n", "")
        result = self.run_text([invoice, "Shipment Cartage Advice", invoice.replace("119.00", "119.00")])
        self.assertEqual(len(result["accounting"]["payables"]), 2)
        self.assertTrue(all(item["invoice_number"] == "" for item in result["accounting"]["payables"]))

    def test_missing_supplier_and_credit_reference_remain_unresolved_or_blank(self):
        unknown = INVOICE.replace("Phocus Direct Communication GmbH", "Unlisted Vendor").replace("Lina-Ammon-Strasse 19b, 90471 Nurnberg, DE", "Unknown address").replace("DE209177122", "XX0000000")
        result = self.run_text([unknown])
        self.assertEqual(result["accounting"]["payables"][0]["supplier"]["supplier_id"], "")

        credit = """Supplier Name: Example Supplier
Kreeditarve arvele nr. 6265
Invoice date: 2025-03-16
Arve kokku -100.00 EUR
Description Quantity Unit price excl VAT Total
1 credit service adjustment 1 EA -100.00 -100.00 EUR
"""
        result = self.run_text([credit])
        self.assertEqual(result["accounting"]["payables"][0]["invoice_type"], "CREDIT_MEMO")
        self.assertEqual(result["accounting"]["payables"][0]["invoice_number"], "")

    def test_ambiguous_supplier_match_stays_blank_in_integrated_output(self):
        masters = MasterData(suppliers=[
            {"supplier_id": "A", "name": "Phocus Direct Communication GmbH", "address": "Lina-Ammon-Strasse 19b, 90471 Nurnberg, DE", "vat_id": ""},
            {"supplier_id": "B", "name": "Phocus Direct Communication GmbH", "address": "Lina-Ammon-Strasse 19b, 90471 Nurnberg, DE", "vat_id": ""},
        ])
        text = INVOICE.replace("VAT ID: DE209177122", "")
        result = self.run_text([text], masters=masters)
        self.assertEqual(result["accounting"]["payables"][0]["supplier"]["supplier_id"], "")
        self.assertEqual(result["diagnostics"]["extraction"]["candidates"][0]["matches"]["supplier"]["status"], "ambiguous")

    def test_incomplete_evidence_is_unresolved_even_if_erp_runs(self):
        text = "TAX INVOICE Invoice number INV-9 Grand total EUR 5.00"
        result = self.run_text([text])
        candidate = result["diagnostics"]["extraction"]["candidates"][0]
        self.assertEqual(candidate["reconciliation"]["status"], "unresolved")

    def test_missing_supplier_identity_blocks_booking(self):
        text = """TAX INVOICE
Invoice number: INV-UNIDENTIFIED
Invoice date: 2026-01-01
Description Quantity Unit price excl VAT Total
1 service 1 EA 100.00 100.00 EUR
Grand total EUR 100.00"""
        result = self.run_text([text])
        candidate = result["diagnostics"]["extraction"]["candidates"][0]
        self.assertEqual(candidate["reconciliation"]["status"], "passed")
        self.assertEqual(candidate["reconciliation"]["erp_gross"], "100.00")
        self.assertEqual(candidate["bookability"]["status"], "unresolved")
        self.assertIn("Neither supplier name nor supplier VAT ID", candidate["bookability"]["reason"])
        self.assertEqual(result["accounting"]["payables"], [])

    def test_source_supported_supplier_vat_id_allows_missing_name_without_master_match(self):
        text = INVOICE.replace("Supplier Name: Phocus Direct Communication GmbH\n", "")
        text = text.replace("Supplier Address: Lina-Ammon-Strasse 19b, 90471 Nurnberg, DE\n", "")
        text = text.replace("DE209177122", "EE398766194")
        result = self.run_text([text])
        payable = result["accounting"]["payables"][0]
        candidate = result["diagnostics"]["extraction"]["candidates"][0]
        self.assertEqual(payable["supplier"]["name"], "")
        self.assertEqual(payable["supplier"]["vat_id"], "EE398766194")
        self.assertEqual(payable["supplier"]["supplier_id"], "")
        self.assertEqual(candidate["matches"]["supplier"]["status"], "unmatched")
        self.assertEqual(candidate["reconciliation"]["erp_gross"], "119.00")
        self.assertEqual(candidate["bookability"]["status"], "passed")

    def test_bank_remittance_identity_allows_unmatched_supplier_code_when_arithmetic_is_supported(self):
        text = """TAX INVOICE
Invoice number: INV-BANK-700
Invoice date: 2026-01-01
Description Quantity Unit price excl VAT Total
1 consulting service 1 EA 100.00 100.00 EUR
Grand total EUR 100.00
Meridian Logistics OÜ
High Street 4, Tartumaa 50703
Oakhaven Bank IBAN: EE682389839721862212"""
        result = self.run_text([text])
        payable = result["accounting"]["payables"][0]
        candidate = result["diagnostics"]["extraction"]["candidates"][0]
        self.assertEqual(payable["supplier"]["name"], "Meridian Logistics OÜ")
        self.assertEqual(payable["supplier"]["supplier_id"], "")
        self.assertEqual(candidate["matches"]["supplier"]["status"], "unmatched")
        self.assertEqual(candidate["reconciliation"]["status"], "passed")
        self.assertEqual(candidate["bookability"]["status"], "passed")

    def test_unknown_line_item_enum_is_not_emitted(self):
        text = """TAX INVOICE
Invoice number: INV-UNKNOWN-TYPE
Invoice date: 2026-01-01
Supplier Name: Example Supplier
Description Quantity Unit price excl VAT Total
1 special adjustment 1 EA 100.00 100.00 EUR
Grand total EUR 100.00"""
        result = self.run_text([text])
        candidate = result["diagnostics"]["extraction"]["candidates"][0]
        self.assertEqual(candidate["reconciliation"]["status"], "passed")
        self.assertEqual(candidate["bookability"]["status"], "unresolved")
        self.assertIn("item type", candidate["bookability"]["reason"])
        self.assertEqual(result["accounting"]["payables"], [])

    def test_ocr_timeout_is_explicit(self):
        class SlowOCR(FakeOCR):
            def recognize(self, *_args, **_kwargs):
                raise subprocess.TimeoutExpired("tesseract", 1)
        result = self.run_text([""], ocr=SlowOCR())
        page = result["diagnostics"]["pages"][0]
        self.assertEqual(page["extraction_method"], "ocr_timeout")
        self.assertIn("timed out", page["error"])
        self.assertEqual(page["evidence"]["ocr_modes_attempted"], [6])
        self.assertNotIn("ocr_attempt_errors", page["evidence"])

    def test_json_encoding_is_valid_utf8_with_multilingual_ocr(self):
        result = {"text": "dotless ı; Estonian õ ä ö ü; Thai ใบวางบิล"}
        output = io.BytesIO()
        write_json_line(result, output)
        self.assertEqual(json.loads(output.getvalue().decode("utf-8")), result)


if __name__ == "__main__":
    unittest.main()
