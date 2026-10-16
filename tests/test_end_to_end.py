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

    def test_explicit_german_plus_vat_wording_establishes_net_price(self):
        text = """Rechnung Nr.: INV-PLUS
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
        text = """Rechnung Nr.: INV-PLUS
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

        credit = """Kreeditarve arvele nr. 6265
Invoice date: 2025-03-16
Arve kokku -100.00 EUR
Description Quantity Unit price excl VAT Total
1 credit adjustment 1 EA -100.00 -100.00 EUR
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

    def test_ocr_timeout_is_explicit(self):
        class SlowOCR(FakeOCR):
            def recognize(self, *_args, **_kwargs):
                raise subprocess.TimeoutExpired("tesseract", 1)
        result = self.run_text([""], ocr=SlowOCR())
        page = result["diagnostics"]["pages"][0]
        self.assertEqual(page["extraction_method"], "ocr_timeout")
        self.assertIn("timed out", page["error"])
        self.assertEqual(page["evidence"]["ocr_modes_attempted"], [6])

    def test_json_encoding_is_valid_utf8_with_multilingual_ocr(self):
        result = {"text": "dotless ı; Estonian õ ä ö ü; Thai ใบวางบิล"}
        output = io.BytesIO()
        write_json_line(result, output)
        self.assertEqual(json.loads(output.getvalue().decode("utf-8")), result)


if __name__ == "__main__":
    unittest.main()
