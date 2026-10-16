from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

try:
    import pymupdf
except ImportError:  # real-PDF integration tests are optional until dependencies are installed
    pymupdf = None

from ingestion.classifier import classify_pages
from ingestion.models import PageRecord
from ingestion.pdf_reader import TesseractOCR, is_usable_text
from ingestion.pipeline import IngestionConfig, IngestionPipeline, enumerate_pdfs


class FakePage:
    def __init__(self, text: str = "", fail_text: bool = False, fail_render: bool = False):
        self.text = text
        self.fail_text = fail_text
        self.fail_render = fail_render

    def get_text(self, _mode: str) -> str:
        if self.fail_text:
            raise ValueError("fixture text failure")
        return self.text


class FakePDF(list):
    def close(self) -> None:
        pass


class FakeReader:
    def __init__(self, pages: list[FakePage], fail_open: bool = False):
        self.pages = FakePDF(pages)
        self.fail_open = fail_open

    def open(self, _path: Path) -> FakePDF:
        if self.fail_open:
            raise ValueError("fixture open failure")
        return self.pages

    def render_png(self, page: FakePage) -> bytes:
        if page.fail_render:
            raise ValueError("fixture render failure")
        return b"fake-png"


class FakeOCR:
    def __init__(self, text: str = "", responses: dict[int, str] | None = None):
        self.text = text
        self.responses = responses or {}
        self.calls = 0

    def recognize(self, _image: bytes, *, language: str, timeout: int, psm: int = 3) -> str:
        self.calls += 1
        return self.responses.get(psm, self.text)


class IngestionTests(unittest.TestCase):
    def test_cli_json_output_round_trips_multilingual_unicode_as_utf8(self):
        from ingestion.cli import write_json_line

        value = {"text": "dotless ı; Estonian õ ä ö ü; Thai ใบวางบิล"}
        output = io.BytesIO()
        write_json_line(value, output)

        encoded = output.getvalue()
        self.assertTrue(encoded.endswith(b"\n"))
        self.assertEqual(json.loads(encoded.decode("utf-8")), value)

    def process(self, pages: list[FakePage], ocr: FakeOCR | None = None):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "anything.pdf"
            path.write_bytes(b"fixture")
            pipeline = IngestionPipeline(IngestionConfig(), FakeReader(pages), ocr or FakeOCR())
            return pipeline.process_file(path)

    def test_native_text_skips_ocr_and_classifies_invoice(self):
        text = "TAX INVOICE Invoice number 12345 Total due EUR 45.00 Supplier Example"
        ocr = FakeOCR("must not be used")
        result = self.process([FakePage(text)], ocr)
        self.assertEqual(result.pages[0].extraction_method, "native_text")
        self.assertEqual(result.pages[0].text, text)
        self.assertEqual(ocr.calls, 0)
        self.assertEqual(result.classification.category, "invoice")

    def test_sparse_scanned_page_uses_ocr(self):
        ocr = FakeOCR("Credit note Amount credited 10.00")
        result = self.process([FakePage("")], ocr)
        self.assertEqual(result.pages[0].extraction_method, "ocr")
        self.assertEqual(result.pages[0].page_number, 1)
        self.assertEqual(result.classification.category, "credit_memo")

    def test_sparse_ocr_retries_page_segmentation_and_keeps_best_result(self):
        ocr = FakeOCR(responses={6: "", 11: "", 3: "No boxes Net Weight Gross Weight UOM Length Width Height Volume"})
        result = self.process([FakePage("")], ocr)
        self.assertEqual(result.pages[0].extraction_method, "ocr")
        self.assertEqual(result.pages[0].evidence["ocr_modes_attempted"], [6, 11, 3])
        self.assertEqual(result.pages[0].evidence["ocr_page_segmentation_mode"], 3)
        self.assertEqual(ocr.calls, 3)

    def test_empty_ocr_is_preserved_and_uncertain(self):
        result = self.process([FakePage("")], FakeOCR(""))
        self.assertEqual(result.pages[0].extraction_method, "ocr_empty")
        self.assertTrue(result.classification.uncertain)

    def test_multi_page_and_mixed_document_are_page_level(self):
        result = self.process([
            FakePage("Invoice number 20 Tax invoice Total due 15.00"),
            FakePage("Delivery note Goods dispatched received by customer"),
        ])
        self.assertEqual([page.page_number for page in result.pages], [1, 2])
        self.assertEqual(result.classification.category, "invoice")
        self.assertTrue(result.classification.mixed_document)
        self.assertTrue(result.classification.uncertain)

    def test_repeated_pages_with_same_invoice_id_are_not_marked_as_multiple(self):
        pages = [
            FakePage("Tax invoice Invoice number INV-100 Total due 15.00"),
            FakePage("Tax invoice Invoice number INV-100 Continued page 2"),
        ]
        result = self.process(pages)
        self.assertFalse(result.classification.potential_multiple_documents)

        pages = [
            FakePage("Tax invoice Invoice number INV-100 Total due 15.00"),
            FakePage("Tax invoice Invoice number INV-200 Total due 20.00"),
        ]
        result = self.process(pages)
        self.assertTrue(result.classification.potential_multiple_documents)
        self.assertEqual(result.classification.possible_document_starts, [2])

    def test_native_extraction_failure_is_recorded_without_losing_other_pages(self):
        result = self.process([
            FakePage("", fail_text=True),
            FakePage("Invoice number 20 Tax invoice Total due 15.00"),
        ])
        self.assertEqual(result.pages[0].extraction_method, "ocr_empty")
        self.assertIn("fixture text failure", result.pages[0].error or "")
        self.assertEqual(result.pages[1].extraction_method, "native_text")

    def test_native_failure_falls_back_to_ocr_and_ocr_failure_is_recorded(self):
        result = self.process([FakePage("", fail_text=True)], FakeOCR("Invoice number 7 Total due 4.00"))
        self.assertEqual(result.pages[0].extraction_method, "ocr")
        self.assertIn("Native text extraction failed", result.pages[0].error or "")

        class BrokenOCR(FakeOCR):
            def recognize(self, *_args, **_kwargs):
                raise RuntimeError("fixture OCR failure")

        failed = self.process([FakePage("")], BrokenOCR())
        self.assertEqual(failed.pages[0].extraction_method, "ocr_failed")
        self.assertIn("fixture OCR failure", failed.pages[0].error or "")

    def test_open_failure_is_document_level_and_uncertain(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.pdf"
            path.write_bytes(b"bad")
            result = IngestionPipeline(reader=FakeReader([], fail_open=True)).process_file(path)
        self.assertEqual(result.pages, [])
        self.assertTrue(result.classification.uncertain)
        self.assertIn("PDF open failed", result.error or "")

    def test_enumeration_is_recursive_and_deterministic(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            (base / "b.PDF").write_bytes(b"")
            (base / "nested").mkdir()
            (base / "nested" / "a.pdf").write_bytes(b"")
            (base / "ignore.txt").write_text("", encoding="utf-8")
            self.assertEqual([p.name for p in enumerate_pdfs(base)], ["b.PDF", "a.pdf"])

    def test_filename_is_not_classification_evidence(self):
        self.assertEqual(classify_pages([]).category, "uncertain")
        self.assertFalse(is_usable_text(""))
        self.assertTrue(is_usable_text("Invoice number 1 Total due 12.00"))

    def test_non_invoice_categories_use_text_evidence(self):
        examples = {
            "delivery_note": "Delivery note Goods dispatched",
            "statement": "Statement of account Balance brought forward",
            "purchase_order": "Purchase order Order confirmation",
            "quotation": "Quotation Valid until 30 June",
            "contract": "Master services agreement Contract agreement",
            "timesheet": "Timesheet Hours worked 8",
            "supporting_document": "Service report Certificate of completion",
            "invoice": "Arve number 1234 Maksetähtaeg 10 days Käibemaks",
            "payment_reminder": "Mahnung Bitte Zahlungsausgleich bis zum Fälligkeitstermin Rechnung 123",
        }
        for expected, text in examples.items():
            with self.subTest(category=expected):
                category = classify_pages([PageRecord("test.pdf", 1, text, "fixture")]).category
                self.assertEqual(category, expected)

    def test_tesseract_preflight_checks_each_requested_model_and_reports_path(self):
        from types import SimpleNamespace

        listing = 'List of available languages in "C:/tessdata/" (3):\neng\nosd\n'
        with patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=listing, stderr="")):
            ocr = TesseractOCR()
            with self.assertRaisesRegex(RuntimeError, "deu.*C:/tessdata"):
                ocr.prepare("eng+deu")

        listing = 'List of available languages in "C:/tessdata/" (3):\neng\nosd\ndeu\n'
        with patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=listing, stderr="")):
            TesseractOCR().prepare("eng+deu")

    def test_tesseract_output_uses_utf8_for_multilingual_text(self):
        from types import SimpleNamespace

        listing = 'List of available languages in "C:/tessdata/" (2):\neng\nosd\n'
        responses = [
            SimpleNamespace(returncode=0, stdout=listing, stderr=""),
            SimpleNamespace(returncode=0, stdout="Fatura de emissão", stderr=""),
        ]
        with patch("subprocess.run", side_effect=responses) as run:
            text = TesseractOCR().recognize(b"png", language="eng")
        self.assertEqual(text, "Fatura de emissão")
        self.assertTrue(all(call.kwargs.get("encoding") == "utf-8" for call in run.call_args_list))

    @unittest.skipIf(pymupdf is None, "PyMuPDF is required for real-PDF fixtures")
    def test_real_native_empty_scanned_and_multi_page_pdf_fixtures(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)

            native_path = root / "native.pdf"
            native = pymupdf.open()
            page = native.new_page()
            page.insert_text((72, 72), "TAX INVOICE Invoice number 12345 Total due EUR 45.00")
            native.save(native_path)
            native.close()
            native_result = IngestionPipeline(ocr=FakeOCR()).process_file(native_path)
            self.assertEqual(native_result.pages[0].extraction_method, "native_text")
            self.assertEqual(native_result.classification.category, "invoice")

            multi_path = root / "multi.pdf"
            multi = pymupdf.open()
            multi.new_page()
            multi.new_page()
            multi.save(multi_path)
            multi.close()
            empty_result = IngestionPipeline(ocr=FakeOCR()).process_file(multi_path)
            self.assertEqual(len(empty_result.pages), 2)
            self.assertTrue(all(page.extraction_method == "ocr_empty" for page in empty_result.pages))
            self.assertEqual([page.page_number for page in empty_result.pages], [1, 2])

            # Rasterize a text page and embed only its pixels to produce a scanned fixture.
            source = pymupdf.open()
            source_page = source.new_page()
            source_page.insert_text((72, 72), "Credit note Amount credited 10.00")
            png = source_page.get_pixmap().tobytes("png")
            source.close()
            scanned_path = root / "scanned.pdf"
            scanned = pymupdf.open()
            scanned_page = scanned.new_page()
            scanned_page.insert_image(scanned_page.rect, stream=png)
            scanned.save(scanned_path)
            scanned.close()
            scanned_result = IngestionPipeline(ocr=FakeOCR("Credit note Amount credited 10.00")).process_file(scanned_path)
            self.assertEqual(scanned_result.pages[0].extraction_method, "ocr")
            self.assertEqual(scanned_result.classification.category, "credit_memo")

            broken_path = root / "broken.pdf"
            broken_path.write_bytes(b"not a pdf")
            broken_result = IngestionPipeline(ocr=FakeOCR()).process_file(broken_path)
            self.assertIn("PDF open failed", broken_result.error or "")


if __name__ == "__main__":
    unittest.main()
