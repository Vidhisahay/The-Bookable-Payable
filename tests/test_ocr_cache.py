from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ingestion.ocr_cache import OCRPageCache
from ingestion.pipeline import IngestionConfig, IngestionPipeline
from test_ingestion import FakeOCR, FakePage, FakeReader


class OCRCacheTests(unittest.TestCase):
    def make_key(self, **overrides):
        values = {
            "pdf_sha256": "a" * 64,
            "page_number": 1,
            "stage": "page",
            "language": "eng+est",
            "psm_modes": (6, 11, 3),
            "dpi": 160,
            "timeout_seconds": 12,
            "retries": 1,
            "minimum_chars": 24,
        }
        values.update(overrides)
        return OCRPageCache.key(**values)

    def test_key_invalidates_for_source_and_ocr_configuration(self):
        base = self.make_key()
        self.assertNotEqual(base, self.make_key(pdf_sha256="b" * 64))
        self.assertNotEqual(base, self.make_key(page_number=2))
        self.assertNotEqual(base, self.make_key(language="eng+deu"))
        self.assertNotEqual(base, self.make_key(psm_modes=(6,)))
        self.assertNotEqual(base, self.make_key(dpi=220))
        self.assertNotEqual(base, self.make_key(timeout_seconds=30))
        self.assertNotEqual(base, self.make_key(retries=0))
        self.assertNotEqual(base, self.make_key(engine_identity="other-tesseract-version"))

    def test_cache_round_trip_and_rejects_invalid_entries(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = OCRPageCache(folder)
            key = self.make_key()
            text = "Invoice number 12345 Total due EUR 45.00 Supplier Example"
            self.assertTrue(cache.store(key, text, {"ocr_page_segmentation_mode": 6}, minimum_chars=24))
            self.assertEqual(cache.load(key, minimum_chars=24), {
                "text": text, "metadata": {"ocr_page_segmentation_mode": 6},
            })
            other = self.make_key(page_number=9)
            (Path(folder) / f"{other}.json").write_text("{broken", encoding="utf-8")
            self.assertIsNone(cache.load(other, minimum_chars=24))
            self.assertFalse(cache.store(other, "", {}, minimum_chars=24))
            self.assertIsNone(cache.load(other, minimum_chars=24))

    def test_interrupted_atomic_write_leaves_no_valid_or_temporary_cache_entry(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = OCRPageCache(folder)
            key = self.make_key()
            with patch("ingestion.ocr_cache.os.replace", side_effect=OSError("simulated interruption")):
                self.assertFalse(cache.store(key, "Invoice total EUR 12.00 Supplier Example", {}, minimum_chars=24))
            self.assertIsNone(cache.load(key, minimum_chars=24))
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_pipeline_cache_hit_skips_ocr_and_preserves_page_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pdf_path = root / "sample.pdf"
            pdf_path.write_bytes(b"stable fixture pdf content")
            cache_path = root / "ocr-cache"
            config = IngestionConfig(ocr_page_segmentation_modes=(6,), ocr_cache_directory=cache_path)
            first_ocr = FakeOCR("Invoice number 12345 Total due EUR 45.00 Supplier Example")
            first = IngestionPipeline(config, FakeReader([FakePage("")]), first_ocr).process_file(pdf_path)
            self.assertEqual(first.pages[0].extraction_method, "ocr")
            self.assertEqual(first.pages[0].evidence["ocr_cache"]["status"], "stored")

            second_ocr = FakeOCR("should not be used")
            second = IngestionPipeline(config, FakeReader([FakePage("")]), second_ocr).process_file(pdf_path)
            self.assertEqual(second_ocr.calls, 0)
            self.assertEqual(second.pages[0].text, first.pages[0].text)
            self.assertEqual(second.pages[0].extraction_method, "ocr")
            self.assertEqual(second.pages[0].evidence["ocr_cache"]["status"], "hit")
            self.assertEqual(second.pages[0].page_number, 1)

            pdf_path.write_bytes(b"changed source pdf content")
            changed_ocr = FakeOCR("Changed source Invoice number 34567 Total due EUR 90.00")
            changed = IngestionPipeline(config, FakeReader([FakePage("")]), changed_ocr).process_file(pdf_path)
            self.assertEqual(changed_ocr.calls, 1)
            self.assertEqual(changed.pages[0].evidence["ocr_cache"]["status"], "stored")

    def test_bounded_timeout_retry_recovers_without_caching_timeout(self):
        class RecoveringOCR(FakeOCR):
            def recognize(self, *_args, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise subprocess.TimeoutExpired("tesseract", 1)
                return "Invoice number 12345 Total due EUR 45.00 Supplier Example"

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pdf_path = root / "retry.pdf"
            pdf_path.write_bytes(b"retry fixture")
            ocr = RecoveringOCR()
            config = IngestionConfig(ocr_page_segmentation_modes=(6,), ocr_retries=1,
                                     ocr_cache_directory=root / "cache")
            result = IngestionPipeline(config, FakeReader([FakePage("")]), ocr).process_file(pdf_path)
            page = result.pages[0]
            self.assertEqual(page.extraction_method, "ocr")
            self.assertEqual(ocr.calls, 2)
            self.assertEqual(page.evidence["ocr_retry_count"], 1)
            self.assertEqual([item["outcome"] for item in page.evidence["ocr_attempts"]], ["timeout", "success"])
            key = page.evidence["ocr_cache"]["key"]
            cached = OCRPageCache(root / "cache").load(key, minimum_chars=24)
            self.assertEqual(cached["text"], page.text)

    def test_timeout_is_not_cached_when_retry_budget_is_zero(self):
        class TimedOutOCR(FakeOCR):
            def recognize(self, *_args, **_kwargs):
                self.calls += 1
                raise subprocess.TimeoutExpired("tesseract", 1)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pdf_path = root / "timeout.pdf"
            pdf_path.write_bytes(b"timeout fixture")
            config = IngestionConfig(ocr_page_segmentation_modes=(6,), ocr_retries=0,
                                     ocr_cache_directory=root / "cache")
            result = IngestionPipeline(config, FakeReader([FakePage("")]), TimedOutOCR()).process_file(pdf_path)
            page = result.pages[0]
            self.assertEqual(page.extraction_method, "ocr_timeout")
            self.assertEqual(page.evidence["ocr_attempts"][0]["outcome"], "timeout")
            key = page.evidence["ocr_cache"]["key"]
            self.assertIsNone(OCRPageCache(root / "cache").load(key, minimum_chars=24))


if __name__ == "__main__":
    unittest.main()
