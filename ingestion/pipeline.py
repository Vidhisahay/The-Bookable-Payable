"""Deterministic directory ingestion with page-level failures and provenance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import subprocess
import time
from typing import Any

from .classifier import classify_pages
from .models import DocumentClassification, DocumentRecord, PageRecord
from .pdf_reader import PyMuPDFReader, TesseractOCR, is_usable_text


@dataclass(frozen=True)
class IngestionConfig:
    ocr_enabled: bool = True
    # Keep the per-page OCR workload within the documented timeout. Additional
    # installed language packs can be selected for targeted documents with the
    # CLI; loading every pack for every page caused widespread timeouts.
    ocr_language: str = "eng+deu"
    ocr_timeout_seconds: int = 120
    minimum_native_chars: int = 24
    # Preserve tabular row structure before trying general page segmentation.
    # On the assessment's flattened invoice tables, PSM 3 often returned enough
    # prose to pass the sparse-text check while losing quantities and amounts.
    ocr_page_segmentation_modes: tuple[int, ...] = (6, 11, 3)
    render_dpi: int = 160


class IngestionPipeline:
    def __init__(self, config: IngestionConfig | None = None, reader: Any | None = None, ocr: Any | None = None,
                 progress: Callable[[str], None] | None = None):
        self.config = config or IngestionConfig()
        self.reader = reader
        self.ocr = ocr
        self.progress = progress

    def _components(self) -> tuple[Any, Any | None]:
        reader = self.reader or PyMuPDFReader(dpi=self.config.render_dpi)
        ocr = self.ocr
        if self.config.ocr_enabled and ocr is None:
            ocr = TesseractOCR()
        return reader, ocr

    def process_file(self, path: Path, page_numbers: list[int] | None = None) -> DocumentRecord:
        filename = path.name
        pages: list[PageRecord] = []
        try:
            reader, ocr = self._components()
            pdf = reader.open(path)
        except Exception as exc:
            return DocumentRecord(filename, pages, _uncertain(), error=f"PDF open failed: {exc}")

        try:
            page_count = len(pdf)
            if page_numbers is None:
                page_indexes = list(range(page_count))
            else:
                if any(number < 1 or number > page_count for number in page_numbers):
                    return DocumentRecord(filename, [], _uncertain(), error=f"Requested page numbers must be between 1 and {page_count}.")
                page_indexes = sorted(set(number - 1 for number in page_numbers))
            for index in page_indexes:
                page_number = index + 1
                if self.progress:
                    self.progress(f"{filename}: page {page_number}/{page_count}")
                native_error: str | None = None
                try:
                    page = pdf[index]
                    try:
                        native_text = page.get_text("text") or ""
                    except Exception as exc:
                        native_text = ""
                        native_error = f"Native text extraction failed: {exc}"
                    if native_error is None and is_usable_text(native_text, self.config.minimum_native_chars):
                        pages.append(PageRecord(
                            filename, page_number, native_text, "native_text",
                            {"native_alphanumeric_chars": sum(c.isalnum() for c in native_text), "ocr_attempted": False},
                        ))
                        continue

                    evidence: dict[str, Any] = {
                        "native_alphanumeric_chars": sum(c.isalnum() for c in native_text),
                        "ocr_attempted": bool(self.config.ocr_enabled),
                    }
                    if native_error:
                        evidence["native_error"] = native_error
                    if not self.config.ocr_enabled:
                        pages.append(PageRecord(filename, page_number, native_text, "native_text_sparse", evidence))
                        continue

                    try:
                        ocr_attempts: list[int] = []
                        deadline = time.monotonic() + self.config.ocr_timeout_seconds
                        prepare = getattr(ocr, "prepare", None)
                        if prepare:
                            try:
                                prepare(self.config.ocr_language, timeout=max(1, min(15, int(deadline - time.monotonic()))))
                            except TypeError:
                                prepare(self.config.ocr_language)
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError(f"OCR page timeout after {self.config.ocr_timeout_seconds}s before rendering")
                        image = reader.render_png(page)
                        # PSM retries share one page budget; a difficult page cannot
                        # consume timeout_seconds once for every OCR mode.
                        text = ""
                        selected_psm = 3
                        best_alnum = -1
                        for psm in self.config.ocr_page_segmentation_modes:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise TimeoutError(f"OCR page timeout after {self.config.ocr_timeout_seconds}s; tried PSM modes {ocr_attempts}")
                            ocr_attempts.append(psm)
                            candidate = ocr.recognize(
                                image, language=self.config.ocr_language,
                                timeout=max(1, int(remaining)), psm=psm,
                            )
                            candidate_alnum = sum(c.isalnum() for c in candidate)
                            if candidate_alnum > best_alnum:
                                text, selected_psm, best_alnum = candidate, psm, candidate_alnum
                            if is_usable_text(candidate, self.config.minimum_native_chars):
                                text, selected_psm, best_alnum = candidate, psm, candidate_alnum
                                break
                        method = "ocr" if is_usable_text(text, self.config.minimum_native_chars) else ("ocr_sparse" if text.strip() else "ocr_empty")
                        pages.append(PageRecord(
                            filename, page_number, text, method,
                            {
                                **evidence,
                                "rendered_bytes": len(image),
                                "ocr_alphanumeric_chars": max(best_alnum, 0),
                                "ocr_modes_attempted": ocr_attempts,
                                "ocr_page_segmentation_mode": selected_psm,
                            },
                            error=native_error,
                        ))
                    except subprocess.TimeoutExpired as exc:
                        evidence["ocr_modes_attempted"] = ocr_attempts
                        pages.append(PageRecord(
                            filename, page_number, native_text, "ocr_timeout", evidence,
                            error="; ".join(part for part in (native_error, f"OCR page timed out after {self.config.ocr_timeout_seconds}s; PSM modes attempted: {ocr_attempts}") if part),
                        ))
                    except TimeoutError as exc:
                        evidence["ocr_modes_attempted"] = ocr_attempts
                        pages.append(PageRecord(
                            filename, page_number, native_text, "ocr_timeout", evidence,
                            error="; ".join(part for part in (native_error, str(exc)) if part),
                        ))
                    except Exception as exc:
                        if "ocr_attempts" in locals():
                            evidence["ocr_modes_attempted"] = ocr_attempts
                        pages.append(PageRecord(
                            filename, page_number, native_text, "ocr_failed", evidence,
                            error="; ".join(part for part in (native_error, f"OCR failed: {exc}") if part),
                        ))
                except Exception as exc:
                    pages.append(PageRecord(filename, page_number, "", "page_failed", {}, error=f"Page extraction failed: {exc}"))
        finally:
            close = getattr(pdf, "close", None)
            if close:
                close()

        return DocumentRecord(filename, pages, classify_pages(pages))


def _uncertain() -> DocumentClassification:
    return DocumentClassification("uncertain", 0.0, True)


def enumerate_pdfs(directory: Path) -> list[Path]:
    """Return PDFs recursively in stable, case-insensitive filename order."""
    return sorted(
        (path for path in directory.rglob("*") if path.is_file() and path.suffix.casefold() == ".pdf"),
        key=lambda path: (path.relative_to(directory).as_posix().casefold(), path.relative_to(directory).as_posix()),
    )


def ingest_directory(directory: Path, pipeline: IngestionPipeline | None = None) -> list[DocumentRecord]:
    pipeline = pipeline or IngestionPipeline()
    return [pipeline.process_file(path) for path in enumerate_pdfs(directory)]
