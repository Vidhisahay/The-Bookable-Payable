"""Deterministic directory ingestion with page-level failures and provenance."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable
import subprocess
import time
import re
from typing import Any

from .classifier import classify_pages
from .models import DocumentClassification, DocumentRecord, PageRecord
from .ocr_cache import OCRPageCache
from .pdf_reader import PyMuPDFReader, TesseractOCR, is_usable_text


@dataclass(frozen=True)
class IngestionConfig:
    ocr_enabled: bool = True
    # English plus Estonian covers the supplied invoice set; callers can still
    # select another installed language combination through the CLI.
    ocr_language: str = "eng+est"
    ocr_timeout_seconds: int = 120
    minimum_native_chars: int = 24
    # Preserve tabular row structure before trying general page segmentation.
    # On the assessment's flattened invoice tables, PSM 3 often returned enough
    # prose to pass the sparse-text check while losing quantities and amounts.
    ocr_page_segmentation_modes: tuple[int, ...] = (6, 11, 3)
    render_dpi: int = 160
    supplemental_table_ocr_dpi: int = 220
    # Each retry receives the same bounded OCR budget. The default therefore
    # caps OCR subprocess time at 24 seconds per page, excluding PDF rendering.
    ocr_retries: int = 1
    ocr_cache_directory: Path | None = None


class IngestionPipeline:
    def __init__(self, config: IngestionConfig | None = None, reader: Any | None = None, ocr: Any | None = None,
                 progress: Callable[[str], None] | None = None):
        self.config = config or IngestionConfig()
        self.reader = reader
        self.ocr = ocr
        self.progress = progress
        self.cache = OCRPageCache(self.config.ocr_cache_directory) if self.config.ocr_cache_directory else None

    def _activity_table_ocr(self, pdf_sha256: str | None, page_number: int, page: Any, text: str,
                            ocr: Any, reader: Any) -> dict[str, Any] | None:
        """Retain a higher-resolution, table-oriented OCR pass for activity tables."""
        if not _needs_activity_table_ocr(text):
            return None
        dpi = self.config.supplemental_table_ocr_dpi
        key = self._cache_key(pdf_sha256, page_number, "activity_table", (6,), dpi, ocr, reader)
        cached = self.cache.load(key, minimum_chars=self.config.minimum_native_chars) if self.cache and key else None
        if cached:
            return {**cached["metadata"], "status": "found", "text": cached["text"], "psm": 6,
                    "cache": {"status": "hit", "key": key}}
        try:
            try:
                image = reader.render_png(page, dpi=dpi)
            except TypeError:
                image = reader.render_png(page)
            table_text, metadata = _recognize_with_retries(
                ocr, image, language=self.config.ocr_language,
                timeout_seconds=self.config.ocr_timeout_seconds, psm_modes=(6,),
                retries=self.config.ocr_retries, minimum_chars=self.config.minimum_native_chars,
            )
            usable = is_usable_text(table_text, self.config.minimum_native_chars)
            stored = bool(usable and self.cache and key and self.cache.store(
                key, table_text, metadata, minimum_chars=self.config.minimum_native_chars,
            ))
            return {**metadata, "status": "found" if usable else "sparse", "text": table_text,
                    "psm": 6, "rendered_bytes": len(image),
                    "cache": {"status": "stored" if stored else ("disabled" if not self.cache else "not_stored"), "key": key}}
        except OCRPageTimeout as exc:
            return {"status": "timeout", "psm": 6, "error": str(exc), "attempts": exc.attempts,
                    "cache": {"status": "miss" if self.cache else "disabled", "key": key}}
        except Exception as exc:
            return {"status": "failed", "psm": 6, "error": f"Supplemental activity-table OCR failed: {exc}",
                    "cache": {"status": "miss" if self.cache else "disabled", "key": key}}

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
            pdf_sha256 = _file_sha256(path) if self.cache else None
        except OSError:
            pdf_sha256 = None
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
                        native_evidence: dict[str, Any] = {
                            "native_alphanumeric_chars": sum(c.isalnum() for c in native_text),
                            "ocr_attempted": False,
                        }
                        # Some native PDFs retain their prose but flatten a
                        # columnar item table into an unusable reading order.
                        # Add bounded OCR evidence for this recognizable table
                        # shape while keeping the native page text authoritative
                        # for other fields.
                        if ocr is not None and _needs_column_table_ocr(native_text):
                            native_evidence["table_ocr_attempted"] = True
                            try:
                                key = self._cache_key(pdf_sha256, page_number, "table", (6,),
                                                      self.config.supplemental_table_ocr_dpi, ocr, reader)
                                cached = self.cache.load(key, minimum_chars=self.config.minimum_native_chars) if self.cache and key else None
                                if cached:
                                    native_evidence["table_ocr"] = {
                                        **cached["metadata"], "status": "found", "text": cached["text"],
                                        "psm": 6, "cache": {"status": "hit", "key": key},
                                    }
                                    pages.append(PageRecord(
                                        filename, page_number, native_text, "native_text", native_evidence,
                                    ))
                                    continue
                                prepare = getattr(ocr, "prepare", None)
                                if prepare:
                                    try:
                                        prepare(self.config.ocr_language, timeout=min(15, self.config.ocr_timeout_seconds))
                                    except TypeError:
                                        prepare(self.config.ocr_language)
                                try:
                                    image = reader.render_png(page, dpi=self.config.supplemental_table_ocr_dpi)
                                except TypeError:
                                    image = reader.render_png(page)
                                table_text, table_meta = _recognize_with_retries(
                                    ocr, image, language=self.config.ocr_language,
                                    timeout_seconds=self.config.ocr_timeout_seconds,
                                    psm_modes=(6,), retries=self.config.ocr_retries,
                                    minimum_chars=self.config.minimum_native_chars,
                                )
                                stored = bool(self.cache and key and self.cache.store(
                                    key, table_text, table_meta, minimum_chars=self.config.minimum_native_chars,
                                ))
                                native_evidence["table_ocr"] = {
                                    "status": "found" if is_usable_text(table_text, self.config.minimum_native_chars) else "sparse",
                                    "text": table_text,
                                    "psm": 6,
                                    "rendered_bytes": len(image),
                                    **table_meta,
                                    "cache": {"status": "stored" if stored else ("disabled" if not self.cache else "not_stored"), "key": key},
                                }
                            except OCRPageTimeout as exc:
                                native_evidence["table_ocr"] = {"status": "timeout", "psm": 6,
                                                                 "error": str(exc), "attempts": exc.attempts,
                                                                 "cache": {"status": "miss" if self.cache else "disabled"}}
                            except subprocess.TimeoutExpired:
                                native_evidence["table_ocr"] = {"status": "timeout", "psm": 6,
                                                                 "error": f"Supplemental table OCR timed out after {self.config.ocr_timeout_seconds}s."}
                            except Exception as exc:
                                native_evidence["table_ocr"] = {"status": "failed", "psm": 6,
                                                                 "error": f"Supplemental table OCR failed: {exc}"}
                        pages.append(PageRecord(
                            filename, page_number, native_text, "native_text",
                            native_evidence,
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

                    key = self._cache_key(pdf_sha256, page_number, "page",
                                          self.config.ocr_page_segmentation_modes, self.config.render_dpi, ocr, reader)
                    cached = self.cache.load(key, minimum_chars=self.config.minimum_native_chars) if self.cache and key else None
                    if cached:
                        cached_evidence = {**evidence, **cached["metadata"], "ocr_attempted": False,
                                           "ocr_cache": {"status": "hit", "key": key}}
                        table_evidence = self._activity_table_ocr(pdf_sha256, page_number, page, cached["text"], ocr, reader)
                        if table_evidence is not None:
                            cached_evidence["activity_table_ocr"] = table_evidence
                        pages.append(PageRecord(
                            filename, page_number, cached["text"], "ocr",
                            cached_evidence,
                            error=native_error,
                        ))
                        continue

                    ocr_attempts: list[int] = []
                    attempt_records: list[dict[str, Any]] = []
                    try:
                        prepare = getattr(ocr, "prepare", None)
                        if prepare:
                            try:
                                prepare(self.config.ocr_language, timeout=min(15, self.config.ocr_timeout_seconds))
                            except TypeError:
                                prepare(self.config.ocr_language)
                        image = reader.render_png(page)
                        text, ocr_meta = _recognize_with_retries(
                            ocr, image, language=self.config.ocr_language,
                            timeout_seconds=self.config.ocr_timeout_seconds,
                            psm_modes=self.config.ocr_page_segmentation_modes,
                            retries=self.config.ocr_retries,
                            minimum_chars=self.config.minimum_native_chars,
                        )
                        ocr_attempts = ocr_meta["ocr_modes_attempted"]
                        attempt_records = ocr_meta["ocr_attempts"]
                        method = "ocr" if is_usable_text(text, self.config.minimum_native_chars) else ("ocr_sparse" if text.strip() else "ocr_empty")
                        stored = bool(method == "ocr" and self.cache and key and self.cache.store(
                            key, text, ocr_meta, minimum_chars=self.config.minimum_native_chars,
                        ))
                        activity_table = self._activity_table_ocr(pdf_sha256, page_number, page, text, ocr, reader)
                        if activity_table is not None:
                            evidence["activity_table_ocr"] = activity_table
                        pages.append(PageRecord(
                            filename, page_number, text, method,
                            {
                                **evidence,
                                "rendered_bytes": len(image),
                                **ocr_meta,
                                "ocr_modes_attempted": ocr_attempts,
                                "ocr_cache": {"status": "stored" if stored else ("disabled" if not self.cache else "not_stored"), "key": key},
                            },
                            error=native_error,
                        ))
                    except OCRPageTimeout as exc:
                        evidence["ocr_modes_attempted"] = exc.modes_attempted
                        evidence["ocr_attempts"] = exc.attempts
                        evidence["ocr_cache"] = {"status": "miss" if self.cache else "disabled", "key": key}
                        pages.append(PageRecord(
                            filename, page_number, native_text, "ocr_timeout", evidence,
                            error="; ".join(part for part in (native_error, str(exc)) if part),
                        ))
                    except subprocess.TimeoutExpired as exc:
                        evidence["ocr_modes_attempted"] = ocr_attempts
                        evidence["ocr_attempts"] = [{"attempt": 1, "outcome": "timeout",
                                                      "error": str(exc)}]
                        pages.append(PageRecord(
                            filename, page_number, native_text, "ocr_timeout", evidence,
                            error=f"Tesseract preflight timed out before OCR: {exc}",
                        ))
                    except Exception as exc:
                        evidence["ocr_modes_attempted"] = ocr_attempts
                        evidence["ocr_attempts"] = attempt_records
                        evidence["ocr_cache"] = {"status": "miss" if self.cache else "disabled", "key": key}
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

    def _cache_key(self, pdf_sha256: str | None, page_number: int, stage: str,
                   psm_modes: tuple[int, ...], dpi: int, ocr: Any, reader: Any) -> str | None:
        if self.cache is None or pdf_sha256 is None:
            return None
        identity = getattr(ocr, "cache_identity", None)
        try:
            engine_identity = identity(self.config.ocr_language) if callable(identity) else (
                f"{type(ocr).__module__}.{type(ocr).__qualname__}"
            )
            fitz = getattr(reader, "_fitz", None)
            rasterizer_identity = getattr(fitz, "VersionBind", "unknown")
            engine_identity = f"{engine_identity}|rasterizer={rasterizer_identity}"
        except Exception:
            # Cache failure must never block extraction; a missing identity simply
            # disables cache reuse for this page.
            return None
        return self.cache.key(
            pdf_sha256=pdf_sha256, page_number=page_number, stage=stage,
            language=self.config.ocr_language, psm_modes=psm_modes, dpi=dpi,
            timeout_seconds=self.config.ocr_timeout_seconds, retries=self.config.ocr_retries,
            minimum_chars=self.config.minimum_native_chars, engine_identity=engine_identity,
        )


class OCRPageTimeout(TimeoutError):
    def __init__(self, attempts: list[dict[str, Any]], modes_attempted: list[int]):
        self.attempts = attempts
        self.modes_attempted = modes_attempted
        super().__init__("OCR timed out after bounded attempts: " + repr(attempts))


def _recognize_with_retries(ocr: Any, image: bytes, *, language: str, timeout_seconds: int,
                            psm_modes: tuple[int, ...], retries: int, minimum_chars: int
                            ) -> tuple[str, dict[str, Any]]:
    """Run a fixed number of full-budget passes; sparse and transient failures retry once."""
    attempt_records: list[dict[str, Any]] = []
    all_modes: list[int] = []
    best_text = ""
    best_alnum = -1
    selected_psm = psm_modes[0] if psm_modes else 3
    for attempt_number in range(1, retries + 2):
        started = time.monotonic()
        deadline = started + timeout_seconds
        tried: list[int] = []
        try:
            for psm in psm_modes:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired("tesseract", timeout_seconds)
                tried.append(psm)
                if psm not in all_modes:
                    all_modes.append(psm)
                candidate = ocr.recognize(
                    image, language=language, timeout=max(1, int(remaining)), psm=psm,
                )
                candidate_alnum = sum(char.isalnum() for char in candidate)
                if candidate_alnum > best_alnum:
                    best_text, selected_psm, best_alnum = candidate, psm, candidate_alnum
                if is_usable_text(candidate, minimum_chars):
                    attempt_records.append({"attempt": attempt_number, "psm_modes": tried,
                                            "elapsed_ms": round((time.monotonic() - started) * 1000),
                                            "outcome": "success"})
                    return candidate, {"ocr_alphanumeric_chars": candidate_alnum,
                                       "ocr_page_segmentation_mode": psm,
                                       "ocr_modes_attempted": all_modes,
                                       "ocr_attempts": attempt_records,
                                       "ocr_retry_count": attempt_number - 1}
            outcome = "sparse"
            attempt_records.append({"attempt": attempt_number, "psm_modes": tried,
                                    "elapsed_ms": round((time.monotonic() - started) * 1000),
                                    "outcome": outcome})
        except subprocess.TimeoutExpired as exc:
            attempt_records.append({"attempt": attempt_number, "psm_modes": tried,
                                    "elapsed_ms": round((time.monotonic() - started) * 1000),
                                    "outcome": "timeout", "error": str(exc)})
            if attempt_number > retries:
                raise OCRPageTimeout(attempt_records, all_modes) from exc
        except Exception as exc:
            attempt_records.append({"attempt": attempt_number, "psm_modes": tried,
                                    "elapsed_ms": round((time.monotonic() - started) * 1000),
                                    "outcome": "failed", "error": str(exc)})
            if attempt_number > retries:
                raise RuntimeError("OCR failed after bounded attempts: " + repr(attempt_records)) from exc
    return best_text, {"ocr_alphanumeric_chars": max(best_alnum, 0),
                       "ocr_page_segmentation_mode": selected_psm,
                       "ocr_modes_attempted": all_modes, "ocr_attempts": attempt_records,
                       "ocr_retry_count": max(0, len(attempt_records) - 1)}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _uncertain() -> DocumentClassification:
    return DocumentClassification("uncertain", 0.0, True)


def _needs_column_table_ocr(text: str) -> bool:
    """Detect a native-text page whose item table has explicit amount columns."""
    normalized = re.sub(r"\s+", " ", text).casefold()
    return bool(
        re.search(r"\bitem\s*(?:no\.?|number)\b", normalized)
        and re.search(r"\bqty\b|quantity|no\.?\s+of\s+users", normalized)
        and re.search(r"\bextended\s+amount\b", normalized)
    )


def _needs_activity_table_ocr(text: str) -> bool:
    """Recognize an activity table even when OCR confuses QTY with ``atv``."""
    normalized = re.sub(r"\s+", " ", text).casefold()
    return bool(re.search(r"\bactivity\b", normalized)
                and re.search(r"\btax\b", normalized)
                and re.search(r"\b(?:qty|quantity|atv)\b", normalized)
                and re.search(r"\brate\b", normalized)
                and re.search(r"\bamount\b", normalized))


def enumerate_pdfs(directory: Path) -> list[Path]:
    """Return PDFs recursively in stable, case-insensitive filename order."""
    return sorted(
        (path for path in directory.rglob("*") if path.is_file() and path.suffix.casefold() == ".pdf"),
        key=lambda path: (path.relative_to(directory).as_posix().casefold(), path.relative_to(directory).as_posix()),
    )


def ingest_directory(directory: Path, pipeline: IngestionPipeline | None = None) -> list[DocumentRecord]:
    pipeline = pipeline or IngestionPipeline()
    return [pipeline.process_file(path) for path in enumerate_pdfs(directory)]
