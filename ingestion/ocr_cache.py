"""Validated, atomic cache for successful page OCR results."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .pdf_reader import is_usable_text


OCR_CACHE_VERSION = 1
OCR_PROCESSING_VERSION = "tesseract-page-selection-v1"


class OCRPageCache:
    """Cache OCR text by source PDF, page, and every text-affecting setting."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    @staticmethod
    def key(
        *, pdf_sha256: str, page_number: int, stage: str, language: str,
        psm_modes: tuple[int, ...], dpi: int, timeout_seconds: int,
        retries: int, minimum_chars: int, engine_identity: str = "unknown",
    ) -> str:
        identity = {
            "cache_version": OCR_CACHE_VERSION,
            "processing_version": OCR_PROCESSING_VERSION,
            "pdf_sha256": pdf_sha256,
            "page_number": page_number,
            "stage": stage,
            "language": language,
            "psm_modes": list(psm_modes),
            "dpi": dpi,
            "timeout_seconds": timeout_seconds,
            "retries": retries,
            "minimum_chars": minimum_chars,
            "engine_identity": engine_identity,
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def load(self, key: str, *, minimum_chars: int) -> dict[str, Any] | None:
        path = self.directory / f"{key}.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if (not isinstance(value, dict) or value.get("cache_version") != OCR_CACHE_VERSION
                or value.get("key") != key or not isinstance(value.get("text"), str)
                or not isinstance(value.get("metadata"), dict)
                or not is_usable_text(value["text"], minimum_chars)):
            return None
        return {"text": value["text"], "metadata": value["metadata"]}

    def store(self, key: str, text: str, metadata: dict[str, Any], *, minimum_chars: int) -> bool:
        """Atomically save only usable OCR text; failures are never cache entries."""
        if not is_usable_text(text, minimum_chars) or not isinstance(metadata, dict):
            return False
        payload = {
            "cache_version": OCR_CACHE_VERSION,
            "key": key,
            "text": text,
            "metadata": metadata,
        }
        temporary: str | None = None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=f".{key}.", suffix=".tmp", dir=self.directory)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.directory / f"{key}.json")
            return True
        except (OSError, TypeError, ValueError):
            if temporary:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
            return False
