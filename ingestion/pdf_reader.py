"""PyMuPDF adapter. Import is lazy so classification tests need no PDF package."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class OCR(Protocol):
    def recognize(self, image_bytes: bytes, *, language: str, timeout: int, psm: int = 3) -> str: ...


class TesseractOCR:
    """Run the local Tesseract binary on a rendered PNG without a paid service."""

    def __init__(self, executable: str = "tesseract") -> None:
        self.executable = executable
        self._cached_failure: str | None = None
        self._available_languages: set[str] | None = None
        self._cache_identities: dict[str, str] = {}

    def prepare(self, language: str, timeout: int = 15) -> None:
        """Fail fast once if the executable or requested trained data is unavailable."""
        import subprocess
        import re

        requested = {part.strip() for part in language.split("+") if part.strip()}
        if self._cached_failure and self._available_languages is None:
            raise RuntimeError(self._cached_failure)
        if self._available_languages is not None:
            missing = sorted(requested - self._available_languages)
            if not missing:
                return
            raise RuntimeError(self._language_error(missing, self._tessdata_dir))
        try:
            result = subprocess.run(
                [self.executable, "--list-langs"], capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=timeout, check=False,
            )
        except Exception as exc:
            self._cached_failure = str(exc)
            raise RuntimeError(self._cached_failure) from exc
        if result.returncode:
            self._cached_failure = result.stderr.strip() or f"Tesseract exited with status {result.returncode}"
        else:
            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            header = lines[0] if lines else ""
            location = re.search(r'\bin\s+"([^"]+)"', header)
            self._tessdata_dir = location.group(1) if location else "(Tesseract default tessdata directory)"
            self._available_languages = set(lines[1:])
            missing = sorted(requested - self._available_languages)
            if missing:
                self._cached_failure = self._language_error(missing, self._tessdata_dir)
        if self._cached_failure:
            raise RuntimeError(self._cached_failure)

    @staticmethod
    def _language_error(missing: list[str], tessdata_dir: str) -> str:
        return f"Missing Tesseract traineddata for: {', '.join(missing)}; tessdata directory: {tessdata_dir}"

    def cache_identity(self, language: str) -> str:
        """Fingerprint engine version and requested traineddata for cache keys."""
        import hashlib
        import subprocess

        if language in self._cache_identities:
            return self._cache_identities[language]
        self.prepare(language)
        result = subprocess.run([self.executable, "--version"], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=15, check=False)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "Could not determine Tesseract version")
        parts = [str(Path(self.executable).resolve()), result.stdout.splitlines()[0].strip()]
        tessdata = Path(getattr(self, "_tessdata_dir", ""))
        for code in sorted(part.strip() for part in language.split("+") if part.strip()):
            model = tessdata / f"{code}.traineddata"
            if model.is_file():
                digest = hashlib.sha256(model.read_bytes()).hexdigest()
                parts.append(f"{code}:{digest}")
            else:
                parts.append(f"{code}:unresolved-at:{tessdata}")
        identity = "|".join(parts)
        self._cache_identities[language] = identity
        return identity

    def recognize(self, image_bytes: bytes, *, language: str = "eng", timeout: int = 120, psm: int = 3) -> str:
        import subprocess
        import tempfile

        self.prepare(language)
        with tempfile.TemporaryDirectory(prefix="pdf-ocr-") as folder:
            image_path = Path(folder) / "page.png"
            image_path.write_bytes(image_bytes)
            result = subprocess.run(
                [self.executable, str(image_path), "stdout", "-l", language, "--psm", str(psm)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=timeout, check=False,
            )
            if result.returncode:
                detail = result.stderr.strip() or f"Tesseract exited with status {result.returncode}"
                if "traineddata" in detail or "Failed loading language" in detail:
                    self._cached_failure = detail
                raise RuntimeError(detail)
            return result.stdout


class PyMuPDFReader:
    """Extract native page text and render sparse pages for optional OCR."""

    def __init__(self, dpi: int = 220) -> None:
        try:
            import pymupdf as fitz  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required to read PDFs; install requirements.txt") from exc
        self._fitz = fitz
        self.dpi = dpi

    def open(self, path: Path) -> Any:
        return self._fitz.open(path)

    def render_png(self, page: Any, *, dpi: int | None = None) -> bytes:
        render_dpi = dpi or self.dpi
        pixmap = page.get_pixmap(matrix=self._fitz.Matrix(render_dpi / 72, render_dpi / 72), alpha=False)
        return pixmap.tobytes("png")


def is_usable_text(text: str, minimum_chars: int = 24) -> bool:
    """Reject blank/sparse extraction, but allow short content when it has words."""
    alnum = sum(char.isalnum() for char in text)
    words = [token for token in text.split() if any(c.isalnum() for c in token)]
    return alnum >= minimum_chars or (alnum >= 12 and len(words) >= 3)
