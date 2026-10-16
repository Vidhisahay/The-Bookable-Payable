"""Controlled Tesseract OCR on INV-02.pdf — experimental only; not used by production pipeline."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ingestion.pdf_reader import PyMuPDFReader, TesseractOCR  # noqa: E402

PDF = ROOT / "documents" / "INV-02.pdf"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

# Match dataset_cli defaults unless noted
DPI = 160
LANG = "eng+est"
PSMS = (6, 11, 3)
PAGE_TIMEOUT = 180  # per PSM attempt budget handled in loop below


def peak_rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return None


def run_tesseract_page(png_bytes: bytes, psm: int, timeout: int) -> tuple[str, float, str | None]:
    ocr = TesseractOCR()
    ocr.prepare(LANG, timeout=15)
    t0 = time.perf_counter()
    err: str | None = None
    text = ""
    try:
        text = ocr.recognize(png_bytes, language=LANG, timeout=timeout, psm=psm)
    except subprocess.TimeoutExpired as exc:
        err = f"TimeoutExpired after {timeout}s: {exc}"
    except Exception as exc:
        err = str(exc)
    elapsed = time.perf_counter() - t0
    return text, elapsed, err


def score_text(text: str) -> dict[str, bool | int]:
    checks = {
        "invoice_9972_907": "9972-907" in text.replace(" ", ""),
        "date_18_03_2026": "18-03-2026" in text or "18.03.2026" in text,
        "due_01_04_2026": "01-04-2026" in text or "01.04.2026" in text,
        "currency_euro": "€" in text or "EUR" in text.upper(),
        "subtotal_538_60": "538,60" in text or "538.60" in text,
        "discount_48_09": "48,09" in text or "48.09" in text,
        "net_490_51": "490,51" in text or "490.51" in text,
        "vat_117_72": "117,72" in text or "117.72" in text,
        "gross_608_23": "608,23" in text or "608.23" in text,
    }
    extensions = ["54,00", "10,00", "90,00", "45,00", "18,00", "103,60", "50,00", "168,00"]
    ext_hits = sum(1 for e in extensions if e in text or e.replace(",", ".") in text)
    checks["line_extensions_hits"] = ext_hits  # type: ignore[assignment]
    checks["all_eight_extensions"] = ext_hits >= 8  # type: ignore[assignment]
    checks["alnum_chars"] = sum(c.isalnum() for c in text)  # type: ignore[assignment]
    return checks


def main() -> int:
    reader = PyMuPDFReader(dpi=DPI)
    doc = reader.open(PDF)
    summary: dict = {
        "pdf": str(PDF),
        "tesseract_version": subprocess.run(
            ["tesseract", "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace"
        ).stdout.splitlines()[0],
        "language": LANG,
        "render_dpi": DPI,
        "page_timeout_seconds_per_psm": PAGE_TIMEOUT,
        "pages": [],
    }
    mem_start = peak_rss_mb()
    for index in range(len(doc)):
        page_num = index + 1
        page = doc[index]
        png = reader.render_png(page)
        png_path = OUT / f"page{page_num}_dpi{DPI}.png"
        png_path.write_bytes(png)
        page_entry = {"page_number": page_num, "png_bytes": len(png), "psm_runs": [], "best": None}
        best_alnum = -1
        best_psm = None
        best_text = ""
        for psm in PSMS:
            rss_before = peak_rss_mb()
            text, elapsed, err = run_tesseract_page(png, psm, PAGE_TIMEOUT)
            rss_after = peak_rss_mb()
            alnum = sum(c.isalnum() for c in text)
            run = {
                "psm": psm,
                "seconds": round(elapsed, 3),
                "error": err,
                "alnum_chars": alnum,
                "line_count": len([ln for ln in text.splitlines() if ln.strip()]),
                "checks": score_text(text),
                "rss_mb_before": round(rss_before, 1) if rss_before else None,
                "rss_mb_after": round(rss_after, 1) if rss_after else None,
            }
            page_entry["psm_runs"].append(run)
            if err is None and alnum > best_alnum:
                best_alnum = alnum
                best_psm = psm
                best_text = text
        if best_psm is not None:
            out_txt = OUT / f"tesseract_page{page_num}_psm{best_psm}.txt"
            out_txt.write_text(best_text, encoding="utf-8")
            page_entry["best"] = {"psm": best_psm, "alnum_chars": best_alnum, "checks": score_text(best_text), "text_file": out_txt.name}
        summary["pages"].append(page_entry)
    doc.close()
    summary["process_rss_mb_start"] = round(mem_start, 1) if mem_start else None
    summary["process_rss_mb_end"] = round(peak_rss_mb(), 1) if peak_rss_mb() else None
    report_path = OUT / "tesseract_summary.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
