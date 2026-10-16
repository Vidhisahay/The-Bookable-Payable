"""Probe PaddleOCR in isolation; run only if paddle_venv exists and imports succeed."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
PNG1 = OUT / "page1_dpi160.png"
PNG2 = OUT / "page2_dpi160.png"


def main() -> int:
    result: dict = {"status": "not_run", "error": None}
    if not PNG1.exists():
        result["error"] = f"Missing rendered PNG: {PNG1}; run run_tesseract_baseline.py first."
        OUT.joinpath("paddle_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 1
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception as exc:
        result["status"] = "import_failed"
        result["error"] = repr(exc)
        OUT.joinpath("paddle_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 1

    # PaddleOCR 2.x/3.x: lang 'en' uses Latin script models; no dedicated 'et' in many builds.
    # Documented approach: use multilingual / latin models where available.
    t0 = time.perf_counter()
    try:
        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    except TypeError:
        ocr = PaddleOCR(use_angle_cls=True, lang="en")
    load_s = time.perf_counter() - t0
    result["model_load_seconds"] = round(load_s, 3)

    pages_out = []
    for page_num, png in [(1, PNG1), (2, PNG2)]:
        if not png.exists():
            continue
        t1 = time.perf_counter()
        try:
            raw = ocr.ocr(str(png), cls=True)
        except Exception as exc:
            pages_out.append({"page": page_num, "error": repr(exc)})
            continue
        elapsed = time.perf_counter() - t1
        lines: list[str] = []
        if raw:
            for block in raw:
                if not block:
                    continue
                for item in block:
                    if item and len(item) >= 2 and item[1]:
                        lines.append(str(item[1][0]))
        text = "\n".join(lines)
        (OUT / f"paddle_page{page_num}.txt").write_text(text, encoding="utf-8")
        pages_out.append(
            {
                "page": page_num,
                "seconds": round(elapsed, 3),
                "line_count": len(lines),
                "alnum_chars": sum(c.isalnum() for c in text),
            }
        )
    result["status"] = "completed"
    result["pages"] = pages_out
    OUT.joinpath("paddle_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
