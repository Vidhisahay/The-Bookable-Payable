"""Write one exact-schema accounting JSON file per input PDF.

Accounting payloads are written to ``output/``; OCR, classification, extraction,
matching, and ERP diagnostics are written to a separate sibling directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .autodraft_pipeline import AutodraftPipeline
from .pipeline import IngestionConfig, IngestionPipeline, enumerate_pdfs


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("documents"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--diagnostics-dir", type=Path, default=Path("output_diagnostics"))
    parser.add_argument("--ocr-language", default="eng+deu")
    parser.add_argument("--page-timeout-seconds", type=int, default=12)
    parser.add_argument("--ocr-psm", default="6,11,3")
    parser.add_argument("--ocr-dpi", type=int, default=160)
    parser.add_argument("--files", nargs="*", help="process these exact PDF basenames only")
    args = parser.parse_args(argv)

    if args.page_timeout_seconds < 1 or args.ocr_dpi < 72:
        parser.error("page timeout must be positive and OCR DPI at least 72")
    try:
        modes = tuple(dict.fromkeys(int(part.strip()) for part in args.ocr_psm.split(",")))
    except ValueError:
        parser.error("--ocr-psm must be a comma-separated list of integers")
    if not modes or any(mode < 0 or mode > 13 for mode in modes):
        parser.error("--ocr-psm values must be between 0 and 13")

    paths = enumerate_pdfs(args.directory)
    if args.files is not None:
        wanted = set(args.files)
        available = {path.name: path for path in paths}
        missing = sorted(wanted - available.keys())
        if missing:
            parser.error("PDF filenames not found exactly: " + ", ".join(missing))
        paths = [path for path in paths if path.name in wanted]
    stems = [path.stem.casefold() for path in paths]
    if len(stems) != len(set(stems)):
        parser.error("input PDFs contain duplicate basenames; output filenames would collide")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    pipeline = AutodraftPipeline(ingestion=IngestionPipeline(
        IngestionConfig(ocr_language=args.ocr_language,
                        ocr_timeout_seconds=args.page_timeout_seconds,
                        ocr_page_segmentation_modes=modes,
                        render_dpi=args.ocr_dpi),
        progress=lambda message: print(f"  {message}", file=sys.stderr, flush=True),
    ))
    print(f"Processing {len(paths)} PDF(s) from {args.directory}.", file=sys.stderr)
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {path.name}", file=sys.stderr, flush=True)
        result = pipeline.process_file(path)
        _write_json(args.output_dir / f"{path.stem}.json", result["accounting"])
        _write_json(args.diagnostics_dir / f"{path.stem}.json", result["diagnostics"])
        print(f"  payables={len(result['accounting']['payables'])} "
              f"declined={len(result['accounting']['declined'])}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
