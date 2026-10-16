"""Run the complete PDF-to-autodraft pipeline. JSON goes to stdout; progress to stderr."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .autodraft_pipeline import AutodraftPipeline
from .cli import write_json_line
from .pipeline import IngestionConfig, IngestionPipeline, enumerate_pdfs


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    diagnostics = result["diagnostics"]
    classification = diagnostics["classification"]
    candidates = diagnostics["extraction"]["candidates"]
    return {
        "accounting": result["accounting"],
        "diagnostics": {
            "document_error": diagnostics["document_error"],
            "classification": classification,
            "pages": [{"page_number": p["page_number"], "extraction_method": p["extraction_method"],
                       "error": p["error"], "text_preview": p["text"][:500]} for p in diagnostics["pages"]],
            "extraction": {"excluded_reason": diagnostics["extraction"]["excluded_reason"],
                           "review_required": diagnostics["extraction"]["review_required"],
                           "warnings": diagnostics["extraction"]["warnings"],
                           "candidates": [{"fields": c["fields"], "master_hints": c["master_hints"],
                                           "matches": c["matches"], "reconciliation": c["reconciliation"]}
                                          for c in candidates]},
        },
    }


def _parse_pages(value: str) -> list[int]:
    """Parse a 1-based page list such as ``1,3-5``."""
    pages: set[int] = set()
    try:
        for part in value.split(","):
            if "-" in part:
                first, last = (int(item.strip()) for item in part.split("-", 1))
                if first < 1 or last < first:
                    raise ValueError
                pages.update(range(first, last + 1))
            else:
                number = int(part.strip())
                if number < 1:
                    raise ValueError
                pages.add(number)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pages must be positive numbers/ranges, e.g. 1,3-5") from exc
    return sorted(pages)


def _parse_psm(value: str) -> tuple[int, ...]:
    try:
        modes = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
        if not modes or any(mode < 0 or mode > 13 for mode in modes):
            raise ValueError
        return modes
    except ValueError as exc:
        raise argparse.ArgumentTypeError("PSM modes must be comma-separated integers from 0 through 13") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("documents"))
    parser.add_argument("--files", nargs="*", help="process these exact PDF basenames only")
    parser.add_argument("--no-ocr", action="store_true", help="native text only")
    parser.add_argument("--ocr-language", default="eng+deu")
    parser.add_argument("--page-timeout-seconds", type=int, default=60,
                        help="shared OCR time budget per page, including segmentation retries")
    parser.add_argument("--pages", type=_parse_pages,
                        help="process only these 1-based pages (for example 3,6-11)")
    parser.add_argument("--ocr-psm", type=_parse_psm, default=(6, 11, 3),
                        help="OCR page segmentation modes to try in order")
    parser.add_argument("--ocr-dpi", type=int, default=160,
                        help="PDF render resolution for OCR")
    parser.add_argument("--summary", action="store_true", help="emit field/evidence summary instead of full page text")
    args = parser.parse_args(argv)
    if args.page_timeout_seconds < 1 or args.ocr_dpi < 72:
        parser.error("--page-timeout-seconds must be positive and --ocr-dpi at least 72")
    paths = enumerate_pdfs(args.directory)
    if args.files is not None:
        selected: list[Path] = []
        missing: list[str] = []
        for wanted in args.files:
            matches = [path for path in paths if path.name == wanted]
            if len(matches) == 1:
                selected.append(matches[0])
            elif not matches:
                missing.append(wanted)
            else:
                print(f"Ambiguous exact basename {wanted!r}; use a unique sample name.", file=sys.stderr)
                return 2
        if missing:
            print("Requested PDF filenames not found exactly: " + ", ".join(missing), file=sys.stderr)
            return 2
        paths = selected

    config = IngestionConfig(ocr_enabled=not args.no_ocr, ocr_language=args.ocr_language,
                             ocr_timeout_seconds=args.page_timeout_seconds,
                             ocr_page_segmentation_modes=args.ocr_psm, render_dpi=args.ocr_dpi)
    pipeline = AutodraftPipeline(ingestion=IngestionPipeline(
        config, progress=lambda message: print(f"  {message}", file=sys.stderr, flush=True)))
    results = []
    print(f"Processing {len(paths)} PDF(s) from {args.directory}.", file=sys.stderr)
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {path.name}", file=sys.stderr)
        result = pipeline.process_file(path, page_numbers=args.pages)
        results.append(_summary(result) if args.summary else result)
        classification = result["diagnostics"]["classification"]
        methods: dict[str, int] = {}
        for page in result["diagnostics"]["pages"]:
            methods[page["extraction_method"]] = methods.get(page["extraction_method"], 0) + 1
        print(f"  type={classification['category']} confidence={classification['confidence']:.2f} "
              f"payables={len(result['accounting']['payables'])} page_methods={methods}", file=sys.stderr)
    write_json_line({"results": results}, sys.stdout.buffer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
