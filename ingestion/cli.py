"""Print page-level extraction and classification as JSON (no accounting output)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, BinaryIO

from .pipeline import IngestionConfig, IngestionPipeline, ingest_directory


def write_json_line(value: Any, output: BinaryIO) -> None:
    """Write one UTF-8 JSON record, independent of the console code page."""
    output.write((json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8"))
    output.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("documents"))
    parser.add_argument("--no-ocr", action="store_true", help="extract native text only")
    parser.add_argument("--ocr-language", default="eng+deu", help="installed Tesseract language codes joined with +")
    args = parser.parse_args()
    pipeline = IngestionPipeline(IngestionConfig(ocr_enabled=not args.no_ocr, ocr_language=args.ocr_language))
    for document in ingest_directory(args.directory, pipeline):
        write_json_line(document.to_dict(), sys.stdout.buffer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
