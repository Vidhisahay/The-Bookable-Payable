# Run guide

The supplied `README.md` is the assessment brief and remains unchanged. From the project root in Windows CMD or PowerShell, run:

```cmd
.venv\Scripts\python.exe -m ingestion.dataset_cli documents --output-dir output --diagnostics-dir output_diagnostics --page-timeout-seconds 60 --ocr-retries 1 --ocr-cache-dir .cache\ocr_pages
```

The command writes one accounting JSON per PDF to `output/` and separate page, extraction, matching, and validation diagnostics to `output_diagnostics/`. Each OCR pass is bounded to 60 seconds, with one bounded retry after a timeout or transient recognition failure. Only usable OCR text is written atomically to the persistent page cache; cache keys include the PDF content, page, OCR settings, Tesseract/language-model identity, rasterizer version, and OCR processing version. A timeout remains explicit and does not count as evidence that a document is non-payable.
