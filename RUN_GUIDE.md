# Run guide

The supplied `README.md` is the assessment brief and remains unchanged; this companion guide supplies its requested run command. Use a CPython 3.10 or newer installation. From the project root in Windows CMD or PowerShell, create a local environment and install the Python dependency:

```cmd
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install Tesseract separately and put `tesseract.exe` on `PATH`. Install its `eng` and `est` traineddata files; verify both appear in `tesseract --list-langs`. Tesseract is a system program, not a Python package in `requirements.txt`. Other languages can be selected with `--ocr-language` after their traineddata is installed.

Run the dataset command from the project root:

```cmd
.venv\Scripts\python.exe -m ingestion.dataset_cli documents --output-dir output --diagnostics-dir output_diagnostics --ocr-language eng+est --page-timeout-seconds 60 --ocr-retries 1 --ocr-cache-dir .cache\ocr_pages
```

The command writes one accounting JSON per PDF to `output/` and separate page, extraction, matching, and validation diagnostics to `output_diagnostics/`. Each OCR pass is bounded to 60 seconds, with one bounded retry after a timeout or transient recognition failure. Only usable OCR text is written atomically to the persistent page cache; cache keys include the PDF content, page, OCR settings, Tesseract/language-model identity, rasterizer version, and OCR processing version. A timeout remains explicit and does not count as evidence that a document is non-payable.

Check the outputs with `.venv\Scripts\python.exe -m ingestion.audit_dataset`. Run automated tests with `.venv\Scripts\python.exe -m pytest -q`; install `pytest` separately for tests (`.venv\Scripts\python.exe -m pip install pytest`). The runtime does not require pytest, a paid API, an existing OCR cache, or the pre-generated outputs. Run the command from the project root because `master_data/` is resolved relative to the working directory.
