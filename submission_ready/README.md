# Invoice AutoDraft — Evidence-Grounded PDF Accounting Pipeline

This project turns supplier PDFs into accounting autodrafts using the supplied schema, master data, and unchanged `erp.py` calculation. A PDF can contain zero, one, or several payables alongside supporting pages. The implementation is a **deterministic extraction and validation pipeline**, not an LLM-driven agent. It emits a payable only when the extracted components, supplier identity, schema values, and ERP check are sufficiently supported. Uncertainty is recorded in diagnostics; it is not automatically treated as a non-payable decline.

The original assessment instructions are preserved byte-for-byte in [ASSESSMENT_BRIEF.md](ASSESSMENT_BRIEF.md). [DESIGN.md](DESIGN.md) answers its three design questions.

## Prerequisites and installation (Windows)

Use a Windows CPython-compatible Python 3.10 or newer installation; ensure `python` resolves to that interpreter. The packaged run was verified with Python 3.13. Install Tesseract separately, add `tesseract.exe` to `PATH`, and install its `eng` and `est` language data. The verified environment used Tesseract 5.5.2; that version is a test record, not a strict pin. PyMuPDF is the only runtime Python package in `requirements.txt`. An internet connection or an equivalent local package source is needed for a fresh installation.

From this directory in PowerShell:

```powershell
python --version
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -c "import pymupdf; print(pymupdf.VersionBind)"
tesseract --version
tesseract --list-langs
```

Confirm that `eng` and `est` appear in the language list. The pipeline uses local OCR; it needs no paid API, credentials, or pre-existing OCR cache.

## Run the dataset

Run from this directory so the program can find `master_data/`:

```powershell
.venv\Scripts\python.exe -m ingestion.dataset_cli documents --output-dir output --diagnostics-dir output_diagnostics --ocr-language eng+est --page-timeout-seconds 60 --ocr-retries 1 --ocr-cache-dir .cache\ocr_pages
```

The positional `documents` argument is the input folder. `--output-dir` and `--diagnostics-dir` separate accounting JSON from evidence and errors. `--ocr-language` selects installed Tesseract languages. The timeout is **per OCR attempt**; `--ocr-retries 1` permits one bounded retry. Successful page OCR is cached under `--ocr-cache-dir`; timeouts and unusable text are not accepted as successful cache entries.

For another PDF folder, replace `documents` with its path and set fresh output and diagnostics directories. For a selected subset, append `--files "invoice A.pdf" "note B.pdf"` to the command; these must be exact PDF basenames in the chosen input folder. The CLI processes PDFs in deterministic order and rejects case-insensitive basename collisions that would overwrite one JSON with another. Existing files with the same output name are rewritten; the CLI does not remove stale files for PDFs no longer present, so use fresh directories when comparing runs.

## Architecture and capabilities

```text
Input PDFs
  -> native text or Tesseract OCR, page evidence and word boxes
  -> document classification and spatial party-role interpretation
  -> payable grouping, fields, lines, taxes and charges
  -> master-data matching and exact-schema mapping
  -> unchanged ERP reconciliation
  -> emitted payable / confirmed decline / unresolved diagnostics
```

`ingestion.pipeline` extracts native text first, OCRs sparse pages, bounds retries, and stores successful OCR atomically by PDF content, page, configuration, engine identity, and processing version. `ingestion.layout` retains native/OCR coordinates and uses explicit labels and proximity for supplier, buyer, ship-to, and payee evidence. `ingestion.classifier` and `ingestion.payables.extractor` classify pages, group document sections, and extract candidates with source page references and field statuses. Flattened or unclear tables remain a limitation.

`ingestion.autodraft` matches supported identifiers against `master_data/`, maps quantities and net unit prices only when evidenced, and keeps line taxes distinct from header taxes. The schema also has separate discounts, freight, other charges, and credit-memo fields; missing components stay blank rather than becoming balancing entries. `ingestion.autodraft_pipeline` invokes the unchanged ERP and emits only a supported draft. It does **not** revise candidates from an ERP difference or guess a Goods/Service type, currency, supplier code, or tax treatment.

## Outputs and decisions

For every input `X.pdf`, the CLI writes `output/X.json` with exactly `file`, `payables`, and `declined`. Each `payables[]` entry uses [AUTODRAFT_SCHEMA.md](AUTODRAFT_SCHEMA.md); credit memos use `CREDIT_MEMO` and positive magnitudes. `declined[]` is reserved for positively identified non-payable documents, such as a payment reminder. Empty `payables` **and** `declined` arrays mean the document remains unresolved, not that it was successfully accounted for.

`output_diagnostics/X.json` holds page text and extraction status, evidence, classification, matching decisions, ERP results, and unresolved reasons. Diagnostics are separate from the accounting payload. Every nonblank master-data code must correspond to a defensible match; arithmetic agreement alone does not prove faithful line or tax structure.

## Validate

```powershell
.venv\Scripts\python.exe -m pip install pytest
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ingestion.audit_dataset
.venv\Scripts\python.exe example_check.py sample_autodraft.json
```

Tests check parsing and safety behavior. The dataset audit checks file coverage, output shape, enums, referenced master codes, OCR outcomes, and ERP execution; it cannot independently prove the PDF interpretation. `example_check.py` accepts **one bare payable object**, not the per-file wrapper. To check an emitted record directly:

```powershell
.venv\Scripts\python.exe -c "import json; from erp import erp_book; p=json.load(open('output/DU-10.json', encoding='utf-8'))['payables'][0]; print(erp_book(p))"
```

Compare the result with the printed gross and the source structure, not merely with JSON syntax.

## Verified development results and limits

The packaged development set contains **42 PDFs, 42 accounting JSONs, and 42 diagnostics files**. It emits **2 payables** and **1 confirmed decline**; **39 documents remain unresolved**. Both emitted payables reconcile to their printed gross through the unchanged ERP. The suite passes **122 tests and 26 subtests**. Across 111 pages there were **0 OCR timeouts** and **2 OCR-empty pages** (INV-23 pages 2–3). These are development-set observations, not unseen-evaluation results.

Conservative gates reduce false bookings but leave complex tables, ambiguous supplier roles, unclear line types, unsupported unit-price bases, and unreadable pages unresolved. Spatial evidence helps identify roles; it does not solve every layout. The system has no runtime LLM or accounting repair loop. [DESIGN.md](DESIGN.md) explains these choices and the remaining accounting limits.

## Project structure

```text
README.md              Candidate guide (this file)
ASSESSMENT_BRIEF.md     Unchanged supplied brief
AUTODRAFT_SCHEMA.md     Supplied accounting contract
erp.py, example_check.py, sample_autodraft.json
requirements.txt        Python runtime dependency
DESIGN.md, RUN_GUIDE.md
master_data/            Supplied reference records
documents/              Supplied development PDFs
ingestion/              OCR, extraction, mapping, CLI, audit
output/                 One accounting JSON per PDF
output_diagnostics/     Per-PDF evidence and decisions
tests/                  Automated tests
```

## Troubleshooting

If Tesseract is not found, install it, put its executable directory on `PATH`, and open a new terminal. If OCR reports missing traineddata, use `tesseract --list-langs` and install `eng` and `est` for that installation. If Python cannot import `pymupdf`, verify the active interpreter and reinstall `requirements.txt` with the virtual environment's Python. Complex pages may take longer or exhaust the bounded OCR budget; inspect that page's diagnostic entry and use `--files` for a targeted retry. Use new output directories when changing the input set to avoid mistaking old JSONs for current results.
