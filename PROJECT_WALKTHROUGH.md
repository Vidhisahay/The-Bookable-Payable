# Project walkthrough: The Bookable Payable

## Problem and output contract

The task is to turn each supplier PDF into zero, one, or multiple bookable accounting drafts. Each `output/<PDF stem>.json` has exactly `file`, `payables`, and `declined`. Each payable must match `AUTODRAFT_SCHEMA.md`; diagnostics belong in the separate `output_diagnostics/` directory. An uncertain document is kept unresolved (both arrays empty), while `declined` is reserved for a positively identified non-payable such as a payment reminder.

## Architecture and data flow

1. `ingestion/dataset_cli.py:main` discovers PDFs with `enumerate_pdfs`, applies CLI OCR settings, calls `AutodraftPipeline.process_file`, then writes one accounting JSON and one sidecar diagnostics JSON per PDF. Progress goes to stderr.
2. `ingestion/autodraft_pipeline.py:AutodraftPipeline.process_file` delegates page processing to `IngestionPipeline.process_file`; `process_document` scopes pages, calls classification and payable extraction, performs master matching, maps the candidate to the schema, and records reconciliation outcomes.
3. `ingestion/pipeline.py:IngestionPipeline.process_file` extracts page text, retains provenance, classifies the assembled document, and catches page-level failures.
4. `ingestion/payables/extractor.py:extract_payables` produces candidates and field-level statuses. `ingestion/autodraft.py:generate_autodraft` maps supported candidate values, matches master data, and calls the unchanged `erp.erp_book` accounting oracle.
5. `ingestion/audit_dataset.py:audit` checks output coverage, schema keys, enumerated values, master references, diagnostics, and whether ERP calculation succeeds.

Key models live in `ingestion/models.py` and `ingestion/payables/models.py`. Numeric locale parsing is in `ingestion/payables/numbers.py`.

## PDF text and OCR

`PyMuPDFReader` in `ingestion/pdf_reader.py` reads native page text and renders sparse pages. The pipeline accepts native text when `is_usable_text` passes; otherwise it invokes local `TesseractOCR` using `subprocess`. Current dataset command settings use `eng+deu`, 160 DPI, a shared 12-second per-page budget, and PSM modes 6, 11, and 3. The modes share one deadline, so a page cannot consume the full timeout for each retry. Available local Tesseract packs also include `est`, `por`, and `tha`, but those are not in the full-run default language string. OCR errors, empty pages, and timeouts remain page records instead of being silently replaced with guessed text.

This is deterministic native extraction/OCR; there is no LLM or learned classifier.

## Classification, extraction, and evidence

`ingestion/classifier.py` uses regular-expression cue groups and a simple evidence-count confidence heuristic. It can return `uncertain`; `DocumentClassification` also records per-page categories, evidence, mixed-document flags, and possible starts. This is rule-based scoring, not a statistical confidence probability.

The payable extractor uses multilingual labels and patterns for invoice identifiers, dates, currencies, totals, rows, taxes, and charges. It does not infer a value merely to complete an object. `ExtractedField` statuses are `found`, `missing`, `ambiguous`, `conflicting`, or `unreadable`; extracted evidence carries filename, page number, and a text excerpt. Line taxes and header taxes are kept separately. The parser is heuristic and flattened OCR tables can lose row associations or values.

## Master data and ERP

`MasterData.load` loads supplier, company, tax, payment-term, and purchase-order JSON. Supplier matching requires a unique normalized VAT ID, or normalized name plus address. Buyer codes are matched only from explicit buyer context and exact normalized master names/codes. Payment terms require a printed alias; a due date alone is not used. A PO requires an explicit PO number and a supplier relationship. Tax matching uses rate and available supplier-country/type evidence; unresolved or ambiguous matches leave codes blank.

`generate_autodraft` maps the supported source components to the exact schema. The fixed `erp.py:erp_book` computes line bases from quantity and net unit price, applies line taxes, header discount and header taxes, then adds charges. A comparison with printed gross is useful only when the source components and tax placement are themselves supported. Current emission is conservative: only candidates whose ERP check passes are emitted; unresolved candidates stay in diagnostics.

## Multi-page documents, errors, and duplicates

Page scoping in `autodraft_pipeline.py:_scope_payable_pages` keeps invoice pages and uncertain continuation pages until explicit attachment/non-payable boundaries. Repeated invoice identifiers alone do not create another payable. A changed invoice identifier or a separator/page structure can suggest another section, but candidates still need source evidence. An OCR timeout is recorded and may leave a document unresolved; it does not prove the page is non-payable. JSON is written as UTF-8 with `ensure_ascii=False`, and progress is separate from the JSON payload.

## Setup and commands

Python dependencies are listed in `requirements.txt` (currently PyMuPDF). OCR also needs a local Tesseract executable and installed trained data; those are OS-level dependencies, not Python packages. On this machine the executable is `C:\msys64\mingw64\bin\tesseract.exe`, with `eng`, `deu`, `est`, `por`, and `tha` installed.

From the project root, install the Python dependency and run tests:

```cmd
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest -q
```

Run all PDFs with the documented configuration:

```cmd
.venv\Scripts\python.exe -m ingestion.dataset_cli documents --output-dir output --diagnostics-dir output_diagnostics --page-timeout-seconds 12
```

The generation command overwrites matching output and diagnostics files; inspect/back up them before intentionally running it. This walkthrough's audit did not run the generation command.

## Examples from the current dataset

- **Emitted payable — `INV-01.pdf`:** invoice 852566, dated 02.02.2026, due 12.02.2026, two service lines (4 and 2 hours at €73), €292 and €146 line totals, printed gross €438, and 0% VAT/reverse-charge context. The net-price evidence is in the source wording, the ERP result is €438 EUR, and this is the one emitted payable. This is an arithmetic pass, not a claim that every master code was resolved.
- **Unresolved invoice — `INV-02.pdf`:** targeted OCR/source review found invoice 9972-907 and printed gross €608.23 with €538.60 subtotal, €48.09 discount, and €117.72 VAT. Several rows show unit prices without printed line totals, and multiplying all of them conflicts with the subtotal, so the record remains unbooked pending line treatment review.
- **Confirmed non-payable — `DU-08.pdf`:** the source is a payment reminder. It references an existing invoice and does not establish a new obligation; the output declines it with that reason.

Other useful manual-review examples: `HLD-01.pdf` is a Thai invoice whose source shows gross THB 8,397.36 and payment after withholding THB 8,161.92; accounting needs to preserve the withholding. `DU-03.pdf` contains a USD 1,040.06 invoice plus a shipment advice and separate SGD 722.76 customs GST page. `INV-31.pdf` has a `$17,657.53` amount due, but the dollar sign and date labels require careful interpretation.

## Current limitations and next improvements

The current audit found 42 PDFs and 42 JSONs, with one emitted payable, one positively declined document, and 40 unresolved documents. The main run recorded many page timeouts; targeted OCR recovered evidence for selected pages, but that does not make all those documents bookable. The parser can miss totals on scanned tables, the full-run default OCR languages do not include every locally available language, and supplier/buyer labeling can be difficult in visual layouts. A sensible next phase is human verification of the prioritized cases in `MANUAL_REVIEW.md`, then targeted extraction fixes only when a human confirms row and tax semantics. Any subsequent full run should be auditable and performed once after such fixes, not used as exploratory OCR tuning.

## Likely technical interview questions

1. **Why not emit every detected invoice?** Detection establishes document type, not complete accounting structure. The pipeline leaves candidates unresolved when required source components or arithmetic evidence are missing.
2. **How is a scanned page handled?** Native text is attempted first; sparse text triggers a rendered image and local Tesseract OCR. Page methods and failures are kept in diagnostics.
3. **What does the confidence value mean?** It is a deterministic cue-score heuristic in `classifier.py`, not a calibrated probability or model confidence.
4. **How are duplicate invoice pages handled?** Repeated IDs alone do not imply a second obligation; page boundaries and changed identifiers provide evidence for candidate starts.
5. **How are tax scopes preserved?** Extracted taxes carry a scope (`line` or `header`) and mapping sends them to the corresponding schema location.
6. **How is a supplier matched?** By unique normalized VAT ID, or unique normalized name plus address; a name-only resemblance is insufficient.
7. **How are payment terms inferred?** They are matched from printed aliases in master data. A due date is not converted into a term.
8. **What is the ERP check?** `erp_book` recomputes net line bases, discounts, line/header taxes, and charges, returning the booked gross and currency. It cannot prove the source structure is correct.
9. **What happens on OCR timeout?** The page receives `ocr_timeout` plus an error and remains part of the document evidence; no text is invented and the document may remain unresolved.
10. **What is the largest current limitation?** Coverage: 40 documents remain unresolved and OCR failures leave important pages unreadable. Manual review is needed before claiming broader bookability.
