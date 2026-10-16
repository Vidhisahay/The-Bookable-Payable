# Design notes

## What I understand now

A PDF is an evidence bundle, not necessarily one invoice. A payable can span several pages, while the same file may include shipping advice, customs papers, delivery notes, or a remittance stub. Those pages can repeat invoice identifiers and amounts without creating another payable. The invoice gross, amount after withholding, and tax-purpose currency conversions are also different quantities. The ERP can calculate a number from submitted parts, but a matching number does not prove that the parts or tax placement are faithful.

## What the system does with an unfamiliar document

The system extracts searchable text first and OCRs pages that lack usable text, trying table-oriented segmentation first. It keeps page numbers, extraction methods, OCR errors, classification evidence, field evidence, master-data decisions, and ERP outcomes in diagnostics separate from the accounting JSON. Classification and extraction use document text rather than filenames. It keeps repeated invoice identifiers together unless a different identifier or explicit document boundary supports a new obligation. It emits an accounting payable only when the source components are complete enough for an exact ERP reconciliation; unresolved candidates remain in diagnostics and are neither booked nor incorrectly declined. It resolves codes only through supported master-data matches and does not add balancing amounts. A matching total alone does not establish the correct document structure.

To run the integrated pipeline and write one accounting JSON per PDF, with diagnostics stored separately:

```powershell
.venv\Scripts\python.exe -m ingestion.dataset_cli documents --output-dir output --diagnostics-dir output_diagnostics --page-timeout-seconds 12
```

## A document that remains unresolved

`INV-31.pdf` has an invoice amount due of `$17,657.53`, but the dollar symbol does not establish a currency. Its remittance stub and billing summary also present conflicting due-date evidence, and the extracted native text does not provide supported electricity line components. The output therefore leaves the currency and dates unresolved and does not claim ERP validation. Choosing USD, or selecting a date based only on which label appears convenient, would add information the evidence does not settle.

Other open cases remain unresolved as well. A targeted Thai+English OCR retry recovered HLD-01's text, including THB 8,397.36, but its invoice number, date, supplier, quantity interpretation, and net-price basis remain incomplete or ambiguous. DU-02's first two pages confirm one consolidated invoice number, but no currency, gross, or invoice lines are recoverable; its customs-detail pages repeat the same identifier and do not establish additional payables. DU-03's first two pages yield one USD 1,040.06 invoice candidate with ten lines, but the document does not establish that the extracted unit prices are tax-exclusive; seven other pages still time out. These are review items, not successful bookings.
