# Run guide

The supplied `README.md` is the assessment brief and is preserved unchanged. Use this companion guide for the single dataset-generation command. From the project root in Windows CMD or PowerShell:

```cmd
.venv\Scripts\python.exe -m ingestion.dataset_cli documents --output-dir output --diagnostics-dir output_diagnostics --page-timeout-seconds 12
```

This writes one accounting JSON per PDF to `output/` and separate extraction and validation diagnostics to `output_diagnostics/`.
