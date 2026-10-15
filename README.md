# PAE CT Radiomics: Three-Month IPSS Prediction

Code-only release snapshot, 15 October 2026.

The primary regression target is **measured three-month IPSS**, not IPSS improvement. Baseline IPSS remains a predictor. The frozen regression cohort has 148 patients, six clinical predictors and 119 CT radiomics features. Classification of at least 25% improvement remains secondary.

## Where to start
- `Code-Final/run_ipss3m.py`: final regression-only entry point (stages 07-11).
- `Code-Final/run_all.py`: complete numbered pipeline (stages 00-12).
- `Code-Final/src/`: image preparation, radiomics, models, validation and interpretation.
- `Code-Final/config/`: example configuration, frozen predictor schema and final regression settings.
- `Code-Final/tests/`: automated tests.
- `figure_tools/`: optional revised SHAP visualizations.
- `Code-Final/README.md`: full scientific methods and execution details.

## Installation and final regression run
Python 3.12 is the documented environment. From the repository root:

```powershell
cd Code-Final
python -m pip install -r requirements.txt
python run_ipss3m.py --dry-run
python run_ipss3m.py --mode quick
python run_ipss3m.py --mode full
# Resume only an existing compatible run:
python run_ipss3m.py --mode full --resume
```

**Authorized data are required.** This public code snapshot contains no patient records or model inputs. Update `config/regression_ipss3m.json` to the authorized frozen CSV, or pass `--data` and `--output-root`. Paths in the supplied config are relative to `Code-Final`. Quick mode is a smoke test, not publishable performance.

To start with images, configure `config/pipeline_config.json` from its example and follow the detailed README. MATLAB and the specified CERR installation are required for radiomics extraction; CERR is a separate third-party dependency and is not bundled.

## Snapshot scope
All included core source/config/test files are byte-identical to the project version; see `SOURCE_MANIFEST.json`. Two workstation-specific watchdog scripts are excluded: they manage background processes and Windows tasks but do not implement the scientific analysis. The standard runners remain included. Old `Codes/`, exploratory `experiments/`, outputs, patient-level predictions, SHAP arrays, scans, workbooks, caches and third-party CERR source are excluded.

The source project's recorded full regression run completed stages 07-11. This release preparation performs syntax and archive checks; it does not rerun that expensive analysis. New machines require their own installation and test verification. Patient data cannot be reproduced from this repository alone.

The primary model's performance is internal validation, not external clinical validation. No software license is assigned in this snapshot; choose a license after confirming ownership and coauthor/institutional requirements.
