# PAE CT Radiomics and Clinical Modelling

This directory is the final, numbered and reproducible code package for the
MSc project. It starts from the authorised DICOM, segmentation and clinical
inputs, extracts 119 Original 3D CERR radiomics features, creates the fixed
modelling cohorts, and reproduces the regression, ablation, model-screening,
bootstrap, interpretation and secondary-classification analyses.

The primary regression outcome is measured 3-month IPSS: `ipss_3m`.

The historical improvement target `baseline IPSS - 3-month IPSS` is retained
for explicit legacy comparisons and is never used as a predictor.

The six pre-treatment clinical predictors are age, prostate volume,
Abrams-Griffiths number, Qmax, post-void residual and baseline IPSS. Baseline
IPSS is retained as a pre-treatment predictor. It is not directly subtracted
in the new primary outcome definition; predictive and causal interpretations
must still be distinguished.

## Directory structure

```text
Code-Final/
  config/
    cerr_core_radiomics_params_full_original.json
    pipeline_config.example.json
    session_series_manifest.example.csv
  src/
    step00_build_session_manifest.m
    01_match_ct_masks.py
    02_export_segmentation_review.py
    03_prepare_cerr_inputs.py
    04_extract_cerr_radiomics.py
    step04_extract_cerr_radiomics.m
    05_build_and_qc_radiomics_table.py
    06_prepare_modelling_data.py
    07_baseline_regression.py
    08_pca_lasso_ablation.py
    09_additional_regression_models.py
    10_bootstrap_stability.py
    11_feature_importance.py
    12_secondary_classification.py
    common/
  tests/
  requirements.txt
  run_all.py
  run_pipeline.ps1
```

Patient data, the generated real session manifest, CERR itself and generated
outputs are deliberately not redistributed in this code directory.

## Software and installation

- Python 3.12
- MATLAB R2024b or compatible
- CERR commit `a0f74fadcd952e066e27a2cea53ae6108c307cad`
- Python versions pinned in `requirements.txt`

From `Code-Final`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CERR is LGPL-2.1 third-party software and is not included. Obtain it from the
CERR repository and check out the commit above.

## Configuration

Create a local configuration and edit the five paths:

```powershell
Copy-Item config\pipeline_config.example.json config\pipeline_config.json
```

The paths identify the approved data root, clinical workbook, saved MATLAB
sessions, DICOM root and CERR installation. The remaining fields freeze random
seeds, parallelism and analysis repetition counts. Do not reduce the full-mode
defaults when generating final reported results.

`session_series_manifest.example.csv` documents the schema only. Step 00
creates `config/session_series_manifest.csv` from `Session.mat`. It can contain
patient-level identifiers and should remain on approved storage.

## Running the pipeline

Run all steps from `Code-Final`:

```powershell
python run_all.py --mode full
```

Equivalent PowerShell wrapper:

```powershell
.\run_pipeline.ps1 -Step all -Mode full
```

Run one stage, a range, or resume checkpointed long analyses:

```powershell
python run_all.py --step 08 --resume
python run_all.py --from-step 07 --to-step 11 --resume
.\run_pipeline.ps1 -Step 10 -Mode full -Resume
```

Preview commands without executing them:

```powershell
python run_all.py --from-step 07 --to-step 12 --dry-run
```

`--mode quick` is a smoke test: it uses four outer LOOCV patients and three
bootstrap replicates and writes analysis results to `outputs_quick`. It is not
valid for reporting. Full analysis outputs are written to sibling directory
`outputs_final` so generated data do not become part of the submitted code.

Every Python stage also supports `--help` and can be executed directly.

## Numbered stages

### 00?06: data preparation

00 derives the CT-series manifest from MATLAB sessions. 01 matches CT series
and masks and checks geometry. 02 exports segmentation-review material. 03
resamples CT linearly and masks with nearest-neighbour interpolation to 1 mm
isotropic spacing. 04 runs complete Original 3D CERR extraction. 05 validates
and preserves the complete 119-feature table while reporting constant and
duplicate features. 06 merges the six clinical variables and constructs the
fixed regression and classification cohorts.

### 07: frozen baseline multimodal regression

This reproduces the original primary comparison with nested LOOCV and the
original fold-internal PCA-to-LASSO radiomics chain. It evaluates clinical,
radiomics, early fusion, learned weighted late fusion and stacking. The final
default estimator is Ridge; alternative legacy estimators remain selectable
for audit purposes.

### 08: PCA and LASSO ablation

Five matched nested-LOOCV configurations use the same Ridge estimator:
clinical only, all 119 early-fusion features, PCA, LASSO, and PCA followed by
LASSO. Scaling, PCA, LASSO selection and hyperparameter tuning are fitted only
inside each training fold. The stage records selected features, reduction
counts, convergence, pooled metrics, and 10,000 paired patient-level bootstrap
comparisons of the out-of-fold absolute errors.

### 09: additional regression models

Ten pre-specified algorithms are screened for clinical and early-fusion input:
Ridge, elastic net, Bayesian Ridge, ARD, PLS, Huber, median quantile regression,
RBF SVR, gradient boosting and Extra Trees. Each configuration uses the same
nested LOOCV framework. Frozen clinical-Ridge and LASSO early-fusion-Ridge OOF
predictions from step 08 are included in the paired comparison. MAE is the
primary selection metric; uncertainty and model simplicity are considered
rather than selecting solely from the smallest point estimate.

### 10: bootstrap stability and overfitting

The analysis requires 500 valid paired bootstrap samples. Each sample repeats
all preprocessing, feature reduction, inner patient-grouped tuning and model
fitting. Apparent in-bag and unique-patient OOB errors quantify optimism. The
same draws are used for clinical Ridge, the four ablation strategies and the
two leading additional early-fusion models (gradient boosting and ARD).
Non-converged attempts are recorded and replaced rather than included.

### 11: feature importance and interpretation

This stage produces:

- LASSO feature-selection frequencies over valid bootstrap fits;
- bootstrap distributions of standardised clinical-Ridge coefficients;
- individual and modality-grouped OOB permutation importance for gradient
  boosting; and
- a full-cohort gradient-boosting SHAP ranking and summary plots.

The bootstrap and OOB analyses describe stability. The full-cohort SHAP result
is explicitly descriptive and must not be presented as external validation or
as a causal analysis.

### 12: secondary classification

Nested stratified cross-validation evaluates the predefined response outcome
(at least 25% IPSS improvement). Thresholds are selected only from inner
training predictions using Youden's J. This remains a secondary analysis and
does not affect primary regression-model selection.

## Leakage controls

No held-out outer patient is used to fit scaling, variance filtering, PCA,
LASSO selection, hyperparameters, fusion weights, stacking models or decision
thresholds. Bootstrap duplicates are grouped by patient identity in inner CV,
preventing copies of one patient from appearing in both inner training and
validation splits.

## Verification

Run fast automated tests:

```powershell
python -m unittest discover -s tests -t . -v
python -m compileall -q src run_all.py
```

Before submission, also run the quick analysis smoke test and then the full
pipeline. A stage is complete only when it creates its `summary.json` with
`"status": "pass"`. Long stages 07?11 support `--resume`; partial checkpoint
files are removed after successful completion.

All performance results are internal resampling estimates from a 148-patient
cohort. They are not independent external validation and do not establish
incremental clinical value or causality.

## Regression endpoint update (2026-10-07)

Use the regression-only entry point for the new supervisor-requested endpoint.
It uses the frozen original 148-patient cohort, six baseline clinical predictors
and the exact 119-feature manifest in `config/regression_feature_schema.json`.
It does not repeat imaging extraction or change secondary classification.

```powershell
python run_ipss3m.py --dry-run
python run_ipss3m.py --mode quick
python run_ipss3m.py --mode full
python run_ipss3m.py --mode full --resume
```

`config/regression_ipss3m.json` freezes the original seed (42), inner three-fold
CV, PCA threshold, bootstrap counts and conservative single-worker execution.
The new outputs are `../outputs/ipss_3m_regression/`: `data_audit`, `baseline`,
`ablation`, `model_screen`, `bootstrap`, `interpretation`, and stage logs.
Quick outputs are isolated beneath `quick/` and carry `smoke_test` status;
they are not complete cohort performance results. Full bootstrap still requires
500 valid paired resamples. All original model grids are retained.

The entry point caches only Python code locally to avoid slow UNC import
searches. Patient data and outputs stay on the project share. Executed source
hashes are checked against the project source before fitting. Worker numerical
libraries use one thread; model seeds and SHAP plot jitter are fixed.

Each stage records data SHA-256, patient order, feature schema, source hashes,
scientific settings, Python and package versions. `--resume` refuses changed
endpoints, inputs, code, seeds, fold limits or scientific settings. It verifies
completed output hashes and skips completed stages. Legacy checkpoints without
these records cannot be adopted. Parallel writers to the same output directory
are refused by an OS-level lock. Start a new output root after intentional
scientific or source-code changes; worker count can change safely.

Follow progress in `pipeline_status.json` and `logs/07.log` through `logs/11.log`.
A failure is recorded and must be diagnosed before resuming. The current target
can be checked in each stage's `summary.json` and `run_manifest.json`.

The measured follow-up target is never overwritten by a change calculation.
Both follow-up and historical change columns are explicitly excluded from
predictors. Raw feature names and the final preparation's `radiomics__` names
are supported with the same fixed feature order. Unexpected columns are refused.

Stage 11 retains descriptive full-cohort SHAP and additionally creates
`patient_cases.csv`, local SHAP values and waterfall plots from outer-held-out
gradient-boosting models. Case selection is deterministic and descriptive:
lowest-error case in each observed-score half, followed by the largest error.
Refitted predictions and SHAP additivity are checked. These cases do not
constitute an independent validation or a causal explanation.

Run all tests with `python -m unittest discover -s tests -v`. Endpoint tests
cover raw/prefixed feature schemas, unchanged follow-up values, leakage column
rejection, cohort validity, incompatible restoration, artifact integrity,
OOF coverage and output locks. Original code is backed up on the project share
in `../outputs/ipss_3m_regression/data_audit/code_before_ipss3m/`.

On this Windows/network-share workstation the validated default is one worker.
Parallel worker startup is slow; increasing n_jobs is optional and should be
benchmarked separately. Numerical libraries remain limited to one thread.

The regression-only runner passes byte-identical frozen cohort and feature-schema
snapshots from `data_audit/` to every stage. Later edits to the original source
cannot change the cohort midway through a long run. The original source and
snapshot SHA-256 values are recorded together.
