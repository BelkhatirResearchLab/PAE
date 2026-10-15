"""Prepare the final classification and regression modelling cohorts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common.io_utils import atomic_csv, atomic_json, atomic_text, normalise_patient_id


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLINICAL = PROJECT_ROOT.parent / "DICOM" / "PAE updated data_SCANS_ANON.xlsx"
DEFAULT_RADIOMICS = (
    PROJECT_ROOT
    / "outputs_final"
    / "05_radiomics_qc"
    / "radiomics_features_CERR_full.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs_final" / "06_modelling_data"

PATIENT_SOURCE = "PAE"
FOLLOW_UP_SOURCE = "IPSS"
CLINICAL_COLUMNS = {
    "Age (at date PAE)": "age_at_pae",
    "Prostate Volume (cc)": "prostate_volume_preop_cc",
    "UDS (AG)": "abrams_griffiths_number",
    "Qmax": "qmax_preop",
    "PVR": "pvr_preop",
    "IPSS preop": "ipss_preop",
}
CLINICAL_FEATURES = list(CLINICAL_COLUMNS.values())
TARGET_CLASSIFICATION = "responder_25_3m"
TARGET_REGRESSION = "ipss_3m"
TARGET_CHANGE = "ipss_change_3m"


def parse_args() -> argparse.Namespace:
    """Read command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Map the six confirmed baseline clinical variables, merge all 119 "
            "radiomics features and prepare task-specific modelling cohorts."
        )
    )
    parser.add_argument(
        "--clinical-xlsx",
        type=Path,
        default=DEFAULT_CLINICAL,
        help="Clinical workbook; column names are read from Excel row 2 by default.",
    )
    parser.add_argument(
        "--sheet-name",
        default="0",
        help="Worksheet name or zero-based worksheet index (default: 0).",
    )
    parser.add_argument(
        "--header-row",
        type=int,
        default=2,
        help="One-based Excel row containing column names (default: 2).",
    )
    parser.add_argument(
        "--radiomics",
        type=Path,
        default=DEFAULT_RADIOMICS,
        help="Step 05 full CERR radiomics CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for final modelling tables, manifests and audits.",
    )
    parser.add_argument(
        "--response-threshold-percent",
        type=float,
        default=25.0,
        help="Minimum 3-month IPSS percentage improvement for response.",
    )
    parser.add_argument(
        "--expected-radiomics-features",
        type=int,
        default=119,
        help="Required number of radiomics predictors (default: 119).",
    )
    return parser.parse_args()


def sheet_selector(value: str) -> str | int:
    """Convert a numeric worksheet selector to an integer index."""
    stripped = value.strip()
    return int(stripped) if stripped.isdigit() else stripped


def to_numeric_strict(table: pd.DataFrame, columns: list[str]) -> None:
    """Convert clinical columns while rejecting non-empty invalid text."""
    for column in columns:
        source = table[column]
        numeric = pd.to_numeric(source, errors="coerce")
        invalid = source.notna() & numeric.isna()
        if invalid.any():
            examples = source.loc[invalid].astype(str).unique()[:5].tolist()
            raise ValueError(f"Non-numeric values in clinical column {column}: {examples}")
        if np.isinf(numeric.to_numpy(float)).any():
            raise ValueError(f"Infinite values in clinical column: {column}")
        table[column] = numeric.astype(float)


def validate_clinical_ranges(table: pd.DataFrame) -> None:
    """Reject impossible non-missing baseline values."""
    ranges = {
        "age_at_pae": (18.0, 100.0),
        "prostate_volume_preop_cc": (0.0, None),
        "qmax_preop": (0.0, None),
        "pvr_preop": (0.0, None),
        "ipss_preop": (0.0, 35.0),
        "ipss_3m": (0.0, 35.0),
    }
    for column, (lower, upper) in ranges.items():
        values = table[column].dropna()
        invalid = values < lower
        if upper is not None:
            invalid |= values > upper
        if invalid.any():
            examples = values.loc[invalid].head(5).tolist()
            raise ValueError(f"Out-of-range values in {column}: {examples}")


def read_clinical_table(
    path: Path,
    sheet_name: str | int,
    header_row: int,
) -> pd.DataFrame:
    """Read row-2 headers and map the six baseline predictors."""
    if not path.is_file():
        raise FileNotFoundError(f"Clinical workbook not found: {path}")
    if header_row < 1:
        raise ValueError("--header-row must be one or greater")
    try:
        raw = pd.read_excel(
            path,
            sheet_name=sheet_name,
            header=header_row - 1,
            engine="openpyxl",
        )
    except ImportError as error:
        raise ImportError(
            "Reading the clinical workbook requires openpyxl. "
            "Install the packages in requirements.txt."
        ) from error
    raw.columns = [str(column).strip() for column in raw.columns]

    required = [PATIENT_SOURCE, *CLINICAL_COLUMNS, FOLLOW_UP_SOURCE]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"Missing required clinical columns: {missing}")
    if raw[PATIENT_SOURCE].isna().any():
        raise ValueError("Clinical table contains missing patient IDs")

    clinical = pd.DataFrame(
        {"patient_id": raw[PATIENT_SOURCE].map(normalise_patient_id)}
    )
    for source, model_column in CLINICAL_COLUMNS.items():
        clinical[model_column] = raw[source]
    clinical["ipss_3m"] = raw[FOLLOW_UP_SOURCE]
    to_numeric_strict(clinical, [*CLINICAL_FEATURES, "ipss_3m"])
    validate_clinical_ranges(clinical)

    if clinical["patient_id"].eq("").any():
        raise ValueError("Clinical table contains blank patient IDs")
    duplicate_ids = clinical.loc[
        clinical["patient_id"].duplicated(keep=False), "patient_id"
    ].unique()
    if len(duplicate_ids):
        raise ValueError(f"Duplicate clinical patient IDs: {duplicate_ids.tolist()}")
    return clinical


def read_radiomics_table(
    path: Path,
    expected_features: int,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Read the preserved Step 05 table and prefix modelling columns."""
    if not path.is_file():
        raise FileNotFoundError(f"Step 05 radiomics table not found: {path}")
    raw = pd.read_csv(path)
    if "Patient" not in raw.columns:
        raise ValueError("Step 05 radiomics table must contain a Patient column")
    if raw["Patient"].isna().any():
        raise ValueError("Radiomics table contains missing patient IDs")

    source_features = [column for column in raw.columns if column != "Patient"]
    if len(source_features) != expected_features:
        raise ValueError(
            f"Expected {expected_features} radiomics features, "
            f"found {len(source_features)}"
        )
    if len(set(source_features)) != len(source_features):
        raise ValueError("Radiomics feature names are not unique")

    radiomics_columns: dict[str, pd.Series] = {
        "patient_id": raw["Patient"].map(normalise_patient_id)
    }
    model_features: list[str] = []
    for source in source_features:
        model_column = f"radiomics__{source}"
        numeric = pd.to_numeric(raw[source], errors="coerce")
        invalid = raw[source].notna() & numeric.isna()
        if invalid.any():
            examples = raw.loc[invalid, source].astype(str).unique()[:5].tolist()
            raise ValueError(f"Non-numeric values in radiomics column {source}: {examples}")
        if np.isinf(numeric.to_numpy(float)).any():
            raise ValueError(f"Infinite values in radiomics column: {source}")
        radiomics_columns[model_column] = numeric.astype(float)
        model_features.append(model_column)

    radiomics = pd.DataFrame(radiomics_columns)

    duplicate_ids = radiomics.loc[
        radiomics["patient_id"].duplicated(keep=False), "patient_id"
    ].unique()
    if len(duplicate_ids):
        raise ValueError(f"Duplicate radiomics patient IDs: {duplicate_ids.tolist()}")
    return radiomics, source_features, model_features


def feature_manifest(
    task: str,
    source_radiomics: list[str],
    model_radiomics: list[str],
) -> pd.DataFrame:
    """Describe every identifier, predictor and task outcome column."""
    rows: list[dict[str, str]] = [
        {
            "task": task,
            "role": "identifier",
            "modality": "identifier",
            "source_column": PATIENT_SOURCE,
            "model_column": "patient_id",
            "preprocessing": "Patient-ID normalisation only",
        }
    ]
    for source, model_column in CLINICAL_COLUMNS.items():
        rows.append(
            {
                "task": task,
                "role": "predictor",
                "modality": "clinical",
                "source_column": source,
                "model_column": model_column,
                "preprocessing": "Fitted inside each training fold",
            }
        )
    for source, model_column in zip(source_radiomics, model_radiomics):
        rows.append(
            {
                "task": task,
                "role": "predictor",
                "modality": "radiomics",
                "source_column": source,
                "model_column": model_column,
                "preprocessing": "Fitted inside each training fold",
            }
        )
    rows.append(
        {
            "task": task,
            "role": "outcome_component",
            "modality": "outcome",
            "source_column": FOLLOW_UP_SOURCE,
            "model_column": "ipss_3m",
            "preprocessing": "Not used as a predictor",
        }
    )
    if task == "classification":
        rows.extend(
            [
                {
                    "task": task,
                    "role": "derived_outcome",
                    "modality": "outcome",
                    "source_column": "IPSS preop and IPSS",
                    "model_column": TARGET_CHANGE,
                    "preprocessing": "Not used as a classification predictor",
                },
                {
                    "task": task,
                    "role": "derived_outcome",
                    "modality": "outcome",
                    "source_column": "IPSS preop and IPSS",
                    "model_column": "ipss_improvement_3m_pct",
                    "preprocessing": "Not used as a predictor",
                },
                {
                    "task": task,
                    "role": "target",
                    "modality": "outcome",
                    "source_column": "IPSS preop and IPSS",
                    "model_column": TARGET_CLASSIFICATION,
                    "preprocessing": "Binary threshold applied before modelling",
                },
            ]
        )
    else:
        rows.append(
            {
                "task": task,
                "role": "target",
                "modality": "outcome",
                "source_column": "IPSS preop and IPSS",
                "model_column": TARGET_CHANGE,
                "preprocessing": "Raw baseline minus 3-month IPSS",
            }
        )
    return pd.DataFrame(rows)


def exclusion_reason(row: pd.Series, task: str) -> str:
    """Create a compact patient-level exclusion reason."""
    reasons: list[str] = []
    if not row["in_clinical_table"]:
        reasons.append("not in clinical table")
    if not row["in_radiomics_table"]:
        reasons.append("not in radiomics table")
    if not row["valid_3m_outcome"]:
        reasons.append("missing/invalid baseline or 3-month IPSS")
    if task == "regression" and not row["six_clinical_predictors_complete"]:
        reasons.append("incomplete clinical predictors")
    if task == "regression" and not row["radiomics_complete"]:
        reasons.append("incomplete radiomics predictors")
    return "; ".join(reasons) if reasons else "included"


def patient_audit(
    clinical: pd.DataFrame,
    radiomics: pd.DataFrame,
    radiomics_features: list[str],
) -> pd.DataFrame:
    """Record why each available patient enters either modelling task."""
    clinical_audit = clinical.copy()
    clinical_audit["in_clinical_table"] = True
    clinical_audit["six_clinical_predictors_complete"] = clinical_audit[
        CLINICAL_FEATURES
    ].notna().all(axis=1)
    clinical_audit["valid_3m_outcome"] = (
        clinical_audit["ipss_preop"].notna()
        & clinical_audit["ipss_3m"].notna()
        & clinical_audit["ipss_preop"].gt(0)
    )

    radiomics_audit = radiomics[["patient_id"]].copy()
    radiomics_audit["in_radiomics_table"] = True
    radiomics_audit["radiomics_complete"] = radiomics[
        radiomics_features
    ].notna().all(axis=1)

    audit = clinical_audit.merge(radiomics_audit, on="patient_id", how="outer")
    boolean_columns = [
        "in_clinical_table",
        "six_clinical_predictors_complete",
        "valid_3m_outcome",
        "in_radiomics_table",
        "radiomics_complete",
    ]
    for column in boolean_columns:
        audit[column] = audit[column].astype("boolean").fillna(False).astype(bool)
    audit["included_classification"] = (
        audit["in_clinical_table"]
        & audit["in_radiomics_table"]
        & audit["valid_3m_outcome"]
    )
    audit["included_regression"] = (
        audit["included_classification"]
        & audit["six_clinical_predictors_complete"]
        & audit["radiomics_complete"]
    )
    audit["classification_exclusion_reason"] = audit.apply(
        exclusion_reason,
        axis=1,
        task="classification",
    )
    audit["regression_exclusion_reason"] = audit.apply(
        exclusion_reason,
        axis=1,
        task="regression",
    )
    return audit[
        [
            "patient_id",
            *boolean_columns,
            "included_classification",
            "classification_exclusion_reason",
            "included_regression",
            "regression_exclusion_reason",
        ]
    ].sort_values("patient_id", kind="stable")


def build_datasets(
    clinical: pd.DataFrame,
    radiomics: pd.DataFrame,
    radiomics_features: list[str],
    response_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the outcome-valid and complete-case task cohorts."""
    merged = clinical.merge(radiomics, on="patient_id", how="inner", validate="one_to_one")
    valid_outcome = (
        merged["ipss_preop"].notna()
        & merged["ipss_3m"].notna()
        & merged["ipss_preop"].gt(0)
    )
    merged = merged.loc[valid_outcome].copy()
    merged["ipss_improvement_3m_pct"] = (
        (merged["ipss_preop"] - merged["ipss_3m"])
        / merged["ipss_preop"]
        * 100.0
    )
    merged[TARGET_CLASSIFICATION] = (
        merged["ipss_improvement_3m_pct"] >= response_threshold
    ).astype(int)
    merged[TARGET_CHANGE] = merged["ipss_preop"] - merged["ipss_3m"]

    classification_columns = [
        "patient_id",
        *CLINICAL_FEATURES,
        *radiomics_features,
        "ipss_3m",
        TARGET_CHANGE,
        "ipss_improvement_3m_pct",
        TARGET_CLASSIFICATION,
    ]
    classification = merged[classification_columns].copy()

    complete_columns = [*CLINICAL_FEATURES, *radiomics_features, TARGET_REGRESSION]
    regression = merged.loc[merged[complete_columns].notna().all(axis=1)].copy()
    regression_columns = [
        "patient_id",
        *CLINICAL_FEATURES,
        *radiomics_features,
        "ipss_3m",
        TARGET_CHANGE,
    ]
    regression = regression[regression_columns]
    return classification, regression


def main() -> None:
    """Write final task-specific datasets, manifests and QC summaries."""
    args = parse_args()
    if not np.isfinite(args.response_threshold_percent):
        raise ValueError("--response-threshold-percent must be finite")
    if args.expected_radiomics_features <= 0:
        raise ValueError("--expected-radiomics-features must be positive")

    clinical = read_clinical_table(
        args.clinical_xlsx,
        sheet_selector(args.sheet_name),
        args.header_row,
    )
    radiomics, source_radiomics, model_radiomics = read_radiomics_table(
        args.radiomics,
        args.expected_radiomics_features,
    )
    classification, regression = build_datasets(
        clinical,
        radiomics,
        model_radiomics,
        args.response_threshold_percent,
    )
    if classification.empty:
        raise ValueError("No patients have a valid classification outcome and radiomics")
    if classification[TARGET_CLASSIFICATION].nunique() != 2:
        raise ValueError("Classification cohort must contain both response classes")
    if regression.empty:
        raise ValueError("No complete patients are available for regression")

    classification_manifest = feature_manifest(
        "classification",
        source_radiomics,
        model_radiomics,
    )
    regression_manifest = feature_manifest(
        "regression",
        source_radiomics,
        model_radiomics,
    )
    audit = patient_audit(clinical, radiomics, model_radiomics)

    missingness_rows = []
    for source, model_column in CLINICAL_COLUMNS.items():
        missing = int(clinical[model_column].isna().sum())
        missingness_rows.append(
            {
                "Source column": source,
                "Model column": model_column,
                "Missing n": missing,
                "Missing %": float(missing / len(clinical) * 100.0),
            }
        )
    clinical_missingness = pd.DataFrame(missingness_rows)
    mapping = pd.DataFrame(
        [
            {"Source column": source, "Model column": model_column}
            for source, model_column in CLINICAL_COLUMNS.items()
        ]
    )

    output = args.output_dir
    atomic_csv(classification, output / "classification_dataset.csv")
    atomic_csv(regression, output / "regression_dataset.csv")
    atomic_csv(
        classification_manifest,
        output / "classification_feature_manifest.csv",
    )
    atomic_csv(regression_manifest, output / "regression_feature_manifest.csv")
    atomic_csv(audit, output / "patient_inclusion_audit.csv")
    atomic_csv(clinical_missingness, output / "clinical_predictor_missingness.csv")
    atomic_csv(mapping, output / "clinical_variable_mapping.csv")

    responders = int(classification[TARGET_CLASSIFICATION].sum())
    summary = {
        "status": "pass",
        "clinical_input": str(args.clinical_xlsx.resolve()),
        "clinical_header_row": int(args.header_row),
        "radiomics_input": str(args.radiomics.resolve()),
        "clinical_records": int(len(clinical)),
        "radiomics_records": int(len(radiomics)),
        "clinical_predictors": CLINICAL_FEATURES,
        "clinical_predictor_count": int(len(CLINICAL_FEATURES)),
        "radiomics_predictor_count": int(len(model_radiomics)),
        "classification": {
            "patients": int(len(classification)),
            "responders": responders,
            "non_responders": int(len(classification) - responders),
            "target": TARGET_CLASSIFICATION,
            "definition": (
                "100 * (baseline IPSS - 3-month IPSS) / baseline IPSS "
                f">= {args.response_threshold_percent:g}%"
            ),
            "predictor_missing_values_allowed": True,
        },
        "regression": {
            "patients": int(len(regression)),
            "target": TARGET_REGRESSION,
            "definition": "3-month IPSS (raw IPSS points)",
            "legacy_target_retained": TARGET_CHANGE,
            "complete_case_required": True,
            "target_mean": float(regression[TARGET_REGRESSION].mean()),
            "target_std": float(regression[TARGET_REGRESSION].std(ddof=1)),
        },
        "global_imputation_or_feature_selection": False,
        "note": "Any model preprocessing must be fitted inside training folds.",
    }
    atomic_json(summary, output / "modelling_data_summary.json")
    report = "\n".join(
        [
            "Final Modelling Data",
            "====================",
            f"Clinical predictors: {len(CLINICAL_FEATURES)}",
            f"Radiomics predictors: {len(model_radiomics)}",
            f"Classification patients: {len(classification)}",
            f"  Responders / non-responders: {responders} / {len(classification) - responders}",
            f"Regression complete cases: {len(regression)}",
            "No global imputation, scaling or feature selection was applied.",
            "",
        ]
    )
    atomic_text(report, output / "modelling_data_summary.txt")
    print(report, end="")
    print(f"Outputs: {output.resolve()}")


if __name__ == "__main__":
    main()
