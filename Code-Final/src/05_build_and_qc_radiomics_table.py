"""Build the complete 119-feature CERR table and descriptive QC files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common.io_utils import atomic_csv, atomic_json, atomic_text, normalise_patient_id


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "outputs_final"
    / "radiomics_full"
    / "radiomics_features_CERR_full.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs_final" / "05_radiomics_qc"
PATIENT_ALIASES = ("Patient", "patient_id", "patient", "PAE")


def parse_args() -> argparse.Namespace:
    """Read command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate the complete CERR radiomics table and report constants, "
            "missingness and exact duplicate columns without deleting features."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="CERR CSV containing one patient column and all radiomics features.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for the preserved full table and QC reports.",
    )
    parser.add_argument(
        "--expected-features",
        type=int,
        default=119,
        help="Required number of radiomics columns (default: 119).",
    )
    return parser.parse_args()


def prepare_table(
    path: Path,
    expected_features: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Load the table, normalise IDs and validate numeric feature values."""
    if not path.is_file():
        raise FileNotFoundError(f"Radiomics table not found: {path}")
    table = pd.read_csv(path)
    if table.empty:
        raise ValueError("Radiomics table is empty")
    if table.columns.duplicated().any():
        duplicates = table.columns[table.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names: {duplicates}")

    patient_matches = [name for name in PATIENT_ALIASES if name in table.columns]
    if len(patient_matches) != 1:
        raise ValueError(
            f"Expected one patient column named {PATIENT_ALIASES}; "
            f"found {patient_matches}"
        )
    patient_column = patient_matches[0]
    if table[patient_column].isna().any():
        raise ValueError("Radiomics table contains missing patient IDs")

    table = table.rename(columns={patient_column: "Patient"}).copy()
    table["Patient"] = table["Patient"].map(normalise_patient_id)
    if table["Patient"].eq("").any():
        raise ValueError("Radiomics table contains blank patient IDs")
    duplicate_ids = table.loc[
        table["Patient"].duplicated(keep=False), "Patient"
    ].unique()
    if len(duplicate_ids):
        raise ValueError(f"Duplicate patient IDs: {duplicate_ids.tolist()}")

    features = [column for column in table.columns if column != "Patient"]
    if expected_features <= 0:
        raise ValueError("--expected-features must be positive")
    if len(features) != expected_features:
        raise ValueError(
            f"Expected {expected_features} radiomics features, found {len(features)}"
        )

    for feature in features:
        source = table[feature]
        numeric = pd.to_numeric(source, errors="coerce")
        invalid = source.notna() & numeric.isna()
        if invalid.any():
            examples = source.loc[invalid].astype(str).unique()[:5].tolist()
            raise ValueError(f"Non-numeric values in {feature}: {examples}")
        if np.isinf(numeric.to_numpy(float)).any():
            raise ValueError(f"Infinite values in radiomics feature: {feature}")
        table[feature] = numeric.astype(float)

    return table[["Patient", *features]], features


def feature_qc(table: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Calculate the requested per-feature descriptive statistics."""
    rows: list[dict[str, object]] = []
    for feature in features:
        values = table[feature]
        has_value = bool(values.notna().any())
        rows.append(
            {
                "Feature": feature,
                "Variance": float(values.var(ddof=0)) if has_value else np.nan,
                "Unique values": int(values.nunique(dropna=True)),
                "Missing %": float(values.isna().mean() * 100.0),
                "Min": float(values.min()) if has_value else np.nan,
                "Max": float(values.max()) if has_value else np.nan,
            }
        )
    return pd.DataFrame(rows)


def exact_duplicates(table: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Find identical columns while treating paired missing values as equal."""
    rows: list[dict[str, str]] = []
    for index, duplicate in enumerate(features):
        duplicate_values = table[duplicate]
        for reference in features[:index]:
            reference_values = table[reference]
            same = duplicate_values.eq(reference_values) | (
                duplicate_values.isna() & reference_values.isna()
            )
            if bool(same.all()):
                rows.append(
                    {
                        "Duplicate feature": duplicate,
                        "Matches feature": reference,
                    }
                )
                break
    return pd.DataFrame(
        rows,
        columns=["Duplicate feature", "Matches feature"],
    )


def main() -> None:
    """Create the preserved table and QC outputs."""
    args = parse_args()
    table, features = prepare_table(args.input, args.expected_features)
    qc = feature_qc(table, features)
    constants = qc.loc[qc["Unique values"] <= 1].reset_index(drop=True)
    duplicates = exact_duplicates(table, features)

    output = args.output_dir
    atomic_csv(table, output / "radiomics_features_CERR_full.csv")
    atomic_csv(qc, output / "radiomics_feature_qc.csv")
    atomic_csv(constants, output / "constant_features.csv")
    atomic_csv(duplicates, output / "exact_duplicate_features.csv")

    summary = {
        "status": "pass",
        "input": str(args.input.resolve()),
        "patients": int(len(table)),
        "radiomics_features": int(len(features)),
        "features_with_missing_values": int((qc["Missing %"] > 0).sum()),
        "constant_features": int(len(constants)),
        "exact_duplicate_features": int(len(duplicates)),
        "variance_definition": "Population variance across available patients (ddof=0)",
        "feature_selection_applied": False,
        "note": (
            "All 119 features are preserved. Constant and exact duplicate "
            "features are reported for audit only."
        ),
    }
    atomic_json(summary, output / "radiomics_qc_summary.json")
    report = "\n".join(
        [
            "Complete CERR Radiomics QC",
            "===========================",
            f"Patients: {len(table)}",
            f"Features preserved: {len(features)}",
            f"Constant features reported: {len(constants)}",
            f"Exact duplicate features reported: {len(duplicates)}",
            "No global feature selection was applied.",
            "",
        ]
    )
    atomic_text(report, output / "radiomics_qc_summary.txt")
    print(report, end="")
    print(f"Outputs: {output.resolve()}")


if __name__ == "__main__":
    main()
