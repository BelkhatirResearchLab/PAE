"""Run the secondary PCA-to-L1 classification analysis."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from common.feature_processing import SafeL1LogisticSelector
from common.io_utils import (
    atomic_csv,
    atomic_json,
    atomic_text,
)
from common.metrics import (
    choose_youden_threshold,
    classification_metrics,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
DEFAULT_DATA = PROJECT_ROOT / "outputs_final/06_modelling_data/classification_dataset.csv"
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "outputs_final/06_modelling_data/classification_feature_manifest.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs_final/12_secondary_classification"
TARGET = "responder_25_3m"
NON_PREDICTORS = {
    "patient_id",
    "ipss_3m",
    "ipss_change_3m",
    "ipss_improvement_3m_pct",
    TARGET,
}
REQUIRED_NON_PREDICTORS = {
    "patient_id",
    "ipss_3m",
    "ipss_improvement_3m_pct",
    TARGET,
}
EXPECTED_CLINICAL_FEATURES = [
    "age_at_pae",
    "prostate_volume_preop_cc",
    "abrams_griffiths_number",
    "qmax_preop",
    "pvr_preop",
    "ipss_preop",
]
MODEL_NAMES = ["elastic_net", "linear_svm", "random_forest", "xgboost"]
MODALITIES = ["clinical_only", "radiomics_only", "early_fusion"]
PRIMARY_MODELS = {
    "clinical_only": "inner_cv_selected",
    "radiomics_only": "inner_cv_selected",
    "early_fusion": "inner_cv_selected",
    "late_fusion": "weighted_probability",
    "stacking_fusion": "logistic_meta",
}


def arguments() -> argparse.Namespace:
    """Read command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PAE classification with nested 3-fold CV and fold-internal "
            "PCA-to-L1 radiomics reduction."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--outer-folds", type=int, default=3)
    parser.add_argument("--outer-repeats", type=int, default=1)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--pca-variance", type=float, default=0.95)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=MODEL_NAMES,
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate cross-validation settings."""
    if args.outer_folds < 2 or args.inner_folds < 2 or args.outer_repeats < 1:
        raise ValueError("CV folds must be >=2 and repeats must be >=1")
    if not 0 < args.pca_variance <= 1:
        raise ValueError("pca-variance must be in (0, 1]")


def load_data(
    data_path: Path,
    manifest_path: Path,
) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    """Load the full cohort and the authoritative predictor manifest."""
    table = pd.read_csv(data_path)
    manifest = pd.read_csv(manifest_path)
    missing = REQUIRED_NON_PREDICTORS - set(table.columns)
    if missing:
        raise ValueError(f"Modelling table is missing columns: {sorted(missing)}")
    if table["patient_id"].duplicated().any() or table["patient_id"].isna().any():
        raise ValueError("Patient IDs must be non-empty and unique")
    if not {"role", "modality", "model_column"}.issubset(manifest.columns):
        raise ValueError("Feature manifest is missing required columns")

    predictors = manifest.loc[manifest["role"] == "predictor"].copy()
    clinical = predictors.loc[
        predictors["modality"] == "clinical", "model_column"
    ].tolist()
    radiomics = predictors.loc[
        predictors["modality"] == "radiomics", "model_column"
    ].tolist()
    expected = clinical + radiomics
    if len(expected) != len(set(expected)):
        raise ValueError("Predictor names in the manifest are not unique")
    missing_predictors = set(expected) - set(table.columns)
    if missing_predictors:
        raise ValueError(f"Missing predictors: {sorted(missing_predictors)}")
    unexpected = set(table.columns) - NON_PREDICTORS - set(expected)
    if unexpected:
        raise ValueError(f"Unclassified modelling columns: {sorted(unexpected)}")
    if clinical != EXPECTED_CLINICAL_FEATURES:
        raise ValueError("Clinical predictors differ from the final six-variable list")
    if len(radiomics) != 119:
        raise ValueError(f"Expected all 119 radiomics features, found {len(radiomics)}")

    y = pd.to_numeric(table[TARGET], errors="raise").to_numpy(int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Classification target must contain both 0 and 1")
    x = table[expected].apply(pd.to_numeric, errors="raise")
    if np.isinf(x.to_numpy(float)).any():
        raise ValueError("Predictors contain infinite values")
    all_missing = x.columns[x.isna().all()].tolist()
    if all_missing:
        raise ValueError(f"All-missing predictors: {all_missing}")
    return table, y, clinical, radiomics


def clinical_preprocessor(
    features: list[str],
    scale: bool,
) -> ColumnTransformer:
    """Impute clinical values and optionally standardise them."""
    numeric = list(features)
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def radiomics_preprocessor(args: argparse.Namespace) -> Pipeline:
    """Fit variance removal, PCA and L1 selection inside each training fold."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("variance", VarianceThreshold(0.0)),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=args.pca_variance, svd_solver="full")),
            (
                "selector",
                SafeL1LogisticSelector(
                    cv_splits=args.inner_folds,
                    random_state=args.random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )


def modality_preprocessor(
    modality: str,
    clinical_features: list[str],
    radiomics_features: list[str],
    scale: bool,
    args: argparse.Namespace,
):
    """Build modality-specific, fold-fitted preprocessing."""
    if modality == "clinical_only":
        return clinical_preprocessor(clinical_features, scale)
    if modality == "radiomics_only":
        return radiomics_preprocessor(args)
    if modality == "early_fusion":
        return ColumnTransformer(
            [
                (
                    "clinical",
                    clinical_preprocessor(clinical_features, scale),
                    clinical_features,
                ),
                (
                    "radiomics",
                    radiomics_preprocessor(args),
                    radiomics_features,
                ),
            ],
            remainder="drop",
            sparse_threshold=0,
        )
    raise ValueError(f"Unknown modality: {modality}")


def candidate_model(
    name: str,
    modality: str,
    clinical_features: list[str],
    radiomics_features: list[str],
    y_train: np.ndarray,
    args: argparse.Namespace,
) -> tuple[Pipeline, dict[str, list[object]]]:
    """Return one classifier and a compact inner-CV parameter grid."""
    is_linear = name in {"elastic_net", "linear_svm"}
    preprocess = modality_preprocessor(
        modality,
        clinical_features,
        radiomics_features,
        scale=is_linear,
        args=args,
    )
    if name == "elastic_net":
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            class_weight="balanced",
            max_iter=10000,
            random_state=args.random_state,
        )
        grid = {
            "model__C": [0.03, 0.10, 0.30],
            "model__l1_ratio": [0.25, 0.50],
        }
    elif name == "linear_svm":
        model = SVC(
            kernel="linear",
            probability=True,
            class_weight="balanced",
            random_state=args.random_state,
        )
        grid = {"model__C": [0.03, 0.10, 0.30]}
    elif name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=args.random_state,
        )
        grid = {
            "model__max_depth": [2, 3],
            "model__min_samples_leaf": [5, 10],
        }
    elif name == "xgboost":
        negative = int((y_train == 0).sum())
        positive = int((y_train == 1).sum())
        model = XGBClassifier(
            n_estimators=150,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=10.0,
            scale_pos_weight=negative / positive,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=args.random_state,
        )
        grid = {
            "model__max_depth": [1, 2],
            "model__min_child_weight": [5, 10],
        }
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("preprocess", preprocess), ("model", model)]), grid


def reduction_audit(estimator: Pipeline, modality: str) -> dict[str, object]:
    """Record the fitted PCA-to-L1 reduction for one outer fold."""
    if modality == "clinical_only":
        return {
            "features_after_variance": np.nan,
            "pca_components": np.nan,
            "pca_explained_variance": np.nan,
            "selected_components": np.nan,
            "selector_c": np.nan,
        }
    preprocessor = estimator.named_steps["preprocess"]
    radiomics = (
        preprocessor
        if modality == "radiomics_only"
        else preprocessor.named_transformers_["radiomics"]
    )
    variance = radiomics.named_steps["variance"]
    pca = radiomics.named_steps["pca"]
    selector = radiomics.named_steps["selector"]
    return {
        "features_after_variance": int(variance.get_support().sum()),
        "pca_components": int(pca.n_components_),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "selected_components": int(selector.get_support().sum()),
        "selector_c": float(selector.selected_c_),
    }


def fit_candidate(
    model_name: str,
    modality: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_validation: pd.DataFrame,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    clinical_features: list[str],
    radiomics_features: list[str],
    args: argparse.Namespace,
) -> dict[str, object]:
    """Tune on inner CV and score the untouched outer fold."""
    estimator, grid = candidate_model(
        model_name,
        modality,
        clinical_features,
        radiomics_features,
        y_train,
        args,
    )
    search = GridSearchCV(
        estimator,
        grid,
        scoring="roc_auc",
        cv=inner_splits,
        n_jobs=args.n_jobs,
        refit=True,
        error_score="raise",
    ).fit(x_train, y_train)
    inner_scores = cross_val_predict(
        clone(search.best_estimator_),
        x_train,
        y_train,
        cv=inner_splits,
        method="predict_proba",
        n_jobs=args.n_jobs,
    )[:, 1]
    threshold = choose_youden_threshold(y_train, inner_scores)
    return {
        "inner_scores": inner_scores,
        "inner_auc": float(roc_auc_score(y_train, inner_scores)),
        "outer_scores": search.best_estimator_.predict_proba(x_validation)[:, 1],
        "threshold": float(threshold),
        "best_params": search.best_params_,
        "reduction_audit": reduction_audit(search.best_estimator_, modality),
    }


def metrics_from_predictions(
    y_true: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float | int]:
    """Calculate pooled metrics when folds use training-selected thresholds."""
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "auc": float(roc_auc_score(y_true, scores)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def input_feature_counts(
    modality: str,
    clinical_count: int,
    radiomics_count: int,
) -> dict[str, int]:
    """Record that all available features enter each relevant model."""
    use_clinical = modality in {
        "clinical_only",
        "early_fusion",
        "late_fusion",
        "stacking_fusion",
    }
    use_radiomics = modality in {
        "radiomics_only",
        "early_fusion",
        "late_fusion",
        "stacking_fusion",
    }
    clinical_used = clinical_count if use_clinical else 0
    radiomics_used = radiomics_count if use_radiomics else 0
    return {
        "clinical_features_entered": clinical_used,
        "radiomics_features_entered": radiomics_used,
        "total_features_entered": clinical_used + radiomics_used,
    }


def add_result(
    *,
    patient_ids: list[str],
    y_validation: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    repeat: int,
    fold: int,
    modality: str,
    model: str,
    inner_auc: float,
    best_params: dict[str, object],
    clinical_count: int,
    radiomics_count: int,
    prediction_rows: list[dict[str, object]],
    fold_rows: list[dict[str, object]],
) -> None:
    """Record patient predictions and outer-fold performance."""
    predicted = (scores >= threshold).astype(int)
    configuration = f"{modality}__{model}"
    for patient_id, truth, score, prediction in zip(
        patient_ids,
        y_validation,
        scores,
        predicted,
    ):
        prediction_rows.append(
            {
                "patient_id": patient_id,
                "y_true": int(truth),
                "repeat": repeat,
                "fold": fold,
                "modality": modality,
                "model": model,
                "configuration": configuration,
                "score": float(score),
                "threshold": float(threshold),
                "predicted_class": int(prediction),
            }
        )
    fold_rows.append(
        {
            "repeat": repeat,
            "fold": fold,
            "modality": modality,
            "model": model,
            "configuration": configuration,
            "validation_patients": len(y_validation),
            "inner_cv_auc": float(inner_auc),
            "best_params": json.dumps(best_params, sort_keys=True),
            **input_feature_counts(modality, clinical_count, radiomics_count),
            **classification_metrics(y_validation, scores, threshold),
        }
    )


def select_modality_model(
    fold_results: dict[tuple[str, str], dict[str, object]],
    modality: str,
    model_order: list[str],
) -> tuple[str, dict[str, object]]:
    """Select one algorithm using inner-training OOF AUC only."""
    available = [
        (model, fold_results[(modality, model)])
        for model in model_order
        if (modality, model) in fold_results
    ]
    return max(
        available,
        key=lambda item: (item[1]["inner_auc"], -model_order.index(item[0])),
    )


def weighted_late_fusion(
    clinical: dict[str, object],
    radiomics: dict[str, object],
    y_train: np.ndarray,
) -> dict[str, object]:
    """Select the probability weight from training OOF predictions."""
    candidates = np.linspace(0, 1, 21)
    scored = []
    for clinical_weight in candidates:
        scores = (
            clinical_weight * clinical["inner_scores"]
            + (1 - clinical_weight) * radiomics["inner_scores"]
        )
        scored.append(
            (
                float(roc_auc_score(y_train, scores)),
                float(clinical_weight),
                scores,
            )
        )
    inner_auc, clinical_weight, inner_scores = max(
        scored,
        key=lambda item: (item[0], -abs(item[1] - 0.5)),
    )
    return {
        "inner_auc": inner_auc,
        "inner_scores": inner_scores,
        "outer_scores": (
            clinical_weight * clinical["outer_scores"]
            + (1 - clinical_weight) * radiomics["outer_scores"]
        ),
        "threshold": choose_youden_threshold(y_train, inner_scores),
        "best_params": {
            "clinical_weight": clinical_weight,
            "radiomics_weight": 1 - clinical_weight,
        },
        "reduction_audit": radiomics["reduction_audit"],
    }


def stacking_fusion(
    clinical: dict[str, object],
    radiomics: dict[str, object],
    y_train: np.ndarray,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> dict[str, object]:
    """Fit a regularised meta-model to training OOF probabilities."""
    x_meta = np.column_stack(
        [clinical["inner_scores"], radiomics["inner_scores"]]
    )
    x_outer = np.column_stack(
        [clinical["outer_scores"], radiomics["outer_scores"]]
    )
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=10000,
                    random_state=args.random_state,
                ),
            ),
        ]
    )
    search = GridSearchCV(
        pipeline,
        {"model__C": [0.01, 0.03, 0.10, 0.30, 1.0]},
        scoring="roc_auc",
        cv=inner_splits,
        n_jobs=args.n_jobs,
        refit=True,
        error_score="raise",
    ).fit(x_meta, y_train)
    inner_scores = cross_val_predict(
        clone(search.best_estimator_),
        x_meta,
        y_train,
        cv=inner_splits,
        method="predict_proba",
        n_jobs=args.n_jobs,
    )[:, 1]
    coefficients = search.best_estimator_.named_steps["model"].coef_[0]
    return {
        "inner_auc": float(roc_auc_score(y_train, inner_scores)),
        "inner_scores": inner_scores,
        "outer_scores": search.best_estimator_.predict_proba(x_outer)[:, 1],
        "threshold": choose_youden_threshold(y_train, inner_scores),
        "best_params": {
            **search.best_params_,
            "clinical_coefficient": float(coefficients[0]),
            "radiomics_coefficient": float(coefficients[1]),
        },
        "reduction_audit": radiomics["reduction_audit"],
    }


def build_repeat_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Pool complete OOF predictions within each repeat."""
    rows = []
    for (configuration, repeat), group in predictions.groupby(
        ["configuration", "repeat"],
        sort=True,
    ):
        if group["patient_id"].duplicated().any():
            raise ValueError(f"Duplicate OOF patients in {configuration}, repeat {repeat}")
        values = metrics_from_predictions(
            group["y_true"].to_numpy(int),
            group["score"].to_numpy(float),
            group["predicted_class"].to_numpy(int),
        )
        rows.append(
            {
                "configuration": configuration,
                "modality": group["modality"].iloc[0],
                "model": group["model"].iloc[0],
                "repeat": int(repeat),
                "patients": len(group),
                "mean_training_selected_threshold": float(group["threshold"].mean()),
                **values,
            }
        )
    return pd.DataFrame(rows)


def summarise_results(
    repeat_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise pooled cross-validation performance."""
    rows = []
    for configuration, group in repeat_metrics.groupby("configuration", sort=True):
        folds = fold_metrics[fold_metrics["configuration"] == configuration]
        row = {
            "configuration": configuration,
            "modality": group["modality"].iloc[0],
            "model": group["model"].iloc[0],
            "patients": int(group["patients"].iloc[0]),
            "outer_repeats": len(group),
            "outer_folds_total": len(folds),
            "clinical_features_entered": int(
                folds["clinical_features_entered"].iloc[0]
            ),
            "radiomics_features_entered": int(
                folds["radiomics_features_entered"].iloc[0]
            ),
            "mean_training_selected_threshold": float(
                group["mean_training_selected_threshold"].mean()
            ),
        }
        for metric in ["accuracy", "sensitivity", "specificity", "auc"]:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = (
                float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            )
        row["fold_auc_mean"] = float(folds["auc"].mean())
        row["fold_auc_sd"] = float(folds["auc"].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("auc_mean", ascending=False)


def primary_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Return the nested model-selection result for each modality."""
    selected = table[
        table.apply(
            lambda row: PRIMARY_MODELS.get(row["modality"]) == row["model"],
            axis=1,
        )
    ].copy()
    order = {name: index for index, name in enumerate(PRIMARY_MODELS)}
    selected["display_order"] = selected["modality"].map(order)
    return selected.sort_values("display_order").drop(columns="display_order")


def save_plot(primary: pd.DataFrame, path: Path) -> None:
    """Plot accuracy, sensitivity, specificity, and AUC."""
    labels = {
        "clinical_only": "Clinical",
        "radiomics_only": "Radiomics",
        "early_fusion": "Early fusion",
        "late_fusion": "Late fusion",
        "stacking_fusion": "Stacking",
    }
    shown = primary.copy()
    shown["label"] = shown["modality"].map(labels)
    metrics = ["accuracy_mean", "sensitivity_mean", "specificity_mean", "auc_mean"]
    names = ["Accuracy", "Sensitivity", "Specificity", "AUC"]
    x_positions = np.arange(len(shown))
    width = 0.18
    fig, axis = plt.subplots(figsize=(12, 6))
    for index, (metric, name) in enumerate(zip(metrics, names)):
        values = shown[metric].to_numpy(float)
        bars = axis.bar(
            x_positions + (index - 1.5) * width,
            values,
            width,
            label=name,
        )
        axis.bar_label(
            bars,
            labels=[f"{value:.2f}" for value in values],
            fontsize=7,
        )
    axis.set_xticks(x_positions, shown["label"])
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Cross-validated performance")
    axis.set_title("Full-Cohort Nested Cross-Validation Performance")
    axis.legend(ncol=4, loc="upper center")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid4().hex}.tmp{path.suffix}")
    try:
        fig.savefig(temporary, dpi=220, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(fig)


def main() -> None:
    """Run candidate and fusion classifiers under nested CV."""
    args = arguments()
    validate_arguments(args)
    table, y, clinical_features, radiomics_features = load_data(
        args.data,
        args.manifest,
    )
    class_counts = pd.Series(y).value_counts()
    if args.outer_folds > int(class_counts.min()):
        raise ValueError("outer-folds exceeds the smallest class count")

    predictor_columns = clinical_features + radiomics_features
    x = table[predictor_columns]
    patient_ids = table["patient_id"].astype(str).tolist()
    outer_cv = RepeatedStratifiedKFold(
        n_splits=args.outer_folds,
        n_repeats=args.outer_repeats,
        random_state=args.random_state,
    )
    outer_splits = list(outer_cv.split(x, y))
    prediction_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []

    for split_number, (training, validation) in enumerate(outer_splits, start=1):
        repeat = (split_number - 1) // args.outer_folds + 1
        fold = (split_number - 1) % args.outer_folds + 1
        x_train = x.iloc[training]
        y_train = y[training]
        x_validation = x.iloc[validation]
        y_validation = y[validation]
        validation_ids = [patient_ids[index] for index in validation]
        if args.inner_folds > int(pd.Series(y_train).value_counts().min()):
            raise ValueError("inner-folds exceeds the smallest outer-training class")
        inner_cv = StratifiedKFold(
            n_splits=args.inner_folds,
            shuffle=True,
            random_state=args.random_state + split_number,
        )
        inner_splits = list(inner_cv.split(x_train, y_train))
        fold_results: dict[tuple[str, str], dict[str, object]] = {}

        for modality in MODALITIES:
            modality_columns = (
                clinical_features
                if modality == "clinical_only"
                else radiomics_features
                if modality == "radiomics_only"
                else predictor_columns
            )
            for model_name in args.models:
                print(
                    f"[{split_number}/{len(outer_splits)}] "
                    f"{modality} / {model_name}",
                    flush=True,
                )
                result = fit_candidate(
                    model_name,
                    modality,
                    x_train[modality_columns],
                    y_train,
                    x_validation[modality_columns],
                    inner_splits,
                    clinical_features,
                    radiomics_features,
                    args,
                )
                fold_results[(modality, model_name)] = result
                add_result(
                    patient_ids=validation_ids,
                    y_validation=y_validation,
                    scores=result["outer_scores"],
                    threshold=result["threshold"],
                    repeat=repeat,
                    fold=fold,
                    modality=modality,
                    model=model_name,
                    inner_auc=result["inner_auc"],
                    best_params=result["best_params"],
                    clinical_count=len(clinical_features),
                    radiomics_count=len(radiomics_features),
                    prediction_rows=prediction_rows,
                    fold_rows=fold_rows,
                )
                parameter_rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "modality": modality,
                        "model": model_name,
                        "inner_cv_auc": result["inner_auc"],
                        "training_selected_threshold": result["threshold"],
                        "best_params": json.dumps(
                            result["best_params"],
                            sort_keys=True,
                        ),
                        **result["reduction_audit"],
                    }
                )

        selected_results: dict[str, dict[str, object]] = {}
        for modality in MODALITIES:
            selected_model, selected = select_modality_model(
                fold_results,
                modality,
                args.models,
            )
            selected_results[modality] = selected
            selected_params = {
                "selected_algorithm": selected_model,
                **selected["best_params"],
            }
            add_result(
                patient_ids=validation_ids,
                y_validation=y_validation,
                scores=selected["outer_scores"],
                threshold=selected["threshold"],
                repeat=repeat,
                fold=fold,
                modality=modality,
                model="inner_cv_selected",
                inner_auc=selected["inner_auc"],
                best_params=selected_params,
                clinical_count=len(clinical_features),
                radiomics_count=len(radiomics_features),
                prediction_rows=prediction_rows,
                fold_rows=fold_rows,
            )
            parameter_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "modality": modality,
                    "model": "inner_cv_selected",
                    "inner_cv_auc": selected["inner_auc"],
                    "training_selected_threshold": selected["threshold"],
                    "best_params": json.dumps(selected_params, sort_keys=True),
                    **selected["reduction_audit"],
                }
            )

        clinical_selected = selected_results["clinical_only"]
        radiomics_selected = selected_results["radiomics_only"]
        late = weighted_late_fusion(clinical_selected, radiomics_selected, y_train)
        stack = stacking_fusion(
            clinical_selected,
            radiomics_selected,
            y_train,
            inner_splits,
            args,
        )
        for modality, model, result in [
            ("late_fusion", "weighted_probability", late),
            ("stacking_fusion", "logistic_meta", stack),
        ]:
            add_result(
                patient_ids=validation_ids,
                y_validation=y_validation,
                scores=result["outer_scores"],
                threshold=result["threshold"],
                repeat=repeat,
                fold=fold,
                modality=modality,
                model=model,
                inner_auc=result["inner_auc"],
                best_params=result["best_params"],
                clinical_count=len(clinical_features),
                radiomics_count=len(radiomics_features),
                prediction_rows=prediction_rows,
                fold_rows=fold_rows,
            )
            parameter_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "modality": modality,
                    "model": model,
                    "inner_cv_auc": result["inner_auc"],
                    "training_selected_threshold": result["threshold"],
                    "best_params": json.dumps(result["best_params"], sort_keys=True),
                    **result["reduction_audit"],
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    parameters = pd.DataFrame(parameter_rows)
    repeat_metrics = build_repeat_metrics(predictions)
    results = summarise_results(repeat_metrics, fold_metrics)
    primary = primary_rows(results)

    incremental_rows = []
    primary_repeat = primary_rows(repeat_metrics)
    for repeat in sorted(primary_repeat["repeat"].unique()):
        current = primary_repeat[
            primary_repeat["repeat"] == repeat
        ].set_index("modality")
        clinical_auc = float(current.loc["clinical_only", "auc"])
        for modality in [
            "radiomics_only",
            "early_fusion",
            "late_fusion",
            "stacking_fusion",
        ]:
            comparison_auc = float(current.loc[modality, "auc"])
            incremental_rows.append(
                {
                    "repeat": int(repeat),
                    "comparison": f"{modality}_vs_clinical_only",
                    "clinical_auc": clinical_auc,
                    "comparison_auc": comparison_auc,
                    "auc_difference": comparison_auc - clinical_auc,
                }
            )
    incremental = pd.DataFrame(incremental_rows)
    radiomics_list = pd.DataFrame(
        {
            "radiomics_feature": radiomics_features,
            "entered_fold_internal_pipeline": True,
            "selected_component_not_original_feature": True,
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(predictions, args.output_dir / "classification_cv_predictions.csv")
    atomic_csv(fold_metrics, args.output_dir / "classification_cv_fold_metrics.csv")
    atomic_csv(repeat_metrics, args.output_dir / "classification_cv_repeat_metrics.csv")
    atomic_csv(results, args.output_dir / "classification_cv_all_results.csv")
    atomic_csv(primary, args.output_dir / "classification_cv_primary_results.csv")
    atomic_csv(
        parameters,
        args.output_dir / "classification_cv_selected_parameters.csv",
    )
    atomic_csv(
        incremental,
        args.output_dir / "classification_cv_incremental_auc.csv",
    )
    atomic_csv(
        radiomics_list,
        args.output_dir / "classification_cv_radiomics_features_used.csv",
    )
    save_plot(primary, args.output_dir / "classification_cv_performance.png")

    best = primary.sort_values("auc_mean", ascending=False).iloc[0]
    summary = {
        "status": "pass",
        "analysis": "Secondary PCA-to-L1 nested stratified classification",
        "patients": len(table),
        "class_0": int((y == 0).sum()),
        "class_1": int((y == 1).sum()),
        "clinical_predictors": clinical_features,
        "radiomics_predictors": len(radiomics_features),
        "all_radiomics_features_entered_pipeline": True,
        "feature_selection_applied": True,
        "pca_variance_target": args.pca_variance,
        "outer_folds": args.outer_folds,
        "outer_repeats": args.outer_repeats,
        "inner_folds": args.inner_folds,
        "candidate_models": args.models,
        "fixed_train_test_split": False,
        "fold_internal_preprocessing": (
            "Radiomics median imputation, zero-variance removal, Z-score scaling, "
            "PCA and L1-logistic component selection were fitted inside training "
            "folds. Clinical imputation and model-specific scaling were also "
            "fitted inside training folds."
        ),
        "threshold_selection": (
            "Youden threshold from inner-training OOF predictions only"
        ),
        "best_observed_primary_configuration": best["configuration"],
        "best_observed_primary_auc": float(best["auc_mean"]),
        "interpretation": (
            "All patients contribute to out-of-fold evaluation. Results are internal "
            "cross-validation estimates, not independent external-test performance."
        ),
    }
    atomic_json(summary, args.output_dir / "classification_cv_summary.json")
    atomic_json(summary, args.output_dir / "summary.json")

    lines = [
        "Classification CV-Only Summary",
        "==============================",
        f"Patients: {len(table)} (class 0/1: {(y == 0).sum()}/{(y == 1).sum()})",
        f"Outer CV: {args.outer_folds}-fold x {args.outer_repeats} repeat(s)",
        f"Inner tuning CV: {args.inner_folds}-fold",
        f"Clinical predictors: {len(clinical_features)}",
        f"Radiomics predictors entering every relevant model: {len(radiomics_features)}",
        (
            "Radiomics reduction: variance removal -> Z-score -> "
            f"PCA ({args.pca_variance:.0%}) -> L1 component selection"
        ),
        "Fixed train/test split: no",
        "",
        "Primary nested-CV results:",
    ]
    for row in primary.itertuples(index=False):
        lines.append(
            f"  {row.modality}: Accuracy={row.accuracy_mean:.3f}, "
            f"Sensitivity={row.sensitivity_mean:.3f}, "
            f"Specificity={row.specificity_mean:.3f}, AUC={row.auc_mean:.3f}"
        )
    lines.extend(
        [
            "",
            f"Best observed primary configuration: {best['configuration']}",
            f"Best observed primary AUC: {best['auc_mean']:.3f}",
            "",
            "These are internal out-of-fold estimates, not independent-test results.",
        ]
    )
    report = "\n".join(lines) + "\n"
    atomic_text(report, args.output_dir / "classification_cv_summary.txt")
    print("\n" + report)


if __name__ == "__main__":
    main()
