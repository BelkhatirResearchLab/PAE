"""Run baseline multimodal regression for the selected IPSS endpoint.

All preprocessing, model selection, and fusion are learned without the
patient held out by the outer leave-one-out cross-validation loop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from joblib import parallel_config
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from common.feature_processing import SafeLassoSelector
from common.regression_data import (
    DEFAULT_DATA as FROZEN_DATA, TARGET, OUTCOMES, CLINICAL_FEATURES,
    add_regression_arguments, load_regression_data, prepare_run, run_metadata,
    finish_run, read_checkpoint, validate_oof, validate_dependency, run_stage,
    output_root, STAGE_FOLDERS,
)

from common.io_utils import atomic_csv, atomic_json, atomic_text

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
DEFAULT_DATA = FROZEN_DATA
DEFAULT_OUTPUT = output_root() / STAGE_FOLDERS["07"]
MODEL_ORDER = ["ridge", "svr", "random_forest", "xgboost"]
RIDGE_ALPHAS = np.logspace(-4, 4, 17)
META_ALPHAS = np.logspace(-4, 3, 15)


def parse_arguments():
    """Read command-line options."""
    parser = argparse.ArgumentParser(
        description="Baseline nested-LOOCV multimodal Ridge regression."
    )
    add_regression_arguments(parser)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-patients", type=int, default=148)
    parser.add_argument("--expected-radiomics-features", type=int, default=119)
    parser.add_argument("--pca-variance", type=float, default=0.95)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--models", default="ridge")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--max-outer-folds", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def select_models(raw):
    """Validate comma-separated model names."""
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = sorted(set(requested) - set(MODEL_ORDER))
    if not requested or invalid:
        raise ValueError(f"Invalid models: {invalid or raw}")
    return [name for name in MODEL_ORDER if name in requested]


def load_data(args):
    return load_regression_data(
        args.data, args.expected_patients, args.expected_radiomics_features,
        target=args.target, feature_manifest=args.feature_manifest,
    )


def make_reducer(args, seed):
    """Create the final fold-fitted radiomics chain."""
    return Pipeline(
        [
            ("variance", VarianceThreshold(0.0)),
            ("scaler", StandardScaler()),
            ("pca", PCA(args.pca_variance, svd_solver="full")),
            (
                "selector",
                SafeLassoSelector(
                    cv_splits=args.inner_folds,
                    random_state=seed,
                    n_jobs=args.n_jobs,
                ),
            ),
        ]
    )


def model_and_grid(name, seed):
    """Return one candidate regressor and its fixed search grid."""
    if name == "ridge":
        return Ridge(), {"model__alpha": [0.1, 1.0, 10.0, 100.0]}
    if name == "svr":
        return SVR(kernel="rbf"), {
            "model__C": [0.1, 1.0, 10.0],
            "model__epsilon": [0.1, 1.0],
        }
    if name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=300, random_state=seed, n_jobs=1
        )
        return model, {
            "model__max_depth": [3, None],
            "model__min_samples_leaf": [2, 5],
        }
    if name == "xgboost":
        if XGBRegressor is None:
            raise ImportError("Install xgboost to run the XGBoost candidate")
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=250,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_lambda=10.0,
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )
        return model, {
            "model__max_depth": [1, 2],
            "model__learning_rate": [0.03, 0.1],
        }
    raise ValueError(f"Unknown model: {name}")


def build_candidate_pipeline(modality, model_name, radio_count, args, seed):
    """Build one clinical, radiomics, or early-fusion pipeline."""
    model, grid = model_and_grid(model_name, seed)
    if modality == "clinical":
        steps = [("preprocess", StandardScaler()), ("model", model)]
    elif modality == "radiomics":
        steps = [("preprocess", make_reducer(args, seed)), ("model", model)]
    else:
        clinical_indices = list(range(len(CLINICAL_FEATURES)))
        radio_indices = list(
            range(len(CLINICAL_FEATURES), len(CLINICAL_FEATURES) + radio_count)
        )
        preprocess = ColumnTransformer(
            [
                ("clinical", StandardScaler(), clinical_indices),
                ("radiomics", make_reducer(args, seed), radio_indices),
            ]
        )
        steps = [("preprocess", preprocess), ("model", model)]
    return Pipeline(steps), grid


def ridge_base_pipeline(modality, args, seed):
    """Build a base model for weighted Late Fusion or Stacking."""
    inner = KFold(args.inner_folds, shuffle=True, random_state=seed)
    ridge = RidgeCV(alphas=RIDGE_ALPHAS, cv=inner, scoring="neg_mean_absolute_error")
    if modality == "clinical":
        return Pipeline([("scaler", StandardScaler()), ("ridge", ridge)])
    steps = list(make_reducer(args, seed).steps)
    steps.append(("ridge", ridge))
    return Pipeline(steps)


def learn_late_weight(clinical_prediction, radiomics_prediction, outcome):
    """Learn a constrained clinical weight from training OOF predictions."""
    difference = clinical_prediction - radiomics_prediction
    denominator = float(np.dot(difference, difference))
    if denominator <= np.finfo(float).eps:
        return 0.5
    weight = np.dot(difference, outcome - radiomics_prediction) / denominator
    return float(np.clip(weight, 0.0, 1.0))


def fit_reduction_audit(x_train, y_train, features, args, fold, patient):
    """Record every dimensionality-reduction decision for one outer fold."""
    transform = make_reducer(args, args.random_state + fold).fit(x_train, y_train)
    variance = transform.named_steps["variance"]
    retained_mask = variance.get_support()
    retained_names = np.asarray(features)[retained_mask]
    pca = transform.named_steps["pca"]
    selector = transform.named_steps["selector"]
    selected_indices = np.flatnonzero(selector.get_support())
    coefficients = np.asarray(selector.estimator_.coef_, dtype=float)
    raw_support = np.abs(coefficients) > 1e-10
    raw_selected = int(raw_support.sum())
    used_fallback = not raw_support.any()

    audit = {
        "outer_fold": fold,
        "held_out_patient": patient,
        "radiomics_input_count": len(features),
        "after_variance_count": int(retained_mask.sum()),
        "removed_constant_features": json.dumps(
            np.asarray(features)[~retained_mask].tolist()
        ),
        "pca_target_variance": args.pca_variance,
        "pca_component_count": int(pca.n_components_),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "lasso_alpha": selector.selected_alpha_,
        "lasso_nonzero_before_fallback": raw_selected,
        "lasso_selected_pc_count": int(selector.get_support().sum()),
        "lasso_selected_pcs": json.dumps((selected_indices + 1).tolist()),
        "fallback_component_used": used_fallback,
    }
    selected_rows = [
        {
            "outer_fold": fold,
            "held_out_patient": patient,
            "component": f"PC{index + 1}",
            "component_index": int(index + 1),
            "lasso_coefficient": float(coefficients[index]),
            "fallback_component": bool(used_fallback and not raw_support[index]),
            "pca_source_feature_count": len(retained_names),
        }
        for index in selected_indices
    ]
    return audit, selected_rows


def configurations(models):
    """Define the fixed model comparison."""
    result = []
    for modality in ("clinical", "radiomics", "early_fusion"):
        result.extend((modality, model) for model in models)
    result.extend(
        [
            ("late_fusion", "ridge_bases"),
            ("stacking_fusion", "ridge_bases"),
        ]
    )
    return result


def configuration_id(modality, model):
    return f"{modality}|pca_lasso|{model}"


def read_partial(path, resume):
    """Load a checkpoint only when resume was requested."""
    return read_checkpoint(path, resume)


def remove_fold(table, fold):
    """Remove an incomplete fold before recomputing it."""
    if table.empty or "outer_fold" not in table:
        return table
    return table.loc[table["outer_fold"].astype(int).ne(fold)].copy()


def safe_correlation(function, observed, predicted):
    """Return a correlation and p-value, including constant-output cases."""
    if np.std(observed) == 0 or np.std(predicted) == 0:
        return np.nan, np.nan
    statistic, p_value = function(observed, predicted)
    return float(statistic), float(p_value)


def run(args):
    """Run or resume nested LOOCV."""
    if not 0.0 < args.pca_variance < 1.0:
        raise ValueError("--pca-variance must be between zero and one")
    if args.inner_folds < 2:
        raise ValueError("--inner-folds must be at least two")

    models = select_models(args.models)
    table, radio_features = load_data(args)
    configs = configurations(models)
    expected_configs = len(configs)
    if prepare_run(args, table, radio_features, "07"):
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)

    partial_prediction = args.output_dir / "regression_oof_predictions.partial.csv"
    partial_audit = args.output_dir / "regression_reduction_audit.partial.csv"
    partial_selected = args.output_dir / "regression_selected_pcs.partial.csv"
    partial_fusion = args.output_dir / "regression_fusion_parameters.partial.csv"
    predictions = read_partial(partial_prediction, args.resume)
    audits = read_partial(partial_audit, args.resume)
    selected_pcs = read_partial(partial_selected, args.resume)
    fusion_parameters = read_partial(partial_fusion, args.resume)

    if not predictions.empty:
        validate_oof(predictions, table, args.target, [configuration_id(*item) for item in configs])
    complete_folds = set()
    if not predictions.empty and not audits.empty and not fusion_parameters.empty:
        prediction_counts = predictions.groupby("outer_fold")["configuration"].nunique()
        fusion_counts = fusion_parameters.groupby("outer_fold")["modality"].nunique()
        audit_folds = set(audits["outer_fold"].astype(int))
        selected_folds = set(selected_pcs["outer_fold"].astype(int))
        complete_folds = {
            int(fold)
            for fold, count in prediction_counts.items()
            if count == expected_configs
            and fusion_counts.get(fold, 0) == 2
            and int(fold) in audit_folds
            and int(fold) in selected_folds
        }

    patient_ids = table["patient_id"].astype(str).to_numpy()
    outcome = table[args.target].to_numpy(float)
    clinical = table[CLINICAL_FEATURES].to_numpy(float)
    radiomics = table[radio_features].to_numpy(float)
    early = np.column_stack([clinical, radiomics])
    outer_splits = list(LeaveOneOut().split(outcome))
    if args.max_outer_folds is not None:
        outer_splits = outer_splits[: args.max_outer_folds]
    print(
        f"Patients={len(table)} outer_folds={len(outer_splits)} "
        f"configurations_per_fold={expected_configs}",
        flush=True,
    )

    for fold, (train_index, test_index) in enumerate(outer_splits, start=1):
        if fold in complete_folds:
            print(f"Fold {fold}/{len(outer_splits)} already complete", flush=True)
            continue

        predictions = remove_fold(predictions, fold)
        audits = remove_fold(audits, fold)
        selected_pcs = remove_fold(selected_pcs, fold)
        fusion_parameters = remove_fold(fusion_parameters, fold)
        patient = patient_ids[test_index][0]
        observed = float(outcome[test_index][0])
        y_train = outcome[train_index]
        seed = args.random_state + fold
        inner_splits = list(
            KFold(args.inner_folds, shuffle=True, random_state=seed).split(y_train)
        )

        audit, fold_selected = fit_reduction_audit(
            radiomics[train_index], y_train, radio_features, args, fold, patient
        )
        fold_predictions = []
        fold_fusion = []

        # Generate leakage-safe base predictions for both fusion methods.
        clinical_base = ridge_base_pipeline("clinical", args, seed)
        radiomics_base = ridge_base_pipeline("radiomics", args, seed)
        clinical_oof = cross_val_predict(
            clone(clinical_base), clinical[train_index], y_train,
            cv=inner_splits, n_jobs=args.n_jobs,
        )
        radiomics_oof = cross_val_predict(
            clone(radiomics_base), radiomics[train_index], y_train,
            cv=inner_splits, n_jobs=args.n_jobs,
        )
        clinical_fitted = clone(clinical_base).fit(clinical[train_index], y_train)
        radiomics_fitted = clone(radiomics_base).fit(radiomics[train_index], y_train)
        clinical_test = float(clinical_fitted.predict(clinical[test_index])[0])
        radiomics_test = float(radiomics_fitted.predict(radiomics[test_index])[0])

        clinical_weight = learn_late_weight(clinical_oof, radiomics_oof, y_train)
        radiomics_weight = 1.0 - clinical_weight
        late_oof = clinical_weight * clinical_oof + radiomics_weight * radiomics_oof
        late_test = clinical_weight * clinical_test + radiomics_weight * radiomics_test
        fold_fusion.append(
            {
                "outer_fold": fold,
                "held_out_patient": patient,
                "modality": "late_fusion",
                "clinical_weight": clinical_weight,
                "radiomics_weight": radiomics_weight,
                "meta_alpha": np.nan,
                "meta_intercept": np.nan,
                "meta_clinical_coefficient": np.nan,
                "meta_radiomics_coefficient": np.nan,
            }
        )

        meta_features = np.column_stack([clinical_oof, radiomics_oof])
        meta_cv = KFold(args.inner_folds, shuffle=True, random_state=seed + 10_000)
        meta_model = RidgeCV(
            alphas=META_ALPHAS,
            cv=meta_cv,
            scoring="neg_mean_absolute_error",
        ).fit(meta_features, y_train)
        stacking_test = float(
            meta_model.predict(np.array([[clinical_test, radiomics_test]]))[0]
        )
        fold_fusion.append(
            {
                "outer_fold": fold,
                "held_out_patient": patient,
                "modality": "stacking_fusion",
                "clinical_weight": np.nan,
                "radiomics_weight": np.nan,
                "meta_alpha": float(meta_model.alpha_),
                "meta_intercept": float(meta_model.intercept_),
                "meta_clinical_coefficient": float(meta_model.coef_[0]),
                "meta_radiomics_coefficient": float(meta_model.coef_[1]),
            }
        )

        for number, (modality, model_name) in enumerate(configs, start=1):
            if modality == "late_fusion":
                predicted = late_test
                inner_mae = float(mean_absolute_error(y_train, late_oof))
                best_parameters = {
                    "clinical_weight": clinical_weight,
                    "radiomics_weight": radiomics_weight,
                }
            elif modality == "stacking_fusion":
                predicted = stacking_test
                fitted_meta = meta_model.predict(meta_features)
                inner_mae = float(mean_absolute_error(y_train, fitted_meta))
                best_parameters = {
                    "meta_alpha": float(meta_model.alpha_),
                    "meta_intercept": float(meta_model.intercept_),
                    "meta_clinical_coefficient": float(meta_model.coef_[0]),
                    "meta_radiomics_coefficient": float(meta_model.coef_[1]),
                }
            else:
                if modality == "clinical":
                    train_values = clinical[train_index]
                    test_values = clinical[test_index]
                elif modality == "radiomics":
                    train_values = radiomics[train_index]
                    test_values = radiomics[test_index]
                else:
                    train_values = early[train_index]
                    test_values = early[test_index]
                pipeline, grid = build_candidate_pipeline(
                    modality,
                    model_name,
                    len(radio_features),
                    args,
                    seed,
                )
                search = GridSearchCV(
                    pipeline,
                    grid,
                    scoring="neg_mean_absolute_error",
                    cv=inner_splits,
                    n_jobs=args.n_jobs,
                    refit=True,
                    error_score="raise",
                ).fit(train_values, y_train)
                predicted = float(search.predict(test_values)[0])
                inner_mae = -float(search.best_score_)
                best_parameters = search.best_params_

            config = configuration_id(modality, model_name)
            fold_predictions.append(
                {
                    "patient_id": patient,
                    "outer_fold": fold,
                    "configuration": config,
                    "modality": modality,
                    "feature_strategy": "pca_lasso",
                    "model": model_name,
                    "observed": observed,
                    "predicted": predicted,
                    "absolute_error": abs(predicted - observed),
                    "inner_best_mae": inner_mae,
                    "best_parameters": json.dumps(best_parameters, sort_keys=True),
                }
            )
            print(
                f"Fold {fold}/{len(outer_splits)} config "
                f"{number}/{expected_configs} {config} complete",
                flush=True,
            )

        predictions = pd.concat(
            [predictions, pd.DataFrame(fold_predictions)], ignore_index=True
        )
        audits = pd.concat([audits, pd.DataFrame([audit])], ignore_index=True)
        selected_pcs = pd.concat(
            [selected_pcs, pd.DataFrame(fold_selected)], ignore_index=True
        )
        fusion_parameters = pd.concat(
            [fusion_parameters, pd.DataFrame(fold_fusion)], ignore_index=True
        )
        atomic_csv(predictions, partial_prediction)
        atomic_csv(audits, partial_audit)
        atomic_csv(selected_pcs, partial_selected)
        atomic_csv(fusion_parameters, partial_fusion)
        print(f"Checkpoint saved after fold {fold}", flush=True)

    expected_folds = len(outer_splits)
    valid_folds = set(range(1, expected_folds + 1))
    predictions = predictions.loc[
        predictions["outer_fold"].astype(int).isin(valid_folds)
    ].copy()
    audits = audits.loc[audits["outer_fold"].astype(int).isin(valid_folds)].copy()
    selected_pcs = selected_pcs.loc[
        selected_pcs["outer_fold"].astype(int).isin(valid_folds)
    ].copy()
    fusion_parameters = fusion_parameters.loc[
        fusion_parameters["outer_fold"].astype(int).isin(valid_folds)
    ].copy()
    completed = predictions.groupby("configuration")["outer_fold"].nunique()
    if len(completed) != expected_configs or not completed.eq(expected_folds).all():
        raise RuntimeError("Not every configuration has a complete LOOCV prediction set")

    validate_oof(predictions, table, args.target, [configuration_id(*item) for item in configs], expected_folds)
    metric_rows = []
    for config, group in predictions.groupby("configuration", sort=False):
        observed_values = group["observed"].to_numpy(float)
        predicted_values = group["predicted"].to_numpy(float)
        mse = float(mean_squared_error(observed_values, predicted_values))
        pearson_r, pearson_p = safe_correlation(
            pearsonr, observed_values, predicted_values
        )
        spearman_rho, spearman_p = safe_correlation(
            spearmanr, observed_values, predicted_values
        )
        first = group.iloc[0]
        metric_rows.append(
            {
                "configuration": config,
                "modality": first["modality"],
                "feature_strategy": "pca_lasso",
                "model": first["model"],
                "patients": len(group),
                "mse": mse,
                "mae": float(mean_absolute_error(observed_values, predicted_values)),
                "rmse": float(np.sqrt(mse)),
                "r2": float(r2_score(observed_values, predicted_values)),
                "pearson_r": pearson_r,
                "pearson_p_value": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p_value": spearman_p,
            }
        )
    metrics = pd.DataFrame(metric_rows).sort_values(["mae", "rmse"])

    frequency = (
        selected_pcs.groupby("component").size().rename("folds_selected").reset_index()
    )
    frequency["selection_frequency"] = frequency["folds_selected"] / expected_folds
    frequency = frequency.sort_values(
        ["selection_frequency", "component"], ascending=[False, True]
    )

    atomic_csv(predictions, args.output_dir / "regression_oof_predictions.csv")
    atomic_csv(metrics, args.output_dir / "regression_model_performance.csv")
    atomic_csv(audits, args.output_dir / "regression_reduction_audit.csv")
    atomic_csv(selected_pcs, args.output_dir / "regression_selected_pcs.csv")
    atomic_csv(frequency, args.output_dir / "regression_pc_selection_frequency.csv")
    atomic_csv(
        fusion_parameters, args.output_dir / "regression_fusion_parameters.csv"
    )

    best = metrics.iloc[0]
    summary = {
        "status": "pass",
        **run_metadata(args),
        "patients": len(table),
        "outcome": OUTCOMES[args.target],
        "clinical_features": CLINICAL_FEATURES,
        "radiomics_input_features": len(radio_features),
        "radiomics_chain": [
            "VarianceThreshold",
            "StandardScaler",
            f"PCA({args.pca_variance:.0%} variance)",
            "LassoCV component selection",
        ],
        "outer_validation": "leave-one-out cross-validation",
        "inner_validation": f"{args.inner_folds}-fold cross-validation",
        "primary_metric": "MAE",
        "models": models,
        "fusion_methods": ["early", "weighted late", "stacking"],
        "best_configuration": {
            "configuration": str(best["configuration"]),
            "modality": str(best["modality"]),
            "model": str(best["model"]),
            "patients": int(best["patients"]),
            "mse": float(best["mse"]),
            "mae": float(best["mae"]),
            "rmse": float(best["rmse"]),
            "r2": float(best["r2"]),
            "pearson_r": float(best["pearson_r"]),
            "spearman_rho": float(best["spearman_rho"]),
        },
        "leakage_control": (
            "All preprocessing, tuning, component selection, and fusion fitting "
            "used outer-training patients only."
        ),
    }
    atomic_json(summary, args.output_dir / "regression_summary.json")
    atomic_json(summary, args.output_dir / "summary.json")
    lines = [
        "Baseline multimodal regression summary",
        "==========================",
        f"Outcome: {summary['outcome']}",
        f"Patients: {len(table)}",
        f"Best configuration: {best['configuration']}",
        f"MAE: {best['mae']:.3f}",
        f"RMSE: {best['rmse']:.3f}",
        f"R2: {best['r2']:.3f}",
        "",
        summary["leakage_control"],
    ]
    atomic_text("\n".join(lines), args.output_dir / "regression_summary.txt")

    plot = metrics.sort_values("mae", ascending=False).copy()
    plot["label"] = plot["modality"] + " | " + plot["model"]
    height = max(6.0, 0.42 * len(plot))
    figure, axes = plt.subplots(1, 2, figsize=(15, height))
    axes[0].barh(plot["label"], plot["mae"], color="#4c78a8")
    axes[0].set_xlabel("Outer-LOOCV MAE (lower is better)")
    axes[0].grid(axis="x", alpha=0.25)
    axes[1].barh(plot["label"], plot["r2"], color="#f58518")
    axes[1].axvline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Outer-LOOCV R-squared (higher is better)")
    axes[1].grid(axis="x", alpha=0.25)
    figure.suptitle("Baseline multimodal regression: PCA + Lasso")
    figure.tight_layout()
    figure.savefig(
        args.output_dir / "regression_model_performance.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)

    best_predictions = predictions.loc[
        predictions["configuration"].eq(best["configuration"])
    ]
    figure, axis = plt.subplots(figsize=(6.5, 6.0))
    axis.scatter(
        best_predictions["observed"],
        best_predictions["predicted"],
        alpha=0.75,
        color="#4c78a8",
    )
    lower = float(
        min(best_predictions["observed"].min(), best_predictions["predicted"].min())
    )
    upper = float(
        max(best_predictions["observed"].max(), best_predictions["predicted"].max())
    )
    axis.plot([lower, upper], [lower, upper], "--", color="black", linewidth=1.0)
    axis.set_xlabel(f"Observed {OUTCOMES[args.target]}")
    axis.set_ylabel(f"Predicted {OUTCOMES[args.target]}")
    axis.set_title(f"Best model: {best['configuration']}")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(
        args.output_dir / "regression_best_observed_vs_predicted.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)

    for checkpoint in (
        partial_prediction, partial_audit, partial_selected, partial_fusion
    ):
        checkpoint.unlink(missing_ok=True)

    finish_run(args)
    print("\nFinal results", flush=True)
    print(metrics.to_string(index=False), flush=True)
    print(f"\nOutputs: {args.output_dir}", flush=True)


if __name__ == "__main__":
    with parallel_config(backend="loky", inner_max_num_threads=1):
        run_stage(run, parse_arguments())
