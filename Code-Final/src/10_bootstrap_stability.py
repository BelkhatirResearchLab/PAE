"""Full-pipeline OOB bootstrap stability and overfitting analysis.

Every bootstrap replicate repeats preprocessing, feature reduction, inner
hyperparameter tuning and model fitting.  All configurations use identical
patient-level draws, so their OOB errors can be compared pairwise.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline

from common.regression_data import (
    DEFAULT_DATA as FROZEN_DATA, TARGET, OUTCOMES, CLINICAL_FEATURES,
    add_regression_arguments, load_regression_data, prepare_run, run_metadata,
    finish_run, read_checkpoint, validate_oof, validate_dependency, run_stage,
    output_root, STAGE_FOLDERS,
)

from common.io_utils import atomic_csv, atomic_json
from common.regression import (
    CLINICAL_FEATURES,
    TARGET,
    additional_model_and_grid,
    additional_preprocessor,
    bootstrap_draws,
    convergence_audit,
    load_regression_data,
    regression_metrics,
    safe_r2,
    strategy_pipeline_and_grid,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
DEFAULT_DATA = FROZEN_DATA
DEFAULT_OUTPUT = output_root() / STAGE_FOLDERS["10"]
CONFIGURATIONS = (
    "clinical_ridge",
    "all119_early_fusion_ridge",
    "pca_early_fusion_ridge",
    "lasso_early_fusion_ridge",
    "pca_lasso_early_fusion_ridge",
    "gradient_boosting_early_fusion",
    "ard_early_fusion",
)


def arguments():
    parser = argparse.ArgumentParser(
        description="Paired full-pipeline OOB bootstrap analysis."
    )
    add_regression_arguments(parser)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-patients", type=int, default=148)
    parser.add_argument("--expected-radiomics-features", type=int, default=119)
    parser.add_argument("--pca-variance", type=float, default=0.95)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=7)
    parser.add_argument("--required-valid-resamples", type=int, default=500)
    parser.add_argument("--max-attempts", type=int, default=650)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def build_pipeline(configuration, radiomics_count, pca_variance, seed):
    if configuration == "clinical_ridge":
        return strategy_pipeline_and_grid("clinical", radiomics_count, pca_variance)
    strategies = {
        "all119_early_fusion_ridge": "all",
        "pca_early_fusion_ridge": "pca",
        "lasso_early_fusion_ridge": "lasso",
        "pca_lasso_early_fusion_ridge": "pca_lasso",
    }
    if configuration in strategies:
        return strategy_pipeline_and_grid(
            strategies[configuration], radiomics_count, pca_variance
        )
    model_name = configuration.removesuffix("_early_fusion")
    estimator, grid = additional_model_and_grid(model_name, "early_fusion", seed)
    if model_name == "ard":
        estimator.set_params(max_iter=20_000)
    return Pipeline([
        ("preprocess", additional_preprocessor("early_fusion", radiomics_count)),
        ("model", estimator),
    ]), grid


def reduction_audit(estimator, configuration, radiomics_features):
    if configuration not in {
        "all119_early_fusion_ridge", "pca_early_fusion_ridge",
        "lasso_early_fusion_ridge", "pca_lasso_early_fusion_ridge",
    }:
        return {}, []
    radiomics = estimator.named_steps["preprocess"].named_transformers_["radiomics"]
    variance = radiomics.named_steps["variance"]
    retained = np.asarray(radiomics_features)[variance.get_support()].tolist()
    audit = {
        "radiomics_after_variance": len(retained),
        "pca_components": np.nan,
        "lasso_selected": np.nan,
        "lasso_raw_nonzero": np.nan,
        "lasso_fallback": False,
        "lasso_n_iter": np.nan,
        "lasso_converged": True,
    }
    labels = retained
    if "pca" in radiomics.named_steps:
        count = int(radiomics.named_steps["pca"].n_components_)
        audit["pca_components"] = count
        labels = [f"PC{index:03d}" for index in range(1, count + 1)]
    selected = []
    if "selector" in radiomics.named_steps:
        selector = radiomics.named_steps["selector"]
        support = selector.get_support()
        selected = [label for label, keep in zip(labels, support) if keep]
        audit.update({
            "lasso_selected": len(selected),
            "lasso_raw_nonzero": int(selector.raw_nonzero_count_),
            "lasso_fallback": bool(selector.fallback_),
            "lasso_n_iter": int(selector.estimator_.n_iter_),
            "lasso_converged": bool(
                selector.estimator_.n_iter_ < selector.estimator_.max_iter
            ),
        })
    return audit, selected


def fit_one(attempt, configuration, inbag, oob, x_clinical, x_early, y,
            patient_ids, radiomics_features, args):
    started = time.perf_counter()
    try:
        seed = args.random_state + attempt
        pipeline, grid = build_pipeline(
            configuration, len(radiomics_features), args.pca_variance, seed
        )
        x = x_clinical if configuration == "clinical_ridge" else x_early
        groups = patient_ids[inbag]
        inner = GroupKFold(n_splits=args.inner_folds)
        search = GridSearchCV(
            pipeline, grid, scoring="neg_mean_absolute_error",
            cv=inner.split(x[inbag], y[inbag], groups), n_jobs=1,
            refit=True, error_score="raise",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            search.fit(x[inbag], y[inbag])
        warning_count = sum(
            issubclass(item.category, ConvergenceWarning) for item in caught
        )
        apparent = np.asarray(search.predict(x[inbag]), float).reshape(-1)
        oob_prediction = np.asarray(search.predict(x[oob]), float).reshape(-1)
        apparent_metrics = regression_metrics(y[inbag], apparent)
        oob_metrics = regression_metrics(y[oob], oob_prediction)
        model = search.best_estimator_.named_steps["model"]
        n_iter, model_converged, nonzero = convergence_audit(model)
        audit, selected = reduction_audit(
            search.best_estimator_, configuration, radiomics_features
        )
        valid = bool(model_converged and audit.get("lasso_converged", True))
        row = {
            "bootstrap_replicate": attempt,
            "configuration": configuration,
            "unique_inbag_patients": int(np.unique(inbag).size),
            "oob_patients": int(len(oob)),
            "apparent_mae": apparent_metrics["mae"],
            "oob_mae": oob_metrics["mae"],
            "mae_optimism": oob_metrics["mae"] - apparent_metrics["mae"],
            "apparent_rmse": apparent_metrics["rmse"],
            "oob_rmse": oob_metrics["rmse"],
            "rmse_optimism": oob_metrics["rmse"] - apparent_metrics["rmse"],
            "apparent_r2": safe_r2(y[inbag], apparent),
            "oob_r2": safe_r2(y[oob], oob_prediction),
            "r2_optimism": safe_r2(y[inbag], apparent) - safe_r2(y[oob], oob_prediction),
            "inner_best_mae": -float(search.best_score_),
            "best_parameters": json.dumps(search.best_params_, sort_keys=True, default=float),
            "selected_model_n_iter": n_iter,
            "selected_model_converged": bool(model_converged),
            "selected_model_nonzero_coefficients": nonzero,
            "candidate_convergence_warning_count": int(warning_count),
            "fit_seconds": float(time.perf_counter() - started),
            **audit,
        }
        selected_rows = [
            {
                "bootstrap_replicate": attempt,
                "configuration": configuration,
                "selected_feature": feature,
            }
            for feature in selected
        ]
        return {"valid": valid, "row": row, "selected": selected_rows, "error": ""}
    except Exception as error:  # recorded and replaced by another draw
        return {
            "valid": False, "row": None, "selected": [],
            "error": f"{type(error).__name__}: {error}",
            "configuration": configuration,
        }


def summarize(results):
    rows = []
    measures = (
        "apparent_mae", "oob_mae", "mae_optimism",
        "apparent_rmse", "oob_rmse", "rmse_optimism",
        "apparent_r2", "oob_r2", "r2_optimism",
    )
    for configuration, group in results.groupby("configuration", sort=False):
        row = {"configuration": configuration, "valid_replicates": len(group)}
        for measure in measures:
            values = group[measure].to_numpy(float)
            row[f"{measure}_mean"] = float(np.nanmean(values))
            row[f"{measure}_median"] = float(np.nanmedian(values))
            row[f"{measure}_ci95_low"] = float(np.nanquantile(values, 0.025))
            row[f"{measure}_ci95_high"] = float(np.nanquantile(values, 0.975))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("oob_mae_mean").reset_index(drop=True)


def paired_deltas(results):
    pivot = results.pivot(
        index="bootstrap_replicate", columns="configuration", values="oob_mae"
    )
    rows = []
    for reference, candidate in combinations(CONFIGURATIONS, 2):
        delta = (pivot[candidate] - pivot[reference]).dropna().to_numpy(float)
        rows.append({
            "reference": reference,
            "candidate": candidate,
            "paired_replicates": len(delta),
            "delta_oob_mae_candidate_minus_reference_mean": float(delta.mean()),
            "delta_oob_mae_ci95_low": float(np.quantile(delta, 0.025)),
            "delta_oob_mae_ci95_high": float(np.quantile(delta, 0.975)),
            "probability_candidate_lower_oob_mae": float(np.mean(delta < 0)),
        })
    return pd.DataFrame(rows)


def run(args):
    if args.required_valid_resamples < 2:
        raise ValueError("At least two valid bootstrap resamples are required")
    if args.max_attempts < args.required_valid_resamples:
        raise ValueError("max-attempts must be at least required-valid-resamples")
    table, radiomics = load_regression_data(
        args.data, args.expected_patients, args.expected_radiomics_features,
        target=args.target, feature_manifest=args.feature_manifest,
    )
    if prepare_run(args, table, radiomics, "10", ()):
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "pipeline_bootstrap_results.partial.csv"
    selected_path = args.output_dir / "bootstrap_selected_features.partial.csv"
    failure_path = args.output_dir / "bootstrap_failures.partial.csv"
    results = read_checkpoint(result_path, args.resume)
    selected = read_checkpoint(selected_path, args.resume)
    failures = read_checkpoint(failure_path, args.resume)
    counts = results.groupby("bootstrap_replicate")["configuration"].nunique() if not results.empty else pd.Series(dtype=int)
    valid_attempts = set(counts.loc[counts.eq(len(CONFIGURATIONS))].index.astype(int))
    if not results.empty and results.duplicated(["bootstrap_replicate", "configuration"]).any():
        raise ValueError("Duplicate bootstrap checkpoint rows")
    if not selected.empty:
        selected_counts = selected.groupby("bootstrap_replicate")["configuration"].nunique()
        valid_attempts &= set(selected_counts.loc[selected_counts.eq(2)].index.astype(int))
    else:
        valid_attempts.clear()
    failed_attempts = set(failures["bootstrap_replicate"].astype(int)) if not failures.empty else set()

    patient_ids = table["patient_id"].astype(str).to_numpy()
    y = table[args.target].to_numpy(float)
    x_clinical = table[CLINICAL_FEATURES].to_numpy(float)
    x_early = table[[*CLINICAL_FEATURES, *radiomics]].to_numpy(float)
    draws = bootstrap_draws(len(table), args.random_state, args.max_attempts)

    for attempt in range(1, args.max_attempts + 1):
        if len(valid_attempts) >= args.required_valid_resamples:
            break
        if attempt in valid_attempts or attempt in failed_attempts:
            continue
        # Remove a partial replicate left by a crash before all files committed.
        if not results.empty:
            results = results.loc[~results["bootstrap_replicate"].eq(attempt)].copy()
        if not selected.empty:
            selected = selected.loc[~selected["bootstrap_replicate"].eq(attempt)].copy()
        inbag, oob = draws[attempt]
        with parallel_config(backend="loky", inner_max_num_threads=1):
            fitted = Parallel(n_jobs=min(args.n_jobs, len(CONFIGURATIONS)))(
                delayed(fit_one)(
                    attempt, configuration, inbag, oob, x_clinical, x_early,
                    y, patient_ids, radiomics, args,
                )
                for configuration in CONFIGURATIONS
            )
        if all(item["valid"] for item in fitted):
            new_rows = pd.DataFrame([item["row"] for item in fitted])
            new_selected = pd.DataFrame(
                [row for item in fitted for row in item["selected"]]
            )
            results = pd.concat([results, new_rows], ignore_index=True)
            if not new_selected.empty:
                selected = pd.concat([selected, new_selected], ignore_index=True)
            valid_attempts.add(attempt)
        else:
            failed = [item for item in fitted if not item["valid"]]
            failure_rows = [{
                "bootstrap_replicate": attempt,
                "configuration": item.get("configuration") or (
                    item["row"]["configuration"] if item["row"] else "unknown"
                ),
                "reason": item.get("error") or "selected estimator did not converge",
            } for item in failed]
            failures = pd.concat([failures, pd.DataFrame(failure_rows)], ignore_index=True)
            failed_attempts.add(attempt)
        atomic_csv(results, result_path)
        if not selected.empty:
            atomic_csv(selected, selected_path)
        if not failures.empty:
            atomic_csv(failures, failure_path)
        print(
            f"Attempt {attempt}/{args.max_attempts}: "
            f"valid={len(valid_attempts)}/{args.required_valid_resamples}",
            flush=True,
        )

    counts = results.groupby("bootstrap_replicate")["configuration"].nunique()
    valid = sorted(counts.loc[counts.eq(len(CONFIGURATIONS))].index.astype(int))
    valid = valid[:args.required_valid_resamples]
    if len(valid) < args.required_valid_resamples:
        raise RuntimeError(
            f"Only {len(valid)} valid paired resamples after {args.max_attempts} attempts"
        )
    results = results.loc[results["bootstrap_replicate"].isin(valid)].copy()
    selected = selected.loc[selected["bootstrap_replicate"].isin(valid)].copy() if not selected.empty else selected
    summary = summarize(results)
    deltas = paired_deltas(results)
    if not selected.empty:
        frequency = (
            selected.groupby(["configuration", "selected_feature"])
            .size().rename("selection_count").reset_index()
        )
        frequency["valid_replicates"] = len(valid)
        frequency["selection_frequency"] = frequency["selection_count"] / len(valid)
        frequency = frequency.sort_values(
            ["configuration", "selection_frequency", "selected_feature"],
            ascending=[True, False, True],
        )
    else:
        frequency = pd.DataFrame(columns=[
            "configuration", "selected_feature", "selection_count",
            "valid_replicates", "selection_frequency",
        ])
    if failures.empty:
        failures = pd.DataFrame(columns=[
            "bootstrap_replicate", "configuration", "reason"
        ])
    atomic_csv(results, args.output_dir / "pipeline_bootstrap_results.csv")
    atomic_csv(summary, args.output_dir / "pipeline_bootstrap_summary.csv")
    atomic_csv(deltas, args.output_dir / "pipeline_bootstrap_oob_deltas.csv")
    atomic_csv(selected, args.output_dir / "bootstrap_selected_features.csv")
    atomic_csv(frequency, args.output_dir / "bootstrap_selection_frequency.csv")
    atomic_csv(failures, args.output_dir / "bootstrap_failures.csv")
    best = summary.iloc[0]
    atomic_json({
        "status": "pass",
        **run_metadata(args),
        "valid_paired_resamples": len(valid),
        "attempts_consumed": max(valid),
        "configurations": list(CONFIGURATIONS),
        "inner_validation": f"patient-grouped {args.inner_folds}-fold CV",
        "best_mean_oob_mae_configuration": str(best["configuration"]),
        "best_mean_oob_mae": float(best["oob_mae_mean"]),
        "overfitting_definition": "OOB error minus apparent in-bag error",
    }, args.output_dir / "summary.json")
    result_path.unlink(missing_ok=True)
    selected_path.unlink(missing_ok=True)
    failure_path.unlink(missing_ok=True)
    finish_run(args)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    run_stage(run, arguments())
