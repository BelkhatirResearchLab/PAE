"""Compare All119, PCA, LASSO and PCA-to-LASSO with the same Ridge."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut

from common.regression_data import (
    DEFAULT_DATA as FROZEN_DATA, TARGET, OUTCOMES, CLINICAL_FEATURES,
    add_regression_arguments, load_regression_data, prepare_run, run_metadata,
    finish_run, read_checkpoint, validate_oof, validate_dependency, run_stage,
    output_root, STAGE_FOLDERS,
)

from common.io_utils import atomic_csv, atomic_json
from common.regression import (
    ABLATION_STRATEGIES,
    CLINICAL_FEATURES,
    TARGET,
    load_regression_data,
    paired_mae_bootstrap,
    regression_metrics,
    strategy_pipeline_and_grid,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
DEFAULT_DATA = FROZEN_DATA
DEFAULT_OUTPUT = output_root() / STAGE_FOLDERS["08"]
CONFIGURATIONS = ("clinical", *ABLATION_STRATEGIES)


def arguments():
    parser = argparse.ArgumentParser(
        description="Leakage-safe PCA and LASSO Ridge ablation."
    )
    add_regression_arguments(parser)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-patients", type=int, default=148)
    parser.add_argument("--expected-radiomics-features", type=int, default=119)
    parser.add_argument("--pca-variance", type=float, default=0.95)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--max-outer-folds", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def reduction_audit(search, strategy, radiomics, fold, patient):
    base = {
        "outer_fold": fold,
        "held_out_patient": patient,
        "configuration": strategy,
        "input_radiomics": len(radiomics),
        "after_variance_count": np.nan,
        "after_pca_count": np.nan,
        "pca_explained_variance": np.nan,
        "after_lasso_count": np.nan,
        "lasso_alpha": np.nan,
        "lasso_n_iter": np.nan,
        "lasso_converged": True,
        "lasso_fallback_used": False,
        "feature_space": "none",
    }
    if strategy == "clinical":
        return base, []
    radio = (
        search.best_estimator_.named_steps["preprocess"]
        .named_transformers_["radiomics"]
    )
    variance = radio.named_steps["variance"].get_support()
    labels = np.asarray(radiomics)[variance].astype(str)
    base["after_variance_count"] = int(variance.sum())
    base["feature_space"] = "original_radiomics"
    if "pca" in radio.named_steps:
        pca = radio.named_steps["pca"]
        labels = np.asarray([
            f"PC{index + 1}" for index in range(int(pca.n_components_))
        ])
        base["after_pca_count"] = int(pca.n_components_)
        base["pca_explained_variance"] = float(
            pca.explained_variance_ratio_.sum()
        )
        base["feature_space"] = "pca_components"
    selected_rows = []
    if "selector" in radio.named_steps:
        selector = radio.named_steps["selector"]
        selected = labels[selector.get_support()].tolist()
        n_iter = int(selector.estimator_.n_iter_)
        converged = n_iter < int(selector.max_iter)
        base.update({
            "after_lasso_count": int(len(selected)),
            "lasso_alpha": float(selector.alpha),
            "lasso_n_iter": n_iter,
            "lasso_converged": bool(converged),
            "lasso_fallback_used": bool(selector.fallback_),
        })
        selected_rows = [{
            "outer_fold": fold,
            "held_out_patient": patient,
            "configuration": strategy,
            "feature_space": base["feature_space"],
            "selected_feature": label,
        } for label in selected]
    return base, selected_rows


def fit_one(fold, train_index, test_index, strategy, x_clinical, x_early,
            y, patient_ids, radiomics, args):
    started = time.perf_counter()
    values = x_clinical if strategy == "clinical" else x_early
    pipeline, grid = strategy_pipeline_and_grid(
        strategy, len(radiomics), args.pca_variance
    )
    inner = KFold(
        args.inner_folds, shuffle=True, random_state=args.random_state + fold
    )
    search = GridSearchCV(
        pipeline, grid, scoring="neg_mean_absolute_error", cv=inner,
        n_jobs=1, refit=True, error_score="raise",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        search.fit(values[train_index], y[train_index])
    patient = str(patient_ids[test_index][0])
    audit, selected = reduction_audit(
        search, strategy, radiomics, fold, patient
    )
    if not audit["lasso_converged"]:
        raise RuntimeError(
            f"Selected LASSO did not converge: fold={fold}, {strategy}"
        )
    predicted = float(search.predict(values[test_index])[0])
    observed = float(y[test_index][0])
    prediction = {
        "patient_id": patient,
        "outer_fold": fold,
        "configuration": strategy,
        "observed": observed,
        "predicted": predicted,
        "absolute_error": abs(predicted - observed),
        "inner_best_mae": -float(search.best_score_),
        "best_parameters": json.dumps(
            search.best_params_, sort_keys=True, default=float
        ),
        "candidate_convergence_warning_count": int(sum(
            issubclass(item.category, ConvergenceWarning) for item in caught
        )),
        "fit_seconds": float(time.perf_counter() - started),
    }
    return prediction, audit, selected


def run(args):
    if not 0 < args.pca_variance < 1:
        raise ValueError("pca-variance must be between zero and one")
    table, radiomics = load_regression_data(
        args.data, args.expected_patients, args.expected_radiomics_features,
        target=args.target, feature_manifest=args.feature_manifest,
    )
    if prepare_run(args, table, radiomics, "08", ()):
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial_predictions = args.output_dir / "ablation_oof.partial.csv"
    partial_audit = args.output_dir / "ablation_reduction.partial.csv"
    partial_selected = args.output_dir / "ablation_selected.partial.csv"
    predictions = read_checkpoint(partial_predictions, args.resume)
    audits = read_checkpoint(partial_audit, args.resume)
    selected = read_checkpoint(partial_selected, args.resume)
    if not predictions.empty:
        validate_oof(predictions, table, args.target, CONFIGURATIONS)
    complete = set()
    if not predictions.empty:
        counts = predictions.groupby("outer_fold")["configuration"].nunique()
        complete = set(
            counts.loc[counts.eq(len(CONFIGURATIONS))].index.astype(int)
        )

    if not audits.empty:
        audit_counts = audits.groupby("outer_fold")["configuration"].nunique()
        complete &= set(audit_counts.loc[audit_counts.eq(len(CONFIGURATIONS))].index.astype(int))
    else:
        complete.clear()
    if not selected.empty:
        selected_counts = selected.groupby("outer_fold")["configuration"].nunique()
        complete &= set(selected_counts.loc[selected_counts.eq(2)].index.astype(int))
    else:
        complete.clear()
    patient_ids = table["patient_id"].astype(str).to_numpy()
    y = table[args.target].to_numpy(float)
    x_clinical = table[CLINICAL_FEATURES].to_numpy(float)
    x_early = table[[*CLINICAL_FEATURES, *radiomics]].to_numpy(float)
    outer = list(LeaveOneOut().split(y))
    if args.max_outer_folds is not None:
        outer = outer[:int(args.max_outer_folds)]
    pending = [
        (fold, train, test)
        for fold, (train, test) in enumerate(outer, start=1)
        if fold not in complete
    ]
    print(
        f"Patients={len(table)} folds={len(outer)} "
        f"complete={len(complete)} pending={len(pending)}",
        flush=True,
    )
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset:offset + args.batch_size]
        tasks = [
            (fold, train, test, strategy)
            for fold, train, test in batch
            for strategy in CONFIGURATIONS
        ]
        with parallel_config(backend="loky", inner_max_num_threads=1):
            returned = Parallel(n_jobs=args.n_jobs)(
                delayed(fit_one)(
                    fold, train, test, strategy, x_clinical, x_early, y,
                    patient_ids, radiomics, args,
                )
                for fold, train, test, strategy in tasks
            )
        batch_folds = [fold for fold, _, _ in batch]
        for frame_name, frame in (
            ("predictions", predictions), ("audits", audits),
            ("selected", selected),
        ):
            if not frame.empty:
                cleaned = frame.loc[
                    ~frame["outer_fold"].astype(int).isin(batch_folds)
                ].copy()
                if frame_name == "predictions":
                    predictions = cleaned
                elif frame_name == "audits":
                    audits = cleaned
                else:
                    selected = cleaned
        predictions = pd.concat([
            predictions, pd.DataFrame([item[0] for item in returned])
        ], ignore_index=True)
        audits = pd.concat([
            audits, pd.DataFrame([item[1] for item in returned])
        ], ignore_index=True)
        selected_rows = [
            row for item in returned for row in item[2]
        ]
        if selected_rows:
            selected = pd.concat(
                [selected, pd.DataFrame(selected_rows)], ignore_index=True
            )
        predictions = predictions.sort_values(
            ["outer_fold", "configuration"]
        ).reset_index(drop=True)
        atomic_csv(predictions, partial_predictions)
        atomic_csv(audits, partial_audit)
        if not selected.empty:
            atomic_csv(selected, partial_selected)
        done = predictions.groupby("outer_fold")["configuration"].nunique()
        print(
            f"Checkpoint: {int(done.eq(len(CONFIGURATIONS)).sum())}/"
            f"{len(outer)} folds", flush=True,
        )

    expected = len(outer)
    counts = predictions.groupby("configuration")["outer_fold"].nunique()
    if len(counts) != len(CONFIGURATIONS) or not counts.eq(expected).all():
        raise RuntimeError("Ablation OOF predictions are incomplete")
    validate_oof(predictions, table, args.target, CONFIGURATIONS, expected)
    performance_rows = []
    for name, group in predictions.groupby("configuration", sort=False):
        performance_rows.append({
            "configuration": name,
            "patients": len(group),
            **regression_metrics(group["observed"], group["predicted"]),
            "mean_inner_best_mae": float(group["inner_best_mae"].mean()),
            "total_fit_seconds": float(group["fit_seconds"].sum()),
            "folds_with_candidate_convergence_warnings": int(
                group["candidate_convergence_warning_count"].gt(0).sum()
            ),
        })
    performance = pd.DataFrame(performance_rows).sort_values(
        "mae"
    ).reset_index(drop=True)
    base = (
        predictions[["patient_id", "outer_fold", "observed"]]
        .drop_duplicates("patient_id")
        .sort_values("patient_id").reset_index(drop=True)
    )
    wide = base.copy()
    for name in CONFIGURATIONS:
        values = predictions.loc[
            predictions["configuration"].eq(name),
            ["patient_id", "predicted"],
        ].rename(columns={"predicted": name})
        wide = wide.merge(values, on="patient_id", validate="1:1")
    ci, deltas = paired_mae_bootstrap(
        wide["observed"].to_numpy(float),
        {name: wide[name].to_numpy(float) for name in CONFIGURATIONS},
        args.bootstrap_resamples,
        args.random_state,
    )
    atomic_csv(predictions, args.output_dir / "ablation_oof.csv")
    atomic_csv(performance, args.output_dir / "ablation_performance.csv")
    atomic_csv(audits, args.output_dir / "ablation_reduction_by_fold.csv")
    if not selected.empty:
        atomic_csv(
            selected, args.output_dir / "ablation_selected_features.csv"
        )
    atomic_csv(wide, args.output_dir / "matched_oof_predictions.csv")
    atomic_csv(ci, args.output_dir / "paired_oof_model_ci.csv")
    atomic_csv(deltas, args.output_dir / "paired_oof_deltas.csv")
    best = performance.iloc[0]
    summary = {
        "status": "pass",
        **run_metadata(args),
        "patients": len(wide),
        "outer_validation": "LOOCV",
        "inner_validation": f"{args.inner_folds}-fold shuffled KFold",
        "strategies": list(CONFIGURATIONS),
        "common_estimator": "Ridge",
        "primary_metric": "MAE",
        "best_configuration": str(best["configuration"]),
        "best_mae": float(best["mae"]),
        "code_final_modified_by_runtime": False,
    }
    atomic_json(summary, args.output_dir / "ablation_summary.json")
    atomic_json(summary, args.output_dir / "summary.json")
    for checkpoint in (
        partial_predictions, partial_audit, partial_selected
    ):
        checkpoint.unlink(missing_ok=True)
    finish_run(args)
    print(performance.to_string(index=False), flush=True)
    print(f"Outputs: {args.output_dir}", flush=True)


if __name__ == "__main__":
    run_stage(run, arguments())
