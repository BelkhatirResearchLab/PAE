"""Nested-LOOCV screening of additional clinical and early-fusion models."""

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
from sklearn.pipeline import Pipeline

from common.regression_data import (
    DEFAULT_DATA as FROZEN_DATA, TARGET, OUTCOMES, CLINICAL_FEATURES,
    add_regression_arguments, load_regression_data, prepare_run, run_metadata,
    finish_run, read_checkpoint, validate_oof, validate_dependency, run_stage,
    output_root, STAGE_FOLDERS,
)

from common.io_utils import atomic_csv, atomic_json
from common.regression import (
    ADDITIONAL_MODELS,
    CLINICAL_FEATURES,
    TARGET,
    additional_model_and_grid,
    additional_preprocessor,
    convergence_audit,
    load_regression_data,
    paired_mae_bootstrap,
    regression_metrics,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
DEFAULT_DATA = FROZEN_DATA
DEFAULT_REFERENCE = output_root() / "ablation/matched_oof_predictions.csv"
DEFAULT_OUTPUT = output_root() / STAGE_FOLDERS["09"]
MODALITIES = ("clinical", "early_fusion")


def arguments():
    parser = argparse.ArgumentParser(
        description="Nested-LOOCV additional regression model screening."
    )
    add_regression_arguments(parser)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--reference-oof", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-patients", type=int, default=148)
    parser.add_argument("--expected-radiomics-features", type=int, default=119)
    parser.add_argument(
        "--models", nargs="+", choices=ADDITIONAL_MODELS,
        default=list(ADDITIONAL_MODELS),
    )
    parser.add_argument(
        "--modalities", nargs="+", choices=MODALITIES,
        default=list(MODALITIES),
    )
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--max-outer-folds", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def fit_one(fold, train_index, test_index, modality, model_name, x, y,
            patient_ids, radiomics_count, args):
    started = time.perf_counter()
    seed = args.random_state + fold
    estimator, grid = additional_model_and_grid(model_name, modality, seed)
    pipeline = Pipeline([
        ("preprocess", additional_preprocessor(modality, radiomics_count)),
        ("model", estimator),
    ])
    inner = KFold(
        args.inner_folds, shuffle=True, random_state=seed
    )
    search = GridSearchCV(
        pipeline, grid, scoring="neg_mean_absolute_error", cv=inner,
        n_jobs=1, refit=True, error_score="raise",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        search.fit(x[train_index], y[train_index])
    warning_count = sum(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )
    selected_model = search.best_estimator_.named_steps["model"]
    n_iter, converged, nonzero = convergence_audit(selected_model)
    if not converged:
        raise RuntimeError(
            f"Final model did not converge: {modality}|{model_name}, "
            f"fold={fold}, n_iter={n_iter}"
        )
    prediction = float(
        np.asarray(search.predict(x[test_index])).reshape(-1)[0]
    )
    observed = float(y[test_index][0])
    return {
        "patient_id": str(patient_ids[test_index][0]),
        "outer_fold": fold,
        "configuration": f"{modality}|{model_name}",
        "modality": modality,
        "model": model_name,
        "observed": observed,
        "predicted": prediction,
        "absolute_error": abs(prediction - observed),
        "inner_best_mae": -float(search.best_score_),
        "best_parameters": json.dumps(
            search.best_params_, sort_keys=True, default=float
        ),
        "selected_model_n_iter": n_iter,
        "selected_model_converged": bool(converged),
        "selected_model_nonzero_coefficients": nonzero,
        "candidate_convergence_warning_count": int(warning_count),
        "fit_seconds": float(time.perf_counter() - started),
    }


def performance_table(predictions):
    rows = []
    for configuration, group in predictions.groupby(
        "configuration", sort=False
    ):
        rows.append({
            "configuration": configuration,
            "modality": group["modality"].iloc[0],
            "model": group["model"].iloc[0],
            "patients": len(group),
            **regression_metrics(group["observed"], group["predicted"]),
            "mean_inner_best_mae": float(group["inner_best_mae"].mean()),
            "total_fit_seconds": float(group["fit_seconds"].sum()),
            "folds_with_candidate_convergence_warnings": int(
                group["candidate_convergence_warning_count"].gt(0).sum()
            ),
        })
    return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)


def frozen_reference_rows(predictions, path):
    reference = pd.read_csv(path)
    needed = {"patient_id", "observed", "clinical", "lasso"}
    missing = needed - set(reference)
    if missing:
        raise ValueError(f"Reference OOF missing: {sorted(missing)}")
    patient_outcome = (
        predictions[["patient_id", "observed"]]
        .drop_duplicates("patient_id")
        .set_index("patient_id")["observed"]
    )
    reference["patient_id"] = reference["patient_id"].astype(str)
    reference = reference.loc[
        reference["patient_id"].isin(patient_outcome.index)
    ].copy()
    if len(reference) != len(patient_outcome):
        raise ValueError("Reference OOF does not match the modelling subset")
    if not np.allclose(
        reference["patient_id"].map(patient_outcome),
        reference["observed"],
    ):
        raise ValueError("Reference OOF outcome mismatch")
    rows = []
    mapping = {
        "clinical": "frozen|clinical_ridge",
        "lasso": "frozen|lasso_early_fusion_ridge",
    }
    for column, configuration in mapping.items():
        for item in reference.itertuples(index=False):
            rows.append({
                "patient_id": str(item.patient_id),
                "outer_fold": int(item.outer_fold),
                "configuration": configuration,
                "modality": "frozen",
                "model": configuration.removeprefix("frozen|"),
                "observed": float(item.observed),
                "predicted": float(getattr(item, column)),
                "absolute_error": abs(
                    float(getattr(item, column)) - float(item.observed)
                ),
                "inner_best_mae": np.nan,
                "best_parameters": "{}",
                "selected_model_n_iter": np.nan,
                "selected_model_converged": True,
                "selected_model_nonzero_coefficients": np.nan,
                "candidate_convergence_warning_count": 0,
                "fit_seconds": 0.0,
            })
    return pd.DataFrame(rows)


def run(args):
    models = list(dict.fromkeys(args.models))
    modalities = list(dict.fromkeys(args.modalities))
    table, radiomics = load_regression_data(
        args.data, args.expected_patients, args.expected_radiomics_features,
        target=args.target, feature_manifest=args.feature_manifest,
    )
    if prepare_run(args, table, radiomics, "09", (args.reference_oof,)):
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configurations = [
        (modality, model) for modality in modalities for model in models
    ]
    partial = args.output_dir / "additional_models_oof.partial.csv"
    predictions = read_checkpoint(partial, args.resume)
    if not predictions.empty:
        validate_oof(predictions, table, args.target, [f"{m}|{n}" for m, n in configurations])
    complete = set()
    if not predictions.empty:
        counts = predictions.groupby("outer_fold")["configuration"].nunique()
        complete = set(
            counts.loc[counts.eq(len(configurations))].index.astype(int)
        )
    patient_ids = table["patient_id"].astype(str).to_numpy()
    y = table[args.target].to_numpy(float)
    x_values = {
        "clinical": table[CLINICAL_FEATURES].to_numpy(float),
        "early_fusion": table[
            [*CLINICAL_FEATURES, *radiomics]
        ].to_numpy(float),
    }
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
        f"configurations={len(configurations)} pending={len(pending)}",
        flush=True,
    )
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset:offset + args.batch_size]
        tasks = [
            (fold, train, test, modality, model)
            for fold, train, test in batch
            for modality, model in configurations
        ]
        with parallel_config(backend="loky", inner_max_num_threads=1):
            rows = Parallel(n_jobs=args.n_jobs)(
                delayed(fit_one)(
                    fold, train, test, modality, model,
                    x_values[modality], y, patient_ids, len(radiomics), args,
                )
                for fold, train, test, modality, model in tasks
            )
        batch_folds = [fold for fold, _, _ in batch]
        if not predictions.empty:
            predictions = predictions.loc[
                ~predictions["outer_fold"].astype(int).isin(batch_folds)
            ].copy()
        predictions = pd.concat(
            [predictions, pd.DataFrame(rows)], ignore_index=True
        ).sort_values(
            ["outer_fold", "configuration"]
        ).reset_index(drop=True)
        atomic_csv(predictions, partial)
        done = predictions.groupby("outer_fold")["configuration"].nunique()
        print(
            f"Checkpoint: {int(done.eq(len(configurations)).sum())}/"
            f"{len(outer)} folds", flush=True,
        )


    expected_rows = len(outer) * len(configurations)
    if len(predictions) != expected_rows:
        raise RuntimeError(
            f"Incomplete model screen: {len(predictions)}/{expected_rows} rows"
        )
    validate_oof(predictions, table, args.target, [f"{m}|{n}" for m, n in configurations], len(outer))
    performance = performance_table(predictions)
    references = frozen_reference_rows(predictions, args.reference_oof)
    combined = pd.concat([predictions, references], ignore_index=True)
    combined_performance = performance_table(combined)

    base = (
        combined[["patient_id", "observed"]]
        .drop_duplicates("patient_id")
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    wide = base.copy()
    names = combined["configuration"].drop_duplicates().tolist()
    for name in names:
        values = combined.loc[
            combined["configuration"].eq(name),
            ["patient_id", "predicted"],
        ].rename(columns={"predicted": name})
        wide = wide.merge(values, on="patient_id", validate="1:1")
    ci, deltas = paired_mae_bootstrap(
        wide["observed"].to_numpy(float),
        {name: wide[name].to_numpy(float) for name in names},
        args.bootstrap_resamples,
        args.random_state,
    )

    atomic_csv(predictions, args.output_dir / "additional_models_oof.csv")
    atomic_csv(performance, args.output_dir / "additional_models_performance.csv")
    atomic_csv(combined, args.output_dir / "all_oof_predictions_with_references.csv")
    atomic_csv(combined_performance, args.output_dir / "all_models_performance.csv")
    atomic_csv(wide, args.output_dir / "all_oof_predictions_wide.csv")
    atomic_csv(ci, args.output_dir / "paired_oof_model_ci.csv")
    atomic_csv(deltas, args.output_dir / "paired_oof_deltas.csv")
    best_new = performance.iloc[0]
    best_all = combined_performance.iloc[0]
    atomic_json({
        "status": "pass",
        **run_metadata(args),
        "patients": len(wide),
        "outer_validation": "LOOCV",
        "inner_validation": f"{args.inner_folds}-fold shuffled KFold",
        "primary_metric": "MAE",
        "screened_models": models,
        "modalities": modalities,
        "best_new_configuration": str(best_new["configuration"]),
        "best_new_mae": float(best_new["mae"]),
        "best_configuration_including_frozen_references": str(
            best_all["configuration"]
        ),
        "best_mae_including_frozen_references": float(best_all["mae"]),
        "selection_rule": (
            "Compare point MAE and paired bootstrap uncertainty; retain the "
            "simpler LASSO early-fusion Ridge when improvement is not robust."
        ),
    }, args.output_dir / "summary.json")
    partial.unlink(missing_ok=True)
    finish_run(args)
    print(combined_performance.to_string(index=False), flush=True)


if __name__ == "__main__":
    run_stage(run, arguments())
