"""Bootstrap feature importance and descriptive SHAP interpretation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed, parallel_config
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error

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
    load_regression_data,
    strategy_pipeline_and_grid,
)
from sklearn.pipeline import Pipeline


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
DEFAULT_DATA = FROZEN_DATA
DEFAULT_BOOTSTRAP = output_root() / "bootstrap"
DEFAULT_SCREEN = output_root() / "model_screen/additional_models_oof.csv"
DEFAULT_OUTPUT = output_root() / STAGE_FOLDERS["11"]


def arguments():
    parser = argparse.ArgumentParser(
        description="Feature importance and model interpretation analysis."
    )
    add_regression_arguments(parser)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--bootstrap-dir", type=Path, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--screening-oof", type=Path, default=DEFAULT_SCREEN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-patients", type=int, default=148)
    parser.add_argument("--expected-radiomics-features", type=int, default=119)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--group-permutation-repeats", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-valid-replicates", type=int)
    parser.add_argument("--case-count", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def valid_replicates(bootstrap_results, maximum):
    expected = bootstrap_results["configuration"].nunique()
    counts = bootstrap_results.groupby("bootstrap_replicate")["configuration"].nunique()
    valid = sorted(counts.loc[counts.eq(expected)].index.astype(int))
    if maximum is not None:
        valid = valid[:int(maximum)]
    if not valid:
        raise ValueError("No complete bootstrap replicates were found")
    return valid


def parameter_row(results, replicate, configuration):
    match = results.loc[
        results["bootstrap_replicate"].eq(replicate)
        & results["configuration"].eq(configuration)
    ]
    if len(match) != 1:
        raise ValueError(f"Missing bootstrap parameter row: {replicate}, {configuration}")
    return json.loads(match.iloc[0]["best_parameters"])


def interpret_one(replicate, inbag, oob, table, radiomics, bootstrap_results,
                  args):
    patient_ids = table["patient_id"].astype(str).to_numpy()
    y = table[args.target].to_numpy(float)
    x_clinical = table[CLINICAL_FEATURES].to_numpy(float)
    feature_names = [*CLINICAL_FEATURES, *radiomics]
    x_early = table[feature_names].to_numpy(float)

    clinical, _ = strategy_pipeline_and_grid("clinical", len(radiomics))
    clinical.set_params(**parameter_row(
        bootstrap_results, replicate, "clinical_ridge"
    ))
    clinical.fit(x_clinical[inbag], y[inbag])
    coefficients = np.asarray(clinical.named_steps["model"].coef_).reshape(-1)
    clinical_rows = [{
        "bootstrap_replicate": replicate,
        "feature": feature,
        "standardized_coefficient": float(value),
        "absolute_standardized_coefficient": abs(float(value)),
    } for feature, value in zip(CLINICAL_FEATURES, coefficients)]

    estimator, _ = additional_model_and_grid(
        "gradient_boosting", "early_fusion", args.random_state + replicate
    )
    gradient = Pipeline([
        ("preprocess", additional_preprocessor("early_fusion", len(radiomics))),
        ("model", estimator),
    ])
    gradient.set_params(**parameter_row(
        bootstrap_results, replicate, "gradient_boosting_early_fusion"
    ))
    gradient.fit(x_early[inbag], y[inbag])
    importance = permutation_importance(
        gradient, x_early[oob], y[oob],
        scoring="neg_mean_absolute_error",
        n_repeats=args.permutation_repeats,
        random_state=args.random_state + 10_000 + replicate,
        n_jobs=1,
    )
    individual_rows = []
    for feature_index, feature in enumerate(feature_names):
        for repeat, value in enumerate(importance.importances[feature_index], start=1):
            individual_rows.append({
                "bootstrap_replicate": replicate,
                "feature": feature,
                "feature_type": "clinical" if feature in CLINICAL_FEATURES else "radiomics",
                "permutation_repeat": repeat,
                "oob_mae_increase": float(value),
            })

    baseline = mean_absolute_error(y[oob], gradient.predict(x_early[oob]))
    group_rows = []
    groups = {
        "clinical": np.arange(len(CLINICAL_FEATURES)),
        "radiomics": np.arange(len(CLINICAL_FEATURES), len(feature_names)),
    }
    for group_index, (group, columns) in enumerate(groups.items()):
        random = np.random.default_rng(
            args.random_state + 1_000_000 + replicate * 10 + group_index
        )
        for repeat in range(1, args.group_permutation_repeats + 1):
            permuted = x_early[oob].copy()
            order = random.permutation(len(oob))
            permuted[:, columns] = permuted[order][:, columns]
            changed = mean_absolute_error(y[oob], gradient.predict(permuted))
            group_rows.append({
                "bootstrap_replicate": replicate,
                "feature_group": group,
                "permutation_repeat": repeat,
                "baseline_oob_mae": float(baseline),
                "permuted_oob_mae": float(changed),
                "oob_mae_increase": float(changed - baseline),
                "oob_patient_count": len(oob),
                "oob_patient_ids": json.dumps(patient_ids[oob].tolist()),
            })
    return clinical_rows, individual_rows, group_rows


def summarize_values(values, key, value):
    per_replicate = (
        values.groupby(["bootstrap_replicate", key])[value]
        .mean().reset_index()
    )
    rows = []
    for name, group in per_replicate.groupby(key):
        array = group[value].to_numpy(float)
        rows.append({
            key: name,
            "bootstrap_replicates": len(array),
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "ci95_low": float(np.quantile(array, 0.025)),
            "ci95_high": float(np.quantile(array, 0.975)),
            "probability_positive": float(np.mean(array > 0)),
        })
    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)


def plot_importance(summary, key, path, title, top=20):
    shown = summary.head(top).sort_values("mean")
    figure_height = max(4.0, 0.28 * len(shown) + 1.5)
    fig, ax = plt.subplots(figsize=(8, figure_height))
    lower = np.maximum(0.0, shown["mean"] - shown["ci95_low"])
    upper = np.maximum(0.0, shown["ci95_high"] - shown["mean"])
    ax.barh(shown[key], shown["mean"], xerr=np.vstack([lower, upper]), alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Increase in OOB MAE after permutation")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def lasso_frequency(bootstrap_dir, output):
    source = pd.read_csv(bootstrap_dir / "bootstrap_selection_frequency.csv")
    frequency = source.loc[
        source["configuration"].eq("lasso_early_fusion_ridge")
    ].sort_values(
        ["selection_frequency", "selected_feature"], ascending=[False, True]
    ).reset_index(drop=True)
    frequency.insert(0, "rank", np.arange(1, len(frequency) + 1))
    atomic_csv(frequency, output / "lasso_selection_frequency.csv")
    return frequency


def shap_analysis(screening_oof, table, radiomics, output, seed, target=TARGET):
    oof = pd.read_csv(screening_oof)
    matches = oof.loc[oof["configuration"].eq("early_fusion|gradient_boosting")]
    if matches.empty:
        raise ValueError("Gradient-boosting OOF parameter records are missing")
    modal_parameters = matches["best_parameters"].mode().iloc[0]
    parameters = json.loads(modal_parameters)
    features = [*CLINICAL_FEATURES, *radiomics]
    x = table[features].to_numpy(float)
    y = table[target].to_numpy(float)
    estimator, _ = additional_model_and_grid("gradient_boosting", "early_fusion", seed)
    pipeline = Pipeline([
        ("preprocess", additional_preprocessor("early_fusion", len(radiomics))),
        ("model", estimator),
    ])
    pipeline.set_params(**parameters)
    pipeline.fit(x, y)
    transformed = np.asarray(pipeline.named_steps["preprocess"].transform(x))
    variance = (
        pipeline.named_steps["preprocess"].named_transformers_["radiomics"]
        .named_steps["variance"]
    )
    transformed_names = [
        *CLINICAL_FEATURES,
        *np.asarray(radiomics)[variance.get_support()].tolist(),
    ]
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    shap_values = np.asarray(explainer.shap_values(transformed))
    mean_absolute = np.mean(np.abs(shap_values), axis=0)
    ranking = pd.DataFrame({
        "feature": transformed_names,
        "mean_absolute_shap": mean_absolute,
        "feature_type": [
            "clinical" if feature in CLINICAL_FEATURES else "radiomics"
            for feature in transformed_names
        ],
    }).sort_values("mean_absolute_shap", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    atomic_csv(ranking, output / "gradient_boosting_shap_ranking.csv")
    np.random.seed(seed)
    plt.figure(figsize=(9, 7))
    shap.summary_plot(
        shap_values, transformed, feature_names=transformed_names,
        plot_type="bar", max_display=20, show=False,
    )
    plt.tight_layout()
    plt.savefig(output / "gradient_boosting_shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()
    np.random.seed(seed)
    plt.figure(figsize=(9, 7))
    shap.summary_plot(
        shap_values, transformed, feature_names=transformed_names,
        max_display=20, show=False,
    )
    plt.tight_layout()
    plt.savefig(output / "gradient_boosting_shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()
    return ranking, parameters


def patient_shap_cases(screening_oof, table, radiomics, output, args):
    """Explain deterministic illustrative cases using their outer-held-out models."""
    oof = pd.read_csv(screening_oof)
    matches = oof.loc[oof["configuration"].eq("early_fusion|gradient_boosting")].copy()
    if matches.empty:
        raise ValueError("No held-out gradient-boosting predictions for patient cases")
    validate_oof(matches, table, args.target, ["early_fusion|gradient_boosting"])
    if args.case_count < 1:
        raise ValueError("case-count must be positive")
    midpoint = float(matches["observed"].median())
    choices = []
    # Predefined descriptive rule: best predicted lower-score case, best
    # predicted higher-score case, then greatest prediction error.
    for reason, candidates in (
        ("lowest_error_lower_observed_score", matches.loc[matches["observed"].le(midpoint)]),
        ("lowest_error_higher_observed_score", matches.loc[matches["observed"].gt(midpoint)]),
        ("largest_absolute_error", matches),
    ):
        candidates = candidates.loc[~candidates["patient_id"].isin([r[1]["patient_id"] for r in choices])]
        if candidates.empty:
            continue
        row = candidates.sort_values(["absolute_error", "patient_id"], ascending=[reason != "largest_absolute_error", True]).iloc[0]
        choices.append((reason, row))
    if args.case_count > len(choices):
        remainder = matches.loc[~matches["patient_id"].isin([r[1]["patient_id"] for r in choices])].sort_values(["absolute_error", "patient_id"], ascending=[False, True])
        choices.extend(("next_largest_absolute_error", row) for _, row in remainder.iterrows())
    features = [*CLINICAL_FEATURES, *radiomics]
    x = table[features].to_numpy(float)
    y = table[args.target].to_numpy(float)
    cases, contributions = [], []
    for number, (reason, row) in enumerate(choices[:args.case_count], start=1):
        fold = int(row["outer_fold"])
        test = fold - 1
        train = np.delete(np.arange(len(table)), test)
        estimator, _ = additional_model_and_grid("gradient_boosting", "early_fusion", args.random_state + fold)
        pipeline = Pipeline([("preprocess", additional_preprocessor("early_fusion", len(radiomics))), ("model", estimator)])
        pipeline.set_params(**json.loads(row["best_parameters"]))
        pipeline.fit(x[train], y[train])
        prediction = float(pipeline.predict(x[[test]])[0])
        if not np.isclose(prediction, row["predicted"], atol=1e-7, rtol=0):
            raise RuntimeError("Held-out case refit does not reproduce the stored OOF prediction")
        transformed = np.asarray(pipeline.named_steps["preprocess"].transform(x[[test]]))
        variance = pipeline.named_steps["preprocess"].named_transformers_["radiomics"].named_steps["variance"]
        names = [*CLINICAL_FEATURES, *np.asarray(radiomics)[variance.get_support()].tolist()]
        explainer = shap.TreeExplainer(pipeline.named_steps["model"])
        values = np.asarray(explainer.shap_values(transformed))[0]
        base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        if not np.isclose(base + values.sum(), prediction, atol=1e-7, rtol=0):
            raise RuntimeError("Local SHAP contributions do not reconstruct the prediction")
        cases.append({"case":number,"selection_rule":reason,"patient_id":str(row["patient_id"]),"outer_fold":fold,"target":args.target,"baseline_ipss":float(table.iloc[test]["ipss_preop"]),"observed":float(y[test]),"predicted":prediction,"absolute_error":abs(prediction-y[test]),"shap_base_value":base,"shap_sum":float(values.sum()),"training_patients":len(train),"scope":"outer-held-out; illustrative, not causal"})
        for feature,value,standardized in zip(names,values,transformed[0]):
            contributions.append({"case":number,"patient_id":str(row["patient_id"]),"target":args.target,"feature":feature,"feature_value_raw":float(table.iloc[test][feature]),"feature_value_transformed":float(standardized),"shap_value":float(value)})
        explanation = shap.Explanation(values=values,base_values=base,data=transformed[0],feature_names=names)
        shap.plots.waterfall(explanation,max_display=12,show=False)
        plt.title(f"Held-out case {number}: {OUTCOMES[args.target]}")
        plt.savefig(output / f"patient_case_{number}_shap.png",dpi=250,bbox_inches="tight")
        plt.close()
    atomic_csv(pd.DataFrame(cases),output / "patient_cases.csv")
    atomic_csv(pd.DataFrame(contributions),output / "patient_case_shap_values.csv")
    atomic_json({"target":args.target,"selection_rule":"Lowest-error case in each observed-score half, then largest error; ties resolved by patient ID. Descriptive selection after evaluation.","case_count":len(cases),"model":"early_fusion|gradient_boosting","scope":"Each patient is excluded from fitting; hyperparameters and seeds reproduce its original outer-fold model."},output / "patient_case_selection.json")
    return cases


def run(args):
    if args.permutation_repeats < 1 or args.group_permutation_repeats < 1:
        raise ValueError("Permutation repeat counts must be positive")
    table, radiomics = load_regression_data(
        args.data, args.expected_patients, args.expected_radiomics_features,
        target=args.target, feature_manifest=args.feature_manifest,
    )
    if prepare_run(args, table, radiomics, "11", (args.bootstrap_dir, args.screening_oof)):
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_results = pd.read_csv(args.bootstrap_dir / "pipeline_bootstrap_results.csv")
    valid = valid_replicates(bootstrap_results, args.max_valid_replicates)
    draws = bootstrap_draws(len(table), args.random_state, max(valid))
    clinical_path = args.output_dir / "clinical_coefficients.partial.csv"
    individual_path = args.output_dir / "gradient_permutation.partial.csv"
    group_path = args.output_dir / "gradient_group_permutation.partial.csv"
    clinical = read_checkpoint(clinical_path, args.resume)
    individual = read_checkpoint(individual_path, args.resume)
    grouped = read_checkpoint(group_path, args.resume)
    complete = set(clinical["bootstrap_replicate"].astype(int)) if not clinical.empty else set()
    complete &= set(individual["bootstrap_replicate"].astype(int)) if not individual.empty else set()
    complete &= set(grouped["bootstrap_replicate"].astype(int)) if not grouped.empty else set()
    pending = [replicate for replicate in valid if replicate not in complete]
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset:offset + args.batch_size]
        with parallel_config(backend="loky", inner_max_num_threads=1):
            fitted = Parallel(n_jobs=args.n_jobs)(
                delayed(interpret_one)(
                    replicate, *draws[replicate], table, radiomics,
                    bootstrap_results, args,
                )
                for replicate in batch
            )
        batch_set = set(batch)
        if not clinical.empty:
            clinical = clinical.loc[~clinical["bootstrap_replicate"].isin(batch_set)]
            individual = individual.loc[~individual["bootstrap_replicate"].isin(batch_set)]
            grouped = grouped.loc[~grouped["bootstrap_replicate"].isin(batch_set)]
        clinical = pd.concat([clinical, pd.DataFrame([r for item in fitted for r in item[0]])], ignore_index=True)
        individual = pd.concat([individual, pd.DataFrame([r for item in fitted for r in item[1]])], ignore_index=True)
        grouped = pd.concat([grouped, pd.DataFrame([r for item in fitted for r in item[2]])], ignore_index=True)
        atomic_csv(clinical, clinical_path)
        atomic_csv(individual, individual_path)
        atomic_csv(grouped, group_path)
        print(f"Interpretability checkpoint: {min(offset + len(batch), len(pending))}/{len(pending)}", flush=True)

    clinical_summary = summarize_values(
        clinical, "feature", "standardized_coefficient"
    )
    clinical_summary["mean_absolute_coefficient"] = (
        clinical.groupby("feature")["absolute_standardized_coefficient"].mean()
        .reindex(clinical_summary["feature"]).to_numpy()
    )
    permutation_summary = summarize_values(individual, "feature", "oob_mae_increase")
    group_summary = summarize_values(grouped, "feature_group", "oob_mae_increase")
    frequency = lasso_frequency(args.bootstrap_dir, args.output_dir)
    shap_ranking, modal_parameters = shap_analysis(
        args.screening_oof, table, radiomics, args.output_dir, args.random_state, args.target
    )
    cases = patient_shap_cases(args.screening_oof, table, radiomics, args.output_dir, args)
    atomic_csv(clinical, args.output_dir / "clinical_bootstrap_coefficients.csv")
    atomic_csv(clinical_summary, args.output_dir / "clinical_coefficient_summary.csv")
    atomic_csv(individual, args.output_dir / "gradient_boosting_permutation_values.csv")
    atomic_csv(permutation_summary, args.output_dir / "gradient_boosting_permutation_summary.csv")
    atomic_csv(grouped, args.output_dir / "gradient_boosting_group_permutation_values.csv")
    atomic_csv(group_summary, args.output_dir / "gradient_boosting_group_permutation_summary.csv")
    plot_importance(
        permutation_summary, "feature",
        args.output_dir / "gradient_boosting_permutation_importance.png",
        "Gradient boosting: bootstrap OOB permutation importance",
    )
    top_clinical = clinical_summary.iloc[
        clinical_summary["mean_absolute_coefficient"].argmax()
    ]
    atomic_json({
        "status": "pass",
        **run_metadata(args),
        "bootstrap_replicates": len(valid),
        "interpretation_scope": (
            "Bootstrap coefficients and permutation importance assess stability; "
            "full-cohort SHAP is descriptive and not an out-of-sample performance estimate."
        ),
        "top_clinical_by_mean_absolute_standardized_coefficient": str(top_clinical["feature"]),
        "top_gradient_permutation_feature": str(permutation_summary.iloc[0]["feature"]),
        "top_lasso_selected_feature": str(frequency.iloc[0]["selected_feature"]) if not frequency.empty else None,
        "top_shap_feature": str(shap_ranking.iloc[0]["feature"]),
        "shap_modal_hyperparameters": modal_parameters,
        "held_out_patient_case_count": len(cases),
        "positive_shap_meaning": "Higher predicted target score; for ipss_3m this means more residual symptoms",
    }, args.output_dir / "summary.json")
    clinical_path.unlink(missing_ok=True)
    individual_path.unlink(missing_ok=True)
    group_path.unlink(missing_ok=True)
    finish_run(args)


if __name__ == "__main__":
    run_stage(run, arguments())
