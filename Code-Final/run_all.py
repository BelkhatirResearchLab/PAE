"""Single entry point for the numbered reproducible pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
STEPS = tuple(f"{number:02d}" for number in range(13))


def arguments():
    parser = argparse.ArgumentParser(description="Run the final PAE pipeline.")
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "config/pipeline_config.json",
    )
    parser.add_argument("--step", choices=("all", *STEPS), default="all")
    parser.add_argument("--from-step", choices=STEPS, default="00")
    parser.add_argument("--to-step", choices=STEPS, default="12")
    parser.add_argument("--mode", choices=("full", "quick"), default="full")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--matlab", default="matlab")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_settings(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {path}\n"
            "Copy config/pipeline_config.example.json to "
            "config/pipeline_config.json and edit the local paths."
        )
    with path.open(encoding="utf-8") as handle:
        settings = json.load(handle)
    required = {
        "data_root", "clinical_xlsx", "session_root", "dicom_root",
        "cerr_path", "random_state", "pca_variance",
        "classification_threshold_percent",
    }
    missing = required - set(settings)
    if missing:
        raise ValueError(f"Configuration keys missing: {sorted(missing)}")
    return settings


def selected_steps(args):
    if args.step != "all":
        return [args.step]
    start = STEPS.index(args.from_step)
    stop = STEPS.index(args.to_step)
    if start > stop:
        raise ValueError("--from-step must not follow --to-step")
    return list(STEPS[start:stop + 1])


def python_command(python, script, *options):
    return [str(python), str(ROOT / "src" / script), *map(str, options)]


def commands(args, settings):
    seed = settings["random_state"]
    pca = settings["pca_variance"]
    n_jobs = settings.get("n_jobs", 8)
    manifest = ROOT / "config/session_series_manifest.csv"
    quick_root = PROJECT_ROOT / "outputs_quick"
    regression_data = Path(settings.get("regression_data", PROJECT_ROOT / "outputs/vigneswaran_radiomics_fusion_regression/matched_complete_modelling_data.csv"))
    classification_data = PROJECT_ROOT / "outputs_final/06_modelling_data/classification_dataset.csv"
    classification_manifest = PROJECT_ROOT / "outputs_final/06_modelling_data/classification_feature_manifest.csv"
    command = {}
    matlab_batch = (
        f"sessionRoot='{str(settings['session_root']).replace(chr(39), chr(39) * 2)}'; "
        f"dicomRoot='{str(settings['dicom_root']).replace(chr(39), chr(39) * 2)}'; "
        f"manifestPath='{str(manifest).replace(chr(39), chr(39) * 2)}'; "
        f"run('{str(ROOT / 'src/step00_build_session_manifest.m').replace(chr(39), chr(39) * 2)}');"
    )
    command["00"] = [args.matlab, "-batch", matlab_batch]
    command["01"] = python_command(
        args.python, "01_match_ct_masks.py",
        "--data-root", settings["data_root"],
        "--session-manifest", manifest,
    )
    command["02"] = python_command(args.python, "02_export_segmentation_review.py")
    command["03"] = python_command(
        args.python, "03_prepare_cerr_inputs.py", "--session-manifest", manifest
    )
    command["04"] = python_command(
        args.python, "04_extract_cerr_radiomics.py",
        "--cerr-path", settings["cerr_path"], "--matlab", args.matlab,
    )
    command["05"] = python_command(args.python, "05_build_and_qc_radiomics_table.py")
    command["06"] = python_command(
        args.python, "06_prepare_modelling_data.py",
        "--clinical-xlsx", settings["clinical_xlsx"],
        "--response-threshold-percent", settings["classification_threshold_percent"],
    )

    outputs = {
        step: quick_root / {
            "07": "07_baseline_regression",
            "08": "08_pca_lasso_ablation",
            "09": "09_additional_regression_models",
            "10": "10_bootstrap_stability",
            "11": "11_feature_importance",
            "12": "12_secondary_classification",
        }[step]
        for step in STEPS[7:]
    } if args.mode == "quick" else {}
    common = ["--random-state", seed, "--pca-variance", pca]
    command["07"] = python_command(
        args.python, "07_baseline_regression.py", "--data", regression_data, *common,
        "--n-jobs", n_jobs, "--models", "ridge",
        *(["--output-dir", outputs["07"], "--max-outer-folds", 4] if outputs else []),
    )
    ablation_output = outputs.get("08", PROJECT_ROOT / "outputs_final/08_pca_lasso_ablation")
    command["08"] = python_command(
        args.python, "08_pca_lasso_ablation.py", "--data", regression_data, *common,
        "--n-jobs", n_jobs,
        "--bootstrap-resamples", (
            1_000 if outputs else settings.get("oof_bootstrap_resamples", 10_000)
        ),
        *(["--output-dir", outputs["08"], "--max-outer-folds", 4,
           ] if outputs else []),
    )
    command["09"] = python_command(
        args.python, "09_additional_regression_models.py",
        "--data", regression_data, "--random-state", seed, "--n-jobs", n_jobs,
        "--reference-oof", ablation_output / "matched_oof_predictions.csv",
        *(["--output-dir", outputs["09"], "--max-outer-folds", 4,
           "--bootstrap-resamples", 1_000,
           "--models", "ridge", "ard", "gradient_boosting",
           "--modalities", "clinical", "early_fusion"] if outputs else []),
    )
    command["10"] = python_command(
        args.python, "10_bootstrap_stability.py", "--data", regression_data, *common,
        "--n-jobs", min(int(n_jobs), 7),
        "--required-valid-resamples", (
            3 if outputs else settings.get("pipeline_bootstrap_resamples", 500)
        ),
        "--max-attempts", (
            10 if outputs else settings.get("pipeline_bootstrap_max_attempts", 650)
        ),
        *(["--output-dir", outputs["10"]] if outputs else []),
    )
    bootstrap_output = outputs.get("10", PROJECT_ROOT / "outputs_final/10_bootstrap_stability")
    screening_output = outputs.get("09", PROJECT_ROOT / "outputs_final/09_additional_regression_models")
    command["11"] = python_command(
        args.python, "11_feature_importance.py",
        "--data", regression_data, "--random-state", seed, "--n-jobs", n_jobs,
        "--bootstrap-dir", bootstrap_output,
        "--screening-oof", screening_output / "additional_models_oof.csv",
        "--permutation-repeats", (
            2 if outputs else settings.get("permutation_repeats", 5)
        ),
        "--group-permutation-repeats", (
            3 if outputs else settings.get("group_permutation_repeats", 20)
        ),
        *(["--output-dir", outputs["11"], "--max-valid-replicates", 3,
           ] if outputs else []),
    )
    command["12"] = python_command(
        args.python, "12_secondary_classification.py",
        "--data", classification_data, "--manifest", classification_manifest,
        "--random-state", seed, "--pca-variance", pca, "--n-jobs", n_jobs,
        *(["--output-dir", outputs["12"]] if outputs else []),
    )
    target = settings.get("regression_target", "ipss_3m")
    regression_root = Path(settings.get("regression_output_root", PROJECT_ROOT / "outputs" / ("ipss_3m_regression" if target == "ipss_3m" else "ipss_change_3m_regression")))
    if args.mode == "quick":
        regression_root = regression_root / "quick"
    folders = {"07":"baseline", "08":"ablation", "09":"model_screen", "10":"bootstrap", "11":"interpretation"}
    def set_option(values, key, value):
        if key in values:
            values[values.index(key) + 1] = str(value)
        else:
            values.extend([key, str(value)])
    for stage, folder in folders.items():
        set_option(command[stage], "--output-dir", regression_root / folder)
        set_option(command[stage], "--target", target)
        set_option(command[stage], "--feature-manifest", ROOT / "config/regression_feature_schema.json")
    set_option(command["09"], "--reference-oof", regression_root / "ablation/matched_oof_predictions.csv")
    set_option(command["11"], "--bootstrap-dir", regression_root / "bootstrap")
    set_option(command["11"], "--screening-oof", regression_root / "model_screen/additional_models_oof.csv")
    if args.resume:
        for step in ("07", "08", "09", "10", "11"):
            command[step].append("--resume")
    return command


def main():
    args = arguments()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    settings = load_settings(config_path.resolve())
    all_commands = commands(args, settings)
    for step in selected_steps(args):
        command = all_commands[step]
        print(f"\n[{step}/12] {subprocess.list2cmdline(command)}", flush=True)
        if not args.dry_run:
            environment = os.environ.copy()
            environment.update({key:"1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
            environment["PYTHONHASHSEED"] = str(settings["random_state"])
            subprocess.run(command, cwd=ROOT, check=True, env=environment)
    print("\nSelected pipeline steps completed successfully.", flush=True)


if __name__ == "__main__":
    main()
