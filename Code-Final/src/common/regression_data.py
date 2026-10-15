"""Frozen regression endpoints, predictor schema and reproducible run records."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .io_utils import atomic_csv, atomic_json

EXECUTION_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = Path(os.environ.get("PAE_CODE_ROOT", str(EXECUTION_ROOT)))
PROJECT_ROOT = CODE_ROOT.parent
DEFAULT_DATA = PROJECT_ROOT / "outputs/vigneswaran_radiomics_fusion_regression/matched_complete_modelling_data.csv"
DEFAULT_SCHEMA = CODE_ROOT / "config/regression_feature_schema.json"
TARGET = "ipss_3m"
TARGETS = (TARGET, "ipss_change_3m")
OUTCOMES = {
    "ipss_3m": "3-month IPSS (raw IPSS points)",
    "ipss_change_3m": "baseline IPSS - 3-month IPSS (raw IPSS points)",
}
CLINICAL_FEATURES = [
    "age_at_pae", "prostate_volume_preop_cc", "abrams_griffiths_number",
    "qmax_preop", "pvr_preop", "ipss_preop",
]
NON_PREDICTORS = {
    "patient_id", *TARGETS, "ipss_improvement_3m_pct", "responder_25_3m",
}
NON_RADIOMICS = {*NON_PREDICTORS, *CLINICAL_FEATURES}
STAGE_FOLDERS = {
    "07": "baseline", "08": "ablation", "09": "model_screen",
    "10": "bootstrap", "11": "interpretation",
}


def output_root(target=TARGET):
    return PROJECT_ROOT / "outputs" / ("ipss_3m_regression" if target == TARGET else "ipss_change_3m_regression")


def add_regression_arguments(parser):
    parser.add_argument("--target", choices=TARGETS, default=TARGET)
    parser.add_argument("--feature-manifest", type=Path, default=DEFAULT_SCHEMA)


def load_regression_data(path, expected_patients=148, expected_radiomics=119,
                         target=TARGET, feature_manifest=DEFAULT_SCHEMA):
    """Read the frozen outcome unchanged; never infer predictors from leftovers."""
    if target not in TARGETS:
        raise ValueError(f"Unsupported target: {target}")
    schema = json.loads(Path(feature_manifest).read_text(encoding="utf-8"))
    clinical = schema["clinical_features"]
    raw_radio = schema["radiomics_features"]
    if clinical != CLINICAL_FEATURES or len(raw_radio) != expected_radiomics:
        raise ValueError("Feature manifest does not match the required 6+119 schema")
    if len(raw_radio) != len(set(raw_radio)) or set(raw_radio) & NON_RADIOMICS:
        raise ValueError("Feature manifest includes duplicate or outcome predictors")
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    if len(header) != len(set(header)):
        raise ValueError("Dataset has duplicate column names")
    table = pd.read_csv(path, dtype={"patient_id": "string"})
    required = {"patient_id", "ipss_3m", *clinical}
    if required - set(table):
        raise ValueError(f"Missing required columns: {sorted(required - set(table))}")
    prefixed = [f"radiomics__{name}" for name in raw_radio]
    if set(raw_radio) <= set(table):
        radiomics = raw_radio
    elif set(prefixed) <= set(table):
        radiomics = prefixed
    else:
        raise ValueError("Dataset radiomics do not match the frozen feature manifest")
    unexpected = set(table) - {*NON_RADIOMICS, *radiomics}
    if unexpected:
        raise ValueError(f"Unexpected columns outside frozen schema: {sorted(unexpected)}")
    if len(table) != expected_patients:
        raise ValueError(f"Expected {expected_patients} patients, found {len(table)}")
    ids = table["patient_id"]
    if ids.isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError("Blank or duplicate patient IDs")
    numeric = [*clinical, *radiomics, "ipss_3m"]
    table[numeric] = table[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(table[numeric].to_numpy(float)).all():
        raise ValueError("Regression data contain missing or infinite values")
    for column in ("ipss_preop", "ipss_3m"):
        if not table[column].between(0, 35).all():
            raise ValueError(f"IPSS outside 0..35: {column}")
    change = table["ipss_preop"] - table["ipss_3m"]
    if "ipss_change_3m" in table:
        supplied = pd.to_numeric(table["ipss_change_3m"], errors="raise")
        if not np.allclose(supplied, change, atol=1e-8, rtol=0):
            raise ValueError("Stored IPSS change is inconsistent")
    elif target == "ipss_change_3m":
        table["ipss_change_3m"] = change
    return table, list(radiomics)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions():
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "matplotlib", "shap", "xgboost", "threadpoolctl")
    return {name: importlib.metadata.version(name) for name in packages}


def read_checkpoint(path, resume):
    if not resume or not Path(path).exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def validate_oof(frame, table, target, configurations, expected_folds=None):
    """Reject duplicate, mislabeled, incomplete or non-finite held-out predictions."""
    needed = {"patient_id", "outer_fold", "configuration", "observed", "predicted"}
    if needed - set(frame):
        raise ValueError("OOF schema is incomplete")
    if frame.duplicated(["outer_fold", "configuration"]).any() or frame.duplicated(["patient_id", "configuration"]).any():
        raise ValueError("Duplicate OOF predictions")
    if not set(frame["configuration"]) <= set(configurations):
        raise ValueError("Unexpected OOF configuration")
    folds = pd.to_numeric(frame["outer_fold"], errors="raise").to_numpy(float)
    if not np.isfinite(folds).all() or not np.equal(folds, np.floor(folds)).all() or (folds < 1).any() or (folds > len(table)).any():
        raise ValueError("Invalid OOF fold indices")
    index = folds.astype(int) - 1
    if frame["patient_id"].astype(str).tolist() != table.iloc[index]["patient_id"].astype(str).tolist():
        raise ValueError("OOF patient/fold mismatch")
    if not np.isfinite(frame[["observed", "predicted"]].to_numpy(float)).all():
        raise ValueError("Non-finite OOF results")
    if not np.allclose(frame["observed"], table.iloc[index][target], atol=1e-8, rtol=0):
        raise ValueError("OOF target mismatch")
    if expected_folds is not None:
        if len(frame) != expected_folds * len(configurations):
            raise ValueError("Incomplete OOF coverage")
        for name in configurations:
            actual = set(frame.loc[frame["configuration"].eq(name), "outer_fold"].astype(int))
            if actual != set(range(1, expected_folds + 1)):
                raise ValueError("Incomplete OOF coverage")


def validate_dependency(path, args):
    """Require a completed dependency with the same endpoint, data and seed."""
    path = Path(path)
    folder = path if path.is_dir() else path.parent
    record = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    identity = record["identity"]
    if record.get("phase") != "completed":
        raise ValueError(f"Dependency is not completed: {folder}")
    if identity["target"] != args.target or identity["data_sha256"] != sha256(args.data) or identity["options"]["random_state"] != args.random_state:
        raise ValueError(f"Dependency endpoint/data/seed mismatch: {folder}")
    artifact = folder / "summary.json" if path.is_dir() else path
    expected = record.get("artifacts", {}).get(artifact.name)
    if expected is None or sha256(artifact) != expected:
        raise ValueError(f"Dependency artifact changed: {artifact}")
    return record


def prepare_run(args, table, radiomics, stage, dependencies=()):
    """Freeze scientific identity before reading a checkpoint or writing results."""
    for key in ("n_jobs", "batch_size", "bootstrap_resamples", "inner_folds", "max_outer_folds", "max_valid_replicates"):
        value = getattr(args, key, None)
        if value is not None and int(value) < 1:
            raise ValueError(f"{key} must be positive")
    if hasattr(args, "inner_folds") and not 2 <= args.inner_folds < len(table):
        raise ValueError("Invalid inner-fold count")
    if getattr(args, "max_outer_folds", None) is not None and args.max_outer_folds > len(table):
        raise ValueError("max-outer-folds exceeds the cohort size")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dependencies = [Path(path) for path in dependencies]
    for path in dependencies:
        validate_dependency(path, args)
    excluded = {"resume", "output_dir", "data", "feature_manifest", "n_jobs", "batch_size"}
    options = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key not in excluded and not key.startswith("_")}
    sources = [*sorted((CODE_ROOT / "src/common").glob("*.py")), CODE_ROOT / "src" / {
        "07": "07_baseline_regression.py", "08": "08_pca_lasso_ablation.py", "09": "09_additional_regression_models.py", "10": "10_bootstrap_stability.py", "11": "11_feature_importance.py",
    }[stage]]
    for source in sources:
        executed = EXECUTION_ROOT / source.relative_to(CODE_ROOT)
        if executed != source and sha256(executed) != sha256(source):
            raise ValueError("Execution copy differs from project source")
    identity = {
        "schema_version": 1, "stage": stage, "target": args.target,
        "data_sha256": sha256(args.data), "feature_manifest_sha256": sha256(args.feature_manifest),
        "clinical_features": CLINICAL_FEATURES, "radiomics_features": radiomics,
        "patient_order_sha256": hashlib.sha256(json.dumps(table["patient_id"].astype(str).tolist()).encode()).hexdigest(),
        "options": options, "python_version": platform.python_version(), "packages": package_versions(),
        "code_sha256": {str(path.relative_to(CODE_ROOT)): sha256(path) for path in sources},
        "dependencies_sha256": {str(path): sha256(path / "run_manifest.json" if path.is_dir() else path) for path in dependencies},
    }
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    args._run_metadata = {"target": args.target, "outcome": OUTCOMES[args.target], "run_signature": signature, "clinical_feature_count": 6, "radiomics_feature_count": len(radiomics)}
    record_path = output / "run_manifest.json"
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["identity"] != identity:
            raise ValueError("Run identity changed (endpoint/data/code/environment/config); use a new output directory")
        if not args.resume:
            raise ValueError("Output already contains a run; use --resume or a new output directory")
        if record.get("phase") == "completed":
            for name, expected in record["artifacts"].items():
                if not (output / name).is_file() or sha256(output / name) != expected:
                    raise ValueError(f"Completed output changed or missing: {name}")
            print(f"Stage {stage}: verified completed outputs; no refitting needed", flush=True)
            return True
    else:
        occupied = [path for path in output.iterdir() if path.name != ".run.lock"]
        if occupied:
            raise ValueError("Existing results without a run manifest cannot be resumed or overwritten")
        atomic_json({"identity": identity, "run_signature": signature, "phase": "running", "runtime": {"python_executable": sys.executable, "thread_limit": 1}}, record_path)
    if stage in {"07", "08", "09"}:
        args._run_metadata["validation_scope"] = "full_cohort" if args.max_outer_folds is None else "smoke_test"
    elif stage == "10":
        args._run_metadata["validation_scope"] = "full_bootstrap" if args.required_valid_resamples >= 500 else "smoke_test"
    else:
        args._run_metadata["validation_scope"] = "smoke_test" if args.max_valid_replicates is not None or validate_dependency(args.bootstrap_dir, args)["identity"]["options"]["required_valid_resamples"] < 500 else "full_bootstrap"
    args._run_metadata["status"] = "smoke_test" if args._run_metadata["validation_scope"] == "smoke_test" else "pass"
    return False


def run_metadata(args):
    return args._run_metadata


def finish_run(args):
    path = args.output_dir / "run_manifest.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["phase"] = "completed"
    record["artifacts"] = {item.name: sha256(item) for item in sorted(args.output_dir.iterdir()) if item.is_file() and item.name not in {"run_manifest.json", ".run.lock"} and ".partial." not in item.name}
    atomic_json(record, path)


@contextmanager
def exclusive_run_lock(output):
    """OS-level lock releases automatically if a process crashes."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Another process is using {output}") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_stage(function, args):
    with exclusive_run_lock(args.output_dir), threadpool_limits(limits=1):
        function(args)
