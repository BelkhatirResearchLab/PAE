"""Run stages 07-11 on the frozen cohort, without rerunning image extraction."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

EXECUTION_ROOT = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("PAE_CODE_ROOT", str(EXECUTION_ROOT)))
# Avoid repeatedly searching a UNC directory for every scientific import.
if os.name == "nt" and str(EXECUTION_ROOT).startswith("\\\\"):
    import shutil
    import tempfile
    import time
    cache = Path(tempfile.mkdtemp(prefix="pae_ipss3m_code_"))
    files = [*EXECUTION_ROOT.glob("*.py"), *EXECUTION_ROOT.glob("src/**/*.py")]
    for source in files:
        destination = cache / source.relative_to(EXECUTION_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(6):
            try:
                destination.write_bytes(source.read_bytes())
                break
            except OSError:
                if attempt == 5: raise
                time.sleep(0.5)
    environment = os.environ.copy()
    environment["PAE_CODE_ROOT"] = str(ROOT)
    result = subprocess.call([sys.executable, "-u", str(cache / "run_ipss3m.py"), *sys.argv[1:]], env=environment, cwd=cache)
    raise SystemExit(result)
sys.path.insert(0, str(EXECUTION_ROOT / "src"))
from common.io_utils import atomic_csv, atomic_json as _atomic_json
from common.regression_data import (
    CLINICAL_FEATURES, OUTCOMES, STAGE_FOLDERS, TARGETS, exclusive_run_lock,
    load_regression_data, package_versions, sha256,
)

def atomic_json(value, path, _retry_timeout=120.0):
    """Retry transient Windows sharing errors without changing atomic content."""
    deadline = time.monotonic() + _retry_timeout
    delay = 0.2
    announced = False
    while True:
        try:
            return _atomic_json(value, path)
        except OSError as error:
            if getattr(error, "winerror", None) not in {5, 32, 33} or time.monotonic() >= deadline:
                raise
            if not announced:
                print(f"Transient Windows sharing error writing {path.name}; retrying atomic replacement", flush=True)
                announced = True
            time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
            delay = min(delay * 1.5, 2.0)


SCRIPTS = {"07":"07_baseline_regression.py", "08":"08_pca_lasso_ablation.py", "09":"09_additional_regression_models.py", "10":"10_bootstrap_stability.py", "11":"11_feature_importance.py"}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/regression_ipss3m.json")
    parser.add_argument("--mode", choices=("full", "quick"), default="full")
    parser.add_argument("--from-stage", choices=SCRIPTS, default="07")
    parser.add_argument("--to-stage", choices=SCRIPTS, default="11")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--require-smoke", type=Path, help="Before stages 08-11 require a completed matching five-stage smoke run")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def build_commands(args, settings):
    target = settings["target"]
    if target not in TARGETS:
        raise ValueError("Unsupported target")
    data = resolve(args.data or settings["data"])
    schema = resolve(settings["feature_manifest"])
    base = resolve(args.output_root or settings["output_root"])
    output = base / "quick" if args.mode == "quick" else base
    folders = {stage:output / name for stage,name in STAGE_FOLDERS.items()}
    shared = ["--data", str(data), "--target", target, "--feature-manifest", str(schema), "--expected-patients", str(settings["expected_patients"]), "--expected-radiomics-features", str(settings["expected_radiomics_features"]), "--random-state", str(settings["random_state"]), "--n-jobs", str(settings["n_jobs"])]
    commands = {}
    for stage, script in SCRIPTS.items():
        command = [sys.executable, "-u", str(EXECUTION_ROOT / "src" / script), *shared, "--output-dir", str(folders[stage])]
        if stage != "11":
            command += ["--pca-variance", str(settings["pca_variance"])] if stage != "09" else []
            command += ["--inner-folds", str(settings["inner_folds"])]
        if stage in {"07", "08", "09"} and args.mode == "quick":
            command += ["--max-outer-folds", "2"]
        if stage in {"08", "09"}:
            command += ["--bootstrap-resamples", str(1000 if args.mode == "quick" else settings["oof_bootstrap_resamples"]), "--batch-size", "2"]
        if stage == "07":
            command += ["--models", "ridge"]
        if stage == "09":
            command += ["--reference-oof", str(folders["08"] / "matched_oof_predictions.csv")]
            if args.mode == "quick":
                command += ["--models", "ridge", "ard", "gradient_boosting"]
        if stage == "10":
            command += ["--required-valid-resamples", str(3 if args.mode == "quick" else settings["pipeline_bootstrap_resamples"]), "--max-attempts", str(10 if args.mode == "quick" else settings["pipeline_bootstrap_max_attempts"])]
        if stage == "11":
            command += ["--bootstrap-dir", str(folders["10"]), "--screening-oof", str(folders["09"] / "additional_models_oof.csv"), "--permutation-repeats", str(2 if args.mode == "quick" else settings["permutation_repeats"]), "--group-permutation-repeats", str(3 if args.mode == "quick" else settings["group_permutation_repeats"]), "--batch-size", str(2 if args.mode == "quick" else settings.get("interpretation_batch_size", 20))]
            if args.mode == "quick":
                command += ["--max-valid-replicates", "3"]
        if args.resume:
            command.append("--resume")
        commands[stage] = command
    return data, schema, output, commands


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    args = arguments()
    settings = json.loads(resolve(args.config).read_text(encoding="utf-8"))
    data, schema, output, commands = build_commands(args, settings)
    stages = [s for s in SCRIPTS if args.from_stage <= s <= args.to_stage]
    if not stages:
        raise ValueError("Invalid stage range")
    if args.dry_run:
        for stage in stages:
            print(subprocess.list2cmdline(commands[stage]), flush=True)
        return
    environment = os.environ.copy()
    environment.update({key:"1" for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS")})
    environment["PYTHONHASHSEED"] = str(settings["random_state"])
    environment["PYTHONUNBUFFERED"] = "1"
    with exclusive_run_lock(output):
        table, radio = load_regression_data(data, settings["expected_patients"], settings["expected_radiomics_features"], settings["target"], schema)
        audit = output / "data_audit"
        audit.mkdir(parents=True, exist_ok=True)
        audit_record = {"target":settings["target"],"outcome":OUTCOMES[settings["target"]],"patients":len(table),"unique_patients":int(table["patient_id"].nunique()),"clinical_features":CLINICAL_FEATURES,"radiomics_feature_count":len(radio),"data_sha256":sha256(data),"feature_schema_sha256":sha256(schema),"preprocessing":"none; fitted inside training folds","source":str(data),"packages":package_versions()}
        old_audit = audit / "dataset_audit.json"
        if old_audit.exists() and json.loads(old_audit.read_text(encoding="utf-8")) != audit_record:
            raise ValueError("Frozen data audit changed; choose a new output root")
        atomic_json(audit_record, old_audit)
        # Keep an exact byte-for-byte data snapshot, not a float-formatted re-export.
        frozen = audit / "frozen_modelling_dataset.csv"
        if not frozen.exists() or sha256(frozen) != audit_record["data_sha256"]:
            temporary = audit / ".frozen_modelling_dataset.csv.tmp"
            temporary.write_bytes(data.read_bytes())
            if sha256(temporary) != audit_record["data_sha256"]:
                raise ValueError("Source dataset changed while making the frozen snapshot")
            os.replace(temporary, frozen)
        frozen_schema = audit / "regression_feature_schema.json"
        if not frozen_schema.exists() or sha256(frozen_schema) != audit_record["feature_schema_sha256"]:
            temporary_schema = audit / ".regression_feature_schema.json.tmp"
            temporary_schema.write_bytes(schema.read_bytes())
            if sha256(temporary_schema) != audit_record["feature_schema_sha256"]:
                raise ValueError("Feature schema changed while making its frozen snapshot")
            os.replace(temporary_schema, frozen_schema)
        for values in commands.values():
            values[values.index("--data") + 1] = str(frozen)
            values[values.index("--feature-manifest") + 1] = str(frozen_schema)
        atomic_csv(__import__("pandas").DataFrame([{"feature":c,"role":"clinical"} for c in CLINICAL_FEATURES]+[{"feature":c,"role":"radiomics"} for c in radio]), audit / "predictor_manifest.csv")
        resolved = {**settings, "data":str(data), "feature_manifest":str(schema), "output_root":str(output), "mode":args.mode, "execution_data":str(frozen), "execution_feature_manifest":str(frozen_schema)}
        atomic_json(resolved, audit / "resolved_settings.json")
        logs = output / "logs"
        logs.mkdir(exist_ok=True)
        state = {"status":"running", "mode":args.mode, "target":settings["target"], "started_utc":now(), "completed_stages":[], "requested_stages":stages, "runner_pid":os.getpid()}
        atomic_json(state, output / "pipeline_status.json")
        for stage in stages:
            if args.require_smoke is not None and stage != "07":
                smoke = resolve(args.require_smoke)
                while True:
                    smoke_state = json.loads((smoke / "pipeline_status.json").read_text(encoding="utf-8"))
                    if smoke_state["status"] == "failed":
                        state.update({"status":"failed", "reason":"Required smoke test failed", "updated_utc":now()})
                        atomic_json(state, output / "pipeline_status.json")
                        raise RuntimeError("Required smoke test failed; full downstream stages were not started")
                    if smoke_state["status"] == "smoke_test_completed":
                        smoke_audit = json.loads((smoke / "data_audit/dataset_audit.json").read_text(encoding="utf-8"))
                        if set(smoke_state["completed_stages"]) != set(SCRIPTS) or smoke_state["target"] != settings["target"] or smoke_audit["data_sha256"] != audit_record["data_sha256"] or smoke_audit["feature_schema_sha256"] != audit_record["feature_schema_sha256"]:
                            raise ValueError("Required smoke test does not cover the same full pipeline, endpoint and frozen data")
                        state["smoke_validation"] = str(smoke)
                        break
                    state.update({"status":"waiting_for_smoke_test", "updated_utc":now()})
                    atomic_json(state, output / "pipeline_status.json")
                    time.sleep(10)
            state.update({"status":"running"})
            state.update({"current_stage":stage,"updated_utc":now(),"log":str(logs / f"{stage}.log")})
            atomic_json(state, output / "pipeline_status.json")
            print(f"Stage {stage}: {STAGE_FOLDERS[stage]} (log: {state['log']})",flush=True)
            with (logs / f"{stage}.log").open("a",encoding="utf-8") as log:
                log.write("\n"+now()+" "+subprocess.list2cmdline(commands[stage])+"\n")
                log.flush()
                process = subprocess.Popen(commands[stage],cwd=EXECUTION_ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
                state["stage_pid"] = process.pid
                atomic_json(state, output / "pipeline_status.json")
                result = process.wait()
            if result:
                state.update({"status":"failed","exit_code":result,"updated_utc":now()})
                atomic_json(state, output / "pipeline_status.json")
                raise RuntimeError(f"Stage {stage} failed; inspect {state['log']}")
            state["completed_stages"].append(stage)
            atomic_json(state, output / "pipeline_status.json")
        state.update({"status":"smoke_test_completed" if args.mode == "quick" else "completed","updated_utc":now()})
        atomic_json(state, output / "pipeline_status.json")
        print(f"Completed requested stages: {output}",flush=True)


if __name__ == "__main__":
    main()
