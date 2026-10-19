"""Fail-closed phase runner for Exp039 locked global-temperature confirmation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ACTIVE_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PRE_REGISTRATION_PATH = EXPERIMENT_ROOT / "pre_registration.json"
EXP037_ROOT = ACTIVE_ROOT / "experiments/037_pie_joint_biu_temporal_ego"
EXP038_ROOT = ACTIVE_ROOT / "experiments/038_pie_nested_softening_calibration"
PREREGISTRATION_SHA256 = "6aca4d5f952bf06246fd1f8c9e781b4199200d7acc255c870af3ae9436d13bb6"
EXPECTED_TEMPERATURES = {
    "joint_temporal_ego_xgb": 1.8395656892513026,
    "v2_bbox_xgb": 2.0149628788881717,
    "v1_bbox_rf": 1.0,
}
RAW_ARMS = ("joint_temporal_ego_xgb", "v2_bbox_xgb", "v1_bbox_rf")
IDENTITY = ["sequence_id", "set_id", "track_id", "video_id", "label"]

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import global_core  # noqa: E402


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _exp037():
    return _load_module(EXP037_ROOT / "scripts/run_experiment.py", "exp039_source_exp037_runner")


def _calibration():
    return _load_module(EXP038_ROOT / "scripts/calibration_core.py", "exp039_source_calibration_core")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {"sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_bytes(path, frame.to_csv(index=False).encode())


def load_preregistration(path: Path = PRE_REGISTRATION_PATH) -> dict[str, Any]:
    path = Path(path)
    if _sha256(path) != PREREGISTRATION_SHA256:
        raise ValueError("Exp039 preregistration differs from the frozen byte contract")
    registration = json.loads(path.read_text(encoding="utf-8"))
    if registration.get("temperatures") != EXPECTED_TEMPERATURES:
        raise ValueError("Exp039 temperatures differ from the frozen contract")
    return registration


def _publish(root: Path, phase: str, outputs: list[Path], **extra: Any) -> None:
    metrics = root / "metrics"
    manifest = {str(path.relative_to(metrics)): _record(path) for path in outputs}
    _write_json(metrics / f".{phase}.complete.json", {
        "phase": phase, "status": "complete", "outputs": manifest, **extra,
    })


def _authenticate_manifest(root: Path, phase: str, required: set[str] | None = None) -> dict[str, Any]:
    metrics = Path(root) / "metrics"
    marker = metrics / f".{phase}.complete.json"
    completion = json.loads(marker.read_text(encoding="utf-8"))
    if completion.get("phase") != phase or completion.get("status") != "complete":
        raise RuntimeError(f"{phase} completion marker is invalid")
    outputs = completion.get("outputs")
    if not isinstance(outputs, dict) or (required is not None and set(outputs) != required):
        raise RuntimeError(f"{phase} completion output manifest is invalid")
    for relative, expected in outputs.items():
        artifact = metrics / relative
        if not artifact.is_file() or expected != _record(artifact):
            raise RuntimeError(f"{phase} output is not authenticated: {relative}")
    return completion


def _authenticate_parent(root: Path, phase: str) -> dict[str, Any]:
    return _authenticate_manifest(root, phase)


def _temperature_lineage() -> dict[str, Any]:
    path = EXP038_ROOT / "metrics/development_temperatures.csv"
    temperatures = pd.read_csv(path)
    observed = {
        arm: float(temperatures.loc[temperatures["raw_arm"].eq(arm), "temperature"].median())
        for arm in RAW_ARMS
    }
    if observed != EXPECTED_TEMPERATURES:
        raise RuntimeError(f"Exp038 median temperatures changed: {observed}")
    if set(temperatures["outer_set"].astype(str)) != {"set01", "set02", "set04", "set05"}:
        raise RuntimeError("Exp038 temperature development set universe changed")
    return {
        "path": str(path.resolve(strict=True)),
        **_record(path),
        "median_temperatures": observed,
    }


def _authenticate_prepare(root: Path) -> dict[str, Any]:
    completion = _authenticate_manifest(
        root, "prepare", {"procedure_lock.json", "source_provenance.json"}
    )
    lock_path = Path(root) / "metrics/procedure_lock.json"
    if completion.get("lock_sha256") != _sha256(lock_path):
        raise RuntimeError("prepare completion does not authenticate lock")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "phase": "prepare",
        "status": "locked",
        "development_sets": ["set01", "set02", "set04", "set05"],
        "confirmation_set": "set06",
        "diagnostic_test_set": "set03",
        "threshold": 0.5,
        "temperatures": EXPECTED_TEMPERATURES,
        "preregistration_sha256": _sha256(Path(root) / "pre_registration.json"),
    }
    if lock != expected:
        raise RuntimeError("Exp039 procedure lock is invalid")
    return lock


def run_prepare(output_root: Path = EXPERIMENT_ROOT) -> str:
    root = Path(output_root)
    marker = root / "metrics/.prepare.complete.json"
    if marker.exists():
        _authenticate_prepare(root)
        return "locked"
    load_preregistration(root / "pre_registration.json")
    parent_records = []
    for parent, phase in ((EXP037_ROOT, "prepare"), (EXP038_ROOT, "development")):
        completion = _authenticate_parent(parent, phase)
        marker_path = parent / f"metrics/.{phase}.complete.json"
        parent_records.append({
            "experiment": parent.name,
            "phase": phase,
            "completion_path": str(marker_path.resolve(strict=True)),
            **_record(marker_path),
            "authenticated_outputs": sorted(completion["outputs"]),
        })
    verification = EXP038_ROOT / "metrics/verification_report.json"
    if json.loads(verification.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("Exp038 verification is not PASS")
    development = EXP037_ROOT / "metrics/prepared_development.csv"
    contract = EXP037_ROOT / "metrics/feature_contract.json"
    if len(pd.read_csv(development, usecols=["sequence_id"])) != 4781:
        raise RuntimeError("Exp037 development cohort changed")
    provenance = {
        "parents": parent_records,
        "development_features": {"path": str(development.resolve(strict=True)), **_record(development)},
        "feature_contract": {"path": str(contract.resolve(strict=True)), **_record(contract)},
        "temperature_lineage": _temperature_lineage(),
        "paper": {
            "path": str((ACTIVE_ROOT / "papers/Guo_2017_Calibration.pdf").resolve(strict=True)),
            **_record(ACTIVE_ROOT / "papers/Guo_2017_Calibration.pdf"),
        },
        "set06_materialized": False,
        "set03_materialized": False,
    }
    lock = {
        "schema_version": 1, "phase": "prepare", "status": "locked",
        "development_sets": ["set01", "set02", "set04", "set05"],
        "confirmation_set": "set06", "diagnostic_test_set": "set03", "threshold": 0.5,
        "temperatures": EXPECTED_TEMPERATURES,
        "preregistration_sha256": _sha256(root / "pre_registration.json"),
    }
    provenance_path = root / "metrics/source_provenance.json"
    lock_path = root / "metrics/procedure_lock.json"
    _write_json(provenance_path, provenance)
    _write_json(lock_path, lock)
    _publish(root, "prepare", [lock_path, provenance_path], lock_sha256=_sha256(lock_path))
    return "locked"


def _load_set06_features():
    return _exp037()._materialize_features(("set06",))


def _load_set03_features():
    return _exp037()._materialize_features(("set03",))


def _feature_arms() -> dict[str, tuple[str, ...]]:
    contract = json.loads((EXP037_ROOT / "metrics/feature_contract.json").read_text(encoding="utf-8"))
    return {arm: tuple(contract["arms"][arm]) for arm in RAW_ARMS}


def _metric_rows(predictions: pd.DataFrame, probability_column: str, scope: str) -> pd.DataFrame:
    calibration = _calibration()
    return pd.DataFrame([
        {
            "scope": scope, "set_id": str(rows["set_id"].iloc[0]), "arm": arm,
            **calibration.compute_metrics(
                rows["label"].to_numpy(int), rows[probability_column].to_numpy(float)
            ),
        }
        for arm, rows in predictions.groupby("arm", sort=True)
    ])


def _predict(train: pd.DataFrame, evaluation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = _exp037()
    arms = _feature_arms()
    raw_parts = []
    calibrated_parts = []
    for arm in RAW_ARMS:
        probability = source.model_core.fit_predict_seed_mean(
            train, evaluation, arms[arm], arm, source.model_core.SEEDS
        )
        raw = evaluation.loc[:, IDENTITY].copy()
        raw["arm"] = arm
        raw["probability"] = probability
        raw_parts.append(raw)
        calibrated = raw.copy()
        calibrated["raw_arm"] = arm
        calibrated["arm"] = f"soft_temp_{arm}"
        calibrated["raw_probability"] = probability
        calibrated["temperature"] = EXPECTED_TEMPERATURES[arm]
        calibrated["probability"] = global_core.apply_locked_temperature(
            probability, EXPECTED_TEMPERATURES[arm]
        )
        global_core.verify_global_invariants(
            probability, calibrated["probability"].to_numpy(float)
        )
        calibrated_parts.append(calibrated)
    return pd.concat(raw_parts, ignore_index=True), pd.concat(calibrated_parts, ignore_index=True)


def _confirmation_bootstrap(calibrated: pd.DataFrame) -> pd.DataFrame:
    source = _exp037()
    wide = calibrated.pivot(
        index=IDENTITY, columns="arm", values="probability"
    ).reset_index()
    results = _metric_rows(calibrated, "probability", "set06_confirmation_calibrated").set_index("arm")
    best_f1 = max(
        ("soft_temp_v2_bbox_xgb", "soft_temp_v1_bbox_rf"),
        key=lambda arm: float(results.loc[arm, "f1_fixed_0p5"]),
    )
    pr = source.model_core.cluster_bootstrap_deltas(
        wide, "soft_temp_v2_bbox_xgb", "soft_temp_joint_temporal_ego_xgb",
        replicates=10000, seed=42,
    ).loc[lambda frame: frame["metric"].eq("pr_auc")].assign(
        reference_arm="soft_temp_v2_bbox_xgb"
    )
    f1 = source.model_core.cluster_bootstrap_deltas(
        wide, best_f1, "soft_temp_joint_temporal_ego_xgb", replicates=10000, seed=42,
    ).loc[lambda frame: frame["metric"].eq("f1_fixed_0p5")].assign(reference_arm=best_f1)
    return pd.concat([pr, f1], ignore_index=True)


CONFIRM_OUTPUTS = {
    "confirmation_bootstrap.csv", "confirmation_features.csv", "confirmation_gate_checks.json",
    "confirmation_lock.json", "confirmation_predictions_calibrated.csv",
    "confirmation_predictions_raw.csv", "confirmation_results_calibrated.csv",
    "confirmation_results_raw.csv", "source_provenance_confirm.json",
}


def _authenticate_confirmation(root: Path) -> dict[str, Any]:
    load_preregistration(Path(root) / "pre_registration.json")
    completion = _authenticate_manifest(root, "confirm", CONFIRM_OUTPUTS)
    lock_path = Path(root) / "metrics/confirmation_lock.json"
    if completion.get("lock_sha256") != _sha256(lock_path):
        raise RuntimeError("confirmation completion does not authenticate lock")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    fixed = {
        "schema_version": 1, "phase": "confirm",
        "development_sets": ["set01", "set02", "set04", "set05"],
        "confirmation_set": "set06", "diagnostic_test_set": "set03", "threshold": 0.5,
        "temperatures": EXPECTED_TEMPERATURES,
        "preregistration_sha256": _sha256(Path(root) / "pre_registration.json"),
    }
    for key, value in fixed.items():
        if lock.get(key) != value:
            raise RuntimeError(f"confirmation lock field is invalid: {key}")
    status = lock.get("status")
    if status not in {"advance", "no_advance"} or completion.get("promotion_status") != status:
        raise RuntimeError("confirmation status is invalid")
    expected_procedure = "soft_temp_joint_temporal_ego_xgb" if status == "advance" else None
    if lock.get("confirmed_procedure") != expected_procedure:
        raise RuntimeError("confirmation procedure is inconsistent")
    gate = json.loads((Path(root) / "metrics/confirmation_gate_checks.json").read_text(encoding="utf-8"))
    if gate.get("status") != status:
        raise RuntimeError("confirmation gate and lock disagree")
    return lock


def run_confirm(output_root: Path = EXPERIMENT_ROOT) -> str:
    root = Path(output_root)
    marker = root / "metrics/.confirm.complete.json"
    if marker.exists():
        return str(_authenticate_confirmation(root)["status"])
    _authenticate_prepare(root)
    confirmation, _ = _load_set06_features()
    if len(confirmation) != 1204 or set(confirmation["set_id"].astype(str)) != {"set06"}:
        raise RuntimeError("set06 confirmation cohort changed")
    development = pd.read_csv(EXP037_ROOT / "metrics/prepared_development.csv")
    raw, calibrated = _predict(development, confirmation)
    raw_results = _metric_rows(raw, "probability", "set06_confirmation_raw")
    calibrated_results = _metric_rows(
        calibrated, "probability", "set06_confirmation_calibrated"
    )
    raw_candidate = raw.loc[raw["arm"].eq("joint_temporal_ego_xgb")]
    calibrated_candidate = calibrated.loc[
        calibrated["arm"].eq("soft_temp_joint_temporal_ego_xgb")
    ]
    global_core.verify_global_invariants(
        raw_candidate["probability"].to_numpy(float),
        calibrated_candidate["probability"].to_numpy(float),
    )
    bootstrap = _confirmation_bootstrap(calibrated)
    decision = global_core.evaluate_confirmation_gate(raw_results, calibrated_results, bootstrap)
    metrics = root / "metrics"
    frames = {
        "confirmation_features.csv": confirmation,
        "confirmation_predictions_raw.csv": raw,
        "confirmation_predictions_calibrated.csv": calibrated,
        "confirmation_results_raw.csv": raw_results,
        "confirmation_results_calibrated.csv": calibrated_results,
        "confirmation_bootstrap.csv": bootstrap,
    }
    outputs = []
    for name, frame in frames.items():
        path = metrics / name
        _write_csv(path, frame)
        outputs.append(path)
    gate_path = metrics / "confirmation_gate_checks.json"
    _write_json(gate_path, {
        "status": decision.status, "checks": decision.checks, "observed": decision.observed,
    })
    outputs.append(gate_path)
    provenance_path = metrics / "source_provenance_confirm.json"
    _write_json(provenance_path, _exp037()._source_manifest(("set06",)))
    outputs.append(provenance_path)
    lock = {
        "schema_version": 1, "phase": "confirm", "status": decision.status,
        "confirmed_procedure": (
            "soft_temp_joint_temporal_ego_xgb" if decision.status == "advance" else None
        ),
        "development_sets": ["set01", "set02", "set04", "set05"],
        "confirmation_set": "set06", "diagnostic_test_set": "set03", "threshold": 0.5,
        "temperatures": EXPECTED_TEMPERATURES,
        "preregistration_sha256": _sha256(root / "pre_registration.json"),
    }
    lock_path = metrics / "confirmation_lock.json"
    _write_json(lock_path, lock)
    outputs.append(lock_path)
    _publish(
        root, "confirm", outputs, lock_sha256=_sha256(lock_path),
        promotion_status=decision.status,
    )
    return decision.status


EVALUATE_OUTPUTS = {
    "diagnostic_set03_predictions_calibrated.csv", "diagnostic_set03_predictions_raw.csv",
    "diagnostic_set03_results_calibrated.csv", "diagnostic_set03_results_raw.csv",
    "evaluation_lock.json", "source_provenance_evaluate.json",
}


def run_evaluate(output_root: Path = EXPERIMENT_ROOT) -> str:
    root = Path(output_root)
    confirmation_lock = _authenticate_confirmation(root)
    if confirmation_lock["status"] == "no_advance":
        return "no_advance"
    marker = root / "metrics/.evaluate.complete.json"
    if marker.exists():
        _authenticate_manifest(root, "evaluate", EVALUATE_OUTPUTS)
        return "advance"
    diagnostic, _ = _load_set03_features()
    if len(diagnostic) != 3760 or set(diagnostic["set_id"].astype(str)) != {"set03"}:
        raise RuntimeError("set03 diagnostic cohort changed")
    metrics = root / "metrics"
    train = pd.concat([
        pd.read_csv(EXP037_ROOT / "metrics/prepared_development.csv"),
        pd.read_csv(metrics / "confirmation_features.csv"),
    ], ignore_index=True)
    raw, calibrated = _predict(train, diagnostic)
    raw_results = _metric_rows(raw, "probability", "set03_test_informed_raw")
    calibrated_results = _metric_rows(
        calibrated, "probability", "set03_test_informed_calibrated"
    )
    outputs = []
    for name, frame in {
        "diagnostic_set03_predictions_raw.csv": raw,
        "diagnostic_set03_predictions_calibrated.csv": calibrated,
        "diagnostic_set03_results_raw.csv": raw_results,
        "diagnostic_set03_results_calibrated.csv": calibrated_results,
    }.items():
        path = metrics / name
        _write_csv(path, frame)
        outputs.append(path)
    provenance_path = metrics / "source_provenance_evaluate.json"
    _write_json(provenance_path, _exp037()._source_manifest(("set03",)))
    outputs.append(provenance_path)
    lock_path = metrics / "evaluation_lock.json"
    _write_json(lock_path, {
        "schema_version": 1, "phase": "evaluate", "status": "complete",
        "scientific_status": "TEST_INFORMED_NOT_INDEPENDENT_CONFIRMATION",
        "training_sets": ["set01", "set02", "set04", "set05", "set06"],
        "diagnostic_test_set": "set03", "threshold": 0.5,
        "temperatures": EXPECTED_TEMPERATURES,
        "confirmation_lock_sha256": _sha256(metrics / "confirmation_lock.json"),
    })
    outputs.append(lock_path)
    _publish(root, "evaluate", outputs, lock_sha256=_sha256(lock_path))
    return "advance"


def _markdown(frame: pd.DataFrame) -> str:
    columns = [str(value) for value in frame.columns]
    lines = [f"| {' | '.join(columns)} |", f"| {' | '.join('---' for _ in columns)} |"]
    lines.extend(
        f"| {' | '.join(str(value).replace('|', chr(92) + '|') for value in row)} |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def run_report(output_root: Path = EXPERIMENT_ROOT) -> str:
    root = Path(output_root)
    lock = _authenticate_confirmation(root)
    metrics = root / "metrics"
    raw = pd.read_csv(metrics / "confirmation_results_raw.csv")
    calibrated = pd.read_csv(metrics / "confirmation_results_calibrated.csv")
    bootstrap = pd.read_csv(metrics / "confirmation_bootstrap.csv")
    gate = json.loads((metrics / "confirmation_gate_checks.json").read_text())
    lines = [
        "# Exp039 Results", "", "## Scientific Status", "",
        f"- Set06 confirmation: `{lock['status']}`.",
        "- Threshold remained fixed at `0.5`.",
        "- Temperatures were frozen before set06 from Exp038 development folds.",
        "- One artifact-free stalled confirmation invocation was terminated and the identical frozen command rerun; see `EXECUTION_LOG.md`.",
        "", "## Set06 Raw", "", _markdown(raw), "", "## Set06 Calibrated", "",
        _markdown(calibrated), "", "## Paired Track Bootstrap", "", _markdown(bootstrap),
        "", "## Gate", "", "```json", json.dumps(gate, indent=2, sort_keys=True), "```", "",
    ]
    if lock["status"] == "advance":
        _authenticate_manifest(root, "evaluate", EVALUATE_OUTPUTS)
        lines.extend([
            "## Set03 Test-Informed Diagnostic", "",
            "This is not independent confirmation because access was conditional on set06.", "",
            "### Raw", "", _markdown(pd.read_csv(metrics / "diagnostic_set03_results_raw.csv")), "",
            "### Calibrated", "",
            _markdown(pd.read_csv(metrics / "diagnostic_set03_results_calibrated.csv")), "",
        ])
    else:
        lines.extend(["Set03 was not materialized because set06 did not advance.", ""])
    _atomic_bytes(root / "RESULTS.md", ("\n".join(lines) + "\n").encode())
    return str(root / "RESULTS.md")


def run_verify(output_root: Path = EXPERIMENT_ROOT) -> str:
    root = Path(output_root)
    _authenticate_prepare(root)
    lock = _authenticate_confirmation(root)
    metrics = root / "metrics"
    raw = pd.read_csv(metrics / "confirmation_predictions_raw.csv")
    calibrated = pd.read_csv(metrics / "confirmation_predictions_calibrated.csv")
    for arm in RAW_ARMS:
        raw_values = raw.loc[raw["arm"].eq(arm)].sort_values("sequence_id")["probability"].to_numpy(float)
        cal_values = calibrated.loc[calibrated["raw_arm"].eq(arm)].sort_values("sequence_id")["probability"].to_numpy(float)
        global_core.verify_global_invariants(raw_values, cal_values)
    forbidden = []
    if lock["status"] == "no_advance":
        forbidden = [path.name for path in metrics.glob("*set03*")]
    elif (metrics / ".evaluate.complete.json").exists():
        _authenticate_manifest(root, "evaluate", EVALUATE_OUTPUTS)
    report = {
        "status": "PASS" if not forbidden else "FAIL",
        "confirmation_status": lock["status"],
        "confirmation_rows_per_arm": int(len(raw) // len(RAW_ARMS)),
        "global_temperature_invariants": "PASS",
        "forbidden_set03_outputs": forbidden,
    }
    _write_json(metrics / "verification_report.json", report)
    if forbidden:
        raise RuntimeError("set03 outputs exist after no-advance")
    return report["status"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("prepare", "confirm", "evaluate", "report", "verify", "all")
    )
    args = parser.parse_args()
    if args.phase in {"prepare", "all"}:
        print(f"prepare: {run_prepare()}")
    if args.phase in {"confirm", "all"}:
        print(f"confirm: {run_confirm()}")
    if args.phase in {"evaluate", "all"}:
        print(f"evaluate: {run_evaluate()}")
    if args.phase in {"report", "all"}:
        print(f"report: {run_report()}")
    if args.phase in {"verify", "all"}:
        print(f"verify: {run_verify()}")


if __name__ == "__main__":
    main()
