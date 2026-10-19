"""Fail-closed runner for Exp038 PIE nested softening calibration."""

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
for item in (SCRIPT_DIR, ACTIVE_ROOT / "experiments/037_pie_joint_biu_temporal_ego/scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import calibration_core  # noqa: E402
import nested_core  # noqa: E402
import feature_core as exp037_feature_core  # noqa: E402
import model_core as exp037_model_core  # noqa: E402


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PRE_REGISTRATION_PATH = EXPERIMENT_ROOT / "pre_registration.json"
EXP037_ROOT = ACTIVE_ROOT / "experiments/037_pie_joint_biu_temporal_ego"
EXP037_RUNNER_PATH = EXP037_ROOT / "scripts/run_experiment.py"
EXPECTED_PREREGISTRATION: dict[str, Any] = {
    "schema_version": 1,
    "experiment": "Exp038",
    "name": "PIE nested softening calibration",
    "status": "PRE_REGISTERED_NOT_RUN",
    "dataset": "PIE",
    "benchmark": "standard anticipatory",
    "observation_frames": 16,
    "tte": "30_60",
    "overlap": 0.6,
    "threshold": 0.5,
    "development_sets": ["set01", "set02", "set04", "set05"],
    "confirmation_set": "set06",
    "diagnostic_test_set": "set03",
    "diagnostic_test_status": "TEST_INFORMED_NOT_INDEPENDENT_CONFIRMATION",
    "raw_arms": ["joint_temporal_ego_xgb", "v2_bbox_xgb", "v1_bbox_rf"],
    "calibrated_arms": ["soft_temp_joint_temporal_ego_xgb", "soft_temp_v2_bbox_xgb", "soft_temp_v1_bbox_rf"],
    "seeds": [42, 43, 44],
    "temperature_bounds": [1.0, 100.0],
    "probability_clip_epsilon": 1e-12,
    "optimizer_absolute_tolerance": 1e-12,
    "objective_tie_tolerance": 1e-12,
    "combined_tail_reduction_fraction": 0.25,
    "pr_auc_noninferiority_margin": 0.002,
    "minimum_baseline_gain": 0.005,
    "maximum_per_set_regression": 0.01,
    "bootstrap_replicates": 10000,
    "bootstrap_seed": 42,
    "high_confidence_fp_min": 0.9,
    "high_confidence_fn_max": 0.1,
    "ece_bins": 10,
    "development_expected_rows": 4781,
    "development_expected_tracks": 824,
    "development_combined_tail_limit": 181,
    "feature_counts": {"joint_temporal_ego_xgb": 187, "v2_bbox_xgb": 109, "v1_bbox_rf": 125},
    "phase_contract": {
        "prepare": "authenticate Exp037 development artifacts without later-cohort access",
        "develop": "nested leave-one-set-out calibration on set01/set02/set04/set05",
        "confirm": "conditional locked set06 confirmation",
        "evaluate": "conditional authenticated set03 test-informed diagnostic",
    },
    "paper": {
        "title": "On Calibration of Modern Neural Networks",
        "year": 2017,
        "official_page": "https://proceedings.mlr.press/v70/guo17a.html",
        "official_pdf": "https://proceedings.mlr.press/v70/guo17a/guo17a.pdf",
        "local_copy": "papers/Guo_2017_Calibration.pdf",
    },
}


def load_preregistration(path: Path = PRE_REGISTRATION_PATH) -> dict[str, Any]:
    registration = json.loads(Path(path).read_text(encoding="utf-8"))
    if registration != EXPECTED_PREREGISTRATION:
        raise ValueError("Exp038 preregistration differs from the frozen semantic contract")
    return registration


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
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


def _publish(root: Path, phase: str, outputs: list[Path], **extra: Any) -> None:
    metrics = root / "metrics"
    manifest = {str(path.relative_to(metrics)): _artifact(path) for path in outputs}
    _write_json(metrics / f".{phase}.complete.json", {"phase": phase, "status": "complete", "outputs": manifest, **extra})


def _authenticate_lock(root: Path, phase: str) -> dict[str, Any]:
    root = Path(root)
    metrics = root / "metrics"
    lock_path = metrics / f"{phase}_lock.json"
    completion = json.loads((metrics / f".{phase}.complete.json").read_text(encoding="utf-8"))
    if completion.get("phase") != phase or completion.get("status") != "complete":
        raise RuntimeError(f"{phase} completion marker is invalid")
    if completion.get("lock_sha256") != _sha256(lock_path):
        raise RuntimeError(f"{phase} completion does not authenticate lock")
    for relative, expected in completion.get("outputs", {}).items():
        path = metrics / relative
        if not path.is_file() or _artifact(path) != expected:
            raise RuntimeError(f"{phase} output is not authenticated: {relative}")
    load_preregistration(root / "pre_registration.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    fixed = {
        "schema_version": 1,
        "phase": phase,
        "development_sets": EXPECTED_PREREGISTRATION["development_sets"],
        "confirmation_set": "set06",
        "diagnostic_test_set": "set03",
        "threshold": 0.5,
        "preregistration_sha256": _sha256(root / "pre_registration.json"),
    }
    for key, expected in fixed.items():
        if lock.get(key) != expected:
            raise RuntimeError(f"{phase} lock field is invalid: {key}")
    if lock.get("status") not in {"advance", "no_advance"}:
        raise RuntimeError(f"{phase} lock status is invalid")
    if completion.get("promotion_status") != lock["status"]:
        raise RuntimeError(f"{phase} promotion status is inconsistent")
    return lock


def _load_exp037_runner():
    spec = importlib.util.spec_from_file_location("exp038_bound_exp037_runner", EXP037_RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_set06_features() -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    return _load_exp037_runner()._materialize_features(("set06",))


def _load_set03_features() -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    return _load_exp037_runner()._materialize_features(("set03",))


def _authenticate_manifest(root: Path, phase: str) -> dict[str, Any]:
    metrics = Path(root) / "metrics"
    completion = json.loads((metrics / f".{phase}.complete.json").read_text(encoding="utf-8"))
    if completion.get("phase") != phase or completion.get("status") != "complete":
        raise RuntimeError(f"{phase} completion marker is invalid")
    outputs = completion.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise RuntimeError(f"{phase} output manifest is invalid")
    for relative, expected in outputs.items():
        path = metrics / relative
        if not path.is_file() or _artifact(path) != expected:
            raise RuntimeError(f"{phase} output is not authenticated: {relative}")
    return completion


def _authenticate_exp037() -> dict[str, Path]:
    _authenticate_manifest(EXP037_ROOT, "prepare")
    _authenticate_manifest(EXP037_ROOT, "develop")
    verification_path = EXP037_ROOT / "metrics/verification_report.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("status") != "PASS":
        raise RuntimeError("Exp037 verification status is not PASS")
    paths = {
        "prepared_development": EXP037_ROOT / "metrics/prepared_development.csv",
        "feature_contract": EXP037_ROOT / "metrics/feature_contract.json",
        "outer_oof": EXP037_ROOT / "metrics/development_oof_predictions.csv",
        "raw_results": EXP037_ROOT / "metrics/development_results.csv",
        "verification": verification_path,
        "materializer": EXP037_RUNNER_PATH,
        "model_core": EXP037_ROOT / "scripts/model_core.py",
    }
    for path in paths.values():
        path.resolve(strict=True)
    return paths


def run_prepare(output_root: Path = EXPERIMENT_ROOT) -> str:
    root = Path(output_root)
    marker = root / "metrics/.prepare.complete.json"
    if marker.exists():
        _authenticate_manifest(root, "prepare")
        return "complete"
    load_preregistration(root / "pre_registration.json")
    sources = _authenticate_exp037()
    development = pd.read_csv(sources["prepared_development"])
    if len(development) != 4781 or development["track_id"].nunique() != 824:
        raise RuntimeError("Exp037 development cohort changed")
    contract = json.loads(sources["feature_contract"].read_text(encoding="utf-8"))
    expected_counts = EXPECTED_PREREGISTRATION["feature_counts"]
    for arm, count in expected_counts.items():
        if len(contract["arms"][arm]) != count:
            raise RuntimeError(f"Exp037 feature count changed for {arm}")
    raw = pd.read_csv(sources["outer_oof"])
    raw = raw.loc[raw["arm"].isin(EXPECTED_PREREGISTRATION["raw_arms"])].copy()
    nested_core.validate_outer_predictions(
        raw,
        development.loc[:, ["sequence_id", "set_id", "track_id", "video_id", "label"]],
    )
    source_records = {name: {"path": str(path.resolve()), **_artifact(path)} for name, path in sources.items()}
    metrics = root / "metrics"
    prepared_path = metrics / "prepared_development.csv"
    raw_path = metrics / "raw_outer_oof_predictions.csv"
    contract_path = metrics / "feature_contract.json"
    provenance_path = metrics / "source_provenance.json"
    _write_csv(prepared_path, development)
    _write_csv(raw_path, raw.sort_values(["arm", "sequence_id"], kind="stable"))
    _write_json(contract_path, contract)
    _write_json(
        provenance_path,
        {
            "authorized_sets": EXPECTED_PREREGISTRATION["development_sets"],
            "set06_materialized": False,
            "set03_materialized": False,
            "sources": source_records,
        },
    )
    _publish(root, "prepare", [prepared_path, raw_path, contract_path, provenance_path])
    return "complete"


def _metric_row(arm: str, rows: pd.DataFrame, scope: str, set_id: str = "pooled") -> dict[str, Any]:
    return {
        "scope": scope,
        "set_id": set_id,
        "arm": arm,
        **calibration_core.compute_metrics(rows["label"].to_numpy(int), rows["probability"].to_numpy(float)),
    }


def _nested_development(
    development: pd.DataFrame,
    raw_outer: pd.DataFrame,
    arms: dict[str, tuple[str, ...]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inner_parts = []
    temperature_rows = []
    outer_parts = []
    for outer in nested_core.nested_fold_plan(tuple(EXPECTED_PREREGISTRATION["development_sets"])):
        for arm in EXPECTED_PREREGISTRATION["raw_arms"]:
            arm_inner = []
            for inner in outer.inner_folds:
                train = development.loc[development["set_id"].astype(str).isin(inner.training_sets)]
                calibration = development.loc[development["set_id"].astype(str).eq(inner.calibration_set)]
                probability = exp037_model_core.fit_predict_seed_mean(
                    train, calibration, arms[arm], arm, tuple(EXPECTED_PREREGISTRATION["seeds"])
                )
                rows = calibration.loc[:, ["sequence_id", "set_id", "track_id", "video_id", "label"]].copy()
                rows["outer_set"] = outer.held_out_set
                rows["arm"] = arm
                rows["probability"] = probability
                rows["inner_training_sets_json"] = json.dumps(list(inner.training_sets), separators=(",", ":"))
                arm_inner.append(rows)
            inner_frame = pd.concat(arm_inner, ignore_index=True).sort_values("sequence_id", kind="stable")
            expected = development.loc[development["set_id"].astype(str).isin(outer.remaining_sets)]
            if set(inner_frame["sequence_id"].astype(str)) != set(expected["sequence_id"].astype(str)):
                raise RuntimeError("nested inner OOF coverage is invalid")
            fit = calibration_core.fit_softening_temperature(
                inner_frame["label"].to_numpy(int), inner_frame["probability"].to_numpy(float)
            )
            inner_parts.append(inner_frame)
            outer_raw = raw_outer.loc[
                raw_outer["arm"].eq(arm) & raw_outer["set_id"].astype(str).eq(outer.held_out_set)
            ].copy()
            calibrated = calibration_core.apply_temperature(outer_raw["probability"].to_numpy(float), fit.temperature)
            calibration_core.verify_softening_invariants(
                outer_raw["probability"].to_numpy(float), calibrated, outer_raw["label"].to_numpy(int)
            )
            outer_raw = outer_raw.rename(columns={"probability": "raw_probability"})
            outer_raw["raw_arm"] = arm
            outer_raw["arm"] = f"soft_temp_{arm}"
            outer_raw["probability"] = calibrated
            outer_raw["temperature"] = fit.temperature
            outer_parts.append(outer_raw)
            temperature_rows.append(
                {
                    "outer_set": outer.held_out_set,
                    "raw_arm": arm,
                    "calibrated_arm": f"soft_temp_{arm}",
                    **fit.__dict__,
                    "inner_rows": int(len(inner_frame)),
                    "outer_rows": int(len(outer_raw)),
                    "remaining_sets_json": json.dumps(list(outer.remaining_sets), separators=(",", ":")),
                }
            )
    return pd.concat(inner_parts, ignore_index=True), pd.DataFrame(temperature_rows), pd.concat(outer_parts, ignore_index=True)


def run_develop(output_root: Path = EXPERIMENT_ROOT) -> str:
    root = Path(output_root)
    marker = root / "metrics/.develop.complete.json"
    if marker.exists():
        return str(_authenticate_lock(root, "development")["status"])
    _authenticate_manifest(root, "prepare")
    metrics = root / "metrics"
    development = pd.read_csv(metrics / "prepared_development.csv")
    raw_outer = pd.read_csv(metrics / "raw_outer_oof_predictions.csv")
    contract = json.loads((metrics / "feature_contract.json").read_text(encoding="utf-8"))
    arms = {arm: tuple(contract["arms"][arm]) for arm in EXPECTED_PREREGISTRATION["raw_arms"]}
    inner, temperatures, calibrated = _nested_development(development, raw_outer, arms)

    raw_results = pd.DataFrame(
        [_metric_row(arm, rows, "raw_development_oof") for arm, rows in raw_outer.groupby("arm", sort=True)]
    )
    calibrated_results = pd.DataFrame(
        [_metric_row(arm, rows, "calibrated_development_oof") for arm, rows in calibrated.groupby("arm", sort=True)]
    )
    per_set_rows = []
    for set_id in EXPECTED_PREREGISTRATION["development_sets"]:
        subset = calibrated.loc[calibrated["set_id"].astype(str).eq(set_id)]
        values = {arm: calibration_core.compute_metrics(rows["label"].to_numpy(int), rows["probability"].to_numpy(float)) for arm, rows in subset.groupby("arm")}
        candidate = values["soft_temp_joint_temporal_ego_xgb"]
        per_set_rows.append(
            {
                "set_id": set_id,
                "pr_auc_delta_vs_v2": candidate["pr_auc"] - values["soft_temp_v2_bbox_xgb"]["pr_auc"],
                "f1_delta_vs_best": candidate["f1_fixed_0p5"] - max(
                    values["soft_temp_v2_bbox_xgb"]["f1_fixed_0p5"], values["soft_temp_v1_bbox_rf"]["f1_fixed_0p5"]
                ),
            }
        )
    per_set = pd.DataFrame(per_set_rows)
    wide = calibrated.pivot(index=["sequence_id", "set_id", "track_id", "video_id", "label"], columns="arm", values="probability").reset_index()
    indexed = calibrated_results.set_index("arm")
    best_f1 = max(
        ("soft_temp_v2_bbox_xgb", "soft_temp_v1_bbox_rf"),
        key=lambda arm: float(indexed.loc[arm, "f1_fixed_0p5"]),
    )
    pr_bootstrap = nested_core.cluster_bootstrap_deltas(
        wide, "soft_temp_v2_bbox_xgb", "soft_temp_joint_temporal_ego_xgb",
        replicates=10000, seed=42,
    ).loc[lambda value: value["metric"].eq("pr_auc")].assign(reference_arm="soft_temp_v2_bbox_xgb")
    f1_bootstrap = nested_core.cluster_bootstrap_deltas(
        wide, best_f1, "soft_temp_joint_temporal_ego_xgb",
        replicates=10000, seed=42,
    ).loc[lambda value: value["metric"].eq("f1_fixed_0p5")].assign(reference_arm=best_f1)
    bootstrap = pd.concat([pr_bootstrap, f1_bootstrap], ignore_index=True)
    decision = nested_core.evaluate_promotion_gate(
        raw_results, calibrated_results, per_set, bootstrap, fixed_tail_limit=181
    )
    output_frames = {
        "development_inner_oof_predictions.csv": inner,
        "development_temperatures.csv": temperatures,
        "development_outer_predictions.csv": calibrated,
        "development_raw_results.csv": raw_results,
        "development_results.csv": calibrated_results,
        "development_per_set.csv": per_set,
        "development_bootstrap.csv": bootstrap,
    }
    outputs = []
    for name, frame in output_frames.items():
        path = metrics / name
        _write_csv(path, frame)
        outputs.append(path)
    checks_path = metrics / "development_gate_checks.json"
    _write_json(checks_path, {"status": decision.status, "checks": decision.checks, "observed": decision.observed})
    outputs.append(checks_path)
    lock = {
        "schema_version": 1,
        "phase": "development",
        "status": decision.status,
        "selected_procedure": "soft_temp_joint_temporal_ego_xgb" if decision.status == "advance" else None,
        "development_sets": EXPECTED_PREREGISTRATION["development_sets"],
        "confirmation_set": "set06",
        "diagnostic_test_set": "set03",
        "threshold": 0.5,
        "preregistration_sha256": _sha256(root / "pre_registration.json"),
    }
    lock_path = metrics / "development_lock.json"
    _write_json(lock_path, lock)
    outputs.append(lock_path)
    _publish(root, "development", outputs, lock_sha256=_sha256(lock_path), promotion_status=decision.status)
    return decision.status


def run_confirm(output_root: Path = EXPERIMENT_ROOT) -> str:
    root = Path(output_root)
    lock = _authenticate_lock(root, "development")
    if lock["status"] == "no_advance":
        confirmation = {
            **lock,
            "phase": "confirmation",
            "status": "no_advance",
            "selected_procedure": None,
        }
        path = root / "metrics/confirmation_lock.json"
        _write_json(path, confirmation)
        _publish(root, "confirmation", [path], lock_sha256=_sha256(path), promotion_status="no_advance")
        return "no_advance"
    _load_set06_features()
    raise NotImplementedError("advancing set06 confirmation is not implemented")


def run_evaluate(output_root: Path = EXPERIMENT_ROOT) -> str:
    root = Path(output_root)
    lock = _authenticate_lock(root, "confirmation")
    if lock["status"] == "no_advance":
        return "no_advance"
    _load_set03_features()
    raise NotImplementedError("advancing set03 diagnostic is not implemented")


def _markdown(frame: pd.DataFrame) -> str:
    columns = [str(value) for value in frame.columns]
    rows = [f"| {' | '.join(columns)} |", f"| {' | '.join('---' for _ in columns)} |"]
    rows.extend(
        f"| {' | '.join(str(value).replace('|', chr(92) + '|') for value in row)} |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def run_report(output_root: Path = EXPERIMENT_ROOT) -> str:
    root = Path(output_root)
    _authenticate_manifest(root, "prepare")
    development_lock = _authenticate_lock(root, "development")
    confirmation_path = root / "metrics/.confirmation.complete.json"
    confirmation_status = "not_run"
    if confirmation_path.exists():
        confirmation_status = str(_authenticate_lock(root, "confirmation")["status"])
    metrics = root / "metrics"
    raw = pd.read_csv(metrics / "development_raw_results.csv")
    calibrated = pd.read_csv(metrics / "development_results.csv")
    temperatures = pd.read_csv(metrics / "development_temperatures.csv")
    checks = json.loads((metrics / "development_gate_checks.json").read_text())
    lines = [
        "# Exp038 Results",
        "",
        "## Scientific Status",
        "",
        f"- Development promotion: `{development_lock['status']}`.",
        f"- Set06 confirmation: `{confirmation_status}`.",
        "- Threshold remained fixed at `0.5`.",
        "- Softening temperature cannot change fixed-0.5 decisions within a cohort.",
        "",
        "## Raw Development OOF",
        "",
        _markdown(raw),
        "",
        "## Calibrated Development OOF",
        "",
        _markdown(calibrated),
        "",
        "## Outer-Fold Temperatures",
        "",
        _markdown(temperatures),
        "",
        "## Gate",
        "",
        "```json",
        json.dumps(checks, indent=2, sort_keys=True),
        "```",
        "",
    ]
    if development_lock["status"] == "no_advance":
        lines.append("Set06 and set03 were not materialized because development did not advance.")
    _atomic_bytes(root / "RESULTS.md", ("\n".join(lines) + "\n").encode())
    return "complete"


def run_verify(output_root: Path = EXPERIMENT_ROOT) -> str:
    root = Path(output_root)
    _authenticate_manifest(root, "prepare")
    development = _authenticate_lock(root, "development")
    metrics = root / "metrics"
    calibrated = pd.read_csv(metrics / "development_outer_predictions.csv")
    temperatures = pd.read_csv(metrics / "development_temperatures.csv")
    for row in temperatures.itertuples(index=False):
        subset = calibrated.loc[
            calibrated["raw_arm"].eq(row.raw_arm) & calibrated["set_id"].astype(str).eq(row.outer_set)
        ]
        reconstructed = calibration_core.apply_temperature(subset["raw_probability"].to_numpy(float), row.temperature)
        if not np.allclose(reconstructed, subset["probability"].to_numpy(float), rtol=0.0, atol=1e-14):
            raise RuntimeError("saved calibrated probabilities do not reconstruct")
        calibration_core.verify_softening_invariants(
            subset["raw_probability"].to_numpy(float), reconstructed, subset["label"].to_numpy(int)
        )
    forbidden = [
        path.name
        for path in metrics.iterdir()
        if development["status"] == "no_advance" and ("confirmation_features" in path.name or "set03" in path.name)
    ]
    if forbidden:
        raise RuntimeError(f"forbidden later-cohort outputs exist: {forbidden}")
    report = {
        "status": "PASS",
        "development_status": development["status"],
        "calibrated_rows": int(len(calibrated)),
        "temperatures": int(len(temperatures)),
        "set06_or_set03_forbidden_outputs": forbidden,
    }
    _write_json(metrics / "verification_report.json", report)
    return "PASS"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "develop", "confirm", "evaluate", "report", "verify"))
    args = parser.parse_args()
    functions = {
        "prepare": run_prepare,
        "develop": run_develop,
        "confirm": run_confirm,
        "evaluate": run_evaluate,
        "report": run_report,
        "verify": run_verify,
    }
    print(functions[args.phase]())


if __name__ == "__main__":
    main()
