#!/usr/bin/env python3
"""Run the retrospective matched BIU-version ablation for Exp072."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
REPO_ROOT = ROOT.parents[1]
EXP072 = REPO_ROOT / "experiments/072_pie_joint_biu_temporal_ego_confirmation"
DEFINITIONS = REPO_ROOT / "experiments/012_pie_biu_v2_30_60_benchmark/metrics/pie_biu_v2_feature_set_definitions.csv"
PARENT_FEATURE_CONTRACT = REPO_ROOT / "experiments/037_pie_joint_biu_temporal_ego/metrics/feature_contract.json"
ARMS = ("base", "v1", "v2", "combined")
SEEDS = (42, 43, 44)
THRESHOLD = 0.5
VERSION_BOOTSTRAP_COMPARISONS = (("v1", "v2"), ("v1", "combined"), ("v2", "combined"))
BASE_BOOTSTRAP_COMPARISONS = (("base", "v1"), ("base", "v2"), ("base", "combined"))
SOURCE_HASHES = {
    "prepared_train.csv": "e0d1b9a35c74de95c4f6c56f13c4db8123b1272f8609b56208d571342f4689ea",
    "prepared_validation.csv": "7b72d2476d38aa9b7efe40680e00bbe5c7aef3aeaedf731708225dacce275356",
    "prepared_test.csv": "72e4e0ded1350d6eb6a095d72d43cbf6b1aae716305f06b49f531f27da8bc7a6",
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


feature_contract = _load_module("exp096_feature_contract_runtime", SCRIPT_DIR / "feature_contract.py")
model_core = _load_module(
    "exp096_exp072_model_core",
    EXP072 / "scripts/model_core.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
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


def atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_bytes(path, frame.to_csv(index=False).encode())


def atomic_text(path: Path, value: str) -> None:
    _atomic_bytes(path, value.encode())


def source_path(split: str) -> Path:
    name = {"train": "prepared_train.csv", "validation": "prepared_validation.csv", "test": "prepared_test.csv"}[split]
    path = EXP072 / "metrics" / name
    if sha256_file(path) != SOURCE_HASHES[name]:
        raise RuntimeError(f"authenticated Exp072 source changed: {name}")
    return path


def arm_columns() -> dict[str, tuple[str, ...]]:
    return feature_contract.build_matched_arm_columns(DEFINITIONS, PARENT_FEATURE_CONTRACT)


def required_runs() -> list[tuple[str, int]]:
    return [(arm, seed) for arm in ARMS for seed in SEEDS]


def _validate_source(frame: pd.DataFrame, split: str, columns: dict[str, tuple[str, ...]]) -> None:
    expected_rows = {"train": 2366, "validation": 3619, "test": 3760}[split]
    expected_sets = {
        "train": {"set01", "set02", "set06"},
        "validation": {"set04", "set05"},
        "test": {"set03"},
    }[split]
    required = {
        "sequence_id", "set_id", "video_id", "track_id", "label", "observation_length",
        "start_frame", "end_frame", "time_to_event_frames",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{split} source missing identity columns: {sorted(missing)}")
    if len(frame) != expected_rows or frame["sequence_id"].duplicated().any():
        raise RuntimeError(f"{split} source row identity changed")
    if set(frame["set_id"].astype(str)) != expected_sets:
        raise RuntimeError(f"{split} source set membership changed")
    if not set(frame["label"].astype(int).unique()).issubset({0, 1}):
        raise RuntimeError(f"{split} labels are not binary")
    exact = (
        frame["observation_length"].astype(int).eq(16)
        & (frame["end_frame"].astype(int) - frame["start_frame"].astype(int)).eq(15)
        & frame["time_to_event_frames"].astype(int).between(30, 60, inclusive="both")
    )
    if not exact.all():
        raise RuntimeError(f"{split} temporal protocol changed")
    all_features = tuple(dict.fromkeys(name for names in columns.values() for name in names))
    absent = set(all_features).difference(frame.columns)
    if absent:
        raise RuntimeError(f"{split} source missing frozen features: {sorted(absent)[:5]}")
    if not np.isfinite(frame.loc[:, all_features].to_numpy(float)).all():
        raise RuntimeError(f"{split} source contains non-finite features")


def _load(split: str) -> pd.DataFrame:
    frame = pd.read_csv(source_path(split))
    _validate_source(frame, split, arm_columns())
    return frame.sort_values("sequence_id", kind="stable").reset_index(drop=True)


def _completion_path(root: Path, phase: str) -> Path:
    return Path(root) / "metrics" / f".{phase}.complete.json"


def _publish_completion(root: Path, phase: str, outputs: list[Path], **extra: Any) -> None:
    root = Path(root).resolve()
    records = {}
    for path in outputs:
        resolved = Path(path).resolve(strict=True)
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as error:
            raise RuntimeError("completion outputs must remain inside the experiment root") from error
        records[relative] = artifact_record(resolved)
    payload = {
        "status": "complete",
        "phase": phase,
        "outputs": records,
        **extra,
    }
    atomic_json(_completion_path(root, phase), payload)


def _authenticate_completion(root: Path, phase: str) -> dict[str, Any]:
    marker = _completion_path(root, phase)
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        outputs = payload["outputs"]
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"missing authenticated {phase} completion") from error
    if payload.get("status") != "complete" or not isinstance(outputs, dict) or not outputs:
        raise RuntimeError(f"missing authenticated {phase} completion")
    for name, expected in outputs.items():
        path = Path(root) / name
        if not path.is_file() or artifact_record(path) != expected:
            raise RuntimeError(f"missing authenticated {phase} completion")
    return payload


def validation_gate_payload(per_seed: pd.DataFrame) -> dict[str, Any]:
    required = {"arm", "seed", "threshold", "f1", "roc_auc", "pr_auc"}
    identities = set(zip(per_seed.get("arm", pd.Series(dtype=str)), per_seed.get("seed", pd.Series(dtype=int))))
    if required.difference(per_seed.columns) or identities != set(required_runs()) or len(per_seed) != 12:
        raise RuntimeError("validation requires a complete 12-run validation grid")
    if not per_seed["threshold"].astype(float).eq(THRESHOLD).all():
        raise RuntimeError("validation requires a complete 12-run validation grid")
    if not np.isfinite(per_seed.loc[:, ["f1", "roc_auc", "pr_auc"]].to_numpy(float)).all():
        raise RuntimeError("validation requires a complete 12-run validation grid")
    return {
        "status": "advance",
        "reason": "all four prespecified arms and three seeds completed on validation",
        "selection_rule": "no arm selected; all prespecified controls advance together",
        "run_count": 12,
        "threshold": THRESHOLD,
        "test_previously_exposed_elsewhere": True,
    }


def authenticate_test_gate(root: Path) -> dict[str, Any]:
    try:
        completion = _authenticate_completion(root, "validate")
        gate = json.loads((Path(root) / "metrics/validation_gate.json").read_text(encoding="utf-8"))
    except (RuntimeError, FileNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError("set03 requires authenticated validation completion") from error
    if completion.get("promotion_status") != "advance" or gate.get("status") != "advance" or gate.get("run_count") != 12:
        raise RuntimeError("set03 requires authenticated validation completion")
    return gate


def _summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    metrics = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier", "log_loss", "ece_10bin")
    return per_seed.groupby(["scope", "arm", "feature_count"], sort=False).agg(
        **{f"{metric}_{stat}": (metric, stat) for metric in metrics for stat in ("mean", "std")}
    ).reset_index()


def _prediction_frame(source: pd.DataFrame, arm: str, seed: int, probability: np.ndarray) -> pd.DataFrame:
    output = source.loc[:, ["sequence_id", "set_id", "video_id", "track_id", "label"]].copy()
    output.insert(0, "seed", int(seed))
    output.insert(0, "arm", arm)
    output["threshold"] = THRESHOLD
    output["probability"] = probability
    output["prediction"] = (probability >= THRESHOLD).astype(int)
    return output


def _fit_grid(train: pd.DataFrame, target: pd.DataFrame, scope: str, root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    columns_by_arm = arm_columns()
    records: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    models: dict[str, Any] = {}
    labels = train["label"].to_numpy(int)
    for arm, seed in required_runs():
        columns = columns_by_arm[arm]
        model = model_core.build_candidate(seed, labels)
        model.fit(train.loc[:, columns].astype(float), labels)
        probability = model.predict_proba(target.loc[:, columns].astype(float))[:, 1]
        metrics = model_core.compute_metrics(target["label"].to_numpy(int), probability)
        records.append({"scope": scope, "arm": arm, "seed": seed, "feature_count": len(columns), **metrics})
        predictions.append(_prediction_frame(target, arm, seed, probability))
        model_path = Path(root) / "models" / f"{arm}_seed{seed}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        models[f"{arm}_seed{seed}"] = artifact_record(model_path)
    return pd.DataFrame(records), pd.concat(predictions, ignore_index=True), models


def run_prepare(root: Path = ROOT) -> str:
    root = Path(root)
    if _completion_path(root, "prepare").is_file():
        _authenticate_completion(root, "prepare")
        return "complete"
    columns = arm_columns()
    train = _load("train")
    validation = _load("validation")
    if set(train["sequence_id"]).intersection(validation["sequence_id"]):
        raise RuntimeError("train and validation cohorts overlap")
    contract_path = root / "metrics/feature_arms.json"
    audit_path = root / "metrics/source_audit.json"
    atomic_json(contract_path, {name: list(values) for name, values in columns.items()})
    atomic_json(
        audit_path,
        {
            "experiment": 96,
            "classification": "retrospective matched control",
            "protocol": "PIE 16 observed frames, TTE30--60, overlap0.6, fixed threshold0.5",
            "reused": "authenticated Exp072 prepared feature matrices",
            "recomputed": "four matched XGBoost arms across three seeds",
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_accessed_by_exp096_prepare": False,
            "source_hashes": {name: SOURCE_HASHES[name] for name in ("prepared_train.csv", "prepared_validation.csv")},
        },
    )
    _publish_completion(root, "prepare", [contract_path, audit_path], test_accessed=False)
    return "complete"


def run_validate(root: Path = ROOT) -> str:
    root = Path(root)
    if _completion_path(root, "validate").is_file():
        _authenticate_completion(root, "validate")
        return "advance"
    _authenticate_completion(root, "prepare")
    train = _load("train")
    validation = _load("validation")
    per_seed, predictions, models = _fit_grid(train, validation, "validation", root)
    gate = validation_gate_payload(per_seed)
    per_seed_path = root / "metrics/validation_per_seed_results.csv"
    summary_path = root / "metrics/validation_summary_results.csv"
    predictions_path = root / "predictions/validation_predictions.csv"
    manifest_path = root / "metrics/model_manifest_pre_gate.json"
    gate_path = root / "metrics/validation_gate.json"
    atomic_csv(per_seed_path, per_seed)
    atomic_csv(summary_path, _summarize(per_seed))
    atomic_csv(predictions_path, predictions)
    atomic_json(manifest_path, {"models": models, "fit_rows": len(train), "fit_rule": "train_only"})
    atomic_json(gate_path, gate)
    model_paths = [root / "models" / f"{arm}_seed{seed}.joblib" for arm, seed in required_runs()]
    _publish_completion(
        root,
        "validate",
        [per_seed_path, summary_path, predictions_path, manifest_path, gate_path, *model_paths],
        promotion_status="advance",
        test_accessed=False,
    )
    return "advance"


def _authenticate_models(root: Path) -> dict[str, Any]:
    manifest_path = Path(root) / "metrics/model_manifest_pre_gate.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for arm, seed in required_runs():
        name = f"{arm}_seed{seed}"
        path = Path(root) / "models" / f"{name}.joblib"
        if artifact_record(path) != manifest["models"][name]:
            raise RuntimeError("frozen validation model changed before test evaluation")
    return manifest


def _pairwise_deltas(per_seed: pd.DataFrame) -> pd.DataFrame:
    comparisons = (("base", "v1"), ("base", "v2"), ("base", "combined"), ("v1", "v2"), ("v1", "combined"), ("v2", "combined"))
    indexed = per_seed.set_index(["arm", "seed"])
    metrics = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
    rows = []
    for left, right in comparisons:
        for seed in SEEDS:
            rows.append({
                "comparison": f"{right}_minus_{left}",
                "seed": seed,
                **{f"delta_{metric}": float(indexed.loc[(right, seed), metric] - indexed.loc[(left, seed), metric]) for metric in metrics},
            })
    return pd.DataFrame(rows)


def _ensemble_bootstrap(
    predictions: pd.DataFrame,
    replicates: int = 10000,
    comparisons: tuple[tuple[str, str], ...] = VERSION_BOOTSTRAP_COMPARISONS,
) -> pd.DataFrame:
    ensemble = predictions.groupby(
        ["arm", "sequence_id", "set_id", "video_id", "track_id", "label"], sort=True
    )["probability"].mean().reset_index()
    by_arm = {arm: frame.set_index("sequence_id") for arm, frame in ensemble.groupby("arm", sort=False)}
    rows = []
    for left, right in comparisons:
        a = by_arm[left].sort_index()
        b = by_arm[right].sort_index()
        if not a.index.equals(b.index) or not a["label"].equals(b["label"]):
            raise RuntimeError("ensemble comparison cohorts differ")
        aligned = b.reset_index().rename(columns={"probability": "probability"})
        aligned["anchor_probability"] = a["probability"].to_numpy(float)
        aligned["seed"] = 0
        result = model_core.cluster_bootstrap_deltas(
            aligned,
            replicates=replicates,
            seed=42,
        )
        result.insert(0, "comparison", f"{right}_minus_{left}")
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def run_base_bootstrap(root: Path = ROOT) -> str:
    """Complete BIU-vs-base uncertainty analysis without replacing benchmark outputs."""
    root = Path(root)
    if _completion_path(root, "base_bootstrap").is_file():
        _authenticate_completion(root, "base_bootstrap")
        return "complete"
    _authenticate_completion(root, "evaluate")
    predictions = pd.read_csv(root / "predictions/test_predictions.csv")
    output = root / "metrics/base_comparison_track_bootstrap.csv"
    atomic_csv(
        output,
        _ensemble_bootstrap(predictions, comparisons=BASE_BOOTSTRAP_COMPARISONS),
    )
    _publish_completion(
        root,
        "base_bootstrap",
        [output],
        source_predictions=artifact_record(root / "predictions/test_predictions.csv"),
        replicates=10000,
        bootstrap_seed=42,
    )
    return "complete"


def run_evaluate(root: Path = ROOT) -> str:
    root = Path(root)
    if _completion_path(root, "evaluate").is_file():
        _authenticate_completion(root, "evaluate")
        return "complete"
    authenticate_test_gate(root)
    manifest_before = _authenticate_models(root)
    test = _load("test")
    columns_by_arm = arm_columns()
    records = []
    prediction_parts = []
    for arm, seed in required_runs():
        model = joblib.load(root / "models" / f"{arm}_seed{seed}.joblib")
        probability = model.predict_proba(test.loc[:, columns_by_arm[arm]].astype(float))[:, 1]
        records.append({
            "scope": "test", "arm": arm, "seed": seed, "feature_count": len(columns_by_arm[arm]),
            **model_core.compute_metrics(test["label"].to_numpy(int), probability),
        })
        prediction_parts.append(_prediction_frame(test, arm, seed, probability))
    if manifest_before != _authenticate_models(root):
        raise RuntimeError("frozen validation models changed during test evaluation")
    per_seed = pd.DataFrame(records)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    per_seed_path = root / "metrics/test_per_seed_results.csv"
    summary_path = root / "metrics/test_summary_results.csv"
    deltas_path = root / "metrics/test_paired_deltas.csv"
    predictions_path = root / "predictions/test_predictions.csv"
    bootstrap_path = root / "metrics/ensemble_track_bootstrap.csv"
    atomic_csv(per_seed_path, per_seed)
    atomic_csv(summary_path, _summarize(per_seed))
    atomic_csv(deltas_path, _pairwise_deltas(per_seed))
    atomic_csv(predictions_path, predictions)
    atomic_csv(bootstrap_path, _ensemble_bootstrap(predictions))
    _publish_completion(
        root,
        "evaluate",
        [per_seed_path, summary_path, deltas_path, predictions_path, bootstrap_path],
        test_rows=len(test),
        test_source_sha256=SOURCE_HASHES["prepared_test.csv"],
        model_refit=False,
    )
    return "complete"


def _markdown(frame: pd.DataFrame) -> str:
    def format_cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            rendered = f"{float(value):.6f}"
        else:
            rendered = str(value)
        return rendered.replace("|", "\\|").replace("\n", " ")

    headers = [format_cell(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(format_cell(value) for value in row) + " |")
    return "\n".join(lines)


def run_report(root: Path = ROOT) -> str:
    root = Path(root)
    _authenticate_completion(root, "evaluate")
    _authenticate_completion(root, "base_bootstrap")
    summary = pd.read_csv(root / "metrics/test_summary_results.csv")
    deltas = pd.read_csv(root / "metrics/test_paired_deltas.csv")
    delta_summary = deltas.groupby("comparison", sort=False).agg(
        f1_delta_mean=("delta_f1", "mean"),
        roc_auc_delta_mean=("delta_roc_auc", "mean"),
        pr_auc_delta_mean=("delta_pr_auc", "mean"),
        f1_favorable_seeds=("delta_f1", lambda values: int((values > 0).sum())),
    ).reset_index()
    bootstrap = pd.read_csv(root / "metrics/ensemble_track_bootstrap.csv")
    base_bootstrap = pd.read_csv(root / "metrics/base_comparison_track_bootstrap.csv")
    report = "\n".join(
        [
            "# Exp096 Final Report",
            "",
            "Retrospective matched control for the Exp072 XGBoost architecture on PIE set03.",
            "All arms use 16 observed frames, TTE30--60, overlap0.6, fixed threshold 0.5,",
            "the same train/validation/test rows, the same seeds, and the same estimator recipe.",
            "",
            "## Fixed-0.5 test summary",
            "",
            _markdown(summary),
            "",
            "## Mean paired deltas",
            "",
            _markdown(delta_summary),
            "",
            "## Seed-ensemble track bootstrap",
            "",
            _markdown(bootstrap),
            "",
            "## BIU arms versus the shared base: seed-ensemble track bootstrap",
            "",
            _markdown(base_bootstrap),
            "",
            "## Interpretation boundary",
            "",
            "This experiment was conceived after prior set03 exposure and is therefore a retrospective",
            "matched control, not pristine confirmation. BIU-Combined reproduces Exp072's exact 187-feature",
            "candidate. The v1 and v2 arms isolate their respective pose representations while retaining",
            "the shared BBox, BBox-temporal, ego-speed, and pose-quality inputs. V2-specific temporal",
            "contrasts remain part of the v2 design. The result supports an architecture-specific version",
            "verdict and does not establish universal dominance across models or datasets.",
        ]
    ) + "\n"
    report_path = root / "FINAL_REPORT.md"
    atomic_text(report_path, report)
    _publish_completion(root, "report", [report_path])
    return "complete"


def run_verify(root: Path = ROOT) -> str:
    root = Path(root)
    for phase in ("prepare", "validate", "evaluate", "base_bootstrap", "report"):
        _authenticate_completion(root, phase)
    authenticate_test_gate(root)
    per_seed = pd.read_csv(root / "metrics/test_per_seed_results.csv")
    if set(zip(per_seed["arm"], per_seed["seed"])) != set(required_runs()) or len(per_seed) != 12:
        raise RuntimeError("final result grid is incomplete")
    completion = {
        "experiment": 96,
        "status": "complete",
        "classification": "retrospective matched control",
        "new_training_runs": 12,
        "reused_feature_matrices": 3,
        "threshold": THRESHOLD,
        "test_rows": 3760,
        "report": artifact_record(root / "FINAL_REPORT.md"),
        "base_comparison_bootstrap": artifact_record(
            root / "metrics/base_comparison_track_bootstrap.csv"
        ),
    }
    atomic_json(root / "EXPERIMENT_COMPLETE.json", completion)
    return "complete"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "validate", "evaluate", "base-bootstrap", "report", "verify", "all"),
    )
    args = parser.parse_args()
    commands = {
        "prepare": run_prepare,
        "validate": run_validate,
        "evaluate": run_evaluate,
        "base-bootstrap": run_base_bootstrap,
        "report": run_report,
        "verify": run_verify,
    }
    if args.command == "all":
        for name in ("prepare", "validate", "evaluate", "base-bootstrap", "report", "verify"):
            print(f"{name}: {commands[name]()}", flush=True)
    else:
        print(f"{args.command}: {commands[args.command]()}", flush=True)


if __name__ == "__main__":
    main()
