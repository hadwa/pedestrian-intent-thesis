"""Fail-closed development runner for Exp040 temporal BIU sequences."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
ACTIVE_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PRE_REGISTRATION_PATH = EXPERIMENT_ROOT / "pre_registration.json"
EXP037_ROOT = ACTIVE_ROOT / "experiments/037_pie_joint_biu_temporal_ego"
EXP039_ROOT = ACTIVE_ROOT / "experiments/039_pie_locked_global_temperature"
ALLOWED_SETS = ("set01", "set02", "set04", "set05", "set06")
FORBIDDEN_SET = "set03"
PREREGISTRATION_SHA256 = "9870ee51101b240117b9810d230fd2656c00f0e2d21ba9fe1bbaa78f52d10973"
IDENTITY = ["sequence_id", "set_id", "track_id", "video_id", "label"]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import sequence_core  # noqa: E402


_EXP037_MODULE = None


def _load_exp037():
    global _EXP037_MODULE
    if _EXP037_MODULE is not None:
        return _EXP037_MODULE
    path = EXP037_ROOT / "scripts/run_experiment.py"
    spec = importlib.util.spec_from_file_location("exp040_exp037_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Exp037 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _EXP037_MODULE = module
    return module


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


def load_preregistration(path: Path = PRE_REGISTRATION_PATH) -> dict[str, Any]:
    path = Path(path)
    digest = _sha256(path)
    if digest != PREREGISTRATION_SHA256:
        raise ValueError("Exp040 preregistration changed after freezing")
    registration = json.loads(path.read_text(encoding="utf-8"))
    if registration.get("authorized_sets") != list(ALLOWED_SETS):
        raise ValueError("Exp040 authorized set universe changed")
    if registration.get("forbidden_set") != FORBIDDEN_SET or registration.get("threshold") != 0.5:
        raise ValueError("Exp040 forbidden set or threshold changed")
    return registration


def authorize_sets(sets: tuple[str, ...]) -> tuple[str, ...]:
    values = tuple(map(str, sets))
    if FORBIDDEN_SET in values:
        raise ValueError("forbidden set03 requested")
    if tuple(sorted(values)) != ALLOWED_SETS or len(set(values)) != len(values):
        raise ValueError("Exp040 requires exactly the five authorized sets")
    return values


def _source_parent_record(root: Path, phase: str) -> dict[str, Any]:
    marker = Path(root) / f"metrics/.{phase}.complete.json"
    completion = _authenticate_manifest(root, phase)
    return {
        "root": str(Path(root).resolve(strict=True)),
        "phase": phase,
        "marker": str(marker.resolve(strict=True)),
        **_record(marker),
        "outputs": sorted(completion["outputs"]),
    }


def _build_sequences(sets: tuple[str, ...]) -> tuple[np.ndarray, pd.DataFrame]:
    authorize_sets(sets)
    source = _load_exp037()
    base = source._load_split_rows(sets)
    needed = source._needed_keys(base)
    keypoints = source._load_keypoints(needed, set(sets))
    bboxes = source._load_bboxes(base, needed)
    obd = source._load_obd(base)
    sequences = []
    metadata = []
    for row in base.itertuples(index=False):
        frames = range(int(row.start_frame), int(row.end_frame) + 1)
        keys = [(str(row.set_id), str(row.video_id), str(row.ped_id), frame) for frame in frames]
        coords = np.asarray([keypoints[key]["keypoints"] for key in keys], dtype=float)
        confidence = np.asarray([keypoints[key]["kp_conf"] for key in keys], dtype=float)
        boxes = np.asarray([bboxes[key] for key in keys], dtype=float)
        speed = np.asarray([
            obd[(str(row.set_id), str(row.video_id))][frame] for frame in frames
        ], dtype=float)
        sequences.append(sequence_core.build_sequence(coords, confidence, boxes, speed))
        metadata.append({name: getattr(row, name) for name in IDENTITY})
    output = np.stack(sequences).astype(np.float32)
    meta = pd.DataFrame(metadata).astype({"label": int})
    if len(output) != len(meta) or set(meta["set_id"].astype(str)) != set(sets):
        raise RuntimeError("sequence cohort materialization changed")
    return output, meta


def _save_npz(path: Path, sequences: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, sequences=sequences)
    _atomic_bytes(path, buffer.getvalue())


PREPARE_OUTPUTS = {
    "prepared_sequences.npz", "prepared_metadata.csv", "sequence_contract.json",
    "source_provenance.json",
}


def run_prepare(output_root: Path = EXPERIMENT_ROOT) -> str:
    root = Path(output_root)
    marker = root / "metrics/.prepare.complete.json"
    if marker.exists():
        _authenticate_manifest(root, "prepare", PREPARE_OUTPUTS)
        return "complete"
    registration = load_preregistration(root / "pre_registration.json")
    sequences, metadata = _build_sequences(ALLOWED_SETS)
    metrics = root / "metrics"
    sequence_path = metrics / "prepared_sequences.npz"
    metadata_path = metrics / "prepared_metadata.csv"
    contract_path = metrics / "sequence_contract.json"
    provenance_path = metrics / "source_provenance.json"
    _save_npz(sequence_path, sequences)
    _write_csv(metadata_path, metadata)
    _write_json(contract_path, {
        "channel_names": list(sequence_core.CHANNEL_NAMES),
        "channel_count": len(sequence_core.CHANNEL_NAMES),
        "shape": list(sequences.shape),
        "authorized_sets": list(ALLOWED_SETS),
        "forbidden_set": FORBIDDEN_SET,
        "architecture": registration["architecture"],
        "optimizer": registration["optimizer"],
    })
    exp037_prepare = _source_parent_record(EXP037_ROOT, "prepare")
    exp039_confirm = _source_parent_record(EXP039_ROOT, "confirm")
    source_manifest = _load_exp037()._source_manifest(ALLOWED_SETS)
    provenance = {
        "source_experiments": [exp037_prepare, exp039_confirm],
        "source_manifest_exp037": source_manifest,
        "set06_reuse": {
            "path": str((EXP039_ROOT / "metrics/confirmation_features.csv").resolve(strict=True)),
            **_record(EXP039_ROOT / "metrics/confirmation_features.csv"),
        },
        "set03_materialized": False,
    }
    _write_json(provenance_path, provenance)
    outputs = [sequence_path, metadata_path, contract_path, provenance_path]
    _publish(root, "prepare", outputs)
    return "complete"


def _load_prepared(root: Path) -> tuple[np.ndarray, pd.DataFrame]:
    _authenticate_manifest(root, "prepare", PREPARE_OUTPUTS)
    metrics = Path(root) / "metrics"
    sequences = np.load(metrics / "prepared_sequences.npz", allow_pickle=False)["sequences"]
    metadata = pd.read_csv(metrics / "prepared_metadata.csv")
    if sequences.shape != (len(metadata), 16, len(sequence_core.CHANNEL_NAMES)):
        raise RuntimeError("prepared sequence shape changed")
    if metadata["sequence_id"].duplicated().any() or set(metadata["set_id"].astype(str)) != set(ALLOWED_SETS):
        raise RuntimeError("prepared metadata cohort changed")
    return sequences.astype(np.float32), metadata


def _aggregate_frame() -> pd.DataFrame:
    source = _load_exp037()
    development = pd.read_csv(EXP037_ROOT / "metrics/prepared_development.csv")
    confirmation = pd.read_csv(EXP039_ROOT / "metrics/confirmation_features.csv")
    frame = pd.concat([development, confirmation], ignore_index=True)
    if frame["sequence_id"].duplicated().any() or set(frame["set_id"].astype(str)) != set(ALLOWED_SETS):
        raise RuntimeError("aggregate anchor cohort changed")
    if len(frame) != 5985:
        raise RuntimeError("aggregate anchor row count changed")
    return frame


def _fit_sequence(train_x: np.ndarray, train_y: np.ndarray, eval_x: np.ndarray, seed: int) -> np.ndarray:
    sequence_core.seed_everything(seed)
    torch.set_num_threads(1)
    mean = train_x.mean(axis=(0, 1), keepdims=True)
    scale = train_x.std(axis=(0, 1), keepdims=True)
    scale = np.where(scale < 1e-6, 1.0, scale)
    train = torch.from_numpy((train_x - mean) / scale).float()
    evaluation = torch.from_numpy((eval_x - mean) / scale).float()
    labels = torch.from_numpy(train_y.astype(np.float32))
    dataset = TensorDataset(train, labels)
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(dataset, batch_size=128, shuffle=True, generator=generator)
    model = sequence_core.TemporalConvClassifier(train.shape[2], seed=seed)
    positives = max(int(train_y.sum()), 1)
    negatives = max(int(len(train_y) - train_y.sum()), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives]))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    model.train()
    for _ in range(80):
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(evaluation)).numpy()


def _loso_sequence_predictions(sequences: np.ndarray, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    plans = sequence_core.loso_plans(metadata)
    parts = []
    audit = []
    for held_out, training_sets in plans:
        train_mask = metadata["set_id"].astype(str).isin(training_sets).to_numpy()
        eval_mask = metadata["set_id"].astype(str).eq(held_out).to_numpy()
        probabilities = []
        for seed in (42, 43, 44):
            probabilities.append(_fit_sequence(
                sequences[train_mask], metadata.loc[train_mask, "label"].to_numpy(int), sequences[eval_mask], seed
            ))
        prediction = metadata.loc[eval_mask, IDENTITY].copy()
        prediction["arm"] = "temporal_biu_tcn"
        prediction["probability"] = np.column_stack(probabilities).mean(axis=1)
        parts.append(prediction)
        audit.append({
            "held_out_set": held_out, "training_sets": list(training_sets),
            "train_rows": int(train_mask.sum()), "evaluation_rows": int(eval_mask.sum()),
            "seed_count": 3, "epochs": 80,
        })
    output = pd.concat(parts, ignore_index=True).sort_values("sequence_id", kind="stable").reset_index(drop=True)
    if len(output) != len(metadata) or output["sequence_id"].duplicated().any():
        raise RuntimeError("sequence OOF coverage changed")
    return output, pd.DataFrame(audit)


def _loso_anchor_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    source = _load_exp037()
    contract = json.loads((EXP037_ROOT / "metrics/feature_contract.json").read_text(encoding="utf-8"))
    arms = {arm: tuple(contract["arms"][arm]) for arm in ("v2_bbox_xgb", "joint_temporal_ego_xgb")}
    outputs = []
    for held_out in ALLOWED_SETS:
        training_sets = tuple(item for item in ALLOWED_SETS if item != held_out)
        train = frame.loc[frame["set_id"].astype(str).isin(training_sets)]
        evaluation = frame.loc[frame["set_id"].astype(str).eq(held_out)]
        for arm, columns in arms.items():
            probability = source.model_core.fit_predict_seed_mean(
                train, evaluation, columns, arm, source.model_core.SEEDS
            )
            prediction = evaluation.loc[:, IDENTITY].copy()
            prediction["arm"] = arm
            prediction["probability"] = probability
            outputs.append(prediction)
    return pd.concat(outputs, ignore_index=True).sort_values(["arm", "sequence_id"], kind="stable").reset_index(drop=True)


def _metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    source = _load_exp037()
    rows = []
    for arm, group in predictions.groupby("arm", sort=True):
        rows.append({"scope": "development_loso", "arm": arm, **source.model_core.compute_metrics(
            group["label"].to_numpy(int), group["probability"].to_numpy(float)
        )})
        for set_id, subset in group.groupby("set_id", sort=True):
            rows.append({"scope": "development_loso", "set_id": str(set_id), "arm": arm, **source.model_core.compute_metrics(
                subset["label"].to_numpy(int), subset["probability"].to_numpy(float)
            )})
    return pd.DataFrame(rows)


def _bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
    source = _load_exp037()
    wide = predictions.pivot(index=IDENTITY, columns="arm", values="probability").reset_index()
    rows = []
    for reference in ("joint_temporal_ego_xgb", "v2_bbox_xgb"):
        evidence = source.model_core.cluster_bootstrap_deltas(
            wide, reference, "temporal_biu_tcn", replicates=10000, seed=42
        )
        rows.append(evidence.assign(reference_arm=reference))
    return pd.concat(rows, ignore_index=True)


def evaluate_gate(results: pd.DataFrame, bootstrap: pd.DataFrame) -> dict[str, Any]:
    indexed = results.loc[
        results["scope"].eq("development_loso")
        & results["set_id"].isna()
        & results["arm"].isin(
            ["temporal_biu_tcn", "joint_temporal_ego_xgb", "v2_bbox_xgb"]
        )
    ].set_index("arm")
    candidate = indexed.loc["temporal_biu_tcn"]
    joint = indexed.loc["joint_temporal_ego_xgb"]
    v2 = indexed.loc["v2_bbox_xgb"]
    evidence = bootstrap.set_index(["reference_arm", "metric"])
    checks = {
        "pr_auc_gain_vs_joint": bool(float(candidate["pr_auc"]) > float(joint["pr_auc"]) + 0.005),
        "f1_gain_vs_joint": bool(float(candidate["f1_fixed_0p5"]) > float(joint["f1_fixed_0p5"]) + 0.005),
        "pr_auc_gain_vs_v2": bool(float(candidate["pr_auc"]) > float(v2["pr_auc"]) + 0.005),
        "f1_gain_vs_v2": bool(float(candidate["f1_fixed_0p5"]) > float(v2["f1_fixed_0p5"]) + 0.005),
        "bootstrap_pr_vs_joint_positive": bool(float(evidence.loc[("joint_temporal_ego_xgb", "pr_auc"), "ci_low"]) > 0),
        "bootstrap_f1_vs_joint_positive": bool(float(evidence.loc[("joint_temporal_ego_xgb", "f1_fixed_0p5"), "ci_low"]) > 0),
        "fp_tail_nonincrease": bool(int(candidate["fp_ge_0p9_count"]) <= int(joint["fp_ge_0p9_count"])),
        "fn_tail_nonincrease": bool(int(candidate["fn_le_0p1_count"]) <= int(joint["fn_le_0p1_count"])),
    }
    return {
        "status": "promising" if all(checks.values()) else "not_promising",
        "checks": checks,
        "observed": {
            "candidate_pr_auc": float(candidate["pr_auc"]), "joint_pr_auc": float(joint["pr_auc"]),
            "candidate_f1": float(candidate["f1_fixed_0p5"]), "joint_f1": float(joint["f1_fixed_0p5"]),
            "candidate_high_confidence_errors": int(candidate["fp_ge_0p9_count"] + candidate["fn_le_0p1_count"]),
            "joint_high_confidence_errors": int(joint["fp_ge_0p9_count"] + joint["fn_le_0p1_count"]),
        },
    }


DEVELOP_OUTPUTS = {
    "sequence_oof_predictions.csv", "anchor_oof_predictions.csv", "development_results.csv",
    "development_per_set.csv", "development_bootstrap.csv", "fold_audit.csv",
    "gate_checks.json", "development_lock.json", "source_provenance_develop.json",
}


def run_develop(output_root: Path = EXPERIMENT_ROOT) -> str:
    root = Path(output_root)
    marker = root / "metrics/.develop.complete.json"
    if marker.exists():
        completion = _authenticate_manifest(root, "develop", DEVELOP_OUTPUTS)
        return str(completion.get("promotion_status", "not_promising"))
    _authenticate_manifest(root, "prepare", PREPARE_OUTPUTS)
    sequences, metadata = _load_prepared(root)
    sequence_predictions, fold_audit = _loso_sequence_predictions(sequences, metadata)
    anchors = _loso_anchor_predictions(_aggregate_frame())
    all_predictions = pd.concat([sequence_predictions, anchors], ignore_index=True)
    results = _metric_rows(all_predictions)
    bootstrap = _bootstrap(all_predictions)
    gate = evaluate_gate(results, bootstrap)
    metrics = root / "metrics"
    outputs = []
    for name, frame in {
        "sequence_oof_predictions.csv": sequence_predictions,
        "anchor_oof_predictions.csv": anchors,
        "development_results.csv": results.loc[results["scope"].eq("development_loso") & ~results["set_id"].notna()].copy(),
        "development_per_set.csv": results.loc[results["set_id"].notna()].copy(),
        "development_bootstrap.csv": bootstrap,
        "fold_audit.csv": fold_audit,
    }.items():
        path = metrics / name
        _write_csv(path, frame)
        outputs.append(path)
    gate_path = metrics / "gate_checks.json"
    _write_json(gate_path, gate)
    outputs.append(gate_path)
    lock = {
        "schema_version": 1, "phase": "develop", "status": gate["status"],
        "authorized_sets": list(ALLOWED_SETS), "forbidden_set": FORBIDDEN_SET,
        "threshold": 0.5, "seeds": [42, 43, 44], "epochs": 80,
        "preregistration_sha256": _sha256(root / "pre_registration.json"),
    }
    lock_path = metrics / "development_lock.json"
    _write_json(lock_path, lock)
    outputs.append(lock_path)
    provenance_path = metrics / "source_provenance_develop.json"
    _write_json(provenance_path, {
        "prepare_completion": _record(metrics / ".prepare.complete.json"),
        "set03_materialized": False,
        "authorized_sets": list(ALLOWED_SETS),
    })
    outputs.append(provenance_path)
    _publish(root, "develop", outputs, promotion_status=gate["status"])
    return gate["status"]


def run_report(output_root: Path = EXPERIMENT_ROOT) -> str:
    root = Path(output_root)
    _authenticate_manifest(root, "develop", DEVELOP_OUTPUTS)
    metrics = root / "metrics"
    results = pd.read_csv(metrics / "development_results.csv")
    per_set = pd.read_csv(metrics / "development_per_set.csv")
    gate = json.loads((metrics / "gate_checks.json").read_text(encoding="utf-8"))
    lock = json.loads((metrics / "development_lock.json").read_text(encoding="utf-8"))
    def markdown_table(frame: pd.DataFrame) -> str:
        """Render a small report table without requiring optional tabulate."""
        columns = [str(column) for column in frame.columns]
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for values in frame.itertuples(index=False, name=None):
            cells = ["" if pd.isna(value) else str(value) for value in values]
            lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |")
        return "\n".join(lines)

    lines = [
        "# Exp040 Results", "", "## Scientific Status", "",
        "Exp040 is a five-set development-only LOSO diagnosis. Set06 was already opened by Exp039; set03 remains sealed.",
        f"", f"Promotion status: `{lock['status']}`.", "", "## Pooled Results", "",
        markdown_table(results), "", "## Per-Set Results", "", markdown_table(per_set),
        "", "## Gate", "", "```json", json.dumps(gate, indent=2, sort_keys=True), "```", "",
        "The temporal sequence model uses no RGB or scene context. The cited TCN paper is methodological motivation only; its results are not PIE evidence.",
    ]
    _atomic_bytes(root / "RESULTS.md", ("\n".join(lines) + "\n").encode())
    return str(root / "RESULTS.md")


def run_verify(output_root: Path = EXPERIMENT_ROOT) -> str:
    root = Path(output_root)
    _authenticate_manifest(root, "prepare", PREPARE_OUTPUTS)
    _authenticate_manifest(root, "develop", DEVELOP_OUTPUTS)
    metrics = root / "metrics"
    forbidden = [path.name for path in root.rglob("*set03*")]
    sequence = pd.read_csv(metrics / "sequence_oof_predictions.csv")
    if len(sequence) != 5985 or set(sequence["set_id"].astype(str)) != set(ALLOWED_SETS):
        raise RuntimeError("sequence OOF cohort changed")
    report = {
        "status": "PASS" if not forbidden else "FAIL",
        "authorized_sets": list(ALLOWED_SETS), "forbidden_set": FORBIDDEN_SET,
        "sequence_oof_rows": int(len(sequence)), "set03_outputs": forbidden,
    }
    _write_json(metrics / "verification_report.json", report)
    if forbidden:
        raise RuntimeError("set03 output exists")
    return report["status"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "develop", "report", "verify", "all"))
    args = parser.parse_args()
    if args.phase in {"prepare", "all"}:
        print(f"prepare: {run_prepare()}")
    if args.phase in {"develop", "all"}:
        print(f"develop: {run_develop()}")
    if args.phase in {"report", "all"}:
        print(f"report: {run_report()}")
    if args.phase in {"verify", "all"}:
        print(f"verify: {run_verify()}")


if __name__ == "__main__":
    main()
