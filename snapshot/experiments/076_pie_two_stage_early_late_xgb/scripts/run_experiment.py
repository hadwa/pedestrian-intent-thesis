"""Phase runner for the frozen Exp076 PIE comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import data_core  # noqa: E402
import evaluation  # noqa: E402
import two_stage_core  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOT = Path(__file__).resolve().parents[3]
EXP072 = ACTIVE_ROOT / "experiments/072_pie_joint_biu_temporal_ego_confirmation"
FEATURE_CONTRACT = ACTIVE_ROOT / "experiments/037_pie_joint_biu_temporal_ego/metrics/feature_contract.json"
DEFINITIONS = ACTIVE_ROOT / "experiments/012_pie_biu_v2_30_60_benchmark/metrics/pie_biu_v2_feature_set_definitions.csv"
KEYPOINTS = Path("/home/abdelrahman/Hadwa/pedestrian-intent-biu/data/data_keypoints_all.jsonl")
ANNOTATIONS = Path("/home/abdelrahman/Hadwa/MSLSTM-PID/PIE/annotations/annotations")
OBD = Path("/home/abdelrahman/Hadwa/MSLSTM-PID/PIE/annotations/annotations_vehicle")
SEEDS = two_stage_core.SEEDS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256(resolved), "size_bytes": resolved.stat().st_size}


def _atomic(path: Path, payload: bytes) -> None:
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


def write_json(path: Path, value: Any) -> None:
    _atomic(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic(path, frame.to_csv(index=False).encode())


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(name) for name in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def completion(phase: str, outputs: list[Path]) -> None:
    write_json(
        ROOT / "metrics" / f".{phase}.complete.json",
        {"phase": phase, "status": "complete", "outputs": [artifact(path) for path in outputs]},
    )


def _contract() -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    contract = two_stage_core.load_joint_contract(DEFINITIONS)
    feature_contract = json.loads(FEATURE_CONTRACT.read_text(encoding="utf-8"))
    full_columns = tuple(feature_contract["arms"]["joint_temporal_ego_xgb"])
    if len(full_columns) != 187 or tuple(full_columns[:182]) != tuple(contract["portable"]):
        raise RuntimeError("Exp072 full feature contract changed")
    return contract, full_columns


def _source_frame(split: str) -> pd.DataFrame:
    name = {"train": "prepared_train.csv", "validation": "prepared_validation.csv", "test": "prepared_test.csv"}[split]
    frame = pd.read_csv(EXP072 / "metrics" / name)
    data_core.validate_window_cohort(frame, tte_column="time_to_event_frames")
    return frame


def _materialize(frame: pd.DataFrame, *, contract: dict[str, tuple[str, ...]], full_columns: tuple[str, ...]) -> tuple[pd.DataFrame, dict[str, Any]]:
    needed = data_core.needed_keys(frame)
    keypoints, keypoint_audit = data_core.load_keypoint_index((KEYPOINTS,), needed)
    bboxes = data_core.load_pie_bboxes(frame, ANNOTATIONS)
    speed = data_core.load_pie_speed(frame, OBD)
    views, view_audit = data_core.materialize_views(frame, keypoints, bboxes, contract=contract, speed=speed)
    portable_differences = []
    for name in contract["portable"]:
        portable_differences.append(
            np.abs(views[f"portable__{name}"].to_numpy(float) - frame[name].to_numpy(float))
        )
    difference = np.concatenate(portable_differences)
    if not np.isfinite(difference).all() or float(difference.max()) > 1e-9:
        raise RuntimeError(f"raw reconstruction differs from Exp072: {float(np.nanmax(difference))}")
    early_columns = [f"early__{name}" for name in contract["joint"]]
    early_ego_columns = [f"early_ego__{name}" for name in full_columns[-5:]]
    meta = [
        name for name in (
            "sequence_id", "split", "set_id", "video_id", "ped_id", "track_id", "start_frame",
            "end_frame", "label", "observation_length", "frame_step", "frame_sampling", "time_to_event_frames",
        ) if name in frame.columns
    ]
    prepared = frame.loc[:, [*meta, *full_columns]].copy()
    prepared = prepared.merge(
        views.loc[:, ["sequence_id", *early_columns, *early_ego_columns]],
        on="sequence_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    data_core.two_stage_core.assert_cohort_parity(frame, prepared)
    return prepared, {
        "keypoints": keypoint_audit,
        "views": view_audit,
        "raw_reconstruction_max_abs_difference": float(difference.max()),
        "full_feature_count": len(full_columns),
        "early_feature_count": len(early_columns) + len(early_ego_columns),
    }


def run_prepare() -> None:
    contract, full_columns = _contract()
    outputs = []
    audits = {}
    for split in ("train", "validation"):
        source = _source_frame(split)
        expected = {"train": 2366, "validation": 3619}[split]
        if len(source) != expected:
            raise RuntimeError(f"Exp076 {split} row count changed")
        prepared, audit = _materialize(source, contract=contract, full_columns=full_columns)
        path = ROOT / "metrics" / f"prepared_{split}.csv"
        write_csv(path, prepared)
        outputs.append(path)
        audits[split] = audit
    audit_path = ROOT / "metrics/preparation_audit.json"
    provenance_path = ROOT / "metrics/source_provenance.json"
    write_json(audit_path, audits)
    write_json(
        provenance_path,
        {
            "pre_registration": artifact(ROOT / "pre_registration.json"),
            "definitions": artifact(DEFINITIONS),
            "keypoints": artifact(KEYPOINTS),
            "exp072_train": artifact(EXP072 / "metrics/prepared_train.csv"),
            "exp072_validation": artifact(EXP072 / "metrics/prepared_validation.csv"),
            "exp037_feature_contract_pinned_by_exp072": artifact(FEATURE_CONTRACT),
        },
    )
    outputs.extend([audit_path, provenance_path])
    completion("prepare", outputs)


def _feature_columns() -> tuple[tuple[str, ...], tuple[str, ...]]:
    contract, full = _contract()
    early = tuple(f"early__{name}" for name in contract["joint"]) + tuple(
        f"early_ego__{name}" for name in full[-5:]
    )
    if len(early) != 166:
        raise RuntimeError("Exp076 early feature count changed")
    return early, full


def _parent_one_stage(seed: int):
    path = EXP072 / "models" / f"candidate_seed{seed}.joblib"
    return joblib.load(path), path


def run_validate() -> None:
    train = pd.read_csv(ROOT / "metrics/prepared_train.csv")
    validation = pd.read_csv(ROOT / "metrics/prepared_validation.csv")
    data_core.two_stage_core.assert_cohort_parity(_source_frame("train"), train)
    data_core.two_stage_core.assert_cohort_parity(_source_frame("validation"), validation)
    early_columns, full_columns = _feature_columns()
    labels = train["label"].to_numpy(int)
    groups = train["track_id"].astype(str).to_numpy()
    early_train = train.loc[:, early_columns].to_numpy(float)
    full_train = train.loc[:, full_columns].to_numpy(float)
    early_validation = validation.loc[:, early_columns].to_numpy(float)
    full_validation = validation.loc[:, full_columns].to_numpy(float)
    prediction_parts = []
    model_manifest = {}
    oof_audits = {}
    outputs = []
    for seed in SEEDS:
        one_stage, one_path = _parent_one_stage(seed)
        two_stage, audit = two_stage_core.fit_two_stage(
            early_train,
            full_train,
            labels,
            groups,
            seed=seed,
            total_estimators=250,
            n_splits=5,
        )
        model_path = ROOT / "models" / f"two_stage_seed{seed}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(two_stage, model_path)
        one_probability = one_stage.predict_proba(full_validation)[:, 1]
        two_probability = two_stage_core.predict_two_stage(two_stage, early_validation, full_validation)
        prediction = validation.loc[:, ["sequence_id", "set_id", "video_id", "track_id", "label"]].copy()
        prediction.insert(0, "seed", seed)
        prediction["one_stage_probability"] = one_probability
        prediction["two_stage_probability"] = two_probability
        prediction_path = ROOT / "predictions" / f"validation_seed{seed}.csv"
        write_csv(prediction_path, prediction)
        outputs.extend([model_path, prediction_path])
        prediction_parts.append(prediction)
        model_manifest[str(seed)] = {"one_stage_parent": artifact(one_path), "two_stage": artifact(model_path)}
        oof_audits[str(seed)] = audit
    predictions = pd.concat(prediction_parts, ignore_index=True)
    per_seed, deltas = evaluation.paired_results(predictions, scope="validation")
    summary = evaluation.summarize(per_seed)
    paths = {
        "predictions": ROOT / "predictions/validation_all_seeds.csv",
        "per_seed": ROOT / "metrics/validation_per_seed_results.csv",
        "summary": ROOT / "metrics/validation_summary_results.csv",
        "deltas": ROOT / "metrics/validation_paired_deltas.csv",
        "manifest": ROOT / "metrics/model_manifest.json",
        "oof": ROOT / "metrics/oof_audit.json",
    }
    write_csv(paths["predictions"], predictions)
    write_csv(paths["per_seed"], per_seed)
    write_csv(paths["summary"], summary)
    write_csv(paths["deltas"], deltas)
    write_json(paths["manifest"], model_manifest)
    write_json(paths["oof"], oof_audits)
    outputs.extend(paths.values())
    completion("validate", outputs)


def run_evaluate() -> None:
    contract, full_columns = _contract()
    source = _source_frame("test")
    if len(source) != 3760:
        raise RuntimeError("Exp076 test row count changed")
    test, audit = _materialize(source, contract=contract, full_columns=full_columns)
    prepared_path = ROOT / "metrics/prepared_test.csv"
    audit_path = ROOT / "metrics/test_preparation_audit.json"
    write_csv(prepared_path, test)
    write_json(audit_path, audit)
    early_columns, full_columns = _feature_columns()
    early_test = test.loc[:, early_columns].to_numpy(float)
    full_test = test.loc[:, full_columns].to_numpy(float)
    prediction_parts = []
    outputs = [prepared_path, audit_path]
    for seed in SEEDS:
        one_stage, _ = _parent_one_stage(seed)
        two_stage = joblib.load(ROOT / "models" / f"two_stage_seed{seed}.joblib")
        prediction = test.loc[:, ["sequence_id", "set_id", "video_id", "track_id", "label"]].copy()
        prediction.insert(0, "seed", seed)
        prediction["one_stage_probability"] = one_stage.predict_proba(full_test)[:, 1]
        prediction["two_stage_probability"] = two_stage_core.predict_two_stage(two_stage, early_test, full_test)
        path = ROOT / "predictions" / f"test_seed{seed}.csv"
        write_csv(path, prediction)
        outputs.append(path)
        prediction_parts.append(prediction)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    per_seed, deltas = evaluation.paired_results(predictions, scope="test")
    summary = evaluation.summarize(per_seed)
    paths = {
        "predictions": ROOT / "predictions/test_all_seeds.csv",
        "per_seed": ROOT / "metrics/test_per_seed_results.csv",
        "summary": ROOT / "metrics/test_summary_results.csv",
        "deltas": ROOT / "metrics/test_paired_deltas.csv",
    }
    write_csv(paths["predictions"], predictions)
    write_csv(paths["per_seed"], per_seed)
    write_csv(paths["summary"], summary)
    write_csv(paths["deltas"], deltas)
    outputs.extend(paths.values())
    completion("evaluate", outputs)


def run_report() -> None:
    validation = pd.read_csv(ROOT / "metrics/validation_summary_results.csv")
    test = pd.read_csv(ROOT / "metrics/test_summary_results.csv")
    deltas = pd.read_csv(ROOT / "metrics/test_paired_deltas.csv")
    lines = [
        "# Exp076 Final Report",
        "",
        "PIE 16 observed frames, TTE 30--60, overlap 0.6, fixed threshold 0.5.",
        "The one-stage arm has 250 trees. The two-stage arm has 125 + 125 trees",
        "and receives a track-group OOF frame-8 probability during stage-2 training.",
        "",
        "## Validation",
        "",
        markdown_table(validation),
        "",
        "## Set03",
        "",
        markdown_table(test),
        "",
        "## Paired Two-Stage Minus One-Stage Deltas",
        "",
        markdown_table(deltas),
        "",
        "Set03 has prior exposure in separate experiments. This is a frozen paired",
        "ablation and not a pristine untouched confirmation.",
    ]
    report = ROOT / "FINAL_REPORT.md"
    _atomic(report, ("\n".join(lines) + "\n").encode())
    completion("report", [report])
    _atomic(ROOT / "EXPERIMENT_COMPLETE", b"complete\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "validate", "evaluate", "report", "all"))
    args = parser.parse_args()
    phases = ("prepare", "validate", "evaluate", "report") if args.phase == "all" else (args.phase,)
    functions = {"prepare": run_prepare, "validate": run_validate, "evaluate": run_evaluate, "report": run_report}
    for phase in phases:
        print(f"[Exp076] {phase}", flush=True)
        functions[phase]()


if __name__ == "__main__":
    main()
