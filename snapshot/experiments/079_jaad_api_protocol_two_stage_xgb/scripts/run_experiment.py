"""Exp077 one/two-stage XGBoost rerun on Exp031 Protocol A JAAD windows."""

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


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOT = Path(__file__).resolve().parents[3]
EXP076 = ACTIVE_ROOT / "experiments/076_pie_two_stage_early_late_xgb"
EXP077 = ACTIVE_ROOT / "experiments/077_jaad_two_stage_early_late_xgb_transfer"
EXP077_ZERO_SHOT_MODELS = EXP077 / "models"
EXP076_SCRIPTS = EXP076 / "scripts"
if str(EXP076_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(EXP076_SCRIPTS))

import data_core  # noqa: E402
import evaluation  # noqa: E402
import two_stage_core  # noqa: E402


PROTOCOL_WINDOWS = (
    ACTIVE_ROOT / "experiments/031_jaad_api_label_protocol_sensitivity/metrics/protocol_a_window_audit.csv"
)
DEFINITIONS = (
    ACTIVE_ROOT / "experiments/012_pie_biu_v2_30_60_benchmark/metrics/pie_biu_v2_feature_set_definitions.csv"
)
POSE_PATHS = (
    ACTIVE_ROOT / "experiments/049_exp048_to_jaad_zero_shot/cache/jaad_keypoints_vitpose_s_exp049_extended.jsonl",
    ROOT / "cache/jaad_api_protocol_missing_pose.jsonl",
)
SEEDS = two_stage_core.SEEDS
TOTAL_TREE_BUDGET = 250
OOF_SPLITS = 5
FIXED_THRESHOLD = 0.5
EXPECTED_ROWS = {"train": 8613, "validation": 1265, "test": 6732}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
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
    columns = list(map(str, frame.columns))
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def complete(phase: str, outputs: list[Path]) -> None:
    write_json(
        ROOT / "metrics" / f".{phase}.complete.json",
        {"phase": phase, "status": "complete", "outputs": [artifact(path) for path in outputs]},
    )


def feature_contract() -> dict[str, tuple[str, ...]]:
    return two_stage_core.load_joint_contract(DEFINITIONS)


def observed_frames(row: Any) -> tuple[int, ...]:
    frames = tuple(int(value) for value in json.loads(str(row.observed_frames_json)))
    if len(frames) != 16 or any(right <= left for left, right in zip(frames, frames[1:])):
        raise ValueError(f"Protocol A sequence {row.sequence_id} does not contain 16 ordered observations")
    return frames


def _validate_protocol_a(frame: pd.DataFrame) -> None:
    required = {
        "sequence_id", "track_id", "label", "observation_length", "frame_step", "frame_sampling",
        "tte_frames", "observed_frames_json", "bbox_sequence_json",
    }
    if required.difference(frame.columns) or frame.empty or frame["sequence_id"].duplicated().any():
        raise ValueError("Protocol A cohort schema or identity is invalid")
    exact = (
        frame["observation_length"].astype(int).eq(16)
        & frame["frame_step"].astype(int).eq(1)
        & frame["frame_sampling"].astype(str).eq("consecutive")
        & frame["tte_frames"].astype(int).between(30, 60, inclusive="both")
        & frame["label"].astype(int).isin([0, 1])
    )
    if not exact.all():
        raise ValueError("Protocol A 16-observation/nominal-TTE contract changed")
    for row in frame.itertuples(index=False):
        observed_frames(row)


def load_jaad_split(split: str) -> pd.DataFrame:
    if split not in EXPECTED_ROWS:
        raise ValueError(f"unsupported split: {split}")
    source_split = "val" if split == "validation" else split
    frame = pd.read_csv(PROTOCOL_WINDOWS)
    frame = frame.loc[frame["split"].astype(str).eq(source_split)].copy()
    frame["set_id"] = "jaad"
    frame["split"] = split
    frame = frame.sort_values("sequence_id", kind="stable").reset_index(drop=True)
    _validate_protocol_a(frame)
    if len(frame) != EXPECTED_ROWS[split]:
        raise RuntimeError(f"Protocol A {split} row count changed: {len(frame)}")
    if not frame["protocol"].astype(str).eq("api_style_reconstruction").all():
        raise RuntimeError("Protocol A identity changed")
    return frame


def _needed_keys(frame: pd.DataFrame) -> set[tuple[str, str, str, int]]:
    return {
        ("jaad", str(row.video_id), str(row.ped_id), frame_idx)
        for row in frame.itertuples(index=False)
        for frame_idx in observed_frames(row)
    }


def _load_bboxes(frame: pd.DataFrame) -> dict[tuple[str, str, str, int], np.ndarray]:
    output = {}
    for row in frame.itertuples(index=False):
        frames = observed_frames(row)
        boxes = np.asarray(json.loads(str(row.bbox_sequence_json)), dtype=float)
        if boxes.shape != (16, 4) or not np.isfinite(boxes).all():
            raise ValueError(f"invalid Protocol A BBoxes for {row.sequence_id}")
        for frame_idx, box in zip(frames, boxes, strict=True):
            key = ("jaad", str(row.video_id), str(row.ped_id), frame_idx)
            if key in output and not np.array_equal(output[key], box):
                raise ValueError(f"conflicting Protocol A BBox for {key}")
            output[key] = box
    missing = _needed_keys(frame).difference(output)
    if missing:
        raise ValueError(f"missing exact Protocol A BBox: {sorted(missing)[0]}")
    return output


def _materialize_views(
    source: pd.DataFrame,
    keypoints: dict[tuple[str, str, str, int], dict[str, Any]],
    bboxes: dict[tuple[str, str, str, int], np.ndarray],
    contract: dict[str, tuple[str, ...]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for row in source.itertuples(index=False):
        frames = observed_frames(row)
        keys = [("jaad", str(row.video_id), str(row.ped_id), frame_idx) for frame_idx in frames]
        if any(key not in keypoints or key not in bboxes for key in keys):
            raise ValueError(f"missing exact source for {row.sequence_id}")
        coords = np.asarray([keypoints[key]["keypoints"] for key in keys], dtype=float)
        confidence = np.asarray([keypoints[key]["kp_conf"] for key in keys], dtype=float)
        boxes = np.asarray([bboxes[key] for key in keys], dtype=float)
        views = two_stage_core.extract_window_views(coords, confidence, boxes, contract=contract)
        result = {
            name: getattr(row, name)
            for name in (
                "sequence_id", "split", "set_id", "video_id", "ped_id", "track_id", "start_frame", "end_frame",
                "label", "observation_length", "frame_step", "frame_sampling", "tte_frames", "protocol",
                "api_flaw_affected", "temporal_position",
            )
        }
        result.update({f"early__{name}": value for name, value in views["early"].items()})
        result.update({f"portable__{name}": value for name, value in views["portable"].items()})
        rows.append(result)
    output = pd.DataFrame(rows)
    if len(output) != len(source) or output["sequence_id"].duplicated().any():
        raise RuntimeError("Protocol A materialized cohort changed")
    two_stage_core.assert_cohort_parity(source, output)
    gap_count = sum(observed_frames(row)[-1] - observed_frames(row)[0] != 15 for row in source.itertuples(index=False))
    return output, {
        "rows": len(output),
        "gapped_api_trajectory_windows": int(gap_count),
        "early_feature_count": 161,
        "portable_feature_count": 182,
    }


def materialize_jaad(split: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = load_jaad_split(split)
    contract = feature_contract()
    needed = _needed_keys(source)
    keypoints, keypoint_audit = data_core.load_keypoint_index(POSE_PATHS, needed)
    bboxes = _load_bboxes(source)
    prepared, view_audit = _materialize_views(source, keypoints, bboxes, contract)
    prepared = prepared.rename(columns={f"portable__{name}": name for name in contract["portable"]})
    two_stage_core.assert_cohort_parity(source, prepared)
    return prepared, {"keypoints": keypoint_audit, "views": view_audit, "source_rows": len(source)}


def run_prepare() -> None:
    outputs = []
    audits = {}
    for split in ("train", "validation"):
        prepared, audit = materialize_jaad(split)
        path = ROOT / "metrics" / f"prepared_jaad_{split}.csv"
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
            "protocol_a_windows": artifact(PROTOCOL_WINDOWS),
            "definitions": artifact(DEFINITIONS),
            "pose_sources": [artifact(path) for path in POSE_PATHS],
            "pie_prepared_train": artifact(EXP076 / "metrics/prepared_train.csv"),
            "exp077_pre_registration": artifact(EXP077 / "pre_registration.json"),
            "reused_zero_shot_models": [
                artifact(EXP077_ZERO_SHOT_MODELS / f"pie_zero_shot_{model}_seed{seed}.joblib")
                for model in ("one_stage", "two_stage")
                for seed in SEEDS
            ],
            "shared_data_core": artifact(EXP076_SCRIPTS / "data_core.py"),
            "shared_two_stage_core": artifact(EXP076_SCRIPTS / "two_stage_core.py"),
            "shared_evaluation": artifact(EXP076_SCRIPTS / "evaluation.py"),
        },
    )
    outputs.extend([audit_path, provenance_path])
    complete("prepare", outputs)


def _matrices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    contract = feature_contract()
    early = frame.loc[:, [f"early__{name}" for name in contract["joint"]]].to_numpy(float)
    full = frame.loc[:, contract["portable"]].to_numpy(float)
    return early, full


def _fit_native(train: pd.DataFrame) -> tuple[dict[int, Any], dict[int, Any], dict[str, Any], list[Path]]:
    early, full = _matrices(train)
    labels = train["label"].to_numpy(int)
    groups = train["track_id"].astype(str).to_numpy()
    one_models, two_models, audits, outputs = {}, {}, {}, []
    for seed in SEEDS:
        one = two_stage_core.build_xgb(seed, labels, n_estimators=TOTAL_TREE_BUDGET)
        one.fit(full, labels)
        two, audit = two_stage_core.fit_two_stage(
            early, full, labels, groups, seed=seed, total_estimators=TOTAL_TREE_BUDGET, n_splits=OOF_SPLITS
        )
        one_path = ROOT / "models" / f"jaad_native_one_stage_seed{seed}.joblib"
        two_path = ROOT / "models" / f"jaad_native_two_stage_seed{seed}.joblib"
        one_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(one, one_path)
        joblib.dump(two, two_path)
        one_models[seed], two_models[seed] = one, two
        audits[str(seed)] = audit
        outputs.extend([one_path, two_path])
    return one_models, two_models, audits, outputs


def _predict_pair(frame: pd.DataFrame, one_models: dict[int, Any], two_models: dict[int, Any]) -> pd.DataFrame:
    early, full = _matrices(frame)
    parts = []
    for seed in SEEDS:
        part = frame.loc[
            :, ["sequence_id", "set_id", "video_id", "track_id", "label", "api_flaw_affected", "temporal_position"]
        ].copy()
        part.insert(0, "seed", seed)
        part["one_stage_probability"] = one_models[seed].predict_proba(full)[:, 1]
        part["two_stage_probability"] = two_stage_core.predict_two_stage(two_models[seed], early, full)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _load_native_models() -> tuple[dict[int, Any], dict[int, Any]]:
    one = {seed: joblib.load(ROOT / "models" / f"jaad_native_one_stage_seed{seed}.joblib") for seed in SEEDS}
    two = {seed: joblib.load(ROOT / "models" / f"jaad_native_two_stage_seed{seed}.joblib") for seed in SEEDS}
    return one, two


def _load_zero_shot_models() -> tuple[dict[int, Any], dict[int, Any]]:
    one = {seed: joblib.load(EXP077_ZERO_SHOT_MODELS / f"pie_zero_shot_one_stage_seed{seed}.joblib") for seed in SEEDS}
    two = {seed: joblib.load(EXP077_ZERO_SHOT_MODELS / f"pie_zero_shot_two_stage_seed{seed}.joblib") for seed in SEEDS}
    return one, two


def run_validate() -> None:
    train = pd.read_csv(ROOT / "metrics/prepared_jaad_train.csv")
    validation = pd.read_csv(ROOT / "metrics/prepared_jaad_validation.csv")
    one, two, oof_audit, model_outputs = _fit_native(train)
    predictions = _predict_pair(validation, one, two)
    per_seed, deltas = evaluation.paired_results(predictions, scope="jaad_api_protocol_native_validation")
    summary = evaluation.summarize(per_seed)
    paths = {
        "predictions": ROOT / "predictions/jaad_native_validation_all_seeds.csv",
        "per_seed": ROOT / "metrics/validation_per_seed_results.csv",
        "summary": ROOT / "metrics/validation_summary_results.csv",
        "deltas": ROOT / "metrics/validation_paired_deltas.csv",
        "oof": ROOT / "metrics/oof_audit.json",
    }
    write_csv(paths["predictions"], predictions)
    write_csv(paths["per_seed"], per_seed)
    write_csv(paths["summary"], summary)
    write_csv(paths["deltas"], deltas)
    write_json(paths["oof"], oof_audit)
    complete("validate", [*model_outputs, *paths.values()])


def run_evaluate() -> None:
    test, audit = materialize_jaad("test")
    test_path = ROOT / "metrics/prepared_jaad_test.csv"
    audit_path = ROOT / "metrics/test_preparation_audit.json"
    write_csv(test_path, test)
    write_json(audit_path, audit)
    metric_parts, delta_parts, outputs = [], [], [test_path, audit_path]
    settings = (
        ("pie_to_jaad_api_zero_shot_test", *_load_zero_shot_models(), "pie_zero_shot"),
        ("jaad_api_protocol_native_test", *_load_native_models(), "jaad_native"),
    )
    for scope, one, two, filename in settings:
        predictions = _predict_pair(test, one, two)
        prediction_path = ROOT / "predictions" / f"{filename}_test_all_seeds.csv"
        write_csv(prediction_path, predictions)
        outputs.append(prediction_path)
        per_seed, deltas = evaluation.paired_results(predictions, scope=scope)
        metric_parts.append(per_seed)
        delta_parts.append(deltas)
    per_seed = pd.concat(metric_parts, ignore_index=True)
    deltas = pd.concat(delta_parts, ignore_index=True)
    summary = evaluation.summarize(per_seed)
    paths = {
        "per_seed": ROOT / "metrics/test_per_seed_results.csv",
        "summary": ROOT / "metrics/test_summary_results.csv",
        "deltas": ROOT / "metrics/test_paired_deltas.csv",
    }
    write_csv(paths["per_seed"], per_seed)
    write_csv(paths["summary"], summary)
    write_csv(paths["deltas"], deltas)
    outputs.extend(paths.values())
    complete("evaluate", outputs)


def run_report() -> None:
    validation = pd.read_csv(ROOT / "metrics/validation_summary_results.csv")
    test = pd.read_csv(ROOT / "metrics/test_summary_results.csv")
    deltas = pd.read_csv(ROOT / "metrics/test_paired_deltas.csv")
    timing = json.loads((ROOT / "metrics/pose_inference_timing.json").read_text())
    comparison = pd.read_csv(ROOT / "metrics/protocol_comparison_summary.csv")
    comparison = comparison.loc[
        :, ["protocol", "source", "model", "f1", "pr_auc", "roc_auc", "accuracy", "fp_ge_0p9_count", "fn_le_0p1_count"]
    ]
    primary_deltas = pd.read_csv(ROOT / "metrics/primary_deltas.csv")
    population = pd.read_csv(ROOT / "metrics/test_population_by_track_type.csv")
    overlap_parity = json.loads((ROOT / "metrics/zero_shot_overlap_prediction_parity.json").read_text())
    feature_parity = json.loads((ROOT / "metrics/exact_overlap_feature_parity.json").read_text())
    one_vs_bbox = primary_deltas.loc[primary_deltas.comparison.eq("one_stage_minus_bbox53")].iloc[0]
    two_vs_one = primary_deltas.loc[primary_deltas.comparison.eq("two_stage_minus_one_stage")].iloc[0]
    report = ROOT / "FINAL_REPORT.md"
    lines = [
        "# Exp079 Final Report", "",
        "Diagnostic reconstruction of the flawed JAAD API-style protocol: 16 observed frames, nominal TTE 30--60, fixed threshold 0.5.",
        "The one/two-stage model recipe is unchanged from Exp077; only the JAAD cohort/protocol differs.", "",
        "## Native Validation", "", markdown_table(validation), "",
        "## Flawed-Protocol Test", "", markdown_table(test), "",
        "## Two-Stage Minus One-Stage Deltas", "", markdown_table(deltas), "",
        "## Protocol References", "", markdown_table(comparison), "",
        "The clean and flawed rows are not directly comparable benchmarks because the populations and label/temporal semantics differ.", "",
        "## Primary Flawed-Protocol Deltas", "", markdown_table(primary_deltas), "",
        "## Population Diagnostic", "", markdown_table(population), "",
        "## Pose Runtime", "",
        f"- CUDA device: `{timing['device']}`.",
        f"- Median synchronized pose inference per unique image: `{timing['inference_ms']['median']:.3f} ms`.",
        f"- Sequential 16-frame estimate from the median: `{timing['estimated_sequential_16_frame_inference_ms_from_median']:.3f} ms`.", "",
        "## Integrity Findings", "",
        f"- Exact-overlap feature parity: `{sum(row['overlap_rows'] for row in feature_parity)}` windows, maximum absolute feature difference `{max(row['max_abs_difference'] for row in feature_parity)}`.",
        f"- Exact-overlap zero-shot prediction parity: `{overlap_parity['overlap_windows']}` test windows across three seeds, maximum probability difference `{max(overlap_parity['one_stage_probability_max_abs_difference'], overlap_parity['two_stage_probability_max_abs_difference'])}`.",
        "- All 4,851 non-behavior test windows are negative, whereas the 1,881 behavior-track windows contain all 1,177 positives.", "",
        "## Conclusions", "",
        f"- Against Exp031 BBox53 at the same flawed protocol and fixed 0.5, one-stage gains F1 `{one_vs_bbox['delta_f1']:.4f}`, PR-AUC `{one_vs_bbox['delta_pr_auc']:.4f}`, and ROC-AUC `{one_vs_bbox['delta_roc_auc']:.4f}`.",
        f"- Two-stage minus one-stage changes F1 by `{two_vs_one['delta_f1']:.4f}` and PR-AUC by `{two_vs_one['delta_pr_auc']:.4f}`; the sequential design is not supported as the better model here.",
        "- PIE zero-shot failure is driven by the added non-behavior negative population, not an implementation mismatch: exact-overlap features and probabilities are identical to Exp077.", "",
        "This is a protocol-sensitivity diagnostic, not the clean JAAD benchmark. The flawed protocol changes both population and temporal/label semantics.",
    ]
    _atomic(report, ("\n".join(lines) + "\n").encode())
    complete("report", [report])
    _atomic(ROOT / "EXPERIMENT_COMPLETE", b"complete\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "validate", "evaluate", "report", "all"))
    args = parser.parse_args()
    phases = ("prepare", "validate", "evaluate", "report") if args.phase == "all" else (args.phase,)
    functions = {"prepare": run_prepare, "validate": run_validate, "evaluate": run_evaluate, "report": run_report}
    for phase in phases:
        print(f"[Exp079] {phase}", flush=True)
        functions[phase]()


if __name__ == "__main__":
    main()
