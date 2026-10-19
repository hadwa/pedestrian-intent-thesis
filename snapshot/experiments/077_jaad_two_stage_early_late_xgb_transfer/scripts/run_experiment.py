"""Frozen zero-shot and JAAD-native runner for Exp077."""

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
EXP076_SCRIPTS = EXP076 / "scripts"
if str(EXP076_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(EXP076_SCRIPTS))

import data_core  # noqa: E402
import evaluation  # noqa: E402
import two_stage_core  # noqa: E402


ARCHIVE_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation")
SPLIT_DIR = ARCHIVE_ROOT / "data/jaad_splits_event_16_30_60_overlap08_default"
FEATURE_DIR = ARCHIVE_ROOT / "data/jaad_features_event_16_30_60_overlap08_default"
DEFINITIONS = ACTIVE_ROOT / "experiments/012_pie_biu_v2_30_60_benchmark/metrics/pie_biu_v2_feature_set_definitions.csv"
JAAD_KEYPOINTS = ACTIVE_ROOT / "data/jaad_keypoints_vitpose_s_extended_short_horizon.jsonl"
JAAD_SUPPLEMENT = ACTIVE_ROOT / "experiments/030_final_confidence_benchmark/jaad30_repair/jaad30_missing_supplemental_keypoints.jsonl"
REPAIR_SEQUENCES = ACTIVE_ROOT / "experiments/030_final_confidence_benchmark/jaad30_repair/jaad30_missing_sequences.csv"
ORIGINAL_JAAD_KEYPOINTS = ARCHIVE_ROOT / "data/jaad_keypoints_vitpose_s.jsonl"
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


def load_jaad_split(split: str) -> pd.DataFrame:
    if split not in {"train", "val", "test"}:
        raise ValueError("JAAD split must be train, val, or test")
    frame = pd.read_csv(SPLIT_DIR / f"{split}.csv")
    frame["set_id"] = "jaad"
    frame["split"] = "validation" if split == "val" else split
    frame = frame.sort_values("sequence_id", kind="stable").reset_index(drop=True)
    data_core.validate_window_cohort(frame, tte_column="tte_frames")
    expected = {"train": 1552, "val": 274, "test": 1501}[split]
    if len(frame) != expected:
        raise RuntimeError(f"JAAD {split} row count changed")
    return frame


def _archived_feature(split: str, family: str, columns: tuple[str, ...]) -> pd.DataFrame:
    source_split = "val" if split == "validation" else split
    frame = pd.read_csv(FEATURE_DIR / f"{source_split}_{family}.csv", usecols=["sequence_id", *columns])
    if frame["sequence_id"].duplicated().any():
        raise RuntimeError(f"archived JAAD {family} identity changed")
    return frame


def assert_expected_mismatch_ids(actual: set[str], expected: set[str], *, family: str) -> None:
    if set(actual) != set(expected):
        raise RuntimeError(
            f"unexpected {family} mismatch population: missing={sorted(expected - actual)[:3]}, "
            f"extra={sorted(actual - expected)[:3]}"
        )


def _parity_audit(source: pd.DataFrame, views: pd.DataFrame, split: str, contract: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    audits = {}
    for family, columns in (("bbox", contract["bbox_only"]), ("biu", contract["biu_v1_only"])):
        archived = _archived_feature(split, family, columns)
        merged = source[["sequence_id"]].merge(archived, on="sequence_id", how="left", validate="one_to_one").merge(
            views[["sequence_id", *[f"full__{name}" for name in columns]]],
            on="sequence_id",
            how="left",
            validate="one_to_one",
        )
        differences = []
        row_mismatch = np.zeros(len(merged), dtype=bool)
        nan_mismatches = 0
        for name in columns:
            left = merged[name].to_numpy(float)
            right = merged[f"full__{name}"].to_numpy(float)
            nan_mismatches += int(np.logical_xor(np.isnan(left), np.isnan(right)).sum())
            row_mismatch |= np.logical_xor(np.isnan(left), np.isnan(right))
            finite = np.isfinite(left) & np.isfinite(right)
            if finite.any():
                difference = np.abs(left - right)
                row_mismatch |= np.isfinite(difference) & (difference > 1e-9)
                differences.append(difference[finite])
        maximum = float(np.concatenate(differences).max()) if differences else 0.0
        mismatch_ids = set(merged.loc[row_mismatch, "sequence_id"].astype(str))
        if family == "bbox":
            assert_expected_mismatch_ids(mismatch_ids, set(), family=family)
            status = "exact"
        else:
            repair = pd.read_csv(REPAIR_SEQUENCES)
            repair_split = "val" if split == "validation" else split
            expected = set(repair.loc[repair["split"].astype(str).eq(repair_split), "sequence_id"].astype(str))
            assert_expected_mismatch_ids(mismatch_ids, expected, family=family)
            status = "expected_exact_pose_vs_historical_nearest_frame_repair_difference"
        audits[family] = {
            "status": status,
            "max_abs_difference": maximum,
            "nan_mismatches": nan_mismatches,
            "mismatch_rows": len(mismatch_ids),
            "rows": len(merged),
        }
    return audits


def materialize_jaad(split: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_split = "val" if split == "validation" else split
    source = load_jaad_split(source_split)
    contract = feature_contract()
    needed = data_core.needed_keys(source)
    keypoints, keypoint_audit = data_core.load_keypoint_index((JAAD_KEYPOINTS, JAAD_SUPPLEMENT), needed)
    bboxes = data_core.load_jaad_bboxes(source)
    views, view_audit = data_core.materialize_views(source, keypoints, bboxes, contract=contract)
    parity = _parity_audit(source, views, split, contract)
    meta = [
        "sequence_id", "split", "set_id", "video_id", "ped_id", "track_id", "start_frame", "end_frame",
        "label", "observation_length", "frame_step", "frame_sampling", "tte_frames",
    ]
    early = [f"early__{name}" for name in contract["joint"]]
    portable_prefixed = [f"portable__{name}" for name in contract["portable"]]
    prepared = views.loc[:, [*meta, *early, *portable_prefixed]].copy()
    prepared = prepared.rename(columns={f"portable__{name}": name for name in contract["portable"]})
    data_core.two_stage_core.assert_cohort_parity(source, prepared)
    return prepared, {"keypoints": keypoint_audit, "views": view_audit, "archived_parity": parity}


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
            "definitions": artifact(DEFINITIONS),
            "jaad_keypoints_primary": artifact(JAAD_KEYPOINTS),
            "jaad_keypoints_supplement": artifact(JAAD_SUPPLEMENT),
            "historical_jaad_keypoints_used_by_archived_biu_table": artifact(ORIGINAL_JAAD_KEYPOINTS),
            "documented_repair_sequences": artifact(REPAIR_SEQUENCES),
            "jaad_train_split": artifact(SPLIT_DIR / "train.csv"),
            "jaad_validation_split": artifact(SPLIT_DIR / "val.csv"),
            "pie_prepared_train": artifact(EXP076 / "metrics/prepared_train.csv"),
            "shared_core": artifact(EXP076_SCRIPTS / "two_stage_core.py"),
        },
    )
    outputs.extend([audit_path, provenance_path])
    complete("prepare", outputs)


def _matrices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    contract = feature_contract()
    early = frame.loc[:, [f"early__{name}" for name in contract["joint"]]].to_numpy(float)
    full = frame.loc[:, contract["portable"]].to_numpy(float)
    return early, full


def _fit_setting(name: str, train: pd.DataFrame) -> tuple[dict[int, Any], dict[int, Any], dict[str, Any], list[Path]]:
    early, full = _matrices(train)
    labels = train["label"].to_numpy(int)
    groups = train["track_id"].astype(str).to_numpy()
    one_models = {}
    two_models = {}
    audits = {}
    outputs = []
    for seed in SEEDS:
        one = two_stage_core.build_xgb(seed, labels, n_estimators=250)
        one.fit(full, labels)
        two, audit = two_stage_core.fit_two_stage(
            early, full, labels, groups, seed=seed, total_estimators=250, n_splits=5
        )
        one_path = ROOT / "models" / f"{name}_one_stage_seed{seed}.joblib"
        two_path = ROOT / "models" / f"{name}_two_stage_seed{seed}.joblib"
        one_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(one, one_path)
        joblib.dump(two, two_path)
        one_models[seed] = one
        two_models[seed] = two
        audits[str(seed)] = audit
        outputs.extend([one_path, two_path])
    return one_models, two_models, audits, outputs


def _predict_pair(
    frame: pd.DataFrame,
    one_models: dict[int, Any],
    two_models: dict[int, Any],
) -> pd.DataFrame:
    early, full = _matrices(frame)
    parts = []
    for seed in SEEDS:
        part = frame.loc[:, ["sequence_id", "set_id", "video_id", "track_id", "label"]].copy()
        part.insert(0, "seed", seed)
        part["one_stage_probability"] = one_models[seed].predict_proba(full)[:, 1]
        part["two_stage_probability"] = two_stage_core.predict_two_stage(two_models[seed], early, full)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def run_validate() -> None:
    pie_train = pd.read_csv(EXP076 / "metrics/prepared_train.csv")
    jaad_train = pd.read_csv(ROOT / "metrics/prepared_jaad_train.csv")
    jaad_validation = pd.read_csv(ROOT / "metrics/prepared_jaad_validation.csv")
    zero_one, zero_two, zero_audit, zero_outputs = _fit_setting("pie_zero_shot", pie_train)
    native_one, native_two, native_audit, native_outputs = _fit_setting("jaad_native", jaad_train)
    predictions = _predict_pair(jaad_validation, native_one, native_two)
    per_seed, deltas = evaluation.paired_results(predictions, scope="jaad_native_validation")
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
    write_json(paths["oof"], {"pie_zero_shot": zero_audit, "jaad_native": native_audit})
    complete("validate", [*zero_outputs, *native_outputs, *paths.values()])


def _load_models(name: str) -> tuple[dict[int, Any], dict[int, Any]]:
    one = {seed: joblib.load(ROOT / "models" / f"{name}_one_stage_seed{seed}.joblib") for seed in SEEDS}
    two = {seed: joblib.load(ROOT / "models" / f"{name}_two_stage_seed{seed}.joblib") for seed in SEEDS}
    return one, two


def run_evaluate() -> None:
    test, audit = materialize_jaad("test")
    test_path = ROOT / "metrics/prepared_jaad_test.csv"
    audit_path = ROOT / "metrics/test_preparation_audit.json"
    write_csv(test_path, test)
    write_json(audit_path, audit)
    metric_parts = []
    delta_parts = []
    outputs = [test_path, audit_path]
    for setting, scope in (("pie_zero_shot", "pie_to_jaad_zero_shot_test"), ("jaad_native", "jaad_native_test")):
        one, two = _load_models(setting)
        predictions = _predict_pair(test, one, two)
        prediction_path = ROOT / "predictions" / f"{setting}_test_all_seeds.csv"
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
    report = ROOT / "FINAL_REPORT.md"
    lines = [
        "# Exp077 Final Report", "",
        "JAAD behavior-pedestrian, 16 observed frames, TTE 30--60, fixed threshold 0.5.",
        "Both architectures use the same no-ego portable inputs and matched 250-tree budget.", "",
        "## Native Validation", "", markdown_table(validation), "",
        "## Final JAAD Test", "", markdown_table(test), "",
        "## Two-Stage Minus One-Stage Deltas", "", markdown_table(deltas), "",
        "Zero-shot and native results are separate. JAAD test has prior exposure in separate work;",
        "all Exp077 decisions were frozen before this evaluation.",
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
        print(f"[Exp077] {phase}", flush=True)
        functions[phase]()


if __name__ == "__main__":
    main()
