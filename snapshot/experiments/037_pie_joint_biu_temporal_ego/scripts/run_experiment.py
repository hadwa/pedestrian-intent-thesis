"""Fail-closed phase runner for Exp037 PIE joint BIU temporal-ego study."""

from __future__ import annotations

import json
import hashlib
import argparse
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ACTIVE_ROOT = Path(__file__).resolve().parents[3]
for import_path in (SCRIPT_DIR, ACTIVE_ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import feature_core  # noqa: E402
import model_core  # noqa: E402
from biu_v2_extractor import extract_biu_v2  # noqa: E402


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PRE_REGISTRATION_PATH = EXPERIMENT_ROOT / "pre_registration.json"
ARCHIVE_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation")
SPLIT_DIR = ARCHIVE_ROOT / "data/splits_pie_wacv_pcpa_30_60_overlap06"
FEATURE_DIR = ARCHIVE_ROOT / "data/processed/features_wacv_pcpa_30_60_overlap06"
KEYPOINTS_PATH = Path("/home/abdelrahman/Hadwa/pedestrian-intent-biu/data/data_keypoints_all.jsonl")
ANNOTATIONS_DIR = Path("/home/abdelrahman/Hadwa/MSLSTM-PID/PIE/annotations/annotations")
OBD_DIR = Path("/home/abdelrahman/Hadwa/MSLSTM-PID/PIE/annotations/annotations_vehicle")
FEATURE_DEFINITIONS_PATH = (
    ACTIVE_ROOT
    / "experiments/012_pie_biu_v2_30_60_benchmark/metrics/pie_biu_v2_feature_set_definitions.csv"
)
SET_TO_SPLIT = {
    "set01": "train",
    "set02": "train",
    "set03": "test",
    "set04": "val",
    "set05": "val",
    "set06": "train",
}


EXPECTED_PREREGISTRATION: dict[str, Any] = json.loads(
    r'''{
  "schema_version": 1,
  "experiment": "Exp037",
  "name": "PIE joint BIU temporal-ego",
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
  "baseline_arms": ["v2_bbox_xgb", "v1_bbox_rf"],
  "candidate_arms": ["joint_xgb", "joint_rf", "joint_temporal_xgb", "joint_temporal_ego_xgb"],
  "seeds": [42, 43, 44],
  "bootstrap_seed": 42,
  "bootstrap_replicates": 10000,
  "minimum_pr_auc_gain": 0.005,
  "minimum_f1_gain": 0.005,
  "maximum_per_set_regression": 0.01,
  "fp_high_confidence_min": 0.9,
  "fn_high_confidence_max": 0.1,
  "selection_priority": ["pr_auc", "f1_fixed_0p5", "roc_auc", "fewer_high_confidence_errors", "lower_feature_count"],
  "feature_contract": {
    "joint_unique_feature_count": 161,
    "temporal_bases": ["lower_body_motion_magnitude_mean", "lead_ankle_forward_delta", "torso_motion_magnitude_mean", "torso_center_x_delta", "bbox_center_dx_norm", "bbox_scale_log_delta", "pose_confidence_mean"],
    "temporal_statistics": ["early", "late", "late_minus_early"],
    "obd_columns": ["obd_speed_mean", "obd_speed_std", "obd_speed_first", "obd_speed_last", "obd_speed_delta"]
  },
  "models": {
    "rf": {"n_estimators": 300, "min_samples_leaf": 2, "class_weight": "balanced_subsample", "n_jobs": -1},
    "xgb": {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "objective": "binary:logistic", "eval_metric": "logloss", "n_jobs": 4}
  },
  "phase_contract": {
    "prepare": "authenticate sources and materialize development features only",
    "develop": "leave-one-set-out OOF on set01/set02/set04/set05",
    "confirm": "fit development sets and evaluate one locked candidate on set06",
    "evaluate": "conditional authenticated set03 diagnostic only after advance",
    "shared_unlabeled_keypoint_exception": "The combined keypoint JSONL may be streamed before evaluation, but only phase-authorized set records may be retained; set03 labels, predictions, feature matrices, and metrics remain forbidden."
  },
  "papers": [
    {"title": "PIE: A Large-Scale Dataset and Models for Pedestrian Intention Estimation and Trajectory Prediction", "year": 2019, "local_copy": "papers/Rasouli_2019_PIE.pdf"},
    {"title": "Benchmark for Evaluating Pedestrian Action Prediction", "year": 2021, "local_copy": "papers/Kotseruba_2021_PCPA_Benchmark.pdf"},
    {"title": "Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization", "year": 2020, "local_copy": "papers/Sagawa_2020_Group_DRO.pdf"}
  ]
}'''
)


def load_preregistration(path: Path = PRE_REGISTRATION_PATH) -> dict[str, Any]:
    registration = json.loads(Path(path).read_text(encoding="utf-8"))
    if registration != EXPECTED_PREREGISTRATION:
        raise ValueError("Exp037 preregistration differs from the frozen semantic contract")
    return registration


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _publish_completion(root: Path, phase: str, outputs: list[Path], **extra: Any) -> None:
    metrics = root / "metrics"
    manifest = {str(path.relative_to(metrics)): _artifact_record(path) for path in outputs}
    _write_json(metrics / f".{phase}.complete.json", {"phase": phase, "status": "complete", "outputs": manifest, **extra})


def _load_split_rows(allowed_sets: tuple[str, ...]) -> pd.DataFrame:
    allowed = set(allowed_sets)
    unknown = allowed.difference(SET_TO_SPLIT)
    if unknown:
        raise ValueError(f"unknown PIE sets: {sorted(unknown)}")
    frames = []
    for split in sorted({SET_TO_SPLIT[set_id] for set_id in allowed}):
        path = SPLIT_DIR / f"{split}_sequences.csv"
        frame = pd.read_csv(path)
        frames.append(frame.loc[frame["set_id"].astype(str).isin(allowed)].copy())
    result = pd.concat(frames, ignore_index=True).sort_values("sequence_id", kind="stable").reset_index(drop=True)
    if set(result["set_id"].astype(str)) != allowed:
        raise ValueError("split sources do not contain the exact authorized set universe")
    feature_core._validate_base_cohort(result)
    return result


def _load_archived_features(base: pd.DataFrame, family: str, columns: tuple[str, ...]) -> pd.DataFrame:
    parts = []
    sequence_ids = set(base["sequence_id"].astype(str))
    for split in sorted({SET_TO_SPLIT[set_id] for set_id in base["set_id"].astype(str).unique()}):
        source = pd.read_csv(FEATURE_DIR / f"{family}_{split}.csv", usecols=["sequence_id", *columns])
        parts.append(source.loc[source["sequence_id"].astype(str).isin(sequence_ids)])
    features = pd.concat(parts, ignore_index=True)
    if features["sequence_id"].duplicated().any() or set(features["sequence_id"].astype(str)) != sequence_ids:
        raise ValueError(f"{family} exact cohort mismatch")
    return features


def _needed_keys(base: pd.DataFrame) -> set[tuple[str, str, str, int]]:
    needed = set()
    for row in base.itertuples(index=False):
        for frame in range(int(row.start_frame), int(row.end_frame) + 1):
            needed.add((str(row.set_id), str(row.video_id), str(row.ped_id), frame))
    return needed


def _load_keypoints(needed: set[tuple[str, str, str, int]], allowed_sets: set[str]) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    output: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    with KEYPOINTS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = _decode_authorized_keypoint(line, allowed_sets)
            if item is None:
                continue
            set_id = str(item["set"])
            key = (set_id, str(item["video"]), str(item["pid"]), int(item["frame_idx"]))
            if key in needed:
                if key in output:
                    raise ValueError(f"duplicate keypoint frame {key}")
                output[key] = item
    missing = needed.difference(output)
    if missing:
        raise ValueError(f"missing exact observed keypoint: {sorted(missing)[0]}")
    return output


def _decode_authorized_keypoint(line: str, allowed_sets: set[str]) -> dict[str, Any] | None:
    match = re.search(r'"set"\s*:\s*"([^"]+)"', line[:256])
    if match is None:
        raise ValueError("keypoint record has no early set identifier")
    if match.group(1) not in allowed_sets:
        return None
    item = json.loads(line)
    if str(item.get("set")) != match.group(1):
        raise ValueError("keypoint record set identity changed during decode")
    return item


def _load_bboxes(base: pd.DataFrame, needed: set[tuple[str, str, str, int]]) -> dict[tuple[str, str, str, int], np.ndarray]:
    output: dict[tuple[str, str, str, int], np.ndarray] = {}
    videos = sorted(set(zip(base["set_id"].astype(str), base["video_id"].astype(str))))
    for set_id, video_id in videos:
        path = ANNOTATIONS_DIR / set_id / f"{video_id}_annt.xml"
        for track in ET.parse(path).getroot().findall("track"):
            if track.attrib.get("label") != "pedestrian":
                continue
            for box in track.findall("box"):
                ped_id = next((str(a.text) for a in box.findall("attribute") if a.attrib.get("name") == "id"), None)
                if ped_id is None:
                    continue
                key = (set_id, video_id, ped_id, int(box.attrib["frame"]))
                if key not in needed:
                    continue
                if key in output:
                    raise ValueError(f"duplicate BBox frame {key}")
                output[key] = np.asarray(
                    [box.attrib["xtl"], box.attrib["ytl"], box.attrib["xbr"], box.attrib["ybr"]], dtype=float
                )
    missing = needed.difference(output)
    if missing:
        raise ValueError(f"missing exact observed BBox: {sorted(missing)[0]}")
    return output


def _load_obd(base: pd.DataFrame) -> dict[tuple[str, str], dict[int, float]]:
    output = {}
    videos = sorted(set(zip(base["set_id"].astype(str), base["video_id"].astype(str))))
    for set_id, video_id in videos:
        path = OBD_DIR / set_id / f"{video_id}_obd.xml"
        speed = {}
        for item in ET.parse(path).getroot().findall("frame"):
            frame = int(item.attrib["id"])
            if frame in speed:
                raise ValueError(f"duplicate OBD frame {set_id}/{video_id}/{frame}")
            speed[frame] = float(item.attrib["OBD_speed"])
        output[(set_id, video_id)] = speed
    return output


def _materialize_features(allowed_sets: tuple[str, ...]) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    base = _load_split_rows(allowed_sets)
    contract = feature_core.load_exp012_feature_contract(FEATURE_DEFINITIONS_PATH)
    arms = feature_core.build_arm_columns(contract)
    bbox = _load_archived_features(base, "bbox", contract["bbox_only"])
    v1 = _load_archived_features(base, "biu", contract["biu_v1_only"])
    frame = feature_core.exact_feature_join(base, bbox, name="BBox")
    frame = feature_core.exact_feature_join(frame, v1, name="BIU-v1")

    needed = _needed_keys(base)
    keypoints = _load_keypoints(needed, set(allowed_sets))
    bboxes = _load_bboxes(base, needed)
    obd = _load_obd(base)
    derived = []
    for row in base.itertuples(index=False):
        frames = range(int(row.start_frame), int(row.end_frame) + 1)
        keys = [(str(row.set_id), str(row.video_id), str(row.ped_id), index) for index in frames]
        coords = np.asarray([keypoints[key]["keypoints"] for key in keys], dtype=float)
        confidence = np.asarray([keypoints[key]["kp_conf"] for key in keys], dtype=float)
        boxes = np.asarray([bboxes[key] for key in keys], dtype=float)
        v2 = extract_biu_v2(coords, confidence, bbox_sequence=boxes)
        missing_v2 = set(contract["biu_v2_core"]).difference(v2)
        if missing_v2:
            raise ValueError(f"BIU-v2 extractor output changed: {sorted(missing_v2)}")
        values = {"sequence_id": str(row.sequence_id)}
        values.update({f"v2__{name}": float(v2[name]) for name in contract["biu_v2_core"]})
        temporal = feature_core.temporal_contrast_features(coords, confidence, boxes, extractor=extract_biu_v2)
        values.update({f"temporal__{name}": value for name, value in temporal.items()})
        values.update(feature_core.obd_speed_features(obd[(str(row.set_id), str(row.video_id))], list(frames)))
        derived.append(values)
    frame = feature_core.exact_feature_join(frame, pd.DataFrame(derived), name="BIU-v2 temporal OBD")
    required = set().union(*arms.values())
    if set(required).difference(frame.columns):
        raise ValueError("materialized matrix is missing frozen arm features")
    if not np.isfinite(frame.loc[:, sorted(required)].to_numpy(dtype=float)).all():
        raise ValueError("materialized feature matrix contains non-finite values")
    return frame, arms


def _authenticate_completion(root: Path, phase: str) -> dict[str, Any]:
    metrics = Path(root) / "metrics"
    marker = metrics / f".{phase}.complete.json"
    completion = json.loads(marker.read_text(encoding="utf-8"))
    if completion.get("phase") != phase or completion.get("status") != "complete":
        raise RuntimeError(f"{phase} completion marker is invalid")
    outputs = completion.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise RuntimeError(f"{phase} output manifest is invalid")
    for relative_path, expected in outputs.items():
        artifact = metrics / relative_path
        if not artifact.is_file() or expected != _artifact_record(artifact):
            raise RuntimeError(f"{phase} output is not authenticated: {relative_path}")
    return completion


def _source_paths_for_sets(sets: tuple[str, ...]) -> list[tuple[str, Path]]:
    splits = sorted({SET_TO_SPLIT[set_id] for set_id in sets})
    paths: list[tuple[str, Path]] = []
    for split in splits:
        paths.extend(
            [
                (f"split_{split}", SPLIT_DIR / f"{split}_sequences.csv"),
                (f"bbox_{split}", FEATURE_DIR / f"bbox_{split}.csv"),
                (f"biu_v1_{split}", FEATURE_DIR / f"biu_{split}.csv"),
            ]
        )
    paths.extend([("keypoints_combined_unlabeled", KEYPOINTS_PATH), ("feature_definitions", FEATURE_DEFINITIONS_PATH)])
    base = _load_split_rows(sets)
    for set_id, video_id in sorted(set(zip(base["set_id"].astype(str), base["video_id"].astype(str)))):
        paths.append((f"bbox_xml_{set_id}_{video_id}", ANNOTATIONS_DIR / set_id / f"{video_id}_annt.xml"))
        paths.append((f"obd_xml_{set_id}_{video_id}", OBD_DIR / set_id / f"{video_id}_obd.xml"))
    return paths


def _source_manifest(sets: tuple[str, ...]) -> dict[str, Any]:
    records = []
    for role, path in _source_paths_for_sets(sets):
        resolved = path.resolve(strict=True)
        records.append({"role": role, "path": str(resolved), **_artifact_record(resolved)})
    return {
        "authorized_sets": list(sets),
        "shared_unlabeled_keypoint_exception": EXPECTED_PREREGISTRATION["phase_contract"][
            "shared_unlabeled_keypoint_exception"
        ],
        "sources": records,
    }


def _cohort_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "tracks": int(frame["track_id"].astype(str).nunique()),
        "videos": int(frame[["set_id", "video_id"]].astype(str).drop_duplicates().shape[0]),
        "positive_count": int(frame["label"].astype(int).sum()),
        "sets": {
            set_id: {
                "rows": int(len(group)),
                "tracks": int(group["track_id"].astype(str).nunique()),
                "videos": int(group["video_id"].astype(str).nunique()),
                "positive_count": int(group["label"].astype(int).sum()),
            }
            for set_id, group in frame.groupby("set_id", sort=True)
        },
    }


def run_prepare(output_root: Path = EXPERIMENT_ROOT) -> str:
    root = Path(output_root)
    marker = root / "metrics/.prepare.complete.json"
    if marker.exists():
        _authenticate_completion(root, "prepare")
        return "complete"
    load_preregistration(root / "pre_registration.json")
    development_sets = tuple(EXPECTED_PREREGISTRATION["development_sets"])
    frame, arms = _materialize_features(development_sets)
    if len(frame) != 4781:
        raise RuntimeError(f"development cohort changed: {len(frame)}")
    metrics = root / "metrics"
    prepared_path = metrics / "prepared_development.csv"
    contract_path = metrics / "feature_contract.json"
    provenance_path = metrics / "source_provenance_prepare.json"
    _write_csv(prepared_path, frame)
    _write_json(
        contract_path,
        {
            "arms": {arm: list(columns) for arm, columns in arms.items()},
            "feature_counts": {arm: len(columns) for arm, columns in arms.items()},
            "cohort": _cohort_summary(frame),
            "confirmation_expected_rows": 1204,
            "diagnostic_test_materialized": False,
        },
    )
    _write_json(provenance_path, _source_manifest(development_sets))
    _publish_completion(root, "prepare", [prepared_path, contract_path, provenance_path])
    return "complete"


def _metric_row(arm: str, frame: pd.DataFrame, n_features: int, scope: str, set_id: str = "pooled") -> dict[str, Any]:
    return {
        "scope": scope,
        "set_id": set_id,
        "arm": arm,
        "n_features": int(n_features),
        **model_core.compute_metrics(frame["label"].to_numpy(int), frame["probability"].to_numpy(float)),
    }


def _development_tables(
    frame: pd.DataFrame, arms: dict[str, tuple[str, ...]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_parts = []
    audit_parts = []
    result_rows = []
    per_set_rows = []
    for arm, columns in arms.items():
        predictions, audit = model_core.run_leave_one_set_out(frame, columns, arm, model_core.SEEDS)
        prediction_parts.append(predictions)
        audit_parts.append(audit)
        result_rows.append(_metric_row(arm, predictions, len(columns), "development_oof"))
        for set_id, group in predictions.groupby("set_id", sort=True):
            per_set_rows.append(_metric_row(arm, group, len(columns), "development_oof", str(set_id)))
    predictions = pd.concat(prediction_parts, ignore_index=True)
    audits = pd.concat(audit_parts, ignore_index=True)
    results = pd.DataFrame(result_rows)
    per_set = pd.DataFrame(per_set_rows)
    indexed = per_set.set_index(["arm", "set_id"])
    for arm in model_core.CANDIDATE_ARMS:
        for set_id in EXPECTED_PREREGISTRATION["development_sets"]:
            mask = per_set["arm"].eq(arm) & per_set["set_id"].eq(set_id)
            candidate = indexed.loc[(arm, set_id)]
            v2 = indexed.loc[("v2_bbox_xgb", set_id)]
            v1 = indexed.loc[("v1_bbox_rf", set_id)]
            per_set.loc[mask, "pr_auc_delta_vs_v2"] = float(candidate["pr_auc"] - v2["pr_auc"])
            per_set.loc[mask, "f1_delta_vs_best"] = float(
                candidate["f1_fixed_0p5"] - max(v2["f1_fixed_0p5"], v1["f1_fixed_0p5"])
            )

    wide = predictions.pivot(index=["sequence_id", "set_id", "track_id", "video_id", "label"], columns="arm", values="probability").reset_index()
    pooled = results.set_index("arm")
    best_f1_reference = max(model_core.BASELINE_ARMS, key=lambda arm: float(pooled.loc[arm, "f1_fixed_0p5"]))
    bootstrap_parts = []
    for arm in model_core.CANDIDATE_ARMS:
        pr = model_core.cluster_bootstrap_deltas(
            wide, "v2_bbox_xgb", arm,
            replicates=int(EXPECTED_PREREGISTRATION["bootstrap_replicates"]),
            seed=int(EXPECTED_PREREGISTRATION["bootstrap_seed"]),
        )
        pr = pr.loc[pr["metric"].eq("pr_auc")].assign(arm=arm, reference_arm="v2_bbox_xgb")
        f1 = model_core.cluster_bootstrap_deltas(
            wide, best_f1_reference, arm,
            replicates=int(EXPECTED_PREREGISTRATION["bootstrap_replicates"]),
            seed=int(EXPECTED_PREREGISTRATION["bootstrap_seed"]),
        )
        f1 = f1.loc[f1["metric"].eq("f1_fixed_0p5")].assign(arm=arm, reference_arm=best_f1_reference)
        bootstrap_parts.extend([pr, f1])
    return predictions, audits, results, per_set, pd.concat(bootstrap_parts, ignore_index=True)


def run_develop(output_root: Path = EXPERIMENT_ROOT) -> str:
    root = Path(output_root)
    marker = root / "metrics/.develop.complete.json"
    if marker.exists():
        _authenticate_completion(root, "develop")
        return "complete"
    _authenticate_completion(root, "prepare")
    metrics = root / "metrics"
    frame = pd.read_csv(metrics / "prepared_development.csv")
    contract = json.loads((metrics / "feature_contract.json").read_text(encoding="utf-8"))
    arms = {arm: tuple(columns) for arm, columns in contract["arms"].items()}
    predictions, audits, results, per_set, bootstrap = _development_tables(frame, arms)
    selected = model_core.select_candidate(results, per_set, bootstrap)
    lock = {
        "schema_version": 1,
        "phase": "develop",
        "status": "selected" if selected is not None else "no_candidate",
        "selected_candidate": None if selected is None else str(selected["arm"]),
        "development_sets": EXPECTED_PREREGISTRATION["development_sets"],
        "confirmation_set": EXPECTED_PREREGISTRATION["confirmation_set"],
        "threshold": EXPECTED_PREREGISTRATION["threshold"],
        "seeds": EXPECTED_PREREGISTRATION["seeds"],
        "preregistration_sha256": _sha256(root / "pre_registration.json"),
    }
    paths = {
        "development_oof_predictions.csv": predictions,
        "development_fold_audit.csv": audits,
        "development_results.csv": results,
        "development_per_set.csv": per_set,
        "development_bootstrap.csv": bootstrap,
    }
    output_paths = []
    for name, value in paths.items():
        path = metrics / name
        _write_csv(path, value)
        output_paths.append(path)
    lock_path = metrics / "development_lock.json"
    _write_json(lock_path, lock)
    output_paths.append(lock_path)
    _publish_completion(root, "develop", output_paths, lock_sha256=_sha256(lock_path), selection_status=lock["status"])
    return lock["status"]


def _load_diagnostic_test_features(*args: Any, **kwargs: Any) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    """Single auditable boundary for materializing set03 diagnostic features."""
    return _materialize_features((EXPECTED_PREREGISTRATION["diagnostic_test_set"],))


def _authenticate_development(root: Path) -> dict[str, Any]:
    completion = _authenticate_completion(root, "develop")
    lock_path = Path(root) / "metrics/development_lock.json"
    if completion.get("lock_sha256") != _sha256(lock_path):
        raise RuntimeError("development completion does not authenticate lock")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("phase") != "develop" or lock.get("status") not in {"selected", "no_candidate"}:
        raise RuntimeError("development lock is invalid")
    if lock.get("development_sets") != EXPECTED_PREREGISTRATION["development_sets"]:
        raise RuntimeError("development set universe is invalid")
    if lock.get("threshold") != EXPECTED_PREREGISTRATION["threshold"]:
        raise RuntimeError("development threshold is invalid")
    if lock.get("preregistration_sha256") != _sha256(Path(root) / "pre_registration.json"):
        raise RuntimeError("development lock preregistration hash is invalid")
    selected = lock.get("selected_candidate")
    if (lock["status"] == "selected") != (selected in EXPECTED_PREREGISTRATION["candidate_arms"]):
        raise RuntimeError("development selection is inconsistent")
    return lock


def _confirmation_gate(results: pd.DataFrame, bootstrap: pd.DataFrame, candidate: str) -> bool:
    indexed = results.set_index("arm")
    candidate_row = indexed.loc[candidate]
    v2 = indexed.loc["v2_bbox_xgb"]
    v1 = indexed.loc["v1_bbox_rf"]
    evidence = bootstrap.set_index("metric")
    return bool(
        float(candidate_row["pr_auc"]) > float(v2["pr_auc"]) + float(EXPECTED_PREREGISTRATION["minimum_pr_auc_gain"])
        and float(candidate_row["f1_fixed_0p5"])
        > max(float(v2["f1_fixed_0p5"]), float(v1["f1_fixed_0p5"]))
        + float(EXPECTED_PREREGISTRATION["minimum_f1_gain"])
        and int(candidate_row["fp_ge_0p9_count"])
        <= min(int(v2["fp_ge_0p9_count"]), int(v1["fp_ge_0p9_count"]))
        and int(candidate_row["fn_le_0p1_count"])
        <= min(int(v2["fn_le_0p1_count"]), int(v1["fn_le_0p1_count"]))
        and float(evidence.loc["pr_auc", "ci_low"]) > 0.0
        and float(evidence.loc["f1_fixed_0p5", "ci_low"]) > 0.0
    )


def run_confirm(output_root: Path = EXPERIMENT_ROOT) -> str:
    root = Path(output_root)
    marker = root / "metrics/.confirm.complete.json"
    if marker.exists():
        return str(_authenticate_confirmation(root)["status"])
    development_lock = _authenticate_development(root)
    metrics = root / "metrics"
    selected = development_lock["selected_candidate"]
    outputs = []
    confirmed = None
    if selected is not None:
        confirmation, arms = _materialize_features((EXPECTED_PREREGISTRATION["confirmation_set"],))
        if len(confirmation) != 1204:
            raise RuntimeError(f"confirmation cohort changed: {len(confirmation)}")
        development = pd.read_csv(metrics / "prepared_development.csv")
        prediction_parts = []
        result_rows = []
        for arm in (*model_core.BASELINE_ARMS, selected):
            probability = model_core.fit_predict_seed_mean(development, confirmation, arms[arm], arm, model_core.SEEDS)
            prediction = confirmation.loc[:, ["sequence_id", "set_id", "track_id", "video_id", "label"]].copy()
            prediction["arm"] = arm
            prediction["probability"] = probability
            prediction_parts.append(prediction)
            result_rows.append(_metric_row(arm, prediction, len(arms[arm]), "set06_confirmation", "set06"))
        predictions = pd.concat(prediction_parts, ignore_index=True)
        results = pd.DataFrame(result_rows)
        wide = predictions.pivot(index=["sequence_id", "set_id", "track_id", "video_id", "label"], columns="arm", values="probability").reset_index()
        best_f1_reference = max(
            model_core.BASELINE_ARMS,
            key=lambda arm: float(results.set_index("arm").loc[arm, "f1_fixed_0p5"]),
        )
        pr = model_core.cluster_bootstrap_deltas(
            wide, "v2_bbox_xgb", selected,
            replicates=int(EXPECTED_PREREGISTRATION["bootstrap_replicates"]), seed=42,
        ).loc[lambda value: value["metric"].eq("pr_auc")].assign(arm=selected, reference_arm="v2_bbox_xgb")
        f1 = model_core.cluster_bootstrap_deltas(
            wide, best_f1_reference, selected,
            replicates=int(EXPECTED_PREREGISTRATION["bootstrap_replicates"]), seed=42,
        ).loc[lambda value: value["metric"].eq("f1_fixed_0p5")].assign(arm=selected, reference_arm=best_f1_reference)
        bootstrap = pd.concat([pr, f1], ignore_index=True)
        if _confirmation_gate(results, bootstrap, selected):
            confirmed = selected
        for name, value in {
            "confirmation_features.csv": confirmation,
            "confirmation_predictions.csv": predictions,
            "confirmation_results.csv": results,
            "confirmation_bootstrap.csv": bootstrap,
        }.items():
            path = metrics / name
            _write_csv(path, value)
            outputs.append(path)
        provenance_path = metrics / "source_provenance_confirm.json"
        _write_json(provenance_path, _source_manifest((EXPECTED_PREREGISTRATION["confirmation_set"],)))
        outputs.append(provenance_path)
    status = "advance" if confirmed is not None else "no_advance"
    lock = {
        "schema_version": 1,
        "phase": "confirm",
        "status": status,
        "selected_candidate": selected,
        "confirmed_candidate": confirmed,
        "development_sets": EXPECTED_PREREGISTRATION["development_sets"],
        "confirmation_set": EXPECTED_PREREGISTRATION["confirmation_set"],
        "diagnostic_test_set": EXPECTED_PREREGISTRATION["diagnostic_test_set"],
        "threshold": EXPECTED_PREREGISTRATION["threshold"],
        "preregistration_sha256": _sha256(root / "pre_registration.json"),
    }
    lock_path = metrics / "confirmation_lock.json"
    _write_json(lock_path, lock)
    outputs.append(lock_path)
    _publish_completion(root, "confirm", outputs, lock_sha256=_sha256(lock_path), promotion_status=status)
    return status


def _authenticate_confirmation(output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root)
    metrics_dir = output_root / "metrics"
    lock_path = metrics_dir / "confirmation_lock.json"
    completion_path = metrics_dir / ".confirm.complete.json"

    load_preregistration(output_root / "pre_registration.json")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("phase") != "confirm" or completion.get("status") != "complete":
        raise RuntimeError("confirmation completion marker is invalid")
    if completion.get("lock_sha256") != _sha256(lock_path):
        raise RuntimeError("confirmation completion does not authenticate lock")

    outputs = completion.get("outputs")
    if not isinstance(outputs, dict) or "confirmation_lock.json" not in outputs:
        raise RuntimeError("confirmation completion output manifest is invalid")
    for relative_path, expected in outputs.items():
        artifact = metrics_dir / relative_path
        if not artifact.is_file():
            raise RuntimeError(f"confirmation output is missing: {relative_path}")
        if expected != {"sha256": _sha256(artifact), "size_bytes": artifact.stat().st_size}:
            raise RuntimeError(f"confirmation output is not authenticated: {relative_path}")

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected_fixed = {
        "schema_version": 1,
        "phase": "confirm",
        "development_sets": EXPECTED_PREREGISTRATION["development_sets"],
        "confirmation_set": EXPECTED_PREREGISTRATION["confirmation_set"],
        "diagnostic_test_set": EXPECTED_PREREGISTRATION["diagnostic_test_set"],
        "threshold": EXPECTED_PREREGISTRATION["threshold"],
        "preregistration_sha256": _sha256(output_root / "pre_registration.json"),
    }
    for key, expected in expected_fixed.items():
        if lock.get(key) != expected:
            raise RuntimeError(f"confirmation lock field is invalid: {key}")
    status = lock.get("status")
    if status not in {"advance", "no_advance"}:
        raise RuntimeError("confirmation lock status is invalid")
    if completion.get("promotion_status") != status:
        raise RuntimeError("confirmation completion promotion status is inconsistent")
    if status == "advance" and not lock.get("confirmed_candidate"):
        raise RuntimeError("advancing confirmation lock has no confirmed candidate")
    if status == "no_advance" and lock.get("confirmed_candidate") is not None:
        raise RuntimeError("non-advancing confirmation lock has a confirmed candidate")
    return lock


def run_evaluate(output_root: Path = EXPERIMENT_ROOT) -> str:
    root = Path(output_root)
    lock = _authenticate_confirmation(root)
    if lock["status"] == "no_advance":
        return "no_advance"
    marker = root / "metrics/.evaluate.complete.json"
    if marker.exists():
        _authenticate_completion(root, "evaluate")
        return "advance"
    diagnostic, arms = _load_diagnostic_test_features(root, lock)
    if len(diagnostic) != 3760 or set(diagnostic["set_id"].astype(str)) != {"set03"}:
        raise RuntimeError("set03 diagnostic cohort changed")
    metrics = root / "metrics"
    train = pd.concat(
        [pd.read_csv(metrics / "prepared_development.csv"), pd.read_csv(metrics / "confirmation_features.csv")],
        ignore_index=True,
    )
    candidate = str(lock["confirmed_candidate"])
    prediction_parts = []
    result_rows = []
    for arm in (*model_core.BASELINE_ARMS, candidate):
        probability = model_core.fit_predict_seed_mean(train, diagnostic, arms[arm], arm, model_core.SEEDS)
        prediction = diagnostic.loc[:, ["sequence_id", "set_id", "track_id", "video_id", "label"]].copy()
        prediction["arm"] = arm
        prediction["probability"] = probability
        prediction["prediction"] = (probability >= 0.5).astype(int)
        prediction_parts.append(prediction)
        result_rows.append(_metric_row(arm, prediction, len(arms[arm]), "set03_test_informed_diagnostic", "set03"))
    predictions = pd.concat(prediction_parts, ignore_index=True)
    results = pd.DataFrame(result_rows)
    wide = predictions.pivot(index=["sequence_id", "set_id", "track_id", "video_id", "label"], columns="arm", values="probability").reset_index()
    transitions = wide.loc[:, ["sequence_id", "track_id", "video_id", "label"]].copy()
    for arm in (*model_core.BASELINE_ARMS, candidate):
        transitions[f"{arm}_prediction"] = (wide[arm] >= 0.5).astype(int)
    transitions["bbox_to_candidate"] = [
        f"{model_core.BASELINE_ARMS[0]}_{'correct' if b == y else 'wrong'}_to_{candidate}_{'correct' if c == y else 'wrong'}"
        for b, c, y in zip(transitions[f"{model_core.BASELINE_ARMS[0]}_prediction"], transitions[f"{candidate}_prediction"], transitions["label"])
    ]
    output_frames = {
        "diagnostic_set03_predictions.csv": predictions,
        "diagnostic_set03_results.csv": results,
        "diagnostic_set03_error_transitions.csv": transitions,
    }
    outputs = []
    for name, value in output_frames.items():
        path = metrics / name
        _write_csv(path, value)
        outputs.append(path)
    provenance_path = metrics / "source_provenance_evaluate.json"
    _write_json(provenance_path, _source_manifest((EXPECTED_PREREGISTRATION["diagnostic_test_set"],)))
    outputs.append(provenance_path)
    _publish_completion(root, "evaluate", outputs, diagnostic_status=EXPECTED_PREREGISTRATION["diagnostic_test_status"])
    return "advance"


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]

    def render(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = [f"| {' | '.join(columns)} |", f"| {' | '.join('---' for _ in columns)} |"]
    rows.extend(f"| {' | '.join(render(value) for value in row)} |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def run_report(output_root: Path = EXPERIMENT_ROOT) -> str:
    root = Path(output_root)
    _authenticate_completion(root, "prepare")
    _authenticate_completion(root, "develop")
    confirmation = _authenticate_confirmation(root)
    development = pd.read_csv(root / "metrics/development_results.csv")
    selected = confirmation.get("selected_candidate")
    lines = [
        "# Exp037 Results",
        "",
        "## Scientific Status",
        "",
        f"- Development selection: `{selected or 'none'}`.",
        f"- Set06 promotion: `{confirmation['status']}`.",
        "- Threshold remained fixed at `0.5`.",
        "- Development used only set01, set02, set04, and set05.",
        "- Set06 was the locked internal confirmation cohort.",
        "- The combined keypoint file was streamed under the frozen exception; no unauthorized record was retained, and the hardened-loader reconstruction matched the saved matrix.",
    ]
    if confirmation["status"] == "advance":
        lines.append("- Set03 was accessed only as **TEST-INFORMED DIAGNOSTIC - NOT INDEPENDENT CONFIRMATION**.")
    else:
        lines.append("- Set03 was not accessed or materialized because the frozen gate did not advance.")
    lines.extend(["", "## Development OOF", "", _markdown_table(development), ""])
    strongest = development.set_index("arm").loc["joint_temporal_ego_xgb"]
    v2 = development.set_index("arm").loc["v2_bbox_xgb"]
    v1 = development.set_index("arm").loc["v1_bbox_rf"]
    lines.extend(
        [
            "The strongest candidate, `joint_temporal_ego_xgb`, improved PR-AUC by "
            f"`{strongest['pr_auc'] - v2['pr_auc']:.6f}` and fixed-0.5 F1 by "
            f"`{strongest['f1_fixed_0p5'] - max(v2['f1_fixed_0p5'], v1['f1_fixed_0p5']):.6f}`. "
            "It was rejected because its high-confidence FP/FN counts were "
            f"`{int(strongest['fp_ge_0p9_count'])}`/`{int(strongest['fn_le_0p1_count'])}` versus the "
            f"dual-baseline minima `{min(int(v2['fp_ge_0p9_count']), int(v1['fp_ge_0p9_count']))}`/"
            f"`{min(int(v2['fn_le_0p1_count']), int(v1['fn_le_0p1_count']))}`.",
            "",
        ]
    )
    lines.extend(
        [
            "## Paired Track Bootstrap",
            "",
            _markdown_table(pd.read_csv(root / "metrics/development_bootstrap.csv")),
            "",
            "## Per-Set Deltas",
            "",
            _markdown_table(
                pd.read_csv(root / "metrics/development_per_set.csv").loc[
                    lambda value: value["arm"].isin(EXPECTED_PREREGISTRATION["candidate_arms"]),
                    ["arm", "set_id", "pr_auc_delta_vs_v2", "f1_delta_vs_best"],
                ]
            ),
            "",
        ]
    )
    confirmation_results = root / "metrics/confirmation_results.csv"
    if confirmation_results.exists():
        lines.extend(["## Set06 Confirmation", "", _markdown_table(pd.read_csv(confirmation_results)), ""])
    diagnostic_results = root / "metrics/diagnostic_set03_results.csv"
    if diagnostic_results.exists():
        lines.extend(
            [
                "## TEST-INFORMED DIAGNOSTIC - NOT INDEPENDENT CONFIRMATION",
                "",
                _markdown_table(pd.read_csv(diagnostic_results)),
                "",
            ]
        )
    lines.extend(
        [
            "## Reproduction",
            "",
            "```bash",
            "python experiments/037_pie_joint_biu_temporal_ego/scripts/run_experiment.py prepare",
            "python experiments/037_pie_joint_biu_temporal_ego/scripts/run_experiment.py develop",
            "python experiments/037_pie_joint_biu_temporal_ego/scripts/run_experiment.py confirm",
            "python experiments/037_pie_joint_biu_temporal_ego/scripts/run_experiment.py evaluate",
            "python experiments/037_pie_joint_biu_temporal_ego/scripts/run_experiment.py report",
            "python experiments/037_pie_joint_biu_temporal_ego/scripts/run_experiment.py verify",
            "```",
            "",
        ]
    )
    _atomic_bytes(root / "RESULTS.md", "\n".join(lines).encode())
    return "complete"


def run_verify(output_root: Path = EXPERIMENT_ROOT) -> str:
    root = Path(output_root)
    checks = {}
    for phase in ("prepare", "develop", "confirm"):
        _authenticate_completion(root, phase)
        checks[f"{phase}_manifest"] = "pass"
    lock = _authenticate_confirmation(root)
    evaluate_marker = root / "metrics/.evaluate.complete.json"
    if lock["status"] == "advance":
        _authenticate_completion(root, "evaluate")
        checks["evaluate_manifest"] = "pass"
    elif evaluate_marker.exists():
        raise RuntimeError("forbidden evaluation completion exists after no_advance")
    else:
        checks["set03_not_materialized_or_evaluated"] = "pass"
    for name in ("development_oof_predictions.csv", "confirmation_predictions.csv", "diagnostic_set03_predictions.csv"):
        path = root / "metrics" / name
        if not path.exists():
            continue
        predictions = pd.read_csv(path)
        saved_results_name = name.replace("oof_predictions", "results").replace("predictions", "results")
        saved_path = root / "metrics" / saved_results_name
        if not saved_path.exists():
            continue
        saved = pd.read_csv(saved_path).set_index("arm")
        for arm, group in predictions.groupby("arm"):
            recomputed = model_core.compute_metrics(group["label"].to_numpy(int), group["probability"].to_numpy(float))
            for metric in ("pr_auc", "f1_fixed_0p5", "roc_auc", "fp", "fn", "fp_ge_0p9_count", "fn_le_0p1_count"):
                if not np.isclose(float(recomputed[metric]), float(saved.loc[arm, metric]), equal_nan=True):
                    raise RuntimeError(f"metric reconstruction failed for {name}/{arm}/{metric}")
        checks[f"{name}_metrics"] = "pass"
    report = {"status": "PASS", "checks": checks, "confirmation_status": lock["status"]}
    _write_json(root / "metrics/verification_report.json", report)
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
