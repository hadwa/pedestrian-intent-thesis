"""Shared leakage-safe two-stage XGBoost and frozen half-window features."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


ACTIVE_ROOT = Path(__file__).resolve().parents[3]
if str(ACTIVE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ACTIVE_ROOT / "src"))

from biu_v2_extractor import extract_biu_v2  # noqa: E402
from features.biu_features import (  # noqa: E402
    aggregate_biu_sequence,
    compute_frame_biu_features,
    frame_has_invalid_required_keypoints,
    frame_has_low_confidence,
)


SEEDS = (42, 43, 44)
TOTAL_ESTIMATORS = 250
TEMPORAL_BASES = (
    "lower_body_motion_magnitude_mean",
    "lead_ankle_forward_delta",
    "torso_motion_magnitude_mean",
    "torso_center_x_delta",
    "bbox_center_dx_norm",
    "bbox_scale_log_delta",
    "pose_confidence_mean",
)


def load_joint_contract(definitions_path: Path) -> dict[str, tuple[str, ...]]:
    definitions = pd.read_csv(Path(definitions_path))
    if {"feature_set", "feature_columns_json"}.difference(definitions.columns):
        raise ValueError("feature definition schema changed")
    rows = definitions.set_index("feature_set")
    raw = {
        name: tuple(json.loads(str(rows.loc[name, "feature_columns_json"])))
        for name in ("bbox_only", "biu_v1_only", "biu_v2_core")
    }
    expected = {"bbox_only": 53, "biu_v1_only": 72, "biu_v2_core": 56}
    for name, count in expected.items():
        if len(raw[name]) != count or len(set(raw[name])) != count:
            raise ValueError(f"frozen {name} contract changed")
    v1_semantics = set(raw["bbox_only"]) | set(raw["biu_v1_only"])
    source_joint = tuple(sorted(v1_semantics | set(raw["biu_v2_core"])))
    joint = tuple(name if name in v1_semantics else f"v2__{name}" for name in source_joint)
    if len(joint) != 161 or len(set(joint)) != 161:
        raise ValueError("frozen joint feature contract changed")
    temporal = tuple(
        f"temporal__{name}_{stat}"
        for name in TEMPORAL_BASES
        for stat in ("early", "late", "late_minus_early")
    )
    return {**raw, "source_joint": source_joint, "joint": joint, "temporal": temporal, "portable": joint + temporal}


def _bbox53(bboxes: np.ndarray) -> dict[str, float]:
    boxes = np.asarray(bboxes, dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or len(boxes) < 2:
        raise ValueError("BBox half-window must have shape [T,4] with T >= 2")
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]
    valid = np.isfinite(boxes).all(axis=1) & (width > 0) & (height > 0)
    values = {
        "center_x": (boxes[:, 0] + boxes[:, 2]) / 2.0,
        "center_y": (boxes[:, 1] + boxes[:, 3]) / 2.0,
        "width": width,
        "height": height,
        "area": width * height,
        "aspect_ratio": np.divide(
            width,
            height,
            out=np.full_like(width, np.nan, dtype=float),
            where=height != 0,
        ),
    }
    for array in values.values():
        array[~valid] = np.nan
    output: dict[str, float] = {}
    for name, array in values.items():
        output.update(
            {
                f"bbox_{name}_mean": float(np.nanmean(array)),
                f"bbox_{name}_std": float(np.nanstd(array, ddof=0)),
                f"bbox_{name}_first": float(array[0]),
                f"bbox_{name}_last": float(array[-1]),
                f"bbox_{name}_delta": float(array[-1] - array[0]),
                f"bbox_{name}_min": float(np.nanmin(array)),
                f"bbox_{name}_max": float(np.nanmax(array)),
            }
        )
    center_x = values["center_x"]
    center_y = values["center_y"]
    area = values["area"]
    output.update(
        {
            "center_x_velocity_mean": float(np.nanmean(np.diff(center_x))),
            "center_y_velocity_mean": float(np.nanmean(np.diff(center_y))),
            "center_x_velocity_std": float(np.nanstd(np.diff(center_x), ddof=0)),
            "center_y_velocity_std": float(np.nanstd(np.diff(center_y), ddof=0)),
            "center_x_displacement": float(center_x[-1] - center_x[0]),
            "center_y_displacement": float(center_y[-1] - center_y[0]),
            "bbox_area_change": float(area[-1] - area[0]),
            "bbox_width_change": float(width[-1] - width[0]),
            "bbox_height_change": float(height[-1] - height[0]),
            "bbox_scale_change_ratio": float(area[-1] / area[0]) if area[0] > 0 else float("nan"),
            "bbox_valid_frame_count": float(valid.sum()),
        }
    )
    if len(output) != 53:
        raise RuntimeError("BBox53 extractor changed dimensionality")
    return output


def _biu_v1(coords: np.ndarray, confidence: np.ndarray) -> dict[str, float]:
    per_frame = [compute_frame_biu_features(frame) for frame in coords]
    output = aggregate_biu_sequence(per_frame)
    output["biu_valid_frame_count"] = float(
        len(coords) - sum(frame_has_invalid_required_keypoints(frame) for frame in coords)
    )
    output["biu_low_confidence_frame_count"] = float(
        sum(frame_has_low_confidence(frame) for frame in confidence)
    )
    if len(output) != 72:
        raise RuntimeError("BIU-v1 extractor changed dimensionality")
    return output


def extract_half_features(
    coords: np.ndarray,
    confidence: np.ndarray,
    bboxes: np.ndarray,
    *,
    contract: Mapping[str, tuple[str, ...]],
) -> dict[str, float]:
    coords = np.asarray(coords, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    bboxes = np.asarray(bboxes, dtype=float)
    if coords.shape != (8, 17, 2) or confidence.shape != (8, 17) or bboxes.shape != (8, 4):
        raise ValueError("half-window extraction requires exact 8-frame COCO-17 observations")
    return extract_joint_features(coords, confidence, bboxes, contract=contract)


def extract_joint_features(
    coords: np.ndarray,
    confidence: np.ndarray,
    bboxes: np.ndarray,
    *,
    contract: Mapping[str, tuple[str, ...]],
) -> dict[str, float]:
    coords = np.asarray(coords, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    bboxes = np.asarray(bboxes, dtype=float)
    if coords.ndim != 3 or coords.shape[1:] != (17, 2) or len(coords) < 2:
        raise ValueError("joint features require at least two COCO-17 frames")
    if confidence.shape != coords.shape[:2] or bboxes.shape != (len(coords), 4):
        raise ValueError("pose, confidence, and BBox windows are not aligned")
    bbox = _bbox53(bboxes)
    v1 = _biu_v1(coords, confidence)
    v2 = extract_biu_v2(coords, confidence, bbox_sequence=bboxes)
    output: dict[str, float] = {}
    v1_semantics = set(contract["bbox_only"]) | set(contract["biu_v1_only"])
    for source_name, output_name in zip(contract["source_joint"], contract["joint"]):
        if source_name in bbox:
            value = bbox[source_name]
        elif source_name in v1:
            value = v1[source_name]
        elif source_name in v2 and source_name not in v1_semantics:
            value = v2[source_name]
        else:
            raise RuntimeError(f"missing frozen half-window feature: {source_name}")
        output[output_name] = float(value)
    if tuple(output) != tuple(contract["joint"]):
        raise RuntimeError("half-window feature order changed")
    return output


def _temporal_features(
    coords: np.ndarray,
    confidence: np.ndarray,
    bboxes: np.ndarray,
) -> dict[str, float]:
    by_half: dict[str, dict[str, float]] = {}
    for name, indices in (("early", slice(0, 8)), ("late", slice(8, 16))):
        half_coords = coords[indices]
        half_confidence = confidence[indices]
        half_bboxes = bboxes[indices]
        v2 = extract_biu_v2(half_coords, half_confidence, bbox_sequence=half_bboxes)
        widths = half_bboxes[:, 2] - half_bboxes[:, 0]
        centers = (half_bboxes[:, 0] + half_bboxes[:, 2]) / 2.0
        areas = widths * (half_bboxes[:, 3] - half_bboxes[:, 1])
        by_half[name] = {
            "lower_body_motion_magnitude_mean": float(v2["lower_body_motion_magnitude_mean"]),
            "lead_ankle_forward_delta": float(v2["lead_ankle_forward_delta"]),
            "torso_motion_magnitude_mean": float(v2["torso_motion_magnitude_mean"]),
            "torso_center_x_delta": float(v2["torso_center_x_delta"]),
            "bbox_center_dx_norm": float((centers[-1] - centers[0]) / widths.mean()),
            "bbox_scale_log_delta": float(np.log(areas[-1] / areas[0])),
            "pose_confidence_mean": float(half_confidence.mean()),
        }
    output: dict[str, float] = {}
    for base in TEMPORAL_BASES:
        early = by_half["early"][base]
        late = by_half["late"][base]
        output[f"temporal__{base}_early"] = early
        output[f"temporal__{base}_late"] = late
        output[f"temporal__{base}_late_minus_early"] = late - early
    return output


def extract_window_views(
    coords: np.ndarray,
    confidence: np.ndarray,
    bboxes: np.ndarray,
    *,
    contract: Mapping[str, tuple[str, ...]],
) -> dict[str, dict[str, float]]:
    coords = np.asarray(coords, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    bboxes = np.asarray(bboxes, dtype=float)
    if coords.shape != (16, 17, 2) or confidence.shape != (16, 17) or bboxes.shape != (16, 4):
        raise ValueError("window views require exact 16-frame COCO-17 observations")
    early = extract_half_features(coords[:8], confidence[:8], bboxes[:8], contract=contract)
    full = extract_joint_features(coords, confidence, bboxes, contract=contract)
    portable = {**full, **_temporal_features(coords, confidence, bboxes)}
    if tuple(portable) != tuple(contract["portable"]):
        raise RuntimeError("portable feature order changed")
    return {"early": early, "full": full, "portable": portable}


def speed_features(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.shape != (8,) or not np.isfinite(values).all():
        raise ValueError("early ego speed requires eight finite frame values")
    return {
        "obd_speed_mean": float(values.mean()),
        "obd_speed_std": float(values.std(ddof=0)),
        "obd_speed_first": float(values[0]),
        "obd_speed_last": float(values[-1]),
        "obd_speed_delta": float(values[-1] - values[0]),
    }


def build_xgb(seed: int, labels: np.ndarray, *, n_estimators: int) -> Pipeline:
    labels = np.asarray(labels, dtype=int)
    if int(seed) not in SEEDS or labels.ndim != 1 or set(np.unique(labels)) != {0, 1}:
        raise ValueError("XGBoost requires an approved seed and both binary classes")
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    model = XGBClassifier(
        n_estimators=int(n_estimators),
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=int(seed),
        n_jobs=4,
        scale_pos_weight=float(negatives / positives),
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def group_oof_probabilities(
    values: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    n_estimators: int,
    n_splits: int = 5,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)
    groups = np.asarray(groups).astype(str)
    if values.ndim != 2 or len(values) != len(labels) or len(groups) != len(labels):
        raise ValueError("OOF inputs are not row-aligned")
    if len(np.unique(groups)) < int(n_splits):
        raise ValueError("OOF requires at least one unique group per fold")
    splitter = StratifiedGroupKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    probabilities = np.full(len(labels), np.nan, dtype=float)
    coverage = np.zeros(len(labels), dtype=int)
    fold_rows = []
    for fold, (train_index, holdout_index) in enumerate(splitter.split(values, labels, groups), start=1):
        train_groups = set(groups[train_index])
        holdout_groups = set(groups[holdout_index])
        overlap = train_groups.intersection(holdout_groups)
        if overlap:
            raise RuntimeError("track group leaked across an OOF fold")
        model = build_xgb(seed, labels[train_index], n_estimators=n_estimators)
        model.fit(values[train_index], labels[train_index])
        probabilities[holdout_index] = model.predict_proba(values[holdout_index])[:, 1]
        coverage[holdout_index] += 1
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": int(len(train_index)),
                "holdout_rows": int(len(holdout_index)),
                "train_groups": int(len(train_groups)),
                "holdout_groups": int(len(holdout_groups)),
                "group_overlap": int(len(overlap)),
            }
        )
    if not np.isfinite(probabilities).all() or not np.all(coverage == 1):
        raise RuntimeError("OOF probabilities do not cover each training row exactly once")
    return probabilities, {
        "covered_rows": int((coverage > 0).sum()),
        "duplicate_rows": int((coverage > 1).sum()),
        "folds": fold_rows,
    }


@dataclass
class TwoStageBundle:
    stage1: Pipeline
    stage2: Pipeline
    stage1_estimators: int
    stage2_estimators: int
    early_feature_count: int
    stage2_feature_count: int


def fit_two_stage(
    early_values: np.ndarray,
    full_values: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    total_estimators: int = TOTAL_ESTIMATORS,
    n_splits: int = 5,
) -> tuple[TwoStageBundle, dict[str, Any]]:
    if int(total_estimators) < 2:
        raise ValueError("two-stage tree budget must be at least two")
    stage1_estimators = int(total_estimators) // 2
    stage2_estimators = int(total_estimators) - stage1_estimators
    oof, oof_audit = group_oof_probabilities(
        early_values,
        labels,
        groups,
        seed=seed,
        n_estimators=stage1_estimators,
        n_splits=n_splits,
    )
    labels = np.asarray(labels, dtype=int)
    early_values = np.asarray(early_values, dtype=float)
    full_values = np.asarray(full_values, dtype=float)
    stage1 = build_xgb(seed, labels, n_estimators=stage1_estimators)
    stage1.fit(early_values, labels)
    stage2_values = np.column_stack([full_values, oof])
    stage2 = build_xgb(seed, labels, n_estimators=stage2_estimators)
    stage2.fit(stage2_values, labels)
    bundle = TwoStageBundle(
        stage1=stage1,
        stage2=stage2,
        stage1_estimators=stage1_estimators,
        stage2_estimators=stage2_estimators,
        early_feature_count=int(early_values.shape[1]),
        stage2_feature_count=int(stage2_values.shape[1]),
    )
    return bundle, {"oof": oof_audit, "total_estimators": int(total_estimators)}


def predict_two_stage(
    bundle: TwoStageBundle,
    early_values: np.ndarray,
    full_values: np.ndarray,
) -> np.ndarray:
    early_probability = bundle.stage1.predict_proba(np.asarray(early_values, dtype=float))[:, 1]
    stage2_values = np.column_stack([np.asarray(full_values, dtype=float), early_probability])
    if stage2_values.shape[1] != bundle.stage2_feature_count:
        raise ValueError("two-stage final feature width changed")
    return bundle.stage2.predict_proba(stage2_values)[:, 1]


def assert_cohort_parity(reference: pd.DataFrame, candidate: pd.DataFrame) -> None:
    columns = ["sequence_id", "track_id", "label"]
    if set(columns).difference(reference.columns) or set(columns).difference(candidate.columns):
        raise ValueError("cohort parity requires sequence, track, and label")
    try:
        pd.testing.assert_frame_equal(
            reference.loc[:, columns].reset_index(drop=True),
            candidate.loc[:, columns].reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as error:
        raise ValueError("cohort parity failed") from error
