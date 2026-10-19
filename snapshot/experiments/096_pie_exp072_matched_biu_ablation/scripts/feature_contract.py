"""Feature-arm construction for the Exp072 matched BIU ablation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


STATS = ("early", "late", "late_minus_early")
BBOX_TEMPORAL_BASES = ("bbox_center_dx_norm", "bbox_scale_log_delta")
V2_TEMPORAL_BASES = (
    "lower_body_motion_magnitude_mean",
    "lead_ankle_forward_delta",
    "torso_motion_magnitude_mean",
    "torso_center_x_delta",
)
POSE_QUALITY_BASES = ("pose_confidence_mean",)
EGO_COLUMNS = (
    "obd_speed_mean",
    "obd_speed_std",
    "obd_speed_first",
    "obd_speed_last",
    "obd_speed_delta",
)


def _temporal_columns(bases: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"temporal__{base}_{stat}" for base in bases for stat in STATS)


def _families(definitions_path: Path) -> dict[str, tuple[str, ...]]:
    frame = pd.read_csv(Path(definitions_path))
    required = {"feature_set", "feature_columns_json"}
    if required.difference(frame.columns) or frame["feature_set"].duplicated().any():
        raise ValueError("Exp012 feature-definition schema changed")
    rows = frame.set_index("feature_set")
    expected = {"bbox_only": 53, "biu_v1_only": 72, "biu_v2_core": 56}
    result: dict[str, tuple[str, ...]] = {}
    for name, count in expected.items():
        if name not in rows.index:
            raise ValueError(f"Exp012 feature definitions omit {name}")
        columns = tuple(map(str, json.loads(str(rows.loc[name, "feature_columns_json"]))))
        if len(columns) != count or len(columns) != len(set(columns)):
            raise ValueError(f"Exp012 {name} contract changed")
        result[name] = columns
    return result


def build_matched_arm_columns(
    definitions_path: Path, parent_contract_path: Path
) -> dict[str, tuple[str, ...]]:
    """Return matched Exp096 arms while preserving Exp072's candidate exactly."""
    families = _families(definitions_path)
    parent = json.loads(Path(parent_contract_path).read_text(encoding="utf-8"))
    try:
        combined = tuple(map(str, parent["arms"]["joint_temporal_ego_xgb"]))
    except (KeyError, TypeError) as error:
        raise ValueError("Exp072 parent contract lacks joint_temporal_ego_xgb") from error
    if len(combined) != 187 or len(combined) != len(set(combined)):
        raise ValueError("Exp072 combined contract must contain 187 unique features")

    bbox = families["bbox_only"]
    v1 = families["biu_v1_only"]
    v2 = tuple(f"v2__{name}" for name in families["biu_v2_core"])
    bbox_temporal = _temporal_columns(BBOX_TEMPORAL_BASES)
    pose_quality = _temporal_columns(POSE_QUALITY_BASES)
    v2_temporal = _temporal_columns(V2_TEMPORAL_BASES)

    base = bbox + bbox_temporal + EGO_COLUMNS
    arms = {
        "base": base,
        "v1": base + v1 + pose_quality,
        "v2": base + v2 + pose_quality + v2_temporal,
        "combined": combined,
    }
    expected_counts = {"base": 64, "v1": 139, "v2": 135, "combined": 187}
    for name, columns in arms.items():
        if len(columns) != expected_counts[name] or len(columns) != len(set(columns)):
            raise ValueError(f"Exp096 {name} arm contract changed")
    return arms
