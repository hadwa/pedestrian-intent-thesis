"""Matched feature views and BIU-specific reliability for the confidence benchmark."""

from __future__ import annotations

import re

import pandas as pd

from src.biu_v2_extractor import COCO_KEYPOINTS, FEATURE_DICTIONARY


REPRESENTATIONS = {
    "kp_xy": "keypoint coordinates only",
    "kp_xyc": "keypoint coordinates plus confidence",
    "biu_v2": "BIU geometry only",
    "biu_v2_plus_global_conf": "BIU geometry plus frozen global confidence",
    "biu_v2_plus_reliability": "BIU geometry plus BIU-specific reliability",
    "bbox": "bounding-box features only",
    "bbox_plus_biu_v2": "bounding-box features plus BIU geometry",
    "bbox_plus_biu_v2_plus_reliability": "bounding-box features plus BIU geometry and reliability",
}

_CONFIDENCE_COLUMN = re.compile(r"kpconf__f(?P<frame>\d+)_kp(?P<joint>\d+)$")


def build_biu_joint_map() -> dict[str, list[int]]:
    """Map each geometry BIU v2 feature to its explicit COCO input joints."""
    keypoint_indices = {name: index for index, name in enumerate(COCO_KEYPOINTS)}
    mapping: dict[str, list[int]] = {}
    for feature in FEATURE_DICTIONARY:
        if feature["group"] == "confidence_metadata":
            continue
        names = feature["required_keypoints"].split(";")
        unknown_names = [name for name in names if name not in keypoint_indices]
        if unknown_names:
            raise ValueError(
                f"Cannot map BIU {feature['feature_name']} to COCO joints: {unknown_names}"
            )
        mapping[feature["feature_name"]] = [keypoint_indices[name] for name in names]
    return mapping


def _confidence_columns_by_frame(df: pd.DataFrame) -> dict[int, dict[int, str]]:
    columns_by_frame: dict[int, dict[int, str]] = {}
    for column in df.columns:
        match = _CONFIDENCE_COLUMN.fullmatch(column)
        if match is None:
            continue
        frame = int(match["frame"])
        joint = int(match["joint"])
        columns_by_frame.setdefault(frame, {})[joint] = column
    return columns_by_frame


def derive_biu_reliability(
    df: pd.DataFrame, biu_joint_map: dict[str, list[int]]
) -> pd.DataFrame:
    """Derive frozen observation-level reliability without altering BIU values."""
    if "sequence_id" not in df.columns:
        raise ValueError("Cannot derive BIU reliability without sequence_id")

    columns_by_frame = _confidence_columns_by_frame(df)
    if not columns_by_frame:
        raise ValueError("Cannot derive BIU reliability without kpconf__f##_kp## columns")

    out = pd.DataFrame({"sequence_id": df["sequence_id"]}, index=df.index)
    for biu_name, joints in biu_joint_map.items():
        if not joints:
            raise ValueError(f"BIU {biu_name} has no contributing joints")
        frame_columns: list[list[str]] = []
        for frame in sorted(columns_by_frame):
            missing_joints = [joint for joint in joints if joint not in columns_by_frame[frame]]
            if missing_joints:
                raise ValueError(
                    f"BIU {biu_name} is missing confidence columns for frame {frame}: {missing_joints}"
                )
            frame_columns.append([columns_by_frame[frame][joint] for joint in joints])

        frame_means = pd.concat(
            [df.loc[:, columns].mean(axis=1) for columns in frame_columns], axis=1
        )
        frame_mins = pd.concat(
            [df.loc[:, columns].min(axis=1) for columns in frame_columns], axis=1
        )
        out[f"rel__{biu_name}__mean_conf"] = frame_means.mean(axis=1)
        out[f"rel__{biu_name}__min_conf"] = frame_mins.min(axis=1)
    return out


def _is_biu_geometry_column(column: str, biu_joint_map: dict[str, list[int]]) -> bool:
    return any(_is_biu_column_for_feature(column, biu_name) for biu_name in biu_joint_map)


def _is_biu_column_for_feature(column: str, biu_name: str) -> bool:
    return any(
        column == prefix or column.startswith(f"{prefix}_")
        for prefix in (biu_name, f"biu__{biu_name}")
    )


def _require_complete_biu_v2_geometry(
    columns: list[str], biu_joint_map: dict[str, list[int]]
) -> None:
    missing_features = [
        biu_name
        for biu_name in biu_joint_map
        if not any(_is_biu_column_for_feature(column, biu_name) for column in columns)
    ]
    if missing_features:
        raise ValueError(
            "BIU source is not BIU-v2 geometry-complete; missing features: "
            f"{missing_features}"
        )


def _reliability_columns_for_biu_columns(
    columns: list[str], biu_joint_map: dict[str, list[int]]
) -> list[str]:
    reliability_columns: list[str] = []
    for biu_name in biu_joint_map:
        if any(
            _is_biu_column_for_feature(column, biu_name)
            for column in columns
        ):
            reliability_columns.extend(
                [
                    f"rel__{biu_name}__mean_conf",
                    f"rel__{biu_name}__min_conf",
                ]
            )
    return reliability_columns


def make_feature_view(
    df: pd.DataFrame,
    representation: str,
    *,
    global_conf_cols: list[str],
    biu_joint_map: dict[str, list[int]],
) -> tuple[pd.DataFrame, list[str]]:
    """Return a source-preserving frame and ordered model-input columns for one view."""
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unsupported feature representation: {representation}")

    view = df.copy()
    biu_columns = [
        column for column in view.columns if _is_biu_geometry_column(column, biu_joint_map)
    ]
    kp_columns = [column for column in view.columns if column.startswith("kp__")]
    kpconf_columns = [column for column in view.columns if column.startswith("kpconf__")]
    bbox_columns = [column for column in view.columns if column.startswith("bbox__")]
    biu_representations = {
        "biu_v2",
        "biu_v2_plus_global_conf",
        "biu_v2_plus_reliability",
        "bbox_plus_biu_v2",
        "bbox_plus_biu_v2_plus_reliability",
    }
    if representation in biu_representations:
        _require_complete_biu_v2_geometry(view.columns.tolist(), biu_joint_map)
    missing_global_conf = [column for column in global_conf_cols if column not in view.columns]
    if representation == "biu_v2_plus_global_conf" and missing_global_conf:
        raise ValueError(
            "BIU global-confidence view is missing frozen global confidence columns: "
            f"{missing_global_conf}"
        )

    if representation in {"biu_v2_plus_reliability", "bbox_plus_biu_v2_plus_reliability"}:
        reliability = derive_biu_reliability(view, biu_joint_map).drop(columns="sequence_id")
        view = pd.concat([view, reliability], axis=1)
        reliability_columns = _reliability_columns_for_biu_columns(biu_columns, biu_joint_map)
    else:
        reliability_columns = []

    columns_by_representation = {
        "kp_xy": kp_columns,
        "kp_xyc": kp_columns + kpconf_columns,
        "biu_v2": biu_columns,
        "biu_v2_plus_global_conf": biu_columns + global_conf_cols,
        "biu_v2_plus_reliability": biu_columns + reliability_columns,
        "bbox": bbox_columns,
        "bbox_plus_biu_v2": bbox_columns + biu_columns,
        "bbox_plus_biu_v2_plus_reliability": bbox_columns + biu_columns + reliability_columns,
    }
    return view, columns_by_representation[representation]
