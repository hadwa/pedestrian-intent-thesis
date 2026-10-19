"""Bounding-box motion features aligned by frozen sequence_id.

The supported bbox source for this ablation is the PIE CVAT-style XML annotation
folder. Boxes are pixel coordinates stored per pedestrian track and per frame as
``xtl, ytl, xbr, ybr``. Feature builders must join by frozen ``sequence_id`` and
must not regenerate windows or labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

BBOX_BASE_FEATURES = ["center_x", "center_y", "width", "height", "area", "aspect_ratio"]
AGGREGATIONS = ["mean", "std", "first", "last", "delta", "min", "max"]
METADATA_COLUMNS = [
    "sequence_id",
    "split",
    "label",
    "set_id",
    "video_id",
    "ped_id",
    "track_id",
    "start_frame",
    "end_frame",
    "bbox_missing_frame_count",
    "bbox_repaired_frame_count",
    "bbox_repair_method",
    "bbox_repaired_frames",
]


@dataclass
class BBoxBuildReport:
    """Diagnostics collected while building bbox feature tables."""

    bbox_source: str
    bbox_schema: str = "PIE CVAT XML pedestrian track boxes: xtl, ytl, xbr, ybr in pixel coordinates"
    xml_files: int = 0
    bbox_rows: int = 0
    duplicate_bbox_keys: int = 0
    missing_required_frames: dict[str, int] = field(default_factory=dict)
    repaired_frame_counts: dict[str, int] = field(default_factory=dict)
    repaired_sequence_counts: dict[str, int] = field(default_factory=dict)
    max_missing_frames_per_sequence: dict[str, int] = field(default_factory=dict)
    invalid_bbox_frames: dict[str, int] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)
    nan_counts: dict[str, int] = field(default_factory=dict)
    feature_nan_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    validation_messages: list[str] = field(default_factory=list)


def _annotation_xml_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.exists():
        raise FileNotFoundError(f"BBox source does not exist: {source}")
    files = sorted(p for p in source.rglob("*_annt.xml") if p.is_file())
    if not files:
        files = sorted(p for p in source.rglob("*.xml") if p.is_file() and "Zone.Identifier" not in p.name)
    if not files:
        raise FileNotFoundError(f"No PIE annotation XML files found under {source}")
    return files


def load_pie_bbox_annotations(bbox_source: Path) -> pd.DataFrame:
    """Load PIE pedestrian boxes from CVAT-style XML annotations.

    Returns one row per available pedestrian frame with columns:
    ``set_id, video_id, ped_id, frame_idx, x1, y1, x2, y2, source_annotation_path``.
    """
    files = _annotation_xml_files(bbox_source)
    rows: list[dict[str, object]] = []
    for xml_path in files:
        set_id = xml_path.parent.name
        if not set_id.startswith("set") and xml_path.parent.parent.name.startswith("set"):
            set_id = xml_path.parent.parent.name
        video_id = xml_path.stem.replace("_annt", "")
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError as exc:
            raise ValueError(f"Failed to parse XML annotation {xml_path}: {exc}") from exc

        for track in tree.getroot().findall("track"):
            if track.attrib.get("label") != "pedestrian":
                continue
            boxes = track.findall("box")
            if not boxes:
                continue
            first_id = None
            for box in boxes:
                pid_attr = box.find("attribute[@name='id']")
                if pid_attr is not None and pid_attr.text:
                    first_id = pid_attr.text.strip()
                    break
            if not first_id:
                continue
            for box in boxes:
                if str(box.attrib.get("outside", "0")) == "1":
                    continue
                try:
                    frame_idx = int(float(box.attrib["frame"]))
                    x1 = float(box.attrib["xtl"])
                    y1 = float(box.attrib["ytl"])
                    x2 = float(box.attrib["xbr"])
                    y2 = float(box.attrib["ybr"])
                except KeyError as exc:
                    raise ValueError(f"Annotation box in {xml_path} missing required bbox attribute: {exc}") from exc
                rows.append(
                    {
                        "set_id": set_id,
                        "video_id": video_id,
                        "ped_id": first_id,
                        "frame_idx": frame_idx,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "source_annotation_path": str(xml_path),
                    }
                )
    if not rows:
        raise RuntimeError(f"No bbox rows loaded from {bbox_source}")
    df = pd.DataFrame(rows)
    duplicate_count = int(df.duplicated(["set_id", "video_id", "ped_id", "frame_idx"]).sum())
    if duplicate_count:
        dup = df.loc[df.duplicated(["set_id", "video_id", "ped_id", "frame_idx"], keep=False), ["set_id", "video_id", "ped_id", "frame_idx"]].head().to_dict("records")
        raise ValueError(f"Duplicate bbox keys found in annotations: count={duplicate_count}, sample={dup}")
    return df.sort_values(["set_id", "video_id", "ped_id", "frame_idx"]).reset_index(drop=True)


def _per_frame_bbox_features(bboxes: np.ndarray) -> pd.DataFrame:
    arr = np.asarray(bboxes, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"Expected bboxes with shape [T, 4], got {arr.shape}")
    width = arr[:, 2] - arr[:, 0]
    height = arr[:, 3] - arr[:, 1]
    valid = np.isfinite(arr).all(axis=1) & (width > 0) & (height > 0)
    center_x = (arr[:, 0] + arr[:, 2]) / 2.0
    center_y = (arr[:, 1] + arr[:, 3]) / 2.0
    area = width * height
    aspect_ratio = np.divide(width, height, out=np.full_like(width, np.nan, dtype=float), where=height != 0)
    frame = pd.DataFrame(
        {
            "center_x": center_x,
            "center_y": center_y,
            "width": width,
            "height": height,
            "area": area,
            "aspect_ratio": aspect_ratio,
            "valid_bbox": valid.astype(int),
        }
    )
    for col in BBOX_BASE_FEATURES:
        frame.loc[~valid, col] = np.nan
    return frame


def compute_bbox_motion_features(bboxes: np.ndarray) -> dict[str, float]:
    """Compute bbox motion features for one 16-frame sequence.

    Args:
        bboxes: Array of shape ``[T, 4]`` in pixel ``x1, y1, x2, y2`` format.
    """
    per_frame = _per_frame_bbox_features(bboxes)
    features: dict[str, float] = {}
    for col in BBOX_BASE_FEATURES:
        values = per_frame[col].to_numpy(dtype=float)
        features[f"bbox_{col}_mean"] = float(np.nanmean(values)) if not np.isnan(values).all() else np.nan
        features[f"bbox_{col}_std"] = float(np.nanstd(values)) if not np.isnan(values).all() else np.nan
        features[f"bbox_{col}_first"] = float(values[0]) if len(values) else np.nan
        features[f"bbox_{col}_last"] = float(values[-1]) if len(values) else np.nan
        features[f"bbox_{col}_delta"] = float(values[-1] - values[0]) if len(values) else np.nan
        features[f"bbox_{col}_min"] = float(np.nanmin(values)) if not np.isnan(values).all() else np.nan
        features[f"bbox_{col}_max"] = float(np.nanmax(values)) if not np.isnan(values).all() else np.nan

    center_x = per_frame["center_x"].to_numpy(dtype=float)
    center_y = per_frame["center_y"].to_numpy(dtype=float)
    area = per_frame["area"].to_numpy(dtype=float)
    width = per_frame["width"].to_numpy(dtype=float)
    height = per_frame["height"].to_numpy(dtype=float)
    vx = np.diff(center_x)
    vy = np.diff(center_y)
    features.update(
        {
            "center_x_velocity_mean": float(np.nanmean(vx)) if len(vx) and not np.isnan(vx).all() else np.nan,
            "center_y_velocity_mean": float(np.nanmean(vy)) if len(vy) and not np.isnan(vy).all() else np.nan,
            "center_x_velocity_std": float(np.nanstd(vx)) if len(vx) and not np.isnan(vx).all() else np.nan,
            "center_y_velocity_std": float(np.nanstd(vy)) if len(vy) and not np.isnan(vy).all() else np.nan,
            "center_x_displacement": float(center_x[-1] - center_x[0]) if len(center_x) else np.nan,
            "center_y_displacement": float(center_y[-1] - center_y[0]) if len(center_y) else np.nan,
            "bbox_area_change": float(area[-1] - area[0]) if len(area) else np.nan,
            "bbox_width_change": float(width[-1] - width[0]) if len(width) else np.nan,
            "bbox_height_change": float(height[-1] - height[0]) if len(height) else np.nan,
            "bbox_scale_change_ratio": float(area[-1] / area[0]) if len(area) and np.isfinite(area[0]) and area[0] > 0 else np.nan,
            "bbox_valid_frame_count": float(per_frame["valid_bbox"].sum()),
        }
    )
    return features


def _build_track_lookup(bbox_df: pd.DataFrame) -> dict[tuple[str, str, str], dict[int, np.ndarray]]:
    """Build same-track frame -> bbox lookup for nearest-frame repair."""
    lookup: dict[tuple[str, str, str], dict[int, np.ndarray]] = {}
    for (set_id, video_id, ped_id), group in bbox_df.groupby(["set_id", "video_id", "ped_id"], sort=False):
        frame_map: dict[int, np.ndarray] = {}
        for rec in group.itertuples(index=False):
            frame_map[int(rec.frame_idx)] = np.asarray([rec.x1, rec.y1, rec.x2, rec.y2], dtype=float)
        lookup[(str(set_id), str(video_id), str(ped_id))] = frame_map
    return lookup


def _repair_missing_bbox(frame_idx: int, frame_map: dict[int, np.ndarray]) -> tuple[np.ndarray, str]:
    """Repair one missing bbox using same-track nearest available bbox.

    Policy: prefer previous adjacent frame, then next adjacent frame, otherwise the
    nearest available frame in the same pedestrian track.
    """
    if frame_idx - 1 in frame_map:
        return frame_map[frame_idx - 1].copy(), "previous_frame"
    if frame_idx + 1 in frame_map:
        return frame_map[frame_idx + 1].copy(), "next_frame"
    if not frame_map:
        raise ValueError(f"Cannot repair missing bbox frame {frame_idx}: track has no available boxes.")
    nearest = min(frame_map, key=lambda available: abs(available - frame_idx))
    return frame_map[nearest].copy(), f"nearest_track_frame:{nearest}"


def build_bbox_features_for_sequences(
    sequences: pd.DataFrame,
    bbox_df: pd.DataFrame,
    split_name: str,
    report: BBoxBuildReport | None = None,
) -> pd.DataFrame:
    """Build one bbox feature row per frozen sequence with rare one-frame repair."""
    required = ["sequence_id", "split", "label", "set_id", "video_id", "ped_id", "track_id", "start_frame", "end_frame"]
    missing = [c for c in required if c not in sequences.columns]
    if missing:
        raise ValueError(f"{split_name} split CSV missing required columns: {missing}")

    track_lookup = _build_track_lookup(bbox_df)
    rows: list[dict[str, object]] = []
    missing_frames_total = 0
    repaired_frames_total = 0
    repaired_sequences = 0
    invalid_frames = 0
    max_missing_for_sequence = 0

    for row in sequences.itertuples(index=False):
        frames = list(range(int(row.start_frame), int(row.end_frame) + 1))
        if len(frames) != 16:
            raise ValueError(f"{row.sequence_id}: expected 16 frames, got {len(frames)}")
        bbox_rows: list[np.ndarray] = []
        missing_for_sequence: list[int] = []
        repair_methods: list[str] = []
        track_key = (str(row.set_id), str(row.video_id), str(row.ped_id))
        frame_map = track_lookup.get(track_key, {})

        for frame_idx in frames:
            if int(frame_idx) in frame_map:
                bbox_rows.append(frame_map[int(frame_idx)].copy())
                continue
            missing_for_sequence.append(frame_idx)
            if len(missing_for_sequence) > 1:
                raise ValueError(
                    f"{row.sequence_id}: more than one missing bbox frame {missing_for_sequence}; "
                    "approved repair policy only allows at most one missing frame per 16-frame sequence."
                )
            repaired_bbox, method = _repair_missing_bbox(frame_idx, frame_map)
            bbox_rows.append(repaired_bbox)
            repair_methods.append(method)

        missing_count = len(missing_for_sequence)
        max_missing_for_sequence = max(max_missing_for_sequence, missing_count)
        missing_frames_total += missing_count
        repaired_frames_total += missing_count
        if missing_count:
            repaired_sequences += 1

        bboxes = np.asarray(bbox_rows, dtype=float)
        if bboxes.shape != (16, 4):
            raise ValueError(f"{row.sequence_id}: repaired bbox sequence has shape {bboxes.shape}, expected (16, 4)")
        widths = bboxes[:, 2] - bboxes[:, 0]
        heights = bboxes[:, 3] - bboxes[:, 1]
        invalid_frames += int((~np.isfinite(bboxes).all(axis=1) | (widths <= 0) | (heights <= 0)).sum())
        feature_values = compute_bbox_motion_features(bboxes)
        meta = {col: getattr(row, col) for col in required}
        meta.update(
            {
                "bbox_missing_frame_count": int(missing_count),
                "bbox_repaired_frame_count": int(missing_count),
                "bbox_repair_method": ";".join(repair_methods) if repair_methods else "none",
                "bbox_repaired_frames": ";".join(str(f) for f in missing_for_sequence) if missing_for_sequence else "",
            }
        )
        rows.append({**meta, **feature_values})

    out = pd.DataFrame(rows)
    if report is not None:
        report.missing_required_frames[split_name] = int(missing_frames_total)
        report.repaired_frame_counts[split_name] = int(repaired_frames_total)
        report.repaired_sequence_counts[split_name] = int(repaired_sequences)
        report.max_missing_frames_per_sequence[split_name] = int(max_missing_for_sequence)
        report.invalid_bbox_frames[split_name] = int(invalid_frames)
        report.row_counts[split_name] = int(len(out))
        feature_cols = [c for c in out.columns if c not in METADATA_COLUMNS]
        report.nan_counts[split_name] = int(out[feature_cols].isna().sum().sum())
        report.feature_nan_counts[split_name] = {k: int(v) for k, v in out[feature_cols].isna().sum().items() if int(v) != 0}
    return out

def validate_bbox_feature_table(feature_df: pd.DataFrame, split_df: pd.DataFrame, split_name: str) -> list[str]:
    """Validate bbox feature table against one frozen split."""
    messages: list[str] = []
    if "sequence_id" not in feature_df.columns:
        raise ValueError(f"{split_name}: bbox feature table is missing sequence_id")
    if feature_df["sequence_id"].duplicated().any():
        raise ValueError(f"{split_name}: duplicate sequence_id in bbox feature table")
    if len(feature_df) != len(split_df):
        raise ValueError(f"{split_name}: row count mismatch bbox={len(feature_df)} split={len(split_df)}")
    split_ids = set(split_df["sequence_id"].astype(str))
    feature_ids = set(feature_df["sequence_id"].astype(str))
    if split_ids != feature_ids:
        raise ValueError(f"{split_name}: bbox feature sequence_ids do not match frozen split")
    labels = split_df[["sequence_id", "label"]].merge(feature_df[["sequence_id", "label"]], on="sequence_id", suffixes=("_split", "_bbox"), validate="one_to_one")
    if not (labels["label_split"].astype(int) == labels["label_bbox"].astype(int)).all():
        raise ValueError(f"{split_name}: bbox labels do not match frozen split")
    feature_cols = [c for c in feature_df.columns if c not in METADATA_COLUMNS]
    entirely_null = [c for c in feature_cols if feature_df[c].isna().all()]
    if entirely_null:
        raise ValueError(f"{split_name}: bbox feature columns entirely null: {entirely_null}")
    messages.append(f"{split_name}: bbox table row count, sequence_id coverage, labels, and non-empty feature columns validated")
    return messages
