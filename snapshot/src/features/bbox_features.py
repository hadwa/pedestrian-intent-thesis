"""Bounding-box motion features aligned by frozen sequence_id."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def compute_bbox_motion_features(bboxes: np.ndarray) -> dict[str, float]:
    """Compute bbox motion features for one sequence.

    Args:
        bboxes: Array of shape [T, 4] in x1, y1, x2, y2 format.
    """
    arr = np.asarray(bboxes, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"Expected bboxes with shape [T, 4], got {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("Cannot compute bbox features for an empty sequence.")

    widths = arr[:, 2] - arr[:, 0]
    heights = arr[:, 3] - arr[:, 1]
    center_x = (arr[:, 0] + arr[:, 2]) / 2.0
    center_y = (arr[:, 1] + arr[:, 3]) / 2.0
    area = widths * heights
    centers = np.stack([center_x, center_y], axis=1)
    velocity = np.diff(centers, axis=0)
    acceleration = np.diff(velocity, axis=0) if len(velocity) > 1 else np.zeros((0, 2))

    return {
        "bbox_center_x_mean": float(center_x.mean()),
        "bbox_center_y_mean": float(center_y.mean()),
        "bbox_width_mean": float(widths.mean()),
        "bbox_height_mean": float(heights.mean()),
        "bbox_area_mean": float(area.mean()),
        "bbox_velocity_x_mean": float(velocity[:, 0].mean()) if len(velocity) else 0.0,
        "bbox_velocity_y_mean": float(velocity[:, 1].mean()) if len(velocity) else 0.0,
        "bbox_speed_mean": float(np.linalg.norm(velocity, axis=1).mean()) if len(velocity) else 0.0,
        "bbox_acceleration_x_mean": float(acceleration[:, 0].mean()) if len(acceleration) else 0.0,
        "bbox_acceleration_y_mean": float(acceleration[:, 1].mean()) if len(acceleration) else 0.0,
        "bbox_scale_change": float(area[-1] - area[0]),
        "bbox_displacement_x": float(center_x[-1] - center_x[0]),
        "bbox_displacement_y": float(center_y[-1] - center_y[0]),
    }


def load_bboxes_for_sequence(sequence_row: pd.Series, bbox_root: Path) -> np.ndarray:
    """Load [16, 4] bboxes for one frozen sequence.

    Adapt this function to the local bbox storage format. It must return the exact
    frames in the frozen sequence and must not alter sequence membership.
    """
    raise NotImplementedError("Adapt bbox loading to your processed PIE bbox files.")


def build_bbox_feature_table(sequence_csv: Path, bbox_root: Path, output_path: Path) -> pd.DataFrame:
    """Build bbox motion features for each frozen sequence, keyed by sequence_id."""
    sequences = pd.read_csv(sequence_csv)
    rows = []
    for _, row in sequences.iterrows():
        bboxes = load_bboxes_for_sequence(row, bbox_root)
        rows.append({"sequence_id": row["sequence_id"], **compute_bbox_motion_features(bboxes)})
    feature_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_path, index=False)
    return feature_df
