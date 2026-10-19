"""Keypoint formatting utilities aligned by frozen sequence_id."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def format_keypoints_for_model(keypoints: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Convert raw keypoints to a model-ready array.

    Args:
        keypoints: Array shaped [T, K, C].
        normalize: If true, center and scale coordinates per sequence using visible
            keypoint statistics. Fit any learned scaler outside this function using
            training data only.
    """
    arr = np.asarray(keypoints, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"Expected keypoints with shape [T, K, C], got {arr.shape}")
    if not normalize:
        return arr
    coords = arr[..., :2]
    center = np.nanmean(coords.reshape(-1, 2), axis=0)
    scale = np.nanstd(coords.reshape(-1, 2), axis=0)
    scale[scale == 0] = 1.0
    out = arr.copy()
    out[..., :2] = (coords - center) / scale
    return out


def flatten_keypoints(keypoints: np.ndarray) -> dict[str, float]:
    """Flatten [T, K, C] keypoints into a tabular feature dict."""
    arr = np.asarray(keypoints, dtype=float)
    flat = arr.reshape(-1)
    return {f"keypoint_{i:04d}": float(value) for i, value in enumerate(flat)}


def load_keypoints_for_sequence(sequence_row: pd.Series, keypoint_root: Path) -> np.ndarray:
    """Load raw keypoints for one frozen sequence.

    Adapt this function to the local pose output format. The returned sequence must
    correspond exactly to start_frame..end_frame from the frozen CSV row.
    """
    raise NotImplementedError("Adapt keypoint loading to your processed PIE keypoint files.")


def build_keypoint_feature_table(sequence_csv: Path, keypoint_root: Path, output_path: Path) -> pd.DataFrame:
    """Build flattened keypoint features keyed by sequence_id."""
    sequences = pd.read_csv(sequence_csv)
    rows = []
    for _, row in sequences.iterrows():
        keypoints = load_keypoints_for_sequence(row, keypoint_root)
        formatted = format_keypoints_for_model(keypoints)
        rows.append({"sequence_id": row["sequence_id"], **flatten_keypoints(formatted)})
    feature_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_path, index=False)
    return feature_df
