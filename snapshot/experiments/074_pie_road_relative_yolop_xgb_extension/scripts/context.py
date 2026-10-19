"""Authenticated Exp063 road-relative cache and frozen Exp074 summaries."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import contract  # noqa: E402


ROAD_CHANNELS = (
    "road_foot_inside", "road_signed_boundary_distance_h", "road_nearest_boundary_dx_h",
    "road_nearest_boundary_dy_h", "road_boundary_tangent_x", "road_boundary_tangent_y",
    "road_boundary_valid", "lane_nearest_distance_h", "lane_nearest_dx_h",
    "lane_nearest_dy_h", "lane_tangent_x", "lane_tangent_y", "lane_geometry_valid",
    "foot_velocity_x_h", "foot_velocity_y_h", "foot_speed_h", "motion_parallel_to_road",
    "motion_toward_drivable_area", "motion_parallel_to_lane", "road_signed_distance_change_h",
)


def context_columns() -> tuple[str, ...]:
    return tuple(
        name
        for channel in ROAD_CHANNELS
        for name in (
            f"road__{channel}__early_mean",
            f"road__{channel}__late_mean",
            f"road__{channel}__late_minus_early",
        )
    )


def summarize_road(sequence: np.ndarray) -> np.ndarray:
    values = np.asarray(sequence, dtype=float)
    if values.shape != (16, 20):
        raise ValueError("road-relative sequence must be exactly 16x20")
    if not np.isfinite(values).all():
        raise ValueError("road-relative sequence must be finite")
    early = values[:8].mean(axis=0)
    late = values[8:].mean(axis=0)
    return np.column_stack([early, late, late - early]).reshape(-1)


def _load_part(path: Path, expected_hash: str) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file() or contract.sha256_file(path) != expected_hash:
        raise RuntimeError(f"Exp063 context part authentication failed: {path}")
    with np.load(path, allow_pickle=False) as bundle:
        ids = np.asarray(bundle["sequence_id"]).astype(str)
        values = np.asarray(bundle["road_relative"], dtype=np.float32)
    if values.shape != (len(ids), 16, 20) or not np.isfinite(values).all():
        raise RuntimeError(f"invalid Exp063 road-relative part: {path}")
    return ids, values


def load_development_context() -> tuple[pd.DataFrame, dict[str, Any]]:
    contract.load_registration()
    for name in ("exp063_complete", "exp063_representations", "exp063_contracts", "exp063_extract", "exp063_yolop"):
        contract.resolve_source(name)
    manifest_path = contract.resolve_source("exp063_manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("set03_accessed") is not False or manifest.get("target_labels_passed_to_perception") is not False:
        raise RuntimeError("Exp063 development cache is not label-sealed")
    part_root = manifest_path.parents[1] / "cache/trainval_parts"
    id_parts = []
    value_parts = []
    part_records = []
    for record in manifest["video_parts"]:
        path = part_root / f"{record['set_id']}_{record['video_id']}.npz"
        ids, values = _load_part(path, str(record["sha256"]))
        if len(ids) != int(record["rows"]):
            raise RuntimeError(f"Exp063 manifest row count changed: {path}")
        id_parts.append(ids)
        value_parts.append(values)
        part_records.append({"path": str(path), **contract.artifact_record(path), "rows": len(ids)})
    ids = np.concatenate(id_parts)
    values = np.concatenate(value_parts)
    if len(ids) != 5985 or len(set(ids.tolist())) != len(ids):
        raise RuntimeError("Exp063 development sequence population changed")
    summarized = np.stack([summarize_road(sequence) for sequence in values])
    frame = pd.DataFrame(summarized, columns=context_columns())
    frame.insert(0, "sequence_id", ids)
    audit = {
        "rows": len(frame), "parts": len(part_records), "part_records": part_records,
        "all_finite": bool(np.isfinite(summarized).all()), "set03_accessed": False,
        "road_geometry_valid_frame_fraction": float(values[:, :, 6].mean()),
        "lane_geometry_valid_frame_fraction": float(values[:, :, 12].mean()),
        "context_nonzero_fraction": {
            column: float(np.count_nonzero(summarized[:, index]) / len(summarized))
            for index, column in enumerate(context_columns())
        },
        "upstream_yolop": {
            key: manifest[key]
            for key in ("device", "elapsed_seconds", "inference_batch_size", "unique_frames")
        },
    }
    return frame, audit


def benchmark_summarizer(sequence: np.ndarray, *, warmups: int = 100, repetitions: int = 5000) -> dict[str, float | int]:
    for _ in range(int(warmups)):
        summarize_road(sequence)
    elapsed = np.empty(int(repetitions), dtype=float)
    for index in range(int(repetitions)):
        started = time.perf_counter_ns()
        summarize_road(sequence)
        elapsed[index] = (time.perf_counter_ns() - started) / 1_000_000.0
    return {
        "warmups": int(warmups), "repetitions": int(repetitions),
        "mean_ms": float(elapsed.mean()), "median_ms": float(np.median(elapsed)),
        "p95_ms": float(np.quantile(elapsed, .95)),
    }
