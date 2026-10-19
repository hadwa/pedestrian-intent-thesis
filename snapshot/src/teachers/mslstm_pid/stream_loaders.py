"""Frozen-row stream loaders for the adapted MSLSTM-style teacher."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import warnings
from typing import Any

import numpy as np
import pandas as pd


STATIC_SIGNALIZED_LABELS = ["n/a", "C", "S", "CS"]
STATIC_INTERSECTION_LABELS = ["midblock", "T", "T-left", "T-right", "four-way"]
STATIC_TRAFFIC_DIRECTION_LABELS = ["OW", "TW"]
STATIC_NUM_LANES_BUCKET_LABELS = ["1", "2", "3", "4", "5+"]
STATIC_NUM_LANES_BUCKETS = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 4, 7: 4, 8: 4}
TRAFFIC_LIGHT_STATE_LABELS = ["undefined", "red", "yellow", "green"]


def _load_pickle(path: Path) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with path.open("rb") as f:
            return pickle.load(f)


def load_pie_database(mslstm_root: Path) -> dict[str, Any]:
    return _load_pickle(mslstm_root / "PIE/annotations/data_cache/pie_database.pkl")


def video_key(set_id: str, video_id: str) -> str:
    return f"{set_id}_{video_id}"


def one_hot(index: int, size: int) -> list[int]:
    out = [0] * size
    out[max(0, min(size - 1, int(index)))] = 1
    return out


@dataclass
class RepairAttempt:
    repaired: bool
    value: np.ndarray | None
    method: str | None


class MSLSTMStreamRepository:
    """Read-only access to PIE-derived MSLSTM teacher artifacts."""

    def __init__(self, mslstm_root: Path):
        self.root = mslstm_root
        self.db = load_pie_database(mslstm_root)
        self._yolop_cache: dict[tuple[str, str], dict[int, dict[str, np.ndarray]] | None] = {}
        self._ped_cache: dict[tuple[str, str, str], tuple[dict[str, Any] | None, dict[str, Any] | None, dict[int, int] | None]] = {}
        self._traffic_light_cache: dict[tuple[str, str], dict[int, tuple[int, bool]]] = {}

    def track(self, set_id: str, video_id: str, ped_id: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        key = (set_id, video_id, ped_id)
        if key in self._ped_cache:
            vdb, pdb, _fmap = self._ped_cache[key]
            return vdb, pdb
        vdb = self.db.get(set_id, {}).get(video_id)
        if vdb is None:
            self._ped_cache[key] = (None, None, None)
            return None, None
        pdb = vdb.get("ped_annotations", {}).get(ped_id)
        fmap = self.frame_index(pdb) if pdb is not None else None
        self._ped_cache[key] = (vdb, pdb, fmap)
        return vdb, pdb

    @staticmethod
    def frame_index(pdb: dict[str, Any]) -> dict[int, int]:
        return {int(frame): idx for idx, frame in enumerate(pdb.get("frames", []))}

    def track_frame_index(self, set_id: str, video_id: str, ped_id: str) -> dict[int, int] | None:
        key = (set_id, video_id, ped_id)
        if key not in self._ped_cache:
            self.track(set_id, video_id, ped_id)
        _vdb, _pdb, fmap = self._ped_cache[key]
        return fmap

    def load_yolop_video(self, set_id: str, video_id: str) -> dict[int, dict[str, np.ndarray]] | None:
        key = (set_id, video_id)
        if key not in self._yolop_cache:
            path = self.root / "Extracted YOLOP" / f"{set_id}_{video_id}_yolop_features.pkl"
            if path.exists():
                self._yolop_cache[key] = _load_pickle(path).get(video_key(set_id, video_id), {})
            else:
                self._yolop_cache[key] = None
        return self._yolop_cache[key]

    def get_bbox(self, set_id: str, video_id: str, ped_id: str, frame_idx: int) -> np.ndarray | None:
        vdb, pdb = self.track(set_id, video_id, ped_id)
        if vdb is None or pdb is None:
            return None
        fmap = self.track_frame_index(set_id, video_id, ped_id)
        if fmap is None:
            return None
        idx = fmap.get(int(frame_idx))
        if idx is None:
            return None
        bbox = pdb.get("bbox", [])
        if idx >= len(bbox):
            return None
        arr = np.asarray(bbox[idx], dtype=float)
        if arr.shape != (4,):
            return None
        return arr

    def get_bbox_normalized(self, set_id: str, video_id: str, ped_id: str, frame_idx: int) -> np.ndarray | None:
        vdb, _pdb = self.track(set_id, video_id, ped_id)
        raw = self.get_bbox(set_id, video_id, ped_id, frame_idx)
        if vdb is None or raw is None:
            return None
        width = float(vdb.get("width", 0) or 0)
        height = float(vdb.get("height", 0) or 0)
        if width <= 0 or height <= 0:
            return None
        x1, y1, x2, y2 = raw
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        if not np.isfinite(w) or not np.isfinite(h) or w <= 0 or h <= 0:
            return None
        cx = ((x1 + x2) / 2.0) / width
        cy = ((y1 + y2) / 2.0) / height
        return np.asarray([cx, cy, w, h], dtype=float)

    def get_vehicle_field(self, set_id: str, video_id: str, frame_idx: int, keys: list[str]) -> np.ndarray | None:
        vdb = self.db.get(set_id, {}).get(video_id)
        if vdb is None:
            return None
        vehicle = vdb.get("vehicle_annotations", {}).get(int(frame_idx))
        if not isinstance(vehicle, dict):
            return None
        values: list[float] = []
        for key in keys:
            value = vehicle.get(key)
            if value is None:
                return None
            values.append(float(value))
        return np.asarray(values, dtype=float)

    def get_yolop(self, set_id: str, video_id: str, ped_id: str, frame_idx: int) -> np.ndarray | None:
        ydata = self.load_yolop_video(set_id, video_id)
        if ydata is None:
            return None
        value = ydata.get(int(frame_idx), {}).get(ped_id)
        if not isinstance(value, np.ndarray) or value.shape != (20,):
            return None
        return np.asarray(value, dtype=float)

    def get_static_scene_only(self, set_id: str, video_id: str, ped_id: str) -> dict[str, int] | None:
        _vdb, pdb = self.track(set_id, video_id, ped_id)
        if pdb is None:
            return None
        attr = pdb.get("attributes", {})
        required = ["signalized", "intersection", "traffic_direction", "num_lanes"]
        if not all(key in attr for key in required):
            return None
        return {
            "signalized": int(attr["signalized"]),
            "intersection": int(attr["intersection"]),
            "traffic_direction": int(attr["traffic_direction"]),
            "num_lanes_bucket": int(STATIC_NUM_LANES_BUCKETS.get(int(attr["num_lanes"]), 4)),
        }

    def get_traffic_light_state(self, set_id: str, video_id: str, frame_idx: int) -> tuple[int, bool]:
        """Return state index and whether any traffic-light object existed at the frame."""
        key = (set_id, video_id)
        if key not in self._traffic_light_cache:
            frame_map: dict[int, tuple[int, bool]] = {}
            vdb = self.db.get(set_id, {}).get(video_id)
            if vdb is not None:
                tldb = vdb.get("traffic_annotations", {})
                for obj in tldb.values():
                    if obj.get("obj_class") != "traffic_light":
                        continue
                    frames = obj.get("frames", [])
                    states = obj.get("state", [])
                    for idx, fr in enumerate(frames):
                        current_state, current_any = frame_map.get(int(fr), (0, False))
                        next_any = True
                        next_state = current_state
                        if idx < len(states) and int(states[idx]) != 0:
                            next_state = int(states[idx])
                        frame_map[int(fr)] = (next_state, current_any or next_any)
            self._traffic_light_cache[key] = frame_map
        state, any_object = self._traffic_light_cache[key].get(int(frame_idx), (0, False))
        return state, any_object

    def repair_series_one_frame(
        self,
        values: dict[int, np.ndarray],
        frame_indices: list[int],
        missing_frame: int,
    ) -> RepairAttempt:
        """Safe one-frame repair using observed-neighbor or nearest track frame."""
        prev_frame = missing_frame - 1
        next_frame = missing_frame + 1
        if prev_frame in values and next_frame in values:
            return RepairAttempt(True, (values[prev_frame] + values[next_frame]) / 2.0, "observed_neighbor_mean")
        if prev_frame in values:
            return RepairAttempt(True, values[prev_frame].copy(), "observed_previous")
        if next_frame in values:
            return RepairAttempt(True, values[next_frame].copy(), "observed_next")
        if not values:
            return RepairAttempt(False, None, None)
        nearest = min(values, key=lambda fr: abs(fr - missing_frame))
        if abs(nearest - missing_frame) == 1:
            return RepairAttempt(True, values[nearest].copy(), f"nearest_track_frame:{nearest}")
        return RepairAttempt(False, None, None)


def temporal_summary(
    values: np.ndarray,
    prefix: str,
    include_velocity: bool = False,
    include_slope: bool = True,
) -> dict[str, float]:
    """Summarize a [T, D] array with NaNs allowed."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    features: dict[str, float] = {}
    time_idx = np.arange(arr.shape[0], dtype=float)
    for dim in range(arr.shape[1]):
        series = arr[:, dim]
        valid = np.isfinite(series)
        name = f"{prefix}_{dim}" if arr.shape[1] > 1 else prefix
        features[f"{name}_first"] = float(series[0]) if np.isfinite(series[0]) else np.nan
        features[f"{name}_last"] = float(series[-1]) if np.isfinite(series[-1]) else np.nan
        features[f"{name}_mean"] = float(np.nanmean(series)) if valid.any() else np.nan
        features[f"{name}_std"] = float(np.nanstd(series)) if valid.any() else np.nan
        features[f"{name}_min"] = float(np.nanmin(series)) if valid.any() else np.nan
        features[f"{name}_max"] = float(np.nanmax(series)) if valid.any() else np.nan
        if np.isfinite(series[0]) and np.isfinite(series[-1]):
            features[f"{name}_delta"] = float(series[-1] - series[0])
        else:
            features[f"{name}_delta"] = np.nan
        if include_slope:
            if valid.sum() >= 2:
                x = time_idx[valid]
                y = series[valid]
                x_mean = float(np.mean(x))
                y_mean = float(np.mean(y))
                denom = float(np.sum((x - x_mean) ** 2))
                if denom > 0:
                    slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
                else:
                    slope = np.nan
                features[f"{name}_slope"] = slope
            else:
                features[f"{name}_slope"] = np.nan
        if include_velocity:
            velocity = np.diff(series)
            valid_v = np.isfinite(velocity)
            features[f"{name}_velocity_mean"] = float(np.nanmean(velocity)) if valid_v.any() else np.nan
            features[f"{name}_velocity_std"] = float(np.nanstd(velocity)) if valid_v.any() else np.nan
            features[f"{name}_velocity_min"] = float(np.nanmin(velocity)) if valid_v.any() else np.nan
            features[f"{name}_velocity_max"] = float(np.nanmax(velocity)) if valid_v.any() else np.nan
    return features


def make_static_scene_features(scene: dict[str, int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for idx, label in enumerate(STATIC_SIGNALIZED_LABELS):
        out[f"static_scene_signalized__{label}"] = int(scene["signalized"] == idx)
    for idx, label in enumerate(STATIC_INTERSECTION_LABELS):
        out[f"static_scene_intersection__{label}"] = int(scene["intersection"] == idx)
    for idx, label in enumerate(STATIC_TRAFFIC_DIRECTION_LABELS):
        out[f"static_scene_traffic_direction__{label}"] = int(scene["traffic_direction"] == idx)
    for idx, label in enumerate(STATIC_NUM_LANES_BUCKET_LABELS):
        out[f"static_scene_num_lanes_bucket__{label}"] = int(scene["num_lanes_bucket"] == idx)
    return out


def make_traffic_light_features(states: list[int], any_object_flags: list[bool]) -> dict[str, float]:
    arr = np.asarray(states, dtype=int)
    any_object = np.asarray(any_object_flags, dtype=int)
    out: dict[str, float] = {}
    for idx, label in enumerate(TRAFFIC_LIGHT_STATE_LABELS):
        out[f"traffic_light_fraction__{label}"] = float(np.mean(arr == idx))
        out[f"traffic_light_last__{label}"] = float(int(arr[-1] == idx))
    out["traffic_light_any_red"] = float(np.any(arr == 1))
    out["traffic_light_any_yellow"] = float(np.any(arr == 2))
    out["traffic_light_any_green"] = float(np.any(arr == 3))
    out["traffic_light_any_defined"] = float(np.any(arr > 0))
    out["traffic_light_any_object"] = float(np.any(any_object == 1))
    out["traffic_light_object_fraction"] = float(np.mean(any_object))
    out["traffic_light_no_object_fraction"] = float(np.mean(any_object == 0))
    out["traffic_light_undefined_with_object_fraction"] = float(np.mean((arr == 0) & (any_object == 1)))
    out["traffic_light_valid"] = 1.0
    return out


def fit_train_standardizer(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.Series, pd.Series]:
    mean = frame[columns].astype(float).mean(axis=0)
    std = frame[columns].astype(float).std(axis=0).replace(0, 1.0).fillna(1.0)
    return mean, std


def apply_standardizer(frame: pd.DataFrame, columns: list[str], mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    out = frame.copy()
    out[columns] = (out[columns].astype(float) - mean) / std
    return out
