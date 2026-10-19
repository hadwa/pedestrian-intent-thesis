"""Authenticated data adapters for Exp084.

Only precomputed Exp050 observation inputs are consumed. Future event boxes are
read solely as auxiliary training targets from the exact PIE annotation frame.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP050 = REPO_ROOT / "experiments/050_mslstm_practical_behavior_replacement_pie_16f_tte30_60"
TRAINVAL_CACHE = EXP050 / "cache/trainval_base"
TEST_CACHE = EXP050 / "cache/set03_final"


class Population(NamedTuple):
    rows: pd.DataFrame
    bbox: np.ndarray
    ego: np.ndarray
    pose: np.ndarray
    biu_v1: np.ndarray
    biu_v2: np.ndarray
    biu_v1_names: tuple[str, ...]
    biu_v2_names: tuple[str, ...]
    source_dir: Path


class StandardizationStats(NamedTuple):
    mean: np.ndarray
    std: np.ndarray


def _center_bbox_to_xyxy(bbox: np.ndarray) -> np.ndarray:
    values = np.asarray(bbox, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (16, 4):
        raise ValueError(f"Expected BBox [N,16,4], got {values.shape}")
    cx, cy, width, height = np.moveaxis(values, -1, 0)
    return np.stack(
        [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2], axis=-1
    ).astype(np.float32)


def load_population(population: str, *, allow_test: bool) -> Population:
    if population == "trainval":
        source = TRAINVAL_CACHE
    elif population == "test":
        if not allow_test:
            raise PermissionError("set03 access requires allow_test=True during final evaluation")
        source = TEST_CACHE
    else:
        raise ValueError(f"Unknown population {population!r}")

    rows = pd.read_csv(source / "rows.csv")
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    bbox = _center_bbox_to_xyxy(np.load(source / "bbox.npy"))
    ego = np.asarray(np.load(source / "ego_speed.npy"), dtype=np.float32)
    pose = np.asarray(np.load(source / "pose.npy"), dtype=np.float32)
    biu_v1 = np.asarray(np.load(source / "biu_v1.npy"), dtype=np.float32)
    biu_v2 = np.asarray(np.load(source / "biu_v2.npy"), dtype=np.float32)

    expected = len(rows)
    arrays = {"bbox": bbox, "ego": ego, "pose": pose, "biu_v1": biu_v1, "biu_v2": biu_v2}
    for name, values in arrays.items():
        if len(values) != expected:
            raise ValueError(f"{name} row count {len(values)} != metadata row count {expected}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")
    if rows["sequence_id"].duplicated().any():
        raise ValueError("Population contains duplicate sequence IDs")
    if not (rows["observation_length"] == 16).all():
        raise ValueError("Population is not the frozen 16-frame cohort")
    if not rows["time_to_event_frames"].between(30, 60).all():
        raise ValueError("Population is not the frozen TTE30--60 cohort")

    v1_names = tuple(metadata["biu_v1_feature_names"])
    v2_names = tuple(metadata["biu_v2_feature_names"])
    if biu_v1.shape[1] != len(v1_names) or biu_v2.shape[1] != len(v2_names):
        raise ValueError("BIU array dimensions do not match authenticated feature names")

    return Population(rows, bbox, ego, pose, biu_v1, biu_v2, v1_names, v2_names, source)


def build_union(
    biu_v1: np.ndarray,
    biu_v2: np.ndarray,
    v1_names: Sequence[str],
    v2_names: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    v1 = np.asarray(biu_v1, dtype=np.float32)
    v2 = np.asarray(biu_v2, dtype=np.float32)
    if v1.ndim != 2 or v2.ndim != 2 or len(v1) != len(v2):
        raise ValueError("BIU union inputs must be aligned 2-D matrices")
    if v1.shape[1] != len(v1_names) or v2.shape[1] != len(v2_names):
        raise ValueError("BIU union feature-name dimensions do not match")
    names = tuple([f"v1::{name}" for name in v1_names] + [f"v2::{name}" for name in v2_names])
    if len(names) != len(set(names)):
        raise ValueError("Namespaced BIU union names are not unique")
    return np.concatenate([v1, v2], axis=1).astype(np.float32), names


def standardize_from_training(
    values: np.ndarray, train_mask: np.ndarray
) -> tuple[np.ndarray, StandardizationStats]:
    matrix = np.asarray(values, dtype=np.float32)
    mask = np.asarray(train_mask, dtype=bool)
    if matrix.ndim != 2 or mask.shape != (len(matrix),) or not mask.any():
        raise ValueError("Standardization requires a 2-D matrix and non-empty aligned train mask")
    mean = matrix[mask].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = matrix[mask].std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    transformed = ((matrix - mean) / std).astype(np.float32)
    if not np.isfinite(transformed).all():
        raise ValueError("Standardization produced non-finite values")
    return transformed, StandardizationStats(mean, std)


def _video_box_index(path: Path) -> dict[tuple[str, int], tuple[float, float, float, float]]:
    root = ET.parse(path).getroot()
    result: dict[tuple[str, int], tuple[float, float, float, float]] = {}
    for track in root.findall("./track"):
        if track.get("label") != "pedestrian":
            continue
        boxes = track.findall("./box")
        if not boxes:
            continue
        id_node = next(
            (node for node in boxes[0].findall("./attribute") if node.get("name") == "id"), None
        )
        if id_node is None or not id_node.text:
            raise ValueError(f"Pedestrian track in {path} has no ID")
        ped_id = id_node.text.strip()
        for box in boxes:
            if box.get("outside") == "1":
                continue
            key = (ped_id, int(box.attrib["frame"]))
            result[key] = (
                float(box.attrib["xtl"]),
                float(box.attrib["ytl"]),
                float(box.attrib["xbr"]),
                float(box.attrib["ybr"]),
            )
    return result


def build_event_bbox_targets(rows: pd.DataFrame) -> np.ndarray:
    required = {
        "sequence_id",
        "source_annotation_path",
        "ped_id",
        "event_frame",
        "frame_width",
        "frame_height",
    }
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Endpoint rows lack required columns: {sorted(missing)}")

    cache: dict[Path, dict[tuple[str, int], tuple[float, float, float, float]]] = {}
    targets = np.empty((len(rows), 4), dtype=np.float32)
    for position, row in enumerate(rows.itertuples(index=False)):
        path = Path(row.source_annotation_path)
        if path not in cache:
            cache[path] = _video_box_index(path)
        key = (str(row.ped_id), int(row.event_frame))
        if key not in cache[path]:
            raise ValueError(f"Exact event BBox missing for {row.sequence_id}: {key} in {path}")
        x1, y1, x2, y2 = cache[path][key]
        targets[position] = [
            x1 / float(row.frame_width),
            y1 / float(row.frame_height),
            x2 / float(row.frame_width),
            y2 / float(row.frame_height),
        ]
    if not np.isfinite(targets).all():
        raise ValueError("Endpoint target construction produced non-finite values")
    return targets
