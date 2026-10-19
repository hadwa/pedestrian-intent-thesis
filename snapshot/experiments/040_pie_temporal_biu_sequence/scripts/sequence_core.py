"""Frozen per-frame BIU/BBox/OBD sequence representation and TCN model."""

from __future__ import annotations

import random
import sys
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn


ACTIVE_ROOT = __import__("pathlib").Path(__file__).resolve().parents[3]
if str(ACTIVE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ACTIVE_ROOT / "src"))

from biu_extractor import extract_biu_features  # noqa: E402
from biu_v2_extractor import (  # noqa: E402
    KPT_INDEX,
    canonicalize_keypoints,
    estimate_direction,
)


OBSERVATION_LENGTH = 16
IMAGE_SIZE = (1920, 1080)
BIU_CHANNELS = (
    "torso_lean_angle", "body_yaw_angle", "head_yaw_angle", "stride_length",
    "lead_foot_elevation", "arm_swing_asymmetry", "hip_center_x", "knee_center_x",
    "ankle_center_x", "leg_center_x", "lead_ankle_forward", "stride_opening",
    "lower_body_motion_magnitude", "torso_center_x", "shoulder_center_x",
    "torso_motion_magnitude", "lower_body_conf", "torso_conf", "head_conf",
)
BBOX_CHANNELS = (
    "bbox_center_x_relative", "bbox_center_y_relative", "bbox_log_area_relative",
    "bbox_center_x_velocity", "bbox_center_y_velocity",
)
OBD_CHANNELS = ("obd_speed", "obd_speed_delta")
CHANNEL_NAMES = BIU_CHANNELS + BBOX_CHANNELS + OBD_CHANNELS


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_observations(
    coords: np.ndarray,
    confidence: np.ndarray,
    bboxes: np.ndarray,
    obd_speed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coords = np.asarray(coords, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    bboxes = np.asarray(bboxes, dtype=float)
    obd_speed = np.asarray(obd_speed, dtype=float)
    if (
        coords.shape != (OBSERVATION_LENGTH, 17, 2)
        or confidence.shape != (OBSERVATION_LENGTH, 17)
        or bboxes.shape != (OBSERVATION_LENGTH, 4)
        or obd_speed.shape != (OBSERVATION_LENGTH,)
    ):
        raise ValueError("exact 16-frame COCO-17 observations are required")
    if not all(np.isfinite(values).all() for values in (coords, confidence, bboxes, obd_speed)):
        raise ValueError("sequence observations must be finite")
    widths = bboxes[:, 2] - bboxes[:, 0]
    heights = bboxes[:, 3] - bboxes[:, 1]
    if np.any(widths <= 0) or np.any(heights <= 0):
        raise ValueError("BBox dimensions must be positive")
    return coords, confidence, bboxes, obd_speed


def build_sequence(
    coords: np.ndarray,
    confidence: np.ndarray,
    bboxes: np.ndarray,
    obd_speed: np.ndarray,
) -> np.ndarray:
    """Build one `[16, 26]` observation-only sequence from cached inputs."""
    coords, confidence, bboxes, obd_speed = _validate_observations(
        coords, confidence, bboxes, obd_speed
    )
    direction = estimate_direction(coords, bboxes)
    canonical = canonicalize_keypoints(coords, direction, image_size=IMAGE_SIZE)
    normalized = canonical.copy()
    normalized[:, :, 0] /= IMAGE_SIZE[0]
    normalized[:, :, 1] /= IMAGE_SIZE[1]
    relative = normalized - normalized[0:1]

    def center(names: tuple[str, str]) -> np.ndarray:
        return np.nanmean(relative[:, [KPT_INDEX[names[0]], KPT_INDEX[names[1]]], :], axis=1)

    hip = center(("left_hip", "right_hip"))
    knee = center(("left_knee", "right_knee"))
    ankle = center(("left_ankle", "right_ankle"))
    shoulder = center(("left_shoulder", "right_shoulder"))
    torso = np.nanmean(np.stack([hip, shoulder], axis=1), axis=1)
    leg = np.nanmean(np.stack([hip, knee, ankle], axis=1), axis=1)
    lead_ankle = np.maximum(
        relative[:, KPT_INDEX["left_ankle"], 0],
        relative[:, KPT_INDEX["right_ankle"], 0],
    )
    stride_opening = np.linalg.norm(
        relative[:, KPT_INDEX["left_ankle"], :]
        - relative[:, KPT_INDEX["right_ankle"], :], axis=1
    )
    lower_motion = np.linalg.norm(np.diff(leg, axis=0, prepend=leg[0:1]), axis=1)
    torso_motion = np.linalg.norm(np.diff(torso, axis=0, prepend=torso[0:1]), axis=1)
    lower_idx = [KPT_INDEX[name] for name in (
        "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"
    )]
    torso_idx = [KPT_INDEX[name] for name in (
        "left_shoulder", "right_shoulder", "left_hip", "right_hip"
    )]
    head_idx = [KPT_INDEX[name] for name in (
        "nose", "left_eye", "right_eye", "left_ear", "right_ear"
    )]
    anatomical = [extract_biu_features({
        name: canonical[t, index].astype(float) for name, index in KPT_INDEX.items()
    }) for t in range(OBSERVATION_LENGTH)]
    rows = np.column_stack([
        np.asarray([row["torso_lean_angle"] for row in anatomical]),
        np.asarray([row["body_yaw_angle"] for row in anatomical]),
        np.asarray([row["head_yaw_angle"] for row in anatomical]),
        np.asarray([row["stride_length"] for row in anatomical]),
        np.asarray([row["lead_foot_elevation"] for row in anatomical]),
        np.asarray([row["arm_swing_asymmetry"] for row in anatomical]),
        hip[:, 0], knee[:, 0], ankle[:, 0], leg[:, 0], lead_ankle, stride_opening,
        lower_motion, torso[:, 0], shoulder[:, 0], torso_motion,
        confidence[:, lower_idx].mean(axis=1), confidence[:, torso_idx].mean(axis=1),
        confidence[:, head_idx].mean(axis=1),
    ])
    centers = np.column_stack([
        (bboxes[:, 0] + bboxes[:, 2]) / 2.0,
        (bboxes[:, 1] + bboxes[:, 3]) / 2.0,
    ])
    widths = bboxes[:, 2] - bboxes[:, 0]
    heights = bboxes[:, 3] - bboxes[:, 1]
    areas = widths * heights
    bbox = np.column_stack([
        (centers[:, 0] - centers[0, 0]) / max(float(widths.mean()), 1e-6),
        (centers[:, 1] - centers[0, 1]) / max(float(heights.mean()), 1e-6),
        np.log(np.maximum(areas, 1e-6) / max(float(areas[0]), 1e-6)),
        np.diff(centers[:, 0], prepend=centers[0, 0]) / max(float(widths.mean()), 1e-6),
        np.diff(centers[:, 1], prepend=centers[0, 1]) / max(float(heights.mean()), 1e-6),
    ])
    obd = np.column_stack([obd_speed, obd_speed - obd_speed[0]])
    output = np.column_stack([rows, bbox, obd]).astype(np.float32)
    if output.shape != (OBSERVATION_LENGTH, len(CHANNEL_NAMES)) or not np.isfinite(output).all():
        raise ValueError("sequence representation is invalid")
    return output


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(value)


class TemporalConvClassifier(nn.Module):
    def __init__(self, input_channels: int, seed: int) -> None:
        seed_everything(seed)
        super().__init__()
        self.input_projection = nn.Conv1d(input_channels, 32, kernel_size=1)
        self.blocks = nn.Sequential(
            ResidualBlock(32, dilation=1, dropout=0.10),
            ResidualBlock(32, dilation=2, dropout=0.10),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.10), nn.Linear(32, 1)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3 or value.shape[2] != len(CHANNEL_NAMES):
            raise ValueError("expected [batch,16,sequence_channels]")
        transformed = self.blocks(self.input_projection(value.transpose(1, 2)))
        pooled = torch.cat([transformed.mean(dim=2), transformed[:, :, -1]], dim=1)
        return self.head(pooled).squeeze(1)


def loso_plans(frame: pd.DataFrame) -> tuple[tuple[str, tuple[str, ...]], ...]:
    required = {"sequence_id", "set_id", "label"}
    if required.difference(frame.columns) or frame.empty or frame["sequence_id"].duplicated().any():
        raise ValueError("LOSO frame identity is invalid")
    sets = tuple(sorted(frame["set_id"].astype(str).unique()))
    expected = ("set01", "set02", "set04", "set05", "set06")
    if sets != expected:
        raise ValueError("Exp040 requires the frozen five-set development universe")
    return tuple(
        (held_out, tuple(item for item in sets if item != held_out)) for held_out in sets
    )
