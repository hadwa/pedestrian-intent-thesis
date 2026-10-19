"""Immutable scientific and source contracts for Exp063."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


CORRECTED_GRID_CHANNELS = tuple(
    [f"drivable_cell_{index}" for index in range(9)]
    + [f"lane_cell_{index}" for index in range(9)]
    + ["vehicle_count", "pedestrian_count"]
)

ROAD_RELATIVE_CHANNELS = (
    "road_foot_inside",
    "road_signed_boundary_distance_h",
    "road_nearest_boundary_dx_h",
    "road_nearest_boundary_dy_h",
    "road_boundary_tangent_x",
    "road_boundary_tangent_y",
    "road_boundary_valid",
    "lane_nearest_distance_h",
    "lane_nearest_dx_h",
    "lane_nearest_dy_h",
    "lane_tangent_x",
    "lane_tangent_y",
    "lane_geometry_valid",
    "foot_velocity_x_h",
    "foot_velocity_y_h",
    "foot_speed_h",
    "motion_parallel_to_road",
    "motion_toward_drivable_area",
    "motion_parallel_to_lane",
    "road_signed_distance_change_h",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """The approved Exp063 protocol; runtime code must not alter these fields."""

    experiment_number: int = 63
    observed_frames: int = 16
    tte: str = "30_60"
    train_cohorts: tuple[str, ...] = ("set01", "set02", "set06")
    val_cohorts: tuple[str, ...] = ("set04", "set05")
    test_cohorts: tuple[str, ...] = ("set03",)
    train_rows: int = 2366
    val_rows: int = 3619
    representations: tuple[str, ...] = ("corrected_grid", "road_relative")
    anchor_families: tuple[str, ...] = ("rf_v1_bbox", "xgb_v2_bbox")
    seeds: tuple[int, ...] = (42, 43, 44)
    threshold: float = 0.5
    scene_shape: tuple[int, int] = (16, 20)
    correction_parameter_count: int = 5932
    yolop_repository_commit: str = "8d8f68df318c71f01d6f813c024df646c7d1978f"
    yolop_checkpoint_sha256: str = (
        "620e7495328cfc8e04807c14cbb8f8df9916953359d190cbf494b2d049f03b0c"
    )
    yolop_notebook_sha256: str = (
        "2310285c26a88e9acc5aa019e1d301e9435b58bd8b69e7c2d9292951575f8b2f"
    )
    yolop_demo_sha256: str = (
        "3a5a2faad6799589793b05fe59a7ecc9406f203af7bedc78cfc5ee1f5ba4971a"
    )
    exp062_rows_sha256: str = (
        "a01ad1f3244157c1b649aada0470ad54d877bbd1e8075c275ee313c35dcc3bf1"
    )
    exp062_ego_sha256: str = (
        "410c4757e4012e223721f5686bffe5f8e4c4f53dd9fd9030e763fcfbe58dd302"
    )
    exp062_model_source_sha256: str = (
        "492b31e3aea2bead1d48310916fb98a6f8c5ced4eac7d267ca35c9e7292aef2d"
    )

    @property
    def development_cohorts(self) -> tuple[str, ...]:
        return tuple(sorted(self.train_cohorts + self.val_cohorts))

    @property
    def fresh_run_count(self) -> int:
        return len(self.representations) * len(self.anchor_families) * len(self.seeds)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash an artifact without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_development_path(path: Path | str) -> Path:
    """Fail before any development-phase operation can touch a set03 path."""
    candidate = Path(path)
    if any("set03" in part.lower() for part in candidate.parts):
        raise ValueError(f"Exp063 development forbids set03 access: {candidate}")
    return candidate


def assert_development_cohort(set_id: str) -> str:
    """Accept only the five frozen train/validation PIE sets."""
    if str(set_id) not in ExperimentConfig().development_cohorts:
        raise ValueError(f"Exp063 development forbids cohort: {set_id}")
    return str(set_id)


__all__ = [
    "CORRECTED_GRID_CHANNELS",
    "ROAD_RELATIVE_CHANNELS",
    "ExperimentConfig",
    "assert_development_cohort",
    "assert_development_path",
    "sha256_file",
]
