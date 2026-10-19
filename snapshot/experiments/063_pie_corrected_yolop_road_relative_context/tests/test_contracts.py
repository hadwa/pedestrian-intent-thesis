from __future__ import annotations

import importlib
from pathlib import Path

import pytest


contracts = importlib.import_module(
    "experiments.063_pie_corrected_yolop_road_relative_context.scripts.contracts"
)


def test_config_forbids_set03_in_development() -> None:
    config = contracts.ExperimentConfig()
    assert config.train_cohorts == ("set01", "set02", "set06")
    assert config.val_cohorts == ("set04", "set05")
    assert config.test_cohorts == ("set03",)
    assert "set03" not in config.development_cohorts
    assert config.fresh_run_count == 12


def test_channel_abis_are_exactly_20d() -> None:
    assert len(contracts.CORRECTED_GRID_CHANNELS) == 20
    assert contracts.CORRECTED_GRID_CHANNELS[:2] == (
        "drivable_cell_0",
        "drivable_cell_1",
    )
    assert contracts.CORRECTED_GRID_CHANNELS[-2:] == (
        "vehicle_count",
        "pedestrian_count",
    )
    assert len(contracts.ROAD_RELATIVE_CHANNELS) == 20
    assert contracts.ROAD_RELATIVE_CHANNELS[-1] == "road_signed_distance_change_h"
    assert len(set(contracts.ROAD_RELATIVE_CHANNELS)) == 20


@pytest.mark.parametrize(
    "path",
    (
        Path("cache/set03/frame.npz"),
        Path("/tmp/authenticated_set03_rows.csv"),
        Path("SET03/video_0001.mp4"),
    ),
)
def test_path_guard_rejects_set03_before_io(path: Path) -> None:
    with pytest.raises(ValueError, match="set03"):
        contracts.assert_development_path(path)


@pytest.mark.parametrize("set_id", ("set01", "set02", "set04", "set05", "set06"))
def test_development_cohort_guard_accepts_only_frozen_sets(set_id: str) -> None:
    assert contracts.assert_development_cohort(set_id) == set_id


@pytest.mark.parametrize("set_id", ("set03", "set00", "train"))
def test_development_cohort_guard_fails_closed(set_id: str) -> None:
    with pytest.raises(ValueError, match="forbids cohort"):
        contracts.assert_development_cohort(set_id)
