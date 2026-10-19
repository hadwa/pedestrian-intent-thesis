import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("psi114_preflight", ROOT / "scripts/preflight.py")
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)
RUN_SPEC = importlib.util.spec_from_file_location("psi114_run", ROOT / "run.py")
RUN = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(RUN)


def test_frozen_feature_dimensions_and_prefix():
    contract = PREFLIGHT.load_feature_contract()
    assert len(contract["S187"]) == 187
    assert len(contract["R247"]) == 247
    assert contract["R247"][:187] == contract["S187"]


def test_speed_blocker_is_exact_and_development_only():
    assert PREFLIGHT.MISSING_PSI2_SPEED_VIDEOS == (
        "video_0032",
        "video_0055",
        "video_0093",
    )
    assert "test" not in {"train", "val"}


def test_amended_cohort_is_exact_and_matched():
    populations = RUN.rows()
    assert len(populations["train"]) == 26295
    assert len(populations["val"]) == 14922
    assert not set(populations["train"].video_id.astype(str)) & RUN.EXCLUDED
    assert not set(populations["val"].video_id.astype(str)) & RUN.EXCLUDED


def test_speed_summary_uses_exact_fifteen_values():
    values = RUN.speed_values({index: float(index) for index in range(15)}, list(range(15)))
    np.testing.assert_allclose(values, [7.0, np.std(np.arange(15)), 0.0, 14.0, 14.0])


def test_road_contract_preserves_source_prefix():
    contract = PREFLIGHT.load_feature_contract()
    assert contract["R247"][:187] == contract["S187"]
    assert all(name.startswith("road__") for name in contract["R247"][187:])
