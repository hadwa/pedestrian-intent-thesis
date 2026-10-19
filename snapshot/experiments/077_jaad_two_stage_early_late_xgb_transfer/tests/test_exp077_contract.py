from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_experiment.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("exp077_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_portable_contract_excludes_ego_and_preserves_expected_widths() -> None:
    runner = _load_runner()
    contract = runner.feature_contract()

    assert len(contract["joint"]) == 161
    assert len(contract["portable"]) == 182
    assert all("obd" not in name for name in contract["portable"])


def test_jaad_split_contract_uses_the_frozen_30_60_population() -> None:
    runner = _load_runner()
    train = runner.load_jaad_split("train")

    assert len(train) == 1552
    assert train["track_id"].nunique() == 154
    assert set(train["set_id"]) == {"jaad"}
    assert train["observation_length"].eq(16).all()
    assert train["tte_frames"].between(30, 60).all()


def test_historical_biu_mismatches_must_equal_documented_repair_population() -> None:
    runner = _load_runner()
    runner.assert_expected_mismatch_ids({"a", "b"}, {"a", "b"}, family="biu")

    with pytest.raises(RuntimeError, match="unexpected biu mismatch"):
        runner.assert_expected_mismatch_ids({"a", "c"}, {"a", "b"}, family="biu")
