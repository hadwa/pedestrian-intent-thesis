from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "scripts" / "exp095_core.py"
RUNNER_PATH = ROOT / "scripts" / "run_experiment.py"


def load_module(name: str, path: Path):
    assert path.is_file(), f"Expected experiment module is missing: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frozen_grid_contains_only_the_three_missing_controls() -> None:
    core = load_module("exp095_test_core_grid", CORE_PATH)
    assert core.VARIANTS == ("J", "K", "L")
    assert core.VARIANT_STREAMS["J"] == (
        "bbox53", "ego_speed", "ego_acc", "yolop", "pose"
    )
    assert core.VARIANT_STREAMS["K"] == (
        "bbox53", "ped_action", "ped_look", "ego_speed", "ego_acc", "yolop"
    )
    assert core.VARIANT_STREAMS["L"] == (
        "bbox53", "ego_speed", "ego_acc", "yolop", "biu_combined"
    )
    assert "static_context" not in core.VARIANT_STREAMS["K"]


def test_combined_biu_retains_all_namespaced_v1_and_v2_values() -> None:
    core = load_module("exp095_test_core_union", CORE_PATH)
    arrays = {
        "biu_v1": np.arange(2 * 72, dtype=np.float32).reshape(2, 72),
        "biu_v2": (1000 + np.arange(2 * 63, dtype=np.float32)).reshape(2, 63),
    }
    combined = core.combined_biu(arrays["biu_v1"], arrays["biu_v2"])
    assert combined.shape == (2, 135)
    np.testing.assert_array_equal(combined[:, :72], arrays["biu_v1"])
    np.testing.assert_array_equal(combined[:, 72:], arrays["biu_v2"])
    assert len(core.BIU_COMBINED_FEATURE_NAMES) == 135
    assert all(name.startswith("v1__") for name in core.BIU_COMBINED_FEATURE_NAMES[:72])
    assert all(name.startswith("v2__") for name in core.BIU_COMBINED_FEATURE_NAMES[72:])


def test_all_models_accept_their_frozen_stream_shapes() -> None:
    core = load_module("exp095_test_core_models", CORE_PATH)
    for variant in core.VARIANTS:
        model = core.MslstmBbox53ControlModel(variant)
        inputs = {}
        for stream in core.VARIANT_STREAMS[variant]:
            dim = core.STREAM_DIMS[stream]
            shape = (2, dim) if stream in core.WINDOW_VECTOR_STREAMS else (2, 16, dim)
            inputs[stream] = torch.zeros(shape)
        assert model(inputs).shape == (2, 2)
        assert model.context_normalization.elementwise_affine is False
    union_projection = core.MslstmBbox53ControlModel("L").vector_projections["biu_combined"]
    assert sum(parameter.numel() for parameter in union_projection.parameters()) == 83456


def test_runner_requires_nine_new_checkpoints_and_has_frozen_comparisons() -> None:
    runner = load_module("exp095_test_runner_grid", RUNNER_PATH)
    assert len(runner.required_training_identities()) == 9
    assert runner.COMPARISONS == (
        ("G", "J"),
        ("G", "K"),
        ("G", "L"),
        ("J", "H"),
        ("J", "I"),
        ("J", "L"),
        ("K", "H"),
        ("K", "I"),
        ("K", "L"),
        ("H", "L"),
        ("I", "L"),
    )


def test_evaluation_gate_rejects_incomplete_grid(tmp_path: Path) -> None:
    runner = load_module("exp095_test_runner_gate", RUNNER_PATH)
    with pytest.raises(RuntimeError, match="nine authenticated checkpoints"):
        runner.assert_training_complete(tmp_path)
