from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "scripts" / "exp053_core.py"
RUNNER_PATH = ROOT / "scripts" / "run_experiment.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frozen_variants_include_v1_and_v2() -> None:
    core = load_module("exp053_test_core_variants", CORE_PATH)
    assert core.VARIANTS == ("G", "H", "I")
    assert core.VARIANT_STREAMS["G"] == ("bbox53", "ego_speed", "ego_acc", "yolop")
    assert core.VARIANT_STREAMS["H"][-1] == "biu_v1"
    assert core.VARIANT_STREAMS["I"][-1] == "biu_v2"
    assert core.RAW_CONTROL_VARIANTS == {"G": "C", "H": "F", "I": "E"}


def test_bbox53_projection_and_biu_projectors_have_frozen_dimensions() -> None:
    core = load_module("exp053_test_core_model", CORE_PATH)
    for variant in core.VARIANTS:
        model = core.MslstmBbox53Model(variant)
        inputs = {}
        for stream in core.VARIANT_STREAMS[variant]:
            dim = core.STREAM_DIMS[stream]
            shape = (2, dim) if stream in core.WINDOW_VECTOR_STREAMS else (2, 16, dim)
            inputs[stream] = torch.zeros(shape)
        assert model(inputs).shape == (2, 2)
        assert sum(p.numel() for p in model.vector_projections["bbox53"].parameters()) == 72960
        assert model.context_normalization.elementwise_affine is False
    assert sum(
        p.numel() for p in core.MslstmBbox53Model("H").vector_projections["biu_v1"].parameters()
    ) == 75392
    assert sum(
        p.numel() for p in core.MslstmBbox53Model("I").vector_projections["biu_v2"].parameters()
    ) == 74240


def test_bbox53_from_normalized_bbox_matches_direct_pixel_definition() -> None:
    core = load_module("exp053_test_core_features", CORE_PATH)
    center = np.linspace(0.25, 0.30, 16, dtype=np.float32)
    normalized = np.column_stack(
        [center, np.full(16, 0.5), np.full(16, 0.1), np.linspace(0.2, 0.3, 16)]
    ).astype(np.float32)

    values = core.bbox53_from_normalized_bbox(normalized, width=1000, height=500)

    assert values.shape == (53,)
    index = {name: i for i, name in enumerate(core.BBOX53_FEATURES)}
    assert values[index["bbox_center_x_first"]] == pytest.approx(250.0)
    assert values[index["bbox_center_x_last"]] == pytest.approx(300.0)
    assert values[index["center_x_displacement"]] == pytest.approx(50.0)
    assert values[index["bbox_valid_frame_count"]] == pytest.approx(16.0)


def test_train_only_scaler_handles_constant_feature() -> None:
    core = load_module("exp053_test_core_scaler", CORE_PATH)
    train = np.asarray([[1.0, 16.0], [3.0, 16.0]], dtype=np.float64)
    scaler = core.fit_bbox53_scaler(train)
    transformed = core.transform_bbox53(train, scaler)
    np.testing.assert_allclose(transformed[:, 0], [-1.0, 1.0])
    np.testing.assert_allclose(transformed[:, 1], [0.0, 0.0])
    assert scaler["std"][1] == 1.0


def test_runner_requires_nine_new_runs_and_maps_three_raw_controls() -> None:
    runner = load_module("exp053_test_runner_grid", RUNNER_PATH)
    assert len(runner.required_training_identities()) == 9
    assert runner.core.RAW_CONTROL_VARIANTS == {"G": "C", "H": "F", "I": "E"}
    assert runner.BASE_CACHE == runner.EXP050_ROOT / "cache" / "trainval_base"


def test_evaluation_gate_rejects_incomplete_grid(tmp_path: Path) -> None:
    runner = load_module("exp053_test_runner_gate", RUNNER_PATH)
    with pytest.raises(RuntimeError, match="nine authenticated checkpoints"):
        runner.assert_training_complete(tmp_path)
