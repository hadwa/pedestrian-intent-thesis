from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "scripts" / "exp055_core.py"


def load_core(name: str):
    spec = importlib.util.spec_from_file_location(name, CORE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_joint_schema_deduplicates_the_twenty_shared_biu_features() -> None:
    core = load_core("exp055_test_core_schema")
    v1 = tuple(f"v1_{index}" for index in range(72))
    shared = tuple(f"v1_{index}" for index in range(20))
    v2 = shared + tuple(f"v2_{index}" for index in range(36))

    schema = core.joint_unique_schema(v1, v2)

    assert len(schema) == 108
    assert schema[:72] == v1
    assert schema[72:] == tuple(f"v2_{index}" for index in range(36))


def test_exp012_biu_v2_core_schema_has_exactly_56_unique_names() -> None:
    core = load_core("exp055_test_core_v2_schema")

    assert len(core.BIU_V2_CORE_FEATURES) == 56
    assert len(set(core.BIU_V2_CORE_FEATURES)) == 56
    assert "lead_ankle_forward_delta" in core.BIU_V2_CORE_FEATURES
    assert "direction_confidence" not in core.BIU_V2_CORE_FEATURES


def test_bbox_motion8_preserves_frame_order_and_uses_zero_initial_delta() -> None:
    core = load_core("exp055_test_core_bbox_motion")
    cx = np.linspace(0.20, 0.35, 16, dtype=np.float32)
    bbox = np.column_stack(
        (
            cx,
            np.full(16, 0.50, dtype=np.float32),
            np.full(16, 0.10, dtype=np.float32),
            np.full(16, 0.20, dtype=np.float32),
        )
    )

    raw4 = core.bbox_temporal_features(bbox, "raw4")
    motion8 = core.bbox_temporal_features(bbox, "motion8")

    assert raw4.shape == (16, 4)
    assert motion8.shape == (16, 8)
    np.testing.assert_allclose(raw4, bbox)
    np.testing.assert_allclose(motion8[0, 4:], 0.0)
    assert motion8[1, 4] == pytest.approx(0.01)
    assert motion8[-1, 0] > motion8[0, 0]


def test_train_scaler_imputes_nonfinite_values_and_handles_constant_columns() -> None:
    core = load_core("exp055_test_core_scaler")
    train = np.asarray([[1.0, 4.0], [3.0, np.nan], [5.0, 4.0]], dtype=np.float64)

    scaler = core.fit_feature_scaler(train)
    transformed = core.transform_features(train, scaler)

    np.testing.assert_allclose(transformed[:, 0], [-1.2247449, 0.0, 1.2247449], rtol=1e-5)
    np.testing.assert_allclose(transformed[:, 1], 0.0)
    assert np.isfinite(transformed).all()
    assert scaler["std"][1] == 1.0


def test_biu_representation_selects_exact_v2_core_and_deduplicated_joint() -> None:
    core = load_core("exp055_test_core_biu_representation")
    v1_names = tuple(f"shared_{i}" for i in range(20)) + tuple(
        f"v1_{i}" for i in range(52)
    )
    v2_names = tuple(core.BIU_V2_CORE_FEATURES) + tuple(f"extra_{i}" for i in range(7))
    v1 = np.arange(2 * 72, dtype=np.float32).reshape(2, 72)
    v2 = np.arange(2 * 63, dtype=np.float32).reshape(2, 63)

    selected_v2, selected_v2_names = core.build_biu_representation(
        "v2_core", v1, v2, v1_names, v2_names
    )
    joint, joint_names = core.build_biu_representation(
        "joint_unique", v1, v2, v1_names, v2_names
    )

    assert selected_v2.shape == (2, 56)
    assert selected_v2_names == core.BIU_V2_CORE_FEATURES
    assert joint.shape == (2, 128)
    assert joint_names[:72] == v1_names
    assert joint_names[72:] == core.BIU_V2_CORE_FEATURES


def test_training_objective_includes_anchor_and_residual_terms() -> None:
    core = load_core("exp055_test_core_objective")
    output = {
        "final_logit": torch.tensor([0.0, 0.0]),
        "anchor_logit": torch.tensor([0.0, 0.0]),
        "applied_correction": torch.tensor([1.0, -1.0]),
    }
    labels = torch.tensor([0.0, 1.0])

    losses = core.training_objective(output, labels)

    expected_bce = torch.tensor(0.69314718)
    torch.testing.assert_close(losses["primary"], expected_bce)
    torch.testing.assert_close(losses["anchor"], expected_bce)
    torch.testing.assert_close(losses["residual"], torch.tensor(1.0))
    torch.testing.assert_close(
        losses["total"],
        expected_bce * 1.30 + torch.tensor(core.RESIDUAL_PENALTY_WEIGHT),
    )


@pytest.mark.parametrize("context_mode", ["none", "ego", "yolop", "ego_yolop"])
def test_model_forward_is_binary_compact_and_preserves_anchor_initially(context_mode: str) -> None:
    core = load_core(f"exp055_test_core_model_{context_mode}")
    config = core.ModelConfig("joint_unique", "motion8", context_mode)
    model = core.CompactResidualBiuLstm(config)
    inputs = {
        "summary": torch.randn(3, 161),
        "bbox_temporal": torch.randn(3, 16, 8),
    }
    if context_mode in {"ego", "ego_yolop"}:
        inputs["ego"] = torch.randn(3, 16, 3)
    if context_mode in {"yolop", "ego_yolop"}:
        inputs["yolop"] = torch.randn(3, 16, 20)

    output = model(inputs)

    assert output["final_logit"].shape == (3,)
    assert output["anchor_logit"].shape == (3,)
    assert output["gate"].shape == (3,)
    assert output["bounded_delta"].shape == (3,)
    torch.testing.assert_close(output["final_logit"], output["anchor_logit"])
    assert core.count_parameters(model) < 150_000
    if context_mode == "none":
        torch.testing.assert_close(output["gate"], torch.zeros(3))
    else:
        assert torch.all((output["gate"] > 0.0) & (output["gate"] < 0.5))


def test_model_rejects_a_stream_not_declared_by_its_context_mode() -> None:
    core = load_core("exp055_test_core_input_guard")
    model = core.CompactResidualBiuLstm(core.ModelConfig("v2_core", "summary_only", "none"))

    with pytest.raises(ValueError, match="Unexpected model inputs"):
        model({"summary": torch.zeros(2, 109), "ego": torch.zeros(2, 16, 3)})
