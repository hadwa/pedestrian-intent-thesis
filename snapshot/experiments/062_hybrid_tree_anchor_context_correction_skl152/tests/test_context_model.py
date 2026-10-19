"""Behavioral contracts for the fixed Exp062 context corrector."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.context_model import (  # noqa: E402
    AdditiveAttention,
    HybridContextCorrector,
    predict_context_corrector,
    train_one_seed,
)


def _force_large_correction(model: HybridContextCorrector) -> HybridContextCorrector:
    """Make the residual approach its declared bound without exceeding it."""
    with torch.no_grad():
        model.correction_head[-1].weight.zero_()
        model.correction_head[-1].bias.fill_(100.0)
        model.gate_head[2].weight.zero_()
        model.gate_head[2].bias.fill_(100.0)
    return model.eval()


def test_model_has_exact_trainable_parameter_count() -> None:
    """Catches an architecture that no longer matches the frozen 5,932 budget."""
    model = HybridContextCorrector()

    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 5932


def test_initial_forward_exactly_reproduces_anchor_probability() -> None:
    """Catches a nonzero residual or an anchor-logit conversion mismatch at initialization."""
    model = HybridContextCorrector().eval()
    base = torch.tensor([0.01, 0.5, 0.99])

    output = model(torch.zeros(3, 16, 3), torch.zeros(3, 16, 20), base)

    torch.testing.assert_close(torch.sigmoid(output.final_logit), base, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(output.applied_correction, torch.zeros(3))
    assert torch.all((output.gate > 0.0) & (output.gate < 0.5))


def test_applied_correction_is_bounded() -> None:
    """Catches a residual path that can move the anchor by more than 1.5 logits."""
    model = _force_large_correction(HybridContextCorrector())

    output = model(torch.zeros(4, 16, 3), torch.zeros(4, 16, 20), torch.full((4,), 0.5))

    assert torch.max(torch.abs(output.bounded_delta)).item() <= 1.5
    assert torch.max(torch.abs(output.applied_correction)).item() <= 1.5


def test_attention_weights_normalize_over_sixteen_frames() -> None:
    """Catches attention normalized over features or batches rather than observation time."""
    attention = AdditiveAttention(input_dim=16, attention_dim=8)

    _, weights = attention(torch.randn(3, 16, 16))

    assert weights.shape == (3, 16)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(3))


def test_forward_shapes_and_anchor_clipping_are_exact() -> None:
    """Catches changed output ABI or unclipped 0/1 anchor probabilities."""
    model = HybridContextCorrector().eval()
    base = torch.tensor([0.0, 1.0])

    output = model(torch.randn(2, 16, 3), torch.randn(2, 16, 20), base)

    for value in (
        output.final_logit,
        output.gate,
        output.bounded_delta,
        output.applied_correction,
    ):
        assert value.shape == (2,)
    expected = torch.logit(base.clamp(1e-6, 1.0 - 1e-6))
    torch.testing.assert_close(output.final_logit, expected, rtol=1e-6, atol=1e-6)


def test_reversing_nonconstant_ego_sequence_can_change_bilstm_context() -> None:
    """Catches an encoder that discards temporal order before context aggregation."""
    torch.manual_seed(17)
    model = HybridContextCorrector().eval()
    ego = torch.arange(48, dtype=torch.float32).reshape(1, 16, 3) / 10.0

    forward_context = model.ego_encoder(ego)
    reversed_context = model.ego_encoder(ego.flip(dims=(1,)))

    assert not torch.allclose(forward_context, reversed_context)


def test_prediction_returns_cpu_numpy_outputs_in_input_order() -> None:
    """Catches prediction helpers that shuffle rows or omit residual diagnostics."""
    model = HybridContextCorrector().eval()
    ego = np.zeros((3, 16, 3), dtype=np.float32)
    yolop = np.zeros((3, 16, 20), dtype=np.float32)
    base = np.array([0.2, 0.5, 0.8], dtype=np.float32)

    prediction = predict_context_corrector(model, ego, yolop, base, device="cpu")

    assert set(prediction) == {
        "probability", "final_logit", "gate", "bounded_delta", "applied_correction"
    }
    assert all(value.shape == (3,) for value in prediction.values())
    np.testing.assert_allclose(prediction["probability"], base, rtol=1e-6, atol=1e-7)


def test_training_rejects_nonfrozen_policy_values() -> None:
    """Catches an accidental hyperparameter search through the fixed training API."""
    values = np.zeros((4, 16, 3), dtype=np.float32)
    yolop = np.zeros((4, 16, 20), dtype=np.float32)
    anchor = np.full(4, 0.5, dtype=np.float32)
    labels = np.array([0, 1, 0, 1], dtype=np.int64)

    with pytest.raises(ValueError, match="exactly 60 epochs and patience 10"):
        train_one_seed(
            values,
            yolop,
            anchor,
            labels,
            values,
            yolop,
            anchor,
            labels,
            seed=42,
            device="cpu",
            max_epochs=1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_forward_runs_on_cuda_when_available() -> None:
    """Catches a CPU-created tensor in the forward path on CUDA-enabled runs."""
    model = HybridContextCorrector().cuda().eval()
    output = model(
        torch.zeros(2, 16, 3, device="cuda"),
        torch.zeros(2, 16, 20, device="cuda"),
        torch.tensor([0.2, 0.8], device="cuda"),
    )

    assert output.final_logit.device.type == "cuda"
    assert output.applied_correction.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_prediction_accepts_cuda_tensors_when_available() -> None:
    """Catches a prediction helper that converts CUDA inputs through NumPy."""
    model = HybridContextCorrector().cuda().eval()

    prediction = predict_context_corrector(
        model,
        torch.zeros(2, 16, 3, device="cuda"),
        torch.zeros(2, 16, 20, device="cuda"),
        torch.tensor([0.2, 0.8], device="cuda"),
        device="cuda",
    )

    assert prediction["probability"].shape == (2,)
