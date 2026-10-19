from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "scripts" / "exp051_core.py"
RUNNER_PATH = ROOT / "scripts" / "run_experiment.py"
EXP050_CORE_PATH = (
    ROOT.parents[0]
    / "050_mslstm_practical_behavior_replacement_pie_16f_tte30_60"
    / "scripts"
    / "exp050_core.py"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parameter_free_normalization_removes_context_scale_and_offset() -> None:
    core = load_module("exp051_test_core_scale", CORE_PATH)
    pattern = torch.arange(core.STREAM_CONTEXT_DIM, dtype=torch.float32).view(1, -1)
    contexts = [pattern, pattern * 100.0 + 500.0]

    fused, normalized = core.normalize_and_fuse_contexts(
        contexts,
        torch.zeros(2),
    )

    torch.testing.assert_close(normalized[:, 0], normalized[:, 1], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(fused, normalized[:, 0], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(normalized.mean(dim=2), torch.zeros((1, 2)), atol=2e-6, rtol=0)
    torch.testing.assert_close(normalized.var(dim=2, unbiased=False), torch.ones((1, 2)), atol=2e-5, rtol=2e-5)


def test_normalized_fusion_preserves_exp050_parameter_counts() -> None:
    core = load_module("exp051_test_core_params", CORE_PATH)
    exp050 = load_module("exp050_reference_core_params", EXP050_CORE_PATH)

    for variant in core.VARIANTS:
        normalized = core.MslstmPracticalModel(variant)
        reference = exp050.MslstmPracticalModel(variant)
        assert core.count_parameters(normalized) == exp050.count_parameters(reference)
        assert normalized.context_normalization.elementwise_affine is False
        assert list(normalized.context_normalization.parameters()) == []


def test_all_variants_use_normalized_fusion_and_keep_output_shape() -> None:
    core = load_module("exp051_test_core_forward", CORE_PATH)
    for variant in core.VARIANTS:
        model = core.MslstmPracticalModel(variant)
        inputs = {}
        for stream in core.VARIANT_STREAMS[variant]:
            dim = core.STREAM_DIMS[stream]
            shape = (2, dim) if stream.startswith("biu_") else (2, core.SEQ_LEN, dim)
            inputs[stream] = torch.zeros(shape)
        assert model(inputs).shape == (2, 2)


def test_runner_keeps_frozen_18_run_grid_and_reuses_exp050_cache() -> None:
    runner = load_module("exp051_test_runner_grid", RUNNER_PATH)
    assert len(runner.base.required_training_identities()) == 18
    assert runner.base.BASE_CACHE == runner.EXP050_ROOT / "cache" / "trainval_base"
    assert runner.base.TEST_CACHE == runner.EXP050_ROOT / "cache" / "set03_final"
    assert runner.base.ROOT == ROOT


def test_evaluation_gate_rejects_incomplete_training(tmp_path: Path) -> None:
    runner = load_module("exp051_test_runner_gate", RUNNER_PATH)
    with pytest.raises(RuntimeError, match="18 authenticated checkpoints"):
        runner.base.assert_training_complete(tmp_path)


def test_exp050_paired_deltas_match_variant_and_seed() -> None:
    runner = load_module("exp051_test_runner_deltas", RUNNER_PATH)
    baseline = pd.DataFrame(
        [
            {"variant": "A", "seed": 42, "accuracy": 0.75, "precision": 0.70, "recall": 0.70, "f1": 0.70, "roc_auc": 0.85, "pr_auc": 0.80},
            {"variant": "A", "seed": 43, "accuracy": 0.76, "precision": 0.72, "recall": 0.72, "f1": 0.72, "roc_auc": 0.86, "pr_auc": 0.82},
        ]
    )
    normalized = pd.DataFrame(
        [
            {"variant": "A", "seed": 43, "accuracy": 0.77, "precision": 0.73, "recall": 0.73, "f1": 0.73, "roc_auc": 0.87, "pr_auc": 0.84},
            {"variant": "A", "seed": 42, "accuracy": 0.74, "precision": 0.71, "recall": 0.71, "f1": 0.71, "roc_auc": 0.84, "pr_auc": 0.79},
        ]
    )

    deltas = runner.exp050_paired_deltas(normalized, baseline)

    assert list(deltas[["variant", "seed"]].itertuples(index=False, name=None)) == [
        ("A", 42),
        ("A", 43),
    ]
    assert deltas["f1_delta"].tolist() == pytest.approx([0.01, 0.01])
    assert deltas["pr_auc_delta"].tolist() == pytest.approx([-0.01, 0.02])


def test_report_rewrite_identifies_exp051_evaluation_without_renaming_baseline() -> None:
    runner = load_module("exp051_test_runner_report", RUNNER_PATH)
    source = (
        "# Exp050 Final Report\n\n"
        "The set03 result is the designated Exp050 final evaluation. "
        "set03 had prior independent post-hoc exposure, but Exp050 was frozen before this evaluation.\n"
        "\n## Exp050 Paired Comparison\n"
    )

    rewritten = runner.rewrite_report_identity(source)

    assert rewritten.startswith("# Exp051 Final Report")
    assert "designated Exp051 final evaluation" in rewritten
    assert "Exp051 was frozen before this evaluation" in rewritten
    assert "## Exp050 Paired Comparison" in rewritten
