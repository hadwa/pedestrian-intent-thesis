"""Executable contracts for the Exp062 foundation."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.contracts import (  # noqa: E402
    ExperimentConfig,
    MetricBundle,
    compute_binary_metrics,
    evaluate_validation_gate,
    harmonic_f1_pr,
    sha256_file,
    write_authenticated_candidate_manifest,
    write_authenticated_final_freeze,
    write_authenticated_json_marker,
)


def aggregate(
    *,
    f1: float,
    pr_auc: float,
    roc_auc: float,
    harmonic: float,
    high_confidence_errors: int,
    per_seed_harmonic: list[float],
    invariants_pass: bool = True,
) -> MetricBundle:
    return MetricBundle(
        f1=f1,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        harmonic=harmonic,
        high_confidence_errors=high_confidence_errors,
        per_seed_harmonic=tuple(per_seed_harmonic),
        invariants_pass=invariants_pass,
    )


def test_frozen_config_matches_approved_spec() -> None:
    """Catches changed cohorts, source hashes, dimensions, or threshold policy."""
    cfg = ExperimentConfig()

    assert cfg.experiment_number == 62
    assert cfg.observed_frames == 16
    assert cfg.tte == "30_60"
    assert cfg.train_cohorts == ("set01", "set02", "set06")
    assert cfg.val_cohorts == ("set04", "set05")
    assert cfg.test_cohorts == ("set03",)
    assert cfg.train_rows == 2366
    assert cfg.val_rows == 3619
    assert cfg.test_rows == 3760
    assert cfg.seeds == (42, 43, 44)
    assert cfg.threshold == 0.5
    assert cfg.rf_feature_dim == 125
    assert cfg.xgb_feature_dim == 109
    assert cfg.rf_v1_bbox_feature_blocks == (
        ("enriched_bbox", 53),
        ("biu_v1_with_quality_counters", 72),
    )
    assert cfg.xgb_v2_bbox_feature_blocks == (
        ("enriched_bbox", 53),
        ("biu_v2_core", 56),
    )
    assert cfg.ego_shape == (16, 3)
    assert cfg.yolop_shape == (16, 20)
    assert cfg.context_parameter_count == 5932
    assert cfg.correction_bound == 1.5
    assert cfg.parity_tolerance == 1e-7
    assert cfg.trainval_rows_sha256 == (
        "a01ad1f3244157c1b649aada0470ad54d877bbd1e8075c275ee313c35dcc3bf1"
    )
    assert cfg.ego_trainval_sha256 == (
        "410c4757e4012e223721f5686bffe5f8e4c4f53dd9fd9030e763fcfbe58dd302"
    )
    assert cfg.yolop_trainval_sha256 == (
        "b353bf9d8fb7d5eedacf726fec643f67138994f5e9a864806f46e382075b65f5"
    )


def test_binary_metrics_apply_fixed_threshold_and_documented_definitions() -> None:
    """Catches accidental threshold selection or an incorrect metric computation."""
    metrics = compute_binary_metrics(
        labels=[0, 0, 1, 1], probabilities=[0.1, 0.6, 0.7, 0.8]
    )

    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(0.8)
    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.pr_auc == pytest.approx(1.0)
    assert metrics.brier == pytest.approx(0.125)
    assert metrics.tp == 2
    assert metrics.tn == 1
    assert metrics.fp == 1
    assert metrics.fn == 0
    assert metrics.high_confidence_errors == 0
    assert metrics.harmonic == pytest.approx(harmonic_f1_pr(0.8, 1.0))


def test_binary_metrics_reject_non_frozen_thresholds() -> None:
    """Catches a threshold-search escape hatch in the fixed-0.5 experiment."""
    with pytest.raises(ValueError, match="fixed 0.5"):
        compute_binary_metrics(labels=[0, 1], probabilities=[0.2, 0.8], threshold=.4)


def test_validation_gate_is_strict_and_anchor_relative() -> None:
    """Catches permissive frozen thresholds or comparisons against the wrong anchor."""
    anchor = aggregate(
        f1=.710,
        pr_auc=.750,
        roc_auc=.903,
        harmonic=.729,
        high_confidence_errors=20,
        per_seed_harmonic=[.72, .73, .74],
    )
    candidate = aggregate(
        f1=.716,
        pr_auc=.755,
        roc_auc=.904,
        harmonic=.735,
        high_confidence_errors=19,
        per_seed_harmonic=[.726, .736, .746],
    )

    decision = evaluate_validation_gate(candidate, anchor)

    assert decision.eligible
    assert decision.failed_conditions == ()


def test_validation_gate_rejects_equal_frozen_bound_and_failed_invariants() -> None:
    """Catches replacement of strict gates with inclusive comparisons or ignored audits."""
    anchor = aggregate(
        f1=.710,
        pr_auc=.750,
        roc_auc=.903,
        harmonic=.729,
        high_confidence_errors=20,
        per_seed_harmonic=[.72, .73, .74],
    )
    candidate = aggregate(
        f1=.7075263341950917,
        pr_auc=.755,
        roc_auc=.904,
        harmonic=.735,
        high_confidence_errors=19,
        per_seed_harmonic=[.726, .736, .746],
        invariants_pass=False,
    )

    decision = evaluate_validation_gate(candidate, anchor)

    assert not decision.eligible
    assert "mean_f1_gt_0.7075263341950917" in decision.failed_conditions
    assert "invariants_pass" in decision.failed_conditions


def test_validation_gate_requires_all_three_frozen_seed_summaries() -> None:
    """Catches a gate decision calculated from fewer than the three approved seeds."""
    summary = aggregate(
        f1=.716,
        pr_auc=.755,
        roc_auc=.904,
        harmonic=.735,
        high_confidence_errors=19,
        per_seed_harmonic=[.726, .736],
    )

    with pytest.raises(ValueError, match="three frozen seed"):
        evaluate_validation_gate(summary, summary)


def test_authenticated_marker_hashes_config_manifest_command_and_inputs(tmp_path: Path) -> None:
    """Catches mutable or unauthenticated phase-marker provenance."""
    input_artifact = tmp_path / "input.txt"
    input_artifact.write_text("frozen input\n", encoding="utf-8")
    marker_path = tmp_path / "phase.json"
    source_manifest = {"rows.csv": "expected-row-hash"}

    payload = write_authenticated_json_marker(
        marker_path,
        config=ExperimentConfig(),
        source_manifest=source_manifest,
        producing_command="python scripts/run_experiment.py prepare",
        input_artifacts={"input.txt": input_artifact},
        payload={"phase": "prepare"},
    )

    assert json.loads(marker_path.read_text(encoding="utf-8")) == payload
    assert payload["input_artifacts"] == {"input.txt": sha256_file(input_artifact)}
    assert payload["producing_command_sha256"] == hashlib.sha256(
        b"python scripts/run_experiment.py prepare"
    ).hexdigest()
    assert len(payload["config_sha256"]) == 64
    assert len(payload["source_manifest_sha256"]) == 64


def test_authenticated_final_freeze_rejects_a_failed_validation_gate(tmp_path: Path) -> None:
    """Catches nominal final freezes that have not passed all ten frozen gates."""
    rejected = aggregate(
        f1=.700,
        pr_auc=.700,
        roc_auc=.700,
        harmonic=.700,
        high_confidence_errors=21,
        per_seed_harmonic=[.70, .70, .70],
    )

    with pytest.raises(ValueError, match="eligible validation gate"):
        write_authenticated_final_freeze(
            tmp_path / "FROZEN_FINAL_MODEL.json",
            config=ExperimentConfig(),
            source_manifest={"rows.csv": "expected-row-hash"},
            producing_command="python scripts/run_experiment.py freeze-or-close",
            winning_candidate="rf_v1_bbox",
            validation_summary=rejected,
            anchor_summary=rejected,
            checkpoint_artifacts={},
            validation_artifacts={},
            candidate_manifest=tmp_path / "candidate.json",
        )


def test_authenticated_final_freeze_binds_gate_candidate_and_checkpoints(
    tmp_path: Path,
) -> None:
    """Catches a final marker lacking the verified candidate or seed checkpoints."""
    anchor = aggregate(
        f1=.710,
        pr_auc=.750,
        roc_auc=.903,
        harmonic=.729,
        high_confidence_errors=20,
        per_seed_harmonic=[.72, .73, .74],
    )
    candidate = aggregate(
        f1=.716,
        pr_auc=.755,
        roc_auc=.904,
        harmonic=.735,
        high_confidence_errors=19,
        per_seed_harmonic=[.726, .736, .746],
    )
    checkpoints = {}
    for seed in ExperimentConfig().seeds:
        checkpoint = tmp_path / f"seed{seed}.pt"
        checkpoint.write_text(f"checkpoint {seed}\n", encoding="utf-8")
        checkpoints[seed] = checkpoint
    validation_artifact = tmp_path / "validation_summary_results.csv"
    validation_artifact.write_text("frozen validation summary\n", encoding="utf-8")
    candidate_manifest = tmp_path / "candidate.json"
    manifest = write_authenticated_candidate_manifest(
        candidate_manifest,
        config=ExperimentConfig(),
        source_manifest={"rows.csv": "expected-row-hash"},
        winning_candidate="rf_v1_bbox",
        validation_summary=candidate,
        anchor_summary=anchor,
        checkpoint_artifacts=checkpoints,
        validation_artifacts={"metrics/validation_summary_results.csv": validation_artifact},
    )

    payload = write_authenticated_final_freeze(
        tmp_path / "FROZEN_FINAL_MODEL.json",
        config=ExperimentConfig(),
        source_manifest={"rows.csv": "expected-row-hash"},
        producing_command="python scripts/run_experiment.py freeze-or-close",
        winning_candidate="rf_v1_bbox",
        validation_summary=candidate,
        anchor_summary=anchor,
        checkpoint_artifacts=checkpoints,
        validation_artifacts={"metrics/validation_summary_results.csv": validation_artifact},
        candidate_manifest=candidate_manifest,
    )

    assert payload["marker_schema"] == "exp062-authenticated-final-freeze-v1"
    assert payload["final_freeze"]["winning_candidate"] == "rf_v1_bbox"
    assert payload["final_freeze"]["validation_gate"] == {
        "eligible": True,
        "failed_conditions": [],
    }
    assert len(payload["final_freeze"]["validation_summary_sha256"]) == 64
    assert len(payload["final_freeze"]["anchor_summary_sha256"]) == 64
    assert payload["final_freeze"]["winning_checkpoint_sha256"] == {
        str(seed): sha256_file(path) for seed, path in checkpoints.items()
    }
    assert payload["final_freeze"]["candidate_manifest_sha256"] == sha256_file(
        candidate_manifest
    )
    assert manifest["validation_summary"]["f1"] == pytest.approx(.716)
    assert manifest["anchor_summary"]["harmonic"] == pytest.approx(.729)
    assert manifest["validation_artifacts"] == {
        "metrics/validation_summary_results.csv": sha256_file(validation_artifact)
    }
    assert payload["input_artifacts"]["checkpoint_seed42"] == sha256_file(checkpoints[42])
    assert json.loads((tmp_path / "FROZEN_FINAL_MODEL.json").read_text(encoding="utf-8")) == payload


def test_authenticated_final_freeze_rejects_inputs_not_bound_by_candidate_manifest(
    tmp_path: Path,
) -> None:
    """Catches a gate summary or checkpoint substituted after candidate freeze."""
    anchor = aggregate(
        f1=.710,
        pr_auc=.750,
        roc_auc=.903,
        harmonic=.729,
        high_confidence_errors=20,
        per_seed_harmonic=[.72, .73, .74],
    )
    candidate = aggregate(
        f1=.716,
        pr_auc=.755,
        roc_auc=.904,
        harmonic=.735,
        high_confidence_errors=19,
        per_seed_harmonic=[.726, .736, .746],
    )
    checkpoints = {}
    for seed in ExperimentConfig().seeds:
        checkpoint = tmp_path / f"seed{seed}.pt"
        checkpoint.write_text(f"checkpoint {seed}\n", encoding="utf-8")
        checkpoints[seed] = checkpoint
    validation_artifact = tmp_path / "validation_summary_results.csv"
    validation_artifact.write_text("frozen validation summary\n", encoding="utf-8")
    candidate_manifest = tmp_path / "candidate.json"
    write_authenticated_candidate_manifest(
        candidate_manifest,
        config=ExperimentConfig(),
        source_manifest={"rows.csv": "expected-row-hash"},
        winning_candidate="rf_v1_bbox",
        validation_summary=candidate,
        anchor_summary=anchor,
        checkpoint_artifacts=checkpoints,
        validation_artifacts={"metrics/validation_summary_results.csv": validation_artifact},
    )

    altered_candidate = aggregate(
        f1=.717,
        pr_auc=.755,
        roc_auc=.904,
        harmonic=.736,
        high_confidence_errors=19,
        per_seed_harmonic=[.727, .737, .747],
    )
    with pytest.raises(ValueError, match="candidate manifest does not match"):
        write_authenticated_final_freeze(
            tmp_path / "FROZEN_FINAL_MODEL.json",
            config=ExperimentConfig(),
            source_manifest={"rows.csv": "expected-row-hash"},
            producing_command="python scripts/run_experiment.py freeze-or-close",
            winning_candidate="rf_v1_bbox",
            validation_summary=altered_candidate,
            anchor_summary=anchor,
            checkpoint_artifacts=checkpoints,
            validation_artifacts={"metrics/validation_summary_results.csv": validation_artifact},
            candidate_manifest=candidate_manifest,
        )

    checkpoints[42].write_text("substituted checkpoint\n", encoding="utf-8")
    with pytest.raises(ValueError, match="candidate manifest does not match"):
        write_authenticated_final_freeze(
            tmp_path / "FROZEN_FINAL_MODEL.json",
            config=ExperimentConfig(),
            source_manifest={"rows.csv": "expected-row-hash"},
            producing_command="python scripts/run_experiment.py freeze-or-close",
            winning_candidate="rf_v1_bbox",
            validation_summary=candidate,
            anchor_summary=anchor,
            checkpoint_artifacts=checkpoints,
            validation_artifacts={"metrics/validation_summary_results.csv": validation_artifact},
            candidate_manifest=candidate_manifest,
        )
