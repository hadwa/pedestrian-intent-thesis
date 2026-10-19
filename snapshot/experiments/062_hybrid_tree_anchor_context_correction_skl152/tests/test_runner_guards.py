"""Phase-order, source-isolation, and leakage-boundary tests for Exp062."""

from __future__ import annotations

import json
import builtins
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.anchors import (  # noqa: E402
    FoldModelArtifact,
    GroupedOOFResult,
    anchor_feature_metadata,
)
from scripts.contracts import (  # noqa: E402
    AnchorComplexity,
    ExperimentConfig,
    MetricBundle,
    canonical_json_sha256,
    compute_binary_metrics,
    evaluate_validation_gate,
    sha256_file,
    write_authenticated_candidate_manifest,
    write_authenticated_final_freeze,
    write_authenticated_json_marker,
)
from scripts.runner import (  # noqa: E402
    TEST_SOURCE,
    ArtifactAuthenticationError,
    ForbiddenSourceError,
    PhaseOrderError,
    PhaseState,
    ProbabilityBundleSpec,
    VerificationError,
    authenticate_probability_bundle,
    build_parser,
    final,
    verify,
    persist_probability_bundle,
    run_authenticated_training,
    select_validation_candidate,
    write_validation_outputs,
    _summary_bundle,
    _verify_final_bundle,
    build_efficiency_rows,
)


def _summary(*, harmonic: float = 0.735, errors: int = 19) -> MetricBundle:
    return MetricBundle(
        f1=0.716,
        pr_auc=0.755,
        roc_auc=0.904,
        harmonic=harmonic,
        high_confidence_errors=errors,
        per_seed_harmonic=(0.726, 0.736, 0.746),
    )


def _anchor_summary() -> MetricBundle:
    return MetricBundle(
        f1=0.710,
        pr_auc=0.750,
        roc_auc=0.903,
        harmonic=0.729,
        high_confidence_errors=20,
        per_seed_harmonic=(0.720, 0.730, 0.740),
    )


def _write_staging_fixture(root: Path, config: ExperimentConfig) -> Path:
    from scripts.contracts import canonical_json_sha256, experiment_config_sha256

    staged = root / "staged_sources"
    staged.mkdir(parents=True, exist_ok=True)
    log = root / "logs" / "stage-sources.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("fixture staging\n", encoding="utf-8")
    paths = {
        name: staged / name
        for name in ("rows.csv", "summary_v1.npy", "summary_v2_core.npy", "ego.npy", "yolop.npy")
    }
    for anchor in config.anchor_families:
        for seed in config.seeds:
            name = f"validation_reference_{anchor}_seed{seed}.csv"
            paths[name] = staged / name
    for name, path in paths.items():
        path.write_text(f"fixture {name}\n", encoding="utf-8")
    paths["stage-sources.log"] = log
    marker = root / "SOURCE_STAGING.json"
    marker.write_text(
        json.dumps(
            {
                "schema": "exp062-source-staging-v1",
                "config_sha256": experiment_config_sha256(config),
                "extraction_rules": {"fixture": "isolated"},
                "original_sources": {},
                "ordered_sequence_ids_sha256": canonical_json_sha256([]),
                "ordered_labels_sha256": canonical_json_sha256([]),
                "train_rows": config.train_rows,
                "validation_rows": config.val_rows,
                "derived_artifacts": {
                    name: {
                        "path": str(path.relative_to(root)),
                        "sha256": sha256_file(path),
                    }
                    for name, path in paths.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return marker


def _test_candidate_selection_payload(
    candidates: dict[str, tuple[MetricBundle, MetricBundle]],
    validation_artifacts: dict[str, Path],
) -> dict:
    selection = select_validation_candidate(candidates)

    def summary_digest(bundle: MetricBundle) -> str:
        return canonical_json_sha256(
            {
                "f1": bundle.f1,
                "pr_auc": bundle.pr_auc,
                "roc_auc": bundle.roc_auc,
                "harmonic": bundle.harmonic,
                "high_confidence_errors": float(bundle.high_confidence_errors),
                "per_seed_harmonic": list(bundle.per_seed_harmonic),
                "invariants_pass": bundle.invariants_pass,
            }
        )

    families = {}
    for family, (candidate, anchor) in candidates.items():
        prediction_hashes = {
            str(seed): sha256_file(
                validation_artifacts[f"predictions/validation_{family}_seed{seed}.csv"]
            )
            for seed in ExperimentConfig().seeds
        }
        decision = evaluate_validation_gate(candidate, anchor)
        families[family] = {
            "candidate_summary_sha256": summary_digest(candidate),
            "anchor_summary_sha256": summary_digest(anchor),
            "validation_prediction_sha256": prediction_hashes,
            "validation_prediction_bundle_sha256": canonical_json_sha256(
                prediction_hashes
            ),
            "validation_gate": {
                "eligible": decision.eligible,
                "failed_conditions": list(decision.failed_conditions),
            },
        }
    return {
        "schema": "exp062-candidate-selection-v1",
        "winner": selection.winner,
        "eligible": list(selection.eligible),
        "families": families,
    }


def _write_valid_freeze(
    root: Path,
    *,
    malformed_summary: bool = False,
    staged_source_manifest: bool = False,
    declared_winner: str = "rf_v1_bbox",
) -> tuple[Path, dict[int, Path]]:
    config = ExperimentConfig()
    source_manifest = _write_staging_fixture(root, config)
    checkpoints: dict[int, Path] = {}
    anchor_models: dict[int, Path] = {}
    validation_artifacts: dict[str, Path] = {}
    for seed in config.seeds:
        checkpoint = root / "checkpoints" / f"{declared_winner}_seed{seed}.pt"
        anchor = root / "models" / f"{declared_winner}_seed{seed}.joblib"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        anchor.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(f"checkpoint {seed}\n", encoding="utf-8")
        anchor.write_text(f"anchor {seed}\n", encoding="utf-8")
        checkpoints[seed] = checkpoint
        anchor_models[seed] = anchor
    summary_path = root / "metrics" / "validation_summary_results.csv"
    per_seed_path = root / "metrics" / "validation_per_seed_results.csv"
    scaler_path = root / "metrics" / "context_scaler.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.arange(config.val_rows, dtype=np.int64) % 2
    sequence_ids = np.asarray([f"v{index:04d}" for index in range(config.val_rows)])
    per_seed_records: list[dict] = []
    summary_records: list[dict] = []
    candidates: dict[str, tuple[MetricBundle, MetricBundle]] = {}
    for family_index, family in enumerate(config.anchor_families):
        anchor_probabilities = np.where(labels == 1, 0.8, 0.2).astype(np.float64)
        anchor_flip_count = 120 + family_index * 20
        anchor_probabilities[:anchor_flip_count] = np.where(
            labels[:anchor_flip_count] == 1, 0.4, 0.6
        )
        corrected_probabilities = np.where(labels == 1, 0.9, 0.1).astype(np.float64)
        if family == "xgb_v2_bbox":
            corrected_probabilities[:2] = np.where(labels[:2] == 1, 0.4, 0.6)
        family_metrics: dict[str, list[MetricBundle]] = {"anchor": [], "corrected": []}
        for seed in config.seeds:
            prediction = root / "predictions" / f"validation_{family}_seed{seed}.csv"
            prediction.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "sequence_id": sequence_ids,
                    "label": labels,
                    "anchor_probability": anchor_probabilities,
                    "corrected_probability": corrected_probabilities,
                    "corrected_class_fixed_0p5": (
                        corrected_probabilities >= config.threshold
                    ).astype(np.int64),
                }
            ).to_csv(prediction, index=False)
            validation_artifacts[f"predictions/{prediction.name}"] = prediction
            for variant, probabilities in (
                ("anchor", anchor_probabilities),
                ("corrected", corrected_probabilities),
            ):
                metric = compute_binary_metrics(labels, probabilities)
                family_metrics[variant].append(metric)
                record = {
                    "anchor_name": family,
                    "seed": seed,
                    "variant": variant,
                    "checkpoint_epoch": 1,
                    "invariants_pass": True,
                    **asdict(metric),
                }
                record["per_seed_harmonic"] = json.dumps(
                    record["per_seed_harmonic"]
                )
                per_seed_records.append(record)
        family_summaries: dict[str, MetricBundle] = {}
        for variant, metrics in family_metrics.items():
            mean_values = {
                name: float(np.mean([getattr(metric, name) for metric in metrics]))
                for name in (
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "pr_auc",
                    "harmonic",
                    "bce",
                    "brier",
                    "high_confidence_errors",
                )
            }
            bundle = MetricBundle(
                **mean_values,
                per_seed_harmonic=tuple(metric.harmonic for metric in metrics),
                invariants_pass=True,
            )
            family_summaries[variant] = bundle
            summary_records.append(
                {
                    "anchor_name": family,
                    "variant": variant,
                    "seed_count": len(config.seeds),
                    **mean_values,
                    "per_seed_harmonic": json.dumps(bundle.per_seed_harmonic),
                    "invariants_pass": True,
                }
            )
        candidates[family] = (
            family_summaries["corrected"],
            family_summaries["anchor"],
        )
    pd.DataFrame(per_seed_records).to_csv(per_seed_path, index=False)
    persisted_per_seed = pd.read_csv(per_seed_path)
    summary_records = []
    candidates = {}
    for family in config.anchor_families:
        family_summaries = {}
        for variant in ("anchor", "corrected"):
            group = persisted_per_seed.loc[
                (persisted_per_seed["anchor_name"] == family)
                & (persisted_per_seed["variant"] == variant)
            ].sort_values("seed")
            mean_values = {
                name: float(group[name].mean())
                for name in (
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "pr_auc",
                    "harmonic",
                    "bce",
                    "brier",
                    "high_confidence_errors",
                )
            }
            bundle = MetricBundle(
                **mean_values,
                per_seed_harmonic=tuple(group["harmonic"].astype(float)),
                invariants_pass=True,
            )
            family_summaries[variant] = bundle
            summary_records.append(
                {
                    "anchor_name": family,
                    "variant": variant,
                    "seed_count": len(config.seeds),
                    **mean_values,
                    "per_seed_harmonic": json.dumps(bundle.per_seed_harmonic),
                    "invariants_pass": True,
                }
            )
        candidates[family] = (
            family_summaries["corrected"],
            family_summaries["anchor"],
        )
    pd.DataFrame(summary_records).to_csv(summary_path, index=False)
    if malformed_summary:
        summary_path.write_text("candidate,metric\nrf_v1_bbox,valid\n", encoding="utf-8")
    scaler_path.write_text("{}\n", encoding="utf-8")
    validation_artifacts["metrics/validation_summary_results.csv"] = summary_path
    validation_artifacts["metrics/validation_per_seed_results.csv"] = per_seed_path
    validation_artifacts["metrics/context_scaler.json"] = scaler_path
    for name in (
        "paired_deltas.csv",
        "error_transitions_fixed_0p5.csv",
        "context_ablation_diagnostics.csv",
    ):
        path = root / "metrics" / name
        path.write_text(f"fixture,{name}\n", encoding="utf-8")
        validation_artifacts[f"metrics/{name}"] = path
    for seed, anchor in anchor_models.items():
        validation_artifacts[f"models/anchor_seed{seed}"] = anchor
    validation_evidence = root / "validation-evidence.txt"
    validation_evidence.write_text("validated\n", encoding="utf-8")
    freeze_log = root / "logs" / "freeze.log"
    freeze_log.parent.mkdir(parents=True, exist_ok=True)
    freeze_log.write_text("frozen\n", encoding="utf-8")
    validation_artifacts["logs/freeze.log"] = freeze_log
    validation_marker = root / "VALIDATION_COMPLETE.json"
    prepare_evidence = root / "prepare-evidence.txt"
    prepare_evidence.write_text("prepared\n", encoding="utf-8")
    write_authenticated_json_marker(
        root / "PREPARED.json",
        config=config,
        source_manifest=source_manifest,
        producing_command="runner.py prepare",
        input_artifacts={
            "SOURCE_STAGING.json": source_manifest,
            "prepare-evidence": prepare_evidence,
        },
        payload={"phase": "prepare"},
    )
    write_authenticated_json_marker(
        root / "ANCHORS_READY.json",
        config=config,
        source_manifest=source_manifest,
        producing_command="runner.py fit-anchors",
        input_artifacts={"PREPARED.json": root / "PREPARED.json"},
        payload={"phase": "fit-anchors"},
    )
    training_artifacts = {"ANCHORS_READY.json": root / "ANCHORS_READY.json"}
    for family in config.anchor_families:
        for seed in config.seeds:
            name = f"TRAINED__{family}__seed{seed}.json"
            path = root / name
            write_authenticated_json_marker(
                path,
                config=config,
                source_manifest=source_manifest,
                producing_command=f"runner.py train --anchor {family} --seed {seed}",
                input_artifacts={"ANCHORS_READY.json": root / "ANCHORS_READY.json"},
                payload={"anchor_name": family, "seed": seed},
            )
            training_artifacts[name] = path
    write_authenticated_json_marker(
        root / "TRAINING_COMPLETE.json",
        config=config,
        source_manifest=source_manifest,
        producing_command="runner.py train",
        input_artifacts=training_artifacts,
        payload={"phase": "train"},
    )
    write_authenticated_json_marker(
        validation_marker,
        config=config,
        source_manifest=source_manifest,
        producing_command="runner.py validate",
        input_artifacts={
            "TRAINING_COMPLETE.json": root / "TRAINING_COMPLETE.json",
            "validation-evidence": validation_evidence,
        },
        payload={"phase": "validate"},
    )
    validation_artifacts["VALIDATION_COMPLETE.json"] = validation_marker
    manifest = root / "FROZEN_CANDIDATE_MANIFEST.json"
    write_authenticated_candidate_manifest(
        manifest,
        config=config,
        source_manifest=source_manifest,
        winning_candidate=declared_winner,
        validation_summary=candidates[declared_winner][0],
        anchor_summary=candidates[declared_winner][1],
        checkpoint_artifacts=checkpoints,
        validation_artifacts=validation_artifacts,
    )
    freeze = root / "FROZEN_FINAL_MODEL.json"
    write_authenticated_final_freeze(
        freeze,
        config=config,
        source_manifest=source_manifest,
        producing_command=f"runner.py freeze --winner {declared_winner}",
        winning_candidate=declared_winner,
        validation_summary=candidates[declared_winner][0],
        anchor_summary=candidates[declared_winner][1],
        checkpoint_artifacts=checkpoints,
        validation_artifacts=validation_artifacts,
        candidate_manifest=manifest,
        payload={
            "anchor_sha256": {str(seed): sha256_file(path) for seed, path in anchor_models.items()},
            "anchor_paths": {str(seed): str(path.resolve()) for seed, path in anchor_models.items()},
            "scaler_sha256": sha256_file(scaler_path),
            "scaler_path": str(scaler_path.resolve()),
            "validation_summary_file_sha256": sha256_file(summary_path),
            "final_command_arguments": ["final", "--threshold", "0.5"],
            "candidate_selection": {
                **_test_candidate_selection_payload(candidates, validation_artifacts),
                "winner": declared_winner,
            },
        },
    )
    return freeze, checkpoints


def test_pre_freeze_phases_reject_test_source_access(tmp_path: Path) -> None:
    """Catches any pre-freeze branch that resolves or opens Set03."""
    state = PhaseState(tmp_path)
    for phase in ("prepare", "fit-anchors", "train", "validate", "freeze"):
        with pytest.raises(ForbiddenSourceError):
            state.authorize_source(TEST_SOURCE, phase=phase)


def test_final_requires_hash_valid_frozen_manifest(tmp_path: Path) -> None:
    """Catches final evaluation without an authenticated ten-gate freeze."""
    state = PhaseState(tmp_path)
    with pytest.raises(PhaseOrderError):
        state.require_final_authorization()


def test_final_cli_accepts_only_the_frozen_threshold(tmp_path: Path) -> None:
    """Catches a final command that records unsupported or tunable arguments."""
    parsed = build_parser().parse_args(["final", "--threshold", "0.5"])
    assert parsed.phase == "final"
    assert parsed.threshold == 0.5
    with pytest.raises(ValueError, match="frozen at 0.5"):
        final(tmp_path, threshold=0.4)


def test_mutated_checkpoint_invalidates_final_authorization(tmp_path: Path) -> None:
    """Catches checkpoint replacement after validation candidate freeze."""
    _, checkpoints = _write_valid_freeze(tmp_path)
    state = PhaseState(tmp_path)

    authorization = state.require_final_authorization()
    assert authorization["final_freeze"]["winning_candidate"] == "rf_v1_bbox"

    checkpoints[42].write_text("substituted\n", encoding="utf-8")
    with pytest.raises(ArtifactAuthenticationError, match="checkpoint"):
        state.require_final_authorization()


def test_final_authorization_rejects_semantically_malformed_bound_summary(
    tmp_path: Path,
) -> None:
    """Catches trusting freeze booleans without reconstructing the ten gates."""
    _write_valid_freeze(tmp_path, malformed_summary=True)

    with pytest.raises(ArtifactAuthenticationError, match="validation summary"):
        PhaseState(tmp_path).require_final_authorization()


def test_predecessor_mutation_after_freeze_invalidates_final_authorization(
    tmp_path: Path,
) -> None:
    """Catches a freeze that authenticates only the validation marker file, not its chain."""
    _write_valid_freeze(tmp_path)
    (tmp_path / "validation-evidence.txt").write_text("mutated\n", encoding="utf-8")

    with pytest.raises(ArtifactAuthenticationError, match="validation-evidence"):
        PhaseState(tmp_path).require_final_authorization()


def test_recursive_prepare_mutation_after_freeze_invalidates_final_authorization(
    tmp_path: Path,
) -> None:
    """Catches a freeze that stops recursive verification at training/validation."""
    _write_valid_freeze(tmp_path)
    (tmp_path / "prepare-evidence.txt").write_text("mutated\n", encoding="utf-8")

    with pytest.raises(ArtifactAuthenticationError, match="prepare-evidence"):
        PhaseState(tmp_path).require_final_authorization()


def test_staged_source_mutation_after_freeze_invalidates_final_authorization(
    tmp_path: Path,
) -> None:
    """Catches trusting only the staging manifest hash after derived arrays drift."""
    _write_valid_freeze(tmp_path, staged_source_manifest=True)
    (tmp_path / "staged_sources" / "summary_v1.npy").write_text(
        "mutated\n", encoding="utf-8"
    )

    with pytest.raises(ArtifactAuthenticationError, match="source staging"):
        PhaseState(tmp_path).require_final_authorization()


def test_malformed_freeze_rejects_before_final_import_or_set03_path_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches final-source import/stat/hash/open before semantic freeze proof."""
    _write_valid_freeze(
        tmp_path, malformed_summary=True, staged_source_manifest=True
    )
    sentinel = os.path.abspath(
        os.fspath(tmp_path.parent / "055_compact_residual_biu_lstm" / "cache" / "test")
    )
    original_import = builtins.__import__
    original_open = Path.open
    original_stat = Path.stat

    def guarded_import(name, *args, **kwargs):
        if name.endswith(".data") or name == "scripts.data":
            pytest.fail("final builder imported before authorization")
        return original_import(name, *args, **kwargs)

    def guarded_open(path: Path, *args, **kwargs):
        if os.path.abspath(os.fspath(path)).startswith(sentinel):
            pytest.fail("Set03 opened before authorization")
        return original_open(path, *args, **kwargs)

    def guarded_stat(path: Path, *args, **kwargs):
        if os.path.abspath(os.fspath(path)).startswith(sentinel):
            pytest.fail("Set03 stat/hash before authorization")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "stat", guarded_stat)
    with pytest.raises(ArtifactAuthenticationError, match="validation summary"):
        final(tmp_path)


def test_substituted_freeze_path_rejects_before_set03_stat_hash_or_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches generic marker hashing of an attacker-selected Set03 path."""
    freeze, _ = _write_valid_freeze(tmp_path, staged_source_manifest=True)
    marker = json.loads(freeze.read_text(encoding="utf-8"))
    sentinel_root = (
        tmp_path.parent / "055_compact_residual_biu_lstm" / "cache" / "test"
    )
    marker["input_artifact_paths"]["metrics/validation_summary_results.csv"] = str(
        sentinel_root / "rows.csv"
    )
    freeze.write_text(json.dumps(marker), encoding="utf-8")
    sentinel = os.path.abspath(os.fspath(sentinel_root))
    original_open = Path.open
    original_stat = Path.stat

    def guarded_open(path: Path, *args, **kwargs):
        if os.path.abspath(os.fspath(path)).startswith(sentinel):
            pytest.fail("Set03 opened before substituted path rejection")
        return original_open(path, *args, **kwargs)

    def guarded_stat(path: Path, *args, **kwargs):
        if os.path.abspath(os.fspath(path)).startswith(sentinel):
            pytest.fail("Set03 stat/hashed before substituted path rejection")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "stat", guarded_stat)
    with pytest.raises(ArtifactAuthenticationError, match="artifact path is invalid"):
        final(tmp_path)


def test_final_authorization_recomputes_both_candidates_before_set03_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches an internally bound but worse declared winner bypassing selection."""
    _write_valid_freeze(
        tmp_path,
        staged_source_manifest=True,
        declared_winner="xgb_v2_bbox",
    )
    freeze_payload = json.loads(
        (tmp_path / "FROZEN_FINAL_MODEL.json").read_text(encoding="utf-8")
    )["payload"]["candidate_selection"]
    assert freeze_payload["eligible"] == list(ExperimentConfig().anchor_families)
    assert freeze_payload["winner"] == "xgb_v2_bbox"
    sentinel = os.path.abspath(
        os.fspath(tmp_path.parent / "055_compact_residual_biu_lstm" / "cache" / "test")
    )
    original_open = Path.open
    original_stat = Path.stat

    def guarded_open(path: Path, *args, **kwargs):
        if os.path.abspath(os.fspath(path)).startswith(sentinel):
            pytest.fail("Set03 opened before cross-family selection rejection")
        return original_open(path, *args, **kwargs)

    def guarded_stat(path: Path, *args, **kwargs):
        if os.path.abspath(os.fspath(path)).startswith(sentinel):
            pytest.fail("Set03 stat/hashed before cross-family selection rejection")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "stat", guarded_stat)
    with pytest.raises(ArtifactAuthenticationError, match="recomputed winner"):
        final(tmp_path)


def test_final_authorization_rejects_mutated_nonwinner_summary_binding(
    tmp_path: Path,
) -> None:
    """Catches authenticating only the winner while the other family digest drifts."""
    freeze, _ = _write_valid_freeze(tmp_path, staged_source_manifest=True)
    marker = json.loads(freeze.read_text(encoding="utf-8"))
    marker["payload"]["candidate_selection"]["families"]["xgb_v2_bbox"][
        "candidate_summary_sha256"
    ] = "0" * 64
    freeze.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(ArtifactAuthenticationError, match="both candidate families"):
        PhaseState(tmp_path).require_final_authorization()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("omitted", "required predecessor"),
        ("alternate", "predecessor path is invalid"),
        ("extra", "extra predecessor"),
    ],
)
def test_final_authorization_requires_exact_marker_chain_topology(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    """Catches standalone, redirected, or multiply-linked phase markers."""
    _write_valid_freeze(tmp_path, staged_source_manifest=True)
    validation_path = tmp_path / "VALIDATION_COMPLETE.json"
    marker = json.loads(validation_path.read_text(encoding="utf-8"))
    if mutation == "omitted":
        marker["input_artifacts"].pop("TRAINING_COMPLETE.json")
        marker["input_artifact_paths"].pop("TRAINING_COMPLETE.json")
    elif mutation == "alternate":
        marker["input_artifact_paths"]["TRAINING_COMPLETE.json"] = str(
            tmp_path / "alternate" / "TRAINING_COMPLETE.json"
        )
    else:
        marker["input_artifacts"]["PREPARED.json"] = sha256_file(
            tmp_path / "PREPARED.json"
        )
        marker["input_artifact_paths"]["PREPARED.json"] = str(
            tmp_path / "PREPARED.json"
        )
    validation_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(ArtifactAuthenticationError, match=message):
        PhaseState(tmp_path).require_final_authorization()


@pytest.mark.parametrize(
    ("marker_name", "predecessor_name"),
    [
        ("VALIDATION_COMPLETE.json", "TRAINING_COMPLETE.json"),
        ("TRAINING_COMPLETE.json", "ANCHORS_READY.json"),
        ("ANCHORS_READY.json", "PREPARED.json"),
        ("PREPARED.json", "SOURCE_STAGING.json"),
    ],
)
def test_final_authorization_requires_every_direct_predecessor_link(
    tmp_path: Path,
    marker_name: str,
    predecessor_name: str,
) -> None:
    """Catches recursive discovery being used instead of mandatory direct links."""
    _write_valid_freeze(tmp_path, staged_source_manifest=True)
    marker_path = tmp_path / marker_name
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["input_artifacts"].pop(predecessor_name)
    marker["input_artifact_paths"].pop(predecessor_name)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(ArtifactAuthenticationError, match="required predecessor"):
        PhaseState(tmp_path).require_final_authorization()


def test_final_authorization_rejects_symlinked_nested_predecessor_before_target_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches following a nested marker symlink before structural preflight."""
    _write_valid_freeze(tmp_path, staged_source_manifest=True)
    training = tmp_path / "TRAINING_COMPLETE.json"
    training.unlink()
    sentinel_root = tmp_path.parent / "set03-sentinel"
    training.symlink_to(sentinel_root / "TRAINING_COMPLETE.json")
    sentinel = os.path.abspath(os.fspath(sentinel_root))
    original_stat = Path.stat

    def guarded_stat(path: Path, *args, **kwargs):
        if os.path.abspath(os.fspath(path)).startswith(sentinel):
            pytest.fail("symlink target statted before marker-chain rejection")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", guarded_stat)
    with pytest.raises(ArtifactAuthenticationError, match="symlink"):
        PhaseState(tmp_path).require_final_authorization()


def test_validation_only_and_final_freeze_are_mutually_exclusive(tmp_path: Path) -> None:
    """Catches contradictory terminal decisions that could expose Set03 after failure."""
    _write_valid_freeze(tmp_path)
    (tmp_path / "VALIDATION_ONLY_COMPLETE").write_text("{}\n", encoding="utf-8")

    with pytest.raises(VerificationError, match="mutually exclusive"):
        PhaseState(tmp_path).verify_terminal_state()


def test_failed_verifier_cannot_leave_experiment_complete(tmp_path: Path) -> None:
    """Catches a stale success marker surviving a failed re-verification."""
    complete = tmp_path / "EXPERIMENT_COMPLETE"
    complete.write_text("{}\n", encoding="utf-8")

    with pytest.raises(VerificationError, match="failed verifier"):
        PhaseState(tmp_path).record_verification(passed=False, evidence={})
    assert not complete.exists()


def test_repeated_successful_verification_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches repeat verify deleting or replacing an authentic completion marker."""
    source_manifest = tmp_path / "SOURCE_STAGING.json"
    source_manifest.write_text('{"fixture":"staged"}\n', encoding="utf-8")
    evidence = tmp_path / "validation.txt"
    evidence.write_text("valid\n", encoding="utf-8")
    from scripts.contracts import write_authenticated_json_marker

    write_authenticated_json_marker(
        tmp_path / "VALIDATION_ONLY_COMPLETE",
        config=ExperimentConfig(),
        source_manifest=source_manifest,
        producing_command="runner.py freeze",
        input_artifacts={"evidence": evidence},
        payload={"set03_accessed": False},
    )
    import scripts.runner as runner

    monkeypatch.setattr(runner, "_verify_validation_bundle", lambda *_args: {})
    state = PhaseState(tmp_path, source_manifest=source_manifest)
    first = verify(tmp_path)
    first_hash = sha256_file(tmp_path / "EXPERIMENT_COMPLETE")
    second = verify(tmp_path)

    assert second == first
    assert sha256_file(tmp_path / "EXPERIMENT_COMPLETE") == first_hash
    monkeypatch.setattr(
        runner,
        "_verify_validation_bundle",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("transient verifier error")),
    )
    with pytest.raises(RuntimeError, match="transient verifier error"):
        verify(tmp_path)
    assert sha256_file(tmp_path / "EXPERIMENT_COMPLETE") == first_hash
    with pytest.raises(VerificationError, match="failed verifier"):
        state.record_verification(passed=False, evidence={})
    assert sha256_file(tmp_path / "EXPERIMENT_COMPLETE") == first_hash


def test_later_phase_recursively_rejects_mutated_earlier_artifact(tmp_path: Path) -> None:
    """Catches a valid prior-marker hash masking mutation of its own bound inputs."""
    state = PhaseState(tmp_path, source_manifest={"rows.csv": "frozen"})
    scaler = tmp_path / "metrics" / "context_scaler.json"
    scaler.parent.mkdir(parents=True, exist_ok=True)
    scaler.write_text("original scaler\n", encoding="utf-8")
    state.write_phase_marker(
        "prepare", command="runner.py prepare", artifacts={"scaler": scaler}
    )
    anchor = tmp_path / "models" / "anchor.joblib"
    anchor.parent.mkdir(parents=True, exist_ok=True)
    anchor.write_text("anchor\n", encoding="utf-8")
    state.write_phase_marker(
        "fit-anchors",
        command="runner.py fit-anchors",
        artifacts={"PREPARED.json": tmp_path / "PREPARED.json", "anchor": anchor},
    )

    scaler.write_text("mutated scaler\n", encoding="utf-8")
    with pytest.raises(ArtifactAuthenticationError, match="scaler"):
        state.require_prior("train")


def _fake_oof(root: Path) -> tuple[GroupedOOFResult, Path]:
    probabilities = np.array([0.1, 0.8, 0.2, 0.7, 0.4], dtype=np.float64)
    fold_artifacts = []
    audit_rows = []
    for fold in range(1, 6):
        artifact = root / "folds" / f"fold{fold}.joblib"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(f"fold {fold}\n", encoding="utf-8")
        fold_artifacts.append(
            FoldModelArtifact(
                fold=fold,
                path=artifact,
                sha256=sha256_file(artifact),
                size_bytes=artifact.stat().st_size,
                complexity=AnchorComplexity(1, 3, 2),
                train_rows=4,
                holdout_rows=1,
                train_positive_rows=2,
                holdout_positive_rows=fold % 2,
                train_negative_rows=2,
                holdout_negative_rows=1 - fold % 2,
                train_unique_tracks=4,
                holdout_unique_tracks=1,
                shared_track_count=0,
            )
        )
        audit_rows.append({"fold": fold, "shared_track_count": 0})
    full_anchor = root / "anchors" / "rf_v1_bbox_seed42.joblib"
    full_anchor.parent.mkdir(parents=True, exist_ok=True)
    full_anchor.write_text("full anchor\n", encoding="utf-8")
    return (
        GroupedOOFResult(
            probabilities=probabilities,
            assignment_count=np.ones(5, dtype=np.int64),
            fold_assignment=np.arange(1, 6, dtype=np.int64),
            fold_audit=pd.DataFrame(audit_rows),
            fold_artifacts=tuple(fold_artifacts),
        ),
        full_anchor,
    )


def _persist_fake_bundle(root: Path) -> tuple[ProbabilityBundleSpec, GroupedOOFResult]:
    result, full_anchor = _fake_oof(root)
    spec = persist_probability_bundle(
        root=root,
        anchor_name="rf_v1_bbox",
        seed=42,
        grouped_oof=result,
        train_sequence_ids=np.array(["t0", "t1", "t2", "t3", "t4"]),
        train_track_ids=np.array(["a", "b", "c", "d", "e"]),
        train_labels=np.array([0, 1, 0, 1, 0]),
        validation_sequence_ids=np.array(["v0", "v1", "v2"]),
        validation_labels=np.array([0, 1, 0]),
        validation_probabilities=np.array([0.2, 0.8, 0.3]),
        parity_verified=True,
        full_anchor_path=full_anchor,
    )
    return spec, result


def test_probability_authentication_accepts_only_bound_grouped_oof(tmp_path: Path) -> None:
    """Catches training on an in-sample, unassigned, or substituted score vector."""
    spec, result = _persist_fake_bundle(tmp_path)

    authenticated = authenticate_probability_bundle(spec, grouped_oof=result)
    np.testing.assert_array_equal(authenticated.train_probabilities, result.probabilities)
    assert authenticated.anchor_name == "rf_v1_bbox"
    assert authenticated.seed == 42

    substituted = GroupedOOFResult(
        probabilities=result.probabilities.copy(),
        assignment_count=result.assignment_count.copy(),
        fold_assignment=result.fold_assignment.copy(),
        fold_audit=result.fold_audit.copy(),
        fold_artifacts=result.fold_artifacts,
    )
    substituted.probabilities[0] = 0.99
    with pytest.raises(ArtifactAuthenticationError, match="OOF probability"):
        authenticate_probability_bundle(spec, grouped_oof=substituted)

    unassigned = GroupedOOFResult(
        probabilities=result.probabilities,
        assignment_count=np.array([0, 1, 1, 1, 1]),
        fold_assignment=result.fold_assignment,
        fold_audit=result.fold_audit,
        fold_artifacts=result.fold_artifacts,
    )
    with pytest.raises(ArtifactAuthenticationError, match="exactly one"):
        authenticate_probability_bundle(spec, grouped_oof=unassigned)


def test_probability_authentication_rejects_unbound_and_full_anchor_substitution(
    tmp_path: Path,
) -> None:
    """Catches bare in-sample arrays and replacement of the parity-verified full anchor."""
    spec, result = _persist_fake_bundle(tmp_path)
    with pytest.raises(TypeError, match="GroupedOOFResult"):
        authenticate_probability_bundle(spec, grouped_oof=result.probabilities)  # type: ignore[arg-type]

    spec.full_anchor_path.write_text("different full anchor\n", encoding="utf-8")
    with pytest.raises(ArtifactAuthenticationError, match="full anchor"):
        authenticate_probability_bundle(spec, grouped_oof=result)


def test_probability_authentication_rejects_validation_substitution_and_wrong_identity(
    tmp_path: Path,
) -> None:
    """Catches non-parity validation scores or cross-seed/anchor bundle reuse."""
    spec, result = _persist_fake_bundle(tmp_path)
    validation = pd.read_csv(spec.validation_path)
    validation.loc[0, "probability"] = 0.99
    validation.to_csv(spec.validation_path, index=False)
    with pytest.raises(ArtifactAuthenticationError, match="validation"):
        authenticate_probability_bundle(spec, grouped_oof=result)

    spec, result = _persist_fake_bundle(tmp_path / "identity")
    with pytest.raises(ArtifactAuthenticationError, match="identity"):
        authenticate_probability_bundle(
            spec, grouped_oof=result, expected_anchor="rf_v1_bbox", expected_seed=43
        )


def test_training_boundary_authenticates_before_calling_primitive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches runner calls to train_one_seed with vectors not proven by Task 3 artifacts."""
    import scripts.runner as runner

    spec, result = _persist_fake_bundle(tmp_path)
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def fake_train(*args, **kwargs):
        calls.append((np.asarray(args[2]), np.asarray(args[6])))
        return "trained"

    monkeypatch.setattr(runner, "train_one_seed", fake_train)
    train_ego = np.zeros((5, 16, 3), dtype=np.float32)
    train_yolop = np.zeros((5, 16, 20), dtype=np.float32)
    val_ego = np.zeros((3, 16, 3), dtype=np.float32)
    val_yolop = np.zeros((3, 16, 20), dtype=np.float32)

    output = run_authenticated_training(
        spec=spec,
        grouped_oof=result,
        train_ego=train_ego,
        train_yolop=train_yolop,
        train_labels=np.array([0, 1, 0, 1, 0]),
        train_sequence_ids=np.array(["t0", "t1", "t2", "t3", "t4"]),
        validation_ego=val_ego,
        validation_yolop=val_yolop,
        validation_labels=np.array([0, 1, 0]),
        validation_sequence_ids=np.array(["v0", "v1", "v2"]),
        checkpoint_path=tmp_path / "checkpoint.pt",
    )
    assert output == "trained"
    np.testing.assert_array_equal(calls[0][0], result.probabilities)
    np.testing.assert_array_equal(calls[0][1], np.array([0.2, 0.8, 0.3]))

    result.assignment_count[0] = 0
    with pytest.raises(ArtifactAuthenticationError, match="exactly one"):
        run_authenticated_training(
            spec=spec,
            grouped_oof=result,
            train_ego=train_ego,
            train_yolop=train_yolop,
            train_labels=np.array([0, 1, 0, 1, 0]),
            train_sequence_ids=np.array(["t0", "t1", "t2", "t3", "t4"]),
            validation_ego=val_ego,
            validation_yolop=val_yolop,
            validation_labels=np.array([0, 1, 0]),
            validation_sequence_ids=np.array(["v0", "v1", "v2"]),
            checkpoint_path=tmp_path / "checkpoint.pt",
        )
    assert len(calls) == 1


def test_training_boundary_rejects_same_length_identity_and_label_permutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches row-order leakage hidden by length-only probability checks."""
    import scripts.runner as runner

    spec, result = _persist_fake_bundle(tmp_path)
    monkeypatch.setattr(
        runner,
        "train_one_seed",
        lambda *_args, **_kwargs: pytest.fail("trainer reached with substituted rows"),
    )
    common = {
        "spec": spec,
        "grouped_oof": result,
        "train_ego": np.zeros((5, 16, 3), dtype=np.float32),
        "train_yolop": np.zeros((5, 16, 20), dtype=np.float32),
        "validation_ego": np.zeros((3, 16, 3), dtype=np.float32),
        "validation_yolop": np.zeros((3, 16, 20), dtype=np.float32),
        "validation_labels": np.array([0, 1, 0]),
        "validation_sequence_ids": np.array(["v0", "v1", "v2"]),
        "checkpoint_path": tmp_path / "checkpoint.pt",
    }
    with pytest.raises(ArtifactAuthenticationError, match="training row identity"):
        run_authenticated_training(
            **common,
            train_labels=np.array([0, 1, 0, 1, 0]),
            train_sequence_ids=np.array(["t1", "t0", "t2", "t3", "t4"]),
        )
    with pytest.raises(ArtifactAuthenticationError, match="training labels"):
        run_authenticated_training(
            **common,
            train_labels=np.array([1, 0, 0, 1, 0]),
            train_sequence_ids=np.array(["t0", "t1", "t2", "t3", "t4"]),
        )
    with pytest.raises(ArtifactAuthenticationError, match="validation row identity"):
        run_authenticated_training(
            **{
                **common,
                "validation_sequence_ids": np.array(["v1", "v0", "v2"]),
            },
            train_labels=np.array([0, 1, 0, 1, 0]),
            train_sequence_ids=np.array(["t0", "t1", "t2", "t3", "t4"]),
        )


def test_candidate_selection_uses_frozen_tie_order() -> None:
    """Catches selecting by anchor name or high-confidence errors before frozen metrics."""
    anchor = _anchor_summary()
    candidates = {
        "rf_v1_bbox": (_summary(harmonic=0.736, errors=18), anchor),
        "xgb_v2_bbox": (_summary(harmonic=0.737, errors=20), anchor),
    }
    assert select_validation_candidate(candidates).winner == "xgb_v2_bbox"

    tied = {
        "rf_v1_bbox": (_summary(harmonic=0.736, errors=18), anchor),
        "xgb_v2_bbox": (_summary(harmonic=0.736, errors=19), anchor),
    }
    assert select_validation_candidate(tied).winner == "rf_v1_bbox"


def test_summary_preserves_fractional_mean_high_confidence_errors() -> None:
    """Catches rounding a 20.333 corrected mean down to an eligible 20 errors."""
    frame = pd.DataFrame(
        [
            {
                "anchor_name": "rf_v1_bbox",
                "variant": "corrected",
                "f1": 0.716,
                "pr_auc": 0.755,
                "roc_auc": 0.904,
                "harmonic": 0.735,
                "high_confidence_errors": 20.333,
                "per_seed_harmonic": "[0.726, 0.736, 0.746]",
                "invariants_pass": True,
            }
        ]
    )

    summary = _summary_bundle(frame, "rf_v1_bbox", "corrected")

    assert summary.high_confidence_errors == pytest.approx(20.333)
    decision = select_validation_candidate(
        {
            "rf_v1_bbox": (summary, _anchor_summary()),
            "xgb_v2_bbox": (_summary(errors=19), _anchor_summary()),
        }
    )
    assert not decision.decisions["rf_v1_bbox"].eligible


def test_validation_output_writer_emits_required_bounded_smoke_bundle(tmp_path: Path) -> None:
    """Catches missing validation predictions, metrics, deltas, transitions, or ablations."""
    rows = pd.DataFrame(
        {
            "sequence_id": ["v0", "v1", "v2", "v3"],
            "label": [0, 1, 0, 1],
        }
    )
    artifacts = write_validation_outputs(
        root=tmp_path,
        anchor_name="rf_v1_bbox",
        seed=42,
        rows=rows,
        anchor_probabilities=np.array([0.7, 0.2, 0.1, 0.8]),
        corrected_probabilities=np.array([0.4, 0.7, 0.2, 0.9]),
        checkpoint_epoch=3,
        gate=np.array([0.2, 0.3, 0.4, 0.5]),
        applied_correction=np.array([-0.2, 0.3, 0.1, 0.1]),
        ablation_probabilities={
            "zero_ego": np.array([0.5, 0.6, 0.2, 0.8]),
            "zero_yolop": np.array([0.4, 0.6, 0.3, 0.8]),
            "reversed_ego": np.array([0.4, 0.7, 0.3, 0.8]),
            "reversed_yolop": np.array([0.4, 0.7, 0.2, 0.8]),
        },
        invariants_pass=True,
    )

    required = {
        "metrics/validation_per_seed_results.csv",
        "metrics/validation_summary_results.csv",
        "metrics/paired_deltas.csv",
        "metrics/error_transitions_fixed_0p5.csv",
        "metrics/context_ablation_diagnostics.csv",
        "predictions/validation_rf_v1_bbox_seed42.csv",
    }
    assert required == set(artifacts)
    assert pd.read_csv(tmp_path / "predictions/validation_rf_v1_bbox_seed42.csv").shape == (4, 5)
    transitions = pd.read_csv(tmp_path / "metrics/error_transitions_fixed_0p5.csv")
    assert transitions.loc[0, "anchor_fp_to_corrected_tn"] == 1
    assert transitions.loc[0, "anchor_fn_to_corrected_tp"] == 1
    assert set(pd.read_csv(tmp_path / "metrics/context_ablation_diagnostics.csv")["ablation"]) == {
        "zero_ego",
        "zero_yolop",
        "reversed_ego",
        "reversed_yolop",
    }


def test_anchor_metadata_is_bound_to_frozen_matrix_abi() -> None:
    """Catches runner dimension-only matching that swaps the Exp012 feature set."""
    assert anchor_feature_metadata("rf_v1_bbox") == {
        "anchor_name": "rf_v1_bbox",
        "feature_dimension": 125,
        "feature_set": "biu_v1_plus_bbox",
    }
    assert anchor_feature_metadata("xgb_v2_bbox") == {
        "anchor_name": "xgb_v2_bbox",
        "feature_dimension": 109,
        "feature_set": "biu_v2_core_plus_bbox",
    }


def _write_final_verifier_fixture(root: Path) -> tuple[ExperimentConfig, dict]:
    from dataclasses import replace

    config = replace(ExperimentConfig(), test_rows=2)
    metrics = root / "metrics"
    predictions = root / "predictions"
    metrics.mkdir(parents=True)
    predictions.mkdir(parents=True)
    rows_path = root / "authenticated_set03_rows.csv"
    pd.DataFrame(
        {"sequence_id": ["s0", "s1"], "set_id": ["set03", "set03"], "label": [0, 1]}
    ).to_csv(rows_path, index=False)
    records = []
    metric_fields = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "harmonic",
        "bce",
        "brier",
        "high_confidence_errors",
        "tp",
        "tn",
        "fp",
        "fn",
        "high_confidence_fp",
        "high_confidence_fn",
    )
    labels = np.array([0, 1], dtype=np.int64)
    probabilities = np.array([0.1, 0.9], dtype=np.float64)
    expected_metric = compute_binary_metrics(labels, probabilities)
    for seed in config.seeds:
        for variant in ("anchor", "corrected"):
            record = {
                "anchor_name": "rf_v1_bbox",
                "seed": seed,
                "variant": variant,
                "threshold": 0.5,
                **asdict(expected_metric),
            }
            record["per_seed_harmonic"] = json.dumps(
                record["per_seed_harmonic"]
            )
            records.append(record)
        pd.DataFrame(
            {
                "sequence_id": ["s0", "s1"],
                "set_id": ["set03", "set03"],
                "label": labels,
                "anchor_probability": probabilities,
                "corrected_probability": probabilities,
                "threshold": [0.5, 0.5],
            }
        ).to_csv(predictions / f"final_rf_v1_bbox_seed{seed}.csv", index=False)
    pd.DataFrame(records).to_csv(metrics / "final_per_seed_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "anchor_name": "rf_v1_bbox",
                "variant": variant,
                "seed_count": 3,
                **{
                    field: float(getattr(expected_metric, field))
                    for field in metric_fields
                },
                **{f"{field}_std": 0.0 for field in metric_fields},
            }
            for variant in ("anchor", "corrected")
        ]
    ).to_csv(metrics / "final_summary_results.csv", index=False)
    marker = {
        "input_artifact_paths": {"set03_rows.csv": str(rows_path.resolve())},
        "input_artifacts": {"set03_rows.csv": sha256_file(rows_path)},
    }
    return config, marker


def test_final_verifier_binds_exact_set03_order_labels_and_unique_summary(
    tmp_path: Path,
) -> None:
    """Catches final seed agreement that ignores the authenticated Set03 cohort."""
    config, marker = _write_final_verifier_fixture(tmp_path)
    _verify_final_bundle(tmp_path, config, "rf_v1_bbox", marker)

    prediction = tmp_path / "predictions" / "final_rf_v1_bbox_seed43.csv"
    frame = pd.read_csv(prediction).iloc[::-1]
    frame.to_csv(prediction, index=False)
    with pytest.raises(VerificationError, match="authenticated Set03"):
        _verify_final_bundle(tmp_path, config, "rf_v1_bbox", marker)

    config, marker = _write_final_verifier_fixture(tmp_path / "labels")
    prediction = (
        tmp_path / "labels" / "predictions" / "final_rf_v1_bbox_seed44.csv"
    )
    frame = pd.read_csv(prediction)
    frame["label"] = [1, 0]
    frame.to_csv(prediction, index=False)
    with pytest.raises(VerificationError, match="authenticated Set03"):
        _verify_final_bundle(tmp_path / "labels", config, "rf_v1_bbox", marker)

    config, marker = _write_final_verifier_fixture(tmp_path / "duplicates")
    summary_path = tmp_path / "duplicates" / "metrics" / "final_summary_results.csv"
    summary = pd.read_csv(summary_path)
    pd.concat([summary, summary.iloc[[0]]], ignore_index=True).to_csv(summary_path, index=False)
    with pytest.raises(VerificationError, match="unique anchor and corrected"):
        _verify_final_bundle(tmp_path / "duplicates", config, "rf_v1_bbox", marker)


@pytest.mark.parametrize(
    ("column", "value"),
    [("f1", 0.5), ("tp", 0), ("high_confidence_fp", 1)],
)
def test_final_verifier_recomputes_per_seed_metrics_and_counts(
    tmp_path: Path,
    column: str,
    value: float,
) -> None:
    """Catches arbitrary finite metrics or altered confusion/high-confidence counts."""
    config, marker = _write_final_verifier_fixture(tmp_path)
    per_seed_path = tmp_path / "metrics" / "final_per_seed_results.csv"
    per_seed = pd.read_csv(per_seed_path)
    per_seed.loc[
        (per_seed["seed"] == 42) & (per_seed["variant"] == "corrected"), column
    ] = value
    per_seed.to_csv(per_seed_path, index=False)

    with pytest.raises(VerificationError, match="does not match predictions"):
        _verify_final_bundle(tmp_path, config, "rf_v1_bbox", marker)


def test_final_verifier_recomputes_metrics_after_prediction_mutation(
    tmp_path: Path,
) -> None:
    """Catches accepting changed probabilities with stale finite metric rows."""
    config, marker = _write_final_verifier_fixture(tmp_path)
    prediction_path = tmp_path / "predictions" / "final_rf_v1_bbox_seed43.csv"
    prediction = pd.read_csv(prediction_path)
    prediction.loc[0, "corrected_probability"] = 0.99
    prediction.to_csv(prediction_path, index=False)

    with pytest.raises(VerificationError, match="does not match predictions"):
        _verify_final_bundle(tmp_path, config, "rf_v1_bbox", marker)


@pytest.mark.parametrize(("column", "value"), [("f1", 0.5), ("f1_std", 0.25)])
def test_final_verifier_recomputes_summary_mean_and_std(
    tmp_path: Path,
    column: str,
    value: float,
) -> None:
    """Catches a finite but unfaithful final aggregate mean or standard deviation."""
    config, marker = _write_final_verifier_fixture(tmp_path)
    summary_path = tmp_path / "metrics" / "final_summary_results.csv"
    summary = pd.read_csv(summary_path)
    summary.loc[summary["variant"] == "corrected", column] = value
    summary.to_csv(summary_path, index=False)

    with pytest.raises(VerificationError, match="summary does not match per-seed"):
        _verify_final_bundle(tmp_path, config, "rf_v1_bbox", marker)


def test_efficiency_rows_separate_cpu_cuda_and_perception_categories() -> None:
    """Catches ambiguous cached-feature latency or invented unavailable runtimes."""
    rows = build_efficiency_rows(
        anchor_name="rf_v1_bbox",
        seed=42,
        sample_count=2,
        anchor_seconds_cpu=0.2,
        correction_seconds_cpu=0.1,
        correction_seconds_cuda=None,
        tree_bytes=100,
        checkpoint_bytes=50,
        tree_count=3,
        node_count=9,
        leaf_count=5,
    )
    by_name = {row["measurement"]: row for row in rows}

    assert len(rows) == 12
    assert len(by_name) == len(rows)
    assert by_name["tree_anchor_cached_feature_latency_cpu"]["value"] == pytest.approx(0.1)
    assert by_name["neural_correction_cached_feature_latency_cpu"]["value"] == pytest.approx(0.05)
    assert by_name["combined_cached_feature_latency_cpu"]["value"] == pytest.approx(0.15)
    assert by_name["neural_correction_cached_feature_latency_cuda"]["status"] == "unavailable"
    assert by_name["combined_cached_feature_latency_cuda"]["status"] == "unavailable"
