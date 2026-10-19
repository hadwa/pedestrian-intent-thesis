"""Immutable scientific, metric, validation-gate, and marker contracts for Exp062."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ExperimentConfig:
    """The approved Exp062 protocol; later code must not substitute these values."""

    experiment_number: int = 62
    observed_frames: int = 16
    tte: str = "30_60"
    train_cohorts: tuple[str, ...] = ("set01", "set02", "set06")
    val_cohorts: tuple[str, ...] = ("set04", "set05")
    test_cohorts: tuple[str, ...] = ("set03",)
    train_rows: int = 2366
    val_rows: int = 3619
    test_rows: int = 3760
    seeds: tuple[int, ...] = (42, 43, 44)
    threshold: float = 0.5
    anchor_families: tuple[str, ...] = ("rf_v1_bbox", "xgb_v2_bbox")
    rf_feature_dim: int = 125
    xgb_feature_dim: int = 109
    rf_v1_bbox_feature_set: str = "biu_v1_plus_bbox"
    xgb_v2_bbox_feature_set: str = "biu_v2_core_plus_bbox"
    rf_v1_bbox_feature_blocks: tuple[tuple[str, int], ...] = (
        ("enriched_bbox", 53),
        ("biu_v1_with_quality_counters", 72),
    )
    xgb_v2_bbox_feature_blocks: tuple[tuple[str, int], ...] = (
        ("enriched_bbox", 53),
        ("biu_v2_core", 56),
    )
    exp012_feature_definition_path: str = (
        "experiments/012_pie_biu_v2_30_60_benchmark/metrics/"
        "pie_biu_v2_feature_set_definitions.csv"
    )
    exp012_feature_definition_sha256: str = (
        "6708d290d9263ad29000786cf6d15c5ec052dc8712c1b5a16741944295d0145c"
    )
    ego_shape: tuple[int, int] = (16, 3)
    yolop_shape: tuple[int, int] = (16, 20)
    context_parameter_count: int = 5932
    correction_bound: float = 1.5
    parity_tolerance: float = 1e-7
    trainval_rows_sha256: str = (
        "a01ad1f3244157c1b649aada0470ad54d877bbd1e8075c275ee313c35dcc3bf1"
    )
    ego_trainval_sha256: str = (
        "410c4757e4012e223721f5686bffe5f8e4c4f53dd9fd9030e763fcfbe58dd302"
    )
    yolop_trainval_sha256: str = (
        "b353bf9d8fb7d5eedacf726fec643f67138994f5e9a864806f46e382075b65f5"
    )

    @property
    def source_hashes(self) -> dict[str, str]:
        """Named immutable-source fingerprints required before Exp062 preparation."""
        return {
            "trainval_rows.csv": self.trainval_rows_sha256,
            "trainval_ego.npy": self.ego_trainval_sha256,
            "trainval_yolop.npy": self.yolop_trainval_sha256,
        }


@dataclass(frozen=True)
class MetricBundle:
    """Fixed-threshold metrics and validation-gate inputs for one result bundle."""

    f1: float
    pr_auc: float
    roc_auc: float
    harmonic: float
    high_confidence_errors: float
    per_seed_harmonic: tuple[float, ...] = ()
    invariants_pass: bool = True
    accuracy: float = float("nan")
    precision: float = float("nan")
    recall: float = float("nan")
    bce: float = float("nan")
    brier: float = float("nan")
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    high_confidence_fp: int = 0
    high_confidence_fn: int = 0


@dataclass(frozen=True)
class ValidationGateDecision:
    """The predeclared, auditable result of all ten validation predicates."""

    eligible: bool
    failed_conditions: tuple[str, ...]


@dataclass(frozen=True)
class AnchorComplexity:
    """Native tree counts recorded for a freshly refit Exp062 anchor."""

    tree_count: int
    node_count: int
    leaf_count: int


@dataclass(frozen=True)
class AnchorParityResult:
    """Fail-closed validation parity result for one fresh/historical anchor pair."""

    passed: bool
    id_match: bool
    label_match: bool
    class_match: bool
    max_probability_error: float
    max_metric_error: float
    failed_checks: tuple[str, ...]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    """Hash canonical ASCII JSON so provenance digests do not depend on key order."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def experiment_config_sha256(config: ExperimentConfig) -> str:
    """Fingerprint every frozen field in the experiment contract."""
    return canonical_json_sha256(asdict(config))


def harmonic_f1_pr(f1: float, pr_auc: float) -> float:
    return 0.0 if f1 + pr_auc == 0 else 2.0 * f1 * pr_auc / (f1 + pr_auc)


def compute_binary_metrics(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    threshold: float = ExperimentConfig().threshold,
) -> MetricBundle:
    """Compute the predeclared binary metrics with no threshold optimization."""
    if threshold != ExperimentConfig().threshold:
        raise ValueError("Exp062 metrics use the fixed 0.5 threshold only")
    label_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    probability_array = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if label_array.size == 0 or label_array.size != probability_array.size:
        raise ValueError("labels and probabilities must be non-empty and equally sized")
    if not np.isin(label_array, (0, 1)).all():
        raise ValueError("labels must contain only binary 0/1 values")
    if not np.isfinite(probability_array).all() or not (
        (0.0 <= probability_array) & (probability_array <= 1.0)
    ).all():
        raise ValueError("probabilities must be finite values in [0, 1]")
    if np.unique(label_array).size != 2:
        raise ValueError("ROC-AUC and PR-AUC require both binary classes")

    predicted = (probability_array >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(label_array, predicted, labels=(0, 1)).ravel()
    high_confidence_fp = int(((label_array == 0) & (probability_array >= 0.9)).sum())
    high_confidence_fn = int(((label_array == 1) & (probability_array <= 0.1)).sum())
    f1 = float(f1_score(label_array, predicted, zero_division=0))
    pr_auc = float(average_precision_score(label_array, probability_array))

    return MetricBundle(
        accuracy=float(accuracy_score(label_array, predicted)),
        precision=float(precision_score(label_array, predicted, zero_division=0)),
        recall=float(recall_score(label_array, predicted, zero_division=0)),
        f1=f1,
        roc_auc=float(roc_auc_score(label_array, probability_array)),
        pr_auc=pr_auc,
        harmonic=harmonic_f1_pr(f1, pr_auc),
        bce=float(log_loss(label_array, probability_array, labels=(0, 1))),
        brier=float(brier_score_loss(label_array, probability_array)),
        tp=int(tp),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        high_confidence_fp=high_confidence_fp,
        high_confidence_fn=high_confidence_fn,
        high_confidence_errors=high_confidence_fp + high_confidence_fn,
    )


def _as_metric_bundle(summary: MetricBundle | Mapping[str, Any]) -> MetricBundle:
    if isinstance(summary, MetricBundle):
        return summary
    return MetricBundle(
        f1=float(summary["f1"]),
        pr_auc=float(summary["pr_auc"]),
        roc_auc=float(summary["roc_auc"]),
        harmonic=float(summary["harmonic"]),
        high_confidence_errors=float(summary["high_confidence_errors"]),
        per_seed_harmonic=tuple(float(item) for item in summary["per_seed_harmonic"]),
        invariants_pass=bool(summary.get("invariants_pass", True)),
    )


def evaluate_validation_gate(
    summary: MetricBundle | Mapping[str, Any],
    anchor_summary: MetricBundle | Mapping[str, Any],
) -> ValidationGateDecision:
    """Apply all ten approved validation predicates with their exact strictness."""
    candidate = _as_metric_bundle(summary)
    anchor = _as_metric_bundle(anchor_summary)
    if len(candidate.per_seed_harmonic) != len(anchor.per_seed_harmonic):
        raise ValueError("candidate and anchor must provide matching per-seed harmonic scores")
    if len(candidate.per_seed_harmonic) != len(ExperimentConfig().seeds):
        raise ValueError("validation gate requires all three frozen seed summaries")

    failures: list[str] = []
    if not candidate.f1 > 0.7075263341950917:
        failures.append("mean_f1_gt_0.7075263341950917")
    if not candidate.pr_auc > 0.7485129299280930:
        failures.append("mean_pr_auc_gt_0.7485129299280930")
    if not candidate.roc_auc > 0.9018675548671817:
        failures.append("mean_roc_auc_gt_0.9018675548671817")
    if not candidate.harmonic > 0.7274142468351926:
        failures.append("mean_harmonic_gt_0.7274142468351926")
    if not candidate.harmonic - anchor.harmonic >= 0.005:
        failures.append("harmonic_delta_gte_0.005")
    if candidate.f1 < anchor.f1 - 0.002:
        failures.append("mean_f1_not_below_anchor_by_more_than_0.002")
    if candidate.pr_auc < anchor.pr_auc - 0.002:
        failures.append("mean_pr_auc_not_below_anchor_by_more_than_0.002")
    if not (candidate.f1 - anchor.f1 > 0.005 or candidate.pr_auc - anchor.pr_auc > 0.005):
        failures.append("f1_or_pr_auc_delta_gt_0.005")
    if any(
        candidate_value - anchor_value < -0.005
        for candidate_value, anchor_value in zip(
            candidate.per_seed_harmonic, anchor.per_seed_harmonic
        )
    ):
        failures.append("no_seed_harmonic_delta_below_negative_0.005")
    if candidate.high_confidence_errors > anchor.high_confidence_errors:
        failures.append("high_confidence_errors_not_above_anchor")
    if not candidate.invariants_pass:
        failures.append("invariants_pass")
    return ValidationGateDecision(not failures, tuple(failures))


def _source_manifest_sha256(source_manifest: Mapping[str, Any] | Path) -> str:
    if isinstance(source_manifest, Path):
        return sha256_file(source_manifest)
    return canonical_json_sha256(dict(source_manifest))


def authenticated_marker_payload(
    *,
    config: ExperimentConfig,
    source_manifest: Mapping[str, Any] | Path,
    producing_command: str,
    input_artifacts: Mapping[str, Path],
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the complete provenance payload every phase marker must carry."""
    if not producing_command:
        raise ValueError("producing_command must not be empty")
    artifact_paths = {
        str(name): str(Path(path).resolve()) for name, path in sorted(input_artifacts.items())
    }
    artifacts = {
        name: sha256_file(Path(path)) for name, path in artifact_paths.items()
    }
    return {
        "marker_schema": "exp062-authenticated-phase-marker-v1",
        "config_sha256": experiment_config_sha256(config),
        "source_manifest_sha256": _source_manifest_sha256(source_manifest),
        "producing_command_sha256": hashlib.sha256(
            producing_command.encode("utf-8")
        ).hexdigest(),
        "input_artifacts": artifacts,
        "input_artifact_paths": artifact_paths,
        "payload": dict(payload or {}),
    }


def verify_authenticated_json_marker(
    path: Path,
    *,
    config: ExperimentConfig,
    source_manifest: Mapping[str, Any] | Path,
    allowed_schemas: Sequence[str] = (
        "exp062-authenticated-phase-marker-v1",
        "exp062-authenticated-final-freeze-v1",
    ),
) -> dict[str, Any]:
    """Load one marker and re-hash every artifact it binds, failing closed."""
    marker_path = Path(path)
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"authenticated marker is missing or invalid: {marker_path}") from exc
    if not isinstance(marker, dict) or marker.get("marker_schema") not in allowed_schemas:
        raise ValueError("authenticated marker schema is invalid")
    if marker.get("config_sha256") != experiment_config_sha256(config):
        raise ValueError("authenticated marker config hash mismatch")
    if marker.get("source_manifest_sha256") != _source_manifest_sha256(source_manifest):
        raise ValueError("authenticated marker source-manifest hash mismatch")
    hashes = marker.get("input_artifacts")
    paths = marker.get("input_artifact_paths")
    if not isinstance(hashes, dict) or not isinstance(paths, dict) or set(hashes) != set(paths):
        raise ValueError("authenticated marker artifact bindings are malformed")
    for name, expected_hash in hashes.items():
        artifact = Path(paths[name])
        if not artifact.is_file() or sha256_file(artifact) != expected_hash:
            raise ValueError(f"authenticated marker artifact hash mismatch: {name}")
    return marker


def _validation_summary_payload(summary: MetricBundle) -> dict[str, Any]:
    """Serialize exactly the values that are inputs to the ten frozen gate checks."""
    return {
        "f1": summary.f1,
        "pr_auc": summary.pr_auc,
        "roc_auc": summary.roc_auc,
        "harmonic": summary.harmonic,
        "high_confidence_errors": float(summary.high_confidence_errors),
        "per_seed_harmonic": list(summary.per_seed_harmonic),
        "invariants_pass": summary.invariants_pass,
    }


def _validation_summary_sha256(summary: MetricBundle) -> str:
    return canonical_json_sha256(_validation_summary_payload(summary))


def _checkpoint_hashes(
    checkpoint_artifacts: Mapping[int, Path], config: ExperimentConfig
) -> dict[str, str]:
    if set(checkpoint_artifacts) != set(config.seeds):
        raise ValueError("candidate manifest requires checkpoints for exactly the frozen seeds")
    return {
        str(seed): sha256_file(Path(checkpoint_artifacts[seed])) for seed in config.seeds
    }


def authenticated_candidate_manifest_payload(
    *,
    config: ExperimentConfig,
    source_manifest: Mapping[str, Any] | Path,
    winning_candidate: str,
    validation_summary: MetricBundle | Mapping[str, Any],
    anchor_summary: MetricBundle | Mapping[str, Any],
    checkpoint_artifacts: Mapping[int, Path],
    validation_artifacts: Mapping[str, Path],
) -> dict[str, Any]:
    """Canonical immutable binding between a selected candidate and its evidence."""
    if winning_candidate not in config.anchor_families:
        raise ValueError(f"unknown Exp062 anchor candidate: {winning_candidate}")
    candidate = _as_metric_bundle(validation_summary)
    anchor = _as_metric_bundle(anchor_summary)
    return {
        "manifest_schema": "exp062-candidate-manifest-v1",
        "config_sha256": experiment_config_sha256(config),
        "source_manifest_sha256": _source_manifest_sha256(source_manifest),
        "winning_candidate": winning_candidate,
        "seeds": list(config.seeds),
        "winning_checkpoint_sha256": _checkpoint_hashes(checkpoint_artifacts, config),
        "validation_summary": _validation_summary_payload(candidate),
        "anchor_summary": _validation_summary_payload(anchor),
        "validation_artifacts": {
            str(name): sha256_file(Path(path))
            for name, path in sorted(validation_artifacts.items())
        },
    }


def _atomic_json_write(value: Mapping[str, Any], path: Path) -> None:
    """Persist marker JSON atomically so an interrupted phase cannot look complete."""
    marker_path = Path(path)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=marker_path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, marker_path)
    finally:
        temporary.unlink(missing_ok=True)


def write_authenticated_json_marker(
    path: Path,
    *,
    config: ExperimentConfig,
    source_manifest: Mapping[str, Any] | Path,
    producing_command: str,
    input_artifacts: Mapping[str, Path],
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically write a canonical authenticated phase marker and return it."""
    marker = authenticated_marker_payload(
        config=config,
        source_manifest=source_manifest,
        producing_command=producing_command,
        input_artifacts=input_artifacts,
        payload=payload,
    )
    _atomic_json_write(marker, path)
    return marker


def write_authenticated_candidate_manifest(
    path: Path,
    *,
    config: ExperimentConfig,
    source_manifest: Mapping[str, Any] | Path,
    winning_candidate: str,
    validation_summary: MetricBundle | Mapping[str, Any],
    anchor_summary: MetricBundle | Mapping[str, Any],
    checkpoint_artifacts: Mapping[int, Path],
    validation_artifacts: Mapping[str, Path],
) -> dict[str, Any]:
    """Atomically create the candidate evidence artifact consumed by final freeze."""
    manifest = authenticated_candidate_manifest_payload(
        config=config,
        source_manifest=source_manifest,
        winning_candidate=winning_candidate,
        validation_summary=validation_summary,
        anchor_summary=anchor_summary,
        checkpoint_artifacts=checkpoint_artifacts,
        validation_artifacts=validation_artifacts,
    )
    _atomic_json_write(manifest, path)
    return manifest


def write_authenticated_final_freeze(
    path: Path,
    *,
    config: ExperimentConfig,
    source_manifest: Mapping[str, Any] | Path,
    producing_command: str,
    winning_candidate: str,
    validation_summary: MetricBundle | Mapping[str, Any],
    anchor_summary: MetricBundle | Mapping[str, Any],
    checkpoint_artifacts: Mapping[int, Path],
    validation_artifacts: Mapping[str, Path],
    candidate_manifest: Path,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write final freeze only when its candidate manifest exactly matches inputs."""
    candidate = _as_metric_bundle(validation_summary)
    anchor = _as_metric_bundle(anchor_summary)
    decision = evaluate_validation_gate(candidate, anchor)
    if not decision.eligible:
        raise ValueError("final freeze requires an eligible validation gate")

    expected_manifest = authenticated_candidate_manifest_payload(
        config=config,
        source_manifest=source_manifest,
        winning_candidate=winning_candidate,
        validation_summary=candidate,
        anchor_summary=anchor,
        checkpoint_artifacts=checkpoint_artifacts,
        validation_artifacts=validation_artifacts,
    )
    manifest_path = Path(candidate_manifest)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("candidate manifest is missing or invalid JSON") from exc
    if not isinstance(manifest, dict) or canonical_json_sha256(manifest) != canonical_json_sha256(
        expected_manifest
    ):
        raise ValueError("candidate manifest does not match final-freeze inputs")

    marker_inputs = dict(validation_artifacts)
    for seed in config.seeds:
        artifact_name = f"checkpoint_seed{seed}"
        if artifact_name in marker_inputs:
            raise ValueError(f"reserved final-freeze artifact name: {artifact_name}")
        checkpoint = Path(checkpoint_artifacts[seed])
        marker_inputs[artifact_name] = checkpoint
    if "candidate_manifest" in marker_inputs:
        raise ValueError("reserved final-freeze artifact name: candidate_manifest")
    marker_inputs["candidate_manifest"] = manifest_path

    marker = authenticated_marker_payload(
        config=config,
        source_manifest=source_manifest,
        producing_command=producing_command,
        input_artifacts=marker_inputs,
        payload=payload,
    )
    marker["marker_schema"] = "exp062-authenticated-final-freeze-v1"
    marker["final_freeze"] = {
        "winning_candidate": winning_candidate,
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "validation_gate": {
            "eligible": decision.eligible,
            "failed_conditions": list(decision.failed_conditions),
        },
        "validation_summary_sha256": _validation_summary_sha256(candidate),
        "anchor_summary_sha256": _validation_summary_sha256(anchor),
        "winning_checkpoint_sha256": expected_manifest["winning_checkpoint_sha256"],
        "winning_checkpoint_paths": {
            str(seed): str(Path(checkpoint_artifacts[seed]).resolve())
            for seed in config.seeds
        },
        "validation_artifact_sha256": expected_manifest["validation_artifacts"],
    }
    _atomic_json_write(marker, path)
    return marker


__all__ = [
    "ExperimentConfig",
    "AnchorComplexity",
    "AnchorParityResult",
    "MetricBundle",
    "ValidationGateDecision",
    "authenticated_candidate_manifest_payload",
    "authenticated_marker_payload",
    "canonical_json_sha256",
    "compute_binary_metrics",
    "evaluate_validation_gate",
    "experiment_config_sha256",
    "harmonic_f1_pr",
    "sha256_file",
    "verify_authenticated_json_marker",
    "write_authenticated_candidate_manifest",
    "write_authenticated_final_freeze",
    "write_authenticated_json_marker",
]
