"""Phase-gated Exp062 runner with fail-closed source and probability guards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import torch

from .anchors import (
    FoldModelArtifact,
    GroupedOOFResult,
    anchor_complexity,
    anchor_feature_metadata,
    build_grouped_oof,
    fit_full_anchor,
    verify_anchor_parity,
)
from .context_model import (
    HybridContextCorrector,
    predict_context_corrector,
    train_one_seed,
)
from .contracts import (
    AnchorComplexity,
    ExperimentConfig,
    MetricBundle,
    authenticated_candidate_manifest_payload,
    canonical_json_sha256,
    compute_binary_metrics,
    evaluate_validation_gate,
    sha256_file,
    verify_authenticated_json_marker,
    write_authenticated_candidate_manifest,
    write_authenticated_final_freeze,
    write_authenticated_json_marker,
)
from .data import (
    ContextScaler,
    TrainValData,
    fit_context_scaler,
    train_row_manifest_sha256,
    transform_context,
    write_context_scaler,
)
from .source_staging import (
    STAGED_DIRECTORY,
    STAGING_MARKER,
    load_staged_trainval,
    read_staged_validation_reference,
    stage_sources as build_source_staging,
    verify_source_staging,
)
from .runtime import assert_runtime_contract


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_VALIDATION_SOURCE = "authenticated-train-validation"
TEST_SOURCE = "set03-test"

PHASE_MARKERS = {
    "prepare": "PREPARED.json",
    "fit-anchors": "ANCHORS_READY.json",
    "train": "TRAINING_COMPLETE.json",
    "validate": "VALIDATION_COMPLETE.json",
    "final": "FINAL_EVALUATION_COMPLETE.json",
}
PRIOR_PHASE = {
    "prepare": "stage-sources",
    "fit-anchors": "prepare",
    "train": "fit-anchors",
    "validate": "train",
    "freeze": "validate",
    "final": "freeze",
    "verify": "terminal",
}


class PhaseOrderError(RuntimeError):
    """Raised when a phase is attempted without its authenticated predecessor."""


class ForbiddenSourceError(RuntimeError):
    """Raised before any forbidden test-source path is returned to a caller."""


class ArtifactAuthenticationError(RuntimeError):
    """Raised when persisted artifact identity or content has drifted."""


class VerificationError(RuntimeError):
    """Raised when terminal outputs cannot be certified."""


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_absent(paths: Sequence[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise PhaseOrderError(f"refusing to overwrite existing phase artifacts: {existing}")


def _start_log(path: Path, phase: str) -> None:
    _require_absent((path,))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"Exp062 {phase} started\n", encoding="utf-8")


def _finish_log(path: Path, phase: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"Exp062 {phase} completed\n")


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _string_array_sha256(values: Sequence[object] | np.ndarray) -> str:
    return canonical_json_sha256([str(value) for value in np.asarray(values).reshape(-1)])


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactAuthenticationError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ArtifactAuthenticationError(f"JSON artifact is not an object: {path}")
    return payload


@dataclass(frozen=True)
class ProbabilityBundleSpec:
    """Paths to one persisted Task 3 OOF/full-anchor probability binding."""

    root: Path
    anchor_name: str
    seed: int
    manifest_path: Path
    oof_path: Path
    validation_path: Path
    fold_audit_path: Path
    full_anchor_path: Path


@dataclass(frozen=True)
class AuthenticatedProbabilityVectors:
    """Opaque proof result required by the Task 4 training boundary."""

    anchor_name: str
    seed: int
    train_probabilities: np.ndarray
    validation_probabilities: np.ndarray
    train_sequence_ids: tuple[str, ...]
    train_labels: np.ndarray
    validation_sequence_ids: tuple[str, ...]
    validation_labels: np.ndarray
    manifest_sha256: str


def _validate_grouped_oof(result: GroupedOOFResult) -> None:
    if not isinstance(result, GroupedOOFResult):
        raise TypeError("training probabilities must come from a GroupedOOFResult")
    size = np.asarray(result.probabilities).reshape(-1).size
    if size == 0 or not np.isfinite(result.probabilities).all():
        raise ArtifactAuthenticationError("OOF probabilities must be finite and nonempty")
    if not ((0 <= result.probabilities) & (result.probabilities <= 1)).all():
        raise ArtifactAuthenticationError("OOF probabilities must be in [0, 1]")
    if np.asarray(result.assignment_count).shape != (size,) or not np.all(result.assignment_count == 1):
        raise ArtifactAuthenticationError("Grouped OOF must assign exactly one prediction per row")
    if np.asarray(result.fold_assignment).shape != (size,) or not np.all(
        (1 <= result.fold_assignment) & (result.fold_assignment <= 5)
    ):
        raise ArtifactAuthenticationError("Grouped OOF fold assignments are invalid")
    if "shared_track_count" not in result.fold_audit or len(result.fold_audit) != 5:
        raise ArtifactAuthenticationError("Grouped OOF track-leakage audit is incomplete")
    if not result.fold_audit["shared_track_count"].eq(0).all():
        raise ArtifactAuthenticationError("Grouped OOF track leakage was detected")
    if len(result.fold_artifacts) != 5 or {item.fold for item in result.fold_artifacts} != set(range(1, 6)):
        raise ArtifactAuthenticationError("Grouped OOF fold artifacts are incomplete")
    for artifact in result.fold_artifacts:
        if artifact.shared_track_count != 0:
            raise ArtifactAuthenticationError("Grouped OOF fold artifact reports track leakage")
        if not artifact.path.is_file() or sha256_file(artifact.path) != artifact.sha256:
            raise ArtifactAuthenticationError(f"Grouped OOF fold artifact hash mismatch: {artifact.fold}")


def persist_probability_bundle(
    *,
    root: Path,
    anchor_name: str,
    seed: int,
    grouped_oof: GroupedOOFResult,
    train_sequence_ids: Sequence[object] | np.ndarray,
    train_track_ids: Sequence[object] | np.ndarray,
    train_labels: Sequence[int] | np.ndarray,
    validation_sequence_ids: Sequence[object] | np.ndarray,
    validation_labels: Sequence[int] | np.ndarray,
    validation_probabilities: Sequence[float] | np.ndarray,
    parity_verified: bool,
    full_anchor_path: Path,
) -> ProbabilityBundleSpec:
    """Persist and bind Task 3 OOF plus parity-verified full-anchor vectors."""
    _validate_grouped_oof(grouped_oof)
    metadata = anchor_feature_metadata(anchor_name)
    config = ExperimentConfig()
    if seed not in config.seeds:
        raise ValueError("probability bundle seed is outside the frozen grid")
    train_ids = np.asarray(train_sequence_ids).astype(str).reshape(-1)
    track_ids = np.asarray(train_track_ids).astype(str).reshape(-1)
    train_label_array = np.asarray(train_labels, dtype=np.int64).reshape(-1)
    validation_ids = np.asarray(validation_sequence_ids).astype(str).reshape(-1)
    validation_label_array = np.asarray(validation_labels, dtype=np.int64).reshape(-1)
    validation_probability_array = np.asarray(validation_probabilities, dtype=np.float64).reshape(-1)
    if (
        train_ids.size != grouped_oof.probabilities.size
        or track_ids.size != train_ids.size
        or train_label_array.size != train_ids.size
        or not np.isin(train_label_array, (0, 1)).all()
    ):
        raise ValueError("OOF row identities must align with its probability vector")
    if len(set(train_ids)) != train_ids.size or len(set(validation_ids)) != validation_ids.size:
        raise ValueError("probability bundle sequence IDs must be unique")
    if validation_ids.size != validation_probability_array.size or validation_ids.size != validation_label_array.size:
        raise ValueError("validation identity, label, and probability rows must align")
    if not parity_verified:
        raise ArtifactAuthenticationError("validation vector must be parity verified")
    if not np.isfinite(validation_probability_array).all() or not (
        (0 <= validation_probability_array) & (validation_probability_array <= 1)
    ).all():
        raise ValueError("validation probabilities must be finite values in [0, 1]")
    full_anchor_path = Path(full_anchor_path).resolve()
    if not full_anchor_path.is_file():
        raise ArtifactAuthenticationError("full anchor artifact is missing")

    bundle_root = Path(root).resolve() / "probability_bundles" / f"{anchor_name}_seed{seed}"
    bundle_root.mkdir(parents=True, exist_ok=True)
    oof_path = bundle_root / "grouped_oof.npz"
    validation_path = bundle_root / "validation_full_anchor.csv"
    fold_audit_path = bundle_root / "oof_fold_audit.csv"
    manifest_path = bundle_root / "binding.json"
    for path in (oof_path, validation_path, fold_audit_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"probability bundle artifact already exists: {path}")

    with tempfile.NamedTemporaryFile(dir=bundle_root, delete=False) as handle:
        temporary = Path(handle.name)
        np.savez(
            handle,
            probabilities=np.asarray(grouped_oof.probabilities, dtype=np.float64),
            assignment_count=np.asarray(grouped_oof.assignment_count, dtype=np.int64),
            fold_assignment=np.asarray(grouped_oof.fold_assignment, dtype=np.int64),
            sequence_ids=train_ids,
            track_ids=track_ids,
            labels=train_label_array,
        )
    os.replace(temporary, oof_path)
    _atomic_csv(grouped_oof.fold_audit, fold_audit_path)
    _atomic_csv(
        pd.DataFrame(
            {
                "sequence_id": validation_ids,
                "label": validation_label_array,
                "probability": validation_probability_array,
            }
        ),
        validation_path,
    )
    fold_artifacts = {
        str(item.fold): {
            "path": str(item.path.resolve()),
            "sha256": item.sha256,
            "shared_track_count": item.shared_track_count,
        }
        for item in grouped_oof.fold_artifacts
    }
    manifest = {
        "schema": "exp062-authenticated-probability-bundle-v1",
        "anchor_name": anchor_name,
        "seed": seed,
        "feature_metadata": metadata,
        "parity_verified": True,
        "oof_path": str(oof_path),
        "oof_sha256": sha256_file(oof_path),
        "oof_probability_sha256": _array_sha256(grouped_oof.probabilities),
        "assignment_count_sha256": _array_sha256(grouped_oof.assignment_count),
        "fold_assignment_sha256": _array_sha256(grouped_oof.fold_assignment),
        "train_sequence_id_sha256": _string_array_sha256(train_ids),
        "train_track_id_sha256": _string_array_sha256(track_ids),
        "train_label_sha256": _array_sha256(train_label_array),
        "fold_audit_path": str(fold_audit_path),
        "fold_audit_sha256": sha256_file(fold_audit_path),
        "fold_artifacts": fold_artifacts,
        "validation_path": str(validation_path),
        "validation_sha256": sha256_file(validation_path),
        "validation_sequence_id_sha256": _string_array_sha256(validation_ids),
        "validation_label_sha256": _array_sha256(validation_label_array),
        "validation_probability_sha256": _array_sha256(validation_probability_array),
        "full_anchor_path": str(full_anchor_path),
        "full_anchor_sha256": sha256_file(full_anchor_path),
    }
    _atomic_json(manifest, manifest_path)
    return ProbabilityBundleSpec(
        root=Path(root).resolve(),
        anchor_name=anchor_name,
        seed=seed,
        manifest_path=manifest_path,
        oof_path=oof_path,
        validation_path=validation_path,
        fold_audit_path=fold_audit_path,
        full_anchor_path=full_anchor_path,
    )


def authenticate_probability_bundle(
    spec: ProbabilityBundleSpec,
    *,
    grouped_oof: GroupedOOFResult,
    expected_anchor: str | None = None,
    expected_seed: int | None = None,
) -> AuthenticatedProbabilityVectors:
    """Re-authenticate saved OOF/full-anchor vectors before neural training."""
    _validate_grouped_oof(grouped_oof)
    manifest = _read_json(spec.manifest_path)
    anchor_name = expected_anchor or spec.anchor_name
    seed = expected_seed if expected_seed is not None else spec.seed
    if (
        manifest.get("schema") != "exp062-authenticated-probability-bundle-v1"
        or manifest.get("anchor_name") != anchor_name
        or manifest.get("seed") != seed
        or spec.anchor_name != anchor_name
        or spec.seed != seed
    ):
        raise ArtifactAuthenticationError("probability bundle identity mismatch")
    if manifest.get("feature_metadata") != anchor_feature_metadata(anchor_name):
        raise ArtifactAuthenticationError("probability bundle feature identity mismatch")
    if manifest.get("parity_verified") is not True:
        raise ArtifactAuthenticationError("validation probability parity was not verified")
    path_bindings = {
        "OOF": (spec.oof_path, "oof_path", "oof_sha256"),
        "validation": (spec.validation_path, "validation_path", "validation_sha256"),
        "fold audit": (spec.fold_audit_path, "fold_audit_path", "fold_audit_sha256"),
        "full anchor": (spec.full_anchor_path, "full_anchor_path", "full_anchor_sha256"),
    }
    for name, (path, path_key, hash_key) in path_bindings.items():
        resolved = Path(path).resolve()
        if str(resolved) != manifest.get(path_key) or not resolved.is_file() or sha256_file(resolved) != manifest.get(hash_key):
            raise ArtifactAuthenticationError(f"{name} artifact hash mismatch")
    if _array_sha256(grouped_oof.probabilities) != manifest.get("oof_probability_sha256"):
        raise ArtifactAuthenticationError("OOF probability vector was substituted")
    if _array_sha256(grouped_oof.assignment_count) != manifest.get("assignment_count_sha256"):
        raise ArtifactAuthenticationError("OOF assignment vector was substituted")
    if _array_sha256(grouped_oof.fold_assignment) != manifest.get("fold_assignment_sha256"):
        raise ArtifactAuthenticationError("OOF fold vector was substituted")
    stored = np.load(spec.oof_path, allow_pickle=False)
    try:
        np.testing.assert_array_equal(stored["probabilities"], grouped_oof.probabilities)
        np.testing.assert_array_equal(stored["assignment_count"], grouped_oof.assignment_count)
        np.testing.assert_array_equal(stored["fold_assignment"], grouped_oof.fold_assignment)
    except AssertionError as exc:
        raise ArtifactAuthenticationError("saved OOF vector does not match GroupedOOFResult") from exc
    fold_manifest = manifest.get("fold_artifacts", {})
    for artifact in grouped_oof.fold_artifacts:
        expected = fold_manifest.get(str(artifact.fold), {})
        if (
            expected.get("path") != str(artifact.path.resolve())
            or expected.get("sha256") != artifact.sha256
            or sha256_file(artifact.path) != expected.get("sha256")
            or expected.get("shared_track_count") != 0
        ):
            raise ArtifactAuthenticationError("OOF fold artifact binding mismatch")
    validation = pd.read_csv(spec.validation_path, float_precision="round_trip")
    if validation.columns.tolist() != ["sequence_id", "label", "probability"]:
        raise ArtifactAuthenticationError("validation probability schema mismatch")
    validation_ids = validation["sequence_id"].astype(str).to_numpy()
    validation_labels = validation["label"].to_numpy(dtype=np.int64)
    validation_probabilities = validation["probability"].to_numpy(dtype=np.float64)
    if (
        _string_array_sha256(stored["sequence_ids"]) != manifest.get("train_sequence_id_sha256")
        or _string_array_sha256(stored["track_ids"]) != manifest.get("train_track_id_sha256")
        or _array_sha256(stored["labels"]) != manifest.get("train_label_sha256")
        or _string_array_sha256(validation_ids) != manifest.get("validation_sequence_id_sha256")
        or _array_sha256(validation_labels) != manifest.get("validation_label_sha256")
        or _array_sha256(validation_probabilities) != manifest.get("validation_probability_sha256")
    ):
        raise ArtifactAuthenticationError("validation or row-identity vector was substituted")
    return AuthenticatedProbabilityVectors(
        anchor_name=anchor_name,
        seed=seed,
        train_probabilities=np.asarray(grouped_oof.probabilities, dtype=np.float64).copy(),
        validation_probabilities=validation_probabilities.copy(),
        train_sequence_ids=tuple(str(value) for value in stored["sequence_ids"]),
        train_labels=np.asarray(stored["labels"], dtype=np.int64).copy(),
        validation_sequence_ids=tuple(validation_ids),
        validation_labels=validation_labels.copy(),
        manifest_sha256=sha256_file(spec.manifest_path),
    )


def run_authenticated_training(
    *,
    spec: ProbabilityBundleSpec,
    grouped_oof: GroupedOOFResult,
    train_ego: np.ndarray,
    train_yolop: np.ndarray,
    train_labels: np.ndarray,
    train_sequence_ids: Sequence[object] | np.ndarray,
    validation_ego: np.ndarray,
    validation_yolop: np.ndarray,
    validation_labels: np.ndarray,
    validation_sequence_ids: Sequence[object] | np.ndarray,
    checkpoint_path: Path,
    device: str | torch.device | None = None,
):
    """Call Task 4 only after authenticating both anchor-probability populations."""
    vectors = authenticate_probability_bundle(
        spec, grouped_oof=grouped_oof, expected_anchor=spec.anchor_name, expected_seed=spec.seed
    )
    if tuple(str(value) for value in train_sequence_ids) != vectors.train_sequence_ids:
        raise ArtifactAuthenticationError("training row identity does not match bound OOF order")
    if not np.array_equal(np.asarray(train_labels, dtype=np.int64), vectors.train_labels):
        raise ArtifactAuthenticationError("training labels do not match bound OOF labels")
    if tuple(str(value) for value in validation_sequence_ids) != vectors.validation_sequence_ids:
        raise ArtifactAuthenticationError("validation row identity does not match bound full-anchor order")
    if not np.array_equal(
        np.asarray(validation_labels, dtype=np.int64), vectors.validation_labels
    ):
        raise ArtifactAuthenticationError("validation labels do not match bound full-anchor labels")
    if len(vectors.train_probabilities) != len(train_labels) or len(vectors.validation_probabilities) != len(validation_labels):
        raise ArtifactAuthenticationError("authenticated probabilities do not align with labels")
    return train_one_seed(
        train_ego,
        train_yolop,
        vectors.train_probabilities,
        train_labels,
        validation_ego,
        validation_yolop,
        vectors.validation_probabilities,
        validation_labels,
        seed=spec.seed,
        device=device,
        checkpoint_path=checkpoint_path,
    )


class PhaseState:
    """Fail-closed phase and source authority rooted at one experiment directory."""

    def __init__(
        self,
        root: Path,
        *,
        config: ExperimentConfig | None = None,
        source_manifest: Mapping[str, Any] | Path | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.config = config or ExperimentConfig()
        self.source_manifest = (
            self.root / STAGING_MARKER if source_manifest is None else source_manifest
        )

    def authorize_source(self, source: str, *, phase: str) -> Path | str:
        if source == TEST_SOURCE:
            if phase != "final":
                raise ForbiddenSourceError(f"Set03 is forbidden during phase {phase}")
            self.require_final_authorization()
            return TEST_SOURCE
        if source == TRAIN_VALIDATION_SOURCE:
            if phase not in {"prepare", "fit-anchors", "train", "validate", "freeze"}:
                raise ForbiddenSourceError(f"train/validation source is not allowed during {phase}")
            verify_source_staging(self.root, config=self.config)
            return self.root / STAGED_DIRECTORY
        raise ForbiddenSourceError(f"unknown source: {source}")

    def _verify_marker(
        self, name: str, *, _seen: frozenset[Path] = frozenset()
    ) -> dict[str, Any]:
        marker_path = (self.root / name).resolve()
        if marker_path in _seen:
            raise ArtifactAuthenticationError("authenticated marker chain contains a cycle")
        try:
            marker = verify_authenticated_json_marker(
                marker_path,
                config=self.config,
                source_manifest=self.source_manifest,
            )
        except ValueError as exc:
            raise ArtifactAuthenticationError(str(exc)) from exc
        next_seen = _seen | {marker_path}
        for artifact_path in marker.get("input_artifact_paths", {}).values():
            candidate = Path(artifact_path).resolve()
            if candidate.parent != self.root or candidate.suffix not in {".json", ""}:
                continue
            try:
                nested = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(nested, dict) and str(nested.get("marker_schema", "")).startswith(
                "exp062-authenticated-"
            ):
                self._verify_marker(candidate.name, _seen=next_seen)
        return marker

    def require_prior(self, phase: str) -> dict[str, Any]:
        prior = PRIOR_PHASE.get(phase)
        if prior is None:
            return {}
        if prior == "terminal":
            return self.verify_terminal_state()
        if prior == "freeze":
            return self.require_final_authorization()
        if prior == "stage-sources":
            if not isinstance(self.source_manifest, Path):
                return {}
            try:
                return verify_source_staging(self.root, config=self.config)
            except RuntimeError as exc:
                raise PhaseOrderError("prepare requires authenticated SOURCE_STAGING.json") from exc
        marker_name = PHASE_MARKERS[prior]
        if not (self.root / marker_name).is_file():
            raise PhaseOrderError(f"{phase} requires {marker_name}")
        return self._verify_marker(marker_name)

    def write_phase_marker(
        self,
        phase: str,
        *,
        command: str,
        artifacts: Mapping[str, Path],
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if phase not in PHASE_MARKERS:
            raise ValueError(f"phase has no ordinary marker: {phase}")
        self.require_prior(phase)
        destination = self.root / PHASE_MARKERS[phase]
        if destination.exists():
            raise PhaseOrderError(f"phase marker already exists: {destination.name}")
        return write_authenticated_json_marker(
            destination,
            config=self.config,
            source_manifest=self.source_manifest,
            producing_command=command,
            input_artifacts=artifacts,
            payload={"phase": phase, **dict(payload or {})},
        )

    def _preflight_path_without_symlinks(self, path: Path, *, label: str) -> None:
        absolute = Path(os.path.abspath(os.fspath(path)))
        try:
            relative = absolute.relative_to(self.root)
        except ValueError as exc:
            raise ArtifactAuthenticationError(
                f"nested marker input escapes experiment root: {label}"
            ) from exc
        current = self.root
        for component in relative.parts:
            current = current / component
            if current.is_symlink():
                raise ArtifactAuthenticationError(
                    f"nested marker path is a symlink: {label}"
                )

    def _preflight_nested_marker_inputs(
        self, marker_path: Path, *, seen: frozenset[Path] = frozenset()
    ) -> None:
        absolute = Path(os.path.abspath(os.fspath(marker_path)))
        self._preflight_path_without_symlinks(absolute, label=absolute.name)
        if absolute in seen:
            raise ArtifactAuthenticationError("authenticated marker chain contains a cycle")
        marker = _read_json(absolute)
        if not str(marker.get("marker_schema", "")).startswith(
            "exp062-authenticated-"
        ):
            return
        paths = marker.get("input_artifact_paths")
        hashes = marker.get("input_artifacts")
        if not isinstance(paths, dict) or not isinstance(hashes, dict) or set(
            paths
        ) != set(hashes):
            raise ArtifactAuthenticationError(
                f"nested marker artifact bindings are malformed: {absolute.name}"
            )
        next_seen = seen | {absolute}
        for name, value in paths.items():
            try:
                candidate = Path(os.path.abspath(os.fspath(value)))
            except TypeError as exc:
                raise ArtifactAuthenticationError(
                    f"nested marker input path is invalid: {name}"
                ) from exc
            self._preflight_path_without_symlinks(candidate, label=str(name))
            if candidate.suffix == ".json":
                try:
                    nested = json.loads(candidate.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(nested, dict) and str(
                    nested.get("marker_schema", "")
                ).startswith("exp062-authenticated-"):
                    self._preflight_nested_marker_inputs(candidate, seen=next_seen)

    def _preflight_required_marker_chain(self) -> None:
        if not isinstance(self.source_manifest, Path):
            raise ArtifactAuthenticationError(
                "final authorization requires exact SOURCE_STAGING.json"
            )
        source_path = self.root / STAGING_MARKER
        if Path(os.path.abspath(self.source_manifest)) != Path(
            os.path.abspath(source_path)
        ):
            raise ArtifactAuthenticationError(
                "final authorization source staging path is invalid"
            )
        chain = (
            ("VALIDATION_COMPLETE.json", "TRAINING_COMPLETE.json"),
            ("TRAINING_COMPLETE.json", "ANCHORS_READY.json"),
            ("ANCHORS_READY.json", "PREPARED.json"),
            ("PREPARED.json", STAGING_MARKER),
        )
        chain_paths = {
            name: Path(os.path.abspath(self.root / name))
            for pair in chain
            for name in pair
        }
        for marker_name, predecessor_name in chain:
            marker_path = chain_paths[marker_name]
            self._preflight_path_without_symlinks(marker_path, label=marker_name)
            marker = _read_json(marker_path)
            if marker.get("marker_schema") != "exp062-authenticated-phase-marker-v1":
                raise ArtifactAuthenticationError(
                    f"required predecessor marker schema is invalid: {marker_name}"
                )
            paths = marker.get("input_artifact_paths")
            hashes = marker.get("input_artifacts")
            if not isinstance(paths, dict) or not isinstance(hashes, dict) or set(
                paths
            ) != set(hashes):
                raise ArtifactAuthenticationError(
                    f"required predecessor bindings are malformed: {marker_name}"
                )
            if predecessor_name not in paths:
                raise ArtifactAuthenticationError(
                    f"required predecessor is missing: {marker_name} -> {predecessor_name}"
                )
            expected_predecessor = chain_paths[predecessor_name]
            try:
                actual_predecessor = Path(
                    os.path.abspath(os.fspath(paths[predecessor_name]))
                )
            except TypeError as exc:
                raise ArtifactAuthenticationError(
                    f"predecessor path is invalid: {marker_name}"
                ) from exc
            if actual_predecessor != expected_predecessor:
                raise ArtifactAuthenticationError(
                    f"predecessor path is invalid: {marker_name} -> {predecessor_name}"
                )
            if sum(
                Path(os.path.abspath(os.fspath(value))) == expected_predecessor
                for value in paths.values()
            ) != 1:
                raise ArtifactAuthenticationError(
                    f"required predecessor must be present exactly once: {marker_name}"
                )
            for name, value in paths.items():
                candidate = Path(os.path.abspath(os.fspath(value)))
                for chain_name, chain_path in chain_paths.items():
                    if candidate == chain_path and chain_name != predecessor_name:
                        raise ArtifactAuthenticationError(
                            f"extra predecessor is forbidden: {marker_name} -> {chain_name}"
                        )
        self._preflight_nested_marker_inputs(self.root / "VALIDATION_COMPLETE.json")

    def require_final_authorization(self) -> dict[str, Any]:
        freeze = self.root / "FROZEN_FINAL_MODEL.json"
        validation_only = self.root / "VALIDATION_ONLY_COMPLETE"
        if validation_only.is_symlink():
            raise ArtifactAuthenticationError(
                "validation-only marker must not be a symlink"
            )
        if validation_only.exists():
            raise PhaseOrderError("validation-only closure forbids final evaluation")
        if freeze.is_symlink():
            raise ArtifactAuthenticationError("final freeze must not be a symlink")
        if not freeze.is_file():
            raise PhaseOrderError("final requires FROZEN_FINAL_MODEL.json")
        raw_marker = _read_json(freeze)
        raw_final = raw_marker.get("final_freeze")
        raw_payload = raw_marker.get("payload")
        if raw_marker.get("marker_schema") != "exp062-authenticated-final-freeze-v1":
            raise ArtifactAuthenticationError(
                "final authorization requires specialized freeze schema"
            )
        if not isinstance(raw_final, dict) or not isinstance(raw_payload, dict):
            raise ArtifactAuthenticationError("final freeze payload is malformed")
        winner = raw_final.get("winning_candidate")
        if winner not in self.config.anchor_families:
            raise ArtifactAuthenticationError("final freeze winning candidate is invalid")
        if raw_payload.get("final_command_arguments") != [
            "final",
            "--threshold",
            "0.5",
        ]:
            raise ArtifactAuthenticationError(
                "final command arguments are unsupported or mutated"
            )

        expected_seed_keys = {str(seed) for seed in self.config.seeds}
        raw_checkpoints = raw_final.get("winning_checkpoint_paths")
        raw_anchor_paths = raw_payload.get("anchor_paths")
        if (
            not isinstance(raw_checkpoints, dict)
            or set(raw_checkpoints) != expected_seed_keys
            or not isinstance(raw_anchor_paths, dict)
            or set(raw_anchor_paths) != expected_seed_keys
        ):
            raise ArtifactAuthenticationError("final freeze model paths are incomplete")

        def _lexical_absolute(value: object) -> Path:
            try:
                return Path(os.path.abspath(os.fspath(value)))
            except TypeError as exc:
                raise ArtifactAuthenticationError(
                    "final freeze artifact path is invalid"
                ) from exc

        expected_validation_paths: dict[str, Path] = {
            "VALIDATION_COMPLETE.json": self.root / "VALIDATION_COMPLETE.json",
            "metrics/validation_summary_results.csv": self.root
            / "metrics"
            / "validation_summary_results.csv",
            "metrics/validation_per_seed_results.csv": self.root
            / "metrics"
            / "validation_per_seed_results.csv",
            "metrics/paired_deltas.csv": self.root / "metrics" / "paired_deltas.csv",
            "metrics/error_transitions_fixed_0p5.csv": self.root
            / "metrics"
            / "error_transitions_fixed_0p5.csv",
            "metrics/context_ablation_diagnostics.csv": self.root
            / "metrics"
            / "context_ablation_diagnostics.csv",
            "metrics/context_scaler.json": self.root / "metrics" / "context_scaler.json",
            "logs/freeze.log": self.root / "logs" / "freeze.log",
            **{
                f"models/anchor_seed{seed}": self.root
                / "models"
                / f"{winner}_seed{seed}.joblib"
                for seed in self.config.seeds
            },
            **{
                f"predictions/validation_{anchor}_seed{seed}.csv": self.root
                / "predictions"
                / f"validation_{anchor}_seed{seed}.csv"
                for anchor in self.config.anchor_families
                for seed in self.config.seeds
            },
        }
        validation_hashes = raw_final.get("validation_artifact_sha256")
        if not isinstance(validation_hashes, dict) or set(validation_hashes) != set(
            expected_validation_paths
        ):
            raise ArtifactAuthenticationError(
                "final freeze validation artifacts are malformed"
            )
        expected_input_paths = {
            **expected_validation_paths,
            **{
                f"checkpoint_seed{seed}": self.root
                / "checkpoints"
                / f"{winner}_seed{seed}.pt"
                for seed in self.config.seeds
            },
            "candidate_manifest": self.root / "FROZEN_CANDIDATE_MANIFEST.json",
        }
        raw_input_paths = raw_marker.get("input_artifact_paths")
        raw_input_hashes = raw_marker.get("input_artifacts")
        if (
            not isinstance(raw_input_paths, dict)
            or set(raw_input_paths) != set(expected_input_paths)
            or not isinstance(raw_input_hashes, dict)
            or set(raw_input_hashes) != set(expected_input_paths)
        ):
            raise ArtifactAuthenticationError("final freeze artifact bindings are malformed")
        for name, expected_path in expected_input_paths.items():
            if _lexical_absolute(raw_input_paths[name]) != _lexical_absolute(expected_path):
                raise ArtifactAuthenticationError(
                    f"final freeze artifact path is invalid: {name}"
                )
            current = self.root
            for component in expected_path.relative_to(self.root).parts:
                current = current / component
                if current.is_symlink():
                    raise ArtifactAuthenticationError(
                        f"final freeze artifact path is a symlink: {name}"
                    )
        for seed in self.config.seeds:
            if _lexical_absolute(raw_checkpoints[str(seed)]) != _lexical_absolute(
                expected_input_paths[f"checkpoint_seed{seed}"]
            ) or _lexical_absolute(raw_anchor_paths[str(seed)]) != _lexical_absolute(
                expected_validation_paths[f"models/anchor_seed{seed}"]
            ):
                raise ArtifactAuthenticationError(
                    f"final freeze model path is invalid for seed {seed}"
                )
        if _lexical_absolute(raw_payload.get("scaler_path")) != _lexical_absolute(
            expected_validation_paths["metrics/context_scaler.json"]
        ):
            raise ArtifactAuthenticationError("final freeze scaler path is invalid")

        self._preflight_required_marker_chain()
        marker = self._verify_marker(freeze.name)
        if marker.get("marker_schema") != "exp062-authenticated-final-freeze-v1":
            raise ArtifactAuthenticationError("final authorization requires specialized freeze schema")
        if isinstance(self.source_manifest, Path):
            try:
                verify_source_staging(self.root, config=self.config)
            except RuntimeError as exc:
                raise ArtifactAuthenticationError(
                    "source staging failed final authorization"
                ) from exc
        final = marker.get("final_freeze")
        if not isinstance(final, dict) or final.get("validation_gate") != {
            "eligible": True,
            "failed_conditions": [],
        }:
            raise ArtifactAuthenticationError("final freeze does not prove all ten gates")
        winner = final.get("winning_candidate")
        if winner not in self.config.anchor_families:
            raise ArtifactAuthenticationError("final freeze winning candidate is invalid")
        checkpoints = final.get("winning_checkpoint_paths")
        hashes = final.get("winning_checkpoint_sha256")
        if (
            not isinstance(checkpoints, dict)
            or set(checkpoints) != expected_seed_keys
            or not isinstance(hashes, dict)
            or set(hashes) != expected_seed_keys
        ):
            raise ArtifactAuthenticationError("final freeze checkpoint paths are incomplete")
        for seed, path in checkpoints.items():
            expected_path = self.root / "checkpoints" / f"{winner}_seed{seed}.pt"
            if Path(path).resolve() != expected_path.resolve() or not Path(path).is_file() or sha256_file(Path(path)) != hashes.get(seed):
                raise ArtifactAuthenticationError(f"checkpoint hash mismatch for seed {seed}")
        payload = marker.get("payload", {})
        if not isinstance(payload, dict):
            raise ArtifactAuthenticationError("final freeze payload is malformed")
        anchor_hashes = payload.get("anchor_sha256")
        if not isinstance(anchor_hashes, dict) or set(anchor_hashes) != expected_seed_keys:
            raise ArtifactAuthenticationError("final freeze anchor hashes are incomplete")
        if not isinstance(payload.get("scaler_sha256"), str):
            raise ArtifactAuthenticationError("final freeze scaler hash is missing")
        if payload.get("final_command_arguments") != ["final", "--threshold", "0.5"]:
            raise ArtifactAuthenticationError("final command arguments are unsupported or mutated")
        anchor_paths = payload.get("anchor_paths")
        if not isinstance(anchor_paths, dict) or set(anchor_paths) != expected_seed_keys:
            raise ArtifactAuthenticationError("final freeze anchor paths are incomplete")
        for seed, path in anchor_paths.items():
            expected_path = self.root / "models" / f"{winner}_seed{seed}.joblib"
            if Path(path).resolve() != expected_path.resolve() or not expected_path.is_file() or sha256_file(expected_path) != anchor_hashes[seed]:
                raise ArtifactAuthenticationError(f"anchor hash or path mismatch for seed {seed}")
        scaler_path = self.root / "metrics" / "context_scaler.json"
        if Path(str(payload.get("scaler_path", ""))).resolve() != scaler_path.resolve():
            raise ArtifactAuthenticationError("final freeze scaler path is invalid")
        if not scaler_path.is_file() or sha256_file(scaler_path) != payload.get("scaler_sha256"):
            raise ArtifactAuthenticationError("final freeze scaler hash mismatch")
        artifact_hashes = set(marker.get("input_artifacts", {}).values())
        if not set(anchor_hashes.values()).issubset(artifact_hashes) or payload["scaler_sha256"] not in artifact_hashes:
            raise ArtifactAuthenticationError("anchor or scaler hash is not bound by final freeze")

        validation_hashes = final.get("validation_artifact_sha256")
        input_paths = marker.get("input_artifact_paths", {})
        if not isinstance(validation_hashes, dict):
            raise ArtifactAuthenticationError("final freeze validation artifacts are malformed")
        validation_artifacts: dict[str, Path] = {}
        for name, expected_hash in validation_hashes.items():
            if name not in input_paths:
                raise ArtifactAuthenticationError(f"validation artifact path is unbound: {name}")
            path = Path(input_paths[name]).resolve()
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise ArtifactAuthenticationError(f"validation artifact hash mismatch: {name}")
            validation_artifacts[name] = path
        validation_marker = validation_artifacts.get("VALIDATION_COMPLETE.json")
        expected_validation_marker = self.root / "VALIDATION_COMPLETE.json"
        if validation_marker is None or validation_marker != expected_validation_marker.resolve():
            raise ArtifactAuthenticationError("final freeze must bind exact VALIDATION_COMPLETE.json")
        self._verify_marker("VALIDATION_COMPLETE.json")

        required_validation = {
            "VALIDATION_COMPLETE.json",
            "metrics/validation_summary_results.csv",
            "metrics/validation_per_seed_results.csv",
            "metrics/paired_deltas.csv",
            "metrics/error_transitions_fixed_0p5.csv",
            "metrics/context_ablation_diagnostics.csv",
            "metrics/context_scaler.json",
            *(f"models/anchor_seed{seed}" for seed in self.config.seeds),
            *(
                f"predictions/validation_{anchor}_seed{seed}.csv"
                for anchor in self.config.anchor_families
                for seed in self.config.seeds
            ),
        }
        if not required_validation.issubset(validation_artifacts):
            missing = sorted(required_validation.difference(validation_artifacts))
            raise ArtifactAuthenticationError(f"final freeze validation evidence is incomplete: {missing}")
        for name in required_validation:
            if name.startswith("models/anchor_seed"):
                seed = name.removeprefix("models/anchor_seed")
                expected_path = Path(anchor_paths[seed]).resolve()
            else:
                expected_path = (self.root / name).resolve()
            if validation_artifacts[name] != expected_path:
                raise ArtifactAuthenticationError(
                    f"final freeze validation artifact path is invalid: {name}"
                )

        try:
            recomputed_candidates, _, _ = _reconstruct_validation_candidates(
                self.root, self.config
            )
        except (OSError, KeyError, ValueError, VerificationError) as exc:
            raise ArtifactAuthenticationError(
                "authenticated validation summary/candidate evidence is malformed"
            ) from exc
        recomputed_selection = select_validation_candidate(recomputed_candidates)
        if recomputed_selection.winner != winner:
            raise ArtifactAuthenticationError(
                "declared winner does not match the recomputed winner"
            )
        expected_selection_payload = _candidate_selection_payload(
            recomputed_candidates, validation_artifacts, self.config
        )
        if canonical_json_sha256(payload.get("candidate_selection")) != canonical_json_sha256(
            expected_selection_payload
        ):
            raise ArtifactAuthenticationError(
                "both candidate families are not exactly bound by final freeze"
            )

        candidate_path_text = input_paths.get("candidate_manifest")
        expected_candidate_path = self.root / "FROZEN_CANDIDATE_MANIFEST.json"
        if candidate_path_text is None or Path(candidate_path_text).resolve() != expected_candidate_path.resolve():
            raise ArtifactAuthenticationError("candidate manifest path is invalid")
        candidate_manifest = _read_json(expected_candidate_path)
        if sha256_file(expected_candidate_path) != final.get("candidate_manifest_sha256"):
            raise ArtifactAuthenticationError("candidate manifest hash mismatch")
        checkpoint_artifacts = {
            seed: Path(checkpoints[str(seed)]) for seed in self.config.seeds
        }
        try:
            candidate_summary = MetricBundle(**{
                **candidate_manifest["validation_summary"],
                "per_seed_harmonic": tuple(candidate_manifest["validation_summary"]["per_seed_harmonic"]),
            })
            anchor_summary = MetricBundle(**{
                **candidate_manifest["anchor_summary"],
                "per_seed_harmonic": tuple(candidate_manifest["anchor_summary"]["per_seed_harmonic"]),
            })
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactAuthenticationError("candidate manifest validation summary is malformed") from exc
        decision = evaluate_validation_gate(candidate_summary, anchor_summary)
        if not decision.eligible or not candidate_summary.invariants_pass:
            raise ArtifactAuthenticationError("candidate manifest does not recompute all ten gates")
        recomputed_candidate, recomputed_anchor = recomputed_candidates[winner]
        if (
            canonical_json_sha256(_gate_summary_payload(candidate_summary))
            != canonical_json_sha256(_gate_summary_payload(recomputed_candidate))
            or canonical_json_sha256(_gate_summary_payload(anchor_summary))
            != canonical_json_sha256(_gate_summary_payload(recomputed_anchor))
        ):
            raise ArtifactAuthenticationError(
                "candidate manifest does not match prediction-derived summaries"
            )
        expected_candidate = authenticated_candidate_manifest_payload(
            config=self.config,
            source_manifest=self.source_manifest,
            winning_candidate=winner,
            validation_summary=candidate_summary,
            anchor_summary=anchor_summary,
            checkpoint_artifacts=checkpoint_artifacts,
            validation_artifacts=validation_artifacts,
        )
        if canonical_json_sha256(candidate_manifest) != canonical_json_sha256(expected_candidate):
            raise ArtifactAuthenticationError("candidate manifest does not reconstruct from freeze inputs")
        summary_path = validation_artifacts["metrics/validation_summary_results.csv"]
        try:
            summary_frame = pd.read_csv(summary_path)
            persisted_candidate = _summary_bundle(summary_frame, winner, "corrected")
            persisted_anchor = _summary_bundle(summary_frame, winner, "anchor")
        except Exception as exc:
            raise ArtifactAuthenticationError("validation summary evidence is malformed") from exc
        if (
            canonical_json_sha256(asdict(persisted_candidate))
            != canonical_json_sha256(asdict(candidate_summary))
            or canonical_json_sha256(asdict(persisted_anchor))
            != canonical_json_sha256(asdict(anchor_summary))
        ):
            raise ArtifactAuthenticationError("validation summary does not match candidate manifest")
        if final.get("validation_summary_sha256") != canonical_json_sha256(
            {
                "f1": candidate_summary.f1,
                "pr_auc": candidate_summary.pr_auc,
                "roc_auc": candidate_summary.roc_auc,
                "harmonic": candidate_summary.harmonic,
                "high_confidence_errors": candidate_summary.high_confidence_errors,
                "per_seed_harmonic": list(candidate_summary.per_seed_harmonic),
                "invariants_pass": candidate_summary.invariants_pass,
            }
        ):
            raise ArtifactAuthenticationError("specialized freeze validation summary hash mismatch")
        if final.get("anchor_summary_sha256") != canonical_json_sha256(
            {
                "f1": anchor_summary.f1,
                "pr_auc": anchor_summary.pr_auc,
                "roc_auc": anchor_summary.roc_auc,
                "harmonic": anchor_summary.harmonic,
                "high_confidence_errors": anchor_summary.high_confidence_errors,
                "per_seed_harmonic": list(anchor_summary.per_seed_harmonic),
                "invariants_pass": anchor_summary.invariants_pass,
            }
        ):
            raise ArtifactAuthenticationError("specialized freeze anchor summary hash mismatch")
        if payload.get("validation_summary_file_sha256") != sha256_file(summary_path):
            raise ArtifactAuthenticationError("validation summary file digest mismatch")
        return marker

    def verify_terminal_state(self) -> dict[str, Any]:
        validation_only = self.root / "VALIDATION_ONLY_COMPLETE"
        freeze = self.root / "FROZEN_FINAL_MODEL.json"
        final = self.root / PHASE_MARKERS["final"]
        if validation_only.exists() and freeze.exists():
            raise VerificationError("VALIDATION_ONLY_COMPLETE and final freeze are mutually exclusive")
        if validation_only.exists():
            if final.exists():
                raise VerificationError("validation-only experiment cannot have final outputs")
            return self._verify_marker(validation_only.name)
        if freeze.exists():
            self.require_final_authorization()
            if not final.exists():
                raise VerificationError("frozen experiment is missing final evaluation marker")
            return self._verify_marker(final.name)
        raise VerificationError("experiment has no valid terminal decision")

    def record_verification(self, *, passed: bool, evidence: Mapping[str, Path]) -> dict[str, Any]:
        complete = self.root / "EXPERIMENT_COMPLETE"
        if not passed:
            if complete.exists():
                try:
                    self._verify_marker("EXPERIMENT_COMPLETE")
                except ArtifactAuthenticationError:
                    complete.unlink(missing_ok=True)
            raise VerificationError("failed verifier cannot coexist with EXPERIMENT_COMPLETE")
        terminal = self.verify_terminal_state()
        if complete.exists():
            existing = self._verify_marker("EXPERIMENT_COMPLETE")
            if existing.get("payload", {}).get("verifier_passed") is not True:
                raise VerificationError("existing completion marker is not verifier-authentic")
            return existing
        terminal_name = (
            "VALIDATION_ONLY_COMPLETE"
            if (self.root / "VALIDATION_ONLY_COMPLETE").exists()
            else PHASE_MARKERS["final"]
        )
        artifacts = {terminal_name: self.root / terminal_name, **dict(evidence)}
        return write_authenticated_json_marker(
            complete,
            config=self.config,
            source_manifest=self.source_manifest,
            producing_command="runner.py verify",
            input_artifacts=artifacts,
            payload={"verifier_passed": True, "terminal_marker_sha256": sha256_file(self.root / terminal_name)},
        )

    def verify_execution_invariants(self) -> bool:
        """Recompute staged-source, parity, grouped-OOF, and phase-chain evidence."""
        verify_source_staging(self.root, config=self.config)
        self._verify_marker("TRAINING_COMPLETE.json")
        parity = pd.read_csv(self.root / "metrics" / "anchor_parity.csv")
        expected = {
            (anchor, seed)
            for anchor in self.config.anchor_families
            for seed in self.config.seeds
        }
        actual = set(
            parity[["anchor_name", "seed"]].itertuples(index=False, name=None)
        )
        passed = parity["passed"].astype(str).str.lower().eq("true")
        if len(parity) != len(expected) or actual != expected or not passed.all():
            raise ArtifactAuthenticationError("anchor parity invariants failed")
        audit = pd.read_csv(self.root / "metrics" / "oof_fold_audit.csv")
        expected_folds = {
            (anchor, seed, fold)
            for anchor in self.config.anchor_families
            for seed in self.config.seeds
            for fold in range(1, 6)
        }
        actual_folds = set(
            audit[["anchor_name", "seed", "fold"]].itertuples(index=False, name=None)
        )
        if (
            len(audit) != len(expected_folds)
            or actual_folds != expected_folds
            or not audit["shared_track_count"].eq(0).all()
        ):
            raise ArtifactAuthenticationError("grouped OOF leakage invariants failed")
        return True


@dataclass(frozen=True)
class CandidateSelection:
    winner: str | None
    eligible: tuple[str, ...]
    decisions: Mapping[str, Any]


def select_validation_candidate(
    candidates: Mapping[str, tuple[MetricBundle, MetricBundle]],
) -> CandidateSelection:
    """Apply the ten gates and exact harmonic/PR/F1/error tie ordering."""
    config = ExperimentConfig()
    if set(candidates) != set(config.anchor_families):
        raise ValueError("selection requires both frozen anchor candidates")
    decisions = {
        name: evaluate_validation_gate(candidate, anchor)
        for name, (candidate, anchor) in candidates.items()
    }
    eligible = tuple(name for name in config.anchor_families if decisions[name].eligible)
    if not eligible:
        return CandidateSelection(None, (), decisions)
    winner = max(
        eligible,
        key=lambda name: (
            candidates[name][0].harmonic,
            candidates[name][0].pr_auc,
            candidates[name][0].f1,
            -candidates[name][0].high_confidence_errors,
        ),
    )
    return CandidateSelection(winner, eligible, decisions)


def _metric_record(bundle: MetricBundle) -> dict[str, Any]:
    record = asdict(bundle)
    record["per_seed_harmonic"] = json.dumps(record["per_seed_harmonic"])
    return record


_METRIC_COLUMNS = (
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

_COUNT_COLUMNS = (
    "tp",
    "tn",
    "fp",
    "fn",
    "high_confidence_fp",
    "high_confidence_fn",
)

_PER_SEED_METRIC_COLUMNS = (*_METRIC_COLUMNS, *_COUNT_COLUMNS)


def _aggregate_metric_rows(
    per_seed: pd.DataFrame,
    *,
    include_validation_evidence: bool,
    include_counts: bool = False,
    include_std: bool = False,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (anchor_name, variant), group in per_seed.groupby(
        ["anchor_name", "variant"], sort=False
    ):
        record: dict[str, Any] = {
            "anchor_name": anchor_name,
            "variant": variant,
            "seed_count": int(group["seed"].nunique()),
        }
        columns = _PER_SEED_METRIC_COLUMNS if include_counts else _METRIC_COLUMNS
        for column in columns:
            record[column] = float(group[column].mean())
            if include_std:
                record[f"{column}_std"] = float(group[column].std(ddof=0))
        if include_validation_evidence:
            record["per_seed_harmonic"] = json.dumps(
                group.sort_values("seed")["harmonic"].tolist()
            )
            record["invariants_pass"] = bool(
                group["invariants_pass"].astype(bool).all()
            )
        records.append(record)
    return pd.DataFrame(records)


def _gate_summary_payload(bundle: MetricBundle) -> dict[str, Any]:
    return {
        "f1": bundle.f1,
        "pr_auc": bundle.pr_auc,
        "roc_auc": bundle.roc_auc,
        "harmonic": bundle.harmonic,
        "high_confidence_errors": float(bundle.high_confidence_errors),
        "per_seed_harmonic": list(bundle.per_seed_harmonic),
        "invariants_pass": bundle.invariants_pass,
    }


def _candidate_selection_payload(
    candidates: Mapping[str, tuple[MetricBundle, MetricBundle]],
    validation_artifacts: Mapping[str, Path],
    config: ExperimentConfig,
) -> dict[str, Any]:
    selection = select_validation_candidate(candidates)
    families: dict[str, Any] = {}
    for family in config.anchor_families:
        candidate, anchor = candidates[family]
        prediction_hashes = {
            str(seed): sha256_file(
                validation_artifacts[
                    f"predictions/validation_{family}_seed{seed}.csv"
                ]
            )
            for seed in config.seeds
        }
        decision = selection.decisions[family]
        families[family] = {
            "candidate_summary_sha256": canonical_json_sha256(
                _gate_summary_payload(candidate)
            ),
            "anchor_summary_sha256": canonical_json_sha256(
                _gate_summary_payload(anchor)
            ),
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


def _transition_record(
    labels: np.ndarray,
    anchor_probabilities: np.ndarray,
    corrected_probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, int]:
    anchor_class = np.asarray(anchor_probabilities) >= threshold
    corrected_class = np.asarray(corrected_probabilities) >= threshold
    labels = np.asarray(labels, dtype=np.int64)
    return {
        "anchor_fp_to_corrected_tn": int(
            ((labels == 0) & anchor_class & ~corrected_class).sum()
        ),
        "anchor_fn_to_corrected_tp": int(
            ((labels == 1) & ~anchor_class & corrected_class).sum()
        ),
        "anchor_tn_to_corrected_fp": int(
            ((labels == 0) & ~anchor_class & corrected_class).sum()
        ),
        "anchor_tp_to_corrected_fn": int(
            ((labels == 1) & anchor_class & ~corrected_class).sum()
        ),
    }


def _upsert(frame: pd.DataFrame, path: Path, keys: Sequence[str]) -> pd.DataFrame:
    if path.is_file():
        existing = pd.read_csv(path)
        mask = np.ones(len(existing), dtype=bool)
        for key in keys:
            mask &= existing[key].astype(str).eq(str(frame.iloc[0][key])).to_numpy()
        existing = existing.loc[~mask]
        frame = pd.concat([existing, frame], ignore_index=True)
    _atomic_csv(frame, path)
    return frame


def write_validation_outputs(
    *,
    root: Path,
    anchor_name: str,
    seed: int,
    rows: pd.DataFrame,
    anchor_probabilities: np.ndarray,
    corrected_probabilities: np.ndarray,
    checkpoint_epoch: int,
    gate: np.ndarray,
    applied_correction: np.ndarray,
    ablation_probabilities: Mapping[str, np.ndarray],
    invariants_pass: bool,
) -> dict[str, Path]:
    """Persist one seed's fixed-0.5 validation result and diagnostics."""
    expected_ablations = {"zero_ego", "zero_yolop", "reversed_ego", "reversed_yolop"}
    if set(ablation_probabilities) != expected_ablations:
        raise ValueError("all four frozen validation ablations are required")
    labels = rows["label"].to_numpy(dtype=np.int64)
    ids = rows["sequence_id"].astype(str).to_numpy()
    anchor_probabilities = np.asarray(anchor_probabilities, dtype=np.float64).reshape(-1)
    corrected_probabilities = np.asarray(corrected_probabilities, dtype=np.float64).reshape(-1)
    if len(ids) != len(anchor_probabilities) or len(ids) != len(corrected_probabilities) or len(set(ids)) != len(ids):
        raise ValueError("validation predictions must align to unique row identities")
    anchor_metric = compute_binary_metrics(labels, anchor_probabilities)
    corrected_metric = compute_binary_metrics(labels, corrected_probabilities)
    root = Path(root)
    prediction_path = root / "predictions" / f"validation_{anchor_name}_seed{seed}.csv"
    _atomic_csv(
        pd.DataFrame(
            {
                "sequence_id": ids,
                "label": labels,
                "anchor_probability": anchor_probabilities,
                "corrected_probability": corrected_probabilities,
                "corrected_class_fixed_0p5": (corrected_probabilities >= 0.5).astype(int),
            }
        ),
        prediction_path,
    )
    per_seed_path = root / "metrics" / "validation_per_seed_results.csv"
    diagnostics = {
        "gate_mean": float(np.mean(gate)),
        "gate_min": float(np.min(gate)),
        "gate_max": float(np.max(gate)),
        "mean_absolute_applied_correction": float(np.mean(np.abs(applied_correction))),
        "max_absolute_applied_correction": float(np.max(np.abs(applied_correction))),
    }
    per_seed_rows = []
    for variant, metric in (("anchor", anchor_metric), ("corrected", corrected_metric)):
        record = {"anchor_name": anchor_name, "seed": seed, "variant": variant, "checkpoint_epoch": checkpoint_epoch, "invariants_pass": bool(invariants_pass)}
        record.update(_metric_record(metric))
        record.update(diagnostics if variant == "corrected" else {key: np.nan for key in diagnostics})
        per_seed_rows.append(record)
    per_seed = _upsert(pd.DataFrame(per_seed_rows), per_seed_path, ("anchor_name", "seed"))
    summary_path = root / "metrics" / "validation_summary_results.csv"
    _atomic_csv(
        _aggregate_metric_rows(per_seed, include_validation_evidence=True), summary_path
    )
    paired_path = root / "metrics" / "paired_deltas.csv"
    paired = pd.DataFrame(
        [{
            "anchor_name": anchor_name,
            "seed": seed,
            **{f"{name}_delta": getattr(corrected_metric, name) - getattr(anchor_metric, name) for name in ("f1", "pr_auc", "roc_auc", "harmonic", "bce", "brier")},
        }]
    )
    _upsert(paired, paired_path, ("anchor_name", "seed"))
    transitions_path = root / "metrics" / "error_transitions_fixed_0p5.csv"
    transitions = pd.DataFrame(
        [{
            "anchor_name": anchor_name,
            "seed": seed,
            **_transition_record(
                labels,
                anchor_probabilities,
                corrected_probabilities,
                threshold=0.5,
            ),
        }]
    )
    _upsert(transitions, transitions_path, ("anchor_name", "seed"))
    ablation_path = root / "metrics" / "context_ablation_diagnostics.csv"
    ablation_rows = []
    for name, probabilities in ablation_probabilities.items():
        metric = compute_binary_metrics(labels, probabilities)
        ablation_rows.append({"anchor_name": anchor_name, "seed": seed, "ablation": name, **_metric_record(metric)})
    if ablation_path.is_file():
        previous = pd.read_csv(ablation_path)
        previous = previous.loc[~((previous["anchor_name"] == anchor_name) & (previous["seed"] == seed))]
        ablation_frame = pd.concat([previous, pd.DataFrame(ablation_rows)], ignore_index=True)
    else:
        ablation_frame = pd.DataFrame(ablation_rows)
    _atomic_csv(ablation_frame, ablation_path)
    return {
        str(path.relative_to(root)): path
        for path in (per_seed_path, summary_path, paired_path, transitions_path, ablation_path, prediction_path)
    }


def _load_scaler(path: Path) -> ContextScaler:
    payload = _read_json(path)
    return ContextScaler(
        ego_mean=np.asarray(payload["ego_mean"], dtype=np.float64),
        ego_std=np.asarray(payload["ego_std"], dtype=np.float64),
        yolop_mean=np.asarray(payload["yolop_mean"], dtype=np.float64),
        yolop_std=np.asarray(payload["yolop_std"], dtype=np.float64),
    )


def _feature_matrix(data: TrainValData, anchor_name: str) -> np.ndarray:
    metadata = anchor_feature_metadata(anchor_name)
    expected = {
        "rf_v1_bbox": ("biu_v1_plus_bbox", data.summary_v1),
        "xgb_v2_bbox": ("biu_v2_core_plus_bbox", data.summary_v2_core),
    }
    feature_set, matrix = expected[anchor_name]
    if metadata != {
        "anchor_name": anchor_name,
        "feature_dimension": matrix.shape[1],
        "feature_set": feature_set,
    }:
        raise ArtifactAuthenticationError("anchor feature matrix ABI does not match Task 3 metadata")
    return matrix


def _bundle_spec(root: Path, anchor_name: str, seed: int) -> ProbabilityBundleSpec:
    bundle = root / "probability_bundles" / f"{anchor_name}_seed{seed}"
    manifest = _read_json(bundle / "binding.json")
    return ProbabilityBundleSpec(
        root=root,
        anchor_name=anchor_name,
        seed=seed,
        manifest_path=bundle / "binding.json",
        oof_path=Path(manifest["oof_path"]),
        validation_path=Path(manifest["validation_path"]),
        fold_audit_path=Path(manifest["fold_audit_path"]),
        full_anchor_path=Path(manifest["full_anchor_path"]),
    )


def _load_grouped_oof(spec: ProbabilityBundleSpec) -> GroupedOOFResult:
    manifest = _read_json(spec.manifest_path)
    stored = np.load(spec.oof_path, allow_pickle=False)
    fold_audit = pd.read_csv(spec.fold_audit_path)
    artifacts = []
    for fold_text, item in sorted(manifest["fold_artifacts"].items(), key=lambda pair: int(pair[0])):
        fold = int(fold_text)
        row = fold_audit.loc[fold_audit["fold"] == fold].iloc[0]
        artifacts.append(
            FoldModelArtifact(
                fold=fold,
                path=Path(item["path"]),
                sha256=item["sha256"],
                size_bytes=int(row.get("size_bytes", Path(item["path"]).stat().st_size)),
                complexity=AnchorComplexity(int(row.get("tree_count", 0)), int(row.get("node_count", 0)), int(row.get("leaf_count", 0))),
                train_rows=int(row.get("train_rows", 0)), holdout_rows=int(row.get("holdout_rows", 0)),
                train_positive_rows=int(row.get("train_positive_rows", 0)), holdout_positive_rows=int(row.get("holdout_positive_rows", 0)),
                train_negative_rows=int(row.get("train_negative_rows", 0)), holdout_negative_rows=int(row.get("holdout_negative_rows", 0)),
                train_unique_tracks=int(row.get("train_unique_tracks", 0)), holdout_unique_tracks=int(row.get("holdout_unique_tracks", 0)),
                shared_track_count=int(row["shared_track_count"]),
            )
        )
    return GroupedOOFResult(
        probabilities=stored["probabilities"], assignment_count=stored["assignment_count"],
        fold_assignment=stored["fold_assignment"], fold_audit=fold_audit,
        fold_artifacts=tuple(artifacts),
    )


def stage_source_inputs(
    root: Path = EXPERIMENT_ROOT,
    *,
    config: ExperimentConfig | None = None,
    build: Callable[[], Any] | None = None,
) -> dict[str, object]:
    """Run the sole mixed-container phase and create immutable isolated sources."""
    return build_source_staging(root, config=config or ExperimentConfig(), build=build)


def prepare(root: Path = EXPERIMENT_ROOT) -> dict[str, Any]:
    state = PhaseState(root)
    state.require_prior("prepare")
    state.authorize_source(TRAIN_VALIDATION_SOURCE, phase="prepare")
    scaler_path = root / "metrics" / "context_scaler.json"
    log_path = root / "logs" / "prepare.log"
    _require_absent((root / "PREPARED.json", scaler_path, log_path))
    for directory in ("logs", "metrics", "models", "checkpoints", "predictions", "probability_bundles"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    _start_log(log_path, "prepare")
    data = load_staged_trainval(root, config=state.config)
    write_context_scaler(scaler_path, data, fit_context_scaler(data))
    _finish_log(log_path, "prepare")
    return state.write_phase_marker(
        "prepare", command="runner.py prepare",
        artifacts={STAGING_MARKER: root / STAGING_MARKER, "metrics/context_scaler.json": scaler_path, "logs/prepare.log": log_path},
        payload={"train_rows": state.config.train_rows, "validation_rows": state.config.val_rows, "train_row_manifest_sha256": train_row_manifest_sha256(data)},
    )


def fit_anchors(root: Path = EXPERIMENT_ROOT) -> dict[str, Any]:
    state = PhaseState(root)
    state.require_prior("fit-anchors")
    state.authorize_source(TRAIN_VALIDATION_SOURCE, phase="fit-anchors")
    log_path = root / "logs" / "fit-anchors.log"
    fit_outputs = [
        root / "ANCHORS_READY.json",
        log_path,
        root / "metrics" / "anchor_parity.csv",
        root / "metrics" / "oof_fold_audit.csv",
    ]
    for anchor in state.config.anchor_families:
        for seed in state.config.seeds:
            fit_outputs.extend(
                [
                    root / "models" / f"{anchor}_seed{seed}.joblib",
                    root / "models" / "oof" / f"{anchor}_seed{seed}",
                    root / "probability_bundles" / f"{anchor}_seed{seed}",
                ]
            )
    _require_absent(tuple(fit_outputs))
    _start_log(log_path, "fit-anchors")
    data = load_staged_trainval(root, config=state.config)
    artifacts: dict[str, Path] = {"PREPARED.json": root / "PREPARED.json"}
    parity_rows: list[dict[str, Any]] = []
    audit_frames: list[pd.DataFrame] = []
    expected_validation = data.rows.loc[data.val_mask, ["sequence_id", "split", "label"]].reset_index(drop=True)
    for anchor_name in state.config.anchor_families:
        matrix = _feature_matrix(data, anchor_name)
        for seed in state.config.seeds:
            full_path = root / "models" / f"{anchor_name}_seed{seed}.joblib"
            fitted = fit_full_anchor(matrix[data.train_mask], data.labels[data.train_mask], anchor_name, seed, path=full_path)
            validation_probabilities = fitted.model.predict_proba(matrix[data.val_mask])[:, 1]
            fresh = expected_validation.assign(probability=validation_probabilities)
            historical = read_staged_validation_reference(
                root, anchor_name, seed, config=state.config
            )
            parity = verify_anchor_parity(fresh, historical)
            if not parity.passed:
                raise ArtifactAuthenticationError(f"validation anchor parity failed: {anchor_name} seed {seed}: {parity.failed_checks}")
            fold_dir = root / "models" / "oof" / f"{anchor_name}_seed{seed}"
            fold_dir.mkdir(parents=True, exist_ok=False)
            oof = build_grouped_oof(
                matrix[data.train_mask], data.labels[data.train_mask],
                data.rows.loc[data.train_mask, "track_id"].to_numpy(), anchor_name, seed,
                output_dir=fold_dir,
            )
            spec = persist_probability_bundle(
                root=root, anchor_name=anchor_name, seed=seed, grouped_oof=oof,
                train_sequence_ids=data.rows.loc[data.train_mask, "sequence_id"].to_numpy(),
                train_track_ids=data.rows.loc[data.train_mask, "track_id"].to_numpy(),
                train_labels=data.labels[data.train_mask],
                validation_sequence_ids=expected_validation["sequence_id"].to_numpy(),
                validation_labels=data.labels[data.val_mask], validation_probabilities=validation_probabilities,
                parity_verified=parity.passed, full_anchor_path=full_path,
            )
            parity_rows.append({"anchor_name": anchor_name, "seed": seed, **asdict(parity)})
            audit = oof.fold_audit.copy()
            audit.insert(0, "seed", seed)
            audit.insert(0, "anchor_name", anchor_name)
            audit_frames.append(audit)
            artifacts[f"models/{full_path.name}"] = full_path
            artifacts[f"probability_bundles/{anchor_name}_seed{seed}/binding.json"] = spec.manifest_path
            for fold in oof.fold_artifacts:
                artifacts[f"models/oof/{anchor_name}_seed{seed}/{fold.path.name}"] = fold.path
    parity_path = root / "metrics" / "anchor_parity.csv"
    audit_path = root / "metrics" / "oof_fold_audit.csv"
    _atomic_csv(pd.DataFrame(parity_rows), parity_path)
    _atomic_csv(pd.concat(audit_frames, ignore_index=True), audit_path)
    artifacts["metrics/anchor_parity.csv"] = parity_path
    artifacts["metrics/oof_fold_audit.csv"] = audit_path
    _finish_log(log_path, "fit-anchors")
    artifacts["logs/fit-anchors.log"] = log_path
    return state.write_phase_marker("fit-anchors", command="runner.py fit-anchors", artifacts=artifacts)


def train(
    root: Path = EXPERIMENT_ROOT,
    *,
    anchor_name: str,
    seed: int,
    device: str | None = None,
) -> dict[str, Any]:
    state = PhaseState(root)
    state.require_prior("train")
    state.authorize_source(TRAIN_VALIDATION_SOURCE, phase="train")
    if anchor_name not in state.config.anchor_families or seed not in state.config.seeds:
        raise ValueError("train requires a frozen anchor and seed")
    checkpoint = root / "checkpoints" / f"{anchor_name}_seed{seed}.pt"
    history_path = root / "metrics" / f"training_history_{anchor_name}_seed{seed}.csv"
    run_marker = root / f"TRAINED__{anchor_name}__seed{seed}.json"
    log_path = root / "logs" / f"train_{anchor_name}_seed{seed}.log"
    _require_absent((root / "TRAINING_COMPLETE.json", checkpoint, history_path, run_marker, log_path))
    _start_log(log_path, f"train {anchor_name} seed {seed}")
    data = transform_context(load_staged_trainval(root, config=state.config), _load_scaler(root / "metrics" / "context_scaler.json"))
    spec = _bundle_spec(root, anchor_name, seed)
    oof = _load_grouped_oof(spec)
    result = run_authenticated_training(
        spec=spec, grouped_oof=oof,
        train_ego=data.ego[data.train_mask], train_yolop=data.yolop[data.train_mask], train_labels=data.labels[data.train_mask],
        train_sequence_ids=data.rows.loc[data.train_mask, "sequence_id"].to_numpy(),
        validation_ego=data.ego[data.val_mask], validation_yolop=data.yolop[data.val_mask], validation_labels=data.labels[data.val_mask],
        validation_sequence_ids=data.rows.loc[data.val_mask, "sequence_id"].to_numpy(),
        checkpoint_path=checkpoint, device=device,
    )
    _atomic_csv(pd.DataFrame(result.history), history_path)
    _finish_log(log_path, f"train {anchor_name} seed {seed}")
    write_authenticated_json_marker(
        run_marker, config=state.config, source_manifest=state.source_manifest,
        producing_command=f"runner.py train --anchor {anchor_name} --seed {seed}",
        input_artifacts={"ANCHORS_READY.json": root / "ANCHORS_READY.json", "probability_binding": spec.manifest_path, "checkpoint": checkpoint, "history": history_path, "log": log_path},
        payload={"anchor_name": anchor_name, "seed": seed, "best_epoch": result.best_epoch, "best_score": result.best_score},
    )
    run_markers = {
        f"TRAINED__{name}__seed{run_seed}.json": root / f"TRAINED__{name}__seed{run_seed}.json"
        for name in state.config.anchor_families for run_seed in state.config.seeds
    }
    if all(path.is_file() for path in run_markers.values()) and not (root / PHASE_MARKERS["train"]).exists():
        state.write_phase_marker(
            "train",
            command="runner.py train (six-run aggregate)",
            artifacts={
                "ANCHORS_READY.json": root / "ANCHORS_READY.json",
                **run_markers,
            },
        )
    return _read_json(run_marker)


def validate(root: Path = EXPERIMENT_ROOT, *, device: str | None = None) -> dict[str, Any]:
    state = PhaseState(root)
    state.require_prior("validate")
    state.authorize_source(TRAIN_VALIDATION_SOURCE, phase="validate")
    log_path = root / "logs" / "validate.log"
    validation_outputs = (
        root / "metrics" / "validation_per_seed_results.csv",
        root / "metrics" / "validation_summary_results.csv",
        root / "metrics" / "paired_deltas.csv",
        root / "metrics" / "error_transitions_fixed_0p5.csv",
        root / "metrics" / "context_ablation_diagnostics.csv",
        *(
            root / "predictions" / f"validation_{anchor}_seed{seed}.csv"
            for anchor in state.config.anchor_families
            for seed in state.config.seeds
        ),
    )
    _require_absent((root / "VALIDATION_COMPLETE.json", log_path, *validation_outputs))
    _start_log(log_path, "validate")
    invariants_pass = state.verify_execution_invariants()
    data = transform_context(load_staged_trainval(root, config=state.config), _load_scaler(root / "metrics" / "context_scaler.json"))
    rows = data.rows.loc[data.val_mask, ["sequence_id", "label"]].reset_index(drop=True)
    artifacts: dict[str, Path] = {"TRAINING_COMPLETE.json": root / "TRAINING_COMPLETE.json"}
    for anchor_name in state.config.anchor_families:
        for seed in state.config.seeds:
            spec = _bundle_spec(root, anchor_name, seed)
            oof = _load_grouped_oof(spec)
            vectors = authenticate_probability_bundle(spec, grouped_oof=oof, expected_anchor=anchor_name, expected_seed=seed)
            checkpoint_path = root / "checkpoints" / f"{anchor_name}_seed{seed}.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if checkpoint.get("seed") != seed or checkpoint.get("total_parameters") != state.config.context_parameter_count:
                raise ArtifactAuthenticationError("checkpoint identity or architecture mismatch")
            model = HybridContextCorrector()
            model.load_state_dict(checkpoint["model_state"])
            ego, yolop = data.ego[data.val_mask], data.yolop[data.val_mask]
            prediction = predict_context_corrector(model, ego, yolop, vectors.validation_probabilities, device=device)
            ablations = {
                "zero_ego": predict_context_corrector(model, np.zeros_like(ego), yolop, vectors.validation_probabilities, device=device)["probability"],
                "zero_yolop": predict_context_corrector(model, ego, np.zeros_like(yolop), vectors.validation_probabilities, device=device)["probability"],
                "reversed_ego": predict_context_corrector(model, ego[:, ::-1].copy(), yolop, vectors.validation_probabilities, device=device)["probability"],
                "reversed_yolop": predict_context_corrector(model, ego, yolop[:, ::-1].copy(), vectors.validation_probabilities, device=device)["probability"],
            }
            generated = write_validation_outputs(
                root=root, anchor_name=anchor_name, seed=seed, rows=rows,
                anchor_probabilities=vectors.validation_probabilities,
                corrected_probabilities=prediction["probability"], checkpoint_epoch=int(checkpoint["best_epoch"]),
                gate=prediction["gate"], applied_correction=prediction["applied_correction"],
                ablation_probabilities=ablations,
                invariants_pass=invariants_pass,
            )
            artifacts.update(generated)
            artifacts[f"checkpoints/{checkpoint_path.name}"] = checkpoint_path
    _finish_log(log_path, "validate")
    artifacts["logs/validate.log"] = log_path
    return state.write_phase_marker("validate", command="runner.py validate", artifacts=artifacts)


def _summary_bundle(frame: pd.DataFrame, anchor_name: str, variant: str) -> MetricBundle:
    row = frame.loc[(frame["anchor_name"] == anchor_name) & (frame["variant"] == variant)]
    if len(row) != 1:
        raise VerificationError("validation summary must have one row per anchor and variant")
    item = row.iloc[0]
    per_seed = tuple(float(value) for value in json.loads(item["per_seed_harmonic"]))
    return MetricBundle(
        f1=float(item["f1"]), pr_auc=float(item["pr_auc"]), roc_auc=float(item["roc_auc"]),
        harmonic=float(item["harmonic"]), high_confidence_errors=float(item["high_confidence_errors"]),
        per_seed_harmonic=per_seed, invariants_pass=bool(item["invariants_pass"]),
    )


def freeze(root: Path = EXPERIMENT_ROOT) -> dict[str, Any]:
    state = PhaseState(root)
    state.require_prior("freeze")
    log_path = root / "logs" / "freeze.log"
    _require_absent(
        (
            root / "FROZEN_FINAL_MODEL.json",
            root / "VALIDATION_ONLY_COMPLETE",
            root / "FROZEN_CANDIDATE_MANIFEST.json",
            log_path,
        )
    )
    _start_log(log_path, "freeze")
    summary_path = root / "metrics" / "validation_summary_results.csv"
    candidates, _, _ = _reconstruct_validation_candidates(root, state.config)
    selection = select_validation_candidate(candidates)
    gate_payload = {
        name: {"eligible": decision.eligible, "failed_conditions": list(decision.failed_conditions)}
        for name, decision in selection.decisions.items()
    }
    if selection.winner is None:
        _finish_log(log_path, "freeze validation-only")
        return write_authenticated_json_marker(
            root / "VALIDATION_ONLY_COMPLETE", config=state.config, source_manifest=state.source_manifest,
            producing_command="runner.py freeze", input_artifacts={"VALIDATION_COMPLETE.json": root / "VALIDATION_COMPLETE.json", "validation_summary": summary_path, "log": log_path},
            payload={
                "validation_gates": gate_payload,
                "candidate_selection": _candidate_selection_payload(
                    candidates,
                    {
                        f"predictions/validation_{anchor}_seed{seed}.csv": root
                        / "predictions"
                        / f"validation_{anchor}_seed{seed}.csv"
                        for anchor in state.config.anchor_families
                        for seed in state.config.seeds
                    },
                    state.config,
                ),
                "set03_accessed": False,
            },
        )
    winner = selection.winner
    checkpoints = {seed: root / "checkpoints" / f"{winner}_seed{seed}.pt" for seed in state.config.seeds}
    anchors = {seed: root / "models" / f"{winner}_seed{seed}.joblib" for seed in state.config.seeds}
    scaler = root / "metrics" / "context_scaler.json"
    validation_artifacts = {
        "VALIDATION_COMPLETE.json": root / "VALIDATION_COMPLETE.json",
        "metrics/validation_summary_results.csv": summary_path,
        "metrics/validation_per_seed_results.csv": root / "metrics" / "validation_per_seed_results.csv",
        "metrics/paired_deltas.csv": root / "metrics" / "paired_deltas.csv",
        "metrics/error_transitions_fixed_0p5.csv": root / "metrics" / "error_transitions_fixed_0p5.csv",
        "metrics/context_ablation_diagnostics.csv": root / "metrics" / "context_ablation_diagnostics.csv",
        "metrics/context_scaler.json": scaler,
        **{f"models/anchor_seed{seed}": path for seed, path in anchors.items()},
        **{
            f"predictions/validation_{anchor}_seed{seed}.csv": (
                root / "predictions" / f"validation_{anchor}_seed{seed}.csv"
            )
            for anchor in state.config.anchor_families
            for seed in state.config.seeds
        },
    }
    _finish_log(log_path, "freeze final-candidate")
    validation_artifacts["logs/freeze.log"] = log_path
    candidate_manifest = root / "FROZEN_CANDIDATE_MANIFEST.json"
    write_authenticated_candidate_manifest(
        candidate_manifest, config=state.config, source_manifest=state.source_manifest,
        winning_candidate=winner, validation_summary=candidates[winner][0], anchor_summary=candidates[winner][1],
        checkpoint_artifacts=checkpoints, validation_artifacts=validation_artifacts,
    )
    final_args = ["final", "--threshold", "0.5"]
    return write_authenticated_final_freeze(
        root / "FROZEN_FINAL_MODEL.json", config=state.config, source_manifest=state.source_manifest,
        producing_command="runner.py freeze", winning_candidate=winner,
        validation_summary=candidates[winner][0], anchor_summary=candidates[winner][1],
        checkpoint_artifacts=checkpoints, validation_artifacts=validation_artifacts,
        candidate_manifest=candidate_manifest,
        payload={
            "anchor_sha256": {str(seed): sha256_file(path) for seed, path in anchors.items()},
            "anchor_paths": {str(seed): str(path.resolve()) for seed, path in anchors.items()},
            "scaler_sha256": sha256_file(scaler), "scaler_path": str(scaler.resolve()),
            "validation_summary_file_sha256": sha256_file(summary_path),
            "final_command_arguments": final_args,
            "validation_gates": gate_payload,
            "candidate_selection": _candidate_selection_payload(
                candidates, validation_artifacts, state.config
            ),
        },
    )


def build_efficiency_rows(
    *,
    anchor_name: str,
    seed: int,
    sample_count: int,
    anchor_seconds_cpu: float,
    correction_seconds_cpu: float,
    correction_seconds_cuda: float | None,
    tree_bytes: int,
    checkpoint_bytes: int,
    tree_count: int,
    node_count: int,
    leaf_count: int,
) -> list[dict[str, Any]]:
    """Build distinct CPU/CUDA cached-feature and serialized-complexity rows."""
    if sample_count <= 0:
        raise ValueError("efficiency measurement requires at least one sample")
    rows = [
        {
            "measurement": "tree_anchor_cached_feature_latency_cpu",
            "value": anchor_seconds_cpu / sample_count,
            "unit": "seconds_per_sample",
            "status": "measured",
        },
        {
            "measurement": "neural_correction_cached_feature_latency_cpu",
            "value": correction_seconds_cpu / sample_count,
            "unit": "seconds_per_sample",
            "status": "measured",
        },
        {
            "measurement": "combined_cached_feature_latency_cpu",
            "value": (anchor_seconds_cpu + correction_seconds_cpu) / sample_count,
            "unit": "seconds_per_sample",
            "status": "measured",
        },
        {
            "measurement": "tree_bytes",
            "value": tree_bytes,
            "unit": "bytes",
            "status": "measured",
        },
        {
            "measurement": "correction_checkpoint_bytes",
            "value": checkpoint_bytes,
            "unit": "bytes",
            "status": "measured",
        },
        {
            "measurement": "combined_classifier_bytes",
            "value": tree_bytes + checkpoint_bytes,
            "unit": "bytes",
            "status": "measured",
        },
        {
            "measurement": "correction_trainable_parameters",
            "value": ExperimentConfig().context_parameter_count,
            "unit": "count",
            "status": "measured",
        },
        {"measurement": "tree_count", "value": tree_count, "unit": "count", "status": "measured"},
        {"measurement": "tree_node_count", "value": node_count, "unit": "count", "status": "measured"},
        {"measurement": "tree_leaf_count", "value": leaf_count, "unit": "count", "status": "measured"},
    ]
    for measurement, combined in (
        ("neural_correction_cached_feature_latency_cuda", False),
        ("combined_cached_feature_latency_cuda", True),
    ):
        available = correction_seconds_cuda is not None
        value = (
            ((anchor_seconds_cpu if combined else 0.0) + correction_seconds_cuda)
            / sample_count
            if available
            else np.nan
        )
        rows.append(
            {
                "measurement": measurement,
                "value": value,
                "unit": "seconds_per_sample",
                "status": "measured" if available else "unavailable",
            }
        )
    return [
        {"anchor_name": anchor_name, "seed": seed, **row}
        for row in rows
    ]


def final(
    root: Path = EXPERIMENT_ROOT,
    *,
    device: str | None = None,
    threshold: float = ExperimentConfig().threshold,
) -> dict[str, Any]:
    """Authorize Set03 before importing its builder; evaluate the winner once."""
    state = PhaseState(root)
    if threshold != state.config.threshold:
        raise ValueError("final evaluation threshold is frozen at 0.5")
    authorization = state.require_final_authorization()
    state.authorize_source(TEST_SOURCE, phase="final")
    if (root / PHASE_MARKERS["final"]).exists():
        raise PhaseOrderError("final evaluation may run only once")
    log_path = root / "logs" / "final.log"
    _require_absent(
        (
            log_path,
            root / "metrics" / "final_per_seed_results.csv",
            root / "metrics" / "final_summary_results.csv",
            root / "metrics" / "final_error_transitions_fixed_0p5.csv",
            root / "metrics" / "efficiency.csv",
            *(
                root
                / "predictions"
                / f"final_{authorization['final_freeze']['winning_candidate']}_seed{seed}.csv"
                for seed in state.config.seeds
            ),
        )
    )
    _start_log(log_path, "final")

    # This import is deliberately unreachable until the hash-valid freeze above.
    from .data import _reconstruct_from_exp012

    test_root = root.parent / "055_compact_residual_biu_lstm" / "cache" / "test"
    metadata_path = test_root / "metadata.json"
    expected_metadata_sha256 = "0527df2110dc61e4332ed144527e60b06c8688048116e1cfb1efc64ef0b24eb1"
    if not metadata_path.is_file() or sha256_file(metadata_path) != expected_metadata_sha256:
        raise ArtifactAuthenticationError("Set03 cache metadata authentication failed")
    metadata = _read_json(metadata_path)
    source_hashes = metadata.get("artifacts", {})
    for filename in ("rows.csv", "ego.npy", "yolop.npy"):
        source = test_root / filename
        if not source.is_file() or sha256_file(source) != source_hashes.get(filename):
            raise ArtifactAuthenticationError(f"Set03 source authentication failed: {filename}")
    rows = pd.read_csv(test_root / "rows.csv")
    required = {"sequence_id", "split", "set_id", "label", "cache_index"}
    if (
        not required.issubset(rows)
        or len(rows) != state.config.test_rows
        or not rows["split"].astype(str).eq("test").all()
        or not rows["set_id"].astype(str).eq("set03").all()
        or rows["sequence_id"].astype(str).duplicated().any()
        or not np.array_equal(rows["cache_index"].to_numpy(), np.arange(len(rows)))
    ):
        raise ArtifactAuthenticationError("Set03 row cohort authentication failed")
    labels = rows["label"].to_numpy(dtype=np.int64)
    if set(labels) != {0, 1}:
        raise ArtifactAuthenticationError("Set03 labels are not binary")
    ego = np.load(test_root / "ego.npy", allow_pickle=False)
    yolop = np.load(test_root / "yolop.npy", allow_pickle=False)
    if ego.shape != (state.config.test_rows, *state.config.ego_shape) or yolop.shape != (
        state.config.test_rows,
        *state.config.yolop_shape,
    ):
        raise ArtifactAuthenticationError("Set03 context shapes are invalid")
    reconstruction_start = time.perf_counter()
    summary_v1, summary_v2 = _reconstruct_from_exp012(rows, state.config)
    reconstruction_seconds = time.perf_counter() - reconstruction_start
    scaler_path = Path(authorization["payload"]["scaler_path"])
    if sha256_file(scaler_path) != authorization["payload"]["scaler_sha256"]:
        raise ArtifactAuthenticationError("frozen scaler was substituted")
    scaler = _load_scaler(scaler_path)
    ego = ((ego - scaler.ego_mean) / scaler.ego_std).astype(np.float32)
    yolop = ((yolop - scaler.yolop_mean) / scaler.yolop_std).astype(np.float32)
    if not np.isfinite(ego).all() or not np.isfinite(yolop).all():
        raise ArtifactAuthenticationError("Set03 standardization produced non-finite context")
    winner = authorization["final_freeze"]["winning_candidate"]
    matrix = summary_v1 if winner == "rf_v1_bbox" else summary_v2
    metadata_abi = anchor_feature_metadata(winner)
    if matrix.shape != (state.config.test_rows, metadata_abi["feature_dimension"]):
        raise ArtifactAuthenticationError("Set03 anchor matrix violates the frozen feature ABI")

    per_seed_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    artifacts: dict[str, Path] = {
        "FROZEN_FINAL_MODEL.json": root / "FROZEN_FINAL_MODEL.json",
        "set03_metadata.json": metadata_path,
        "set03_rows.csv": test_root / "rows.csv",
        "set03_ego.npy": test_root / "ego.npy",
        "set03_yolop.npy": test_root / "yolop.npy",
        "logs/final.log": log_path,
    }
    latency_rows: list[dict[str, Any]] = []
    for seed in state.config.seeds:
        anchor_path = Path(authorization["payload"]["anchor_paths"][str(seed)])
        checkpoint_path = Path(
            authorization["final_freeze"]["winning_checkpoint_paths"][str(seed)]
        )
        if sha256_file(anchor_path) != authorization["payload"]["anchor_sha256"][str(seed)]:
            raise ArtifactAuthenticationError(f"frozen anchor was substituted for seed {seed}")
        anchor_model = joblib.load(anchor_path)
        anchor_start = time.perf_counter()
        anchor_probability = anchor_model.predict_proba(matrix)[:, 1]
        anchor_seconds = time.perf_counter() - anchor_start
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("seed") != seed or checkpoint.get("total_parameters") != state.config.context_parameter_count:
            raise ArtifactAuthenticationError("frozen checkpoint identity mismatch")
        corrector = HybridContextCorrector()
        corrector.load_state_dict(checkpoint["model_state"])
        prediction = predict_context_corrector(
            corrector, ego, yolop, anchor_probability, device=device
        )
        cpu_start = time.perf_counter()
        predict_context_corrector(
            corrector, ego, yolop, anchor_probability, device="cpu"
        )
        correction_seconds_cpu = time.perf_counter() - cpu_start
        correction_seconds_cuda: float | None = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            cuda_start = time.perf_counter()
            predict_context_corrector(
                corrector, ego, yolop, anchor_probability, device="cuda"
            )
            torch.cuda.synchronize()
            correction_seconds_cuda = time.perf_counter() - cuda_start
        corrected_probability = prediction["probability"]
        anchor_metric = compute_binary_metrics(labels, anchor_probability)
        corrected_metric = compute_binary_metrics(labels, corrected_probability)
        for variant, metric in (("anchor", anchor_metric), ("corrected", corrected_metric)):
            per_seed_rows.append(
                {
                    "anchor_name": winner,
                    "seed": seed,
                    "variant": variant,
                    "threshold": state.config.threshold,
                    **_metric_record(metric),
                }
            )
        transition_rows.append(
            {
                "anchor_name": winner,
                "seed": seed,
                **_transition_record(
                    labels,
                    anchor_probability,
                    corrected_probability,
                    threshold=state.config.threshold,
                ),
            }
        )
        prediction_path = root / "predictions" / f"final_{winner}_seed{seed}.csv"
        _atomic_csv(
            pd.DataFrame(
                {
                    "sequence_id": rows["sequence_id"].astype(str),
                    "set_id": rows["set_id"].astype(str),
                    "label": labels,
                    "anchor_probability": anchor_probability,
                    "corrected_probability": corrected_probability,
                    "threshold": state.config.threshold,
                }
            ),
            prediction_path,
        )
        artifacts[f"predictions/{prediction_path.name}"] = prediction_path
        complexity = anchor_complexity(anchor_model)
        latency_rows.extend(
            build_efficiency_rows(
                anchor_name=winner,
                seed=seed,
                sample_count=len(rows),
                anchor_seconds_cpu=anchor_seconds,
                correction_seconds_cpu=correction_seconds_cpu,
                correction_seconds_cuda=correction_seconds_cuda,
                tree_bytes=anchor_path.stat().st_size,
                checkpoint_bytes=checkpoint_path.stat().st_size,
                tree_count=complexity.tree_count,
                node_count=complexity.node_count,
                leaf_count=complexity.leaf_count,
            )
        )
    latency_rows.extend(
        [
            {
                "anchor_name": winner,
                "seed": "all",
                "measurement": "BIU_anchor_reconstruction_from_cached_keypoints",
                "value": reconstruction_seconds / len(rows),
                "unit": "seconds_per_sample",
                "status": "measured",
            },
            {
                "anchor_name": winner,
                "seed": "all",
                "measurement": "YOLOP_20D_composition_from_cached_perception",
                "value": np.nan,
                "unit": "seconds_per_sample",
                "status": "needs verification: source cache stores precomposed 20D arrays",
            },
            {
                "anchor_name": winner,
                "seed": "all",
                "measurement": "full_YOLOP_perception_runtime",
                "value": np.nan,
                "unit": "seconds_per_sample",
                "status": "needs verification: no traced runtime artifact was available",
            },
        ]
    )
    per_seed_path = root / "metrics" / "final_per_seed_results.csv"
    summary_path = root / "metrics" / "final_summary_results.csv"
    transitions_path = root / "metrics" / "final_error_transitions_fixed_0p5.csv"
    efficiency_path = root / "metrics" / "efficiency.csv"
    per_seed_frame = pd.DataFrame(per_seed_rows)
    _atomic_csv(per_seed_frame, per_seed_path)
    _atomic_csv(
        _aggregate_metric_rows(
            per_seed_frame,
            include_validation_evidence=False,
            include_counts=True,
            include_std=True,
        ),
        summary_path,
    )
    _atomic_csv(pd.DataFrame(transition_rows), transitions_path)
    _atomic_csv(pd.DataFrame(latency_rows), efficiency_path)
    artifacts.update(
        {
            "metrics/final_per_seed_results.csv": per_seed_path,
            "metrics/final_summary_results.csv": summary_path,
            "metrics/final_error_transitions_fixed_0p5.csv": transitions_path,
            "metrics/efficiency.csv": efficiency_path,
        }
    )
    payload = {
        "winning_candidate": winner,
        "expected_rows": state.config.test_rows,
        "seeds": list(state.config.seeds),
        "threshold": state.config.threshold,
        "set03_constructed_once": True,
    }
    _finish_log(log_path, "final")
    return state.write_phase_marker("final", command="runner.py final", artifacts=artifacts, payload=payload)


def _require_finite_metrics(frame: pd.DataFrame, *, name: str) -> None:
    if not set(_METRIC_COLUMNS).issubset(frame) or not np.isfinite(
        frame.loc[:, _METRIC_COLUMNS].to_numpy(dtype=np.float64)
    ).all():
        raise VerificationError(f"{name} has missing or non-finite metrics")


def _require_matching_metric_row(
    frame: pd.DataFrame,
    *,
    anchor_name: str,
    seed: int,
    variant: str,
    expected: MetricBundle,
) -> None:
    row = frame.loc[
        (frame["anchor_name"] == anchor_name)
        & (frame["seed"] == seed)
        & (frame["variant"] == variant)
    ]
    if len(row) != 1:
        raise VerificationError("metric evidence does not identify exactly one seed row")
    actual = row.iloc[0]
    for column in _PER_SEED_METRIC_COLUMNS:
        actual_value = float(actual[column])
        expected_value = float(getattr(expected, column))
        matches = (
            actual_value == expected_value
            if column in (*_COUNT_COLUMNS, "high_confidence_errors")
            else np.isclose(
                actual_value,
                expected_value,
                rtol=1e-10,
                atol=1e-12,
            )
        )
        if not matches:
            raise VerificationError(
                f"persisted {variant} metric does not match predictions: {column}"
            )


def _reconstruct_validation_candidates(
    root: Path, config: ExperimentConfig
) -> tuple[
    dict[str, tuple[MetricBundle, MetricBundle]],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Recompute both validation candidates from saved per-seed predictions."""
    per_seed = pd.read_csv(root / "metrics" / "validation_per_seed_results.csv")
    expected_rows = {
        (family, seed, variant)
        for family in config.anchor_families
        for seed in config.seeds
        for variant in ("anchor", "corrected")
    }
    actual_rows = set(
        per_seed.loc[:, ["anchor_name", "seed", "variant"]].itertuples(
            index=False, name=None
        )
    )
    if actual_rows != expected_rows or len(per_seed) != len(expected_rows):
        raise VerificationError(
            "validation metrics do not contain the exact anchor/seed grid"
        )
    if not set(_PER_SEED_METRIC_COLUMNS).issubset(per_seed) or not np.isfinite(
        per_seed.loc[:, _PER_SEED_METRIC_COLUMNS].to_numpy(dtype=np.float64)
    ).all():
        raise VerificationError("validation per-seed evidence is incomplete or non-finite")
    if "invariants_pass" not in per_seed or not per_seed["invariants_pass"].astype(
        bool
    ).all():
        raise VerificationError("validation per-seed invariants are not all authenticated")

    expected_ids: np.ndarray | None = None
    expected_labels: np.ndarray | None = None
    for family in config.anchor_families:
        for seed in config.seeds:
            prediction = pd.read_csv(
                root / "predictions" / f"validation_{family}_seed{seed}.csv",
                float_precision="round_trip",
            )
            required = {
                "sequence_id",
                "label",
                "anchor_probability",
                "corrected_probability",
                "corrected_class_fixed_0p5",
            }
            if (
                not required.issubset(prediction)
                or len(prediction) != config.val_rows
                or prediction["sequence_id"].astype(str).duplicated().any()
            ):
                raise VerificationError(
                    f"validation prediction cohort is malformed: {family} seed {seed}"
                )
            ids = prediction["sequence_id"].astype(str).to_numpy()
            labels = prediction["label"].to_numpy(dtype=np.int64)
            if expected_ids is None:
                expected_ids = ids
                expected_labels = labels
            elif not np.array_equal(ids, expected_ids) or not np.array_equal(
                labels, expected_labels
            ):
                raise VerificationError(
                    "validation prediction families do not share exact IDs/order/labels"
                )
            anchor_probabilities = prediction["anchor_probability"].to_numpy(
                dtype=np.float64
            )
            corrected_probabilities = prediction["corrected_probability"].to_numpy(
                dtype=np.float64
            )
            if not np.isfinite(
                np.column_stack((anchor_probabilities, corrected_probabilities))
            ).all() or not np.array_equal(
                prediction["corrected_class_fixed_0p5"].to_numpy(dtype=np.int64),
                (corrected_probabilities >= config.threshold).astype(np.int64),
            ):
                raise VerificationError(
                    f"validation probabilities or fixed threshold drifted: {family} seed {seed}"
                )
            for variant, probabilities in (
                ("anchor", anchor_probabilities),
                ("corrected", corrected_probabilities),
            ):
                _require_matching_metric_row(
                    per_seed,
                    anchor_name=family,
                    seed=seed,
                    variant=variant,
                    expected=compute_binary_metrics(labels, probabilities),
                )

    reconstructed = _aggregate_metric_rows(
        per_seed, include_validation_evidence=True
    ).sort_values(["anchor_name", "variant"]).reset_index(drop=True)
    persisted = pd.read_csv(
        root / "metrics" / "validation_summary_results.csv"
    ).sort_values(["anchor_name", "variant"]).reset_index(drop=True)
    if (
        len(persisted) != len(reconstructed)
        or not np.array_equal(
            persisted[["anchor_name", "variant"]].astype(str).to_numpy(),
            reconstructed[["anchor_name", "variant"]].astype(str).to_numpy(),
        )
    ):
        raise VerificationError("validation summary identity grid is malformed")
    for column in (*_METRIC_COLUMNS, "seed_count"):
        if not np.allclose(
            persisted[column].to_numpy(dtype=np.float64),
            reconstructed[column].to_numpy(dtype=np.float64),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise VerificationError(
                f"validation summary does not match prediction-derived evidence: {column}"
            )
    persisted_harmonic = [
        np.asarray(json.loads(value), dtype=np.float64)
        for value in persisted["per_seed_harmonic"].astype(str)
    ]
    reconstructed_harmonic = [
        np.asarray(json.loads(value), dtype=np.float64)
        for value in reconstructed["per_seed_harmonic"].astype(str)
    ]
    if any(
        not np.allclose(left, right, rtol=1e-10, atol=1e-12)
        for left, right in zip(persisted_harmonic, reconstructed_harmonic)
    ) or not persisted["invariants_pass"].astype(bool).all():
        raise VerificationError("validation summary provenance evidence drifted")
    candidates = {
        family: (
            _summary_bundle(reconstructed, family, "corrected"),
            _summary_bundle(reconstructed, family, "anchor"),
        )
        for family in config.anchor_families
    }
    return candidates, per_seed, reconstructed


def _verify_validation_bundle(root: Path, config: ExperimentConfig) -> dict[str, Path]:
    per_seed_path = root / "metrics" / "validation_per_seed_results.csv"
    summary_path = root / "metrics" / "validation_summary_results.csv"
    _reconstruct_validation_candidates(root, config)
    evidence = {
        "metrics/validation_per_seed_results.csv": per_seed_path,
        "metrics/validation_summary_results.csv": summary_path,
    }
    for anchor in config.anchor_families:
        for seed in config.seeds:
            bundle = _bundle_spec(root, anchor, seed)
            authenticate_probability_bundle(
                bundle,
                grouped_oof=_load_grouped_oof(bundle),
                expected_anchor=anchor,
                expected_seed=seed,
            )
            expected = pd.read_csv(bundle.validation_path, float_precision="round_trip")
            prediction_path = root / "predictions" / f"validation_{anchor}_seed{seed}.csv"
            prediction = pd.read_csv(prediction_path, float_precision="round_trip")
            if (
                len(prediction) != config.val_rows
                or prediction["sequence_id"].astype(str).duplicated().any()
                or not np.array_equal(
                    prediction["sequence_id"].astype(str).to_numpy(),
                    expected["sequence_id"].astype(str).to_numpy(),
                )
                or not np.array_equal(
                    prediction["label"].to_numpy(dtype=np.int64),
                    expected["label"].to_numpy(dtype=np.int64),
                )
                or not np.array_equal(
                    prediction["anchor_probability"].to_numpy(dtype=np.float64),
                    expected["probability"].to_numpy(dtype=np.float64),
                )
                or not np.isfinite(
                    prediction[["anchor_probability", "corrected_probability"]].to_numpy(
                        dtype=np.float64
                    )
                ).all()
                or not np.array_equal(
                    prediction["corrected_class_fixed_0p5"].to_numpy(dtype=np.int64),
                    (
                        prediction["corrected_probability"].to_numpy(dtype=np.float64)
                        >= config.threshold
                    ).astype(np.int64),
                )
            ):
                raise VerificationError(
                    f"validation prediction cohort or fixed threshold failed: {anchor} seed {seed}"
                )
            evidence[f"predictions/{prediction_path.name}"] = prediction_path
    return evidence


def _verify_final_bundle(
    root: Path,
    config: ExperimentConfig,
    winner: str,
    terminal_marker: Mapping[str, Any],
) -> dict[str, Path]:
    per_seed_path = root / "metrics" / "final_per_seed_results.csv"
    summary_path = root / "metrics" / "final_summary_results.csv"
    per_seed = pd.read_csv(per_seed_path)
    expected_rows = {
        (winner, seed, variant)
        for seed in config.seeds
        for variant in ("anchor", "corrected")
    }
    if (
        set(per_seed.loc[:, ["anchor_name", "seed", "variant"]].itertuples(index=False, name=None))
        != expected_rows
        or len(per_seed) != len(expected_rows)
        or not per_seed["threshold"].eq(config.threshold).all()
    ):
        raise VerificationError("final metrics violate the frozen anchor/seed/threshold grid")
    _require_finite_metrics(per_seed, name="final per-seed results")
    if not set(_PER_SEED_METRIC_COLUMNS).issubset(per_seed) or not np.isfinite(
        per_seed.loc[:, _PER_SEED_METRIC_COLUMNS].to_numpy(dtype=np.float64)
    ).all():
        raise VerificationError("final per-seed metric/count evidence is incomplete")
    summary = pd.read_csv(summary_path)
    expected_summary = {(winner, "anchor"), (winner, "corrected")}
    actual_summary = set(
        summary[["anchor_name", "variant"]].itertuples(index=False, name=None)
    )
    if (
        len(summary) != 2
        or actual_summary != expected_summary
        or not summary["seed_count"].eq(len(config.seeds)).all()
    ):
        raise VerificationError(
            "final summary must contain unique anchor and corrected winner rows"
        )
    _require_finite_metrics(summary, name="final summary")
    summary_columns = (
        *_PER_SEED_METRIC_COLUMNS,
        *(f"{column}_std" for column in _PER_SEED_METRIC_COLUMNS),
    )
    if not set(summary_columns).issubset(summary) or not np.isfinite(
        summary.loc[:, summary_columns].to_numpy(dtype=np.float64)
    ).all():
        raise VerificationError("final summary mean/std evidence is incomplete")
    evidence = {
        "metrics/final_per_seed_results.csv": per_seed_path,
        "metrics/final_summary_results.csv": summary_path,
    }
    set03_path_text = terminal_marker.get("input_artifact_paths", {}).get("set03_rows.csv")
    set03_hash = terminal_marker.get("input_artifacts", {}).get("set03_rows.csv")
    if not isinstance(set03_path_text, str):
        raise VerificationError("final marker does not bind authenticated Set03 rows")
    set03_path = Path(set03_path_text)
    if not set03_path.is_file() or sha256_file(set03_path) != set03_hash:
        raise VerificationError("authenticated Set03 row artifact hash mismatch")
    authenticated_rows = pd.read_csv(set03_path)
    expected_ids = authenticated_rows["sequence_id"].astype(str).to_numpy()
    expected_labels = authenticated_rows["label"].to_numpy(dtype=np.int64)
    if len(authenticated_rows) != config.test_rows or len(set(expected_ids)) != config.test_rows:
        raise VerificationError("authenticated Set03 cohort is malformed")
    for seed in config.seeds:
        prediction_path = root / "predictions" / f"final_{winner}_seed{seed}.csv"
        prediction = pd.read_csv(prediction_path, float_precision="round_trip")
        ids = prediction["sequence_id"].astype(str).to_numpy()
        labels = prediction["label"].to_numpy(dtype=np.int64)
        if (
            len(prediction) != config.test_rows
            or len(set(ids)) != config.test_rows
            or not prediction["set_id"].astype(str).eq("set03").all()
            or not prediction["threshold"].eq(config.threshold).all()
            or not np.isfinite(
                prediction[["anchor_probability", "corrected_probability"]].to_numpy(
                    dtype=np.float64
                )
            ).all()
        ):
            raise VerificationError(f"final prediction bundle is malformed for seed {seed}")
        if not np.array_equal(ids, expected_ids) or not np.array_equal(
            labels, expected_labels
        ):
            raise VerificationError(
                "final prediction IDs/order/labels differ from authenticated Set03 rows"
            )
        for variant, probabilities in (
            (
                "anchor",
                prediction["anchor_probability"].to_numpy(dtype=np.float64),
            ),
            (
                "corrected",
                prediction["corrected_probability"].to_numpy(dtype=np.float64),
            ),
        ):
            _require_matching_metric_row(
                per_seed,
                anchor_name=winner,
                seed=seed,
                variant=variant,
                expected=compute_binary_metrics(labels, probabilities),
            )
        evidence[f"predictions/{prediction_path.name}"] = prediction_path
    expected_summary_frame = _aggregate_metric_rows(
        per_seed,
        include_validation_evidence=False,
        include_counts=True,
        include_std=True,
    ).sort_values(["anchor_name", "variant"]).reset_index(drop=True)
    actual_summary_frame = summary.sort_values(
        ["anchor_name", "variant"]
    ).reset_index(drop=True)
    for column in (*summary_columns, "seed_count"):
        if not np.allclose(
            actual_summary_frame[column].to_numpy(dtype=np.float64),
            expected_summary_frame[column].to_numpy(dtype=np.float64),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise VerificationError(
                f"final summary does not match per-seed mean/std: {column}"
            )
    return evidence


def verify(root: Path = EXPERIMENT_ROOT) -> dict[str, Any]:
    state = PhaseState(root)
    try:
        terminal = state.verify_terminal_state()
        frozen = (root / "FROZEN_FINAL_MODEL.json").exists()
        expected_rows = state.config.test_rows if frozen else state.config.val_rows
        payload = terminal.get("payload", {})
        if "expected_rows" in payload and int(payload["expected_rows"]) != expected_rows:
            raise VerificationError("terminal row count does not match the frozen cohort")
        if frozen:
            authorization = state.require_final_authorization()
            evidence = _verify_validation_bundle(root, state.config)
            evidence.update(_verify_final_bundle(
                root,
                state.config,
                authorization["final_freeze"]["winning_candidate"],
                terminal,
            ))
        else:
            evidence = _verify_validation_bundle(root, state.config)
        return state.record_verification(passed=True, evidence=evidence)
    except Exception:
        complete = root / "EXPERIMENT_COMPLETE"
        if complete.exists():
            try:
                state._verify_marker("EXPERIMENT_COMPLETE")
            except ArtifactAuthenticationError:
                complete.unlink(missing_ok=True)
        raise


def status(root: Path = EXPERIMENT_ROOT) -> dict[str, bool]:
    names = (STAGING_MARKER, *PHASE_MARKERS.values(), "FROZEN_FINAL_MODEL.json", "VALIDATION_ONLY_COMPLETE", "EXPERIMENT_COMPLETE")
    return {name: (root / name).is_file() for name in names}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=EXPERIMENT_ROOT)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    for phase in ("stage-sources", "prepare", "fit-anchors", "freeze", "verify", "status"):
        subparsers.add_parser(phase)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--anchor", required=True, choices=ExperimentConfig().anchor_families)
    train_parser.add_argument("--seed", required=True, type=int, choices=ExperimentConfig().seeds)
    train_parser.add_argument("--device")
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--device")
    final_parser = subparsers.add_parser("final")
    final_parser.add_argument("--device")
    final_parser.add_argument("--threshold", type=float, default=ExperimentConfig().threshold)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    _runtime_checker: Callable[[], None] | None = None,
) -> int:
    checker = assert_runtime_contract if _runtime_checker is None else _runtime_checker
    checker()
    args = build_parser().parse_args(argv)
    functions: dict[str, Callable[..., Any]] = {
        "stage-sources": stage_source_inputs, "prepare": prepare, "fit-anchors": fit_anchors, "validate": validate,
        "freeze": freeze, "final": final, "verify": verify, "status": status,
    }
    if args.phase == "train":
        result = train(args.root, anchor_name=args.anchor, seed=args.seed, device=args.device)
    elif args.phase == "validate":
        result = functions[args.phase](args.root, device=args.device)
    elif args.phase == "final":
        result = final(args.root, device=args.device, threshold=args.threshold)
    else:
        result = functions[args.phase](args.root)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "TEST_SOURCE", "TRAIN_VALIDATION_SOURCE", "ArtifactAuthenticationError",
    "ForbiddenSourceError", "PhaseOrderError", "VerificationError", "PhaseState",
    "ProbabilityBundleSpec", "AuthenticatedProbabilityVectors", "CandidateSelection",
    "authenticate_probability_bundle", "persist_probability_bundle",
    "run_authenticated_training", "select_validation_candidate", "write_validation_outputs",
    "stage_source_inputs", "prepare", "fit_anchors", "train", "validate", "freeze", "final", "verify", "main",
]
