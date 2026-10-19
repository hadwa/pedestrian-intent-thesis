"""Exact, parity-checked Exp062 tree anchors and track-grouped OOF predictions."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .contracts import (
    AnchorComplexity,
    AnchorParityResult,
    ExperimentConfig,
    compute_binary_metrics,
    sha256_file,
)


ANCHOR_FEATURE_DIMENSIONS = {
    "rf_v1_bbox": ExperimentConfig().rf_feature_dim,
    "xgb_v2_bbox": ExperimentConfig().xgb_feature_dim,
}

ANCHOR_FEATURE_SETS = {
    "rf_v1_bbox": ExperimentConfig().rf_v1_bbox_feature_set,
    "xgb_v2_bbox": ExperimentConfig().xgb_v2_bbox_feature_set,
}


@dataclass(frozen=True)
class AnchorFitResult:
    """A persisted full-train anchor and immutable post-write metadata."""

    model: Pipeline
    path: Path
    sha256: str
    size_bytes: int
    complexity: AnchorComplexity


@dataclass(frozen=True)
class FoldModelArtifact:
    """Persisted metadata for one exact OOF fold model."""

    fold: int
    path: Path
    sha256: str
    size_bytes: int
    complexity: AnchorComplexity
    train_rows: int
    holdout_rows: int
    train_positive_rows: int
    holdout_positive_rows: int
    train_negative_rows: int
    holdout_negative_rows: int
    train_unique_tracks: int
    holdout_unique_tracks: int
    shared_track_count: int


@dataclass(frozen=True)
class GroupedOOFResult:
    """Original-order OOF probabilities and the fold-level leakage audit."""

    probabilities: np.ndarray
    assignment_count: np.ndarray
    fold_assignment: np.ndarray
    fold_audit: pd.DataFrame
    fold_artifacts: tuple[FoldModelArtifact, ...]


def _validate_anchor_inputs(
    features: np.ndarray, labels: np.ndarray, anchor_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Validate aligned binary training arrays before an anchor fit."""
    feature_array = np.asarray(features, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    if feature_array.ndim != 2 or feature_array.shape[0] == 0:
        raise ValueError("anchor features must be a nonempty 2D matrix")
    expected_dimension = ANCHOR_FEATURE_DIMENSIONS.get(anchor_name)
    if expected_dimension is None:
        raise ValueError(f"unknown anchor: {anchor_name}")
    if feature_array.shape[1] != expected_dimension:
        raise ValueError(
            f"{anchor_name} feature dimension must be exactly {expected_dimension}, "
            f"got {feature_array.shape[1]}"
        )
    if feature_array.shape[0] != label_array.size:
        raise ValueError("anchor features and labels must have matching row counts")
    if not np.isin(label_array, (0, 1)).all() or np.unique(label_array).size != 2:
        raise ValueError("anchor labels must contain both binary classes")
    if np.isinf(feature_array).any():
        raise ValueError("anchor features must not contain infinite values")
    return feature_array, label_array


def _validate_anchor_identity(anchor_name: str, seed: int) -> None:
    config = ExperimentConfig()
    if anchor_name not in config.anchor_families:
        raise ValueError(f"unknown anchor: {anchor_name}")
    if seed not in config.seeds:
        raise ValueError("anchor seed is not one of the frozen Exp062 seeds")


def anchor_feature_metadata(anchor_name: str) -> dict[str, str | int]:
    """Return the frozen anchor identity and matrix ABI used by the runner."""
    if anchor_name not in ANCHOR_FEATURE_DIMENSIONS:
        raise ValueError(f"unknown anchor: {anchor_name}")
    return {
        "anchor_name": anchor_name,
        "feature_dimension": ANCHOR_FEATURE_DIMENSIONS[anchor_name],
        "feature_set": ANCHOR_FEATURE_SETS[anchor_name],
    }


def build_anchor(anchor_name: str, seed: int, labels: np.ndarray) -> Pipeline:
    """Construct one exact historical anchor pipeline without fitting it."""
    _validate_anchor_identity(anchor_name, seed)
    label_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    if label_array.size == 0 or not np.isin(label_array, (0, 1)).all():
        raise ValueError("anchor labels must be nonempty binary values")
    if anchor_name == "rf_v1_bbox":
        classifier: Any = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    else:
        negatives = int((label_array == 0).sum())
        positives = int((label_array == 1).sum())
        if negatives == 0 or positives == 0:
            raise ValueError("XGBoost anchor requires both binary classes")
        classifier = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=4,
            scale_pos_weight=negatives / positives,
        )
    return Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("classifier", classifier)]
    )


def _count_xgboost_tree(node: dict[str, Any]) -> tuple[int, int]:
    children = node.get("children", [])
    if not children:
        return 1, 1
    nodes, leaves = 1, 0
    for child in children:
        child_nodes, child_leaves = _count_xgboost_tree(child)
        nodes += child_nodes
        leaves += child_leaves
    return nodes, leaves


def anchor_complexity(model: Pipeline) -> AnchorComplexity:
    """Count trees, nodes, and leaves using the fitted estimator's native APIs."""
    if not isinstance(model, Pipeline) or "classifier" not in model.named_steps:
        raise ValueError("anchor complexity requires a fitted anchor pipeline")
    classifier = model.named_steps["classifier"]
    if isinstance(classifier, RandomForestClassifier):
        if not hasattr(classifier, "estimators_"):
            raise ValueError("random forest anchor has not been fitted")
        tree_count = len(classifier.estimators_)
        node_count = sum(tree.tree_.node_count for tree in classifier.estimators_)
        leaf_count = sum(
            int((tree.tree_.children_left == -1).sum()) for tree in classifier.estimators_
        )
        return AnchorComplexity(tree_count, int(node_count), int(leaf_count))
    if isinstance(classifier, XGBClassifier):
        try:
            trees = [json.loads(tree) for tree in classifier.get_booster().get_dump(dump_format="json")]
        except Exception as exc:  # pragma: no cover - xgboost exception varies by pinned build
            raise ValueError("XGBoost anchor has not been fitted") from exc
        counts = [_count_xgboost_tree(tree) for tree in trees]
        return AnchorComplexity(
            tree_count=len(counts),
            node_count=sum(item[0] for item in counts),
            leaf_count=sum(item[1] for item in counts),
        )
    raise ValueError("anchor complexity requires an approved RF or XGBoost classifier")


def fit_full_anchor(
    features: np.ndarray,
    labels: np.ndarray,
    anchor_name: str,
    seed: int,
    *,
    path: Path,
) -> AnchorFitResult:
    """Fit a full-train anchor, atomically serialize it, then hash the closed file."""
    feature_array, label_array = _validate_anchor_inputs(features, labels, anchor_name)
    model = build_anchor(anchor_name, seed, label_array)
    model.fit(feature_array, label_array)

    destination = Path(path)
    _serialize_anchor(model, destination)
    return AnchorFitResult(
        model=model,
        path=destination,
        sha256=sha256_file(destination),
        size_bytes=destination.stat().st_size,
        complexity=anchor_complexity(model),
    )


def _serialize_anchor(model: Pipeline, destination: Path) -> None:
    """Atomically serialize one fitted model, refusing to replace existing artifacts."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"anchor artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        joblib.dump(model, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_fold_output_directory(output_dir: Path) -> Path:
    """Require a dedicated empty directory so OOF artifacts cannot be mixed or replaced."""
    directory = Path(output_dir)
    if not directory.exists() or not directory.is_dir():
        raise ValueError("OOF output_dir must be an existing empty directory")
    if any(directory.iterdir()):
        raise ValueError("OOF output_dir must be empty to prevent ambiguous artifacts")
    return directory


def build_grouped_oof(
    features: np.ndarray,
    labels: np.ndarray,
    track_ids: np.ndarray,
    anchor_name: str,
    seed: int,
    *,
    output_dir: Path,
) -> GroupedOOFResult:
    """Fit five fresh track-grouped anchor folds and return original-order OOF scores."""
    feature_array, label_array = _validate_anchor_inputs(features, labels, anchor_name)
    directory = _validate_fold_output_directory(output_dir)
    group_array = np.asarray(track_ids).reshape(-1)
    if group_array.size != label_array.size or pd.isna(group_array).any():
        raise ValueError("OOF track IDs must be nonmissing and align with labels")
    if np.unique(group_array).size < 5:
        raise ValueError("grouped OOF requires at least five unique track IDs")

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    probabilities = np.full(label_array.shape, np.nan, dtype=np.float64)
    assignment_count = np.zeros(label_array.shape, dtype=np.int64)
    fold_assignment = np.zeros(label_array.shape, dtype=np.int64)
    audit_rows: list[dict[str, int | str]] = []
    fold_artifacts: list[FoldModelArtifact] = []
    for fold, (train_index, holdout_index) in enumerate(
        splitter.split(feature_array, label_array, groups=group_array), start=1
    ):
        train_tracks = np.unique(group_array[train_index])
        holdout_tracks = np.unique(group_array[holdout_index])
        shared_track_count = int(np.intersect1d(train_tracks, holdout_tracks).size)
        if shared_track_count:
            raise RuntimeError("track-grouped OOF split leaked track IDs")
        fold_model = build_anchor(anchor_name, seed, label_array[train_index])
        fold_model.fit(feature_array[train_index], label_array[train_index])
        probabilities[holdout_index] = fold_model.predict_proba(feature_array[holdout_index])[:, 1]
        assignment_count[holdout_index] += 1
        fold_assignment[holdout_index] = fold
        destination = directory / f"{anchor_name}__seed{seed}__fold{fold}.joblib"
        _serialize_anchor(fold_model, destination)
        artifact = FoldModelArtifact(
            fold=fold,
            path=destination,
            sha256=sha256_file(destination),
            size_bytes=destination.stat().st_size,
            complexity=anchor_complexity(fold_model),
            train_rows=int(train_index.size),
            holdout_rows=int(holdout_index.size),
            train_positive_rows=int(label_array[train_index].sum()),
            holdout_positive_rows=int(label_array[holdout_index].sum()),
            train_negative_rows=int((label_array[train_index] == 0).sum()),
            holdout_negative_rows=int((label_array[holdout_index] == 0).sum()),
            train_unique_tracks=int(train_tracks.size),
            holdout_unique_tracks=int(holdout_tracks.size),
            shared_track_count=shared_track_count,
        )
        fold_artifacts.append(artifact)
        audit_rows.append(
            {
                "fold": artifact.fold,
                "artifact_path": str(artifact.path),
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
                "tree_count": artifact.complexity.tree_count,
                "node_count": artifact.complexity.node_count,
                "leaf_count": artifact.complexity.leaf_count,
                "train_rows": artifact.train_rows,
                "holdout_rows": artifact.holdout_rows,
                "train_positive_rows": artifact.train_positive_rows,
                "holdout_positive_rows": artifact.holdout_positive_rows,
                "train_negative_rows": artifact.train_negative_rows,
                "holdout_negative_rows": artifact.holdout_negative_rows,
                "train_unique_tracks": artifact.train_unique_tracks,
                "holdout_unique_tracks": artifact.holdout_unique_tracks,
                "shared_track_count": artifact.shared_track_count,
            }
        )
    if (
        not np.all(assignment_count == 1)
        or not np.all((1 <= fold_assignment) & (fold_assignment <= 5))
        or not np.isfinite(probabilities).all()
        or len(fold_artifacts) != 5
    ):
        raise RuntimeError("grouped OOF must assign every original training row exactly once")
    return GroupedOOFResult(
        probabilities=probabilities,
        assignment_count=assignment_count,
        fold_assignment=fold_assignment,
        fold_audit=pd.DataFrame.from_records(audit_rows),
        fold_artifacts=tuple(fold_artifacts),
    )


def _parity_arrays(frame: pd.DataFrame, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {"sequence_id", "label", "probability"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{name} parity frame is missing required columns")
    ordered = frame.loc[:, ["sequence_id", "label", "probability"]].reset_index(drop=True)
    sequence_ids = ordered["sequence_id"].astype(str).to_numpy()
    labels = ordered["label"].to_numpy(dtype=np.int64)
    probabilities = ordered["probability"].to_numpy(dtype=np.float64)
    if (
        len(ordered) == 0
        or pd.isna(ordered["sequence_id"]).any()
        or pd.Series(sequence_ids).duplicated().any()
        or not np.isin(labels, (0, 1)).all()
        or not np.isfinite(probabilities).all()
        or not ((0.0 <= probabilities) & (probabilities <= 1.0)).all()
    ):
        raise ValueError(f"{name} parity frame is malformed")
    return sequence_ids, labels, probabilities


def verify_anchor_parity(
    fresh: pd.DataFrame,
    historical: pd.DataFrame,
    *,
    tolerance: float = ExperimentConfig().parity_tolerance,
) -> AnchorParityResult:
    """Compare exact-order validation predictions and fixed-0.5 metrics fail-closed."""
    if tolerance != ExperimentConfig().parity_tolerance:
        raise ValueError("anchor parity tolerance is frozen at 1e-7")
    fresh_ids, fresh_labels, fresh_probabilities = _parity_arrays(fresh, "fresh")
    historical_ids, historical_labels, historical_probabilities = _parity_arrays(
        historical, "historical"
    )
    same_length = fresh_ids.size == historical_ids.size
    id_match = same_length and np.array_equal(fresh_ids, historical_ids)
    label_match = same_length and np.array_equal(fresh_labels, historical_labels)
    if same_length:
        fresh_classes = (fresh_probabilities >= 0.5).astype(np.int64)
        historical_classes = (historical_probabilities >= 0.5).astype(np.int64)
        class_match = np.array_equal(fresh_classes, historical_classes)
        max_probability_error = float(
            np.max(np.abs(fresh_probabilities - historical_probabilities))
        )
    else:
        class_match = False
        max_probability_error = float("inf")

    # Metrics intentionally use each vector's own labels so label parity cannot
    # be masked by recomputing both sides against a substituted label array.
    try:
        fresh_metrics = compute_binary_metrics(fresh_labels, fresh_probabilities)
        historical_metrics = compute_binary_metrics(historical_labels, historical_probabilities)
        max_metric_error = float(
            max(
                abs(fresh_metrics.f1 - historical_metrics.f1),
                abs(fresh_metrics.pr_auc - historical_metrics.pr_auc),
                abs(fresh_metrics.roc_auc - historical_metrics.roc_auc),
            )
        )
    except ValueError:
        max_metric_error = float("inf")

    failed_checks: list[str] = []
    if not id_match:
        failed_checks.append("sequence_id")
    if not label_match:
        failed_checks.append("label")
    if not class_match:
        failed_checks.append("fixed_0.5_class")
    if max_probability_error > tolerance:
        failed_checks.append("probability")
    if max_metric_error > 1e-10:
        failed_checks.append("metrics")
    return AnchorParityResult(
        passed=not failed_checks,
        id_match=id_match,
        label_match=label_match,
        class_match=class_match,
        max_probability_error=max_probability_error,
        max_metric_error=max_metric_error,
        failed_checks=tuple(failed_checks),
    )


__all__ = [
    "ANCHOR_FEATURE_DIMENSIONS",
    "ANCHOR_FEATURE_SETS",
    "AnchorFitResult",
    "FoldModelArtifact",
    "GroupedOOFResult",
    "anchor_complexity",
    "anchor_feature_metadata",
    "build_anchor",
    "build_grouped_oof",
    "fit_full_anchor",
    "verify_anchor_parity",
]
