"""Frozen models, metrics, grouped evaluation, and gates for Exp037."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


THRESHOLD = 0.5
SEEDS = (42, 43, 44)
BASELINE_ARMS = ("v2_bbox_xgb", "v1_bbox_rf")
CANDIDATE_ARMS = ("joint_xgb", "joint_rf", "joint_temporal_xgb", "joint_temporal_ego_xgb")


@dataclass(frozen=True)
class FoldPlan:
    held_out_set: str
    training_sets: tuple[str, ...]


def _validate_binary(labels: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or probabilities.ndim != 1 or len(labels) != len(probabilities) or not len(labels):
        raise ValueError("labels and probabilities must be aligned nonempty vectors")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels must be binary")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be finite in [0,1]")
    return labels, probabilities


def calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    labels, probabilities = _validate_binary(labels, probabilities)
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(probabilities, edges[1:-1], right=True), bins - 1)
    total = len(labels)
    ece = 0.0
    for index in range(bins):
        mask = assignments == index
        if mask.any():
            ece += float(mask.sum() / total) * abs(float(labels[mask].mean()) - float(probabilities[mask].mean()))
    return ece


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    labels, probabilities = _validate_binary(labels, probabilities)
    predictions = (probabilities >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "n": int(len(labels)),
        "positive_count": int(labels.sum()),
        "threshold": THRESHOLD,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1_fixed_0p5": float(f1_score(labels, predictions, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if fp + tn else float("nan"),
        "fnr": float(fn / (fn + tp)) if fn + tp else float("nan"),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else float("nan"),
        "brier": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "ece": calibration_error(labels, probabilities),
        "fp_ge_0p9_count": int(np.sum((labels == 0) & (probabilities >= 0.9))),
        "fn_le_0p1_count": int(np.sum((labels == 1) & (probabilities <= 0.1))),
    }


def leave_one_set_out_plan(frame: pd.DataFrame) -> tuple[FoldPlan, ...]:
    required = {"sequence_id", "set_id", "track_id", "label"}
    if required.difference(frame.columns) or frame.empty or frame["sequence_id"].duplicated().any():
        raise ValueError("development cohort schema or identity is invalid")
    sets = tuple(sorted(frame["set_id"].astype(str).unique()))
    if sets != ("set01", "set02", "set04", "set05"):
        raise ValueError("leave-one-set-out requires the frozen development set universe")
    return tuple(FoldPlan(held_out, tuple(item for item in sets if item != held_out)) for held_out in sets)


def _model_kind(arm: str) -> str:
    if arm in {"v2_bbox_xgb", "joint_xgb", "joint_temporal_xgb", "joint_temporal_ego_xgb"}:
        return "xgb"
    if arm in {"v1_bbox_rf", "joint_rf"}:
        return "rf"
    raise ValueError(f"unknown frozen arm {arm}")


def build_model(arm: str, seed: int, labels: np.ndarray) -> Pipeline:
    labels = np.asarray(labels, dtype=int)
    kind = _model_kind(arm)
    if kind == "rf":
        estimator = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=int(seed),
            n_jobs=-1,
        )
    else:
        positives = int((labels == 1).sum())
        negatives = int((labels == 0).sum())
        estimator = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=int(seed),
            n_jobs=4,
            scale_pos_weight=float(negatives / max(positives, 1)),
        )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])


def fit_predict_seed_mean(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    feature_columns: Sequence[str],
    arm: str,
    seeds: Sequence[int] = SEEDS,
) -> np.ndarray:
    if tuple(sorted(map(int, seeds))) != SEEDS:
        raise ValueError("Exp037 requires exactly seeds 42,43,44")
    columns = tuple(map(str, feature_columns))
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("feature columns must be unique and nonempty")
    if set(columns).difference(train.columns) or set(columns).difference(evaluation.columns):
        raise ValueError("model frame is missing frozen features")
    labels = train["label"].to_numpy(dtype=int)
    predictions = []
    for seed in sorted(map(int, seeds)):
        model = build_model(arm, seed, labels)
        model.fit(train.loc[:, columns].astype(float), labels)
        predictions.append(model.predict_proba(evaluation.loc[:, columns].astype(float))[:, 1])
    return np.column_stack(predictions).mean(axis=1)


def run_leave_one_set_out(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    arm: str,
    seeds: Sequence[int] = SEEDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    plans = leave_one_set_out_plan(frame)
    outputs = []
    audits = []
    for plan in plans:
        train = frame.loc[frame["set_id"].astype(str).isin(plan.training_sets)]
        held_out = frame.loc[frame["set_id"].astype(str).eq(plan.held_out_set)]
        probability = fit_predict_seed_mean(train, held_out, feature_columns, arm, seeds)
        outputs.append(
            held_out.loc[:, ["sequence_id", "set_id", "track_id", "video_id", "label"]]
            .assign(arm=arm, probability=probability)
        )
        audits.append(
            {
                "arm": arm,
                "held_out_set": plan.held_out_set,
                "training_sets": list(plan.training_sets),
                "train_rows": len(train),
                "evaluation_rows": len(held_out),
            }
        )
    predictions = pd.concat(outputs, ignore_index=True).sort_values("sequence_id", kind="stable").reset_index(drop=True)
    if len(predictions) != len(frame) or predictions["sequence_id"].duplicated().any():
        raise RuntimeError("OOF prediction coverage is invalid")
    return predictions, pd.DataFrame(audits)


def select_candidate(
    results: pd.DataFrame,
    per_set: pd.DataFrame,
    bootstrap: pd.DataFrame,
    *,
    minimum_pr_gain: float = 0.005,
    minimum_f1_gain: float = 0.005,
    maximum_per_set_regression: float = 0.01,
) -> dict[str, Any] | None:
    required = {
        "arm", "pr_auc", "f1_fixed_0p5", "roc_auc", "fp_ge_0p9_count",
        "fn_le_0p1_count", "n_features",
    }
    if required.difference(results.columns) or results["arm"].duplicated().any():
        raise ValueError("selection result schema is invalid")
    indexed = results.set_index("arm")
    if any(arm not in indexed.index for arm in BASELINE_ARMS):
        raise ValueError("selection results are missing dual baselines")
    v2 = indexed.loc["v2_bbox_xgb"]
    v1 = indexed.loc["v1_bbox_rf"]
    best_f1 = max(float(v2["f1_fixed_0p5"]), float(v1["f1_fixed_0p5"]))
    fp_limit = min(int(v2["fp_ge_0p9_count"]), int(v1["fp_ge_0p9_count"]))
    fn_limit = min(int(v2["fn_le_0p1_count"]), int(v1["fn_le_0p1_count"]))
    eligible = []
    for arm in CANDIDATE_ARMS:
        if arm not in indexed.index:
            continue
        row = indexed.loc[arm]
        arm_sets = per_set.loc[per_set["arm"].eq(arm)]
        arm_bootstrap = bootstrap.loc[bootstrap["arm"].eq(arm)].set_index("metric")
        if set(arm_sets.get("set_id", [])) != {"set01", "set02", "set04", "set05"}:
            continue
        if not {"pr_auc", "f1_fixed_0p5"}.issubset(arm_bootstrap.index):
            continue
        qualifies = (
            float(row["pr_auc"]) > float(v2["pr_auc"]) + minimum_pr_gain
            and float(row["f1_fixed_0p5"]) > best_f1 + minimum_f1_gain
            and int(row["fp_ge_0p9_count"]) <= fp_limit
            and int(row["fn_le_0p1_count"]) <= fn_limit
            and float(arm_bootstrap.loc["pr_auc", "ci_low"]) > 0.0
            and float(arm_bootstrap.loc["f1_fixed_0p5", "ci_low"]) > 0.0
            and float(arm_sets["pr_auc_delta_vs_v2"].min()) >= -maximum_per_set_regression
            and float(arm_sets["f1_delta_vs_best"].min()) >= -maximum_per_set_regression
        )
        if qualifies:
            eligible.append(row.to_dict() | {"arm": arm})
    eligible.sort(
        key=lambda row: (
            -float(row["pr_auc"]),
            -float(row["f1_fixed_0p5"]),
            -float(row["roc_auc"]),
            int(row["fp_ge_0p9_count"]) + int(row["fn_le_0p1_count"]),
            int(row["n_features"]),
        )
    )
    return eligible[0] if eligible else None


def cluster_bootstrap_deltas(
    frame: pd.DataFrame,
    reference_column: str,
    candidate_column: str,
    *,
    replicates: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    required = {"track_id", "label", reference_column, candidate_column}
    if required.difference(frame.columns) or frame.empty:
        raise ValueError("bootstrap frame schema is invalid")
    tracks = np.asarray(sorted(frame["track_id"].astype(str).unique()))
    groups = {track: frame.index[frame["track_id"].astype(str).eq(track)].to_numpy() for track in tracks}
    rng = np.random.default_rng(seed)
    deltas = {"pr_auc": [], "f1_fixed_0p5": []}
    for _ in range(int(replicates)):
        sampled = rng.choice(tracks, size=len(tracks), replace=True)
        indices = np.concatenate([groups[track] for track in sampled])
        labels = frame.loc[indices, "label"].to_numpy(dtype=int)
        reference = compute_metrics(labels, frame.loc[indices, reference_column].to_numpy(dtype=float))
        candidate = compute_metrics(labels, frame.loc[indices, candidate_column].to_numpy(dtype=float))
        for metric in deltas:
            deltas[metric].append(float(candidate[metric]) - float(reference[metric]))
    rows = []
    for metric, values in deltas.items():
        array = np.asarray(values, dtype=float)
        rows.append(
            {
                "metric": metric,
                "replicates": int(replicates),
                "mean_delta": float(array.mean()),
                "ci_low": float(np.quantile(array, 0.025)),
                "ci_high": float(np.quantile(array, 0.975)),
            }
        )
    return pd.DataFrame(rows)
