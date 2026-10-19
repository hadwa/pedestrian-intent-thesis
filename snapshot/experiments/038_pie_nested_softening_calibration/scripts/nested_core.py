"""Nested folds, paired uncertainty, and frozen gates for Exp038."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


DEVELOPMENT_SETS = ("set01", "set02", "set04", "set05")
CANDIDATE = "soft_temp_joint_temporal_ego_xgb"
CALIBRATED_BASELINES = ("soft_temp_v2_bbox_xgb", "soft_temp_v1_bbox_rf")


@dataclass(frozen=True)
class InnerFold:
    training_sets: tuple[str, ...]
    calibration_set: str


@dataclass(frozen=True)
class OuterFold:
    held_out_set: str
    remaining_sets: tuple[str, ...]
    inner_folds: tuple[InnerFold, ...]


@dataclass(frozen=True)
class GateDecision:
    status: Literal["advance", "no_advance"]
    checks: dict[str, bool]
    observed: dict[str, float | int]


def nested_fold_plan(sets: tuple[str, ...]) -> tuple[OuterFold, ...]:
    canonical = tuple(sorted(map(str, sets)))
    if canonical != DEVELOPMENT_SETS or len(canonical) != len(set(canonical)):
        raise ValueError("nested calibration requires the frozen development set universe")
    plans = []
    for held_out in canonical:
        remaining = tuple(set_id for set_id in canonical if set_id != held_out)
        inner = tuple(
            InnerFold(tuple(set_id for set_id in remaining if set_id != calibration), calibration)
            for calibration in remaining
        )
        plans.append(OuterFold(held_out, remaining, inner))
    return tuple(plans)


def validate_outer_predictions(predictions: pd.DataFrame, cohort: pd.DataFrame) -> None:
    identity = ["sequence_id", "set_id", "track_id", "video_id", "label"]
    if set(identity + ["arm", "probability"]).difference(predictions.columns):
        raise ValueError("outer prediction schema is invalid")
    if predictions.duplicated(["sequence_id", "arm"]).any():
        raise ValueError("outer prediction identity is duplicated")
    expected_ids = cohort.loc[:, identity].astype({"label": int}).sort_values("sequence_id").reset_index(drop=True)
    for arm, rows in predictions.groupby("arm", sort=True):
        observed = rows.loc[:, identity].astype({"label": int}).sort_values("sequence_id").reset_index(drop=True)
        if not observed.equals(expected_ids):
            raise ValueError(f"outer prediction cohort mismatch for {arm}")
        probability = rows["probability"].to_numpy(float)
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError("outer probabilities are invalid")


def evaluate_promotion_gate(
    raw_results: pd.DataFrame,
    calibrated_results: pd.DataFrame,
    per_set: pd.DataFrame,
    bootstrap: pd.DataFrame,
    *,
    fixed_tail_limit: int,
) -> GateDecision:
    raw = raw_results.set_index("arm").loc["joint_temporal_ego_xgb"]
    calibrated = calibrated_results.set_index("arm")
    candidate = calibrated.loc[CANDIDATE]
    v2 = calibrated.loc["soft_temp_v2_bbox_xgb"]
    v1 = calibrated.loc["soft_temp_v1_bbox_rf"]
    evidence = bootstrap.set_index("metric")
    confusion = ("tn", "fp", "fn", "tp")
    combined_tail = int(candidate["fp_ge_0p9_count"] + candidate["fn_le_0p1_count"])
    checks = {
        "pr_auc_noninferiority": bool(float(candidate["pr_auc"]) >= float(raw["pr_auc"]) - 0.002),
        "f1_invariant": bool(float(candidate["f1_fixed_0p5"]) == float(raw["f1_fixed_0p5"])),
        "confusion_invariant": all(int(candidate[name]) == int(raw[name]) for name in confusion),
        "pr_auc_gain": bool(float(candidate["pr_auc"]) > float(v2["pr_auc"]) + 0.005),
        "f1_gain": bool(
            float(candidate["f1_fixed_0p5"])
            > max(float(v2["f1_fixed_0p5"]), float(v1["f1_fixed_0p5"])) + 0.005
        ),
        "bootstrap_pr_positive": bool(float(evidence.loc["pr_auc", "ci_low"]) > 0.0),
        "bootstrap_f1_positive": bool(float(evidence.loc["f1_fixed_0p5", "ci_low"]) > 0.0),
        "per_set_pr": bool(
            set(per_set["set_id"].astype(str)) == set(DEVELOPMENT_SETS)
            and float(per_set["pr_auc_delta_vs_v2"].min()) >= -0.01
        ),
        "per_set_f1": bool(
            set(per_set["set_id"].astype(str)) == set(DEVELOPMENT_SETS)
            and float(per_set["f1_delta_vs_best"].min()) >= -0.01
        ),
        "brier_improved": bool(float(candidate["brier"]) < float(raw["brier"])),
        "log_loss_improved": bool(float(candidate["log_loss"]) < float(raw["log_loss"])),
        "ece_nonincrease": bool(float(candidate["ece"]) <= float(raw["ece"])),
        "fp_tail_nonincrease": bool(int(candidate["fp_ge_0p9_count"]) <= int(raw["fp_ge_0p9_count"])),
        "fn_tail_nonincrease": bool(int(candidate["fn_le_0p1_count"]) <= int(raw["fn_le_0p1_count"])),
        "combined_tail_reduction": bool(combined_tail <= int(fixed_tail_limit)),
    }
    return GateDecision(
        status="advance" if all(checks.values()) else "no_advance",
        checks=checks,
        observed={
            "candidate_pr_auc": float(candidate["pr_auc"]),
            "candidate_f1": float(candidate["f1_fixed_0p5"]),
            "combined_tail_count": combined_tail,
            "combined_tail_limit": int(fixed_tail_limit),
        },
    )


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
    groups = {track: np.flatnonzero(frame["track_id"].astype(str).to_numpy() == track) for track in tracks}
    labels = frame["label"].to_numpy(int)
    reference = frame[reference_column].to_numpy(float)
    candidate = frame[candidate_column].to_numpy(float)
    rng = np.random.default_rng(seed)
    pr_values = np.empty(replicates, dtype=float)
    f1_values = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = rng.choice(tracks, size=len(tracks), replace=True)
        positions = np.concatenate([groups[track] for track in sampled])
        selected_labels = labels[positions]
        pr_values[index] = average_precision_score(selected_labels, candidate[positions]) - average_precision_score(
            selected_labels, reference[positions]
        )
        f1_values[index] = f1_score(selected_labels, candidate[positions] >= 0.5) - f1_score(
            selected_labels, reference[positions] >= 0.5
        )
    rows = []
    for metric, values in (("pr_auc", pr_values), ("f1_fixed_0p5", f1_values)):
        rows.append(
            {
                "metric": metric,
                "replicates": int(replicates),
                "seed": int(seed),
                "mean_delta": float(values.mean()),
                "median_delta": float(np.median(values)),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
            }
        )
    return pd.DataFrame(rows)
