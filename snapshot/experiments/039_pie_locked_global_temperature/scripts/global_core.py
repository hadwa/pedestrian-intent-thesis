"""Pure locked-temperature transformation and set06 gate for Exp039."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.special import expit


@dataclass(frozen=True)
class GateDecision:
    status: Literal["advance", "no_advance"]
    checks: dict[str, bool]
    observed: dict[str, float | int | str]


def apply_locked_temperature(probability: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(probability, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("probabilities must be a nonempty finite vector")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("probabilities must lie in [0,1]")
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature < 1.0:
        raise ValueError("temperature must be finite and at least one")
    clipped = np.clip(values, 1e-12, 1.0 - 1e-12)
    return expit((np.log(clipped) - np.log1p(-clipped)) / temperature)


def verify_global_invariants(raw: np.ndarray, calibrated: np.ndarray) -> None:
    raw = np.asarray(raw, dtype=float)
    calibrated = np.asarray(calibrated, dtype=float)
    if raw.shape != calibrated.shape or raw.ndim != 1 or not len(raw):
        raise ValueError("raw and calibrated probabilities must align")
    if not np.array_equal(raw >= 0.5, calibrated >= 0.5):
        raise ValueError("fixed-0.5 decision invariant failed")
    if not np.array_equal(np.argsort(raw, kind="stable"), np.argsort(calibrated, kind="stable")):
        raise ValueError("global ranking invariant failed")
    if np.any(np.abs(calibrated - 0.5) > np.abs(raw - 0.5) + 1e-14):
        raise ValueError("softening direction invariant failed")


def evaluate_confirmation_gate(
    raw_results: pd.DataFrame,
    calibrated_results: pd.DataFrame,
    bootstrap: pd.DataFrame,
    *,
    minimum_pr_gain: float = 0.005,
    minimum_f1_gain: float = 0.005,
    tail_reduction_fraction: float = 0.25,
) -> GateDecision:
    raw = raw_results.set_index("arm").loc["joint_temporal_ego_xgb"]
    calibrated = calibrated_results.set_index("arm")
    candidate = calibrated.loc["soft_temp_joint_temporal_ego_xgb"]
    v2 = calibrated.loc["soft_temp_v2_bbox_xgb"]
    v1 = calibrated.loc["soft_temp_v1_bbox_rf"]
    evidence = bootstrap.set_index("metric")
    best_f1_arm = max(
        ("soft_temp_v2_bbox_xgb", "soft_temp_v1_bbox_rf"),
        key=lambda arm: float(calibrated.loc[arm, "f1_fixed_0p5"]),
    )
    raw_tail = int(raw["fp_ge_0p9_count"] + raw["fn_le_0p1_count"])
    calibrated_tail = int(candidate["fp_ge_0p9_count"] + candidate["fn_le_0p1_count"])
    tail_limit = int(np.floor(raw_tail * (1.0 - float(tail_reduction_fraction))))
    confusion = ("tn", "fp", "fn", "tp")
    checks = {
        "pr_auc_gain": bool(float(candidate["pr_auc"]) > float(v2["pr_auc"]) + minimum_pr_gain),
        "f1_gain": bool(
            float(candidate["f1_fixed_0p5"])
            > float(calibrated.loc[best_f1_arm, "f1_fixed_0p5"]) + minimum_f1_gain
        ),
        "bootstrap_pr_positive": bool(float(evidence.loc["pr_auc", "ci_low"]) > 0.0),
        "bootstrap_f1_positive": bool(float(evidence.loc["f1_fixed_0p5", "ci_low"]) > 0.0),
        "confusion_invariant": all(int(candidate[name]) == int(raw[name]) for name in confusion),
        "brier_improved": bool(float(candidate["brier"]) < float(raw["brier"])),
        "log_loss_improved": bool(float(candidate["log_loss"]) < float(raw["log_loss"])),
        "ece_nonincrease": bool(float(candidate["ece"]) <= float(raw["ece"])),
        "fp_tail_nonincrease": bool(
            int(candidate["fp_ge_0p9_count"]) <= int(raw["fp_ge_0p9_count"])
        ),
        "fn_tail_nonincrease": bool(
            int(candidate["fn_le_0p1_count"]) <= int(raw["fn_le_0p1_count"])
        ),
        "combined_tail_reduction": bool(calibrated_tail <= tail_limit),
    }
    return GateDecision(
        status="advance" if all(checks.values()) else "no_advance",
        checks=checks,
        observed={
            "candidate_pr_auc": float(candidate["pr_auc"]),
            "v2_pr_auc": float(v2["pr_auc"]),
            "candidate_f1": float(candidate["f1_fixed_0p5"]),
            "best_f1_reference_arm": best_f1_arm,
            "best_baseline_f1": float(calibrated.loc[best_f1_arm, "f1_fixed_0p5"]),
            "raw_combined_tail": raw_tail,
            "calibrated_combined_tail": calibrated_tail,
            "combined_tail_limit": tail_limit,
        },
    )
