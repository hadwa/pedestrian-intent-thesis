"""Frozen four-arm XGBoost assembly and evaluation for Exp081."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
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


SCRIPT_DIR = Path(__file__).resolve().parent


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Exp081 dependency from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CONTRACT = _load(SCRIPT_DIR / "contract.py", "exp081_contract_for_model")
TWO_STAGE_PATH = CONTRACT.resolve_source("exp076_two_stage_core")
TWO_STAGE_CORE = _load(TWO_STAGE_PATH, "exp081_authenticated_two_stage_core")
DEFINITIONS_PATH = CONTRACT.resolve_source("exp012_feature_definitions")
ANCHOR_COLUMNS = tuple(
    TWO_STAGE_CORE.load_joint_contract(DEFINITIONS_PATH)["portable"]
)
if len(ANCHOR_COLUMNS) != 182 or len(set(ANCHOR_COLUMNS)) != 182:
    raise RuntimeError("Exp081 authenticated portable anchor is not 182-D")

PROXY_COLUMNS = tuple(
    f"motion_proxy__{channel}__{summary}"
    for channel in ("translation_x_norm", "translation_y_norm", "rotation_rad", "log_scale")
    for summary in ("early_mean", "late_mean", "late_minus_early")
) + (
    "motion_proxy__valid_transition_fraction",
    "motion_proxy__mean_valid_inlier_fraction",
    "motion_proxy__final_cumulative_translation_norm",
)
ROAD_CHANNELS = tuple(CONTRACT.load_registration()["road_features"]["channels"])
ROAD_BASE_COLUMNS = tuple(
    f"{channel}__{summary}"
    for channel in ROAD_CHANNELS
    for summary in ("early_mean", "late_mean", "late_minus_early")
)
RAW_ROAD_COLUMNS = tuple(f"road_raw__{name}" for name in ROAD_BASE_COLUMNS)
COMPENSATED_ROAD_COLUMNS = tuple(
    f"road_compensated__{name}" for name in ROAD_BASE_COLUMNS
)
if not (
    len(PROXY_COLUMNS) == 15
    and len(RAW_ROAD_COLUMNS) == 60
    and len(COMPENSATED_ROAD_COLUMNS) == 60
):
    raise RuntimeError("Exp081 context feature ABI changed")


def anchor_columns() -> tuple[str, ...]:
    return ANCHOR_COLUMNS


def proxy_columns() -> tuple[str, ...]:
    return PROXY_COLUMNS


def raw_road_columns() -> tuple[str, ...]:
    return RAW_ROAD_COLUMNS


def compensated_road_columns() -> tuple[str, ...]:
    return COMPENSATED_ROAD_COLUMNS


def feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(column for column in frame.columns if column not in {"sequence_id", "label"})


def _checked_anchor(anchor: pd.DataFrame) -> pd.DataFrame:
    required = {"sequence_id", "label", *ANCHOR_COLUMNS}
    missing = required - set(anchor.columns)
    if missing:
        raise ValueError(f"Exp081 anchor is missing columns: {sorted(missing)[:5]}")
    checked = anchor.loc[:, ["sequence_id", "label", *ANCHOR_COLUMNS]].copy()
    if checked["sequence_id"].astype(str).duplicated().any():
        raise ValueError("Exp081 anchor sequence IDs are not unique")
    if not checked["label"].astype(int).isin([0, 1]).all():
        raise ValueError("Exp081 anchor labels are not binary")
    return checked


def assemble_variants(
    anchor: pd.DataFrame, context: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Join context one-to-one without changing anchor population or order."""
    checked = _checked_anchor(anchor)
    required_context = {
        "sequence_id",
        *PROXY_COLUMNS,
        *RAW_ROAD_COLUMNS,
        *COMPENSATED_ROAD_COLUMNS,
    }
    missing = required_context - set(context.columns)
    if missing:
        raise ValueError(f"Exp081 context is missing columns: {sorted(missing)[:5]}")
    if "label" in context.columns:
        raise ValueError("Exp081 extracted context must not contain labels")
    selected_context = context.loc[:, list(required_context)].copy()
    if selected_context["sequence_id"].astype(str).duplicated().any():
        raise ValueError("Exp081 context sequence IDs are not unique")
    merged = checked.merge(
        selected_context, on="sequence_id", how="left", validate="one_to_one", sort=False
    )
    if len(merged) != len(checked):
        raise RuntimeError("Exp081 context join changed the cohort")
    np.testing.assert_array_equal(
        merged["sequence_id"].astype(str).to_numpy(),
        checked["sequence_id"].astype(str).to_numpy(),
    )
    if merged.loc[:, [*PROXY_COLUMNS, *RAW_ROAD_COLUMNS, *COMPENSATED_ROAD_COLUMNS]].isna().any().any():
        raise RuntimeError("Exp081 context join is incomplete")
    meta = ["sequence_id", "label"]
    variants = {
        "A": merged.loc[:, [*meta, *ANCHOR_COLUMNS]].copy(),
        "B": merged.loc[:, [*meta, *ANCHOR_COLUMNS, *PROXY_COLUMNS]].copy(),
        "C": merged.loc[:, [*meta, *ANCHOR_COLUMNS, *RAW_ROAD_COLUMNS]].copy(),
        "D": merged.loc[:, [*meta, *ANCHOR_COLUMNS, *COMPENSATED_ROAD_COLUMNS]].copy(),
    }
    expected_dimensions = CONTRACT.load_registration()["dimensions"]
    for name, frame in variants.items():
        if len(feature_columns(frame)) != int(expected_dimensions[name]):
            raise RuntimeError(f"Exp081 Variant {name} dimensionality changed")
        np.testing.assert_array_equal(frame["label"], checked["label"])
    return variants


def build_xgb(seed: int, labels: np.ndarray):
    """Delegate exactly to the authenticated Exp077 model constructor."""
    pipeline = TWO_STAGE_CORE.build_xgb(
        int(seed), np.asarray(labels, dtype=int), n_estimators=250
    )
    model = pipeline.named_steps["model"]
    frozen = CONTRACT.load_registration()["xgboost"]
    params = model.get_params()
    checks = {
        "n_estimators": int(params["n_estimators"]),
        "max_depth": int(params["max_depth"]),
        "learning_rate": float(params["learning_rate"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample_bytree"]),
        "reg_lambda": float(params["reg_lambda"]),
        "n_jobs": int(params["n_jobs"]),
    }
    for name, value in checks.items():
        if value != frozen[name]:
            raise RuntimeError(f"Exp081 XGBoost parameter drifted: {name}")
    return pipeline


def fixed_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if (
        labels.ndim != 1
        or probabilities.shape != labels.shape
        or set(np.unique(labels)) != {0, 1}
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0.0) | (probabilities > 1.0))
    ):
        raise ValueError("Exp081 metrics require aligned binary labels and probabilities")
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    high_fp = int(((labels == 0) & (probabilities >= 0.9)).sum())
    high_fn = int(((labels == 1) & (probabilities <= 0.1)).sum())
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "high_confidence_fp": high_fp,
        "high_confidence_fn": high_fn,
        "high_confidence_errors": high_fp + high_fn,
    }


def paired_transitions(
    sequence_ids: np.ndarray,
    labels: np.ndarray,
    left_probabilities: np.ndarray,
    right_probabilities: np.ndarray,
    *,
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    ids = np.asarray(sequence_ids).astype(str)
    labels = np.asarray(labels, dtype=int)
    left = np.asarray(left_probabilities, dtype=float)
    right = np.asarray(right_probabilities, dtype=float)
    if not (ids.shape == labels.shape == left.shape == right.shape):
        raise ValueError("Exp081 paired transitions are not aligned")
    left_correct = (left >= 0.5).astype(int) == labels
    right_correct = (right >= 0.5).astype(int) == labels
    return pd.DataFrame(
        {
            "sequence_id": ids,
            "label": labels,
            f"{left_name}_probability": left,
            f"{right_name}_probability": right,
            "transition": np.select(
                [~left_correct & right_correct, left_correct & ~right_correct],
                [f"{left_name}_wrong_to_{right_name}_correct", f"{left_name}_correct_to_{right_name}_wrong"],
                default="unchanged_correctness",
            ),
        }
    )


def validation_gate(per_seed: pd.DataFrame) -> dict[str, Any]:
    required = {"seed", "variant", "f1", "pr_auc", "roc_auc", "high_confidence_errors"}
    missing = required - set(per_seed.columns)
    if missing:
        raise ValueError(f"Exp081 validation gate is missing: {sorted(missing)}")
    selected = per_seed.loc[per_seed["variant"].isin(["A", "C", "D"])].copy()
    if selected.duplicated(["seed", "variant"]).any():
        raise ValueError("Exp081 validation results contain duplicate arms")
    seeds = tuple(CONTRACT.load_registration()["seeds"])
    if set(selected["seed"].astype(int)) != set(seeds):
        raise ValueError("Exp081 validation seeds changed")
    indexed = selected.set_index(["seed", "variant"])
    for seed in seeds:
        for variant in ("A", "C", "D"):
            if (seed, variant) not in indexed.index:
                raise ValueError(f"Exp081 validation result missing {variant}{seed}")

    def values(variant: str, metric: str) -> np.ndarray:
        return np.asarray(
            [float(indexed.loc[(seed, variant), metric]) for seed in seeds]
        )

    d_minus_a_pr = values("D", "pr_auc") - values("A", "pr_auc")
    d_minus_a_f1 = float(values("D", "f1").mean() - values("A", "f1").mean())
    d_minus_a_roc = float(
        values("D", "roc_auc").mean() - values("A", "roc_auc").mean()
    )
    d_minus_c_pr = float(
        values("D", "pr_auc").mean() - values("C", "pr_auc").mean()
    )
    high_error_delta = int(
        values("D", "high_confidence_errors").sum()
        - values("A", "high_confidence_errors").sum()
    )
    conditions = {
        "D_minus_A_pr_auc_positive_all_seeds": bool((d_minus_a_pr > 0.0).all()),
        "D_minus_A_mean_f1_noninferior": d_minus_a_f1 >= -0.005,
        "D_minus_A_mean_roc_auc_noninferior": d_minus_a_roc >= -0.005,
        "D_high_confidence_errors_no_more_than_A": high_error_delta <= 0,
        "D_minus_C_mean_pr_auc_positive": d_minus_c_pr > 0.0,
    }
    return {
        "gate_status": "advance" if all(conditions.values()) else "no_advance",
        "conditions": conditions,
        "deltas": {
            "D_minus_A_pr_auc_by_seed": {
                str(seed): float(delta) for seed, delta in zip(seeds, d_minus_a_pr)
            },
            "D_minus_A_mean_f1": d_minus_a_f1,
            "D_minus_A_mean_roc_auc": d_minus_a_roc,
            "D_minus_C_mean_pr_auc": d_minus_c_pr,
            "D_minus_A_combined_high_confidence_errors": high_error_delta,
        },
    }

