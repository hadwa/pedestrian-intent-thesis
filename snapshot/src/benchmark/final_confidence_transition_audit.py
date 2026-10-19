from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import json
import math
from pathlib import Path

import pandas as pd

from scripts.experiments import final_confidence_benchmark as final_benchmark
from scripts.experiments import jaad_biu_v2_benchmark_comparison as jaad_benchmark
from src.benchmark.final_confidence_feature_views import (
    build_biu_joint_map,
    make_feature_view,
)
from src.benchmark.final_confidence_sources import collect_global_confidence_columns


STATE_MAP = {
    (0, 0): "TN",
    (0, 1): "FP",
    (1, 0): "FN",
    (1, 1): "TP",
}

REPAIRABLE_JAAD_CONFIG = ("jaad", "30_60")
CLEAN_ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_JAAD_CACHE = Path(
    "/home/abdelrahman/Hadwa/BIU-PID-Distillation/data/jaad_keypoints_vitpose_s_extended_short_horizon.jsonl"
)
SUPPLEMENTAL_JAAD_CACHE = (
    CLEAN_ROOT
    / "experiments/030_final_confidence_benchmark/jaad30_repair/jaad30_missing_supplemental_keypoints.jsonl"
)


def load_selected_prediction_payload(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text())
    frame = pd.DataFrame(payload)
    required = {"seed", "selected_threshold", "test_predictions"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction payload {path} is missing columns: {sorted(missing)}")
    return frame.sort_values("seed").reset_index(drop=True)


def select_seed_prediction_row(payload: pd.DataFrame, seed: int) -> pd.Series:
    matches = payload.loc[payload["seed"].eq(int(seed))].reset_index(drop=True)
    if matches.empty:
        raise ValueError(f"Prediction payload is missing required seed {seed}")
    if len(matches) != 1:
        raise ValueError(f"Prediction payload has duplicate rows for seed {seed}")
    return matches.iloc[0]


def rebuild_fixed_threshold_predictions(
    predictions: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    rebuilt = predictions.copy(deep=True)
    rebuilt["selected_threshold"] = float(threshold)
    new_rows = []
    for rows in rebuilt["test_predictions"]:
        updated = []
        for row in rows:
            copied = dict(row)
            copied["prediction"] = int(float(copied["probability"]) >= threshold)
            updated.append(copied)
        new_rows.append(updated)
    rebuilt["test_predictions"] = new_rows
    return rebuilt


def classify_binary_state(label: int, prediction: int) -> str:
    return STATE_MAP[(int(label), int(prediction))]


def assign_transition_label(
    label: int, baseline_prediction: int, augmented_prediction: int
) -> str:
    baseline_state = classify_binary_state(label, baseline_prediction)
    augmented_state = classify_binary_state(label, augmented_prediction)
    return f"{baseline_state}->{augmented_state}"


def summarize_transition_family_denominator(transition_label: str) -> str:
    return transition_label.split("->", maxsplit=1)[0]


def compute_transition_counts(rows: pd.DataFrame) -> pd.DataFrame:
    baseline_denominators = rows["baseline_state"].value_counts().to_dict()
    grouped = (
        rows.groupby(["transition_label", "baseline_state"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    grouped["baseline_population"] = grouped["baseline_state"].map(baseline_denominators)
    grouped["percentage_of_baseline_population"] = (
        grouped["count"] / grouped["baseline_population"]
    )
    return grouped


def compute_transition_distribution_summary(rows: pd.DataFrame) -> pd.DataFrame:
    counts = compute_transition_counts(rows)
    distributions = (
        rows.groupby(["transition_label", "baseline_state"], dropna=False)
        .agg(
            primary_reliability_mean=("primary_reliability", "mean"),
            primary_reliability_median=("primary_reliability", "median"),
            conservative_reliability_mean=("conservative_reliability", "mean"),
            conservative_reliability_median=("conservative_reliability", "median"),
            probability_shift_mean=("probability_shift", "mean"),
            probability_shift_median=("probability_shift", "median"),
            baseline_probability_mean=("probability_baseline", "mean"),
            baseline_probability_median=("probability_baseline", "median"),
            augmented_probability_mean=("probability_augmented", "mean"),
            augmented_probability_median=("probability_augmented", "median"),
        )
        .reset_index()
    )
    return counts.merge(
        distributions,
        on=["transition_label", "baseline_state"],
        how="left",
        validate="one_to_one",
    )


def compute_error_accounting(rows: pd.DataFrame) -> dict[str, float]:
    baseline_errors = int(rows["baseline_state"].isin(["FP", "FN"]).sum())
    baseline_correct = int(rows["baseline_state"].isin(["TN", "TP"]).sum())
    corrected = int(rows["transition_label"].isin(["FP->TN", "FN->TP"]).sum())
    new_errors = int(rows["transition_label"].isin(["TN->FP", "TP->FN"]).sum())
    correction_rate = corrected / baseline_errors if baseline_errors else math.nan
    error_introduction_rate = new_errors / baseline_correct if baseline_correct else math.nan
    return {
        "baseline_errors": baseline_errors,
        "baseline_correct_cases": baseline_correct,
        "corrected_errors": corrected,
        "new_errors": new_errors,
        "net_corrected_errors": corrected - new_errors,
        "correction_rate": correction_rate,
        "error_introduction_rate": error_introduction_rate,
        "net_error_change": new_errors - corrected,
    }


@lru_cache(maxsize=1)
def _load_supplemental_jaad_keypoints() -> dict[tuple[str, str, int], dict]:
    if not SUPPLEMENTAL_JAAD_CACHE.exists():
        raise ValueError(
            f"Supplemental JAAD repair cache is missing: {SUPPLEMENTAL_JAAD_CACHE}"
        )
    records: dict[tuple[str, str, int], dict] = {}
    with SUPPLEMENTAL_JAAD_CACHE.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = (str(record["video"]), str(record["pid"]), int(record["frame_idx"]))
            records[key] = record
    return records


@contextmanager
def _patched_jaad_keypoint_index(config_key: tuple[str, str]):
    if config_key != REPAIRABLE_JAAD_CONFIG:
        yield
        return

    original_loader = jaad_benchmark._load_keypoint_index
    supplemental = _load_supplemental_jaad_keypoints()

    def merged_loader(path: Path, needed):
        index = original_loader(path, needed)
        if Path(path) == ORIGINAL_JAAD_CACHE:
            for key in needed:
                if key not in index and key in supplemental:
                    index[key] = supplemental[key]
        return index

    jaad_benchmark._load_keypoint_index = merged_loader
    try:
        yield
    finally:
        jaad_benchmark._load_keypoint_index = original_loader


@lru_cache(maxsize=8)
def _get_canonical_frame(config_key: tuple[str, str]) -> pd.DataFrame:
    with _patched_jaad_keypoint_index(config_key):
        return final_benchmark.reconstruct_canonical_biu_v2(config_key)


def build_primary_scalar_frame(
    config_key: tuple[str, str], comparison: str
) -> pd.DataFrame:
    canonical = _get_canonical_frame(config_key).copy()
    joint_map = build_biu_joint_map()
    global_conf_cols = collect_global_confidence_columns(canonical)
    if comparison == "KP-XY -> KP-XYC":
        view, columns = make_feature_view(
            canonical,
            "kp_xyc",
            global_conf_cols=global_conf_cols,
            biu_joint_map=joint_map,
        )
        kpconf_cols = [column for column in columns if column.startswith("kpconf__")]
        if not kpconf_cols:
            raise ValueError("No kpconf__ columns found for KP-XYC scalar extraction")
        return canonical[["sequence_id", "split"]].assign(
            primary_reliability=view[kpconf_cols].mean(axis=1),
            conservative_reliability=view[kpconf_cols].min(axis=1),
        )

    representation = {
        "BIU-v2 -> BIU-v2 + BIU-specific reliability": "biu_v2_plus_reliability",
        "BBox + BIU-v2 -> BBox + BIU-v2 + reliability": "bbox_plus_biu_v2_plus_reliability",
    }[comparison]
    view, columns = make_feature_view(
        canonical,
        representation,
        global_conf_cols=global_conf_cols,
        biu_joint_map=joint_map,
    )
    rel_mean_cols = [
        column
        for column in columns
        if column.startswith("rel__") and column.endswith("__mean_conf")
    ]
    rel_min_cols = [
        column
        for column in columns
        if column.startswith("rel__") and column.endswith("__min_conf")
    ]
    if not rel_mean_cols:
        raise ValueError(
            f"No rel__*__mean_conf columns found for comparison: {comparison}"
        )
    if not rel_min_cols:
        raise ValueError(
            f"No rel__*__min_conf columns found for comparison: {comparison}"
        )
    return canonical[["sequence_id", "split"]].assign(
        primary_reliability=view[rel_mean_cols].mean(axis=1),
        conservative_reliability=view[rel_min_cols].mean(axis=1),
    )


def assemble_transition_rows(
    dataset: str,
    horizon: str,
    comparison: str,
    seed: int,
    baseline_payload: pd.DataFrame,
    augmented_payload: pd.DataFrame,
    *,
    threshold_regime: str,
    scalar_frame: pd.DataFrame,
) -> pd.DataFrame:
    baseline_row = select_seed_prediction_row(baseline_payload, seed)
    augmented_row = select_seed_prediction_row(augmented_payload, seed)
    baseline_rows = pd.DataFrame(baseline_row["test_predictions"]).reset_index(drop=True)
    augmented_rows = pd.DataFrame(augmented_row["test_predictions"]).reset_index(drop=True)
    identity_columns = ["sequence_id", "split", "label"]
    if not baseline_rows[identity_columns].equals(augmented_rows[identity_columns]):
        raise ValueError(
            f"Baseline and augmented predictions disagree on identity columns for "
            f"{dataset} {horizon} {comparison} seed {seed}"
        )
    merged = baseline_rows.rename(
        columns={
            "prediction": "prediction_baseline",
            "probability": "probability_baseline",
        }
    ).copy()
    merged["prediction_augmented"] = augmented_rows["prediction"].to_numpy()
    merged["probability_augmented"] = augmented_rows["probability"].to_numpy()
    merged["baseline_state"] = merged.apply(
        lambda row: classify_binary_state(row["label"], row["prediction_baseline"]),
        axis=1,
    )
    merged["augmented_state"] = merged.apply(
        lambda row: classify_binary_state(row["label"], row["prediction_augmented"]),
        axis=1,
    )
    merged["transition_label"] = merged.apply(
        lambda row: assign_transition_label(
            row["label"], row["prediction_baseline"], row["prediction_augmented"]
        ),
        axis=1,
    )
    merged["probability_shift"] = (
        merged["probability_augmented"] - merged["probability_baseline"]
    )
    merged["dataset"] = dataset
    merged["horizon"] = horizon
    merged["comparison"] = comparison
    merged["seed"] = int(seed)
    merged["threshold_regime"] = threshold_regime
    merged = merged.merge(
        scalar_frame, on=["sequence_id", "split"], how="left", validate="one_to_one"
    )
    if merged[["primary_reliability", "conservative_reliability"]].isna().any().any():
        raise ValueError(
            f"Missing reliability scalar values after merge for {dataset} {horizon} {comparison}"
        )
    return merged
