from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.stats import beta
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


SEEDS = (42, 43, 44)


@dataclass(frozen=True)
class ParityResult:
    pearson_r: float
    max_abs_diff: float
    class_agreement: float
    abs_f1_diff: float
    abs_pr_auc_diff: float
    abs_roc_auc_diff: float
    exact_probability_match: bool
    metrics_pass_tolerance: bool
    parity_status: str


@dataclass(frozen=True)
class ExperimentPaths:
    repo: Path
    exp_dir: Path
    fig_dir: Path


def build_experiment_paths() -> ExperimentPaths:
    repo = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean")
    exp_dir = repo / "experiments/026_jaad_calibrated_selective_early_exit"
    return ExperimentPaths(repo=repo, exp_dir=exp_dir, fig_dir=exp_dir / "figures")


def _midpoints(scores: np.ndarray) -> np.ndarray:
    uniq = np.unique(np.asarray(scores, dtype=float))
    if uniq.size == 1:
        return uniq
    return (uniq[:-1] + uniq[1:]) / 2.0


def negative_exit_threshold_candidates(scores: np.ndarray) -> np.ndarray:
    uniq = np.unique(np.asarray(scores, dtype=float))
    mids = _midpoints(uniq)
    lower = np.nextafter(float(uniq.min()), float("-inf"))
    return np.unique(np.concatenate(([lower], uniq, mids, [1.0])))


def positive_exit_threshold_candidates(scores: np.ndarray) -> np.ndarray:
    uniq = np.unique(np.asarray(scores, dtype=float))
    mids = _midpoints(uniq)
    upper = np.nextafter(float(uniq.max()), float("inf"))
    return np.unique(np.concatenate(([0.0], mids, uniq, [upper])))


def clopper_pearson_upper_bound(errors: int, total: int, alpha: float = 0.05) -> float:
    if total <= 0:
        return float("nan")
    if errors >= total:
        return 1.0
    return float(beta.ppf(1.0 - alpha, errors + 1, total - errors))


def support_strength_label(total_exits: int) -> str:
    if total_exits >= 10:
        return "meaningful"
    if total_exits >= 5:
        return "exploratory"
    return "insufficient"


def more_conservative_threshold(a: float, b: float, exit_kind: str) -> float:
    if exit_kind == "positive":
        return max(a, b)
    if exit_kind == "negative":
        return min(a, b)
    raise ValueError(exit_kind)


def _calibration_inputs(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if scores.ndim != 1 or labels.ndim != 1 or len(scores) != len(labels) or len(scores) == 0:
        raise ValueError("calibration scores and labels must be non-empty one-dimensional arrays of equal length")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("calibration scores must be finite values in [0, 1]")
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("calibration labels must be binary")
    return scores, labels


def fit_platt_calibrator(scores: np.ndarray, labels: np.ndarray):
    """Fit sigmoid calibration on seed-mean OOF train scores only."""
    scores, labels = _calibration_inputs(scores, labels)
    if np.unique(labels).size != 2:
        raise ValueError("Platt calibration requires both label classes")
    model = LogisticRegression(random_state=12345, max_iter=2000)
    model.fit(scores.reshape(-1, 1), labels)
    return lambda x: model.predict_proba(np.asarray(x, dtype=float).reshape(-1, 1))[:, 1]


def fit_isotonic_calibrator(scores: np.ndarray, labels: np.ndarray):
    """Fit isotonic calibration on seed-mean OOF train scores only."""
    scores, labels = _calibration_inputs(scores, labels)
    if np.unique(labels).size != 2:
        raise ValueError("isotonic calibration requires both label classes")
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(scores, labels)
    return lambda x: model.predict(np.asarray(x, dtype=float))


def build_reliability_bins(
    labels: np.ndarray, scores: np.ndarray, n_bins: int = 15
) -> pd.DataFrame:
    """Build equal-width reliability bins, retaining empty bins for reporting."""
    scores, labels = _calibration_inputs(scores, labels)
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        count = int(mask.sum())
        rows.append({
            "bin_id": bin_id,
            "bin_lower": float(edges[bin_id]),
            "bin_upper": float(edges[bin_id + 1]),
            "bin_count": count,
            "mean_score": float(scores[mask].mean()) if count else float("nan"),
            "empirical_rate": float(labels[mask].mean()) if count else float("nan"),
        })
    return pd.DataFrame(rows)


def compute_calibration_metrics(
    labels: np.ndarray, scores: np.ndarray, n_bins: int = 15
) -> dict[str, float | int]:
    """Compute calibration metrics without treating one-class ROC-AUC as defined."""
    scores, labels = _calibration_inputs(scores, labels)
    bins = build_reliability_bins(labels, scores, n_bins=n_bins)
    populated = bins.loc[bins["bin_count"] > 0]
    gaps = np.abs(populated["empirical_rate"] - populated["mean_score"])
    weights = populated["bin_count"] / len(labels)
    return {
        "n_samples": int(len(labels)),
        "brier_score": float(np.mean((scores - labels) ** 2)),
        "ece": float((weights * gaps).sum()),
        "mce": float(gaps.max()) if len(gaps) else float("nan"),
        "roc_auc": float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else float("nan"),
    }


def authorize_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    exit_kind: str,
    risk_target: float,
    alpha: float = 0.05,
) -> dict[str, object]:
    """Authorize one validation threshold using a one-sided Clopper-Pearson bound."""
    scores, labels = _calibration_inputs(scores, labels)
    if not 0.0 <= threshold <= 1.0 or not 0.0 < risk_target <= 1.0 or not 0.0 < alpha < 1.0:
        raise ValueError("threshold, risk_target, and alpha must be within their valid probability ranges")
    if exit_kind == "negative":
        mask = scores <= threshold
        errors = int(((labels == 1) & mask).sum())
    elif exit_kind == "positive":
        mask = scores >= threshold
        errors = int(((labels == 0) & mask).sum())
    else:
        raise ValueError(exit_kind)
    n_exits = int(mask.sum())
    upper_bound = clopper_pearson_upper_bound(errors, n_exits, alpha) if n_exits else float("nan")
    support_strength = support_strength_label(n_exits)
    risk_authorized = bool(n_exits and upper_bound <= risk_target)
    return {
        "threshold": float(threshold),
        "exit_kind": exit_kind,
        "n_exits": n_exits,
        "n_errors": errors,
        "empirical_risk": float(errors / n_exits) if n_exits else float("nan"),
        "upper_bound": upper_bound,
        "support_strength": support_strength,
        "risk_authorized": risk_authorized,
        "claim_eligible": bool(risk_authorized and support_strength == "meaningful"),
        "coverage": float(mask.mean()),
    }


def select_authorized_threshold(
    candidate_rows: Iterable[dict[str, object]], exit_kind: str
) -> dict[str, object] | None:
    """Choose the highest-coverage authorized validation rule deterministically."""
    candidates = [row for row in candidate_rows if row["exit_kind"] == exit_kind and row["risk_authorized"]]
    if not candidates:
        return None
    if exit_kind == "negative":
        return min(candidates, key=lambda row: (-float(row["coverage"]), float(row["threshold"])))
    if exit_kind == "positive":
        return min(candidates, key=lambda row: (-float(row["coverage"]), -float(row["threshold"])))
    raise ValueError(exit_kind)


def _authorize_stage(
    scores: np.ndarray,
    labels: np.ndarray,
    negative_thresholds: Iterable[float],
    negative_risk_target: float,
    positive_thresholds: Iterable[float] = (),
    positive_risk_target: float = 1.0,
    allow_positive: bool = False,
    alpha: float = 0.05,
) -> dict[str, object]:
    scores, labels = _calibration_inputs(scores, labels)
    rows = [
        authorize_threshold(scores, labels, threshold, "negative", negative_risk_target, alpha)
        for threshold in negative_thresholds
    ]
    if allow_positive:
        rows.extend(
            authorize_threshold(scores, labels, threshold, "positive", positive_risk_target, alpha)
            for threshold in positive_thresholds
        )
    selected_negative = select_authorized_threshold(rows, "negative")
    selected_positive = select_authorized_threshold(rows, "positive") if allow_positive else None
    exit_mask = np.zeros(len(scores), dtype=bool)
    if selected_negative is not None:
        exit_mask |= scores <= selected_negative["threshold"]
    if selected_positive is not None:
        exit_mask |= scores >= selected_positive["threshold"]
    return {
        "candidate_rows": rows,
        "selected_negative": selected_negative,
        "selected_positive": selected_positive,
        "exit_mask": exit_mask,
    }


def authorize_stage1(
    scores: np.ndarray,
    labels: np.ndarray,
    negative_thresholds: Iterable[float],
    negative_risk_target: float,
    positive_thresholds: Iterable[float] = (),
    positive_risk_target: float = 1.0,
    allow_positive: bool = False,
    alpha: float = 0.05,
) -> dict[str, object]:
    """Authorize frozen Stage 1 exits from validation rows only."""
    return _authorize_stage(
        scores, labels, negative_thresholds, negative_risk_target, positive_thresholds,
        positive_risk_target, allow_positive, alpha,
    )


def authorize_stage2_on_survivors(
    stage1: dict[str, object],
    scores: np.ndarray,
    labels: np.ndarray,
    negative_thresholds: Iterable[float],
    negative_risk_target: float,
    positive_thresholds: Iterable[float] = (),
    positive_risk_target: float = 1.0,
    allow_positive: bool = False,
    alpha: float = 0.05,
) -> dict[str, object]:
    """Authorize Stage 2 only on validation rows not exited by frozen Stage 1."""
    scores, labels = _calibration_inputs(scores, labels)
    stage1_exit_mask = np.asarray(stage1.get("exit_mask"), dtype=bool)
    if stage1_exit_mask.ndim != 1 or len(stage1_exit_mask) != len(scores):
        raise ValueError("stage1 exit mask must align exactly with Stage 2 validation rows")
    survivor_mask = ~stage1_exit_mask
    if not survivor_mask.any():
        return {
            "candidate_rows": [],
            "selected_negative": None,
            "selected_positive": None,
            "exit_mask": np.zeros(0, dtype=bool),
            "survivor_mask": survivor_mask,
        }
    result = _authorize_stage(
        scores[survivor_mask], labels[survivor_mask], negative_thresholds, negative_risk_target,
        positive_thresholds, positive_risk_target, allow_positive, alpha,
    )
    result["survivor_mask"] = survivor_mask
    return result


def load_selected_bbox_specs() -> pd.DataFrame:
    """Load the validation-selected canonical BBox models used by exp-024."""
    paths = build_experiment_paths()
    selection = pd.read_csv(
        paths.repo / "experiments/024_jaad_adaptive_early_decision/model_selection.csv"
    )
    rows = selection[selection["method"].isin((
        "single_bbox_30_60",
        "single_bbox_15_30_corrected",
        "single_bbox_5_15",
    ))].copy()
    if len(rows) != 3:
        raise ValueError("exp-024 does not contain all three canonical BBox selections")

    rows["horizon_label"] = rows["method"].str.removeprefix("single_bbox_")
    rows = rows.rename(columns={"selected_model": "model_name"})
    return rows[["horizon_label", "model_name", "threshold_rule", "threshold", "source"]].sort_values(
        "horizon_label"
    ).reset_index(drop=True)


def build_horizon_feature_frames() -> dict[str, pd.DataFrame]:
    """Return canonical BBox frames keyed by the three exp-026 horizons.

    The 15_30 corrected frame is reconstructed from exp-021's corrected BBox
    table; the two uncorrected horizons reuse the canonical benchmark loader.
    """
    import sys

    scripts_dir = build_experiment_paths().repo / "scripts/experiments"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import jaad_biu_v2_benchmark_comparison as base

    split_rows = {horizon: base._load_split_rows(horizon) for horizon in ("30_60", "5_15")}
    placeholder_v2_rows = {
        horizon: rows[["sequence_id"]].copy() for horizon, rows in split_rows.items()
    }
    feature_dict = pd.read_csv(base.DICT_PATH)
    feature_sets, _, _ = base._build_feature_sets(split_rows, placeholder_v2_rows, feature_dict)

    frames = {}
    for horizon in ("30_60", "5_15"):
        frame = feature_sets[horizon]["bbox_only"].copy()
        frame["track_key"] = frame["video_id"].astype(str) + ":" + frame["ped_id"].astype(str)
        frame.attrs["canonical_bbox_columns"] = tuple(feature_sets[horizon]["_column_map"]["bbox_only"])
        frames[horizon] = frame

    import jaad_corrected_15_30_rebuild_benchmark as corrected_source

    corrected_base_rows = corrected_source.load_base_rows()
    corrected_feature_sets, corrected_columns = corrected_source.build_feature_sets(corrected_base_rows)
    corrected = corrected_feature_sets["bbox_only"].copy()
    corrected["track_key"] = corrected["video_id"].astype(str) + ":" + corrected["ped_id"].astype(str)
    corrected.attrs["canonical_bbox_columns"] = tuple(corrected_columns["bbox_only"])
    frames["15_30_corrected"] = corrected
    return frames


def canonical_bbox_feature_columns(
    frame: pd.DataFrame, requested_columns: Iterable[str] | None = None
) -> list[str]:
    """Return the exact BBox column set emitted by the canonical builder."""
    columns = frame.attrs.get("canonical_bbox_columns")
    if columns is None:
        raise ValueError("frame is missing canonical BBox columns from the feature builder")
    canonical = list(columns)
    if requested_columns is not None and list(requested_columns) != canonical:
        raise ValueError("requested feature columns differ from canonical BBox columns")
    if set(canonical).difference(frame.columns):
        raise ValueError("canonical BBox columns are missing from the feature frame")
    return canonical


def _valid_group_disjoint_folds(
    df: pd.DataFrame, folds: list[np.ndarray], group_column: str
) -> bool:
    positions = np.arange(len(df), dtype=int)
    if not folds or any(idx.dtype.kind not in "iu" or idx.size == 0 for idx in folds):
        return False
    combined = np.concatenate(folds)
    if not np.array_equal(np.sort(combined), positions) or len(np.unique(combined)) != len(positions):
        return False
    held_out_groups: set[object] = set()
    for idx in folds:
        test = df.iloc[idx]
        train = df.iloc[np.setdiff1d(positions, idx, assume_unique=True)]
        if test["label"].nunique() != 2 or train["label"].nunique() != 2:
            return False
        groups = set(test[group_column])
        if held_out_groups.intersection(groups):
            return False
        held_out_groups.update(groups)
    return True


def _solve_valid_group_assignment(
    df: pd.DataFrame, n_folds: int, group_column: str
) -> list[np.ndarray] | None:
    """Prove whether a valid grouped assignment exists using binary feasibility."""
    grouped = list(df.reset_index(drop=True).groupby(group_column, sort=True))
    n_groups = len(grouped)
    positives = np.asarray([int(group["label"].sum()) for _, group in grouped], dtype=float)
    negatives = np.asarray([int((group["label"] == 0).sum()) for _, group in grouped], dtype=float)
    n_variables = n_folds * n_groups

    assignment = np.zeros((n_groups, n_variables), dtype=float)
    for group_id in range(n_groups):
        assignment[group_id, group_id::n_groups] = 1.0
    class_support = np.zeros((2 * n_folds, n_variables), dtype=float)
    for fold_id in range(n_folds):
        start = fold_id * n_groups
        class_support[2 * fold_id, start:start + n_groups] = positives
        class_support[2 * fold_id + 1, start:start + n_groups] = negatives
    constraints = [
        LinearConstraint(assignment, lb=np.ones(n_groups), ub=np.ones(n_groups)),
        LinearConstraint(class_support, lb=np.ones(2 * n_folds), ub=np.full(2 * n_folds, np.inf)),
    ]
    result = milp(
        c=np.zeros(n_variables),
        integrality=np.ones(n_variables),
        bounds=Bounds(0, 1),
        constraints=constraints,
        options={"disp": False},
    )
    if result.status == 2:
        return None
    if not result.success or result.x is None:
        raise RuntimeError(f"grouped-fold feasibility solver failed: {result.message}")

    assignments = np.rint(result.x).astype(int).reshape(n_folds, n_groups)
    folds = []
    for fold_id in range(n_folds):
        positions = [
            group.index.to_numpy(dtype=int)
            for group_id, (_, group) in enumerate(grouped)
            if assignments[fold_id, group_id]
        ]
        folds.append(np.sort(np.concatenate(positions)).astype(int))
    if not _valid_group_disjoint_folds(df, folds, group_column):
        raise RuntimeError("feasibility solver returned an invalid grouped fold assignment")
    return folds


def _make_group_disjoint_folds(
    df: pd.DataFrame, group_column: str, preferred_folds: int, min_folds: int
) -> list[np.ndarray]:
    required = {group_column, "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing required fold columns: {sorted(missing)}")
    if min_folds < 2 or preferred_folds < min_folds:
        raise ValueError("preferred_folds must be >= min_folds >= 2")
    if df["label"].nunique() != 2:
        raise ValueError("grouped OOF requires both label classes")

    labels = df["label"].to_numpy(dtype=int)
    n_groups = df[group_column].nunique()
    for n_folds in range(min(preferred_folds, n_groups), min_folds - 1, -1):
        if np.min(np.bincount(labels)) < n_folds:
            continue
        folds = _solve_valid_group_assignment(df, n_folds, group_column)
        if folds is not None:
            return folds
    raise ValueError(f"no valid {group_column}-disjoint split with at least {min_folds} folds exists")


def make_video_disjoint_folds(
    df: pd.DataFrame, preferred_folds: int = 5, min_folds: int = 3
) -> list[np.ndarray]:
    """Create a validated video-disjoint, class-supported OOF fold scheme."""
    return _make_group_disjoint_folds(df, "video_id", preferred_folds, min_folds)


def make_track_grouped_folds(
    df: pd.DataFrame, preferred_folds: int = 5, min_folds: int = 3
) -> list[np.ndarray]:
    """Create the approved track-grouped fallback OOF fold scheme."""
    return _make_group_disjoint_folds(df, "track_key", preferred_folds, min_folds)


def make_grouped_oof_folds(
    df: pd.DataFrame, preferred_folds: int = 5, min_folds: int = 3
) -> tuple[str, list[np.ndarray]]:
    """Prefer video-disjoint OOF and fall back to valid track grouping only."""
    try:
        return "video_disjoint", make_video_disjoint_folds(df, preferred_folds, min_folds)
    except ValueError as exc:
        if not str(exc).startswith("no valid video_id-disjoint split"):
            raise
        return "track_grouped", make_track_grouped_folds(df, preferred_folds, min_folds)


def describe_fold_scheme(df: pd.DataFrame, folds: Iterable[np.ndarray]) -> pd.DataFrame:
    """Summarize a validated fold scheme for the experiment provenance table."""
    rows = []
    for fold_id, indices in enumerate(folds):
        held_out = df.iloc[np.asarray(indices, dtype=int)]
        rows.append({
            "fold_id": fold_id,
            "n_rows": len(held_out),
            "n_videos": held_out["video_id"].nunique(),
            "n_positive": int(held_out["label"].sum()),
            "n_negative": int((held_out["label"] == 0).sum()),
        })
    return pd.DataFrame(rows)


def make_grouped_oof_predictions(
    frame: pd.DataFrame,
    feature_columns: Iterable[str] | None,
    model_name: str,
    seeds: Iterable[int] = SEEDS,
    folds: Iterable[np.ndarray] | None = None,
) -> pd.DataFrame:
    """Fit seed-specific OOF models on canonical train rows only."""
    import sys

    if "split" not in frame or not frame["split"].eq("train").all():
        raise ValueError("grouped OOF requires canonical train rows only")

    scripts_dir = build_experiment_paths().repo / "scripts/experiments"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import jaad_biu_v2_benchmark_comparison as base

    features = canonical_bbox_feature_columns(frame, feature_columns)
    fold_scheme, approved_folds = make_grouped_oof_folds(frame)
    fold_list = [np.asarray(indices, dtype=int) for indices in approved_folds]
    if folds is not None:
        provided_folds = [np.asarray(indices, dtype=int) for indices in folds]
        if len(provided_folds) != len(fold_list) or any(
            not np.array_equal(provided, approved)
            for provided, approved in zip(provided_folds, fold_list)
        ):
            raise ValueError("caller-provided folds do not match the approved grouped OOF scheme")

    y = frame["label"].to_numpy(dtype=int)
    result = frame[["sequence_id", "video_id", "track_key", "label", "split"]].copy()
    result["fold_scheme"] = fold_scheme
    for seed in seeds:
        scores = np.full(len(frame), np.nan, dtype=float)
        for test_idx in fold_list:
            test_idx = np.asarray(test_idx, dtype=int)
            train_idx = np.setdiff1d(np.arange(len(frame)), test_idx, assume_unique=True)
            model = base._build_model(model_name, int(seed), y[train_idx])
            model.fit(frame.iloc[train_idx][features], y[train_idx])
            scores[test_idx] = model.predict_proba(frame.iloc[test_idx][features])[:, 1]
        if np.isnan(scores).any():
            raise RuntimeError("OOF generation did not cover every row exactly once")
        result[f"score_seed{int(seed)}"] = scores
    seed_columns = [col for col in result if col.startswith("score_seed")]
    result["score_seed_mean"] = result[seed_columns].mean(axis=1)
    return result


def align_saved_scores_by_key(frame: pd.DataFrame, saved_scores: pd.DataFrame) -> np.ndarray:
    """Align saved ensemble probabilities to reconstruction rows by sequence ID."""
    if "sequence_id" not in frame or "sequence_id" not in saved_scores:
        raise ValueError("saved-score alignment requires sequence_id in both inputs")
    if frame["sequence_id"].duplicated().any():
        raise ValueError("reconstruction frame has duplicate sequence IDs")
    if saved_scores["sequence_id"].duplicated().any():
        raise ValueError("saved scores have duplicate sequence IDs")
    score_columns = [column for column in ("prob", "score_seed_mean") if column in saved_scores]
    if len(score_columns) != 1:
        raise ValueError("saved scores must contain exactly one of prob or score_seed_mean")
    saved = saved_scores[["sequence_id", score_columns[0]]].copy()
    frame_keys = frame[["sequence_id"]].assign(_frame_present=True)
    coverage = saved.merge(frame_keys, on="sequence_id", how="left", validate="one_to_one")
    if len(coverage) != len(saved) or coverage["_frame_present"].isna().any():
        raise ValueError("saved-score sequence IDs include unmatched reconstruction rows")
    for column in ("label", "split"):
        if column in frame and column in saved_scores:
            expected = frame.set_index("sequence_id")[column].reindex(saved_scores["sequence_id"])
            actual = saved_scores.set_index("sequence_id")[column]
            if not actual.eq(expected).all():
                raise ValueError(f"saved-score {column} values do not match reconstruction rows")
    frame_subset = frame.loc[frame["sequence_id"].isin(saved["sequence_id"]), ["sequence_id"]]
    merged = frame_subset.merge(saved, on="sequence_id", how="left", validate="one_to_one")
    return merged[score_columns[0]].to_numpy(dtype=float)


def parity_rows_for_saved_scores(frame: pd.DataFrame, saved_scores: pd.DataFrame) -> pd.DataFrame:
    """Select and validate the saved val/test rows used for parity evaluation."""
    if "split" not in frame or "split" not in saved_scores:
        raise ValueError("parity alignment requires split metadata")
    expected = frame.loc[frame["split"].isin(("val", "test"))].copy()
    if expected.empty:
        raise ValueError("reconstruction frame has no val/test rows for parity")
    saved_keys = set(saved_scores["sequence_id"])
    expected_keys = set(expected["sequence_id"])
    if saved_keys != expected_keys:
        raise ValueError("saved parity scores must cover the exact val/test key set")
    if not saved_scores["split"].isin(("val", "test")).all():
        raise ValueError("saved parity scores must contain val/test rows only")
    aligned_scores = align_saved_scores_by_key(expected, saved_scores)
    rows = expected.copy()
    rows["saved_score"] = aligned_scores
    return rows


def reconstruct_seed_mean_scores(
    frame: pd.DataFrame,
    feature_columns: Iterable[str] | None,
    model_name: str,
    saved_scores: pd.DataFrame,
    threshold: float,
    seeds: Iterable[int] = SEEDS,
) -> tuple[pd.DataFrame, ParityResult]:
    """Recreate saved train-split model scores and reject failed parity.

    This intentionally differs from OOF generation: saved exp-011/021 scores
    came from models fitted on the canonical ``train`` partition.  Reproducing
    those scores first prevents an incompatible feature/model pipeline from
    being used to create calibration OOF predictions.
    """
    import sys

    scripts_dir = build_experiment_paths().repo / "scripts/experiments"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import jaad_biu_v2_benchmark_comparison as base

    if "split" not in frame:
        raise ValueError("parity reconstruction requires the canonical split column")
    features = canonical_bbox_feature_columns(frame, feature_columns)
    train_mask = frame["split"].to_numpy() == "train"
    if not train_mask.any() or np.unique(frame.loc[train_mask, "label"]).size != 2:
        raise ValueError("canonical train partition must contain both classes")

    y_train = frame.loc[train_mask, "label"].to_numpy(dtype=int)
    reproduced = frame[["sequence_id", "video_id", "label", "split"]].copy()
    for seed in seeds:
        model = base._build_model(model_name, int(seed), y_train)
        model.fit(frame.loc[train_mask, features], y_train)
        reproduced[f"score_seed{int(seed)}"] = model.predict_proba(frame[features])[:, 1]
    score_columns = [column for column in reproduced if column.startswith("score_seed")]
    reproduced["score_seed_mean"] = reproduced[score_columns].mean(axis=1)
    parity_frame = frame
    expected_keys = set(frame.loc[frame["split"].isin(("val", "test")), "sequence_id"])
    saved_keys = set(saved_scores["sequence_id"])
    if saved_keys != expected_keys:
        parity_frame = canonicalize_track_rows(frame, str(frame.get("horizon_label", pd.Series(["unknown"])).iloc[0]))
    parity_rows = parity_rows_for_saved_scores(parity_frame, saved_scores)
    reproduced_by_key = reproduced.set_index("sequence_id")["score_seed_mean"]
    reproduced_scores = reproduced_by_key.reindex(parity_rows["sequence_id"]).to_numpy(dtype=float)
    if np.isnan(reproduced_scores).any():
        raise ValueError("reconstructed scores do not cover saved parity keys")
    parity = evaluate_parity(
        parity_rows["label"].to_numpy(dtype=int), parity_rows["saved_score"].to_numpy(dtype=float), reproduced_scores, threshold
    )
    if parity.parity_status == "parity_failed":
        raise ValueError("reconstruction failed the approved parity tolerances")
    return reproduced, parity


def evaluate_parity(
    y_true: np.ndarray, saved_scores: np.ndarray, reproduced_scores: np.ndarray, threshold: float
) -> ParityResult:
    """Compare saved and reproduced scores against the true labels only."""
    y_true = np.asarray(y_true, dtype=int)
    saved = np.asarray(saved_scores, dtype=float)
    reproduced = np.asarray(reproduced_scores, dtype=float)
    if not (len(y_true) == len(saved) == len(reproduced)) or len(y_true) == 0:
        raise ValueError("parity inputs must be non-empty and have equal length")

    saved_pred = (saved >= threshold).astype(int)
    reproduced_pred = (reproduced >= threshold).astype(int)
    pearson_r = float(np.corrcoef(saved, reproduced)[0, 1])
    max_abs_diff = float(np.max(np.abs(saved - reproduced)))
    class_agreement = float((saved_pred == reproduced_pred).mean())
    abs_f1_diff = float(abs(f1_score(y_true, saved_pred, zero_division=0) - f1_score(y_true, reproduced_pred, zero_division=0)))
    if np.unique(y_true).size == 2:
        abs_pr_auc_diff = float(abs(average_precision_score(y_true, saved) - average_precision_score(y_true, reproduced)))
        abs_roc_auc_diff = float(abs(roc_auc_score(y_true, saved) - roc_auc_score(y_true, reproduced)))
    else:
        abs_pr_auc_diff = float("nan")
        abs_roc_auc_diff = float("nan")
    exact_probability_match = bool(pearson_r >= 0.999 and max_abs_diff <= 1e-4 and class_agreement >= 0.995)
    metrics_pass_tolerance = bool(
        abs_f1_diff <= 0.005 and abs_pr_auc_diff <= 0.005 and abs_roc_auc_diff <= 0.005
    )
    parity_status = (
        "exact_match" if exact_probability_match else
        "pipeline_matched_reconstruction" if metrics_pass_tolerance else
        "parity_failed"
    )
    return ParityResult(
        pearson_r=pearson_r,
        max_abs_diff=max_abs_diff,
        class_agreement=class_agreement,
        abs_f1_diff=abs_f1_diff,
        abs_pr_auc_diff=abs_pr_auc_diff,
        abs_roc_auc_diff=abs_roc_auc_diff,
        exact_probability_match=exact_probability_match,
        metrics_pass_tolerance=metrics_pass_tolerance,
        parity_status=parity_status,
    )


def make_canonical_grouped_oof_predictions(
    full_frame: pd.DataFrame,
    canonical_train: pd.DataFrame,
    feature_columns: Iterable[str] | None,
    model_name: str,
    seeds: Iterable[int] = SEEDS,
) -> tuple[pd.DataFrame, str, list[np.ndarray]]:
    """Score canonical train rows honestly while training on the full-window train pool."""
    import sys

    if not full_frame["split"].eq("train").all():
        raise ValueError("full_frame must contain train rows only for canonical OOF")
    if not canonical_train["split"].eq("train").all():
        raise ValueError("canonical_train must contain train rows only for canonical OOF")

    scripts_dir = build_experiment_paths().repo / "scripts/experiments"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import jaad_biu_v2_benchmark_comparison as base

    features = canonical_bbox_feature_columns(full_frame, feature_columns)
    fold_scheme, folds = make_grouped_oof_folds(canonical_train)
    y_full = full_frame["label"].to_numpy(dtype=int)
    result = canonical_train[["sequence_id", "video_id", "track_key", "label", "split"]].copy()
    result["fold_scheme"] = fold_scheme
    for seed in seeds:
        scores = np.full(len(canonical_train), np.nan, dtype=float)
        for fold_idx in folds:
            held_out = canonical_train.iloc[np.asarray(fold_idx, dtype=int)]
            if fold_scheme == "video_disjoint":
                blocked_groups = set(held_out["video_id"])
                train_mask = ~full_frame["video_id"].isin(blocked_groups)
            elif fold_scheme == "track_grouped":
                blocked_groups = set(held_out["track_key"])
                train_mask = ~full_frame["track_key"].isin(blocked_groups)
            else:
                raise ValueError(fold_scheme)
            model = base._build_model(model_name, int(seed), y_full[train_mask.to_numpy()])
            model.fit(full_frame.loc[train_mask, features], y_full[train_mask.to_numpy()])
            scores[np.asarray(fold_idx, dtype=int)] = model.predict_proba(held_out[features])[:, 1]
        if np.isnan(scores).any():
            raise RuntimeError("canonical OOF generation did not cover every held-out row exactly once")
        result[f"score_seed{int(seed)}"] = scores
    score_columns = [column for column in result if column.startswith("score_seed")]
    result["score_seed_mean"] = result[score_columns].mean(axis=1)
    return result, fold_scheme, folds


def canonicalize_track_rows(frame: pd.DataFrame, horizon_label: str) -> pd.DataFrame:
    """Apply the canonical staged-track rule: min TTE, then latest start frame."""
    required = {"track_key", "split", "sequence_id", "label", "tte_frames", "start_frame"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"frame is missing canonicalization columns: {sorted(missing)}")
    canonical = (
        frame.sort_values(["track_key", "split", "tte_frames", "start_frame"], ascending=[True, True, True, False])
        .groupby(["track_key", "split"], as_index=False)
        .first()
        .copy()
    )
    canonical["horizon_label"] = horizon_label
    canonical.attrs.update(frame.attrs)
    return canonical


def load_saved_track_rows(horizon_label: str, model_name: str) -> pd.DataFrame:
    """Load the canonical saved staged-track rows for one BBox horizon."""
    paths = build_experiment_paths()
    if horizon_label == "15_30_corrected":
        path = paths.repo / "experiments/021_jaad_corrected_15_30_rebuild/metrics/corrected_staged_track_table.csv"
    else:
        path = paths.repo / "experiments/011_jaad_staged_bbox_biu_v2/metrics/staged_track_table.csv"
    table = pd.read_csv(path)
    rows = table[
        (table["horizon_label"] == horizon_label)
        & (table["feature_set"] == "bbox_only")
        & (table["model_name"] == model_name)
    ].copy()
    if rows.empty:
        raise ValueError(f"no saved track rows found for {horizon_label}/{model_name}")
    return rows


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_saved_common_track_frame(model_name: str = "logreg") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match the exp-024 common-track construction using saved canonical rows."""
    bbox30 = load_saved_track_rows("30_60", model_name)
    bbox15 = load_saved_track_rows("15_30_corrected", model_name)
    bbox5 = load_saved_track_rows("5_15", model_name)

    def attach(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        cols = {
            "sequence_id": f"{prefix}_sequence_id",
            "prob": f"{prefix}_saved_raw_score",
            "tte_frames": f"{prefix}_tte",
            "start_frame": f"{prefix}_start_frame",
        }
        return df.rename(columns=cols)[["track_key", "split", "label", "group", *cols.values()]].copy()

    df30 = attach(bbox30, "bbox30")
    df15 = attach(bbox15, "bbox15")
    df5 = attach(bbox5, "bbox5")
    merged = df30.merge(df15, on=["track_key", "split", "label", "group"], how="outer", indicator="m15")
    merged = merged.merge(df5, on=["track_key", "split", "label", "group"], how="outer", indicator="m5")
    common_mask = (
        merged["bbox30_sequence_id"].notna()
        & merged["bbox15_sequence_id"].notna()
        & merged["bbox5_sequence_id"].notna()
    )
    merged["availability"] = np.where(common_mask, "common_all_horizons", "partial_or_missing")
    common = merged.loc[common_mask].copy().sort_values(["split", "track_key"]).reset_index(drop=True)
    missing = merged.loc[~common_mask].copy().sort_values(["split", "track_key"]).reset_index(drop=True)
    return common, missing


def attach_score_variants(
    common: pd.DataFrame,
    horizon_prefix: str,
    score_table: pd.DataFrame,
) -> pd.DataFrame:
    required = {"sequence_id", "raw_score", "platt_score", "isotonic_score"}
    missing = required.difference(score_table.columns)
    if missing:
        raise ValueError(f"score table is missing score-variant columns: {sorted(missing)}")
    lookup = score_table[list(required)].rename(
        columns={
            "sequence_id": f"{horizon_prefix}_sequence_id",
            "raw_score": f"{horizon_prefix}_raw_score",
            "platt_score": f"{horizon_prefix}_platt_score",
            "isotonic_score": f"{horizon_prefix}_isotonic_score",
        }
    )
    merged = common.merge(lookup, on=f"{horizon_prefix}_sequence_id", how="left", validate="one_to_one")
    score_cols = [col for col in merged.columns if col.startswith(f"{horizon_prefix}_") and col.endswith("_score")]
    if merged[score_cols].isna().any().any():
        raise ValueError(f"{horizon_prefix} calibrated scores do not fully cover the common track frame")
    return merged


def _risk_table_threshold_candidates(scores: np.ndarray, exit_kind: str) -> np.ndarray:
    uniq = np.unique(np.asarray(scores, dtype=float))
    mids = _midpoints(uniq)
    if exit_kind == "negative":
        return np.unique(np.clip(np.concatenate(([0.0], mids, uniq, [1.0])), 0.0, 1.0))
    if exit_kind == "positive":
        return np.unique(np.clip(np.concatenate(([0.0], mids, uniq, [1.0])), 0.0, 1.0))
    raise ValueError(exit_kind)


def _precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    positive = y_pred == 1
    if not positive.any():
        return float("nan")
    return float(y_true[positive].mean())


def _npv(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    negative = y_pred == 0
    if not negative.any():
        return float("nan")
    return float((y_true[negative] == 0).mean())


def _fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    negative = y_true == 0
    if not negative.any():
        return float("nan")
    return float(((y_pred == 1) & negative).sum() / negative.sum())


def evaluate_threshold_policy(
    df: pd.DataFrame,
    score_kind: str,
    family: str,
    threshold: float,
    final_thresholds: dict[str, dict[str, float]],
) -> dict[str, object]:
    y_true = df["label"].to_numpy(dtype=int)
    groups = df["group"].to_numpy(dtype=object)
    p30 = df[f"bbox30_{score_kind}_score"].to_numpy(dtype=float)
    p15 = df[f"bbox15_{score_kind}_score"].to_numpy(dtype=float)
    p5 = df[f"bbox5_{score_kind}_score"].to_numpy(dtype=float)
    t5 = float(final_thresholds[score_kind]["5_15"])

    if family == "30_60_negative":
        exit_kind = "negative"
        exit_mask = p30 <= threshold
        final_pred = np.where(exit_mask, 0, (p5 >= t5).astype(int)).astype(int)
        exit_stage = np.where(exit_mask, "30_60", "5_15")
        decision_tte = np.where(exit_mask, df["bbox30_tte"], df["bbox5_tte"]).astype(float)
        stages_evaluated = np.where(exit_mask, 1, 3).astype(float)
    elif family == "15_30_negative":
        exit_kind = "negative"
        exit_mask = p15 <= threshold
        final_pred = np.where(exit_mask, 0, (p5 >= t5).astype(int)).astype(int)
        exit_stage = np.where(exit_mask, "15_30_corrected", "5_15")
        decision_tte = np.where(exit_mask, df["bbox15_tte"], df["bbox5_tte"]).astype(float)
        stages_evaluated = np.where(exit_mask, 2, 3).astype(float)
    elif family == "15_30_positive":
        exit_kind = "positive"
        exit_mask = p15 >= threshold
        final_pred = np.where(exit_mask, 1, (p5 >= t5).astype(int)).astype(int)
        exit_stage = np.where(exit_mask, "15_30_corrected", "5_15")
        decision_tte = np.where(exit_mask, df["bbox15_tte"], df["bbox5_tte"]).astype(float)
        stages_evaluated = np.where(exit_mask, 2, 3).astype(float)
    elif family == "5_15_final":
        exit_kind = "final"
        exit_mask = np.ones(len(df), dtype=bool)
        final_pred = (p5 >= threshold).astype(int)
        exit_stage = np.full(len(df), "5_15", dtype=object)
        decision_tte = df["bbox5_tte"].to_numpy(dtype=float)
        stages_evaluated = np.full(len(df), 3.0)
    else:
        raise ValueError(family)

    n_exits = int(exit_mask.sum())
    if exit_kind == "negative":
        n_errors = int(((y_true == 1) & exit_mask).sum())
        empirical_error = float(n_errors / n_exits) if n_exits else float("nan")
        pos_precision = float("nan")
        neg_npv = float((y_true[exit_mask] == 0).mean()) if n_exits else float("nan")
        cp_upper = clopper_pearson_upper_bound(n_errors, n_exits) if n_exits else float("nan")
    elif exit_kind == "positive":
        n_errors = int(((y_true == 0) & exit_mask).sum())
        empirical_error = float(n_errors / n_exits) if n_exits else float("nan")
        pos_precision = float((y_true[exit_mask] == 1).mean()) if n_exits else float("nan")
        neg_npv = float("nan")
        cp_upper = clopper_pearson_upper_bound(n_errors, n_exits) if n_exits else float("nan")
    else:
        n_errors = int((final_pred != y_true).sum())
        empirical_error = float(n_errors / n_exits) if n_exits else float("nan")
        pos_precision = _precision(y_true, final_pred)
        neg_npv = _npv(y_true, final_pred)
        cp_upper = clopper_pearson_upper_bound(n_errors, n_exits) if n_exits else float("nan")

    ambiguous_mask = groups == "ambiguous_negative"
    clean_mask = groups == "clean_negative"
    return {
        "policy_family": family,
        "score_kind": score_kind,
        "threshold": float(threshold),
        "final_5_15_threshold": t5,
        "exit_kind": exit_kind,
        "support_count": n_exits,
        "coverage": float(exit_mask.mean()),
        "empirical_error_rate": empirical_error,
        "cp_upper_95": cp_upper,
        "positive_exit_precision": pos_precision,
        "negative_exit_npv": neg_npv,
        "overall_f1": float(f1_score(y_true, final_pred, zero_division=0)),
        "overall_fpr": _fpr(y_true, final_pred),
        "ambiguous_negative_fpr": _fpr(y_true[ambiguous_mask], final_pred[ambiguous_mask]) if ambiguous_mask.any() else float("nan"),
        "clean_negative_fpr": _fpr(y_true[clean_mask], final_pred[clean_mask]) if clean_mask.any() else float("nan"),
        "mean_decision_tte": float(decision_tte.mean()),
        "average_stages_evaluated": float(stages_evaluated.mean()),
    }


def build_risk_coverage_tables(
    common_scores: pd.DataFrame,
    final_thresholds: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows = []
    for score_kind in ("raw", "platt", "isotonic"):
        val = common_scores[common_scores["split"] == "val"].copy()
        test = common_scores[common_scores["split"] == "test"].copy()
        family_specs = [
            ("30_60_negative", "bbox30", "negative"),
            ("15_30_negative", "bbox15", "negative"),
            ("15_30_positive", "bbox15", "positive"),
        ]
        for family, prefix, exit_kind in family_specs:
            candidates = _risk_table_threshold_candidates(val[f"{prefix}_{score_kind}_score"].to_numpy(dtype=float), exit_kind)
            for threshold in candidates:
                for split_name, split_df in (("val", val), ("test", test)):
                    row = evaluate_threshold_policy(split_df, score_kind, family, float(threshold), final_thresholds)
                    row["evaluation_split"] = split_name
                    row["threshold_origin"] = "validation_candidate_score"
                    rows.append(row)
        final_threshold = float(final_thresholds[score_kind]["5_15"])
        for split_name, split_df in (("val", val), ("test", test)):
            row = evaluate_threshold_policy(split_df, score_kind, "5_15_final", final_threshold, final_thresholds)
            row["evaluation_split"] = split_name
            row["threshold_origin"] = "fixed_final_decision_rule"
            rows.append(row)
    table = pd.DataFrame(rows)
    order = [
        "score_kind",
        "policy_family",
        "threshold_origin",
        "evaluation_split",
        "threshold",
        "final_5_15_threshold",
        "exit_kind",
        "support_count",
        "coverage",
        "empirical_error_rate",
        "cp_upper_95",
        "positive_exit_precision",
        "negative_exit_npv",
        "overall_f1",
        "overall_fpr",
        "ambiguous_negative_fpr",
        "clean_negative_fpr",
        "mean_decision_tte",
        "average_stages_evaluated",
    ]
    return table[order].sort_values(
        ["score_kind", "policy_family", "evaluation_split", "threshold"], kind="stable"
    ).reset_index(drop=True)


def plot_reliability_figure(horizon_label: str, bins: pd.DataFrame, output_path: Path) -> None:
    split_order = ["oof_train", "val", "test"]
    method_order = ["raw", "platt", "isotonic"]
    colors = {"raw": "#8c564b", "platt": "#1f77b4", "isotonic": "#2ca02c"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, split_name in zip(axes, split_order):
        ax.plot([0.0, 1.0], [0.0, 1.0], "--", color="#b0b0b0", linewidth=1)
        for method in method_order:
            sub = bins[
                (bins["horizon_label"] == horizon_label)
                & (bins["split_name"] == split_name)
                & (bins["score_kind"] == method)
                & (bins["bin_count"] > 0)
            ]
            ax.plot(
                sub["mean_score"],
                sub["empirical_rate"],
                marker="o",
                linewidth=1.5,
                markersize=4,
                color=colors[method],
                label=method,
            )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(split_name.replace("_", " "))
        ax.set_xlabel("mean score")
        ax.set_ylabel("empirical positive rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"JAAD {horizon_label} reliability", y=1.05)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def determine_calibration_verdict(calibration_metrics: pd.DataFrame) -> str:
    eval_rows = calibration_metrics[calibration_metrics["split_name"].isin(["val", "test"])].copy()

    def generalizes(score_kind: str) -> bool:
        subset = eval_rows[eval_rows["score_kind"] == score_kind]
        if subset.empty:
            return False
        stable = (subset["ece_delta_vs_raw"] <= 1e-9) | (subset["brier_delta_vs_raw"] <= 1e-9)
        improved = (subset["ece_delta_vs_raw"] < -1e-6) | (subset["brier_delta_vs_raw"] < -1e-6)
        return bool(stable.all() and improved.any())

    platt_generalizes = generalizes("platt")
    isotonic_generalizes = generalizes("isotonic")
    oof_rows = calibration_metrics[calibration_metrics["split_name"] == "oof_train"]
    train_only_pattern = False
    for score_kind in ("platt", "isotonic"):
        train_subset = oof_rows[oof_rows["score_kind"] == score_kind]
        eval_subset = eval_rows[eval_rows["score_kind"] == score_kind]
        if not train_subset.empty and not eval_subset.empty:
            train_improves = ((train_subset["ece_delta_vs_raw"] < -1e-6) | (train_subset["brier_delta_vs_raw"] < -1e-6)).any()
            eval_worsens = ((eval_subset["ece_delta_vs_raw"] > 1e-6) | (eval_subset["brier_delta_vs_raw"] > 1e-6)).any()
            train_only_pattern = train_only_pattern or bool(train_improves and eval_worsens)

    if platt_generalizes:
        return "Platt calibration improves JAAD confidence reliably."
    if isotonic_generalizes and not platt_generalizes:
        return "Isotonic improves calibration but is unstable due to sample size."
    if train_only_pattern:
        return "Calibration improvement does not generalize beyond OOF training."
    return "JAAD confidence remains too unstable for confidence-driven early exits."


def build_final_thresholds(specs: pd.DataFrame) -> dict[str, dict[str, float]]:
    spec_map = {
        row["horizon_label"]: float(row["threshold"])
        for row in specs.to_dict("records")
    }
    return {
        "raw": {
            "30_60": spec_map["30_60"],
            "15_30_corrected": spec_map["15_30_corrected"],
            "5_15": spec_map["5_15"],
        },
        "platt": {"30_60": 0.5, "15_30_corrected": 0.5, "5_15": 0.5},
        "isotonic": {"30_60": 0.5, "15_30_corrected": 0.5, "5_15": 0.5},
    }


def main() -> None:
    paths = build_experiment_paths()
    paths.exp_dir.mkdir(parents=True, exist_ok=True)
    paths.fig_dir.mkdir(parents=True, exist_ok=True)

    specs = load_selected_bbox_specs()
    frames = build_horizon_feature_frames()

    parity_rows = []
    fold_rows = []
    calibration_rows = []
    reliability_rows = []
    score_tables: dict[str, pd.DataFrame] = {}

    for spec in specs.to_dict("records"):
        horizon_label = spec["horizon_label"]
        model_name = spec["model_name"]
        full_frame = frames[horizon_label].copy()
        canonical = canonicalize_track_rows(full_frame, horizon_label)
        saved_rows = load_saved_track_rows(horizon_label, model_name)
        reproduced, parity = reconstruct_seed_mean_scores(
            full_frame, None, model_name, saved_rows, float(spec["threshold"])
        )
        parity_rows.append(
            {
                "horizon_label": horizon_label,
                "model_name": model_name,
                "selected_threshold": float(spec["threshold"]),
                "threshold_rule": spec["threshold_rule"],
                "pearson_r": parity.pearson_r,
                "max_abs_diff": parity.max_abs_diff,
                "class_agreement": parity.class_agreement,
                "abs_f1_diff": parity.abs_f1_diff,
                "abs_pr_auc_diff": parity.abs_pr_auc_diff,
                "abs_roc_auc_diff": parity.abs_roc_auc_diff,
                "exact_probability_match": parity.exact_probability_match,
                "metrics_pass_tolerance": parity.metrics_pass_tolerance,
                "parity_status": parity.parity_status,
            }
        )

        full_train = full_frame.loc[full_frame["split"] == "train"].copy().reset_index(drop=True)
        full_train.attrs.update(full_frame.attrs)
        canonical_train = canonical.loc[canonical["split"] == "train"].copy().reset_index(drop=True)
        canonical_train.attrs.update(canonical.attrs)
        oof, fold_scheme, folds = make_canonical_grouped_oof_predictions(
            full_train, canonical_train, None, model_name
        )
        fold_summary = describe_fold_scheme(canonical_train, folds)
        fold_summary.insert(0, "fold_scheme", fold_scheme)
        fold_summary.insert(0, "horizon_label", horizon_label)
        fold_rows.append(fold_summary)

        platt = fit_platt_calibrator(
            oof["score_seed_mean"].to_numpy(dtype=float),
            oof["label"].to_numpy(dtype=int),
        )
        isotonic = fit_isotonic_calibrator(
            oof["score_seed_mean"].to_numpy(dtype=float),
            oof["label"].to_numpy(dtype=int),
        )

        full_scores = canonical[["sequence_id", "track_key", "split", "label", "group", "tte_frames"]].merge(
            reproduced[["sequence_id", "score_seed_mean"]],
            on="sequence_id",
            how="left",
            validate="one_to_one",
        )
        if full_scores["score_seed_mean"].isna().any():
            raise ValueError(f"reconstructed score coverage is incomplete for {horizon_label}")
        full_scores = full_scores.rename(columns={"score_seed_mean": "raw_score"})
        full_scores["platt_score"] = platt(full_scores["raw_score"].to_numpy(dtype=float))
        full_scores["isotonic_score"] = isotonic(full_scores["raw_score"].to_numpy(dtype=float))
        score_tables[horizon_label] = full_scores

        eval_frames = {
            "oof_train": oof.rename(columns={"score_seed_mean": "raw_score"}).copy(),
            "val": full_scores.loc[full_scores["split"] == "val"].copy(),
            "test": full_scores.loc[full_scores["split"] == "test"].copy(),
        }
        eval_frames["oof_train"]["platt_score"] = platt(eval_frames["oof_train"]["raw_score"].to_numpy(dtype=float))
        eval_frames["oof_train"]["isotonic_score"] = isotonic(eval_frames["oof_train"]["raw_score"].to_numpy(dtype=float))

        for split_name, eval_df in eval_frames.items():
            labels = eval_df["label"].to_numpy(dtype=int)
            raw_metrics = compute_calibration_metrics(labels, eval_df["raw_score"].to_numpy(dtype=float))
            for score_kind, score_col in (
                ("raw", "raw_score"),
                ("platt", "platt_score"),
                ("isotonic", "isotonic_score"),
            ):
                scores = eval_df[score_col].to_numpy(dtype=float)
                metrics = compute_calibration_metrics(labels, scores)
                bins = build_reliability_bins(labels, scores)
                bins.insert(0, "score_kind", score_kind)
                bins.insert(0, "split_name", split_name)
                bins.insert(0, "horizon_label", horizon_label)
                reliability_rows.append(bins)
                calibration_rows.append(
                    {
                        "horizon_label": horizon_label,
                        "split_name": split_name,
                        "score_kind": score_kind,
                        "n_samples": int(metrics["n_samples"]),
                        "brier_score": float(metrics["brier_score"]),
                        "ece": float(metrics["ece"]),
                        "mce": float(metrics["mce"]),
                        "roc_auc": float(metrics["roc_auc"]),
                        "brier_delta_vs_raw": float(metrics["brier_score"] - raw_metrics["brier_score"]),
                        "ece_delta_vs_raw": float(metrics["ece"] - raw_metrics["ece"]),
                    }
                )

    parity_df = pd.DataFrame(parity_rows).sort_values("horizon_label").reset_index(drop=True)
    fold_df = pd.concat(fold_rows, ignore_index=True)
    calibration_df = pd.DataFrame(calibration_rows).sort_values(
        ["horizon_label", "split_name", "score_kind"]
    ).reset_index(drop=True)
    reliability_df = pd.concat(reliability_rows, ignore_index=True)

    for horizon_label in ("30_60", "15_30_corrected", "5_15"):
        plot_reliability_figure(
            horizon_label,
            reliability_df,
            paths.fig_dir / f"{horizon_label}_reliability.png",
        )

    common_scores, missing_tracks = build_saved_common_track_frame(model_name="logreg")
    common_scores = attach_score_variants(common_scores, "bbox30", score_tables["30_60"])
    common_scores = attach_score_variants(common_scores, "bbox15", score_tables["15_30_corrected"])
    common_scores = attach_score_variants(common_scores, "bbox5", score_tables["5_15"])
    final_thresholds = build_final_thresholds(specs)
    risk_coverage_df = build_risk_coverage_tables(common_scores, final_thresholds)

    verdict = determine_calibration_verdict(calibration_df)
    calibration_protocol = (
        "# Experiment 026 Calibration-First Protocol\n\n"
        "- Canonical track rows are reconstructed per horizon with the staged rule: minimum `tte_frames`, then latest `start_frame` within each `(track_key, split)`.\n"
        "- Parity is checked against the saved JAAD BBox staged-track tables before any calibration claim is made.\n"
        "- OOF calibration uses training rows only, grouped by video when feasible and by track only as a documented fallback.\n"
        "- Seeds `42/43/44` are fit separately for OOF generation and averaged per training sample before fitting the calibrator.\n"
        "- Calibration evaluation is reported separately on OOF-train, validation, and test.\n"
        "- Validation-derived threshold candidates are used only for diagnostic risk-coverage tables; Clopper-Pearson bounds are exported as optional conservative diagnostics.\n"
        "- Raw scores are treated as classifier scores, not calibrated probabilities.\n"
    )
    readme = (
        "# Experiment 026: JAAD Calibration-First Study\n\n"
        f"Calibration verdict: `{verdict}`\n\n"
        "This revision preserves the parity and honest OOF reconstruction work, then stops after:\n"
        "- parity checks for canonical JAAD BBox models\n"
        "- OOF-only Platt and isotonic calibration\n"
        "- calibration metrics and reliability figures on OOF-train, validation, and test\n"
        "- simple validation-candidate risk-coverage tables for possible early exits\n"
    )

    files_used = "\n".join(
        [
            "scripts/experiments/jaad_calibrated_selective_early_exit.py",
            "scripts/experiments/jaad_biu_v2_benchmark_comparison.py",
            "scripts/experiments/jaad_staged_bbox_biu_v2.py",
            "scripts/experiments/jaad_corrected_15_30_rebuild_benchmark.py",
            "experiments/024_jaad_adaptive_early_decision/model_selection.csv",
            "experiments/011_jaad_staged_bbox_biu_v2/metrics/staged_track_table.csv",
            "experiments/021_jaad_corrected_15_30_rebuild/metrics/corrected_staged_track_table.csv",
        ]
    )
    commands_run = "python3 scripts/experiments/jaad_calibrated_selective_early_exit.py\n"

    parity_df.to_csv(paths.exp_dir / "parity_check.csv", index=False)
    fold_df.to_csv(paths.exp_dir / "oof_fold_scheme.csv", index=False)
    calibration_df.to_csv(paths.exp_dir / "calibration_metrics.csv", index=False)
    reliability_df.to_csv(paths.exp_dir / "reliability_bins.csv", index=False)
    risk_coverage_df.to_csv(paths.exp_dir / "risk_coverage_tables.csv", index=False)
    common_scores.to_csv(paths.exp_dir / "common_track_scores.csv", index=False)
    missing_tracks.to_csv(paths.exp_dir / "missing_track_audit.csv", index=False)
    write_text(paths.exp_dir / "calibration_protocol.md", calibration_protocol)
    write_text(paths.exp_dir / "README.md", readme)
    write_text(paths.exp_dir / "files_used.txt", files_used + "\n")
    write_text(paths.exp_dir / "commands_run.txt", commands_run)
    write_text(paths.exp_dir / "verdict.md", verdict + "\n")
    write_text(
        paths.exp_dir / "run_summary.json",
        json.dumps(
            {
                "verdict": verdict,
                "parity_rows": len(parity_df),
                "calibration_rows": len(calibration_df),
                "risk_coverage_rows": len(risk_coverage_df),
                "common_val_tracks": int((common_scores["split"] == "val").sum()),
                "common_test_tracks": int((common_scores["split"] == "test").sum()),
            },
            indent=2,
        ) + "\n",
    )


if __name__ == "__main__":
    main()
