"""Audit staged-prediction behavior across current JAAD and multi-horizon PIE/JAAD runs."""

from __future__ import annotations

import argparse
import json
import math
import textwrap
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.benchmark.build_jaad_event_aligned_candidates import TrackRecord, load_jaad_tracks
from src.benchmark.build_pie_benchmark_sequences import read_config
from src.benchmark.build_pie_window_horizon_sequences import build_sequences as build_pie_horizon_sequences
from src.benchmark.build_pie_window_horizon_sequences import write_outputs as write_pie_horizon_outputs
from src.benchmark.metrics import compute_binary_metrics, compute_confusion_counts, find_best_threshold_on_validation
from src.features.biu_features import FeatureBuildReport, build_biu_feature_table, load_keypoint_index, validate_biu_feature_outputs
from src.features.build_bbox_feature_tables import build_tables as build_pie_bbox_tables
from src.features.build_biu_bbox_feature_tables import build_combined_tables as build_pie_biu_bbox_tables
from src.features.build_biu_kp_feature_tables import build_combined_tables as build_pie_biu_kp_tables
from src.features.build_jaad_bbox_feature_tables import build_tables as build_jaad_bbox_tables
from src.features.build_jaad_kp_biu_feature_tables import build_all as build_jaad_kp_biu_tables
from src.features.build_keypoint_feature_tables import build_tables as build_pie_kp_tables
from src.features.build_pose_bbox_feature_tables import build_all as build_pie_pose_bbox_tables
from src.students.run_jaad_bbox_lightweight_baseline import _apply_medians, _fit_medians
from src.students.train_biu_supervised_suite import _build_model_specs, _predict_positive_proba


FEATURE_SETS = ["bbox", "biu", "kp", "biu_bbox", "kp_bbox", "biu_kp", "biu_kp_bbox"]
FEATURE_BUCKETS = {
    "bbox_based": ["bbox", "biu_bbox", "kp_bbox", "biu_kp_bbox"],
    "biu_based": ["biu", "biu_bbox", "biu_kp", "biu_kp_bbox"],
}
THRESHOLD_RULES = {
    "f1_opt": "f1",
    "balanced_accuracy_opt": "balanced_accuracy",
    "fixed_0_5": None,
}
SPLITS = ["train", "val", "test"]
BASE_META_COLUMNS = {
    "sequence_id",
    "split",
    "label",
    "set_id",
    "video_id",
    "ped_id",
    "track_id",
    "start_frame",
    "end_frame",
    "event_frame",
    "last_observed_frame",
    "tte_frames",
    "pred_start_frame",
    "pred_end_frame",
    "has_future_crossing",
    "has_observed_crossing",
    "first_future_cross_frame",
    "tte_to_first_future_cross",
    "track_eventually_crosses",
    "observation_length",
    "frame_step",
    "sample_stride_between_windows",
    "frame_sampling",
    "source_annotation_path",
    "parser_mode",
    "crossing_attr",
    "crossing_point",
    "decision_point",
    "bbox_sequence_json",
    "observed_frames_json",
    "observed_crosses_json",
    "negative_alignment_source",
    "onset_source",
    "track_length",
    "time_to_event_frames",
    "track_never_crossing",
    "num_crossing_transitions",
    "negative_event_policy",
    "bbox_missing_frame_count",
    "bbox_repaired_frame_count",
    "bbox_repair_method",
    "bbox_repaired_frames",
}

JAAD_DATASET_ROOT = Path("/home/abdelrahman/Hadwa/datasets/JAAD")
JAAD_KEYPOINT_JSONL = Path("data/jaad_keypoints_vitpose_s.jsonl")
JAAD_CURRENT_SPLITS = Path("data/jaad_splits_event_16_30_60_overlap08_default")
JAAD_CURRENT_FEATURES = Path("data/jaad_features_event_16_30_60_overlap08_default")
JAAD_CURRENT_RESULTS = Path("results/jaad_event_16_30_60_overlap08_default")
JAAD_SELECTED_THRESHOLDS = JAAD_CURRENT_RESULTS / "selected_thresholds.csv"
JAAD_SPLIT_AUDIT_ROOT = Path("data/jaad_splits_stage_horizon_audit")
JAAD_FEATURE_AUDIT_ROOT = Path("data/jaad_features_stage_horizon_audit")
JAAD_STAGE_RESULTS = Path("results/jaad_stage_horizon_audit")

PIE_JSONL = Path("/home/abdelrahman/Hadwa/pedestrian-intent-biu/data/data_keypoints_all.jsonl")
PIE_BBOX_SOURCE = Path("/home/abdelrahman/Hadwa/MSLSTM-PID/PIE/annotations/annotations")
PIE_CURRENT_SPLITS = Path("data/splits_pie_wacv_pcpa_30_60_overlap06")
PIE_CURRENT_FEATURES = Path("data/processed/features_wacv_pcpa_30_60_overlap06")
PIE_CURRENT_RESULTS = Path("results/wacv_pcpa_30_60_overlap06")
PIE_STAGE_SPLIT_ROOT = Path("data/splits_pie_stage_horizon_audit")
PIE_STAGE_FEATURE_ROOT = Path("data/processed/features_pie_stage_horizon_audit")
PIE_STAGE_RESULTS = Path("results/pie_stage_horizon_audit")
PIE_BASE_CONFIG = Path("configs/benchmark_pie_window_horizon_30_60.yaml")

HORIZONS = [
    {"label": "60_90", "tte_min": 60, "tte_max": 90},
    {"label": "30_60", "tte_min": 30, "tte_max": 60},
    {"label": "15_30", "tte_min": 15, "tte_max": 30},
    {"label": "5_15", "tte_min": 5, "tte_max": 15},
]

PER_SAMPLE_JAAD_OUTPUT = JAAD_CURRENT_RESULTS / "per_sample_fixed_0p5_predictions.csv"
JAAD_SUBGROUP_OUTPUT = JAAD_CURRENT_RESULTS / "jaad_test_negative_subgroups.csv"
JAAD_SCORE_GROUP_OUTPUT = JAAD_CURRENT_RESULTS / "stage_hypothesis_score_by_group.csv"

JAAD_GROUP_REPORT = Path("reports/JAAD_CLEAN_VS_AMBIGUOUS_NEGATIVE_AUDIT.md")
JAAD_STAGE_REPORT = Path("reports/JAAD_STAGE_HYPOTHESIS_SCORE_AUDIT.md")
FINAL_REPORT = Path("reports/STAGED_PREDICTION_HYPOTHESIS_RESULTS.md")
PARTIAL_REPORT = Path("reports/STAGED_PREDICTION_HYPOTHESIS_RESULTS_PARTIAL.md")
HORIZON_SKIP_NOTES: list[dict[str, Any]] = []
MODEL_ORDER = ["logistic_regression", "random_forest", "lightgbm", "xgboost"]
JAAD_SCOPES = [
    "all_test",
    "negative_clean_only",
    "negative_ambiguous_only",
    "positives_vs_clean_negatives",
    "positives_vs_ambiguous_negatives",
]


def _load_jaad_keypoint_frame_index(keypoint_jsonl: Path | None = None) -> tuple[set[tuple[str, str]], dict[tuple[str, str], set[int]]]:
    jsonl_path = keypoint_jsonl or JAAD_KEYPOINT_JSONL
    track_ids: set[tuple[str, str]] = set()
    track_frames: dict[tuple[str, str], set[int]] = {}
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            video_id = str(record.get("video", ""))
            ped_id = str(record.get("pid", ""))
            frame_idx = record.get("frame_idx")
            if video_id and ped_id and frame_idx is not None:
                track_key = (video_id, ped_id)
                track_ids.add(track_key)
                track_frames.setdefault(track_key, set()).add(int(frame_idx))
    return track_ids, track_frames


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _stage_dir(dataset: str) -> Path:
    return JAAD_STAGE_RESULTS if dataset == "JAAD" else PIE_STAGE_RESULTS


def _metrics_checkpoint_path(dataset: str) -> Path:
    return _stage_dir(dataset) / "metrics_all_horizons_checkpoint.csv"


def _selected_checkpoint_path(dataset: str) -> Path:
    return _stage_dir(dataset) / "selected_metrics_all_horizons_checkpoint.csv"


def _fixed_checkpoint_path(dataset: str) -> Path:
    return _stage_dir(dataset) / "fixed_0p5_metrics_all_horizons_checkpoint.csv"


def _clean_checkpoint_path() -> Path:
    return JAAD_STAGE_RESULTS / "clean_vs_ambiguous_metrics_jaad_checkpoint.csv"


def _failed_path(dataset: str) -> Path:
    return _stage_dir(dataset) / "failed_combinations.csv"


def _atomic_write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def _atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _upsert_rows(path: Path, new_rows: pd.DataFrame, key_cols: list[str], sort_cols: list[str]) -> pd.DataFrame:
    existing = _read_csv_optional(path)
    if existing.empty:
        combined = new_rows.copy()
    elif new_rows.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, new_rows], ignore_index=True)
    if combined.empty:
        _atomic_write_df(combined, path)
        return combined
    combined = combined.drop_duplicates(subset=key_cols, keep="last")
    combined = combined.sort_values(sort_cols).reset_index(drop=True)
    _atomic_write_df(combined, path)
    return combined


def _filter_df(df: pd.DataFrame, **criteria: Any) -> pd.DataFrame:
    if df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    for column, value in criteria.items():
        mask &= df[column].astype(str).eq(str(value))
    return df.loc[mask].copy()


def _log_progress(
    *,
    status: str,
    dataset: str,
    horizon_label: str,
    feature_set: str,
    model_name: str,
    split_sizes: dict[str, int],
    start_time: str,
    end_time: str,
    elapsed_seconds: float,
    reason: str = "",
) -> None:
    payload = {
        "status": status,
        "dataset": dataset,
        "horizon": horizon_label,
        "feature_set": feature_set,
        "model": model_name,
        "split_sizes": split_sizes,
        "start_time": start_time,
        "end_time": end_time,
        "elapsed_seconds": round(float(elapsed_seconds), 3),
    }
    if reason:
        payload["reason"] = reason
    print(payload)


def _append_failure_row(
    dataset: str,
    row: dict[str, Any],
) -> None:
    path = _failed_path(dataset)
    _upsert_rows(
        path,
        pd.DataFrame([row]),
        key_cols=["dataset", "horizon_label", "feature_set", "model_name", "stage", "status"],
        sort_cols=["dataset", "horizon_label", "feature_set", "model_name", "stage", "status"],
    )


def _remove_failure_rows(dataset: str, **criteria: Any) -> None:
    path = _failed_path(dataset)
    df = _read_csv_optional(path)
    if df.empty:
        return
    mask = pd.Series(True, index=df.index)
    for column, value in criteria.items():
        if column not in df.columns:
            continue
        mask &= df[column].astype(str).eq(str(value))
    updated = df.loc[~mask].copy().reset_index(drop=True)
    _atomic_write_df(updated, path)


def _remove_checkpoint_rows(dataset: str, horizon_label: str, feature_set: str) -> None:
    for path in [_metrics_checkpoint_path(dataset), _selected_checkpoint_path(dataset), _fixed_checkpoint_path(dataset), _clean_checkpoint_path()]:
        df = _read_csv_optional(path)
        if df.empty:
            continue
        mask = pd.Series(True, index=df.index)
        for column, value in [("dataset", dataset), ("horizon_label", horizon_label), ("feature_set", feature_set)]:
            if column not in df.columns:
                continue
            mask &= df[column].astype(str).eq(str(value))
        updated = df.loc[~mask].copy().reset_index(drop=True)
        _atomic_write_df(updated, path)


def _resolve_feature_table_path(features_dir: Path, split: str, feature_set: str) -> Path:
    candidates = [
        features_dir / f"{split}_{feature_set}.csv",
        features_dir / f"{feature_set}_{split}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing feature table for split={split} feature_set={feature_set} under {features_dir}")


def _load_feature_tables(features_dir: Path, feature_set: str) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        path = _resolve_feature_table_path(features_dir, split, feature_set)
        frame = pd.read_csv(path)
        if "sequence_id" not in frame.columns:
            raise ValueError(f"{path} missing sequence_id")
        if frame["sequence_id"].duplicated().any():
            raise ValueError(f"{path} contains duplicate sequence_id values")
        frames[split] = frame
    return frames


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in BASE_META_COLUMNS and pd.api.types.is_numeric_dtype(df[column])]


def _predict_with_selected_model(feature_set: str, model_name: str, features_dir: Path) -> tuple[dict[str, pd.DataFrame], list[str], dict[str, np.ndarray]]:
    frames = _load_feature_tables(features_dir, feature_set)
    feature_columns = _feature_columns(frames["train"])
    medians = _fit_medians(frames["train"], feature_columns)
    prepared = {split: _apply_medians(frame.copy(), feature_columns, medians) for split, frame in frames.items()}

    x_train = prepared["train"][feature_columns].astype(float).to_numpy()
    y_train = prepared["train"]["label"].astype(int).to_numpy()

    model_specs, warnings = _build_model_specs(random_seed=42, positive_rate=float(np.mean(y_train)))
    model_lookup = {name: model for name, model in model_specs}
    if model_name not in model_lookup:
        raise ValueError(f"{feature_set}: selected model {model_name!r} unavailable")
    model = model_lookup[model_name]
    model.fit(x_train, y_train)

    outputs: dict[str, np.ndarray] = {}
    for split in SPLITS:
        outputs[split] = _predict_positive_proba(model, prepared[split][feature_columns].astype(float).to_numpy())
    return prepared, feature_columns, outputs


def _confusion_category(y_true: int, y_pred: int) -> str:
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 1:
        return "FP"
    if y_true == 0 and y_pred == 0:
        return "TN"
    return "FN"


def _load_jaad_track_maps() -> tuple[dict[str, TrackRecord], dict[str, str], dict[str, str]]:
    tracks, _ = load_jaad_tracks(JAAD_DATASET_ROOT, "default")
    track_map = {track.track_id: track for track in tracks}
    attr_snippets: dict[str, str] = {}
    box_snippets: dict[str, str] = {}
    for path in (JAAD_DATASET_ROOT / "annotations_attributes").glob("*_attributes.xml"):
        video_id = path.stem.removesuffix("_attributes")
        root = ET.parse(path).getroot()
        for ped in root.findall("./pedestrian"):
            ped_id = ped.get("id")
            if ped_id:
                attr_snippets[f"{video_id}:{ped_id}"] = ET.tostring(ped, encoding="unicode")
    for path in (JAAD_DATASET_ROOT / "annotations").glob("*.xml"):
        video_id = path.stem
        root = ET.parse(path).getroot()
        for track_el in root.findall("./track"):
            if track_el.get("label") != "pedestrian":
                continue
            visible = [box for box in track_el.findall("./box") if int(box.get("outside", "0")) == 0]
            if not visible:
                continue
            ped_id = None
            for attr in visible[0].findall("./attribute"):
                if attr.get("name") == "id":
                    ped_id = (attr.text or "").strip()
                    break
            if ped_id:
                box_snippets[f"{video_id}:{ped_id}"] = ET.tostring(visible[0], encoding="unicode")
    return track_map, attr_snippets, box_snippets


def _jaad_group_markers(track: TrackRecord) -> dict[str, Any]:
    crossing_attr = track.crossing_attr
    crossing_point = -1 if track.crossing_point is None else int(track.crossing_point)
    decision_point = -1 if track.decision_point is None else int(track.decision_point)
    return {
        "has_frame_crossing_anywhere": int(any(cross == 1 for cross in track.crosses)),
        "has_cross_0_to_1_transition": int(bool(track.transition_frames)),
        "starts_already_crossing": int(track.starts_already_crossing),
        "crossing_attr": crossing_attr,
        "crossing_point": crossing_point,
        "decision_point": decision_point,
        "legacy_jaad_intention_positive": int(track.ped_id.endswith("b") and crossing_attr is not None and crossing_attr != -1),
        "ambiguity_crossing_attr_zero": int(crossing_attr == 0),
        "ambiguity_crossing_point_positive": int(crossing_point > 0),
        "ambiguity_decision_point_positive": int(decision_point > 0),
        "ambiguity_crossing_attr_minus1_but_crossing_point_positive": int(crossing_attr == -1 and crossing_point > 0),
    }


def _classify_jaad_sample_group(row: pd.Series, track: TrackRecord) -> str:
    markers = _jaad_group_markers(track)
    if int(row["label"]) == 1 and markers["has_cross_0_to_1_transition"] == 1:
        return "positive_actual_crossing"
    if int(row["label"]) == 0:
        if (
            markers["crossing_attr"] == -1
            and markers["has_frame_crossing_anywhere"] == 0
            and markers["has_cross_0_to_1_transition"] == 0
            and markers["crossing_point"] <= 0
        ):
            return "negative_clean"
        if (
            markers["has_frame_crossing_anywhere"] == 0
            and markers["has_cross_0_to_1_transition"] == 0
            and (
                markers["ambiguity_crossing_attr_zero"] == 1
                or markers["ambiguity_crossing_point_positive"] == 1
                or markers["legacy_jaad_intention_positive"] == 1
                or markers["ambiguity_decision_point_positive"] == 1
            )
        ):
            return "negative_ambiguous"
    return "other"


def build_current_jaad_per_sample_predictions() -> pd.DataFrame:
    selected = pd.read_csv(JAAD_SELECTED_THRESHOLDS)
    rows: list[dict[str, Any]] = []
    for item in selected.sort_values("feature_set").to_dict(orient="records"):
        feature_set = str(item["feature_set"])
        model_name = str(item["selected_model"])
        frames, _, outputs = _predict_with_selected_model(feature_set, model_name, JAAD_CURRENT_FEATURES)
        for split in SPLITS:
            frame = frames[split]
            probs = outputs[split]
            preds = (probs >= 0.5).astype(int)
            for index, row in frame.reset_index(drop=True).iterrows():
                y_true = int(row["label"])
                y_pred = int(preds[index])
                rows.append(
                    {
                        "split": split,
                        "video_id": row.get("video_id", ""),
                        "pid": row.get("ped_id", ""),
                        "sequence_id": row["sequence_id"],
                        "start_frame": int(row["start_frame"]),
                        "end_frame": int(row["end_frame"]),
                        "event_frame_or_anchor_frame": int(row.get("event_frame", row.get("last_observed_frame", -1))),
                        "y_true": y_true,
                        "y_score": float(probs[index]),
                        "y_pred_fixed_0p5": y_pred,
                        "confusion_category": _confusion_category(y_true, y_pred),
                        "feature_set": feature_set,
                        "selected_model": model_name,
                        "probability_source": "refit_selected_model",
                    }
                )
    df = pd.DataFrame(rows).sort_values(["feature_set", "split", "sequence_id"]).reset_index(drop=True)
    PER_SAMPLE_JAAD_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PER_SAMPLE_JAAD_OUTPUT, index=False)
    return df


def build_current_jaad_subgroups() -> pd.DataFrame:
    test_df = pd.read_csv(JAAD_CURRENT_SPLITS / "test.csv")
    track_map, attr_snippets, box_snippets = _load_jaad_track_maps()
    rows: list[dict[str, Any]] = []
    for record in test_df.to_dict(orient="records"):
        track = track_map[str(record["track_id"])]
        markers = _jaad_group_markers(track)
        group = _classify_jaad_sample_group(pd.Series(record), track)
        rows.append(
            {
                "sequence_id": record["sequence_id"],
                "split": "test",
                "video_id": record["video_id"],
                "pid": record["ped_id"],
                "track_id": record["track_id"],
                "start_frame": int(record["start_frame"]),
                "end_frame": int(record["end_frame"]),
                "event_frame_or_anchor_frame": int(record["event_frame"]),
                "y_true": int(record["label"]),
                "subgroup": group,
                **markers,
                "attr_xml_snippet": attr_snippets.get(str(record["track_id"]), ""),
                "box_xml_snippet": box_snippets.get(str(record["track_id"]), ""),
            }
        )
    df = pd.DataFrame(rows).sort_values(["subgroup", "sequence_id"]).reset_index(drop=True)
    df.to_csv(JAAD_SUBGROUP_OUTPUT, index=False)
    return df


def _group_score_rows(pred_df: pd.DataFrame, subgroup_df: pd.DataFrame) -> pd.DataFrame:
    merged = pred_df.loc[pred_df["split"] == "test"].merge(
        subgroup_df[
            [
                "sequence_id",
                "track_id",
                "subgroup",
                "y_true",
                "crossing_attr",
                "crossing_point",
                "decision_point",
                "legacy_jaad_intention_positive",
            ]
        ],
        on=["sequence_id", "y_true"],
        how="left",
        validate="many_to_one",
    )
    rows: list[dict[str, Any]] = []
    for (feature_set, model_name, subgroup), frame in merged.groupby(["feature_set", "selected_model", "subgroup"]):
        scores = frame["y_score"].astype(float)
        rows.append(
            {
                "feature_set": feature_set,
                "selected_model": model_name,
                "subgroup": subgroup,
                "count": int(len(frame)),
                "track_count": int(frame["track_id"].nunique()),
                "mean_score": float(scores.mean()),
                "median_score": float(scores.median()),
                "p10_score": float(scores.quantile(0.10)),
                "p25_score": float(scores.quantile(0.25)),
                "p50_score": float(scores.quantile(0.50)),
                "p75_score": float(scores.quantile(0.75)),
                "p90_score": float(scores.quantile(0.90)),
                "predicted_positive_rate_fixed_0p5": float(frame["y_pred_fixed_0p5"].mean()),
                "false_positive_rate_clean_or_ambiguous": float(frame["y_pred_fixed_0p5"].mean())
                if subgroup.startswith("negative_")
                else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["feature_set", "subgroup"]).reset_index(drop=True)


def _write_jaad_group_report(subgroup_df: pd.DataFrame) -> None:
    lines = [
        "# JAAD Clean Vs Ambiguous Negative Audit",
        "",
        "## Counts",
        "",
        "| subgroup | sample_count | track_count |",
        "|---|---:|---:|",
    ]
    summary = subgroup_df.groupby("subgroup").agg(sample_count=("sequence_id", "size"), track_count=("track_id", "nunique")).reset_index()
    for row in summary.to_dict(orient="records"):
        lines.append(f"| {row['subgroup']} | {int(row['sample_count'])} | {int(row['track_count'])} |")

    lines.extend(
        [
            "",
            "## Confirmation",
            "",
            f"- Negative-clean rows with any frame-level crossing anywhere: `{int(subgroup_df.loc[subgroup_df['subgroup'] == 'negative_clean', 'has_frame_crossing_anywhere'].sum())}`.",
            f"- Negative-ambiguous rows with any frame-level crossing anywhere: `{int(subgroup_df.loc[subgroup_df['subgroup'] == 'negative_ambiguous', 'has_frame_crossing_anywhere'].sum())}`.",
            f"- Negative-ambiguous rows with any `0->1` transition anywhere: `{int(subgroup_df.loc[subgroup_df['subgroup'] == 'negative_ambiguous', 'has_cross_0_to_1_transition'].sum())}`.",
            "- This confirms that the ambiguous negatives are still event-negative samples: they carry crossing-relevant metadata, but no actual future crossing onset exists in the full JAAD track.",
            "",
            "## Example Raw XML",
            "",
        ]
    )

    for subgroup in ["positive_actual_crossing", "negative_clean", "negative_ambiguous"]:
        sample = subgroup_df.loc[subgroup_df["subgroup"] == subgroup].head(1)
        if sample.empty:
            continue
        attr_xml = textwrap.shorten(str(sample.iloc[0]["attr_xml_snippet"]), width=260, placeholder=" ...")
        box_xml = textwrap.shorten(str(sample.iloc[0]["box_xml_snippet"]), width=260, placeholder=" ...")
        lines.extend(
            [
                f"### {subgroup}",
                "",
                "```xml",
                attr_xml,
                "```",
                "",
                "```xml",
                box_xml,
                "```",
                "",
            ]
        )

    JAAD_GROUP_REPORT.parent.mkdir(parents=True, exist_ok=True)
    JAAD_GROUP_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_jaad_stage_report(group_scores: pd.DataFrame, pred_df: pd.DataFrame, subgroup_df: pd.DataFrame) -> None:
    merged = pred_df.loc[pred_df["split"] == "test"].merge(
        subgroup_df[["sequence_id", "subgroup"]],
        on="sequence_id",
        how="left",
        validate="many_to_one",
    )
    fp_summary = (
        merged.loc[merged["confusion_category"] == "FP"]
        .groupby(["feature_set", "subgroup"])
        .size()
        .rename("count")
        .reset_index()
    )

    lines = [
        "# JAAD Stage Hypothesis Score Audit",
        "",
        "## Score Summary",
        "",
        "| feature_set | selected_model | subgroup | count | track_count | mean_score | median_score | p10 | p25 | p50 | p75 | p90 | pred_positive_rate@0.5 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in group_scores.to_dict(orient="records"):
        lines.append(
            f"| {row['feature_set']} | {row['selected_model']} | {row['subgroup']} | {int(row['count'])} | {int(row['track_count'])} | "
            f"{row['mean_score']:.4f} | {row['median_score']:.4f} | {row['p10_score']:.4f} | {row['p25_score']:.4f} | "
            f"{row['p50_score']:.4f} | {row['p75_score']:.4f} | {row['p90_score']:.4f} | {row['predicted_positive_rate_fixed_0p5']:.4f} |"
        )

    lines.extend(["", "## Current Fixed-0.5 False Positives", "", "| feature_set | FP clean | FP ambiguous | ambiguous_share |", "|---|---:|---:|---:|"])
    for feature_set in FEATURE_SETS:
        clean = int(fp_summary.loc[(fp_summary["feature_set"] == feature_set) & (fp_summary["subgroup"] == "negative_clean"), "count"].sum())
        ambiguous = int(fp_summary.loc[(fp_summary["feature_set"] == feature_set) & (fp_summary["subgroup"] == "negative_ambiguous"), "count"].sum())
        total = clean + ambiguous
        share = float(ambiguous / total) if total else float("nan")
        lines.append(f"| {feature_set} | {clean} | {ambiguous} | {share:.4f} |")

    lines.extend(["", "## Answers", ""])
    for bucket_name, bucket_features in FEATURE_BUCKETS.items():
        subset = group_scores.loc[group_scores["feature_set"].isin(bucket_features)].copy()
        amb = subset.loc[subset["subgroup"] == "negative_ambiguous", "mean_score"]
        clean = subset.loc[subset["subgroup"] == "negative_clean", "mean_score"]
        if not amb.empty and not clean.empty:
            lines.append(
                f"- {bucket_name} mean ambiguous-vs-clean score delta: `{float(amb.mean() - clean.mean()):+.4f}` over `{bucket_features}`."
            )
    bbox_clean = group_scores.loc[(group_scores["feature_set"] == "bbox") & (group_scores["subgroup"] == "negative_clean"), "predicted_positive_rate_fixed_0p5"]
    bbox_amb = group_scores.loc[(group_scores["feature_set"] == "bbox") & (group_scores["subgroup"] == "negative_ambiguous"), "predicted_positive_rate_fixed_0p5"]
    biu_bbox_clean = group_scores.loc[(group_scores["feature_set"] == "biu_bbox") & (group_scores["subgroup"] == "negative_clean"), "predicted_positive_rate_fixed_0p5"]
    kp_bbox_clean = group_scores.loc[(group_scores["feature_set"] == "kp_bbox") & (group_scores["subgroup"] == "negative_clean"), "predicted_positive_rate_fixed_0p5"]
    if not bbox_clean.empty and not bbox_amb.empty:
        lines.append(
            f"- BBox-only predicted-positive rate at `0.5`: clean negatives `{float(bbox_clean.iloc[0]):.4f}`, ambiguous negatives `{float(bbox_amb.iloc[0]):.4f}`."
        )
    if not biu_bbox_clean.empty and not kp_bbox_clean.empty:
        lines.append(
            f"- On clean negatives, `biu_bbox` vs `kp_bbox` predicted-positive rate at `0.5`: `{float(biu_bbox_clean.iloc[0]):.4f}` vs `{float(kp_bbox_clean.iloc[0]):.4f}`."
        )
    lines.append("- High JAAD false-positive rates should therefore be read in two parts: some mass is concentrated in crossing-relevant event-negative samples, but the clean-negative false-positive rate is still non-trivial for the KP-heavy sets.")

    JAAD_STAGE_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _jaad_window_stride(overlap: float) -> int:
    return max(1, int(round(16 * (1.0 - overlap))))


def _jaad_window_end_frames(track: TrackRecord, event_frame: int, tte_min: int, tte_max: int, window_stride: int) -> list[int]:
    earliest_end = max(track.frames[0] + 16 - 1, event_frame - tte_max)
    latest_end = event_frame - tte_min
    if earliest_end > latest_end:
        return []
    return list(range(earliest_end, latest_end + 1, int(window_stride)))


def _jaad_build_sample(track: TrackRecord, event_frame: int, end_frame: int, label: int, onset_source: str, negative_alignment_source: str, tte_min: int, tte_max: int, window_stride: int) -> dict[str, Any] | None:
    start_frame = end_frame - 16 + 1
    observed_frames = list(range(start_frame, end_frame + 1))
    if any(frame not in track.frame_to_cross for frame in observed_frames):
        return None
    observed_crosses = [track.frame_to_cross[frame] for frame in observed_frames]
    if any(cross == 1 for cross in observed_crosses):
        return None
    bbox_sequence = [track.frame_to_bbox[frame] for frame in observed_frames]
    tte_frames = event_frame - end_frame
    if not (tte_min <= tte_frames <= tte_max):
        return None
    sequence_id = f"{track.track_id}:{start_frame}:{end_frame}:{event_frame}:{label}"
    return {
        "sequence_id": sequence_id,
        "split": track.split,
        "video_id": track.video_id,
        "ped_id": track.ped_id,
        "track_id": track.track_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "event_frame": event_frame,
        "last_observed_frame": end_frame,
        "tte_frames": tte_frames,
        "label": label,
        "observation_length": 16,
        "frame_step": 1,
        "sample_stride_between_windows": int(window_stride),
        "frame_sampling": "consecutive",
        "onset_source": onset_source,
        "negative_alignment_source": negative_alignment_source,
        "track_length": track.track_length,
        "crossing_attr": track.crossing_attr,
        "crossing_point": track.crossing_point,
        "decision_point": track.decision_point,
        "has_observed_crossing": 0,
        "observed_frames_json": json.dumps(observed_frames),
        "observed_crosses_json": json.dumps(observed_crosses),
        "bbox_sequence_json": json.dumps(bbox_sequence),
        "source_annotation_path": track.source_annotation_path,
        "source_attributes_path": track.source_attributes_path,
        "parser_mode": "jaad_raw_annotations",
    }


def build_jaad_horizon_splits(
    tte_min: int,
    tte_max: int,
    overlap: float,
    output_dir: Path,
    *,
    keypoint_jsonl: Path,
    force_rebuild: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    can_reuse_existing = (
        not force_rebuild
        and (output_dir / "train.csv").exists()
        and (output_dir / "val.csv").exists()
        and (output_dir / "test.csv").exists()
    )
    if can_reuse_existing:
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            metadata = {"tte_min": tte_min, "tte_max": tte_max, "overlap": overlap, "window_stride": _jaad_window_stride(overlap)}
        if metadata.get("keypoint_jsonl") == str(keypoint_jsonl):
            split_videos = None
            df = pd.read_csv(output_dir / "all_samples.csv") if (output_dir / "all_samples.csv").exists() else pd.concat(
                [pd.read_csv(output_dir / f"{split}.csv") for split in SPLITS],
                ignore_index=True,
            )
        else:
            can_reuse_existing = False
    if not can_reuse_existing:
        tracks, split_videos = load_jaad_tracks(JAAD_DATASET_ROOT, "default")
        window_stride = _jaad_window_stride(overlap)
        samples: list[dict[str, Any]] = []
        metadata = {
            "tte_min": tte_min,
            "tte_max": tte_max,
            "overlap": overlap,
            "window_stride": window_stride,
            "keypoint_jsonl": str(keypoint_jsonl),
        }
        for track in tracks:
            if track.crossing_attr is None:
                continue
            if track.crossing_attr > 0:
                if track.starts_already_crossing or len(track.transition_frames) > 1 or not track.is_continuous or track.track_length < 16:
                    continue
                onset_frame: int | None = None
                onset_source = ""
                if len(track.transition_frames) == 1:
                    onset_frame = track.transition_frames[0]
                    onset_source = "frame_cross_transition"
                else:
                    cp = track.crossing_point
                    if cp is None or cp < 0 or cp < track.frames[0] or cp > track.frames[-1]:
                        continue
                    onset_frame = cp
                    onset_source = "crossing_point_fallback"
                for end_frame in _jaad_window_end_frames(track, onset_frame, tte_min, tte_max, window_stride):
                    sample = _jaad_build_sample(track, onset_frame, end_frame, 1, onset_source, "", tte_min, tte_max, window_stride)
                    if sample is not None:
                        samples.append(sample)
            else:
                if not track.is_continuous or track.track_length < 16 or track.starts_already_crossing or track.transition_frames or any(cross == 1 for cross in track.crosses):
                    continue
                event_frame = track.frames[-1]
                for end_frame in _jaad_window_end_frames(track, event_frame, tte_min, tte_max, window_stride):
                    sample = _jaad_build_sample(track, event_frame, end_frame, 0, "", "last_visible_frame", tte_min, tte_max, window_stride)
                    if sample is not None:
                        samples.append(sample)
        df = pd.DataFrame(samples).sort_values(["split", "video_id", "track_id", "start_frame"]).reset_index(drop=True)

    keypoint_tracks, keypoint_frames = _load_jaad_keypoint_frame_index(keypoint_jsonl)
    keep_mask_values: list[bool] = []
    drop_reason_values: list[str] = []
    for record in df.to_dict(orient="records"):
        track_key = (str(record["video_id"]), str(record["ped_id"]))
        if track_key not in keypoint_tracks:
            keep_mask_values.append(False)
            drop_reason_values.append("missing_track_in_keypoints")
            continue
        observed_frames = json.loads(record["observed_frames_json"])
        available_frames = keypoint_frames.get(track_key, set())
        if any(int(frame_idx) not in available_frames for frame_idx in observed_frames):
            keep_mask_values.append(False)
            drop_reason_values.append("missing_exact_observed_frames")
            continue
        keep_mask_values.append(True)
        drop_reason_values.append("")
    keep_mask = pd.Series(keep_mask_values, index=df.index)
    dropped_df = df.loc[~keep_mask].copy()
    if not dropped_df.empty:
        dropped_df["drop_reason"] = [reason for keep, reason in zip(keep_mask_values, drop_reason_values, strict=False) if not keep]
    df = df.loc[keep_mask].copy().sort_values(["split", "video_id", "track_id", "start_frame"]).reset_index(drop=True)
    df.to_csv(output_dir / "all_samples.csv", index=False)
    for split in SPLITS:
        df.loc[df["split"] == split].copy().to_csv(output_dir / f"{split}.csv", index=False)

    metadata.update(
        {
            "keypoint_jsonl": str(keypoint_jsonl),
            "num_rows": int(len(df)),
            "num_positive": int((df["label"] == 1).sum()),
            "num_negative": int((df["label"] == 0).sum()),
            "track_counts": {split: int(df.loc[df["split"] == split, "track_id"].nunique()) for split in SPLITS},
            "keypoint_alignment": {
                "dropped_rows": int(len(dropped_df)),
                "dropped_tracks": int(dropped_df["track_id"].nunique()) if not dropped_df.empty else 0,
                "dropped_rows_by_split": {split: int((dropped_df["split"] == split).sum()) for split in SPLITS},
                "dropped_tracks_by_split": {
                    split: int(dropped_df.loc[dropped_df["split"] == split, "track_id"].nunique()) for split in SPLITS
                },
                "dropped_rows_by_reason": dropped_df["drop_reason"].value_counts().sort_index().to_dict() if not dropped_df.empty else {},
            },
        }
    )
    if split_videos is not None:
        metadata["split_video_counts"] = {split: int(len(split_videos[split])) for split in SPLITS}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_dir


def ensure_jaad_horizon_features(split_dir: Path, feature_dir: Path, *, keypoint_jsonl: Path, force_rebuild: bool = False) -> Path:
    metadata_path = feature_dir / "feature_build_metadata.json"
    required = [feature_dir / f"train_{feature_set}.csv" for feature_set in FEATURE_SETS]
    can_reuse_existing = (
        not force_rebuild
        and all(path.exists() for path in required)
        and metadata_path.exists()
    )
    if can_reuse_existing:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("keypoint_jsonl") == str(keypoint_jsonl) and metadata.get("splits_dir") == str(split_dir):
            return feature_dir
    if all(path.exists() for path in required) and force_rebuild:
        pass
    feature_dir.mkdir(parents=True, exist_ok=True)
    tmp_report_dir = Path("/tmp/staged_prediction_jaad_reports")
    tmp_report_dir.mkdir(parents=True, exist_ok=True)
    build_jaad_bbox_tables(
        splits_dir=split_dir,
        output_dir=feature_dir,
        report_path=tmp_report_dir / f"{split_dir.name}_bbox_report.md",
    )
    build_jaad_kp_biu_tables(
        jsonl_path=keypoint_jsonl,
        splits_dir=split_dir,
        output_dir=feature_dir,
        audit_report_path=tmp_report_dir / f"{split_dir.name}_kp_biu_report.md",
        bbox_reference_summary_path=None,
    )
    metadata_path.write_text(
        json.dumps(
            {
                "splits_dir": str(split_dir),
                "feature_dir": str(feature_dir),
                "keypoint_jsonl": str(keypoint_jsonl),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return feature_dir


def _build_jaad_horizon_context(horizon: dict[str, Any], *, force_rebuild: bool, keypoint_jsonl: Path) -> dict[str, Any]:
    split_dir = build_jaad_horizon_splits(
        horizon["tte_min"],
        horizon["tte_max"],
        0.8,
        JAAD_SPLIT_AUDIT_ROOT / f"tte_{horizon['label']}",
        keypoint_jsonl=keypoint_jsonl,
        force_rebuild=force_rebuild,
    )
    split_sizes = {split: int(len(pd.read_csv(split_dir / f"{split}.csv"))) for split in SPLITS}
    if any(size == 0 for size in split_sizes.values()):
        return {
            "splits_dir": split_dir,
            "features_dir": None,
            "overlap": 0.8,
            "feasible": False,
            "skip_reason": f"empty split after exact keypoint alignment: {split_sizes}",
        }
    feature_dir = ensure_jaad_horizon_features(
        split_dir,
        JAAD_FEATURE_AUDIT_ROOT / f"tte_{horizon['label']}",
        keypoint_jsonl=keypoint_jsonl,
        force_rebuild=force_rebuild,
    )
    return {"splits_dir": split_dir, "features_dir": feature_dir, "overlap": 0.8, "feasible": True}


def _build_pie_biu_tables(jsonl_path: Path, splits_dir: Path, output_dir: Path) -> None:
    report = FeatureBuildReport(source_jsonl=str(jsonl_path), splits_dir=str(splits_dir), output_dir=str(output_dir))
    keypoint_index = load_keypoint_index(jsonl_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        build_biu_feature_table(splits_dir / f"{split}_sequences.csv", keypoint_index, output_dir / f"biu_{split}.csv", report)
    validate_biu_feature_outputs(splits_dir, output_dir)


def build_pie_horizon_splits(tte_min: int, tte_max: int, output_dir: Path) -> Path:
    if (output_dir / "train_sequences.csv").exists() and (output_dir / "val_sequences.csv").exists() and (output_dir / "test_sequences.csv").exists():
        return output_dir
    config = read_config(PIE_BASE_CONFIG)
    config["prediction_horizon_frames"] = [int(tte_min), int(tte_max)]
    df = build_pie_horizon_sequences(PIE_JSONL, config, parser_mode="jsonl_keypoints")
    write_pie_horizon_outputs(
        df.loc[df["split"] == "train"].copy(),
        df.loc[df["split"] == "val"].copy(),
        df.loc[df["split"] == "test"].copy(),
        config,
        output_dir,
    )
    return output_dir


def ensure_pie_horizon_features(split_dir: Path, feature_dir: Path) -> Path:
    required = [feature_dir / f"{feature_set}_train.csv" for feature_set in FEATURE_SETS]
    if all(path.exists() for path in required):
        return feature_dir
    feature_dir.mkdir(parents=True, exist_ok=True)
    tmp_report_dir = Path("/tmp/staged_prediction_pie_reports")
    tmp_report_dir.mkdir(parents=True, exist_ok=True)
    build_pie_bbox_tables(PIE_BBOX_SOURCE, split_dir, feature_dir, tmp_report_dir / f"{split_dir.name}_bbox_report.md")
    build_pie_kp_tables(PIE_JSONL, split_dir, feature_dir, tmp_report_dir / f"{split_dir.name}_kp_report.md")
    _build_pie_biu_tables(PIE_JSONL, split_dir, feature_dir)
    build_pie_biu_bbox_tables(feature_dir, split_dir, feature_dir, tmp_report_dir / f"{split_dir.name}_biu_bbox_report.md")
    build_pie_biu_kp_tables(feature_dir, split_dir, feature_dir, tmp_report_dir / f"{split_dir.name}_biu_kp_report.md")
    build_pie_pose_bbox_tables(feature_dir, split_dir, feature_dir, tmp_report_dir)
    return feature_dir


def _dataset_context(dataset: str, horizon: dict[str, Any]) -> dict[str, Any]:
    if dataset == "JAAD":
        if horizon["label"] == "30_60":
            return {
                "splits_dir": JAAD_CURRENT_SPLITS,
                "features_dir": JAAD_CURRENT_FEATURES,
                "overlap": 0.8,
                "feasible": True,
            }
        raise RuntimeError("_dataset_context no longer supports JAAD non-30_60 without explicit keypoint_jsonl")
    if horizon["label"] == "30_60":
        return {
            "splits_dir": PIE_CURRENT_SPLITS,
            "features_dir": PIE_CURRENT_FEATURES,
            "overlap": 0.6,
            "feasible": True,
        }
    split_dir = build_pie_horizon_splits(horizon["tte_min"], horizon["tte_max"], PIE_STAGE_SPLIT_ROOT / f"tte_{horizon['label']}")
    feature_dir = ensure_pie_horizon_features(split_dir, PIE_STAGE_FEATURE_ROOT / f"tte_{horizon['label']}")
    return {"splits_dir": split_dir, "features_dir": feature_dir, "overlap": 0.6, "feasible": True}


def _balanced_accuracy(conf: dict[str, int]) -> float:
    tn, fp, fn, tp = [int(conf[key]) for key in ["tn", "fp", "fn", "tp"]]
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return float((tpr + tnr) / 2.0)


def _prepare_feature_set_data(
    feature_set: str,
    features_dir: Path,
    model_filter: set[str] | None,
) -> tuple[dict[str, pd.DataFrame], list[str], list[tuple[str, Any]], list[str], dict[str, int]]:
    frames = _load_feature_tables(features_dir, feature_set)
    feature_columns = _feature_columns(frames["train"])
    medians = _fit_medians(frames["train"], feature_columns)
    prepared = {split: _apply_medians(frame.copy(), feature_columns, medians) for split, frame in frames.items()}
    x_train = prepared["train"][feature_columns].astype(float).to_numpy()
    y_train = prepared["train"]["label"].astype(int).to_numpy()
    model_specs, warnings = _build_model_specs(random_seed=42, positive_rate=float(np.mean(y_train)))
    if model_filter is not None:
        model_specs = [(name, model) for name, model in model_specs if name in model_filter]
    split_sizes = {split: int(len(prepared[split])) for split in SPLITS}
    return prepared, feature_columns, model_specs, warnings, split_sizes


def _single_class_splits(prepared: dict[str, pd.DataFrame]) -> dict[str, list[int]]:
    summary: dict[str, list[int]] = {}
    for split, frame in prepared.items():
        labels = sorted({int(value) for value in frame["label"].dropna().astype(int).tolist()})
        if len(labels) < 2:
            summary[split] = labels
    return summary


def _fit_single_model(
    prepared: dict[str, pd.DataFrame],
    feature_columns: list[str],
    model_name: str,
    model_specs: list[tuple[str, Any]],
) -> dict[str, np.ndarray]:
    model_lookup = {name: model for name, model in model_specs}
    if model_name not in model_lookup:
        raise ValueError(f"Requested model {model_name!r} is unavailable in the current environment.")
    model = model_lookup[model_name]
    x_train = prepared["train"][feature_columns].astype(float).to_numpy()
    y_train = prepared["train"]["label"].astype(int).to_numpy()
    model.fit(x_train, y_train)
    outputs: dict[str, np.ndarray] = {}
    for split in SPLITS:
        x = prepared[split][feature_columns].astype(float).to_numpy()
        outputs[split] = _predict_positive_proba(model, x)
    return outputs


def _metrics_row(dataset: str, overlap: float, horizon: dict[str, Any], feature_set: str, model_name: str, threshold: float, val_metrics: dict[str, float], test_metrics: dict[str, float], val_conf: dict[str, int], test_conf: dict[str, int], feature_count: int) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "overlap": overlap,
        "horizon_label": horizon["label"],
        "tte_min": int(horizon["tte_min"]),
        "tte_max": int(horizon["tte_max"]),
        "feature_set": feature_set,
        "model_name": model_name,
        "feature_count": int(feature_count),
        "selected_threshold": float(threshold),
        "validation_f1": float(val_metrics["f1"]),
        "validation_precision": float(val_metrics["precision"]),
        "validation_recall": float(val_metrics["recall"]),
        "validation_accuracy": float(val_metrics["accuracy"]),
        "validation_balanced_accuracy": _balanced_accuracy(val_conf),
        "validation_roc_auc": float(val_metrics["roc_auc"]),
        "validation_pr_auc": float(val_metrics["pr_auc"]),
        "test_f1": float(test_metrics["f1"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_balanced_accuracy": _balanced_accuracy(test_conf),
        "test_roc_auc": float(test_metrics["roc_auc"]),
        "test_pr_auc": float(test_metrics["pr_auc"]),
        "test_tn": int(test_conf["tn"]),
        "test_fp": int(test_conf["fp"]),
        "test_fn": int(test_conf["fn"]),
        "test_tp": int(test_conf["tp"]),
        "test_fpr": float(test_conf["fp"] / (test_conf["fp"] + test_conf["tn"])) if (test_conf["fp"] + test_conf["tn"]) else float("nan"),
        "test_fnr": float(test_conf["fn"] / (test_conf["fn"] + test_conf["tp"])) if (test_conf["fn"] + test_conf["tp"]) else float("nan"),
    }


def _evaluate_test_subset(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    metrics = compute_binary_metrics(y_true, y_prob, threshold)
    conf = compute_confusion_counts(y_true, y_prob, threshold)
    return {
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": _balanced_accuracy(conf),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "roc_auc": float(metrics["roc_auc"]),
        "pr_auc": float(metrics["pr_auc"]),
        "tn": int(conf["tn"]),
        "fp": int(conf["fp"]),
        "fn": int(conf["fn"]),
        "tp": int(conf["tp"]),
        "fpr": float(conf["fp"] / (conf["fp"] + conf["tn"])) if (conf["fp"] + conf["tn"]) else float("nan"),
        "fnr": float(conf["fn"] / (conf["fn"] + conf["tp"])) if (conf["fn"] + conf["tp"]) else float("nan"),
        "predicted_positive_rate": float((y_prob >= threshold).mean()),
    }


def _load_sequence_frame(splits_dir: Path, dataset: str) -> pd.DataFrame:
    if dataset == "JAAD":
        frames = [pd.read_csv(splits_dir / f"{split}.csv") for split in SPLITS]
    else:
        frames = [pd.read_csv(splits_dir / f"{split}_sequences.csv") for split in SPLITS]
    return pd.concat(frames, ignore_index=True)


def _jaad_subgroups_for_split(test_df: pd.DataFrame, track_map: dict[str, TrackRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in test_df.to_dict(orient="records"):
        track = track_map[str(record["track_id"])]
        markers = _jaad_group_markers(track)
        subgroup = _classify_jaad_sample_group(pd.Series(record), track)
        rows.append({"sequence_id": record["sequence_id"], "track_id": record["track_id"], "subgroup": subgroup, **markers})
    return pd.DataFrame(rows)


def _materialize_dataset_outputs(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stage_dir = _stage_dir(dataset)
    stage_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = _read_csv_optional(_metrics_checkpoint_path(dataset))
    selected_df = _read_csv_optional(_selected_checkpoint_path(dataset))
    fixed_df = _read_csv_optional(_fixed_checkpoint_path(dataset))
    _atomic_write_df(metrics_df, stage_dir / "metrics_all_horizons.csv")
    _atomic_write_df(selected_df, stage_dir / "selected_metrics_all_horizons.csv")
    _atomic_write_df(fixed_df, stage_dir / "fixed_0p5_metrics_all_horizons.csv")
    return metrics_df, selected_df, fixed_df


def _materialize_jaad_clean_outputs() -> pd.DataFrame:
    JAAD_STAGE_RESULTS.mkdir(parents=True, exist_ok=True)
    clean_df = _read_csv_optional(_clean_checkpoint_path())
    _atomic_write_df(clean_df, JAAD_STAGE_RESULTS / "clean_vs_ambiguous_metrics_jaad.csv")
    return clean_df


def _selected_datasets(args: argparse.Namespace) -> list[str]:
    datasets = ["JAAD", "PIE"]
    if args.skip_jaad:
        datasets = [dataset for dataset in datasets if dataset != "JAAD"]
    if args.skip_pie:
        datasets = [dataset for dataset in datasets if dataset != "PIE"]
    return datasets


def _selected_horizons(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.horizons:
        return HORIZONS
    wanted = set(args.horizons)
    return [horizon for horizon in HORIZONS if horizon["label"] in wanted]


def _selected_feature_sets(args: argparse.Namespace) -> list[str]:
    if not args.feature_sets:
        return FEATURE_SETS
    return [feature_set for feature_set in FEATURE_SETS if feature_set in set(args.feature_sets)]


def _selected_models(args: argparse.Namespace) -> set[str] | None:
    if not args.models:
        return None
    return set(args.models)


def _all_threshold_rows_exist(
    dataset: str,
    horizon_label: str,
    feature_set: str,
    selected_model: str,
) -> bool:
    selected_df = _read_csv_optional(_selected_checkpoint_path(dataset))
    fixed_df = _read_csv_optional(_fixed_checkpoint_path(dataset))
    if "threshold_rule" not in selected_df.columns or "threshold_rule" not in fixed_df.columns:
        return False
    expected_selected = {"f1_opt", "balanced_accuracy_opt"}
    got_selected = set(
        _filter_df(selected_df, dataset=dataset, horizon_label=horizon_label, feature_set=feature_set)["threshold_rule"].astype(str)
    )
    got_fixed = set(
        _filter_df(fixed_df, dataset=dataset, horizon_label=horizon_label, feature_set=feature_set)["threshold_rule"].astype(str)
    )
    return got_selected == expected_selected and got_fixed == {"fixed_0_5"}


def _all_clean_rows_exist(horizon_label: str, feature_set: str, threshold_rule: str) -> bool:
    clean_df = _read_csv_optional(_clean_checkpoint_path())
    if "scope" not in clean_df.columns:
        return False
    got_scopes = set(
        _filter_df(clean_df, dataset="JAAD", horizon_label=horizon_label, feature_set=feature_set, threshold_rule=threshold_rule)["scope"].astype(str)
    )
    return got_scopes == set(JAAD_SCOPES)


def _run_threshold_rows(
    *,
    dataset: str,
    horizon: dict[str, Any],
    overlap: float,
    feature_set: str,
    feature_count: int,
    best_model: str,
    prepared: dict[str, pd.DataFrame],
    feature_columns: list[str],
    model_specs: list[tuple[str, Any]],
    split_frame: pd.DataFrame,
    jaad_track_map: dict[str, TrackRecord] | None,
    force: bool,
) -> None:
    selected_cp = _selected_checkpoint_path(dataset)
    fixed_cp = _fixed_checkpoint_path(dataset)
    if not force and _all_threshold_rows_exist(dataset, horizon["label"], feature_set, best_model):
        if dataset != "JAAD" or all(_all_clean_rows_exist(horizon["label"], feature_set, rule) for rule in THRESHOLD_RULES):
            return

    split_sizes = {split: int(len(prepared[split])) for split in SPLITS}
    model_outputs = _fit_single_model(prepared, feature_columns, best_model, model_specs)
    y_val = prepared["val"]["label"].astype(int).to_numpy()
    y_test = prepared["test"]["label"].astype(int).to_numpy()
    test_meta = split_frame.loc[split_frame["split"] == "test"].copy()
    subgroup_df = _jaad_subgroups_for_split(test_meta, jaad_track_map) if dataset == "JAAD" and jaad_track_map is not None else pd.DataFrame()

    for rule_name, metric_name in THRESHOLD_RULES.items():
        rule_path = fixed_cp if rule_name == "fixed_0_5" else selected_cp
        existing_rule = _filter_df(
            _read_csv_optional(rule_path),
            dataset=dataset,
            horizon_label=horizon["label"],
            feature_set=feature_set,
            threshold_rule=rule_name,
        )
        if not force and not existing_rule.empty:
            if dataset != "JAAD" or _all_clean_rows_exist(horizon["label"], feature_set, rule_name):
                continue

        start_time = _now_iso()
        start_perf = time.perf_counter()
        try:
            threshold = 0.5 if metric_name is None else find_best_threshold_on_validation(y_val, model_outputs["val"], metric=metric_name)
            val_metrics = compute_binary_metrics(y_val, model_outputs["val"], threshold)
            test_metrics = compute_binary_metrics(y_test, model_outputs["test"], threshold)
            val_conf = compute_confusion_counts(y_val, model_outputs["val"], threshold)
            test_conf = compute_confusion_counts(y_test, model_outputs["test"], threshold)
            row = _metrics_row(
                dataset,
                overlap,
                horizon,
                feature_set,
                best_model,
                threshold,
                val_metrics,
                test_metrics,
                val_conf,
                test_conf,
                feature_count,
            )
            row["selected_model"] = best_model
            row["threshold_rule"] = rule_name
            _upsert_rows(
                rule_path,
                pd.DataFrame([row]),
                key_cols=["dataset", "horizon_label", "feature_set", "threshold_rule"],
                sort_cols=["dataset", "horizon_label", "feature_set", "threshold_rule"],
            )

            if dataset == "JAAD":
                test_probs_df = pd.DataFrame(
                    {
                        "sequence_id": prepared["test"]["sequence_id"].astype(str),
                        "y_score": model_outputs["test"],
                        "y_true": y_test,
                    }
                )
                merged = (
                    test_meta[["sequence_id", "track_id", "label"]]
                    .merge(test_probs_df, on="sequence_id", how="left", validate="one_to_one")
                    .merge(subgroup_df[["sequence_id", "subgroup"]], on="sequence_id", how="left", validate="one_to_one")
                )
                scope_masks = {
                    "all_test": np.ones(len(merged), dtype=bool),
                    "negative_clean_only": merged["subgroup"].eq("negative_clean").to_numpy(),
                    "negative_ambiguous_only": merged["subgroup"].eq("negative_ambiguous").to_numpy(),
                    "positives_vs_clean_negatives": merged["subgroup"].isin(["positive_actual_crossing", "negative_clean"]).to_numpy(),
                    "positives_vs_ambiguous_negatives": merged["subgroup"].isin(["positive_actual_crossing", "negative_ambiguous"]).to_numpy(),
                }
                clean_rows: list[dict[str, Any]] = []
                for scope, mask in scope_masks.items():
                    subset = merged.loc[mask].copy()
                    if subset.empty:
                        continue
                    subset_metrics = _evaluate_test_subset(
                        subset["label"].astype(int).to_numpy(),
                        subset["y_score"].astype(float).to_numpy(),
                        threshold,
                    )
                    clean_rows.append(
                        {
                            "dataset": dataset,
                            "overlap": overlap,
                            "horizon_label": horizon["label"],
                            "tte_min": int(horizon["tte_min"]),
                            "tte_max": int(horizon["tte_max"]),
                            "feature_set": feature_set,
                            "selected_model": best_model,
                            "threshold_rule": rule_name,
                            "threshold": float(threshold),
                            "scope": scope,
                            "sample_count": int(len(subset)),
                            "track_count": int(subset["track_id"].nunique()),
                            **subset_metrics,
                        }
                    )
                if clean_rows:
                    _upsert_rows(
                        _clean_checkpoint_path(),
                        pd.DataFrame(clean_rows),
                        key_cols=["dataset", "horizon_label", "feature_set", "threshold_rule", "scope"],
                        sort_cols=["dataset", "horizon_label", "feature_set", "threshold_rule", "scope"],
                    )
                    _materialize_jaad_clean_outputs()

            end_time = _now_iso()
            elapsed = time.perf_counter() - start_perf
            _log_progress(
                status="fitted",
                dataset=dataset,
                horizon_label=horizon["label"],
                feature_set=feature_set,
                model_name=f"{best_model}:{rule_name}",
                split_sizes=split_sizes,
                start_time=start_time,
                end_time=end_time,
                elapsed_seconds=elapsed,
            )
        except Exception as exc:
            end_time = _now_iso()
            elapsed = time.perf_counter() - start_perf
            _append_failure_row(
                dataset,
                {
                    "dataset": dataset,
                    "horizon_label": horizon["label"],
                    "feature_set": feature_set,
                    "model_name": best_model,
                    "threshold_rule": rule_name,
                    "stage": "threshold_eval",
                    "status": "failed",
                    "reason": str(exc),
                    "split_sizes": json.dumps(split_sizes, sort_keys=True),
                    "start_time": start_time,
                    "end_time": end_time,
                    "elapsed_seconds": float(elapsed),
                },
            )
            _log_progress(
                status="failed",
                dataset=dataset,
                horizon_label=horizon["label"],
                feature_set=feature_set,
                model_name=f"{best_model}:{rule_name}",
                split_sizes=split_sizes,
                start_time=start_time,
                end_time=end_time,
                elapsed_seconds=elapsed,
                reason=str(exc),
            )


def run_horizon_audit(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    datasets = _selected_datasets(args)
    horizons = _selected_horizons(args)
    feature_sets = _selected_feature_sets(args)
    model_filter = _selected_models(args)
    jaad_track_map, _, _ = _load_jaad_track_maps()

    for dataset in datasets:
        _stage_dir(dataset).mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        for horizon in horizons:
            if dataset == "JAAD" and horizon["label"] != "30_60":
                context = _build_jaad_horizon_context(
                    horizon,
                    force_rebuild=bool(args.force),
                    keypoint_jsonl=args.jaad_keypoint_jsonl,
                )
            else:
                context = _dataset_context(dataset, horizon)
            if not context.get("feasible", True):
                reason = str(context.get("skip_reason", "infeasible horizon"))
                HORIZON_SKIP_NOTES.append({"dataset": dataset, "horizon_label": horizon["label"], "reason": reason})
                _append_failure_row(
                    dataset,
                    {
                        "dataset": dataset,
                        "horizon_label": horizon["label"],
                        "feature_set": "",
                        "model_name": "",
                        "threshold_rule": "",
                        "stage": "dataset_horizon",
                        "status": "infeasible",
                        "reason": reason,
                        "split_sizes": json.dumps({}, sort_keys=True),
                        "start_time": _now_iso(),
                        "end_time": _now_iso(),
                        "elapsed_seconds": 0.0,
                    },
                )
                continue
            _remove_failure_rows(dataset, horizon_label=horizon["label"], stage="dataset_horizon", status="infeasible")

            features_dir = Path(context["features_dir"])
            overlap = float(context["overlap"])
            split_frame = _load_sequence_frame(Path(context["splits_dir"]), dataset)

            for feature_set in feature_sets:
                _remove_failure_rows(dataset, horizon_label=horizon["label"], feature_set=feature_set)
                try:
                    prepared, feature_columns, model_specs, warnings, split_sizes = _prepare_feature_set_data(
                        feature_set,
                        features_dir,
                        model_filter,
                    )
                except Exception as exc:
                    _append_failure_row(
                        dataset,
                        {
                            "dataset": dataset,
                            "horizon_label": horizon["label"],
                            "feature_set": feature_set,
                            "model_name": "",
                            "threshold_rule": "",
                            "stage": "feature_prepare",
                            "status": "failed",
                            "reason": str(exc),
                            "split_sizes": json.dumps({}, sort_keys=True),
                            "start_time": _now_iso(),
                            "end_time": _now_iso(),
                            "elapsed_seconds": 0.0,
                        },
                    )
                    continue

                degenerate_splits = _single_class_splits(prepared)
                if degenerate_splits:
                    _remove_checkpoint_rows(dataset, horizon["label"], feature_set)
                    _append_failure_row(
                        dataset,
                        {
                            "dataset": dataset,
                            "horizon_label": horizon["label"],
                            "feature_set": feature_set,
                            "model_name": "",
                            "threshold_rule": "",
                            "stage": "feature_prepare",
                            "status": "failed",
                            "reason": f"single_class_split_labels: {degenerate_splits}",
                            "split_sizes": json.dumps(split_sizes, sort_keys=True),
                            "start_time": _now_iso(),
                            "end_time": _now_iso(),
                            "elapsed_seconds": 0.0,
                        },
                    )
                    _log_progress(
                        status="failed",
                        dataset=dataset,
                        horizon_label=horizon["label"],
                        feature_set=feature_set,
                        model_name="",
                        split_sizes=split_sizes,
                        start_time=_now_iso(),
                        end_time=_now_iso(),
                        elapsed_seconds=0.0,
                        reason=f"single_class_split_labels: {degenerate_splits}",
                    )
                    continue

                if warnings:
                    print({"dataset": dataset, "horizon": horizon["label"], "feature_set": feature_set, "warnings": warnings})
                feature_count = len(feature_columns)
                metrics_path = _metrics_checkpoint_path(dataset)
                existing_metrics = _read_csv_optional(metrics_path)
                model_metrics: list[dict[str, Any]] = []

                for model_name, _model in model_specs:
                    existing_row = _filter_df(
                        existing_metrics,
                        dataset=dataset,
                        horizon_label=horizon["label"],
                        feature_set=feature_set,
                        model_name=model_name,
                    )
                    if not args.force and not existing_row.empty:
                        model_metrics.append(existing_row.iloc[-1].to_dict())
                        _log_progress(
                            status="skipped",
                            dataset=dataset,
                            horizon_label=horizon["label"],
                            feature_set=feature_set,
                            model_name=model_name,
                            split_sizes=split_sizes,
                            start_time=_now_iso(),
                            end_time=_now_iso(),
                            elapsed_seconds=0.0,
                            reason="checkpoint_exists",
                        )
                        continue

                    start_time = _now_iso()
                    start_perf = time.perf_counter()
                    try:
                        split_outputs = _fit_single_model(prepared, feature_columns, model_name, model_specs)
                        y_val = prepared["val"]["label"].astype(int).to_numpy()
                        y_test = prepared["test"]["label"].astype(int).to_numpy()
                        threshold = find_best_threshold_on_validation(y_val, split_outputs["val"], metric="f1")
                        val_metrics = compute_binary_metrics(y_val, split_outputs["val"], threshold)
                        test_metrics = compute_binary_metrics(y_test, split_outputs["test"], threshold)
                        val_conf = compute_confusion_counts(y_val, split_outputs["val"], threshold)
                        test_conf = compute_confusion_counts(y_test, split_outputs["test"], threshold)
                        row = _metrics_row(
                            dataset,
                            overlap,
                            horizon,
                            feature_set,
                            model_name,
                            threshold,
                            val_metrics,
                            test_metrics,
                            val_conf,
                            test_conf,
                            feature_count,
                        )
                        model_metrics.append(row)
                        existing_metrics = _upsert_rows(
                            metrics_path,
                            pd.DataFrame([row]),
                            key_cols=["dataset", "horizon_label", "feature_set", "model_name"],
                            sort_cols=["dataset", "horizon_label", "feature_set", "model_name"],
                        )
                        end_time = _now_iso()
                        elapsed = time.perf_counter() - start_perf
                        _log_progress(
                            status="fitted",
                            dataset=dataset,
                            horizon_label=horizon["label"],
                            feature_set=feature_set,
                            model_name=model_name,
                            split_sizes=split_sizes,
                            start_time=start_time,
                            end_time=end_time,
                            elapsed_seconds=elapsed,
                        )
                    except Exception as exc:
                        end_time = _now_iso()
                        elapsed = time.perf_counter() - start_perf
                        _append_failure_row(
                            dataset,
                            {
                                "dataset": dataset,
                                "horizon_label": horizon["label"],
                                "feature_set": feature_set,
                                "model_name": model_name,
                                "threshold_rule": "",
                                "stage": "model_fit",
                                "status": "failed",
                                "reason": str(exc),
                                "split_sizes": json.dumps(split_sizes, sort_keys=True),
                                "start_time": start_time,
                                "end_time": end_time,
                                "elapsed_seconds": float(elapsed),
                            },
                        )
                        _log_progress(
                            status="failed",
                            dataset=dataset,
                            horizon_label=horizon["label"],
                            feature_set=feature_set,
                            model_name=model_name,
                            split_sizes=split_sizes,
                            start_time=start_time,
                            end_time=end_time,
                            elapsed_seconds=elapsed,
                            reason=str(exc),
                        )

                if not model_metrics:
                    continue

                best = sorted(model_metrics, key=lambda row: (row["validation_f1"], row["validation_pr_auc"]), reverse=True)[0]
                _run_threshold_rows(
                    dataset=dataset,
                    horizon=horizon,
                    overlap=overlap,
                    feature_set=feature_set,
                    feature_count=feature_count,
                    best_model=str(best["model_name"]),
                    prepared=prepared,
                    feature_columns=feature_columns,
                    model_specs=model_specs,
                    split_frame=split_frame,
                    jaad_track_map=jaad_track_map if dataset == "JAAD" else None,
                    force=args.force,
                )

            _materialize_dataset_outputs(dataset)
            if dataset == "JAAD":
                _materialize_jaad_clean_outputs()

    metrics_df = pd.concat(
        [_read_csv_optional(_metrics_checkpoint_path(dataset)) for dataset in datasets],
        ignore_index=True,
    ) if datasets else pd.DataFrame()
    selected_df = pd.concat(
        [_read_csv_optional(_selected_checkpoint_path(dataset)) for dataset in datasets],
        ignore_index=True,
    ) if datasets else pd.DataFrame()
    fixed_df = pd.concat(
        [_read_csv_optional(_fixed_checkpoint_path(dataset)) for dataset in datasets],
        ignore_index=True,
    ) if datasets else pd.DataFrame()
    clean_df = _read_csv_optional(_clean_checkpoint_path()) if "JAAD" in datasets else pd.DataFrame()
    return metrics_df, selected_df, fixed_df, clean_df


def _write_final_report(group_scores: pd.DataFrame, subgroup_df: pd.DataFrame, selected_df: pd.DataFrame, fixed_df: pd.DataFrame, clean_df: pd.DataFrame) -> None:
    lines = [
        "# Staged Prediction Hypothesis Results",
        "",
        "## Current JAAD Clean vs Ambiguous Pattern",
        "",
    ]

    for feature_set in ["bbox", "biu", "kp", "biu_bbox", "kp_bbox", "biu_kp", "biu_kp_bbox"]:
        subset = group_scores.loc[group_scores["feature_set"] == feature_set].set_index("subgroup")
        if {"negative_clean", "negative_ambiguous"}.issubset(subset.index):
            lines.append(
                f"- `{feature_set}` ambiguous-vs-clean mean score delta: `{float(subset.loc['negative_ambiguous', 'mean_score'] - subset.loc['negative_clean', 'mean_score']):+.4f}`; "
                f"predicted-positive rate delta at `0.5`: `{float(subset.loc['negative_ambiguous', 'predicted_positive_rate_fixed_0p5'] - subset.loc['negative_clean', 'predicted_positive_rate_fixed_0p5']):+.4f}`."
            )

    lines.extend(["", "## Horizon Findings", ""])
    for note in HORIZON_SKIP_NOTES:
        lines.append(f"- Skipped `{note['dataset']}` horizon `{note['horizon_label']}`: {note['reason']}.")
    for horizon in ["60_90", "15_30", "5_15"]:
        metadata_path = JAAD_SPLIT_AUDIT_ROOT / f"tte_{horizon}" / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        alignment = metadata.get("keypoint_alignment", {})
        dropped_rows = int(alignment.get("dropped_rows", 0))
        dropped_tracks = int(alignment.get("dropped_tracks", 0))
        if dropped_rows > 0:
            lines.append(
                f"- JAAD `{horizon}` horizon audit was intersected with frozen keypoint coverage before KP/BIU feature building: dropped `{dropped_rows}` rows across `{dropped_tracks}` tracks."
            )

    jaad_selected = selected_df.loc[(selected_df["dataset"] == "JAAD") & (selected_df["threshold_rule"] == "f1_opt")].copy()
    pie_selected = selected_df.loc[(selected_df["dataset"] == "PIE") & (selected_df["threshold_rule"] == "f1_opt")].copy()
    jaad_fixed = fixed_df.loc[fixed_df["dataset"] == "JAAD"].copy()
    short_horizon = "5_15"
    far_horizon = "60_90"
    for feature_set in ["biu", "biu_bbox", "kp", "kp_bbox", "bbox"]:
        short_row = jaad_selected.loc[(jaad_selected["feature_set"] == feature_set) & (jaad_selected["horizon_label"] == short_horizon)]
        far_row = jaad_selected.loc[(jaad_selected["feature_set"] == feature_set) & (jaad_selected["horizon_label"] == far_horizon)]
        if not short_row.empty and not far_row.empty:
            lines.append(
                f"- JAAD `{feature_set}` test F1 from `{far_horizon}` to `{short_horizon}` under validation-F1 thresholds: "
                f"`{float(far_row.iloc[0]['test_f1']):.4f} -> {float(short_row.iloc[0]['test_f1']):.4f}` "
                f"(delta `{float(short_row.iloc[0]['test_f1'] - far_row.iloc[0]['test_f1']):+.4f}`)."
            )

    lines.extend(["", "## Answers", ""])
    bbox_group = group_scores.loc[group_scores["feature_set"] == "bbox"].set_index("subgroup")
    biu_group = group_scores.loc[group_scores["feature_set"] == "biu"].set_index("subgroup")
    if {"negative_clean", "negative_ambiguous"}.issubset(bbox_group.index):
        lines.append(
            f"1. Early BBox/context-like features identify crossing-relevant event-negative samples: ambiguous JAAD negatives score higher than clean negatives for `bbox` by `{float(bbox_group.loc['negative_ambiguous', 'mean_score'] - bbox_group.loc['negative_clean', 'mean_score']):+.4f}` in mean score."
        )
    else:
        lines.append("1. Early BBox/context-like feature evidence is inconclusive in the saved subgroup summaries.")
    lines.append(
        f"2. Ambiguous JAAD negatives are generally scored higher than clean negatives across the current overlap-0.8 selected models; see the subgroup deltas listed above."
    )
    biu_short = jaad_selected.loc[(jaad_selected["feature_set"] == "biu") & (jaad_selected["horizon_label"] == short_horizon), "test_f1"]
    biu_far = jaad_selected.loc[(jaad_selected["feature_set"] == "biu") & (jaad_selected["horizon_label"] == far_horizon), "test_f1"]
    if not biu_short.empty and not biu_far.empty:
        lines.append(
            f"3. BIU performance as horizon shortens: JAAD `biu` changes from `{float(biu_far.iloc[0]):.4f}` at `{far_horizon}` to `{float(biu_short.iloc[0]):.4f}` at `{short_horizon}` under the selected-model F1-threshold view."
        )
    biu_bbox_short = jaad_selected.loc[(jaad_selected["feature_set"] == "biu_bbox") & (jaad_selected["horizon_label"] == short_horizon), "test_f1"]
    biu_bbox_far = jaad_selected.loc[(jaad_selected["feature_set"] == "biu_bbox") & (jaad_selected["horizon_label"] == far_horizon), "test_f1"]
    if not biu_bbox_short.empty and not biu_bbox_far.empty:
        lines.append(
            f"4. `biu_bbox` at short vs long horizons on JAAD: `{float(biu_bbox_far.iloc[0]):.4f}` at `{far_horizon}` and `{float(biu_bbox_short.iloc[0]):.4f}` at `{short_horizon}`."
        )
    kp_short = jaad_selected.loc[(jaad_selected["feature_set"] == "kp") & (jaad_selected["horizon_label"] == short_horizon), "test_f1"]
    kp_far = jaad_selected.loc[(jaad_selected["feature_set"] == "kp") & (jaad_selected["horizon_label"] == far_horizon), "test_f1"]
    if not kp_short.empty and not kp_far.empty and not biu_short.empty and not biu_far.empty:
        lines.append(
            f"5. KP behaves differently from BIU if their horizon deltas diverge: JAAD `kp` changes `{float(kp_far.iloc[0]):.4f} -> {float(kp_short.iloc[0]):.4f}`, while `biu` changes `{float(biu_far.iloc[0]):.4f} -> {float(biu_short.iloc[0]):.4f}`."
        )
    clean_compare = clean_df.loc[
        (clean_df["threshold_rule"] == "fixed_0_5")
        & (clean_df["scope"] == "positives_vs_ambiguous_negatives")
        & (clean_df["horizon_label"] == "30_60")
        & (clean_df["feature_set"].isin(["bbox", "biu_bbox"]))
    ]
    if len(clean_compare) == 2:
        pivot = clean_compare.set_index("feature_set")
        lines.append(
            f"6. On current JAAD 30-60 under fixed `0.5`, `biu_bbox` vs `bbox` F1 against positives-plus-ambiguous-negatives is `{float(pivot.loc['biu_bbox', 'f1']):.4f}` vs `{float(pivot.loc['bbox', 'f1']):.4f}`."
        )
    clean_neg_fpr = clean_df.loc[
        (clean_df["threshold_rule"] == "fixed_0_5") & (clean_df["scope"] == "negative_clean_only") & (clean_df["horizon_label"] == "30_60")
    ][["feature_set", "fpr"]]
    amb_neg_fpr = clean_df.loc[
        (clean_df["threshold_rule"] == "fixed_0_5") & (clean_df["scope"] == "negative_ambiguous_only") & (clean_df["horizon_label"] == "30_60")
    ][["feature_set", "fpr"]]
    if not clean_neg_fpr.empty and not amb_neg_fpr.empty:
        merged_fpr = clean_neg_fpr.merge(amb_neg_fpr, on="feature_set", suffixes=("_clean", "_ambiguous"))
        lines.append(
            "7. JAAD high false-positive behavior is not only an ambiguous-negative issue; compare the clean-vs-ambiguous fixed-0.5 negative-only FPRs in `clean_vs_ambiguous_metrics_jaad.csv`."
        )
        best_examples = merged_fpr.sort_values("fpr_clean", ascending=False).head(2).to_dict(orient="records")
        for example in best_examples:
            lines.append(
                f"   Example `{example['feature_set']}` clean vs ambiguous FPR: `{float(example['fpr_clean']):.4f}` vs `{float(example['fpr_ambiguous']):.4f}`."
            )
    lines.append(
        "8. The staged interpretation is supported only if two patterns hold together: early geometry scores ambiguous event-negative samples above clean negatives, and BIU/BIU+BBox become relatively more informative as the horizon shrinks. The audit outputs should be read with that standard rather than as proof by assertion."
    )
    lines.append(
        "9. The next experiment should be chosen from the strength of those patterns. If subgroup separation is strong but clean-negative FPR stays high, context features are the next controlled addition. If short-horizon BIU gains are strong but noisy, confidence-aware BIUs are the next controlled addition. A true cascade should come only after those two audits are quantified."
    )

    _atomic_write_text("\n".join(lines) + "\n", FINAL_REPORT)


def _load_current_jaad_stage_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_df = _read_csv_optional(PER_SAMPLE_JAAD_OUTPUT)
    subgroup_df = _read_csv_optional(JAAD_SUBGROUP_OUTPUT)
    group_scores = _read_csv_optional(JAAD_SCORE_GROUP_OUTPUT)
    if group_scores.empty and not pred_df.empty and not subgroup_df.empty:
        group_scores = _group_score_rows(pred_df, subgroup_df)
        _atomic_write_df(group_scores, JAAD_SCORE_GROUP_OUTPUT)
    return pred_df, subgroup_df, group_scores


def _write_partial_report(
    *,
    datasets: list[str],
    horizons: list[dict[str, Any]],
    feature_sets: list[str],
    model_filter: set[str] | None,
    metrics_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    fixed_df: pd.DataFrame,
) -> None:
    models = MODEL_ORDER if model_filter is None else [name for name in MODEL_ORDER if name in model_filter]
    lines = [
        "# Staged Prediction Hypothesis Results Partial",
        "",
        "## Completion Status",
        "",
    ]

    failed_frames = {dataset: _read_csv_optional(_failed_path(dataset)) for dataset in datasets}
    for dataset in datasets:
        failed_df = failed_frames[dataset]
        lines.append(f"### {dataset}")
        lines.append("")
        for horizon in horizons:
            horizon_label = horizon["label"]
            infeasible = not failed_df.loc[
                (failed_df.get("horizon_label", pd.Series(dtype=str)).astype(str) == horizon_label)
                & (failed_df.get("status", pd.Series(dtype=str)).astype(str) == "infeasible")
            ].empty
            if infeasible:
                reason = failed_df.loc[
                    (failed_df["horizon_label"].astype(str) == horizon_label) & (failed_df["status"].astype(str) == "infeasible"),
                    "reason",
                ].iloc[-1]
                lines.append(f"- `{horizon_label}`: infeasible, `{reason}`.")
                continue
            expected_metrics = len(feature_sets) * len(models)
            got_metrics = len(_filter_df(metrics_df, dataset=dataset, horizon_label=horizon_label))
            expected_selected = len(feature_sets) * 2
            got_selected = len(_filter_df(selected_df, dataset=dataset, horizon_label=horizon_label))
            expected_fixed = len(feature_sets)
            got_fixed = len(_filter_df(fixed_df, dataset=dataset, horizon_label=horizon_label))
            status = "completed" if (got_metrics >= expected_metrics and got_selected >= expected_selected and got_fixed >= expected_fixed) else "incomplete"
            lines.append(
                f"- `{horizon_label}`: {status}; metrics `{got_metrics}/{expected_metrics}`, selected `{got_selected}/{expected_selected}`, fixed `{got_fixed}/{expected_fixed}`."
            )
        lines.append("")

    lines.extend(["## Failed Combinations", ""])
    any_failures = False
    for dataset in datasets:
        failed_df = failed_frames[dataset]
        failed_df = failed_df.loc[failed_df.get("status", pd.Series(dtype=str)).astype(str) == "failed"].copy()
        if failed_df.empty:
            continue
        any_failures = True
        lines.append(f"### {dataset}")
        lines.append("")
        lines.append("| horizon | feature_set | model | stage | reason |")
        lines.append("|---|---|---|---|---|")
        for row in failed_df.tail(20).to_dict(orient="records"):
            lines.append(
                f"| {row.get('horizon_label','')} | {row.get('feature_set','')} | {row.get('model_name','')} | {row.get('stage','')} | {str(row.get('reason','')).replace('|', '/')} |"
            )
        lines.append("")
    if not any_failures:
        lines.append("- No failed combinations are currently recorded.")
        lines.append("")

    jaad_ready = not _filter_df(selected_df, dataset="JAAD", horizon_label="30_60").empty and not _read_csv_optional(_clean_checkpoint_path()).empty
    pie_ready = not _filter_df(selected_df, dataset="PIE", horizon_label="30_60").empty
    lines.extend(["## Interpretation Readiness", ""])
    lines.append(f"- JAAD staged audit is {'ready for interpretation' if jaad_ready else 'not ready for interpretation yet'}.")
    lines.append(f"- PIE horizon audit is {'ready for interpretation' if pie_ready else 'not ready for interpretation yet'}.")
    if jaad_ready and not pie_ready:
        lines.append("- A JAAD-only staged reading is possible, but the cross-dataset staged hypothesis is still incomplete.")
    elif not jaad_ready and not pie_ready:
        lines.append("- The staged hypothesis should not be interpreted yet from checkpoints alone.")
    else:
        lines.append("- The staged hypothesis can be interpreted only for the completed portions listed above.")

    _atomic_write_text("\n".join(lines) + "\n", PARTIAL_REPORT)


def finalize_outputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    datasets = _selected_datasets(args)
    horizons = _selected_horizons(args)
    feature_sets = _selected_feature_sets(args)
    model_filter = _selected_models(args)

    all_metrics: list[pd.DataFrame] = []
    all_selected: list[pd.DataFrame] = []
    all_fixed: list[pd.DataFrame] = []
    for dataset in datasets:
        metrics_df, selected_df, fixed_df = _materialize_dataset_outputs(dataset)
        all_metrics.append(metrics_df)
        all_selected.append(selected_df)
        all_fixed.append(fixed_df)
    clean_df = _materialize_jaad_clean_outputs() if "JAAD" in datasets else pd.DataFrame()

    metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    selected_df = pd.concat(all_selected, ignore_index=True) if all_selected else pd.DataFrame()
    fixed_df = pd.concat(all_fixed, ignore_index=True) if all_fixed else pd.DataFrame()

    pred_df, subgroup_df, group_scores = _load_current_jaad_stage_inputs()
    complete = True
    for dataset in datasets:
        failed_df = _read_csv_optional(_failed_path(dataset))
        if not failed_df.loc[failed_df.get("status", pd.Series(dtype=str)).astype(str) == "failed"].empty:
            complete = False
    models = MODEL_ORDER if model_filter is None else [name for name in MODEL_ORDER if name in model_filter]
    for dataset in datasets:
        failed_df = _read_csv_optional(_failed_path(dataset))
        infeasible_horizons = set(
            failed_df.loc[failed_df.get("status", pd.Series(dtype=str)).astype(str) == "infeasible", "horizon_label"].astype(str)
        ) if not failed_df.empty else set()
        for horizon in horizons:
            if dataset == "JAAD" and horizon["label"] in infeasible_horizons:
                continue
            expected_metrics = len(feature_sets) * len(models)
            expected_selected = len(feature_sets) * 2
            expected_fixed = len(feature_sets)
            if len(_filter_df(metrics_df, dataset=dataset, horizon_label=horizon["label"])) < expected_metrics:
                complete = False
            if len(_filter_df(selected_df, dataset=dataset, horizon_label=horizon["label"])) < expected_selected:
                complete = False
            if len(_filter_df(fixed_df, dataset=dataset, horizon_label=horizon["label"])) < expected_fixed:
                complete = False

    if complete and not group_scores.empty and not subgroup_df.empty:
        _write_final_report(group_scores, subgroup_df, selected_df, fixed_df, clean_df)
    else:
        _write_partial_report(
            datasets=datasets,
            horizons=horizons,
            feature_sets=feature_sets,
            model_filter=model_filter,
            metrics_df=metrics_df,
            selected_df=selected_df,
            fixed_df=fixed_df,
        )
    return metrics_df, selected_df, fixed_df, clean_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkpointed staged-prediction audit.")
    parser.add_argument("--finalize-only", action="store_true", help="Finalize outputs from existing checkpoints without fitting models.")
    parser.add_argument("--force", action="store_true", help="Refit and rewrite checkpoint rows even if they already exist.")
    parser.add_argument("--skip-pie", action="store_true", help="Skip PIE horizon work.")
    parser.add_argument("--skip-jaad", action="store_true", help="Skip JAAD horizon work.")
    parser.add_argument("--horizons", nargs="*", choices=[h["label"] for h in HORIZONS], help="Restrict to specific horizon labels.")
    parser.add_argument("--feature-sets", nargs="*", choices=FEATURE_SETS, help="Restrict to specific feature sets.")
    parser.add_argument("--models", nargs="*", choices=MODEL_ORDER, help="Restrict to specific model names.")
    parser.add_argument("--jaad-keypoint-jsonl", type=Path, default=JAAD_KEYPOINT_JSONL, help="JAAD keypoint JSONL to use for non-30_60 horizon rebuilds.")
    parser.add_argument("--skip-current-jaad-audits", action="store_true", help="Skip rebuilding current JAAD 30-60 per-sample/subgroup audit artifacts before horizon evaluation.")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    global JAAD_KEYPOINT_JSONL
    JAAD_KEYPOINT_JSONL = args.jaad_keypoint_jsonl
    if not args.finalize_only:
        if not args.skip_current_jaad_audits:
            pred_df = build_current_jaad_per_sample_predictions()
            subgroup_df = build_current_jaad_subgroups()
            group_scores = _group_score_rows(pred_df, subgroup_df)
            _atomic_write_df(group_scores, JAAD_SCORE_GROUP_OUTPUT)
            _write_jaad_group_report(subgroup_df)
            _write_jaad_stage_report(group_scores, pred_df, subgroup_df)
        run_horizon_audit(args)
    finalize_outputs(args)


if __name__ == "__main__":
    run(_parse_args())
