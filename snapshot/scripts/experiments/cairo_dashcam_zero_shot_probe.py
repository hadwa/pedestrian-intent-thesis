from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

CLEAN_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean")
ARCHIVE_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation")
SCRIPTS_DIR = CLEAN_ROOT / "scripts/experiments"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import jaad_biu_v2_benchmark_comparison as base
from jaad_corrected_15_30_rebuild import DATA_DIR as CORRECTED_DATA_DIR
from jaad_corrected_15_30_rebuild import FEAT_DIR as CORRECTED_FEAT_DIR

OUT_ROOT = CLEAN_ROOT / "experiments/029_cairo_zero_shot_probe"
METRICS_DIR = OUT_ROOT / "metrics"
PRED_DIR = OUT_ROOT / "predictions"
REPORT_PATH = OUT_ROOT / "README.md"
LABEL_SHEET = CLEAN_ROOT / "experiments/028_cairo_dashcam_case_study/labeling_sheet.csv"
TRACE_ROOT = CLEAN_ROOT / "experiments/028_cairo_dashcam_case_study/traces"
TRACK_ROOT = CLEAN_ROOT / "experiments/028_cairo_dashcam_case_study/tracks"
DICT_PATH = CLEAN_ROOT / "experiments/008_biu_v2_design_spec/metrics/biu_v2_feature_dictionary.csv"
RESULTS_021 = CLEAN_ROOT / "experiments/021_jaad_corrected_15_30_rebuild/metrics/corrected_staged_results.csv"

SEEDS = (42, 43, 44)
HORIZON_RULES = {
    "30_60": {"min": 30, "max": 60},
    "15_30_corrected": {"min": 15, "max": 30},
}


@dataclass(frozen=True)
class StrategySpec:
    name: str
    horizon: str | None
    feature_set: str | None
    model_name: str | None
    threshold: float | None
    stage1_threshold: float | None = None
    stage2_horizon: str | None = None
    stage2_feature_set: str | None = None
    stage2_model_name: str | None = None
    stage2_threshold: float | None = None


def read_labeling_sheet(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    header = rows[0]
    expected = len(header)
    repaired = []
    for row in rows[1:]:
        if len(row) > expected:
            row = row[: expected - 1] + [",".join(row[expected - 1 :])]
        elif len(row) < expected:
            row = row + [""] * (expected - len(row))
        repaired.append(row)
    frame = pd.DataFrame(repaired, columns=header)
    if "hesitation_observed" in frame.columns and "notes" in frame.columns:
        repaired_notes = []
        repaired_hesitation = []
        for hesitation, notes in zip(frame["hesitation_observed"], frame["notes"]):
            hesitation_text = _normalize_label(hesitation)
            notes_text = _normalize_text(notes)
            if hesitation_text not in {"", "yes", "no"}:
                merged = ",".join(part for part in [str(hesitation).strip(), notes_text] if part)
                repaired_notes.append(merged)
                repaired_hesitation.append("")
            else:
                repaired_notes.append(notes_text)
                repaired_hesitation.append(hesitation_text)
        frame["hesitation_observed"] = repaired_hesitation
        frame["notes"] = repaired_notes
    return frame


def _normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _normalize_label(value: Any) -> str:
    return _normalize_text(value).lower()


def prepare_track_labels(frame: pd.DataFrame) -> pd.DataFrame:
    def parse_int_field(record: dict[str, Any], field: str) -> int | None:
        raw = _normalize_text(record.get(field))
        return int(float(raw)) if raw else None

    rows = []
    for record in frame.to_dict(orient="records"):
        label_text = _normalize_label(record.get("crossing_label"))
        notes = _normalize_label(record.get("notes"))
        combined = ",".join(x for x in [label_text, notes] if x)
        if not label_text and not notes:
            continue
        if "not_pedestrian" in combined:
            continue
        if "duplicate_of:" in combined or "duplicate-of:" in combined:
            continue
        if "id_switch" in combined:
            continue
        if label_text not in {"yes", "no", "ambiguous"}:
            continue
        onset_raw = _normalize_text(record.get("crossing_onset_frame"))
        onset_frame = int(float(onset_raw)) if onset_raw else None
        binary_label: int | None
        if label_text == "yes":
            binary_label = 1
        elif label_text == "no":
            binary_label = 0
        else:
            binary_label = None
        rows.append(
            {
                "clip_id": _normalize_text(record["clip_id"]),
                "track_id": int(float(record["track_id"])),
                "first_frame": parse_int_field(record, "first_frame"),
                "last_frame": parse_int_field(record, "last_frame"),
                "label_group": label_text,
                "binary_label": binary_label,
                "crossing_onset_frame": onset_frame,
                "notes": notes,
            }
        )
    out = pd.DataFrame(rows).sort_values(["clip_id", "track_id"]).reset_index(drop=True)
    if not out.empty:
        out["binary_label"] = out["binary_label"].astype(object)
        out.loc[out["label_group"] == "ambiguous", "binary_label"] = None
    return out


def select_track_window(
    trace_df: pd.DataFrame,
    label_group: str,
    onset_frame: int | None,
    horizon_min: int,
    horizon_max: int,
    anchor_frame: int | None = None,
) -> pd.Series | None:
    sort_cols = [c for c in ["window_end_frame", "window_start_frame"] if c in trace_df.columns]
    ordered = trace_df.sort_values(sort_cols).reset_index(drop=True)
    if label_group == "yes" and onset_frame is not None:
        anchor = onset_frame
    elif anchor_frame is not None:
        anchor = anchor_frame
    else:
        anchor = None
    if anchor is None:
        return ordered.iloc[-1]
    deltas = anchor - ordered["window_end_frame"].astype(int)
    eligible = ordered[(deltas >= horizon_min) & (deltas <= horizon_max)].copy()
    if eligible.empty:
        return None
    eligible["tte_delta"] = anchor - eligible["window_end_frame"].astype(int)
    eligible_sort_cols = ["tte_delta"] + (["window_start_frame"] if "window_start_frame" in eligible.columns else [])
    eligible_ascending = [True] + ([False] if "window_start_frame" in eligible.columns else [])
    eligible = eligible.sort_values(eligible_sort_cols, ascending=eligible_ascending)
    return eligible.iloc[0]


def _read_split_feature_frame(feature_dir: Path, name: str) -> pd.DataFrame:
    parts = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(feature_dir / f"{split}_{name}.csv")
        df["split"] = split
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def _merge_base_and_features(base_df: pd.DataFrame, feat_df: pd.DataFrame) -> pd.DataFrame:
    keep = ["sequence_id"] + [c for c in feat_df.columns if c != "sequence_id" and c not in base_df.columns]
    return base_df.merge(feat_df[keep], on="sequence_id", how="inner")


def _numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in base.META_COLS and pd.api.types.is_numeric_dtype(df[c])]


def _select_prefixed_columns(df: pd.DataFrame, prefixes: list[str]) -> list[str]:
    selected = []
    for col in df.columns:
        if col in base.META_COLS:
            continue
        for prefix in prefixes:
            if col == prefix or col.startswith(prefix + "_"):
                selected.append(col)
                break
    return sorted(selected)


def load_cairo_feature_specs() -> dict[str, tuple[pd.DataFrame, list[str]]]:
    feature_dict = pd.read_csv(DICT_PATH)
    anatomical_prefixes = feature_dict.loc[feature_dict["group"].isin(["original_anatomical_biu"]), "feature_name"].tolist()
    lower_prefixes = feature_dict.loc[feature_dict["group"].isin(["lower_body_commitment_biu"]), "feature_name"].tolist()
    torso_prefixes = feature_dict.loc[feature_dict["group"].isin(["torso_center_commitment_biu"]), "feature_name"].tolist()

    base_rows = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(CORRECTED_DATA_DIR / f"{split}.csv")
        df["split"] = split
        df["horizon_label"] = "15_30_corrected"
        df["group"] = df.apply(base._classify_group, axis=1)
        base_rows.append(df)
    corrected_base = pd.concat(base_rows, ignore_index=True)

    bbox_df = _merge_base_and_features(corrected_base, _read_split_feature_frame(CORRECTED_FEAT_DIR, "bbox"))
    v2_df = _merge_base_and_features(corrected_base, _read_split_feature_frame(CORRECTED_FEAT_DIR, "biu_v2"))

    bbox_cols = _numeric_feature_cols(bbox_df)
    v2_anatomical_cols = _select_prefixed_columns(v2_df, anatomical_prefixes)
    v2_lower_cols = _select_prefixed_columns(v2_df, lower_prefixes)
    v2_torso_cols = _select_prefixed_columns(v2_df, torso_prefixes)
    v2_core_cols = sorted(set(v2_anatomical_cols + v2_lower_cols + v2_torso_cols))

    corrected_v2bbox = v2_df[[c for c in v2_df.columns if c not in bbox_cols]].merge(
        bbox_df[["sequence_id"] + bbox_cols], on="sequence_id", how="inner"
    )
    corrected_v2bbox_cols = sorted(set(v2_core_cols + bbox_cols))

    bbox30 = _read_split_feature_frame(ARCHIVE_ROOT / "data/jaad_features_event_16_30_60_overlap08_default", "bbox")
    bbox30_cols = _numeric_feature_cols(bbox30)

    return {
        "bbox_only_30_60": (bbox30, bbox30_cols),
        "biu_v2_core_plus_bbox_15_30_corrected": (corrected_v2bbox, corrected_v2bbox_cols),
    }


def _compute_bbox_features(track_window: pd.DataFrame) -> dict[str, float]:
    ordered = track_window.sort_values("frame_idx").copy()
    frame_range = np.arange(int(ordered["frame_idx"].min()), int(ordered["frame_idx"].max()) + 1)
    ordered = ordered.drop_duplicates("frame_idx").set_index("frame_idx").reindex(frame_range)
    ordered[["x1", "y1", "x2", "y2", "conf"]] = ordered[["x1", "y1", "x2", "y2", "conf"]].interpolate(
        method="linear", limit_direction="both"
    )
    center_x = (ordered["x1"] + ordered["x2"]) / 2.0
    center_y = (ordered["y1"] + ordered["y2"]) / 2.0
    width = ordered["x2"] - ordered["x1"]
    height = ordered["y2"] - ordered["y1"]
    area = width * height
    aspect = width / height.replace(0, np.nan)
    dx = center_x.diff().fillna(0.0)
    dy = center_y.diff().fillna(0.0)
    return {
        "bbox_center_x_mean": float(center_x.mean()),
        "bbox_center_x_std": float(center_x.std(ddof=0)),
        "bbox_center_x_first": float(center_x.iloc[0]),
        "bbox_center_x_last": float(center_x.iloc[-1]),
        "bbox_center_x_delta": float(center_x.iloc[-1] - center_x.iloc[0]),
        "bbox_center_x_min": float(center_x.min()),
        "bbox_center_x_max": float(center_x.max()),
        "bbox_center_y_mean": float(center_y.mean()),
        "bbox_center_y_std": float(center_y.std(ddof=0)),
        "bbox_center_y_first": float(center_y.iloc[0]),
        "bbox_center_y_last": float(center_y.iloc[-1]),
        "bbox_center_y_delta": float(center_y.iloc[-1] - center_y.iloc[0]),
        "bbox_center_y_min": float(center_y.min()),
        "bbox_center_y_max": float(center_y.max()),
        "bbox_width_mean": float(width.mean()),
        "bbox_width_std": float(width.std(ddof=0)),
        "bbox_width_first": float(width.iloc[0]),
        "bbox_width_last": float(width.iloc[-1]),
        "bbox_width_delta": float(width.iloc[-1] - width.iloc[0]),
        "bbox_width_min": float(width.min()),
        "bbox_width_max": float(width.max()),
        "bbox_height_mean": float(height.mean()),
        "bbox_height_std": float(height.std(ddof=0)),
        "bbox_height_first": float(height.iloc[0]),
        "bbox_height_last": float(height.iloc[-1]),
        "bbox_height_delta": float(height.iloc[-1] - height.iloc[0]),
        "bbox_height_min": float(height.min()),
        "bbox_height_max": float(height.max()),
        "bbox_area_mean": float(area.mean()),
        "bbox_area_std": float(area.std(ddof=0)),
        "bbox_area_first": float(area.iloc[0]),
        "bbox_area_last": float(area.iloc[-1]),
        "bbox_area_delta": float(area.iloc[-1] - area.iloc[0]),
        "bbox_area_min": float(area.min()),
        "bbox_area_max": float(area.max()),
        "bbox_aspect_ratio_mean": float(aspect.mean()),
        "bbox_aspect_ratio_std": float(aspect.std(ddof=0)),
        "bbox_aspect_ratio_first": float(aspect.iloc[0]),
        "bbox_aspect_ratio_last": float(aspect.iloc[-1]),
        "bbox_aspect_ratio_delta": float(aspect.iloc[-1] - aspect.iloc[0]),
        "bbox_aspect_ratio_min": float(aspect.min()),
        "bbox_aspect_ratio_max": float(aspect.max()),
        "center_x_velocity_mean": float(dx.mean()),
        "center_y_velocity_mean": float(dy.mean()),
        "center_x_velocity_std": float(dx.std(ddof=0)),
        "center_y_velocity_std": float(dy.std(ddof=0)),
        "center_x_displacement": float(center_x.iloc[-1] - center_x.iloc[0]),
        "center_y_displacement": float(center_y.iloc[-1] - center_y.iloc[0]),
        "bbox_area_change": float(area.iloc[-1] - area.iloc[0]),
        "bbox_width_change": float(width.iloc[-1] - width.iloc[0]),
        "bbox_height_change": float(height.iloc[-1] - height.iloc[0]),
        "bbox_scale_change_ratio": float(area.iloc[-1] / area.iloc[0]) if float(area.iloc[0]) != 0.0 else float("nan"),
        "bbox_valid_frame_count": int(len(ordered)),
    }


def _build_cairo_row(label_row: pd.Series, trace_row: pd.Series, bbox_features: dict[str, float]) -> dict[str, Any]:
    row = {k: trace_row[k] for k in trace_row.index}
    row.update(bbox_features)
    row.update(
        {
            "clip_id": label_row["clip_id"],
            "track_id": label_row["track_id"],
            "label_group": label_row["label_group"],
            "binary_label": label_row["binary_label"],
            "crossing_onset_frame": label_row["crossing_onset_frame"],
            "first_frame": label_row["first_frame"],
            "last_frame": label_row["last_frame"],
        }
    )
    return row


def build_cairo_feature_rows(labels: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    trace_cache: dict[tuple[str, int], pd.DataFrame] = {}
    track_cache: dict[tuple[str, int], pd.DataFrame] = {}
    rows_by_horizon = {"30_60": [], "15_30_corrected": []}
    audit_rows = []

    for clip_id in labels["clip_id"].unique():
        track_path = TRACK_ROOT / f"{clip_id}_tracks.csv"
        if not track_path.exists():
            track_path = TRACK_ROOT / f"{clip_id}_tracks_raw.csv"
        if not track_path.exists():
            continue
        track_df = pd.read_csv(track_path)
        for track_id, sub in track_df.groupby("track_id"):
            track_cache[(clip_id, int(track_id))] = sub.copy()

    for record in labels.to_dict(orient="records"):
        key = (record["clip_id"], int(record["track_id"]))
        trace_path = TRACE_ROOT / record["clip_id"] / f"track_{int(record['track_id']):04d}_trace.csv"
        trace_df = pd.read_csv(trace_path) if trace_path.exists() else None
        track_df = track_cache.get(key)
        for horizon, cfg in HORIZON_RULES.items():
            trace_row = None if trace_df is None else select_track_window(
                trace_df,
                label_group=record["label_group"],
                onset_frame=record["crossing_onset_frame"],
                horizon_min=cfg["min"],
                horizon_max=cfg["max"],
                anchor_frame=record["last_frame"],
            )
            status = "available"
            if trace_df is None:
                status = "missing_trace_file"
            elif trace_row is None:
                status = "no_valid_window"
            elif track_df is None:
                status = "missing_track_boxes"
            else:
                start_frame = int(trace_row["window_start_frame"])
                end_frame = int(trace_row["window_end_frame"])
                window_boxes = track_df[(track_df["frame_idx"] >= start_frame) & (track_df["frame_idx"] <= end_frame)].copy()
                if window_boxes.empty:
                    status = "no_track_boxes_for_window"
                else:
                    bbox_features = _compute_bbox_features(window_boxes)
                    rows_by_horizon[horizon].append(_build_cairo_row(pd.Series(record), trace_row, bbox_features))
            audit_rows.append(
                {
                    "clip_id": record["clip_id"],
                    "track_id": record["track_id"],
                    "label_group": record["label_group"],
                    "horizon_label": horizon,
                    "status": status,
                }
            )
    frames = {k: pd.DataFrame(v) for k, v in rows_by_horizon.items()}
    return frames, pd.DataFrame(audit_rows)


def _score_with_ensemble(train_df: pd.DataFrame, feature_cols: list[str], model_name: str, cairo_df: pd.DataFrame) -> np.ndarray:
    X_train = train_df[feature_cols].astype(float)
    y_train = train_df["label"].astype(int).to_numpy()
    X_cairo = cairo_df.reindex(columns=feature_cols).astype(float)
    probs = []
    for seed in SEEDS:
        pipe = base._build_model(model_name, seed, y_train)
        if pipe is None:
            raise RuntimeError(f"model {model_name} is unavailable in this environment")
        pipe.fit(X_train, y_train)
        probs.append(pipe.predict_proba(X_cairo)[:, 1])
    return np.mean(np.vstack(probs), axis=0)


def build_strategy_specs() -> list[StrategySpec]:
    results = pd.read_csv(RESULTS_021)
    bbox = results[results["config"] == "single|bbox_only|30_60"].iloc[0]
    v2bbox = results[results["config"] == "single|biu_v2_core_plus_bbox|15_30_corrected"].iloc[0]
    staged = results[results["config"] == "staged|bbox30_60->biu_v2_core_plus_bbox@15_30_corrected"].iloc[0]
    return [
        StrategySpec(
            name="bbox_only_30_60",
            horizon="30_60",
            feature_set="bbox_only",
            model_name=str(bbox["model"]),
            threshold=float(bbox["t1"]),
        ),
        StrategySpec(
            name="biu_v2_core_plus_bbox_15_30_corrected",
            horizon="15_30_corrected",
            feature_set="biu_v2_core_plus_bbox",
            model_name=str(v2bbox["model"]),
            threshold=float(v2bbox["t1"]),
        ),
        StrategySpec(
            name="staged_bbox30_60_to_biu_v2_core_plus_bbox_15_30_corrected",
            horizon=None,
            feature_set=None,
            model_name=None,
            threshold=None,
            stage1_threshold=float(staged["t1"]),
            stage2_horizon="15_30_corrected",
            stage2_feature_set="biu_v2_core_plus_bbox",
            stage2_model_name="logreg",
            stage2_threshold=float(staged["t2"]),
        ),
    ]


def _safe_metric(y_true: np.ndarray, y_pred: np.ndarray, fn) -> float:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(fn(y_true, y_pred, zero_division=0))


def evaluate_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, sub in pred_df.groupby("strategy"):
        main = sub[sub["binary_label"].notna()].copy()
        amb = sub[sub["label_group"] == "ambiguous"].copy()
        y = main["binary_label"].astype(int).to_numpy()
        pred = main["pred"].astype(int).to_numpy()
        rows.append(
            {
                "strategy": strategy,
                "n_scored_main": int(len(main)),
                "n_yes_main": int((main["binary_label"] == 1).sum()),
                "n_no_main": int((main["binary_label"] == 0).sum()),
                "n_ambiguous_probe": int(len(amb)),
                "f1": _safe_metric(y, pred, f1_score),
                "precision": _safe_metric(y, pred, precision_score),
                "recall": _safe_metric(y, pred, recall_score),
                "ambiguous_fire_rate": float(amb["pred"].mean()) if len(amb) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("strategy").reset_index(drop=True)


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    labels_raw = read_labeling_sheet(LABEL_SHEET)
    labels = prepare_track_labels(labels_raw)
    labels.to_csv(METRICS_DIR / "prepared_labels.csv", index=False)

    cairo_rows, audit = build_cairo_feature_rows(labels)
    audit.to_csv(METRICS_DIR / "window_selection_audit.csv", index=False)

    feature_specs = load_cairo_feature_specs()
    bbox_train, bbox_cols = feature_specs["bbox_only_30_60"]
    v2bbox_train, v2bbox_cols = feature_specs["biu_v2_core_plus_bbox_15_30_corrected"]

    bbox_cairo = cairo_rows["30_60"].copy()
    v2bbox_cairo = cairo_rows["15_30_corrected"].copy()
    if bbox_cairo.empty and v2bbox_cairo.empty:
        raise RuntimeError("no Cairo feature rows were available for scoring")

    if not bbox_cairo.empty:
        bbox_cairo["prob"] = _score_with_ensemble(bbox_train, bbox_cols, "logreg", bbox_cairo)
    if not v2bbox_cairo.empty:
        v2bbox_cairo["prob"] = _score_with_ensemble(v2bbox_train, v2bbox_cols, "logreg", v2bbox_cairo)

    bbox_cairo.to_csv(PRED_DIR / "bbox_only_30_60_track_rows.csv", index=False)
    v2bbox_cairo.to_csv(PRED_DIR / "biu_v2_core_plus_bbox_15_30_corrected_track_rows.csv", index=False)

    strategies = build_strategy_specs()
    pred_rows = []

    bbox_lookup = bbox_cairo.set_index(["clip_id", "track_id"]) if not bbox_cairo.empty else pd.DataFrame()
    v2bbox_lookup = v2bbox_cairo.set_index(["clip_id", "track_id"]) if not v2bbox_cairo.empty else pd.DataFrame()

    for spec in strategies:
        if spec.name == "bbox_only_30_60":
            for _, row in bbox_cairo.iterrows():
                pred_rows.append(
                    {
                        "strategy": spec.name,
                        "clip_id": row["clip_id"],
                        "track_id": int(row["track_id"]),
                        "label_group": row["label_group"],
                        "binary_label": row["binary_label"],
                        "prob": float(row["prob"]),
                        "pred": int(float(row["prob"]) >= float(spec.threshold)),
                    }
                )
        elif spec.name == "biu_v2_core_plus_bbox_15_30_corrected":
            for _, row in v2bbox_cairo.iterrows():
                pred_rows.append(
                    {
                        "strategy": spec.name,
                        "clip_id": row["clip_id"],
                        "track_id": int(row["track_id"]),
                        "label_group": row["label_group"],
                        "binary_label": row["binary_label"],
                        "prob": float(row["prob"]),
                        "pred": int(float(row["prob"]) >= float(spec.threshold)),
                    }
                )
        else:
            common_keys = bbox_lookup.index
            for key in common_keys:
                b_row = bbox_lookup.loc[key]
                if isinstance(b_row, pd.DataFrame):
                    b_row = b_row.iloc[0]
                if key in v2bbox_lookup.index:
                    s2_row = v2bbox_lookup.loc[key]
                    if isinstance(s2_row, pd.DataFrame):
                        s2_row = s2_row.iloc[0]
                    s2_pred = int(float(s2_row["prob"]) >= float(spec.stage2_threshold))
                    s2_prob = float(s2_row["prob"])
                else:
                    s2_pred = 1
                    s2_prob = float("nan")
                s1_pred = int(float(b_row["prob"]) >= float(spec.stage1_threshold))
                pred_rows.append(
                    {
                        "strategy": spec.name,
                        "clip_id": key[0],
                        "track_id": int(key[1]),
                        "label_group": b_row["label_group"],
                        "binary_label": b_row["binary_label"],
                        "prob": float(b_row["prob"]),
                        "stage2_prob": s2_prob,
                        "pred": int(s1_pred and s2_pred),
                    }
                )

    pred_df = pd.DataFrame(pred_rows).sort_values(["strategy", "clip_id", "track_id"]).reset_index(drop=True)
    pred_df.to_csv(PRED_DIR / "track_predictions.csv", index=False)

    metrics = evaluate_predictions(pred_df)
    metrics.to_csv(METRICS_DIR / "metrics_summary.csv", index=False)

    label_counts = labels["label_group"].value_counts(dropna=False).rename_axis("label_group").reset_index(name="count")
    label_counts.to_csv(METRICS_DIR / "label_counts.csv", index=False)

    files_used = [
        str(LABEL_SHEET),
        str(TRACE_ROOT),
        str(TRACK_ROOT),
        str(ARCHIVE_ROOT / "data/jaad_features_event_16_30_60_overlap08_default"),
        str(CORRECTED_DATA_DIR),
        str(CORRECTED_FEAT_DIR),
        str(DICT_PATH),
        str(RESULTS_021),
    ]
    (OUT_ROOT / "files_used.txt").write_text("\n".join(files_used) + "\n", encoding="utf-8")
    (OUT_ROOT / "commands_run.txt").write_text(
        "python3 scripts/experiments/cairo_dashcam_zero_shot_probe.py\n",
        encoding="utf-8",
    )

    report = [
        "# Experiment 029 — Cairo dashcam zero-shot probe",
        "",
        "Scored the labeled Cairo subset using frozen JAAD-derived strategies with no Cairo tuning.",
        "",
        "Strategies:",
        "- `bbox_only_30_60` using the corrected exp-021/011 val-F1 threshold.",
        "- `biu_v2_core_plus_bbox_15_30_corrected` using the corrected exp-021 val-F1 threshold.",
        "- `staged_bbox30_60_to_biu_v2_core_plus_bbox_15_30_corrected` using the corrected exp-021 staged thresholds.",
        "",
        "Caveat: the bbox family is transferred cross-resolution from JAAD to Cairo, so absolute bbox-scale features are not directly comparable. Results are diagnostic zero-shot evidence, not a benchmark.",
        "",
        "Outputs:",
        "- `metrics/metrics_summary.csv`",
        "- `metrics/window_selection_audit.csv`",
        "- `predictions/track_predictions.csv`",
        "",
        metrics.to_string(index=False),
        "",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
