"""Audit JAAD API-style protocol sensitivity against the clean 30_60 benchmark.

Protocol A:
    Documented reconstruction of the upstream JAAD API-style flawed code path
    used by Noor's pipeline lineage.

Protocol B:
    Clean event-aligned anticipatory JAAD benchmark.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CLEAN_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean")
ARCHIVE_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation")
DATASET_ROOT = Path("/home/abdelrahman/Hadwa/datasets/JAAD")
UPSTREAM_JAAD_PATH = Path("/tmp/jaad-upstream-audit/jaad_data.py")
UPSTREAM_CACHE_DIR = Path("/tmp/jaad_upstream_cache_exp031")

OUT_ROOT = CLEAN_ROOT / "experiments/031_jaad_api_label_protocol_sensitivity"
METRICS_DIR = OUT_ROOT / "metrics"
PROVENANCE_DIR = OUT_ROOT / "provenance"
DOC_PATH = CLEAN_ROOT / "docs/JAAD_API_PROTOCOL_SENSITIVITY.md"

CLEAN_SPLIT_DIR = ARCHIVE_ROOT / "data/jaad_splits_event_16_30_60_overlap08_default"
CLEAN_FEATURE_DIR = ARCHIVE_ROOT / "data/jaad_features_event_16_30_60_overlap08_default"
KEYPOINT_JSONL = ARCHIVE_ROOT / "data/jaad_keypoints_vitpose_s_extended_short_horizon.jsonl"

NOOR_REPO_COMMIT = "65ecd1249797ca82806c3d373fec73f5f47b106c"
UPSTREAM_JAAD_COMMIT = "d2feeac998366e47ecc54e5533c8a3fa61141dda"
UPSTREAM_MECHANISM_COMMIT = "a04cb8511b85413bbddafd049911da1b8c3c214e"

OBS_LEN = 16
TTE_MIN = 30
TTE_MAX = 60
OVERLAP = 0.8
STRIDE = max(1, int((1.0 - OVERLAP) * OBS_LEN))
SPLITS = ("train", "val", "test")
PROTOCOL_A = "api_style_reconstruction"
PROTOCOL_B = "clean_event_aligned"
REPRESENTATION = "bbox"
META_COLUMNS = [
    "sequence_id",
    "split",
    "label",
    "set_id",
    "video_id",
    "ped_id",
    "track_id",
    "start_frame",
    "end_frame",
]


@dataclass
class ProtocolResult:
    summary_rows: pd.DataFrame
    prediction_rows: pd.DataFrame
    candidate_rows: pd.DataFrame


@dataclass
class UpstreamTrackRecord:
    video_id: str
    ped_id: str
    track_id: str
    frames: list[int]
    crosses: list[int]
    frame_to_cross: dict[int, int]
    crossing_attr: int | None
    crossing_point: int | None
    decision_point: int | None
    track_length: int
    transition_frames: list[int]
    starts_already_crossing: bool
    source_annotation_path: str
    source_attributes_path: str


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_upstream_jaad_class():
    if "cv2" not in sys.modules:
        sys.modules["cv2"] = types.ModuleType("cv2")
    mod = _load_module("exp031_upstream_jaad", UPSTREAM_JAAD_PATH)
    UPSTREAM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mod.JAAD.cache_path = property(lambda self: str(UPSTREAM_CACHE_DIR))
    return mod.JAAD


ARCHIVE_BBOX = _load_module(
    "exp031_archive_bbox_features",
    ARCHIVE_ROOT / "src/features/bbox_features.py",
)
BENCHMARK = _load_module(
    "exp031_benchmark_backbone",
    CLEAN_ROOT / "scripts/experiments/jaad_biu_v2_benchmark_comparison.py",
)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


def _json_loads(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if pd.isna(value):
        return None
    return json.loads(value)


def _image_frame(image_path: str) -> int:
    return int(Path(image_path).stem)


def _image_video(image_path: str) -> str:
    return Path(image_path).parent.name


def _flatten_pid(value: Any) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        if not value:
            return ""
        return _flatten_pid(value[0])
    return str(value)


def _nominal_tte_bins(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {"lt30": 0, "30_60": 0, "gt60": 0}
    return {
        "lt30": int((series < 30).sum()),
        "30_60": int(series.between(30, 60).sum()),
        "gt60": int((series > 60).sum()),
    }


def _float_or_nan(value: float | int | None) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _maybe_float_stat(series: pd.Series, fn: str) -> float:
    if series.empty:
        return float("nan")
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    if fn == "min":
        return float(clean.min())
    if fn == "max":
        return float(clean.max())
    if fn == "mean":
        return float(clean.mean())
    if fn == "median":
        return float(clean.median())
    raise ValueError(f"Unsupported stat {fn}")


def _load_track_map(database: dict[str, Any]) -> dict[str, UpstreamTrackRecord]:
    track_map: dict[str, UpstreamTrackRecord] = {}
    for video_id, video_record in database.items():
        ped_annotations = video_record["ped_annotations"]
        for ped_id, ped_record in ped_annotations.items():
            frames = [int(frame) for frame in ped_record["frames"]]
            raw_crosses = ped_record.get("behavior", {}).get("cross", [])
            if raw_crosses:
                crosses = [int(value) for value in raw_crosses]
            else:
                crosses = [0] * len(frames)
            if len(crosses) != len(frames):
                raise ValueError(f"{video_id}:{ped_id} has mismatched frame/cross lengths")
            transition_frames = [
                frames[idx]
                for idx in range(1, len(frames))
                if crosses[idx - 1] == 0 and crosses[idx] == 1
            ]
            attrs = ped_record.get("attributes", {}) or {}
            track_id = f"{video_id}:{ped_id}"
            track_map[track_id] = UpstreamTrackRecord(
                video_id=video_id,
                ped_id=ped_id,
                track_id=track_id,
                frames=frames,
                crosses=crosses,
                frame_to_cross={frame: cross for frame, cross in zip(frames, crosses, strict=True)},
                crossing_attr=attrs.get("crossing"),
                crossing_point=attrs.get("crossing_point"),
                decision_point=attrs.get("decision_point"),
                track_length=len(frames),
                transition_frames=transition_frames,
                starts_already_crossing=bool(crosses and crosses[0] == 1),
                source_annotation_path=str(DATASET_ROOT / "annotations" / f"{video_id}.xml"),
                source_attributes_path=str(DATASET_ROOT / "annotations_attributes" / f"{video_id}_attributes.xml"),
            )
    return track_map


def _resolve_track_id(track_id: str, track_map: dict[str, Any]) -> str:
    if track_id in track_map:
        return track_id
    behavior_track_id = f"{track_id}b"
    if behavior_track_id in track_map:
        return behavior_track_id
    raise KeyError(f"Missing raw JAAD track for {track_id}")


def _load_upstream_artifacts() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    JAAD = _load_upstream_jaad_class()
    imdb = JAAD(data_path=str(DATASET_ROOT))
    database = imdb.generate_database()
    raw_by_split: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        raw_by_split[split] = imdb.generate_data_trajectory_sequence(
            split,
            fstride=1,
            sample_type="all",
            subset="default",
            data_split_type="default",
            seq_type="crossing",
            min_track_size=76,
        )
    return raw_by_split, database


def _build_protocol_a_windows(raw_by_split: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, raw in raw_by_split.items():
        bbox_seqs = raw["bbox"]
        center_seqs = raw["center"]
        pid_seqs = raw["pid"]
        activity_seqs = raw["activities"]
        image_seqs = raw["image"]

        for i in range(len(bbox_seqs)):
            seq_len = len(bbox_seqs[i])
            start_idx = seq_len - OBS_LEN - TTE_MAX
            end_idx = seq_len - OBS_LEN - TTE_MIN
            if start_idx < 0:
                continue

            api_anchor_frame = _image_frame(image_seqs[i][-1])
            api_anchor_source = "api_sequence_end"
            for w in range(start_idx, end_idx + 1, STRIDE):
                image_window = image_seqs[i][w : w + OBS_LEN]
                frame_window = [_image_frame(path) for path in image_window]
                video_id = _image_video(image_window[0])
                pid = _flatten_pid(pid_seqs[i][0])
                crossing = int(activity_seqs[i][w + OBS_LEN - 1][0])
                nominal_tte = int(seq_len - (w + OBS_LEN))
                start_frame = int(frame_window[0])
                end_frame = int(frame_window[-1])
                sequence_id = (
                    f"{PROTOCOL_A}:{split}:{video_id}:{pid}:{start_frame}:{end_frame}:{nominal_tte}:{crossing}"
                )
                rows.append(
                    {
                        "protocol": PROTOCOL_A,
                        "sequence_id": sequence_id,
                        "split": split,
                        "label": crossing,
                        "set_id": "jaad",
                        "video_id": video_id,
                        "ped_id": pid,
                        "track_id": f"{video_id}:{pid}",
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "event_frame": api_anchor_frame,
                        "last_observed_frame": end_frame,
                        "tte_frames": nominal_tte,
                        "nominal_tte_frames": nominal_tte,
                        "observation_length": OBS_LEN,
                        "frame_step": 1,
                        "sample_stride_between_windows": STRIDE,
                        "frame_sampling": "consecutive",
                        "onset_source": "",
                        "negative_alignment_source": "",
                        "track_length": seq_len,
                        "crossing_attr": np.nan,
                        "crossing_point": np.nan,
                        "decision_point": np.nan,
                        "has_observed_crossing": np.nan,
                        "observed_frames_json": _json_dumps(frame_window),
                        "observed_crosses_json": "",
                        "bbox_sequence_json": _json_dumps(bbox_seqs[i][w : w + OBS_LEN]),
                        "source_annotation_path": "",
                        "source_attributes_path": "",
                        "parser_mode": "upstream_jaad_api_crossing_sequence",
                        "api_track_length": seq_len,
                        "api_anchor_frame": api_anchor_frame,
                        "api_anchor_source": api_anchor_source,
                        "api_window_start_index": int(w),
                        "api_window_end_index": int(w + OBS_LEN - 1),
                        "api_track_end_frame": api_anchor_frame,
                        "api_center_sequence_json": _json_dumps(center_seqs[i][w : w + OBS_LEN]),
                    }
                )

    df = pd.DataFrame(rows)
    if df["sequence_id"].duplicated().any():
        raise ValueError("Protocol A produced duplicate sequence IDs")
    return df.sort_values(["split", "video_id", "ped_id", "start_frame"]).reset_index(drop=True)


def _load_protocol_b_windows() -> pd.DataFrame:
    frames = []
    for split in SPLITS:
        frame = pd.read_csv(CLEAN_SPLIT_DIR / f"{split}.csv")
        frame["split"] = split
        frame["set_id"] = "jaad"
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df["protocol"] = PROTOCOL_B
    df["nominal_tte_frames"] = df["tte_frames"].astype(int)
    df["api_anchor_frame"] = df["event_frame"].astype(int)
    df["api_anchor_source"] = "verified_event_alignment"
    df["api_track_length"] = np.nan
    df["api_window_start_index"] = np.nan
    df["api_window_end_index"] = np.nan
    df["api_track_end_frame"] = np.nan
    df["api_center_sequence_json"] = ""
    return df.sort_values(["split", "video_id", "ped_id", "start_frame"]).reset_index(drop=True)


def _annotate_windows(df: pd.DataFrame, track_map: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        original_track_id = str(record["track_id"])
        resolved_track_id = _resolve_track_id(original_track_id, track_map)
        track = track_map[resolved_track_id]
        resolved_ped_id = resolved_track_id.split(":", 1)[1]

        observed_frames = [int(frame) for frame in _json_loads(record["observed_frames_json"])]
        observed_crosses = [int(track.frame_to_cross[frame]) for frame in observed_frames]
        physical_onset = (
            int(track.transition_frames[0]) if len(track.transition_frames) == 1 else None
        )
        last_cross_frame = (
            int(max(frame for frame, cross in zip(track.frames, track.crosses, strict=True) if cross == 1))
            if any(track.crosses)
            else None
        )
        contains_cross1 = int(any(value == 1 for value in observed_crosses))
        broad_hazard_track = int(track.crossing_attr == 1 and track.crossing_point == -1)
        physical_onset_available = int(physical_onset is not None)
        physical_tte = (
            int(physical_onset - int(record["end_frame"])) if physical_onset is not None else np.nan
        )
        label = int(record["label"])

        if label != 1:
            temporal_position = "negative"
        elif physical_onset is None:
            temporal_position = "onset_unavailable"
        elif contains_cross1:
            temporal_position = "already_crossing"
        elif last_cross_frame is not None and int(record["start_frame"]) > last_cross_frame:
            temporal_position = "post_crossing"
        elif int(record["end_frame"]) < physical_onset:
            temporal_position = "genuinely_pre_crossing"
        elif int(record["start_frame"]) > physical_onset:
            temporal_position = "after_onset_without_cross_flag"
        else:
            temporal_position = "ambiguous_temporal_position"

        api_anchor_frame = int(record["api_anchor_frame"])
        nominal_anchor_matches_verified_onset = int(
            physical_onset is not None and api_anchor_frame == physical_onset
        )
        nominal_tte_not_verified_anchor = int(
            label == 1 and not nominal_anchor_matches_verified_onset
        )
        api_label_semantics_affected = int(
            label == 1
            and broad_hazard_track == 1
            and temporal_position in {"already_crossing", "post_crossing"}
        )
        api_temporal_anchor_affected = int(label == 1 and nominal_tte_not_verified_anchor == 1)
        api_flaw_affected = int(
            record["protocol"] == PROTOCOL_A
            and (api_label_semantics_affected == 1 or api_temporal_anchor_affected == 1)
        )

        row = dict(record)
        row.update(
            {
                "track_id": resolved_track_id,
                "ped_id": resolved_ped_id,
                "api_track_id_original": original_track_id,
                "api_ped_id_original": str(record["ped_id"]),
                "crossing_attr": track.crossing_attr,
                "crossing_point": track.crossing_point,
                "decision_point": track.decision_point,
                "source_annotation_path": track.source_annotation_path,
                "source_attributes_path": track.source_attributes_path,
                "track_length": track.track_length,
                "observed_crosses_json": _json_dumps(observed_crosses),
                "has_observed_crossing": contains_cross1,
                "physical_onset_frame": physical_onset,
                "last_cross_frame": last_cross_frame,
                "physical_onset_available": physical_onset_available,
                "physical_tte_frames": physical_tte,
                "temporal_position": temporal_position,
                "broad_hazard_track": broad_hazard_track,
                "contains_frame_level_cross1": contains_cross1,
                "partly_overlapping_actual_crossing": int(label == 1 and contains_cross1 == 1),
                "entirely_after_true_crossing_onset": int(
                    physical_onset is not None and int(record["start_frame"]) > physical_onset
                ),
                "entirely_after_crossing_segment": int(
                    last_cross_frame is not None and int(record["start_frame"]) > last_cross_frame
                ),
                "nominal_anchor_matches_verified_onset": nominal_anchor_matches_verified_onset,
                "nominal_tte_not_verified_anchor": nominal_tte_not_verified_anchor,
                "api_label_semantics_affected": api_label_semantics_affected,
                "api_temporal_anchor_affected": api_temporal_anchor_affected,
                "api_flaw_affected": api_flaw_affected,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _build_bbox_features(windows_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in windows_df.to_dict(orient="records"):
        bboxes = np.asarray(_json_loads(record["bbox_sequence_json"]), dtype=float)
        feats = ARCHIVE_BBOX.compute_bbox_motion_features(bboxes)
        base = {key: record[key] for key in META_COLUMNS}
        rows.append({**base, **feats})
    out = pd.DataFrame(rows)
    if out["sequence_id"].duplicated().any():
        raise ValueError("BBox features contain duplicate sequence IDs")
    return out.sort_values(["split", "sequence_id"]).reset_index(drop=True)


def _load_protocol_b_bbox_features() -> pd.DataFrame:
    frames = []
    for split in SPLITS:
        frame = pd.read_csv(CLEAN_FEATURE_DIR / f"{split}_bbox.csv")
        frame["split"] = split
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    if out["sequence_id"].duplicated().any():
        raise ValueError("Protocol B bbox table contains duplicate sequence IDs")
    return out.sort_values(["split", "sequence_id"]).reset_index(drop=True)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if col not in META_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
    ]


def _run_selected_bbox_benchmark(
    protocol_name: str,
    windows_df: pd.DataFrame,
    bbox_df: pd.DataFrame,
) -> ProtocolResult:
    print(f"[Exp031] Starting bbox benchmark for {protocol_name}", flush=True)
    frame = windows_df[META_COLUMNS].merge(
        bbox_df,
        on=META_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    feature_columns = _feature_columns(frame)
    if not feature_columns:
        raise ValueError(f"{protocol_name}: no numeric bbox features")

    per_seed_summary: list[dict[str, Any]] = []
    per_seed_predictions: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    split_frames = {split: frame.loc[frame["split"] == split].copy() for split in SPLITS}
    train_df = split_frames["train"]
    val_df = split_frames["val"]
    test_df = split_frames["test"]

    X_train = train_df[feature_columns].astype(float)
    y_train = train_df["label"].astype(int).to_numpy()
    X_val = val_df[feature_columns].astype(float)
    y_val = val_df["label"].astype(int).to_numpy()
    X_test = test_df[feature_columns].astype(float)
    y_test = test_df["label"].astype(int).to_numpy()

    for seed in BENCHMARK.SEEDS:
        print(f"[Exp031] {protocol_name} seed={seed}: fitting candidate models", flush=True)
        candidates: list[tuple[str, Any, np.ndarray, np.ndarray]] = []
        for model_name in BENCHMARK.MODEL_ORDER:
            print(f"[Exp031] {protocol_name} seed={seed} model={model_name}: fit", flush=True)
            model = BENCHMARK._build_model(model_name, seed, y_train)
            if model is None:
                continue
            model.fit(X_train, y_train)
            val_prob = model.predict_proba(X_val)[:, 1]
            test_prob = model.predict_proba(X_test)[:, 1]
            selection = next(
                item
                for item in BENCHMARK._select_thresholds(y_val, val_prob)
                if item.rule == "val_f1"
            )
            if selection.status != "available" or selection.threshold is None:
                continue
            candidates.append((model_name, selection, val_prob, test_prob))
            candidate_rows.append(
                {
                    "protocol": protocol_name,
                    "representation": REPRESENTATION,
                    "seed": seed,
                    "model_name": model_name,
                    "selected_threshold": float(selection.threshold),
                    "candidate_val_f1": float(selection.val_metrics["f1"]),
                    "candidate_val_pr_auc": float(selection.val_metrics["pr_auc"]),
                }
            )
            print(
                f"[Exp031] {protocol_name} seed={seed} model={model_name}: "
                f"val_f1={selection.val_metrics['f1']:.4f} thr={float(selection.threshold):.4f}",
                flush=True,
            )

        if not candidates:
            raise RuntimeError(f"{protocol_name}: no usable candidate models")

        model_name, selection, val_prob, test_prob = max(
            candidates, key=lambda item: item[1].val_metrics["f1"]
        )
        threshold = float(selection.threshold)
        val_metrics = selection.val_metrics
        test_metrics = BENCHMARK._compute_metrics(y_test, test_prob, threshold)
        test_pred = (test_prob >= threshold).astype(int)
        val_pred = (val_prob >= threshold).astype(int)
        print(
            f"[Exp031] {protocol_name} seed={seed}: selected {model_name} "
            f"val_f1={val_metrics['f1']:.4f} test_f1={test_metrics['f1']:.4f}",
            flush=True,
        )

        per_seed_summary.append(
            {
                "protocol": protocol_name,
                "representation": REPRESENTATION,
                "seed": seed,
                "model_name": model_name,
                "threshold_rule": "val_f1",
                "selected_threshold": threshold,
                "n_features": len(feature_columns),
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
                **{f"val_{key}": value for key, value in val_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        )

        for split_name, split_frame, probabilities, predictions in [
            ("val", val_df, val_prob, val_pred),
            ("test", test_df, test_prob, test_pred),
        ]:
            for base_record, prob, pred in zip(
                split_frame[META_COLUMNS].to_dict(orient="records"),
                probabilities,
                predictions,
                strict=True,
            ):
                per_seed_predictions.append(
                    {
                        "protocol": protocol_name,
                        "representation": REPRESENTATION,
                        "seed": seed,
                        "model_name": model_name,
                        "threshold_rule": "val_f1",
                        "selected_threshold": threshold,
                        "probability": float(prob),
                        "prediction": int(pred),
                        "prediction_split": split_name,
                        **base_record,
                    }
                )

    return ProtocolResult(
        summary_rows=pd.DataFrame(per_seed_summary),
        prediction_rows=pd.DataFrame(per_seed_predictions),
        candidate_rows=pd.DataFrame(candidate_rows),
    )


def _summarize_population(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for protocol in [PROTOCOL_A, PROTOCOL_B]:
        protocol_df = df.loc[df["protocol"] == protocol].copy()
        for split in ["all", *SPLITS]:
            sub = protocol_df if split == "all" else protocol_df.loc[protocol_df["split"] == split].copy()
            positives = sub.loc[sub["label"] == 1].copy()
            physical_known = positives.loc[positives["physical_onset_available"] == 1].copy()
            phys_bins = _nominal_tte_bins(physical_known["physical_tte_frames"])
            nom_bins = _nominal_tte_bins(positives["nominal_tte_frames"])
            rows.append(
                {
                    "protocol": protocol,
                    "split": split,
                    "total_windows": int(len(sub)),
                    "positive_windows": int((sub["label"] == 1).sum()),
                    "negative_windows": int((sub["label"] == 0).sum()),
                    "positive_rate": float((sub["label"] == 1).mean()) if len(sub) else float("nan"),
                    "unique_tracks": int(sub["track_id"].nunique()),
                    "unique_positive_tracks": int(sub.loc[sub["label"] == 1, "track_id"].nunique()),
                    "unique_negative_tracks": int(sub.loc[sub["label"] == 0, "track_id"].nunique()),
                    "crossing1_crossingpoint_minus1_tracks": int(
                        sub.loc[sub["broad_hazard_track"] == 1, "track_id"].nunique()
                    ),
                    "crossing1_crossingpoint_minus1_positive_windows": int(
                        positives["broad_hazard_track"].sum()
                    ),
                    "windows_containing_frame_cross1": int(sub["contains_frame_level_cross1"].sum()),
                    "windows_partly_overlapping_actual_crossing": int(
                        sub["partly_overlapping_actual_crossing"].sum()
                    ),
                    "windows_entirely_after_true_crossing_onset": int(
                        sub["entirely_after_true_crossing_onset"].sum()
                    ),
                    "windows_entirely_after_crossing_segment": int(
                        sub["entirely_after_crossing_segment"].sum()
                    ),
                    "positive_windows_nominal_tte_not_verified_anchor": int(
                        positives["nominal_tte_not_verified_anchor"].sum()
                    ),
                    "positive_windows_pre_crossing": int(
                        (positives["temporal_position"] == "genuinely_pre_crossing").sum()
                    ),
                    "positive_windows_already_crossing": int(
                        (positives["temporal_position"] == "already_crossing").sum()
                    ),
                    "positive_windows_post_crossing": int(
                        (positives["temporal_position"] == "post_crossing").sum()
                    ),
                    "positive_windows_onset_unavailable": int(
                        (positives["temporal_position"] == "onset_unavailable").sum()
                    ),
                    "positive_windows_other_temporal": int(
                        positives["temporal_position"].isin(
                            ["after_onset_without_cross_flag", "ambiguous_temporal_position"]
                        ).sum()
                    ),
                    "positive_windows_api_flaw_affected": int(positives["api_flaw_affected"].sum()),
                    "positive_windows_physical_onset_available": int(
                        positives["physical_onset_available"].sum()
                    ),
                    "nominal_tte_min": _maybe_float_stat(positives["nominal_tte_frames"], "min"),
                    "nominal_tte_max": _maybe_float_stat(positives["nominal_tte_frames"], "max"),
                    "nominal_tte_mean": _maybe_float_stat(positives["nominal_tte_frames"], "mean"),
                    "nominal_tte_median": _maybe_float_stat(positives["nominal_tte_frames"], "median"),
                    "nominal_tte_lt30": nom_bins["lt30"],
                    "nominal_tte_30_60": nom_bins["30_60"],
                    "nominal_tte_gt60": nom_bins["gt60"],
                    "physical_tte_min": _maybe_float_stat(physical_known["physical_tte_frames"], "min"),
                    "physical_tte_max": _maybe_float_stat(physical_known["physical_tte_frames"], "max"),
                    "physical_tte_mean": _maybe_float_stat(physical_known["physical_tte_frames"], "mean"),
                    "physical_tte_median": _maybe_float_stat(physical_known["physical_tte_frames"], "median"),
                    "physical_tte_lt30": phys_bins["lt30"],
                    "physical_tte_30_60": phys_bins["30_60"],
                    "physical_tte_gt60": phys_bins["gt60"],
                }
            )
    return pd.DataFrame(rows)


def _keypoint_coverage(protocol_a_df: pd.DataFrame) -> pd.DataFrame:
    needed: set[tuple[str, str, int]] = set()
    for record in protocol_a_df.to_dict(orient="records"):
        frames = _json_loads(record["observed_frames_json"])
        for frame in frames:
            needed.add((str(record["video_id"]), str(record["ped_id"]), int(frame)))

    found: set[tuple[str, str, int]] = set()
    with KEYPOINT_JSONL.open("r", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            key = (str(rec["video"]), str(rec["pid"]), int(rec["frame_idx"]))
            if key in needed:
                found.add(key)

    rows = []
    for split in ["all", *SPLITS]:
        sub = protocol_a_df if split == "all" else protocol_a_df.loc[protocol_a_df["split"] == split]
        total_windows = len(sub)
        fully_covered = 0
        for record in sub.to_dict(orient="records"):
            frames = _json_loads(record["observed_frames_json"])
            keys = {(str(record["video_id"]), str(record["ped_id"]), int(frame)) for frame in frames}
            if keys.issubset(found):
                fully_covered += 1
        rows.append(
            {
                "split": split,
                "windows": int(total_windows),
                "fully_covered_windows": int(fully_covered),
                "fully_covered_ratio": float(fully_covered / total_windows) if total_windows else float("nan"),
                "needed_frames_total": int(
                    len(
                        {
                            (str(record["video_id"]), str(record["ped_id"]), int(frame))
                            for record in sub.to_dict(orient="records")
                            for frame in _json_loads(record["observed_frames_json"])
                        }
                    )
                ),
                "found_frames_total": int(
                    len(
                        {
                            key
                            for key in found
                            if any(
                                key[0] == str(record["video_id"]) and key[1] == str(record["ped_id"])
                                for record in sub.to_dict(orient="records")
                            )
                        }
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def _delta_summary(per_seed_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["accuracy", "roc_auc", "pr_auc", "f1", "precision", "recall"]:
        clean = per_seed_results.loc[per_seed_results["protocol"] == PROTOCOL_B, f"test_{metric}"]
        api = per_seed_results.loc[per_seed_results["protocol"] == PROTOCOL_A, f"test_{metric}"]
        rows.append(
            {
                "representation": REPRESENTATION,
                "metric": metric,
                "correct_protocol_mean": float(clean.mean()),
                "correct_protocol_std": float(clean.std(ddof=1)),
                "api_style_protocol_mean": float(api.mean()),
                "api_style_protocol_std": float(api.std(ddof=1)),
                "delta_api_minus_correct": float(api.mean() - clean.mean()),
            }
        )
    return pd.DataFrame(rows)


def _motion_strength_summary(df: pd.DataFrame, bbox_df: pd.DataFrame) -> pd.DataFrame:
    motion = df[["sequence_id", "protocol", "split", "label", "temporal_position"]].merge(
        bbox_df[
            [
                "sequence_id",
                "center_x_displacement",
                "center_x_velocity_mean",
                "bbox_area_change",
                "bbox_scale_change_ratio",
            ]
        ],
        on="sequence_id",
        how="left",
    )
    motion["abs_center_x_displacement"] = motion["center_x_displacement"].abs()
    motion["abs_center_x_velocity_mean"] = motion["center_x_velocity_mean"].abs()

    groups = [
        (PROTOCOL_B, "genuinely_pre_crossing"),
        (PROTOCOL_A, "genuinely_pre_crossing"),
        (PROTOCOL_A, "already_crossing"),
        (PROTOCOL_A, "post_crossing"),
        (PROTOCOL_A, "onset_unavailable"),
    ]
    rows = []
    for protocol, temporal_position in groups:
        sub = motion.loc[
            (motion["protocol"] == protocol)
            & (motion["label"] == 1)
            & (motion["temporal_position"] == temporal_position)
        ].copy()
        if sub.empty:
            continue
        rows.append(
            {
                "protocol": protocol,
                "temporal_position": temporal_position,
                "n_windows": int(len(sub)),
                "abs_center_x_displacement_mean": float(sub["abs_center_x_displacement"].mean()),
                "abs_center_x_velocity_mean_mean": float(sub["abs_center_x_velocity_mean"].mean()),
                "bbox_area_change_mean": float(sub["bbox_area_change"].mean()),
                "bbox_scale_change_ratio_mean": float(sub["bbox_scale_change_ratio"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _exact_overlap(protocol_a: pd.DataFrame, protocol_b: pd.DataFrame) -> pd.DataFrame:
    merge_cols = ["split", "video_id", "ped_id", "start_frame", "end_frame"]
    keep_cols = [
        *merge_cols,
        "sequence_id",
        "label",
        "nominal_tte_frames",
        "physical_tte_frames",
        "temporal_position",
        "api_anchor_frame",
    ]
    overlap = protocol_a[keep_cols].merge(
        protocol_b[keep_cols],
        on=merge_cols,
        how="inner",
        suffixes=("_api", "_clean"),
    )
    if overlap.empty:
        return overlap
    overlap["label_differs"] = (
        overlap["label_api"].astype(int) != overlap["label_clean"].astype(int)
    ).astype(int)
    overlap["nominal_tte_delta_api_minus_clean"] = (
        pd.to_numeric(overlap["nominal_tte_frames_api"], errors="coerce")
        - pd.to_numeric(overlap["nominal_tte_frames_clean"], errors="coerce")
    )
    return overlap.sort_values(merge_cols).reset_index(drop=True)


def _exact_overlap_prediction_comparison(
    prediction_rows: pd.DataFrame,
    overlap_rows: pd.DataFrame,
) -> pd.DataFrame:
    if overlap_rows.empty:
        return overlap_rows
    merge_cols = ["seed", "prediction_split", "video_id", "ped_id", "start_frame", "end_frame"]
    base = prediction_rows.loc[
        prediction_rows["protocol"] == PROTOCOL_A,
        [
            "seed",
            "prediction_split",
            "video_id",
            "ped_id",
            "start_frame",
            "end_frame",
            "probability",
            "prediction",
            "label",
        ],
    ].rename(
        columns={
            "probability": "probability_api",
            "prediction": "prediction_api",
            "label": "label_api",
        }
    )
    clean = prediction_rows.loc[
        prediction_rows["protocol"] == PROTOCOL_B,
        [
            "seed",
            "prediction_split",
            "video_id",
            "ped_id",
            "start_frame",
            "end_frame",
            "probability",
            "prediction",
            "label",
        ],
    ].rename(
        columns={
            "probability": "probability_clean",
            "prediction": "prediction_clean",
            "label": "label_clean",
        }
    )
    overlap_base = overlap_rows[["split", "video_id", "ped_id", "start_frame", "end_frame"]].drop_duplicates()
    base = overlap_base.merge(
        base,
        left_on=["split", "video_id", "ped_id", "start_frame", "end_frame"],
        right_on=["prediction_split", "video_id", "ped_id", "start_frame", "end_frame"],
        how="inner",
    )
    clean = overlap_base.merge(
        clean,
        left_on=["split", "video_id", "ped_id", "start_frame", "end_frame"],
        right_on=["prediction_split", "video_id", "ped_id", "start_frame", "end_frame"],
        how="inner",
    )
    merged = base.merge(clean, on=merge_cols, how="inner", validate="one_to_one")
    merged["probability_delta_api_minus_clean"] = (
        merged["probability_api"] - merged["probability_clean"]
    )
    return merged.sort_values(merge_cols).reset_index(drop=True)


def _write_commands_used() -> None:
    lines = [
        "# Commands used for Exp031",
        "",
        "Primary execution command:",
        "```bash",
        "cd /home/abdelrahman/Hadwa/BIU-PID-Distillation-clean",
        "python3 scripts/experiments/jaad_api_label_protocol_sensitivity.py",
        "```",
        "",
        "Recovered provenance inputs:",
        f"- Noor repository commit: `{NOOR_REPO_COMMIT}`",
        f"- Upstream JAAD commit inspected: `{UPSTREAM_JAAD_COMMIT}`",
        f"- Upstream mechanism already present by: `{UPSTREAM_MECHANISM_COMMIT}`",
        f"- Clean split dir: `{CLEAN_SPLIT_DIR}`",
        f"- Clean bbox dir: `{CLEAN_FEATURE_DIR}`",
        f"- Current JAAD snapshot: `{DATASET_ROOT}`",
    ]
    (PROVENANCE_DIR / "commands_used.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(
    population: pd.DataFrame,
    per_seed_results: pd.DataFrame,
    delta_summary: pd.DataFrame,
    motion_summary: pd.DataFrame,
    overlap_rows: pd.DataFrame,
    overlap_predictions: pd.DataFrame,
    keypoint_coverage: pd.DataFrame,
) -> None:
    pop_all = population.loc[population["split"] == "all"].set_index("protocol")
    clean = pop_all.loc[PROTOCOL_B]
    api = pop_all.loc[PROTOCOL_A]

    seed_lines = []
    for row in per_seed_results.sort_values(["protocol", "seed"]).to_dict(orient="records"):
        seed_lines.append(
            f"| `{row['protocol']}` | {row['seed']} | `{row['model_name']}` | "
            f"{row['test_accuracy']:.3f} | {row['test_roc_auc']:.3f} | {row['test_pr_auc']:.3f} | "
            f"{row['test_f1']:.3f} | {row['test_precision']:.3f} | {row['test_recall']:.3f} |"
        )

    delta_lines = []
    for row in delta_summary.to_dict(orient="records"):
        delta_lines.append(
            f"| `{row['metric']}` | {row['correct_protocol_mean']:.3f} +/- {row['correct_protocol_std']:.3f} | "
            f"{row['api_style_protocol_mean']:.3f} +/- {row['api_style_protocol_std']:.3f} | "
            f"{row['delta_api_minus_correct']:+.3f} |"
        )

    motion_lines = []
    for row in motion_summary.to_dict(orient="records"):
        motion_lines.append(
            f"| `{row['protocol']}` | `{row['temporal_position']}` | {row['n_windows']} | "
            f"{row['abs_center_x_displacement_mean']:.3f} | {row['abs_center_x_velocity_mean_mean']:.3f} | "
            f"{row['bbox_area_change_mean']:.3f} | {row['bbox_scale_change_ratio_mean']:.3f} |"
        )

    overlap_summary = "No exact overlapping physical windows were found between the two protocol constructions."
    if not overlap_rows.empty:
        overlap_summary = (
            f"Exact-overlap windows found: `{len(overlap_rows)}` total, "
            f"`{int(overlap_rows['label_differs'].sum())}` with different labels."
        )
    overlap_pred_summary = "No overlap prediction comparison was produced."
    if not overlap_predictions.empty:
        overlap_pred_summary = (
            f"Exact-overlap prediction rows: `{len(overlap_predictions)}` across seeds. "
            f"Mean probability delta API-clean on overlap rows: "
            f"`{overlap_predictions['probability_delta_api_minus_clean'].mean():.3f}`."
        )

    keypoint_text = []
    for row in keypoint_coverage.to_dict(orient="records"):
        keypoint_text.append(
            f"- Protocol A keypoint coverage `{row['split']}`: "
            f"{row['fully_covered_windows']}/{row['windows']} fully covered "
            f"({row['fully_covered_ratio']:.3f})."
        )

    lines = [
        "# JAAD API Protocol Sensitivity",
        "",
        "## Exact API Flaw Under Test",
        "",
        "- Protocol A is a documented reconstruction of the upstream JAAD API-style flawed code path used by Noor's pipeline lineage.",
        "- Noor repository commit inspected: "
        f"`{NOOR_REPO_COMMIT}`.",
        "- Upstream JAAD commit inspected: "
        f"`{UPSTREAM_JAAD_COMMIT}`.",
        "- The core mechanism already exists by upstream commit "
        f"`{UPSTREAM_MECHANISM_COMMIT}`.",
        "- Protocol B is the clean event-aligned 16-frame 30_60 overlap-0.8 JAAD benchmark already used in the thesis.",
        "",
        "## Population Differences",
        "",
        f"- Clean protocol windows: `{int(clean['total_windows'])}` total, `{int(clean['positive_windows'])}` positive, "
        f"`{clean['positive_rate']:.3f}` positive rate.",
        f"- API-style protocol windows: `{int(api['total_windows'])}` total, `{int(api['positive_windows'])}` positive, "
        f"`{api['positive_rate']:.3f}` positive rate.",
        f"- API-style positives that were not nominally anchored to a verified physical crossing onset: "
        f"`{int(api['positive_windows_nominal_tte_not_verified_anchor'])}`.",
        f"- API-style positives by temporal position: pre-crossing=`{int(api['positive_windows_pre_crossing'])}`, "
        f"already-crossing=`{int(api['positive_windows_already_crossing'])}`, "
        f"post-crossing=`{int(api['positive_windows_post_crossing'])}`, "
        f"onset-unavailable=`{int(api['positive_windows_onset_unavailable'])}`.",
        f"- API-style hazard-track population (`crossing=1`, `crossing_point=-1`): "
        f"`{int(api['crossing1_crossingpoint_minus1_tracks'])}` tracks contributing "
        f"`{int(api['crossing1_crossingpoint_minus1_positive_windows'])}` positive windows.",
        f"- API-style physical TTE among positives with verified onset: "
        f"mean=`{api['physical_tte_mean']:.2f}`, median=`{api['physical_tte_median']:.2f}`, "
        f"`<30`=`{int(api['physical_tte_lt30'])}`, `30-60`=`{int(api['physical_tte_30_60'])}`, "
        f"`>60`=`{int(api['physical_tte_gt60'])}`.",
        "",
        "## BBox Benchmark Results",
        "",
        "| Protocol | Seed | Selected model | Acc | ROC-AUC | PR-AUC | F1 | Precision | Recall |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        *seed_lines,
        "",
        "| Metric | Clean protocol | API-style protocol | Delta API - Clean |",
        "|---|---:|---:|---:|",
        *delta_lines,
        "",
        "## Why The Task Changes",
        "",
        "- Protocol A does not only change labels. It also changes the temporal anchor used to define nominal `30_60` windows.",
        "- The clean protocol keeps only verified pre-crossing windows. Protocol A can retain already-crossing or post-crossing positive observations on hazard tracks, and can also retain positives whose nominal TTE is measured relative to an API-generated sequence end rather than a verified physical onset.",
        "",
        "| Protocol | Positive window type | N | mean |vx displacement| mean |vx velocity| mean area change | mean scale ratio |",
        "|---|---|---:|---:|---:|---:|---:|",
        *motion_lines,
        "",
        "## Exact-Overlap Diagnostic",
        "",
        f"- {overlap_summary}",
        f"- {overlap_pred_summary}",
        "",
        "## Secondary BBox + BIU-v2 Feasibility",
        "",
        *keypoint_text,
        "- Secondary `BBox + BIU-v2` was not executed in Exp031. The primary goal here is protocol sensitivity under a stable representation, and adding non-benchmark BIU-v2 reconstruction for Protocol A would change the experiment into a feature-lineage exercise.",
        "",
        "## What This Means",
        "",
        "- This audit does not replace the clean JAAD benchmark.",
        "- It quantifies how the upstream API-style construction can alter the task population and apparent performance under the same model-selection recipe.",
        "- Any performance increase under Protocol A should be interpreted as protocol sensitivity or an easier task construction, not as a better anticipatory benchmark.",
        "",
        "## What It Does Not Mean",
        "",
        "- It does not prove that every JAAD paper used this exact flaw.",
        "- It does not invalidate the clean event-aligned benchmark.",
        "- It does not imply that all protocol differences are due only to post-crossing positives; temporal-anchor drift is a separate effect and is reported separately here.",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)

    print("[Exp031] Loading upstream artifacts", flush=True)
    raw_by_split, upstream_database = _load_upstream_artifacts()
    track_map = _load_track_map(upstream_database)
    print("[Exp031] Building Protocol A/B window tables", flush=True)
    protocol_a_windows = _build_protocol_a_windows(raw_by_split)
    protocol_b_windows = _load_protocol_b_windows()

    print("[Exp031] Annotating windows against raw JAAD tracks", flush=True)
    protocol_a_annotated = _annotate_windows(protocol_a_windows, track_map)
    protocol_b_annotated = _annotate_windows(protocol_b_windows, track_map)
    combined = pd.concat([protocol_a_annotated, protocol_b_annotated], ignore_index=True)
    protocol_a_annotated.to_csv(METRICS_DIR / "protocol_a_window_audit.csv", index=False)
    protocol_b_annotated.to_csv(METRICS_DIR / "protocol_b_window_audit.csv", index=False)

    print("[Exp031] Building/loading bbox features", flush=True)
    protocol_a_bbox = _build_bbox_features(protocol_a_annotated)
    protocol_b_bbox = _load_protocol_b_bbox_features()

    population = _summarize_population(combined)
    keypoint_coverage = _keypoint_coverage(protocol_a_annotated)
    motion_summary = _motion_strength_summary(
        combined,
        pd.concat([protocol_a_bbox, protocol_b_bbox], ignore_index=True),
    )
    overlap_rows = _exact_overlap(protocol_a_annotated, protocol_b_annotated)
    population.to_csv(METRICS_DIR / "protocol_population_summary.csv", index=False)
    keypoint_coverage.to_csv(METRICS_DIR / "protocol_a_keypoint_cache_coverage.csv", index=False)
    motion_summary.to_csv(METRICS_DIR / "motion_strength_by_window_type.csv", index=False)
    overlap_rows.to_csv(METRICS_DIR / "exact_overlap_window_comparison.csv", index=False)
    protocol_a_annotated.loc[protocol_a_annotated["api_flaw_affected"] == 1].to_csv(
        METRICS_DIR / "affected_sample_audit.csv",
        index=False,
    )
    print("[Exp031] Population audit written; starting benchmark", flush=True)

    protocol_a_result = _run_selected_bbox_benchmark(PROTOCOL_A, protocol_a_annotated, protocol_a_bbox)
    protocol_b_result = _run_selected_bbox_benchmark(PROTOCOL_B, protocol_b_annotated, protocol_b_bbox)
    per_seed_results = pd.concat(
        [protocol_a_result.summary_rows, protocol_b_result.summary_rows], ignore_index=True
    )
    prediction_rows = pd.concat(
        [protocol_a_result.prediction_rows, protocol_b_result.prediction_rows], ignore_index=True
    )
    candidate_rows = pd.concat(
        [protocol_a_result.candidate_rows, protocol_b_result.candidate_rows], ignore_index=True
    )
    delta_summary = _delta_summary(per_seed_results)
    overlap_predictions = _exact_overlap_prediction_comparison(prediction_rows, overlap_rows)

    per_seed_results.to_csv(METRICS_DIR / "per_seed_results.csv", index=False)
    delta_summary.to_csv(METRICS_DIR / "protocol_delta_summary.csv", index=False)
    candidate_rows.to_csv(METRICS_DIR / "candidate_model_scores.csv", index=False)
    prediction_rows.to_csv(METRICS_DIR / "selected_model_predictions.csv", index=False)
    overlap_predictions.to_csv(METRICS_DIR / "exact_overlap_prediction_comparison.csv", index=False)

    _write_commands_used()
    _write_report(
        population=population,
        per_seed_results=per_seed_results,
        delta_summary=delta_summary,
        motion_summary=motion_summary,
        overlap_rows=overlap_rows,
        overlap_predictions=overlap_predictions,
        keypoint_coverage=keypoint_coverage,
    )
    print("[Exp031] Final outputs written", flush=True)

    print(f"Wrote Exp031 outputs to {OUT_ROOT}")
    print(f"Population summary: {METRICS_DIR / 'protocol_population_summary.csv'}")
    print(f"Per-seed results: {METRICS_DIR / 'per_seed_results.csv'}")
    print(f"Delta summary: {METRICS_DIR / 'protocol_delta_summary.csv'}")
    print(f"Report: {DOC_PATH}")


if __name__ == "__main__":
    main()
