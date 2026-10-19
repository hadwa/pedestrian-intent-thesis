"""Build frozen-row T1/T2 feature tables for the adapted MSLSTM-style teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.students.datasets import load_split_csvs
from src.teachers.mslstm_pid.stream_loaders import (
    MSLSTMStreamRepository,
    apply_standardizer,
    fit_train_standardizer,
    make_static_scene_features,
    make_traffic_light_features,
    temporal_summary,
)


META_COLUMNS = ["sequence_id", "split", "set_id", "video_id", "ped_id", "start_frame", "end_frame", "label"]
T1_VARIANT = "T1"
T2_VARIANT = "T2"


def _numeric_summary_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    numeric = [c for c in frame.columns if c not in META_COLUMNS]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    return frame


def _observed_frames(row: pd.Series) -> list[int]:
    return list(range(int(row["start_frame"]), int(row["end_frame"]) + 1))


def _extract_train_ego_scaler_frame(repo: MSLSTMStreamRepository, train_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for row in train_df.itertuples(index=False):
        for frame_idx in range(int(row.start_frame), int(row.end_frame) + 1):
            speed = repo.get_vehicle_field(row.set_id, row.video_id, frame_idx, ["OBD_speed"])
            if speed is None or float(speed[0]) == 0.0:
                speed = repo.get_vehicle_field(row.set_id, row.video_id, frame_idx, ["GPS_speed"])
            acc = repo.get_vehicle_field(row.set_id, row.video_id, frame_idx, ["accX", "accY"])
            gyro = repo.get_vehicle_field(row.set_id, row.video_id, frame_idx, ["gyroZ"])
            rows.append(
                {
                    "ego_speed": float(speed[0]) if speed is not None else np.nan,
                    "ego_acc_0": float(acc[0]) if acc is not None else np.nan,
                    "ego_acc_1": float(acc[1]) if acc is not None else np.nan,
                    "ego_gyro": float(gyro[0]) if gyro is not None else np.nan,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.isna().all().any():
        bad = frame.columns[frame.isna().all()].tolist()
        raise ValueError(f"Training scaler frame contains entirely missing ego columns: {bad}")
    return frame


def _build_core_sequence_features(
    row: dict[str, Any],
    repo: MSLSTMStreamRepository,
    ego_mean: pd.Series,
    ego_std: pd.Series,
    variant: str,
    repair_stats: dict[str, dict[str, int]],
) -> dict[str, Any]:
    set_id = str(row["set_id"])
    video_id = str(row["video_id"])
    ped_id = str(row["ped_id"])
    frames = _observed_frames(row)
    feature_row: dict[str, Any] = {key: row[key] for key in META_COLUMNS}

    bbox_values: dict[int, np.ndarray] = {}
    bbox_missing: list[int] = []
    for frame_idx in frames:
        val = repo.get_bbox_normalized(set_id, video_id, ped_id, frame_idx)
        if val is None:
            bbox_missing.append(frame_idx)
        else:
            bbox_values[frame_idx] = val
    bbox_series = []
    bbox_repaired = 0
    for frame_idx in frames:
        if frame_idx in bbox_values:
            bbox_series.append(bbox_values[frame_idx])
            continue
        repaired = repo.repair_series_one_frame(bbox_values, frames, frame_idx) if len(bbox_missing) == 1 else None
        if repaired and repaired.repaired and repaired.value is not None:
            bbox_series.append(repaired.value)
            bbox_repaired += 1
            repair_stats[variant]["bbox_repaired_frames"] += 1
            repair_stats[variant]["bbox_repaired_sequences"] += int(bbox_repaired == 1)
        else:
            bbox_series.append(np.full(4, np.nan, dtype=float))
    bbox_arr = np.asarray(bbox_series, dtype=float)
    feature_row.update(temporal_summary(bbox_arr, "bbox", include_velocity=True))
    feature_row["bbox_missing_frame_count"] = float(len(bbox_missing))
    feature_row["bbox_missing_any"] = float(len(bbox_missing) > 0)
    feature_row["bbox_valid_frame_fraction"] = float(np.mean(np.isfinite(bbox_arr).all(axis=1)))
    feature_row["bbox_repaired_frame_count"] = float(bbox_repaired)

    center_speed = np.sqrt(np.diff(bbox_arr[:, 0]) ** 2 + np.diff(bbox_arr[:, 1]) ** 2)
    feature_row["bbox_center_speed_mean"] = float(np.nanmean(center_speed)) if np.isfinite(center_speed).any() else np.nan
    feature_row["bbox_center_speed_std"] = float(np.nanstd(center_speed)) if np.isfinite(center_speed).any() else np.nan

    ego_speed_values = []
    ego_acc_values = []
    ego_gyro_values = []
    for frame_idx in frames:
        speed = repo.get_vehicle_field(set_id, video_id, frame_idx, ["OBD_speed"])
        if speed is None or float(speed[0]) == 0.0:
            speed = repo.get_vehicle_field(set_id, video_id, frame_idx, ["GPS_speed"])
        acc = repo.get_vehicle_field(set_id, video_id, frame_idx, ["accX", "accY"])
        gyro = repo.get_vehicle_field(set_id, video_id, frame_idx, ["gyroZ"])
        ego_speed_values.append(float(speed[0]) if speed is not None else np.nan)
        ego_acc_values.append(acc if acc is not None else np.asarray([np.nan, np.nan], dtype=float))
        ego_gyro_values.append(float(gyro[0]) if gyro is not None else np.nan)
    ego_frame = pd.DataFrame(
        {
            "ego_speed": ego_speed_values,
            "ego_acc_0": [float(v[0]) for v in ego_acc_values],
            "ego_acc_1": [float(v[1]) for v in ego_acc_values],
            "ego_gyro": ego_gyro_values,
        }
    )
    scaled = apply_standardizer(ego_frame, list(ego_frame.columns), ego_mean, ego_std)
    feature_row.update(temporal_summary(scaled["ego_speed"].to_numpy(dtype=float), "ego_speed"))
    feature_row.update(temporal_summary(scaled[["ego_acc_0", "ego_acc_1"]].to_numpy(dtype=float), "ego_acc"))
    feature_row.update(temporal_summary(scaled["ego_gyro"].to_numpy(dtype=float), "ego_gyro"))
    feature_row["ego_missing_any"] = float(ego_frame.isna().any(axis=None))
    return feature_row

def _add_t2_features(
    feature_row: dict[str, Any],
    row: dict[str, Any],
    repo: MSLSTMStreamRepository,
    include_yolop: bool,
    repair_stats: dict[str, dict[str, int]],
) -> dict[str, Any]:
    set_id = str(row["set_id"])
    video_id = str(row["video_id"])
    ped_id = str(row["ped_id"])
    frames = _observed_frames(row)
    tl_states: list[int] = []
    tl_any_object: list[bool] = []
    for frame_idx in frames:
        state, any_object = repo.get_traffic_light_state(set_id, video_id, frame_idx)
        tl_states.append(state)
        tl_any_object.append(any_object)
    feature_row.update(make_traffic_light_features(tl_states, tl_any_object))

    static_scene = repo.get_static_scene_only(set_id, video_id, ped_id)
    if static_scene is None:
        feature_row.update({k: np.nan for k in make_static_scene_features({"signalized": 0, "intersection": 0, "traffic_direction": 0, "num_lanes_bucket": 0})})
        feature_row["static_scene_valid"] = 0.0
        repair_stats[T2_VARIANT]["static_scene_missing_rows"] += 1
    else:
        feature_row.update(make_static_scene_features(static_scene))
        feature_row["static_scene_valid"] = 1.0

    if include_yolop:
        yolop_values: dict[int, np.ndarray] = {}
        missing: list[int] = []
        for frame_idx in frames:
            value = repo.get_yolop(set_id, video_id, ped_id, frame_idx)
            if value is None:
                missing.append(frame_idx)
            else:
                yolop_values[frame_idx] = value
        series = []
        repaired_count = 0
        for frame_idx in frames:
            if frame_idx in yolop_values:
                series.append(yolop_values[frame_idx])
                continue
            repaired = repo.repair_series_one_frame(yolop_values, frames, frame_idx) if len(missing) == 1 else None
            if repaired and repaired.repaired and repaired.value is not None:
                series.append(repaired.value)
                repaired_count += 1
                repair_stats[T2_VARIANT]["yolop_repaired_frames"] += 1
                repair_stats[T2_VARIANT]["yolop_repaired_sequences"] += int(repaired_count == 1)
            else:
                series.append(np.full(20, np.nan, dtype=float))
        yolop_arr = np.asarray(series, dtype=float)
        feature_row.update(temporal_summary(yolop_arr, "yolop", include_slope=False))
        feature_row["yolop_missing_frame_count"] = float(len(missing))
        feature_row["yolop_missing_any"] = float(len(missing) > 0)
        feature_row["yolop_valid_frame_fraction"] = float(np.mean(np.isfinite(yolop_arr).all(axis=1)))
        feature_row["yolop_repaired_frame_count"] = float(repaired_count)
    else:
        feature_row["yolop_skipped"] = 1.0
    return feature_row


def build_variant_tables(
    repo: MSLSTMStreamRepository,
    splits: dict[str, pd.DataFrame],
    include_yolop: bool,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, Any]]:
    ego_scaler_frame = _extract_train_ego_scaler_frame(repo, splits["train"])
    ego_mean, ego_std = fit_train_standardizer(ego_scaler_frame, list(ego_scaler_frame.columns))
    repair_stats: dict[str, dict[str, dict[str, int]]] = {T1_VARIANT: {}, T2_VARIANT: {}}
    for variant in [T1_VARIANT, T2_VARIANT]:
        for split_name in splits:
            repair_stats[variant][split_name] = {
                "bbox_repaired_frames": 0,
                "bbox_repaired_sequences": 0,
                "static_scene_missing_rows": 0,
            }
            if variant == T2_VARIANT:
                repair_stats[variant][split_name]["yolop_repaired_frames"] = 0
                repair_stats[variant][split_name]["yolop_repaired_sequences"] = 0
    variant_tables: dict[str, dict[str, pd.DataFrame]] = {T1_VARIANT: {}, T2_VARIANT: {}}
    for split_name, split_df in splits.items():
        t1_rows: list[dict[str, Any]] = []
        t2_rows: list[dict[str, Any]] = []
        for row in split_df.itertuples(index=False):
            row_meta = {
                "sequence_id": str(row.sequence_id),
                "split": str(row.split),
                "set_id": str(row.set_id),
                "video_id": str(row.video_id),
                "ped_id": str(row.ped_id),
                "start_frame": int(row.start_frame),
                "end_frame": int(row.end_frame),
                "label": int(row.label),
            }
            core = _build_core_sequence_features(
                row_meta,
                repo,
                ego_mean,
                ego_std,
                variant=T1_VARIANT,
                repair_stats={T1_VARIANT: repair_stats[T1_VARIANT][split_name]},
            )
            t1_rows.append(core)
            t2_rows.append(
                _add_t2_features(
                    core.copy(),
                    row_meta,
                    repo,
                    include_yolop=include_yolop,
                    repair_stats={T2_VARIANT: repair_stats[T2_VARIANT][split_name]},
                )
            )
        variant_tables[T1_VARIANT][split_name] = _numeric_summary_frame(t1_rows)
        variant_tables[T2_VARIANT][split_name] = _numeric_summary_frame(t2_rows)
        print(f"built {split_name}: rows={len(split_df)}")
    metadata = {
        "ego_train_mean": ego_mean.to_dict(),
        "ego_train_std": ego_std.to_dict(),
        "include_yolop": include_yolop,
        "repair_stats": repair_stats,
    }
    return variant_tables[T1_VARIANT], variant_tables[T2_VARIANT], metadata


def write_tables(variant_name: str, tables: dict[str, pd.DataFrame], output_dir: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    for split_name, frame in tables.items():
        path = output_dir / f"mslstm_frozen_{variant_name}_{split_name}.csv"
        frame.to_csv(path, index=False)
        paths[split_name] = str(path)
    return paths


def write_report(path: Path, t1: dict[str, pd.DataFrame], t2: dict[str, pd.DataFrame], metadata: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# MSLSTM Frozen T1/T2 Stream Extraction Report",
        "",
        "## Summary",
        "",
        "Built frozen-row tabular feature tables for the adapted MSLSTM-style Stage 1/2 teacher.",
        "This uses the frozen 16-frame / 30-60 TTE benchmark protocol and does not use the original local `pred_len=1` recurrent setup.",
        "Observed frames were extracted strictly from `start_frame` through `end_frame` inclusive for every frozen row; no future or target frames were accessed.",
        "",
        "## Variants",
        "",
        "- `T1`: bbox + ego_speed + ego_acc + ego_gyro",
        "- `T2`: T1 + traffic_light + sanitized static scene context" + (" + YOLOP" if metadata["include_yolop"] else " (YOLOP skipped)"),
        "- Sanitized static scene fields: `signalized`, `intersection`, `traffic_direction`, `num_lanes_bucket`",
        "- Excluded main-teacher streams: `ped_action`, `ped_look`, `ped_occlusion`, `pose`, `age`, `gender`, `crossing`, `intention_prob`, `crossing_point`, `critical_point`, `exp_start_point`",
        "",
        "## Train-Only Ego Scaling",
        "",
        f"- Means: `{metadata['ego_train_mean']}`",
        f"- Standard deviations: `{metadata['ego_train_std']}`",
        "",
        "## Feature Counts",
        "",
        f"- T1 feature count: `{len([c for c in t1['train'].columns if c not in META_COLUMNS])}`",
        f"- T2 feature count: `{len([c for c in t2['train'].columns if c not in META_COLUMNS])}`",
        "",
        "## Split-Wise Repair Counts",
        "",
    ]
    for split_name in ["train", "val", "test"]:
        split_t1 = metadata["repair_stats"][T1_VARIANT][split_name]
        split_t2 = metadata["repair_stats"][T2_VARIANT][split_name]
        t1_bbox_frames = int(t1[split_name].get("bbox_repaired_frame_count", pd.Series(dtype=float)).fillna(0).sum())
        t1_bbox_sequences = int((t1[split_name].get("bbox_repaired_frame_count", pd.Series(dtype=float)).fillna(0) > 0).sum())
        t2_bbox_frames = int(t2[split_name].get("bbox_repaired_frame_count", pd.Series(dtype=float)).fillna(0).sum())
        t2_bbox_sequences = int((t2[split_name].get("bbox_repaired_frame_count", pd.Series(dtype=float)).fillna(0) > 0).sum())
        lines.append(f"### {split_name}")
        lines.append("")
        lines.append(f"- `T1` bbox repaired frames: `{t1_bbox_frames}`")
        lines.append(f"- `T1` bbox repaired sequences: `{t1_bbox_sequences}`")
        lines.append(f"- `T2` bbox repaired frames: `{t2_bbox_frames}`")
        lines.append(f"- `T2` bbox repaired sequences: `{t2_bbox_sequences}`")
        if metadata["include_yolop"]:
            lines.append(f"- `T2` YOLOP repaired frames: `{split_t2['yolop_repaired_frames']}`")
            lines.append(f"- `T2` YOLOP repaired sequences: `{split_t2['yolop_repaired_sequences']}`")
        lines.append(f"- `T2` static scene missing rows: `{split_t2['static_scene_missing_rows']}`")
        lines.append("")
    lines.extend(["## Output Tables", ""])
    for variant_name, tables in [(T1_VARIANT, t1), (T2_VARIANT, t2)]:
        for split_name in ["train", "val", "test"]:
            lines.append(f"- `{output_dir / f'mslstm_frozen_{variant_name}_{split_name}.csv'}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frozen-row T1/T2 MSLSTM-style teacher tables.")
    parser.add_argument("--mslstm_root", type=Path, default=Path("/home/abdelrahman/Hadwa/MSLSTM-PID"))
    parser.add_argument("--splits_dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed/teacher"))
    parser.add_argument("--report_path", type=Path, default=Path("MSLSTM_FROZEN_T1_T2_STREAM_EXTRACTION_REPORT.md"))
    parser.add_argument("--include_yolop", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = load_split_csvs(args.splits_dir)
    repo = MSLSTMStreamRepository(args.mslstm_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t1, t2, metadata = build_variant_tables(repo, splits, include_yolop=args.include_yolop)
    t1_paths = write_tables(T1_VARIANT, t1, args.output_dir)
    t2_paths = write_tables(T2_VARIANT, t2, args.output_dir)
    write_report(args.report_path, t1, t2, metadata, args.output_dir)
    print(json.dumps({"T1": t1_paths, "T2": t2_paths, "include_yolop": metadata["include_yolop"]}, indent=2))


if __name__ == "__main__":
    main()
