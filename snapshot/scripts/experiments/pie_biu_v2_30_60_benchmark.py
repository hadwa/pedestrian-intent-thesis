"""Benchmark BIU v2 vs BIU v1 on the main PIE benchmark (16 obs frames, TTE 30-60, overlap 0.6, test=set03).

PIE counterpart of experiment 009 (scripts/experiments/jaad_biu_v2_benchmark_comparison.py).
Reuses the JAAD script's model builders, threshold selection, metric computation, and
feature-set prefix helpers by importing it as a module. BIU v2 features are extracted
per window with src/biu_v2_extractor.py:extract_biu_v2 from the PIE keypoints JSONL.

Inputs (all read-only):
- Splits: /home/abdelrahman/Hadwa/BIU-PID-Distillation/data/splits_pie_wacv_pcpa_30_60_overlap06/{train,val,test}_sequences.csv
- Existing bbox / BIU v1 tables: /home/abdelrahman/Hadwa/BIU-PID-Distillation/data/processed/features_wacv_pcpa_30_60_overlap06/{bbox,biu}_{split}.csv
- Keypoints: /home/abdelrahman/Hadwa/pedestrian-intent-biu/data/data_keypoints_all.jsonl
- Per-frame bboxes for direction fallback: PIE CVAT XMLs via archive src/features/bbox_features.py:load_pie_bbox_annotations

Outputs: experiments/012_pie_biu_v2_30_60_benchmark/{metrics,predictions}/
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CLEAN_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean")
ARCHIVE_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation")
SCRIPTS_DIR = CLEAN_ROOT / "scripts/experiments"
SRC_ROOT = CLEAN_ROOT / "src"
for p in (str(SCRIPTS_DIR), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import jaad_biu_v2_benchmark_comparison as jaad  # noqa: E402  (reused helpers)
from biu_v2_extractor import extract_biu_v2  # noqa: E402

SPLIT_DIR = ARCHIVE_ROOT / "data/splits_pie_wacv_pcpa_30_60_overlap06"
FEATURE_DIR = ARCHIVE_ROOT / "data/processed/features_wacv_pcpa_30_60_overlap06"
KEYPOINTS_JSONL = Path("/home/abdelrahman/Hadwa/pedestrian-intent-biu/data/data_keypoints_all.jsonl")
PIE_ANNOTATIONS_DIR = Path("/home/abdelrahman/Hadwa/MSLSTM-PID/PIE/annotations/annotations")
ARCHIVE_BBOX_FEATURES_PY = ARCHIVE_ROOT / "src/features/bbox_features.py"
ANCHOR_SOURCE = ARCHIVE_ROOT / "results/wacv_pcpa_30_60_overlap06/lightweight_baseline_metrics.csv"
DICT_PATH = CLEAN_ROOT / "experiments/008_biu_v2_design_spec/metrics/biu_v2_feature_dictionary.csv"

OUT_ROOT = CLEAN_ROOT / "experiments/012_pie_biu_v2_30_60_benchmark"
METRICS_DIR = OUT_ROOT / "metrics"
PREDICTIONS_DIR = OUT_ROOT / "predictions"

SPLITS = ["train", "val", "test"]
SEEDS = jaad.SEEDS  # [42, 43, 44]
MODEL_ORDER = jaad.MODEL_ORDER  # ["logreg", "rf", "xgb"]
FEATURE_SET_ORDER = ["bbox_only", "biu_v1_only", "biu_v1_plus_bbox", "biu_v2_core", "biu_v2_core_plus_bbox"]

# Archive published anchors (seed 42, validation-F1 threshold), traceable to ANCHOR_SOURCE.
ANCHORS = {
    ("bbox_only", "rf"): {"archive_feature_set": "bbox", "archive_model": "random_forest", "test_f1": 0.857564},
    ("biu_v1_plus_bbox", "rf"): {"archive_feature_set": "biu_bbox", "archive_model": "random_forest", "test_f1": 0.866359},
}

# Meta columns: union of the JAAD script meta set and the archive PIE trainer metadata
# (src/students/datasets.py FEATURE_METADATA_COLUMNS) so numeric QC columns such as
# bbox_missing_frame_count are excluded exactly as in the archived benchmark (bbox=53 features).
META_COLS = set(jaad.META_COLS) | {
    "time_to_event_frames",
    "event_frame_definition",
    "track_type",
    "has_observed_irrelevant_cross",
    "first_crossing_onset_frame",
    "official_crossing_point",
    "official_crossing_point_abs_diff",
    "bbox_missing_frame_count",
    "bbox_missing_any",
    "bbox_valid_frame_fraction",
    "bbox_repaired_frame_count",
    "bbox_repaired_frames",
    "bbox_repair_method",
}


def _load_archive_bbox_module():
    spec = importlib.util.spec_from_file_location("archive_bbox_features", ARCHIVE_BBOX_FEATURES_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules["archive_bbox_features"] = module
    spec.loader.exec_module(module)
    return module


def _numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in META_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def _load_split_rows() -> pd.DataFrame:
    dfs = []
    for split in SPLITS:
        df = pd.read_csv(SPLIT_DIR / f"{split}_sequences.csv")
        if "split" not in df.columns:
            df["split"] = split
        assert (df["split"] == split).all(), f"split column mismatch in {split}_sequences.csv"
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    # The PIE 30-60 protocol uses consecutive raw frames; observed frames are recoverable
    # as start_frame..end_frame inclusive. Assert this holds for every window.
    assert (out["frame_step"] == 1).all(), "expected frame_step == 1 for all windows"
    assert (out["frame_sampling"] == "consecutive").all(), "expected consecutive frame sampling"
    assert (out["observation_length"] == 16).all(), "expected 16 observed frames"
    assert ((out["end_frame"] - out["start_frame"]) == 15).all(), "expected end_frame - start_frame == 15"
    assert out["sequence_id"].is_unique, "sequence_id must be unique across splits"
    return out


def _observed_frames(row: pd.Series) -> list[int]:
    return list(range(int(row["start_frame"]), int(row["end_frame"]) + 1))


def _needed_frame_keys(split_df: pd.DataFrame) -> set[tuple[str, str, str, int]]:
    needed: set[tuple[str, str, str, int]] = set()
    for _, row in split_df.iterrows():
        for frame_idx in _observed_frames(row):
            needed.add((str(row["set_id"]), str(row["video_id"]), str(row["ped_id"]), frame_idx))
    return needed


def _load_keypoint_index(needed: set[tuple[str, str, str, int]]) -> dict[tuple[str, str, str, int], dict]:
    out: dict[tuple[str, str, str, int], dict] = {}
    with KEYPOINTS_JSONL.open("r", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            key = (str(rec["set"]), str(rec["video"]), str(rec["pid"]), int(rec["frame_idx"]))
            if key in needed:
                out[key] = rec
    return out


def _load_bbox_index(needed: set[tuple[str, str, str, int]]) -> dict[tuple[str, str, str, int], list[float]]:
    archive_bbox = _load_archive_bbox_module()
    bbox_df = archive_bbox.load_pie_bbox_annotations(PIE_ANNOTATIONS_DIR)
    out: dict[tuple[str, str, str, int], list[float]] = {}
    for set_id, video_id, ped_id, frame_idx, x1, y1, x2, y2 in bbox_df[
        ["set_id", "video_id", "ped_id", "frame_idx", "x1", "y1", "x2", "y2"]
    ].itertuples(index=False):
        key = (str(set_id), str(video_id), str(ped_id), int(frame_idx))
        if key in needed:
            out[key] = [float(x1), float(y1), float(x2), float(y2)]
    return out


def _extract_v2_for_rows(
    split_df: pd.DataFrame,
    kp_index: dict[tuple[str, str, str, int], dict],
    bbox_index: dict[tuple[str, str, str, int], list[float]],
) -> tuple[pd.DataFrame, list[dict], dict[str, int]]:
    rows = []
    missing = []
    bbox_fallback_stats = {"windows_with_full_bbox": 0, "windows_with_partial_bbox": 0, "windows_without_bbox": 0}
    for _, rec in split_df.iterrows():
        frames = _observed_frames(rec)
        kp_rows = []
        missing_frames = []
        bbox_seq = []
        for frame in frames:
            key = (str(rec["set_id"]), str(rec["video_id"]), str(rec["ped_id"]), int(frame))
            item = kp_index.get(key)
            if item is None:
                missing_frames.append(int(frame))
            else:
                kp_rows.append(item)
            bbox = bbox_index.get(key)
            if bbox is not None:
                bbox_seq.append(bbox)
        if missing_frames:
            missing.append(
                {
                    "sequence_id": rec["sequence_id"],
                    "set_id": rec["set_id"],
                    "video_id": rec["video_id"],
                    "ped_id": rec["ped_id"],
                    "split": rec["split"],
                    "label": int(rec["label"]),
                    "n_missing_frames": len(missing_frames),
                    "missing_frames": json.dumps(missing_frames),
                }
            )
            continue
        if len(bbox_seq) == len(frames):
            bbox_fallback_stats["windows_with_full_bbox"] += 1
        elif bbox_seq:
            bbox_fallback_stats["windows_with_partial_bbox"] += 1
        else:
            bbox_fallback_stats["windows_without_bbox"] += 1
        coords = np.asarray([r["keypoints"] for r in kp_rows], dtype=float)
        conf = np.asarray([r["kp_conf"] for r in kp_rows], dtype=float)
        bbox_arr = np.asarray(bbox_seq, dtype=float) if bbox_seq else None
        feats = extract_biu_v2(coords, conf, bbox_sequence=bbox_arr)
        base = {
            "sequence_id": rec["sequence_id"],
            "split": rec["split"],
            "label": int(rec["label"]),
            "set_id": rec["set_id"],
            "video_id": rec["video_id"],
            "ped_id": rec["ped_id"],
            "track_id": rec["track_id"],
            "start_frame": int(rec["start_frame"]),
            "end_frame": int(rec["end_frame"]),
        }
        base.update(feats)
        rows.append(base)
    return pd.DataFrame(rows), missing, bbox_fallback_stats


def _load_existing_feature_frame(name: str) -> pd.DataFrame:
    dfs = []
    for split in SPLITS:
        path = FEATURE_DIR / f"{name}_{split}.csv"
        df = pd.read_csv(path)
        if "split" not in df.columns:
            df["split"] = split
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _merge_base_and_features(base: pd.DataFrame, feat_df: pd.DataFrame) -> pd.DataFrame:
    keep = ["sequence_id"] + [c for c in feat_df.columns if c != "sequence_id" and c not in base.columns]
    return base.merge(feat_df[keep], on="sequence_id", how="inner")


def _build_feature_sets(split_df: pd.DataFrame, v2_df_raw: pd.DataFrame, feature_dict: pd.DataFrame):
    base = split_df.copy()
    bbox_df = _merge_base_and_features(base, _load_existing_feature_frame("bbox"))
    biu_v1_df = _merge_base_and_features(base, _load_existing_feature_frame("biu"))
    v2_df = _merge_base_and_features(base, v2_df_raw)

    bbox_cols = _numeric_feature_cols(bbox_df)
    v1_cols = _numeric_feature_cols(biu_v1_df)

    anatomical_prefixes = jaad._base_feature_prefixes(feature_dict, ["original_anatomical_biu"])
    lower_prefixes = jaad._base_feature_prefixes(feature_dict, ["lower_body_commitment_biu"])
    torso_prefixes = jaad._base_feature_prefixes(feature_dict, ["torso_center_commitment_biu"])
    v2_anatomical_cols = jaad._select_prefixed_columns(v2_df, anatomical_prefixes)
    v2_lower_cols = jaad._select_prefixed_columns(v2_df, lower_prefixes)
    v2_torso_cols = jaad._select_prefixed_columns(v2_df, torso_prefixes)
    v2_core_cols = sorted(set(v2_anatomical_cols + v2_lower_cols + v2_torso_cols))

    v1_plus_bbox_df = biu_v1_df[[c for c in biu_v1_df.columns if c not in bbox_cols]].merge(
        bbox_df[["sequence_id"] + bbox_cols], on="sequence_id", how="inner"
    )
    v2_plus_bbox_df = v2_df[[c for c in v2_df.columns if c not in bbox_cols]].merge(
        bbox_df[["sequence_id"] + bbox_cols], on="sequence_id", how="inner"
    )

    feature_sets = {
        "bbox_only": bbox_df,
        "biu_v1_only": biu_v1_df,
        "biu_v1_plus_bbox": v1_plus_bbox_df,
        "biu_v2_core": v2_df,
        "biu_v2_core_plus_bbox": v2_plus_bbox_df,
    }
    column_map = {
        "bbox_only": bbox_cols,
        "biu_v1_only": v1_cols,
        "biu_v1_plus_bbox": sorted(set(v1_cols + bbox_cols)),
        "biu_v2_core": v2_core_cols,
        "biu_v2_core_plus_bbox": sorted(set(v2_core_cols + bbox_cols)),
    }
    notes_map = {
        "bbox_only": "Archived PIE enriched bbox features (data/processed/features_wacv_pcpa_30_60_overlap06/bbox_*.csv); QC metadata columns excluded as in the archive trainer.",
        "biu_v1_only": "Archived BIU v1 features (data/processed/features_wacv_pcpa_30_60_overlap06/biu_*.csv), as used in the published PIE benchmark.",
        "biu_v1_plus_bbox": "BIU v1 concatenated with bbox features (matches archived biu_bbox set).",
        "biu_v2_core": "Direction-aware BIU v2 core: original anatomical + lower-body commitment + torso/body-center commitment groups from the BIU v2 feature dictionary; extracted here from PIE keypoints JSONL.",
        "biu_v2_core_plus_bbox": "BIU v2 core concatenated with bbox features.",
    }
    definitions = []
    for feature_set in FEATURE_SET_ORDER:
        df = feature_sets[feature_set]
        definitions.append(
            {
                "feature_set": feature_set,
                "n_features": len(column_map[feature_set]),
                "n_rows": len(df),
                "n_unique_sequence_ids": int(df["sequence_id"].nunique()),
                "missing_feature_rows_vs_split": int(len(split_df) - df["sequence_id"].nunique()),
                "notes": notes_map[feature_set],
                "feature_columns_json": json.dumps(column_map[feature_set]),
            }
        )
    return feature_sets, column_map, definitions


def _run_benchmark(feature_sets: dict[str, pd.DataFrame], column_map: dict[str, list[str]]):
    results = []
    for feature_set in FEATURE_SET_ORDER:
        cols = column_map[feature_set]
        df = feature_sets[feature_set]
        train_df = df[df["split"] == "train"]
        val_df = df[df["split"] == "val"]
        test_df = df[df["split"] == "test"]
        X_train = train_df[cols].astype(float)
        y_train = train_df["label"].astype(int).to_numpy()
        X_val = val_df[cols].astype(float)
        y_val = val_df["label"].astype(int).to_numpy()
        X_test = test_df[cols].astype(float)
        y_test = test_df["label"].astype(int).to_numpy()

        for seed in SEEDS:
            for model_name in MODEL_ORDER:
                pipe = jaad._build_model(model_name, seed, y_train)
                if pipe is None:
                    print(f"WARNING: model {model_name} unavailable, skipping")
                    continue
                t0 = time.time()
                pipe.fit(X_train, y_train)
                p_val = pipe.predict_proba(X_val)[:, 1]
                p_test = pipe.predict_proba(X_test)[:, 1]

                pred_rows = pd.concat(
                    [
                        pd.DataFrame({"sequence_id": val_df["sequence_id"].to_numpy(), "split": "val", "label": y_val, "prob": p_val}),
                        pd.DataFrame({"sequence_id": test_df["sequence_id"].to_numpy(), "split": "test", "label": y_test, "prob": p_test}),
                    ],
                    ignore_index=True,
                )
                pred_rows.insert(0, "feature_set", feature_set)
                pred_rows.insert(1, "model_name", model_name)
                pred_rows.insert(2, "seed", seed)
                pred_rows.to_csv(PREDICTIONS_DIR / f"{feature_set}__{model_name}__seed{seed}.csv", index=False)

                for sel in jaad._select_thresholds(y_val, p_val):
                    row = {
                        "feature_set": feature_set,
                        "model_name": model_name,
                        "seed": seed,
                        "threshold_rule": sel.rule,
                        "threshold_status": sel.status,
                        "selected_threshold": sel.threshold,
                        "n_features": len(cols),
                        "train_rows": len(train_df),
                        "val_rows": len(val_df),
                        "test_rows": len(test_df),
                        "fit_seconds": time.time() - t0,
                    }
                    if sel.status == "available" and sel.threshold is not None:
                        for k, v in sel.val_metrics.items():
                            row[f"val_{k}"] = v
                        row.update({f"test_{k}": v for k, v in jaad._compute_metrics(y_test, p_test, float(sel.threshold)).items()})
                    results.append(row)
    return pd.DataFrame(results)


def _aggregate_results(results_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in results_df.columns if c.startswith("test_") or c.startswith("val_") or c == "selected_threshold"]
    agg = results_df.groupby(["feature_set", "model_name", "threshold_rule", "threshold_status"], dropna=False)[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


def _delta_table(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparisons = [
        ("biu_v2_core", "biu_v1_only"),
        ("biu_v2_core_plus_bbox", "biu_v1_plus_bbox"),
    ]
    metrics = ["test_f1", "test_precision", "test_recall", "test_pr_auc", "test_roc_auc", "test_balanced_accuracy"]
    for rule in ["fixed_0_5", "val_f1"]:
        sub = results_df[(results_df["threshold_rule"] == rule) & (results_df["threshold_status"] == "available")]
        for model_name in MODEL_ORDER:
            for v2_set, v1_set in comparisons:
                v2 = sub[(sub["feature_set"] == v2_set) & (sub["model_name"] == model_name)]
                v1 = sub[(sub["feature_set"] == v1_set) & (sub["model_name"] == model_name)]
                if v2.empty or v1.empty:
                    continue
                row = {
                    "threshold_rule": rule,
                    "model_name": model_name,
                    "comparison": f"{v2_set}_vs_{v1_set}",
                    "n_seeds": int(min(len(v2), len(v1))),
                }
                for m in metrics:
                    row[f"{m}_v2_mean"] = float(v2[m].mean())
                    row[f"{m}_v1_mean"] = float(v1[m].mean())
                    row[f"{m}_delta"] = float(v2[m].mean() - v1[m].mean())
                rows.append(row)
    return pd.DataFrame(rows)


def _sanity_anchor_table(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sub = results_df[(results_df["threshold_rule"] == "val_f1") & (results_df["threshold_status"] == "available")]
    for (feature_set, model_name), anchor in ANCHORS.items():
        ours = sub[(sub["feature_set"] == feature_set) & (sub["model_name"] == model_name)]
        if ours.empty:
            continue
        seed42 = ours[ours["seed"] == 42]
        mean_f1 = float(ours["test_f1"].mean())
        rows.append(
            {
                "feature_set": feature_set,
                "model_name": model_name,
                "archive_feature_set": anchor["archive_feature_set"],
                "archive_model": anchor["archive_model"],
                "archive_test_f1_seed42": anchor["test_f1"],
                "this_run_test_f1_seed42": float(seed42["test_f1"].iloc[0]) if not seed42.empty else float("nan"),
                "this_run_test_f1_mean_seeds": mean_f1,
                "this_run_test_f1_std_seeds": float(ours["test_f1"].std()),
                "abs_deviation_seed42": abs(float(seed42["test_f1"].iloc[0]) - anchor["test_f1"]) if not seed42.empty else float("nan"),
                "abs_deviation_mean": abs(mean_f1 - anchor["test_f1"]),
                "anchor_source": str(ANCHOR_SOURCE),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    feature_dict = pd.read_csv(DICT_PATH)
    split_df = _load_split_rows()
    print(f"split rows: {len(split_df)} "
          f"(train={int((split_df['split']=='train').sum())}, val={int((split_df['split']=='val').sum())}, test={int((split_df['split']=='test').sum())})")

    t0 = time.time()
    needed = _needed_frame_keys(split_df)
    print(f"needed (set, video, pid, frame) keys: {len(needed)}")
    kp_index = _load_keypoint_index(needed)
    print(f"keypoint index size: {len(kp_index)} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    bbox_index = _load_bbox_index(needed)
    print(f"bbox index size: {len(bbox_index)} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    v2_df_raw, missing_rows, bbox_stats = _extract_v2_for_rows(split_df, kp_index, bbox_index)
    print(f"BIU v2 extracted for {len(v2_df_raw)} windows; missing-keypoint windows: {len(missing_rows)} ({time.time() - t0:.1f}s)")
    print(f"bbox availability for direction fallback: {bbox_stats}")

    missing_df = pd.DataFrame(
        missing_rows,
        columns=["sequence_id", "set_id", "video_id", "ped_id", "split", "label", "n_missing_frames", "missing_frames"],
    )
    missing_df.to_csv(METRICS_DIR / "pie_biu_v2_missing_keypoint_rows.csv", index=False)

    feature_sets, column_map, definitions = _build_feature_sets(split_df, v2_df_raw, feature_dict)
    pd.DataFrame(definitions).to_csv(METRICS_DIR / "pie_biu_v2_feature_set_definitions.csv", index=False)

    results_df = _run_benchmark(feature_sets, column_map)
    mean_std_df = _aggregate_results(results_df)
    delta_df = _delta_table(results_df)
    anchor_df = _sanity_anchor_table(results_df)

    results_df.to_csv(METRICS_DIR / "pie_biu_v2_benchmark_results.csv", index=False)
    mean_std_df.to_csv(METRICS_DIR / "pie_biu_v2_benchmark_results_mean_std.csv", index=False)
    delta_df.to_csv(METRICS_DIR / "pie_biu_v2_vs_v1_delta_table.csv", index=False)
    anchor_df.to_csv(METRICS_DIR / "pie_biu_v2_sanity_anchor_comparison.csv", index=False)
    pd.DataFrame([{**bbox_stats, "missing_keypoint_windows": len(missing_rows), "total_windows": len(split_df)}]).to_csv(
        METRICS_DIR / "pie_biu_v2_extraction_coverage.csv", index=False
    )

    print(json.dumps({
        "results_rows": int(len(results_df)),
        "missing_keypoint_windows": int(len(missing_rows)),
        "missing_fraction": float(len(missing_rows) / len(split_df)),
        "prediction_files": len(list(PREDICTIONS_DIR.glob("*.csv"))),
    }, indent=2))


if __name__ == "__main__":
    main()
