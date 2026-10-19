"""Matched-cohort execution helpers for the final confidence benchmark."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from scripts.experiments import jaad_biu_v2_benchmark_comparison as benchmark
from src.benchmark.final_confidence_feature_views import (
    build_biu_joint_map,
    make_feature_view,
)
from src.benchmark.final_confidence_sources import (
    collect_global_confidence_columns,
    resolve_benchmark_sources,
    validate_kp_xy_vs_xyc_is_additive,
)
from src.biu_v2_extractor import extract_biu_v2


MATCH_KEYS = ["sequence_id", "split"]
CLEAN_ROOT = Path(__file__).resolve().parents[2]
BIU_V2_EXTRACTOR_PATH = CLEAN_ROOT / "src/biu_v2_extractor.py"
SPLITS = ("train", "val", "test")
SEEDS = tuple(benchmark.SEEDS)
MODEL_ORDER = tuple(benchmark.MODEL_ORDER)


def _extract_complete_biu_v2(
    coords: np.ndarray,
    confidence: np.ndarray,
    bbox_sequence: np.ndarray | None,
) -> dict[str, float | int | str]:
    return extract_biu_v2(
        coords,
        confidence,
        bbox_sequence=bbox_sequence,
        include_optional=True,
    )


def _git_commit_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=CLEAN_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _ordered_sha256(values: list[str]) -> str:
    return sha256("\n".join(values).encode("utf-8")).hexdigest()


def _biu_geometry_columns(frame: pd.DataFrame) -> list[str]:
    names = build_biu_joint_map()
    return [
        column
        for column in frame.columns
        if any(
            column == prefix or column.startswith(f"{prefix}_")
            for name in names
            for prefix in (name, f"biu__{name}")
        )
    ]


def fingerprint_reconstruction(
    frame: pd.DataFrame, *, config_key: tuple[str, str]
) -> dict[str, str]:
    """Fingerprint code, ordered BIU-v2 schema, and split-specific row order."""
    missing = {"sequence_id", "split"} - set(frame.columns)
    if missing:
        raise ValueError(f"Cannot fingerprint reconstruction without {sorted(missing)}")
    if frame["sequence_id"].isna().any() or frame["sequence_id"].duplicated().any():
        raise ValueError(f"Reconstruction {config_key} has missing or duplicate sequence IDs")

    fingerprints = {
        "git_commit_sha": _git_commit_sha(),
        "biu_v2_extractor_sha256": sha256(BIU_V2_EXTRACTOR_PATH.read_bytes()).hexdigest(),
        "biu_v2_feature_schema_sha256": _ordered_sha256(_biu_geometry_columns(frame)),
    }
    for split in SPLITS:
        sequence_ids = frame.loc[frame["split"].eq(split), "sequence_id"].astype(str).tolist()
        if sequence_ids:
            fingerprints[f"{split}_sequence_id_sha256"] = _ordered_sha256(sequence_ids)
    return fingerprints


def _load_frozen_split_rows(source: dict[str, str]) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        frame = pd.read_csv(source[f"{split}_split_path"])
        if "split" in frame.columns and not frame["split"].astype(str).eq(split).all():
            raise ValueError(f"Frozen {split} rows contain conflicting split membership")
        frame["split"] = split
        frame["sequence_id"] = frame["sequence_id"].astype(str)
        if frame["sequence_id"].isna().any() or frame["sequence_id"].duplicated().any():
            raise ValueError(f"Frozen {split} rows have missing or duplicate sequence IDs")
        rows.append(frame)
    frozen = pd.concat(rows, ignore_index=True)
    if frozen["sequence_id"].duplicated().any():
        raise ValueError("Frozen sequence IDs overlap across splits")
    return frozen


def _assert_frozen_sequence_order(
    reconstructed: pd.DataFrame,
    frozen: pd.DataFrame,
    *,
    config_key: tuple[str, str],
) -> None:
    required = {"sequence_id", "split"}
    if required - set(reconstructed.columns):
        raise ValueError(f"Reconstructed BIU-v2 for {config_key} lacks sequence_id/split")
    if reconstructed["sequence_id"].isna().any() or reconstructed["sequence_id"].duplicated().any():
        raise ValueError(f"Reconstructed BIU-v2 for {config_key} has invalid sequence IDs")
    for split in SPLITS:
        expected = frozen.loc[frozen["split"].eq(split), "sequence_id"].astype(str).tolist()
        actual = (
            reconstructed.loc[reconstructed["split"].eq(split), "sequence_id"]
            .astype(str)
            .tolist()
        )
        if actual != expected:
            raise ValueError(
                f"Reconstructed BIU-v2 sequence order does not match frozen {config_key} {split} rows"
            )


def _reconstruct_jaad_biu_v2(
    config_key: tuple[str, str], source: dict[str, str], frozen: pd.DataFrame
) -> pd.DataFrame:
    horizon = config_key[1]
    runner_source = benchmark.HORIZON_CONFIGS[horizon]
    if Path(source["split_dir"]) != Path(runner_source["split_dir"]):
        raise ValueError(f"JAAD runner split lineage differs from resolved source for {config_key}")

    runner_rows = benchmark._load_split_rows(horizon)
    _assert_frozen_sequence_order(runner_rows, frozen, config_key=config_key)
    split_rows = {
        split: runner_rows.loc[runner_rows["split"].eq(split)].copy() for split in SPLITS
    }
    needed = benchmark._needed_frame_keys(split_rows)
    keypoints = benchmark._load_keypoint_index(Path(runner_source["keypoints_jsonl"]), needed)
    rows = []
    missing = []
    for _, record in runner_rows.iterrows():
        frames = benchmark._parse_json_list(record["observed_frames_json"])
        kp_rows = [
            keypoints.get((str(record["video_id"]), str(record["ped_id"]), int(frame)))
            for frame in frames
        ]
        missing_frames = [
            int(frame) for frame, keypoint in zip(frames, kp_rows) if keypoint is None
        ]
        if missing_frames:
            missing.append(
                {"sequence_id": record["sequence_id"], "missing_frames": missing_frames}
            )
            continue
        coords = np.asarray([keypoint["keypoints"] for keypoint in kp_rows], dtype=float)
        confidence = np.asarray([keypoint["kp_conf"] for keypoint in kp_rows], dtype=float)
        bbox_sequence = np.asarray(
            benchmark._parse_json_list(record["bbox_sequence_json"]), dtype=float
        )
        row = {"sequence_id": record["sequence_id"], "split": record["split"]}
        row.update(_extract_complete_biu_v2(coords, confidence, bbox_sequence))
        rows.append(row)
    reconstructed = pd.DataFrame(rows)
    if missing:
        first = missing[0]
        raise ValueError(
            f"JAAD BIU-v2 reconstruction is missing cached keypoints for {len(missing)} rows; first={first}"
        )
    return reconstructed


def _reconstruct_pie_biu_v2(
    config_key: tuple[str, str], source: dict[str, str], frozen: pd.DataFrame
) -> pd.DataFrame:
    from scripts.experiments import pie_biu_v2_30_60_benchmark as pie

    if Path(source["split_dir"]) != pie.SPLIT_DIR:
        raise ValueError(f"PIE runner split lineage differs from resolved source for {config_key}")
    runner_rows = pie._load_split_rows()
    _assert_frozen_sequence_order(runner_rows, frozen, config_key=config_key)
    needed = pie._needed_frame_keys(runner_rows)
    keypoints = pie._load_keypoint_index(needed)
    bboxes = pie._load_bbox_index(needed)
    rows = []
    missing = []
    for _, record in runner_rows.iterrows():
        frames = pie._observed_frames(record)
        keys = [
            (
                str(record["set_id"]),
                str(record["video_id"]),
                str(record["ped_id"]),
                int(frame),
            )
            for frame in frames
        ]
        kp_rows = [keypoints.get(key) for key in keys]
        missing_frames = [
            int(frame) for frame, keypoint in zip(frames, kp_rows) if keypoint is None
        ]
        if missing_frames:
            missing.append(
                {"sequence_id": record["sequence_id"], "missing_frames": missing_frames}
            )
            continue
        coords = np.asarray([keypoint["keypoints"] for keypoint in kp_rows], dtype=float)
        confidence = np.asarray([keypoint["kp_conf"] for keypoint in kp_rows], dtype=float)
        bbox_rows = [bboxes[key] for key in keys if key in bboxes]
        bbox_sequence = np.asarray(bbox_rows, dtype=float) if bbox_rows else None
        row = {"sequence_id": record["sequence_id"], "split": record["split"]}
        row.update(_extract_complete_biu_v2(coords, confidence, bbox_sequence))
        rows.append(row)
    reconstructed = pd.DataFrame(rows)
    if missing:
        first = missing[0]
        raise ValueError(
            f"PIE BIU-v2 reconstruction is missing cached keypoints for {len(missing)} rows; first={first}"
        )
    return reconstructed


def _load_cached_rows(
    source: dict[str, str], frozen: pd.DataFrame, table_key: str
) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        cached = pd.read_csv(source[f"{split}_{table_key}_path"])
        cached["sequence_id"] = cached["sequence_id"].astype(str)
        if cached["sequence_id"].duplicated().any():
            raise ValueError(f"Cached {table_key} {split} rows have duplicate sequence IDs")
        expected = frozen.loc[frozen["split"].eq(split), "sequence_id"].astype(str).tolist()
        missing = set(expected) - set(cached["sequence_id"])
        if missing:
            raise ValueError(f"Cached {table_key} {split} rows are missing frozen sequence IDs")
        matched = cached.set_index("sequence_id", verify_integrity=True).loc[expected].reset_index()
        if "split" in matched.columns and not matched["split"].astype(str).eq(split).all():
            raise ValueError(f"Cached {table_key} rows conflict with frozen {split} membership")
        matched["split"] = split
        rows.append(matched)
    return pd.concat(rows, ignore_index=True)


def _bbox_feature_columns(frame: pd.DataFrame, dataset: str) -> list[str]:
    if dataset == "jaad":
        return benchmark._numeric_feature_cols(frame)
    from scripts.experiments import pie_biu_v2_30_60_benchmark as pie

    return pie._numeric_feature_cols(frame)


def _assemble_canonical_frame(
    config_key: tuple[str, str],
    source: dict[str, str],
    frozen: pd.DataFrame,
    reconstructed: pd.DataFrame,
) -> pd.DataFrame:
    canonical = frozen.copy()
    reconstructed_columns = [
        "sequence_id",
        *[
            column
            for column in reconstructed.columns
            if column != "sequence_id" and column not in canonical.columns
        ],
    ]
    canonical = canonical.merge(
        reconstructed[reconstructed_columns],
        on="sequence_id",
        how="left",
        sort=False,
        validate="one_to_one",
    )

    kp_rows = _load_cached_rows(source, frozen, "kp_xy")
    kp_columns = [
        column
        for column in kp_rows.columns
        if column.startswith("kp__") or column.startswith("kpconf__")
    ]
    canonical = canonical.merge(
        kp_rows[["sequence_id", *kp_columns]],
        on="sequence_id",
        how="left",
        sort=False,
        validate="one_to_one",
    )

    bbox_rows = _load_cached_rows(source, frozen, "bbox")
    bbox_columns = _bbox_feature_columns(bbox_rows, config_key[0])
    bbox_features = bbox_rows[["sequence_id", *bbox_columns]].rename(
        columns={column: f"bbox__{column}" for column in bbox_columns}
    )
    canonical = canonical.merge(
        bbox_features,
        on="sequence_id",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    if "lower_body_conf_mean" in canonical.columns:
        canonical["lower_body_kp_conf_mean"] = canonical["lower_body_conf_mean"]
    canonical["coord_norm_version"] = "v2"
    _assert_frozen_sequence_order(canonical, frozen, config_key=config_key)
    return canonical


def reconstruct_canonical_biu_v2(config_key: tuple[str, str]) -> pd.DataFrame:
    """Reconstruct one BIU-v2 frame and merge trusted KP/BBox views in frozen order."""
    sources = resolve_benchmark_sources()
    if config_key not in sources:
        raise ValueError(f"Unsupported final-confidence benchmark configuration: {config_key}")
    source = sources[config_key]
    frozen = _load_frozen_split_rows(source)
    if config_key[0] == "jaad":
        reconstructed = _reconstruct_jaad_biu_v2(config_key, source, frozen)
    elif config_key == ("pie", "30_60"):
        reconstructed = _reconstruct_pie_biu_v2(config_key, source, frozen)
    else:
        raise ValueError(f"No benchmark-aligned BIU-v2 reconstruction for {config_key}")
    _assert_frozen_sequence_order(reconstructed, frozen, config_key=config_key)
    canonical = _assemble_canonical_frame(
        config_key, source, frozen, reconstructed
    )
    canonical.attrs["reconstruction_fingerprint"] = fingerprint_reconstruction(
        canonical, config_key=config_key
    )
    return canonical


def run_representation_benchmark(
    config_key: tuple[str, str], representation: str
) -> pd.DataFrame:
    """Fit the frozen benchmark candidates and retain each seed's val-F1 winner."""
    canonical = reconstruct_canonical_biu_v2(config_key)
    fingerprints = fingerprint_reconstruction(canonical, config_key=config_key)
    joint_map = build_biu_joint_map()
    global_confidence = collect_global_confidence_columns(canonical)

    if representation in {"kp_xy", "kp_xyc"}:
        xy_frame, xy_columns = make_feature_view(
            canonical,
            "kp_xy",
            global_conf_cols=global_confidence,
            biu_joint_map=joint_map,
        )
        xyc_frame, xyc_columns = make_feature_view(
            canonical,
            "kp_xyc",
            global_conf_cols=global_confidence,
            biu_joint_map=joint_map,
        )
        validate_kp_xy_vs_xyc_is_additive(
            xy_frame[[*MATCH_KEYS, "coord_norm_version", *xy_columns]],
            xyc_frame[[*MATCH_KEYS, "coord_norm_version", *xyc_columns]],
        )

    view, feature_columns = make_feature_view(
        canonical,
        representation,
        global_conf_cols=global_confidence,
        biu_joint_map=joint_map,
    )
    if not feature_columns:
        raise ValueError(f"Representation {representation} has no model-input columns")

    split_frames = {
        split: view.loc[view["split"].eq(split)].copy() for split in SPLITS
    }
    for split, frame in split_frames.items():
        if frame.empty or frame["label"].nunique() < 2:
            raise ValueError(f"Representation {representation} has unusable {split} labels")

    train = split_frames["train"]
    validation = split_frames["val"]
    test = split_frames["test"]
    X_train = train[feature_columns].astype(float)
    y_train = train["label"].astype(int).to_numpy()
    X_val = validation[feature_columns].astype(float)
    y_val = validation["label"].astype(int).to_numpy()
    X_test = test[feature_columns].astype(float)
    y_test = test["label"].astype(int).to_numpy()

    rows = []
    for seed in SEEDS:
        candidates = []
        for model_name in MODEL_ORDER:
            model = benchmark._build_model(model_name, seed, y_train)
            if model is None:
                raise RuntimeError(f"Required benchmark model is unavailable: {model_name}")
            model.fit(X_train, y_train)
            val_prob = model.predict_proba(X_val)[:, 1]
            test_prob = model.predict_proba(X_test)[:, 1]
            selection = next(
                item
                for item in benchmark._select_thresholds(y_val, val_prob)
                if item.rule == "val_f1"
            )
            if selection.status != "available" or selection.threshold is None:
                raise RuntimeError(f"Validation-F1 threshold unavailable for {model_name}")
            candidates.append((model_name, selection, test_prob))

        model_name, selection, test_prob = max(
            candidates, key=lambda item: item[1].val_metrics["f1"]
        )
        threshold = float(selection.threshold)
        test_metrics = benchmark._compute_metrics(y_test, test_prob, threshold)
        test_pred = (test_prob >= threshold).astype(int)
        prediction_columns = ["sequence_id", "split", "label"]
        if "track_id" in test.columns:
            prediction_columns.append("track_id")
        predictions = test[prediction_columns].copy()
        predictions["probability"] = test_prob
        predictions["prediction"] = test_pred

        row = {
            "dataset": config_key[0],
            "horizon": config_key[1],
            "representation": representation,
            "seed": seed,
            "model_name": model_name,
            "threshold_rule": "val_f1",
            "selected_threshold": threshold,
            "n_features": len(feature_columns),
            "train_rows": len(train),
            "val_rows": len(validation),
            "test_rows": len(test),
            "test_predictions": predictions.to_dict("records"),
            "candidate_val_f1": {
                candidate_name: float(candidate_selection.val_metrics["f1"])
                for candidate_name, candidate_selection, _probability in candidates
            },
            **{f"val_{key}": value for key, value in selection.val_metrics.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
            **fingerprints,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def match_comparison_cohort(
    baseline_df: pd.DataFrame, candidate_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align two representations to the same sequence and frozen split rows."""
    for name, frame in (("baseline", baseline_df), ("candidate", candidate_df)):
        missing = set(MATCH_KEYS) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} cohort is missing keys: {sorted(missing)}")
        if frame.duplicated(MATCH_KEYS).any():
            raise ValueError(f"{name} cohort has duplicate sequence/split rows")

    baseline_splits = baseline_df.set_index("sequence_id")["split"]
    candidate_splits = candidate_df.set_index("sequence_id")["split"]
    common_ids = baseline_splits.index.intersection(candidate_splits.index)
    if not baseline_splits.loc[common_ids].equals(candidate_splits.loc[common_ids]):
        raise ValueError("Baseline and candidate split membership differs")

    matched_keys = (
        baseline_df[MATCH_KEYS]
        .merge(candidate_df[MATCH_KEYS], on=MATCH_KEYS, how="inner", validate="one_to_one")
        .sort_values(MATCH_KEYS)
        .reset_index(drop=True)
    )
    if matched_keys.empty:
        raise ValueError("Baseline and candidate cohorts have no matched rows")

    baseline = matched_keys.merge(
        baseline_df, on=MATCH_KEYS, how="left", validate="one_to_one"
    )
    candidate = matched_keys.merge(
        candidate_df, on=MATCH_KEYS, how="left", validate="one_to_one"
    )
    return baseline, candidate


def paired_test_track_bootstrap(
    test_df: pd.DataFrame,
    baseline_col: str,
    candidate_col: str,
    track_col: str,
    metric_name: str,
    n_boot: int = 1000,
) -> dict[str, float]:
    """Bootstrap a paired test delta by resampling held-out tracks."""
    required = {"label", baseline_col, candidate_col, track_col}
    missing = required - set(test_df.columns)
    if missing:
        raise ValueError(f"Paired bootstrap is missing columns: {sorted(missing)}")
    supported_metrics = {"accuracy", "roc_auc", "pr_auc", "f1", "precision", "recall"}
    if metric_name not in supported_metrics:
        raise ValueError(f"Unsupported bootstrap metric: {metric_name}")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if test_df.empty or test_df[track_col].isna().any():
        raise ValueError("Paired bootstrap requires non-empty rows with track IDs")
    if "split" in test_df.columns and not test_df["split"].eq("test").all():
        raise ValueError("Paired bootstrap accepts held-out test rows only")

    def score(frame: pd.DataFrame, column: str) -> float:
        labels = frame["label"].astype(int).to_numpy()
        values = frame[column].astype(float).to_numpy()
        if metric_name == "roc_auc":
            return float(roc_auc_score(labels, values)) if len(np.unique(labels)) == 2 else float("nan")
        if metric_name == "pr_auc":
            return float(average_precision_score(labels, values)) if len(np.unique(labels)) == 2 else float("nan")
        predictions = (values >= 0.5).astype(int)
        if metric_name == "accuracy":
            return float(accuracy_score(labels, predictions))
        if metric_name == "f1":
            return float(f1_score(labels, predictions, zero_division=0))
        if metric_name == "precision":
            return float(precision_score(labels, predictions, zero_division=0))
        return float(recall_score(labels, predictions, zero_division=0))

    track_ids = test_df[track_col].drop_duplicates().to_numpy()
    grouped_indices = {
        track_id: test_df.index[test_df[track_col].eq(track_id)].to_numpy()
        for track_id in track_ids
    }
    rng = np.random.default_rng(42)
    deltas: list[float] = []
    for _ in range(n_boot):
        sampled_tracks = rng.choice(track_ids, size=len(track_ids), replace=True)
        sampled_indices = np.concatenate(
            [grouped_indices[track_id] for track_id in sampled_tracks]
        )
        sampled = test_df.loc[sampled_indices]
        deltas.append(score(sampled, candidate_col) - score(sampled, baseline_col))

    baseline_score = score(test_df, baseline_col)
    candidate_score = score(test_df, candidate_col)
    return {
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "delta": candidate_score - baseline_score,
        "delta_mean": float(np.nanmean(deltas)),
        "ci_low": float(np.nanquantile(deltas, 0.025)),
        "ci_high": float(np.nanquantile(deltas, 0.975)),
        "n_tracks": float(len(track_ids)),
        "n_boot": float(n_boot),
    }
