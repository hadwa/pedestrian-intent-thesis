"""Authenticated Exp062 train/validation inputs and validation-only history readers."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import (
    ExperimentConfig,
    canonical_json_sha256,
    experiment_config_sha256,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP055_TRAINVAL = REPO_ROOT / "experiments/055_compact_residual_biu_lstm/cache/trainval"
EXP012_ROOT = REPO_ROOT / "experiments/012_pie_biu_v2_30_60_benchmark"
EXP012_BUILDER = REPO_ROOT / "scripts/experiments/pie_biu_v2_30_60_benchmark.py"
EXP012_PREDICTIONS = EXP012_ROOT / "predictions"
_EXP012_ANCHOR_CACHE: tuple[str, np.ndarray, np.ndarray] | None = None

SOURCE_MANIFEST = {
    "summary_v1.npy": "6a92d072e4e0ecd360ad819bf180428c1a75508502e152c534989d2cdbe29784",
    "summary_v2_core.npy": "06ee8693c0ccd819bc7bddd4db7393cd501093e21bc9d130c5e8f0a38d683867",
    "ego.npy": "410c4757e4012e223721f5686bffe5f8e4c4f53dd9fd9030e763fcfbe58dd302",
    "yolop.npy": "b353bf9d8fb7d5eedacf726fec643f67138994f5e9a864806f46e382075b65f5",
    "rows.csv": "a01ad1f3244157c1b649aada0470ad54d877bbd1e8075c275ee313c35dcc3bf1",
}

# The Exp050 YOLOP cache serializes its 3x3 drivable grid, then its 3x3 lane
# grid, followed by object counts. This semantic order is part of Exp062 input ABI.
YOLOP_CHANNELS = tuple(
    [f"drivable_cell_{index}" for index in range(9)]
    + [f"lane_cell_{index}" for index in range(9)]
    + ["vehicle_count", "pedestrian_count"]
)


@dataclass(frozen=True)
class TrainValData:
    """The only original train/validation population available to Exp062."""

    rows: pd.DataFrame
    summary_v1: np.ndarray
    summary_v2_core: np.ndarray
    ego: np.ndarray
    yolop: np.ndarray
    labels: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray


@dataclass(frozen=True)
class ContextScaler:
    """Per-channel context standardization fitted exclusively on train samples."""

    ego_mean: np.ndarray
    ego_std: np.ndarray
    yolop_mean: np.ndarray
    yolop_std: np.ndarray

def _authenticate_sources(config: ExperimentConfig) -> None:
    expected = dict(SOURCE_MANIFEST)
    if expected["rows.csv"] != config.trainval_rows_sha256:
        raise RuntimeError("Exp062 rows manifest no longer matches its frozen contract")
    if expected["ego.npy"] != config.ego_trainval_sha256:
        raise RuntimeError("Exp062 ego manifest no longer matches its frozen contract")
    if expected["yolop.npy"] != config.yolop_trainval_sha256:
        raise RuntimeError("Exp062 YOLOP manifest no longer matches its frozen contract")
    for filename, expected_hash in expected.items():
        path = EXP055_TRAINVAL / filename
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Exp062 source authentication failed: {filename}")


def _load_array(filename: str, expected_shape: tuple[int, ...]) -> np.ndarray:
    array = np.load(EXP055_TRAINVAL / filename, allow_pickle=False)
    if array.dtype.hasobject:
        raise RuntimeError(f"Exp062 source array must not use object dtype: {filename}")
    if array.shape != expected_shape:
        raise RuntimeError(
            f"Exp062 source shape drifted for {filename}: {array.shape} != {expected_shape}"
        )
    if not np.isfinite(array).all():
        raise RuntimeError(f"Exp062 source contains non-finite values: {filename}")
    return np.asarray(array)


def _validate_rows(rows: pd.DataFrame, config: ExperimentConfig) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    required = {
        "sequence_id", "split", "set_id", "track_id", "start_frame", "end_frame",
        "event_frame", "time_to_event_frames", "observation_length", "frame_step",
        "frame_sampling", "label", "cache_index",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise RuntimeError(f"Exp062 rows are missing required columns: {missing}")
    checked = rows.reset_index(drop=True).copy()
    if len(checked) != config.train_rows + config.val_rows:
        raise RuntimeError("Exp062 train/validation row count drifted")
    if checked["sequence_id"].isna().any() or checked["sequence_id"].astype(str).duplicated().any():
        raise RuntimeError("Exp062 sequence IDs must be nonempty and unique")
    splits = checked["split"].astype(str)
    if set(splits) != {"train", "val"}:
        raise RuntimeError("Exp062 rows contain an unexpected split")
    train_mask = splits.eq("train").to_numpy()
    val_mask = splits.eq("val").to_numpy()
    if train_mask.sum() != config.train_rows or val_mask.sum() != config.val_rows:
        raise RuntimeError("Exp062 frozen train/validation counts drifted")
    if not train_mask[: config.train_rows].all() or not val_mask[config.train_rows :].all():
        raise RuntimeError("Exp062 row order must be all train rows followed by validation rows")
    if not np.array_equal(checked["cache_index"].to_numpy(dtype=np.int64), np.arange(len(checked))):
        raise RuntimeError("Exp062 cache_index order drifted")
    for split, cohorts in (("train", config.train_cohorts), ("val", config.val_cohorts)):
        actual = tuple(sorted(checked.loc[splits.eq(split), "set_id"].astype(str).unique()))
        if actual != tuple(sorted(cohorts)):
            raise RuntimeError(f"Exp062 {split} cohort membership drifted: {actual}")
    if checked.loc[train_mask, "track_id"].isna().any():
        raise RuntimeError("Exp062 train rows must retain track_id values")
    labels = checked["label"].to_numpy(dtype=np.int64)
    if not np.isin(labels, (0, 1)).all():
        raise RuntimeError("Exp062 labels must be binary")
    start = checked["start_frame"].to_numpy(dtype=np.int64)
    end = checked["end_frame"].to_numpy(dtype=np.int64)
    observed = start[:, None] + np.arange(config.observed_frames, dtype=np.int64)
    consecutive = (
        checked["observation_length"].to_numpy(dtype=np.int64) == config.observed_frames
    ) & (checked["frame_step"].to_numpy(dtype=np.int64) == 1) & (
        checked["frame_sampling"].astype(str).to_numpy() == "consecutive"
    ) & ((end - start) == config.observed_frames - 1) & (observed[:, -1] == end)
    if not consecutive.all():
        raise RuntimeError("Exp062 windows must be exactly 16 consecutive observation frames")
    tte = checked["time_to_event_frames"].to_numpy(dtype=np.int64)
    if not ((30 <= tte) & (tte <= 60)).all() or not np.array_equal(
        checked["event_frame"].to_numpy(dtype=np.int64) - end, tte
    ):
        raise RuntimeError("Exp062 30--60 frame event timing drifted")
    return checked, labels, train_mask, val_mask


def _load_exp012_builder():
    spec = importlib.util.spec_from_file_location("exp062_exp012_feature_builder", EXP012_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load authoritative Exp012 feature builder: {EXP012_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # The builder's file-loading helper consults this module-level value. Keeping
    # this tuple train/val-only prevents it from opening an Exp012 test source.
    module.SPLITS = ("train", "val")
    return module


def _authenticate_exp012_definition(config: ExperimentConfig) -> tuple[Path, str]:
    """Authenticate the frozen Exp012 definition contract before cache reuse."""
    frozen = ExperimentConfig()
    if (
        config.exp012_feature_definition_path != frozen.exp012_feature_definition_path
        or config.exp012_feature_definition_sha256
        != frozen.exp012_feature_definition_sha256
    ):
        raise RuntimeError("Exp012 feature-definition contract drifted")
    definitions_path = REPO_ROOT / config.exp012_feature_definition_path
    if not definitions_path.is_file():
        raise RuntimeError("Exp012 feature-definition source is missing")
    definition_sha256 = sha256_file(definitions_path)
    if definition_sha256 != config.exp012_feature_definition_sha256:
        raise RuntimeError("Exp012 feature-definition authentication failed")
    return definitions_path, definition_sha256


def _exp012_anchor_cache_key(
    rows: pd.DataFrame, config: ExperimentConfig, definitions_path: Path, definition_sha256: str
) -> str:
    """Fingerprint every input that determines the authoritative anchor matrices."""
    builder_rows = rows.drop(columns=["cache_index", "frame_width", "frame_height"])
    return canonical_json_sha256(
        {
            "config_sha256": experiment_config_sha256(config),
            "definition_source": {
                "path": str(definitions_path.relative_to(REPO_ROOT)),
                "sha256": definition_sha256,
            },
            "builder_rows": {
                "columns": builder_rows.columns.tolist(),
                "values": builder_rows.to_numpy().tolist(),
            },
        }
    )


def _align_authoritative_features(
    feature_frame: pd.DataFrame, feature_names: tuple[str, ...], rows: pd.DataFrame
) -> np.ndarray:
    required = {"sequence_id", "split", "label", *feature_names}
    missing = sorted(required.difference(feature_frame.columns))
    if missing or feature_frame["sequence_id"].astype(str).duplicated().any():
        raise RuntimeError("Authoritative Exp012 feature rows are malformed")
    indexed = feature_frame.set_index(feature_frame["sequence_id"].astype(str), drop=False)
    ids = rows["sequence_id"].astype(str).to_numpy()
    if set(indexed.index) != set(ids):
        raise RuntimeError("Authoritative Exp012 feature cohort drifted")
    aligned = indexed.loc[ids]
    if not np.array_equal(aligned["split"].astype(str).to_numpy(), rows["split"].astype(str).to_numpy()):
        raise RuntimeError("Authoritative Exp012 split order drifted")
    if not np.array_equal(aligned["label"].to_numpy(dtype=np.int64), rows["label"].to_numpy(dtype=np.int64)):
        raise RuntimeError("Authoritative Exp012 labels drifted")
    values = aligned.loc[:, list(feature_names)].to_numpy(dtype=np.float64)
    if values.shape != (len(rows), len(feature_names)) or not np.isfinite(values).all():
        raise RuntimeError("Authoritative Exp012 anchor values drifted")
    return values


def _load_exp012_trainval_bbox_index(
    builder: object, needed: set[tuple[str, str, str, int]]
) -> dict[tuple[str, str, str, int], list[float]]:
    """Read only train/validation XML records needed by Exp012 BIU-v2 extraction."""
    by_video: dict[tuple[str, str], set[tuple[str, str, str, int]]] = {}
    for key in needed:
        by_video.setdefault(key[:2], set()).add(key)
    output: dict[tuple[str, str, str, int], list[float]] = {}
    for (set_id, video_id), video_keys in by_video.items():
        xml_path = builder.PIE_ANNOTATIONS_DIR / set_id / f"{video_id}_annt.xml"
        root = ET.parse(xml_path).getroot()
        for track in root.findall("track"):
            if track.attrib.get("label") != "pedestrian":
                continue
            pedestrian_id = None
            for box in track.findall("box"):
                item = box.find("attribute[@name='id']")
                if item is not None and item.text:
                    pedestrian_id = item.text.strip()
                    break
            if pedestrian_id is None:
                continue
            for box in track.findall("box"):
                if box.attrib.get("outside", "0") == "1":
                    continue
                frame_idx = int(float(box.attrib["frame"]))
                key = (set_id, video_id, pedestrian_id, frame_idx)
                if key in video_keys:
                    output[key] = [
                        float(box.attrib["xtl"]),
                        float(box.attrib["ytl"]),
                        float(box.attrib["xbr"]),
                        float(box.attrib["ybr"]),
                    ]
    return output


def _load_exp012_trainval_keypoint_index(
    builder: object, needed: set[tuple[str, str, str, int]]
) -> dict[tuple[str, str, str, int], dict]:
    """Load only train/validation keypoint records required by the Exp012 builder."""
    allowed_sets = {key[0] for key in needed}
    output: dict[tuple[str, str, str, int], dict] = {}
    with builder.KEYPOINTS_JSONL.open("r", encoding="utf-8") as handle:
        for line in handle:
            # Avoid deserializing any record outside the authorized cohorts.
            if not any(f'"set": "{set_id}"' in line for set_id in allowed_sets):
                continue
            record = json.loads(line)
            key = (
                str(record["set"]),
                str(record["video"]),
                str(record["pid"]),
                int(record["frame_idx"]),
            )
            if key in needed:
                output[key] = record
    return output


def _reconstruct_from_exp012(rows: pd.DataFrame, config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    """Build the two frozen anchor matrices from Exp012 train/validation inputs only."""
    global _EXP012_ANCHOR_CACHE
    definitions_path, definition_sha256 = _authenticate_exp012_definition(config)
    cache_key = _exp012_anchor_cache_key(rows, config, definitions_path, definition_sha256)
    if _EXP012_ANCHOR_CACHE is not None and _EXP012_ANCHOR_CACHE[0] == cache_key:
        return _EXP012_ANCHOR_CACHE[1].copy(), _EXP012_ANCHOR_CACHE[2].copy()
    builder = _load_exp012_builder()
    builder_rows = rows.drop(columns=["cache_index", "frame_width", "frame_height"])
    needed = builder._needed_frame_keys(builder_rows)
    keypoint_index = _load_exp012_trainval_keypoint_index(builder, needed)
    bbox_index = _load_exp012_trainval_bbox_index(builder, needed)
    v2_frame, missing_v2, _ = builder._extract_v2_for_rows(
        builder_rows, keypoint_index, bbox_index
    )
    if missing_v2:
        raise RuntimeError("Authoritative Exp012 BIU-v2 reconstruction has missing rows")
    feature_definitions = pd.read_csv(definitions_path).set_index("feature_set")
    feature_dict = pd.read_csv(builder.DICT_PATH)
    feature_sets, column_map, _ = builder._build_feature_sets(builder_rows, v2_frame, feature_dict)
    matrices: list[np.ndarray] = []
    for feature_set, expected_dim in (
        ("biu_v1_plus_bbox", config.rf_feature_dim),
        ("biu_v2_core_plus_bbox", config.xgb_feature_dim),
    ):
        names = tuple(column_map[feature_set])
        defined_names = tuple(json.loads(feature_definitions.loc[feature_set, "feature_columns_json"]))
        if names != defined_names:
            raise RuntimeError(f"Exp012 {feature_set} feature order drifted")
        if len(names) != expected_dim or len(names) != len(set(names)):
            raise RuntimeError(f"Exp012 {feature_set} feature names drifted")
        matrices.append(_align_authoritative_features(feature_sets[feature_set], names, rows))
    _EXP012_ANCHOR_CACHE = (cache_key, matrices[0], matrices[1])
    return matrices[0].copy(), matrices[1].copy()


def load_authenticated_trainval(config: ExperimentConfig) -> TrainValData:
    """Load the frozen train/validation population after every source check passes."""
    _authenticate_sources(config)
    rows, labels, train_mask, val_mask = _validate_rows(
        pd.read_csv(EXP055_TRAINVAL / "rows.csv"), config
    )
    summary_v1, summary_v2_core = _reconstruct_from_exp012(rows, config)
    data = TrainValData(
        rows=rows,
        summary_v1=summary_v1,
        summary_v2_core=summary_v2_core,
        ego=_load_array("ego.npy", (len(rows), *config.ego_shape)),
        yolop=_load_array("yolop.npy", (len(rows), *config.yolop_shape)),
        labels=labels,
        train_mask=train_mask,
        val_mask=val_mask,
    )
    rebuilt_v1, rebuilt_v2 = _reconstruct_from_exp012(rows, config)
    np.testing.assert_allclose(data.summary_v1, rebuilt_v1, rtol=0, atol=0)
    np.testing.assert_allclose(data.summary_v2_core, rebuilt_v2, rtol=0, atol=0)
    return data


def fit_context_scaler(data: TrainValData) -> ContextScaler:
    """Fit per-channel context statistics over training rows and 16 frame positions."""
    if data.train_mask.dtype != np.bool_ or data.train_mask.shape != (len(data.rows),):
        raise ValueError("train_mask must align with the train/validation rows")
    if not data.train_mask.any() or not np.isfinite(data.ego).all() or not np.isfinite(data.yolop).all():
        raise ValueError("context scaler requires finite nonempty training context")
    ego_train = data.ego[data.train_mask]
    yolop_train = data.yolop[data.train_mask]
    ego_mean = ego_train.mean(axis=(0, 1))
    yolop_mean = yolop_train.mean(axis=(0, 1))
    ego_std = np.where(ego_train.std(axis=(0, 1)) > 0.0, ego_train.std(axis=(0, 1)), 1.0)
    yolop_std = np.where(yolop_train.std(axis=(0, 1)) > 0.0, yolop_train.std(axis=(0, 1)), 1.0)
    return ContextScaler(ego_mean=ego_mean, ego_std=ego_std, yolop_mean=yolop_mean, yolop_std=yolop_std)


def transform_context(data: TrainValData, scaler: ContextScaler) -> TrainValData:
    """Apply a previously fitted context scaler without altering labels or anchors."""
    ego = ((data.ego - scaler.ego_mean) / scaler.ego_std).astype(np.float32)
    yolop = ((data.yolop - scaler.yolop_mean) / scaler.yolop_std).astype(np.float32)
    if not np.isfinite(ego).all() or not np.isfinite(yolop).all():
        raise ValueError("context transformation produced non-finite values")
    return TrainValData(
        rows=data.rows,
        summary_v1=data.summary_v1,
        summary_v2_core=data.summary_v2_core,
        ego=ego,
        yolop=yolop,
        labels=data.labels,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
    )


def train_row_manifest_sha256(data: TrainValData) -> str:
    """Hash the exact ordered original training-row identity needed by a scaler."""
    required = {"sequence_id", "split", "label", "track_id"}
    if not required.issubset(data.rows.columns) or data.train_mask.shape != (len(data.rows),):
        raise ValueError("train-row manifest requires authenticated row identity columns")
    train_rows = data.rows.loc[data.train_mask, ["sequence_id", "split", "label", "track_id"]]
    if train_rows.empty or not train_rows["split"].astype(str).eq("train").all():
        raise ValueError("train-row manifest requires original training rows")
    records = [
        {
            "sequence_id": str(row.sequence_id),
            "split": str(row.split),
            "label": int(row.label),
            "track_id": str(row.track_id),
        }
        for row in train_rows.itertuples(index=False)
    ]
    return canonical_json_sha256(records)


def write_context_scaler(path: Path, data: TrainValData, scaler: ContextScaler) -> dict[str, object]:
    """Atomically persist a train-only scaler with its derived row-manifest hash."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_row_manifest_sha256": train_row_manifest_sha256(data),
        "ego_mean": scaler.ego_mean.astype(np.float64).tolist(),
        "ego_std": scaler.ego_std.astype(np.float64).tolist(),
        "yolop_mean": scaler.yolop_mean.astype(np.float64).tolist(),
        "yolop_std": scaler.yolop_std.astype(np.float64).tolist(),
    }
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
    temporary.replace(destination)
    return payload


def read_historical_validation_prefix(
    path: Path,
    expected_model: str,
    seed: int,
    expected_validation: pd.DataFrame,
    expected_rows: int = 3619,
) -> pd.DataFrame:
    """Read exactly the approved validation prefix, never the historical test suffix."""
    allowed = {
        "rf_v1_bbox": ("biu_v1_plus_bbox", "rf"),
        "xgb_v2_bbox": ("biu_v2_core_plus_bbox", "xgb"),
    }
    if expected_model not in allowed or seed not in ExperimentConfig().seeds:
        raise ValueError("historical reader requires an approved anchor family and seed")
    if expected_rows != ExperimentConfig().val_rows:
        raise ValueError("historical reader must parse exactly 3,619 validation rows")
    feature_set, model = allowed[expected_model]
    expected_path = EXP012_PREDICTIONS / f"{feature_set}__{model}__seed{seed}.csv"
    if Path(path).resolve() != expected_path.resolve():
        raise ValueError("historical reader path is not an approved anchor prediction file")
    records: list[dict[str, object]] = []
    with expected_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"feature_set", "model_name", "seed", "sequence_id", "split", "label", "prob"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("approved prediction file has an unexpected schema")
        for row in reader:
            if len(records) == expected_rows:
                break
            if row["feature_set"] != feature_set or row["model_name"] != model or int(row["seed"]) != seed:
                raise ValueError("approved prediction file identity drifted")
            if row["split"] != "val":
                raise ValueError("historical validation prefix contains a non-validation row")
            label = int(row["label"])
            probability = float(row["prob"])
            if label not in (0, 1) or not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError("historical validation prefix contains invalid values")
            records.append({"sequence_id": row["sequence_id"], "split": "val", "label": label, "probability": probability})
    if len(records) != expected_rows:
        raise ValueError("approved prediction file is shorter than its frozen validation prefix")
    frame = pd.DataFrame.from_records(records, columns=["sequence_id", "split", "label", "probability"])
    if frame["sequence_id"].duplicated().any():
        raise ValueError("historical validation prefix has duplicate sequence IDs")
    required_cohort = {"sequence_id", "split", "label"}
    if not required_cohort.issubset(expected_validation.columns) or len(expected_validation) != expected_rows:
        raise ValueError("historical validation cohort has an unexpected schema or length")
    expected = expected_validation.loc[:, ["sequence_id", "split", "label"]].reset_index(drop=True)
    if (
        not expected["split"].astype(str).eq("val").all()
        or expected["sequence_id"].astype(str).duplicated().any()
        or not np.isin(expected["label"].to_numpy(dtype=np.int64), (0, 1)).all()
    ):
        raise ValueError("historical validation cohort is malformed")
    if not np.array_equal(frame["sequence_id"].astype(str).to_numpy(), expected["sequence_id"].astype(str).to_numpy()) or not np.array_equal(
        frame["label"].to_numpy(dtype=np.int64), expected["label"].to_numpy(dtype=np.int64)
    ):
        raise ValueError("historical validation prefix does not match the authenticated cohort")
    return frame
