#!/usr/bin/env python3
"""Resumable phase-separated runner for Exp050."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import exp050_core as core  # noqa: E402


ROOT = SCRIPT_DIR.parent
REPO_ROOT = ROOT.parents[1]
ARCHIVE_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation")
MSLSTM_ROOT = Path("/home/abdelrahman/Hadwa/MSLSTM-PID")
SPLIT_ROOT = ARCHIVE_ROOT / "data/splits_pie_wacv_pcpa_30_60_overlap06"
PIE_DB_PATH = MSLSTM_ROOT / "PIE/annotations/data_cache/pie_database.pkl"
POSE_PATH = Path("/home/abdelrahman/Hadwa/pedestrian-intent-biu/data/data_keypoints_all.jsonl")
POSE_SHA256 = "1f0907ebb33bb04c3a3434d9c89e7bd597e2b9e3d47fe30f488ffa9410054d87"
YOLOP_ROOT = MSLSTM_ROOT / "Extracted YOLOP"

BASE_CACHE = ROOT / "cache/trainval_base"
TEST_CACHE = ROOT / "cache/set03_final"
PREPARE_MARKER = ROOT / ".prepare.complete.json"
EVALUATION_MARKER = ROOT / ".evaluation.complete.json"
COMPLETE_MARKER = ROOT / "EXPERIMENT_COMPLETE"

ARRAY_NAMES = (
    "bbox",
    "ped_action",
    "ped_look",
    "ego_speed",
    "ego_acc",
    "static_context",
    "yolop",
    "pose",
    "biu_v2",
    "biu_v1",
)
TEMPORAL_STREAMS = {"bbox", "ped_action", "ped_look", "ego_speed", "ego_acc", "yolop", "pose"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_line(path: Path, message: str) -> None:
    line = f"{utc_now()} {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def atomic_json(value: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_npy(value: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    temporary.replace(path)


def atomic_torch_save(value: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(value), temporary)
    temporary.replace(path)


def frame_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact table without pandas' optional tabulate dependency."""
    def cell(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}".rstrip("0").rstrip(".")
        return str(value).replace("|", "\\|")

    columns = [str(column) for column in frame.columns]
    lines = [f"| {' | '.join(columns)} |", f"|{'|'.join('---' for _ in columns)}|"]
    lines.extend(f"| {' | '.join(cell(value) for value in row)} |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def required_training_identities() -> list[tuple[str, int]]:
    return [(variant, seed) for variant in core.VARIANTS for seed in core.SEEDS]


def training_marker(root: Path, variant: str, seed: int) -> Path:
    return Path(root) / f".train_{variant}_seed{seed}.complete.json"


def select_phase_rows(rows: pd.DataFrame, *, phase: str) -> pd.DataFrame:
    if phase == "prepare":
        selected = rows.loc[rows["split"].isin({"train", "val"})].reset_index(drop=True)
        if selected["set_id"].astype(str).eq("set03").any():
            raise ValueError("Preparation must not access set03.")
        return selected
    if phase == "evaluate":
        selected = rows.loc[rows["split"].eq("test")].reset_index(drop=True)
        if set(selected["set_id"].astype(str)) != {"set03"}:
            raise ValueError("Final evaluation must use designated set03 only.")
        return selected
    raise ValueError(f"Unknown phase: {phase}")


def authenticate_artifacts(root: Path, artifacts: Mapping[str, str]) -> None:
    for relative, expected in artifacts.items():
        path = Path(root) / relative
        if not path.is_file() or core.sha256_file(path) != str(expected):
            raise ValueError(f"Authenticated artifact drifted: {path}")


def assert_training_complete(root: Path = ROOT) -> list[dict[str, object]]:
    authenticated: list[dict[str, object]] = []
    for variant, seed in required_training_identities():
        marker = training_marker(root, variant, seed)
        if not marker.is_file():
            raise RuntimeError("Final evaluation requires all 18 authenticated checkpoints.")
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("variant") != variant or int(payload.get("seed", -1)) != seed:
            raise RuntimeError("Final evaluation requires all 18 authenticated checkpoints.")
        try:
            authenticate_artifacts(root, payload.get("artifacts", {}))
        except ValueError as error:
            raise RuntimeError("Final evaluation requires all 18 authenticated checkpoints.") from error
        authenticated.append(payload)
    return authenticated


def load_phase_rows(phase: str) -> pd.DataFrame:
    splits = ("train", "val") if phase == "prepare" else ("test",)
    frames = []
    for split in splits:
        path = SPLIT_ROOT / f"{split}_sequences.csv"
        frame = pd.read_csv(path)
        if not frame["split"].astype(str).eq(split).all():
            raise ValueError(f"Split identity drifted: {path}")
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    allowed = set(splits)
    rows = core.validate_split_rows(rows, allowed_splits=allowed)
    for split in splits:
        count = int((rows["split"] == split).sum())
        if count != core.EXPECTED_SPLIT_COUNTS[split]:
            raise ValueError(f"Frozen {split} count drifted: {count}")
        actual_sets = set(rows.loc[rows["split"] == split, "set_id"].astype(str))
        if actual_sets != core.EXPECTED_SPLIT_SETS[split]:
            raise ValueError(f"Frozen {split} membership drifted: {sorted(actual_sets)}")
    return rows


class YolopStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.cache: dict[tuple[str, str], Mapping[object, object]] = {}

    def video(self, set_id: str, video_id: str) -> Mapping[object, object]:
        identity = (str(set_id), str(video_id))
        if identity not in self.cache:
            path = self.root / f"{identity[0]}_{identity[1]}_yolop_features.pkl"
            if not path.is_file():
                raise FileNotFoundError(f"Missing exact YOLOP cache: {path}")
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            key = f"{identity[0]}_{identity[1]}"
            if not isinstance(payload, Mapping) or not isinstance(payload.get(key), Mapping):
                raise ValueError(f"Malformed YOLOP cache: {path}")
            self.cache[identity] = payload[key]
        return self.cache[identity]

    def window(self, set_id: str, video_id: str, ped_id: str, frames: list[int]) -> np.ndarray:
        video = self.video(set_id, video_id)
        output = []
        for frame_id in frames:
            frame = video.get(frame_id, video.get(str(frame_id)))
            if not isinstance(frame, Mapping) or str(ped_id) not in frame:
                raise ValueError(f"Missing YOLOP {set_id}/{video_id}/{ped_id}/{frame_id}")
            value = np.asarray(frame[str(ped_id)], dtype=np.float32)
            if value.shape != (20,) or not np.isfinite(value).all():
                raise ValueError("YOLOP vectors must be finite 20-D values.")
            output.append(value)
        return np.stack(output)


def load_pose_index(rows: pd.DataFrame) -> dict[tuple[str, str, str, int], tuple[np.ndarray, np.ndarray]]:
    needed = {
        (str(row.set_id), str(row.video_id), str(row.ped_id), frame)
        for row in rows.itertuples(index=False)
        for frame in core.observed_frames(row)
    }
    result: dict[tuple[str, str, str, int], tuple[np.ndarray, np.ndarray]] = {}
    with POSE_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            key = (
                str(record["set"]),
                str(record["video"]),
                str(record["pid"]),
                int(record["frame_idx"]),
            )
            if key not in needed:
                continue
            coords = np.asarray(record["keypoints"], dtype=np.float32)
            confidence = np.asarray(record["kp_conf"], dtype=np.float32)
            if coords.shape != (17, 2) or confidence.shape != (17,):
                raise ValueError(f"Malformed pose record: {key}")
            result[key] = (coords, confidence)
    missing = needed.difference(result)
    if missing:
        raise ValueError(f"Pose cache misses {len(missing)} exact observation frames; first={next(iter(missing))}")
    return result


def static_context(attributes: Mapping[str, object]) -> np.ndarray:
    categories = (
        (int(attributes.get("signalized", 0)), 4),
        (int(attributes.get("intersection", 0)), 5),
        (int(attributes.get("age", 2)), 4),
        (int(attributes.get("gender", 0)), 3),
        (int(attributes.get("traffic_direction", 0)), 2),
    )
    lane_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 4, 7: 4, 8: 4}
    categories += ((lane_map.get(int(attributes.get("num_lanes", 2)), 4), 5),)
    pieces = []
    for value, size in categories:
        one_hot = np.zeros(size, dtype=np.float32)
        one_hot[value if 0 <= value < size else 0] = 1.0
        pieces.append(one_hot)
    vector = np.concatenate(pieces)
    if vector.shape != (23,):
        raise RuntimeError(f"Static context must be 23-D, got {vector.shape}")
    return vector


def cache_metadata(cache_root: Path) -> dict[str, object]:
    path = cache_root / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing cache metadata: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    authenticate_artifacts(cache_root, metadata.get("artifacts", {}))
    return metadata


def save_cache(cache_root: Path, rows: pd.DataFrame, arrays: Mapping[str, np.ndarray], extra: Mapping[str, object]) -> dict[str, object]:
    cache_root.mkdir(parents=True, exist_ok=True)
    rows_path = cache_root / "rows.csv"
    atomic_csv(rows, rows_path)
    for name, value in arrays.items():
        atomic_npy(np.asarray(value), cache_root / f"{name}.npy")
    artifacts = {"rows.csv": core.sha256_file(rows_path)}
    artifacts.update({f"{name}.npy": core.sha256_file(cache_root / f"{name}.npy") for name in arrays})
    metadata = {
        "created_at": utc_now(),
        "n_rows": int(len(rows)),
        "sequence_id_sha256": core.canonical_json_sha256(rows["sequence_id"].astype(str).tolist()),
        "artifacts": artifacts,
        **dict(extra),
    }
    atomic_json(metadata, cache_root / "metadata.json")
    return metadata


def load_cache(cache_root: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, object]]:
    metadata = cache_metadata(cache_root)
    rows = pd.read_csv(cache_root / "rows.csv")
    arrays = {name: np.load(cache_root / f"{name}.npy", mmap_mode="r") for name in ARRAY_NAMES}
    if any(len(value) != len(rows) for value in arrays.values()):
        raise ValueError(f"Cache population is misaligned: {cache_root}")
    return rows, arrays, metadata


def assemble_base_cache(rows: pd.DataFrame, cache_root: Path, *, scalers: Mapping[str, float] | None) -> dict[str, object]:
    log = ROOT / "logs/prepare.log"
    log_line(log, f"assemble status=loading_pie rows={len(rows)} cache={cache_root.name}")
    with PIE_DB_PATH.open("rb") as handle:
        pie_db = pickle.load(handle)
    log_line(log, "assemble status=indexing_pose")
    pose_index = load_pose_index(rows)
    yolop = YolopStore(YOLOP_ROOT)
    n = len(rows)
    arrays = {
        "bbox": np.empty((n, core.SEQ_LEN, 4), dtype=np.float32),
        "ped_action": np.empty((n, core.SEQ_LEN, 1), dtype=np.float32),
        "ped_look": np.empty((n, core.SEQ_LEN, 1), dtype=np.float32),
        "ego_speed": np.empty((n, core.SEQ_LEN, 1), dtype=np.float32),
        "ego_acc": np.empty((n, core.SEQ_LEN, 2), dtype=np.float32),
        "static_context": np.empty((n, 23), dtype=np.float32),
        "yolop": np.empty((n, core.SEQ_LEN, 20), dtype=np.float32),
        "pose": np.empty((n, core.SEQ_LEN, 34), dtype=np.float32),
        "biu_v2": np.empty((n, 63), dtype=np.float32),
        "biu_v1": np.empty((n, 72), dtype=np.float32),
    }
    fallback = np.zeros(n, dtype=np.bool_)
    valid_joint_fraction = np.zeros(n, dtype=np.float32)
    output_rows = rows.copy()
    output_rows["frame_width"] = 1920
    output_rows["frame_height"] = 1080
    v1_names: tuple[str, ...] | None = None
    v2_names: tuple[str, ...] | None = None
    for position, row in enumerate(rows.itertuples(index=False)):
        video = pie_db[str(row.set_id)][str(row.video_id)]
        width, height = int(video.get("width", 1920)), int(video.get("height", 1080))
        output_rows.loc[position, ["frame_width", "frame_height"]] = [width, height]
        pedestrian = video["ped_annotations"][str(row.ped_id)]
        frames = core.observed_frames(row)
        frame_to_index = {int(frame): index for index, frame in enumerate(pedestrian["frames"])}
        bbox_pixels = []
        coords = []
        confidence = []
        action_source = pedestrian.get("behavior", {}).get("action", [])
        look_source = pedestrian.get("behavior", {}).get("look", [])
        for temporal_index, frame_id in enumerate(frames):
            pedestrian_index = frame_to_index.get(frame_id)
            if pedestrian_index is None:
                raise ValueError(f"Missing PIE pedestrian frame {row.sequence_id}/{frame_id}")
            box = np.asarray(pedestrian["bbox"][pedestrian_index], dtype=np.float32)
            if box.shape != (4,) or not np.isfinite(box).all():
                raise ValueError(f"Invalid PIE BBox {row.sequence_id}/{frame_id}")
            x1, y1, x2, y2 = box
            arrays["bbox"][position, temporal_index] = [
                ((x1 + x2) / 2.0) / width,
                ((y1 + y2) / 2.0) / height,
                (x2 - x1) / width,
                (y2 - y1) / height,
            ]
            bbox_pixels.append(box)
            arrays["ped_action"][position, temporal_index, 0] = float(action_source[pedestrian_index])
            arrays["ped_look"][position, temporal_index, 0] = float(look_source[pedestrian_index])
            ego = video.get("vehicle_annotations", {}).get(frame_id)
            if ego is None:
                raise ValueError(f"Missing ego annotation {row.sequence_id}/{frame_id}")
            speed = float(ego.get("OBD_speed", 0.0) or ego.get("GPS_speed", 0.0))
            arrays["ego_speed"][position, temporal_index, 0] = speed
            arrays["ego_acc"][position, temporal_index] = [float(ego.get("accX", 0.0)), float(ego.get("accY", 0.0))]
            keypoint, keypoint_confidence = pose_index[(str(row.set_id), str(row.video_id), str(row.ped_id), frame_id)]
            coords.append(keypoint)
            confidence.append(keypoint_confidence)
        coords_array = np.stack(coords).astype(np.float32)
        confidence_array = np.stack(confidence).astype(np.float32)
        arrays["pose"][position] = (coords_array / np.asarray([width, height], dtype=np.float32)).reshape(core.SEQ_LEN, 34)
        arrays["yolop"][position] = yolop.window(str(row.set_id), str(row.video_id), str(row.ped_id), frames)
        arrays["static_context"][position] = static_context(pedestrian.get("attributes", {}))
        v1, names1 = core.extract_biu_v1_window(coords_array, confidence_array)
        v2, names2, direction = core.extract_biu_v2_window(
            arrays["pose"][position], arrays["bbox"][position], width=width, height=height
        )
        if v1_names is not None and names1 != v1_names:
            raise RuntimeError("BIU-v1 schema drifted within the cache.")
        if v2_names is not None and names2 != v2_names:
            raise RuntimeError("BIU-v2 schema drifted within the cache.")
        v1_names, v2_names = names1, names2
        arrays["biu_v1"][position] = v1
        arrays["biu_v2"][position] = v2
        fallback[position] = direction == "bbox_center_x"
        valid_joint_fraction[position] = float(np.mean(np.any(coords_array != 0.0, axis=2)))
        if (position + 1) % 1000 == 0 or position + 1 == n:
            log_line(log, f"assemble rows={position + 1}/{n}")

    if scalers is None:
        training = output_rows["split"].eq("train").to_numpy()
        if not training.any():
            raise ValueError("Scaler fitting requires training rows.")
        speed = arrays["ego_speed"][training]
        acceleration = arrays["ego_acc"][training]
        scalers = {
            "ego_speed_mean": float(speed.mean()),
            "ego_speed_std": float(max(speed.std(), 1e-6)),
            "accX_mean": float(acceleration[..., 0].mean()),
            "accX_std": float(max(acceleration[..., 0].std(), 1e-6)),
            "accY_mean": float(acceleration[..., 1].mean()),
            "accY_std": float(max(acceleration[..., 1].std(), 1e-6)),
        }
    arrays["ego_speed"] = ((arrays["ego_speed"] - scalers["ego_speed_mean"]) / scalers["ego_speed_std"]).astype(np.float32)
    arrays["ego_acc"][..., 0] = (arrays["ego_acc"][..., 0] - scalers["accX_mean"]) / scalers["accX_std"]
    arrays["ego_acc"][..., 1] = (arrays["ego_acc"][..., 1] - scalers["accY_mean"]) / scalers["accY_std"]
    metadata = save_cache(
        cache_root,
        output_rows,
        arrays,
        {
            "phase_population": sorted(set(output_rows["split"].astype(str))),
            "scalers": dict(scalers),
            "pose_sha256": POSE_SHA256,
            "pose_valid_window_count": int((valid_joint_fraction > 0).sum()),
            "mean_valid_joint_fraction": float(valid_joint_fraction.mean()),
            "median_valid_joint_fraction": float(np.median(valid_joint_fraction)),
            "bbox_direction_fallback_count": int(fallback.sum()),
            "bbox_direction_fallback_rate": float(fallback.mean()),
            "biu_v1_feature_names": list(v1_names or ()),
            "biu_v2_feature_names": list(v2_names or ()),
        },
    )
    return metadata


def build_augmented_cache(seed: int, base_rows: pd.DataFrame, base_arrays: Mapping[str, np.ndarray]) -> dict[str, object]:
    train_rows = base_rows.loc[base_rows["split"].eq("train")].reset_index(drop=True)
    manifest = core.build_oversampling_manifest(train_rows, seed)
    output = {
        name: np.empty((len(manifest), *base_arrays[name].shape[1:]), dtype=np.float32)
        for name in ARRAY_NAMES
    }
    parity_records = []
    log = ROOT / "logs/prepare.log"
    for position, sampled in enumerate(manifest.itertuples(index=False)):
        source_index = int(sampled.source_cache_index)
        for name in ARRAY_NAMES:
            output[name][position] = base_arrays[name][source_index]
        if bool(sampled.augmented):
            measurements = {
                name: np.asarray(base_arrays[name][source_index], dtype=np.float32)
                for name in ("bbox", "ego_speed", "ego_acc", "yolop", "pose")
            }
            augmented = core.precompute_augmented_row(
                measurements,
                width=int(base_rows.iloc[source_index]["frame_width"]),
                height=int(base_rows.iloc[source_index]["frame_height"]),
                noise_seed=int(sampled.noise_seed),
                base_biu_v1=np.asarray(base_arrays["biu_v1"][source_index], dtype=np.float32),
            )
            for name in ("bbox", "ego_speed", "ego_acc", "yolop", "pose", "biu_v1", "biu_v2"):
                output[name][position] = augmented[name]
            if len(parity_records) < 8:
                direct = core.precompute_augmented_row(
                    measurements,
                    width=int(base_rows.iloc[source_index]["frame_width"]),
                    height=int(base_rows.iloc[source_index]["frame_height"]),
                    noise_seed=int(sampled.noise_seed),
                    base_biu_v1=np.asarray(base_arrays["biu_v1"][source_index], dtype=np.float32),
                )
                parity_records.append(
                    {
                        "seed": seed,
                        "manifest_position": position,
                        "source_cache_index": source_index,
                        "biu_v1_exact": bool(np.array_equal(output["biu_v1"][position], direct["biu_v1"])),
                        "biu_v2_exact": bool(np.array_equal(output["biu_v2"][position], direct["biu_v2"])),
                    }
                )
        if (position + 1) % 1000 == 0 or position + 1 == len(manifest):
            log_line(log, f"augment seed={seed} rows={position + 1}/{len(manifest)}")
    parity = pd.DataFrame(parity_records)
    if parity.empty or not parity[["biu_v1_exact", "biu_v2_exact"]].all().all():
        raise RuntimeError(f"Offline augmented BIU parity failed for seed {seed}.")
    augmented_root = ROOT / f"cache/augmented_seed{seed}"
    metadata = save_cache(
        augmented_root,
        manifest,
        output,
        {
            "seed": seed,
            "source_base_metadata_sha256": core.sha256_file(BASE_CACHE / "metadata.json"),
            "class_counts": {str(key): int(value) for key, value in manifest["label"].value_counts().items()},
            "augmented_count": int(manifest["augmented"].sum()),
        },
    )
    atomic_csv(parity, ROOT / f"metrics/augmentation_parity_seed{seed}.csv")
    atomic_csv(manifest, ROOT / f"metrics/oversampling_manifest_seed{seed}.csv")
    return metadata


def run_prepare() -> None:
    if PREPARE_MARKER.is_file():
        payload = json.loads(PREPARE_MARKER.read_text(encoding="utf-8"))
        authenticate_artifacts(ROOT, payload["artifacts"])
        log_line(ROOT / "logs/prepare.log", "status=already_complete_authenticated")
        return
    if core.sha256_file(POSE_PATH) != POSE_SHA256:
        raise ValueError("Pinned Exp012 pose SHA-256 mismatch.")
    rows = select_phase_rows(load_phase_rows("prepare"), phase="prepare")
    if set(rows["split"].astype(str)) != {"train", "val"} or rows["set_id"].eq("set03").any():
        raise RuntimeError("Preparation population leaked set03.")
    metadata = assemble_base_cache(rows, BASE_CACHE, scalers=None)
    base_rows, base_arrays, _ = load_cache(BASE_CACHE)
    for seed in core.SEEDS:
        build_augmented_cache(seed, base_rows, base_arrays)
    artifacts = {"cache/trainval_base/metadata.json": core.sha256_file(BASE_CACHE / "metadata.json")}
    for seed in core.SEEDS:
        artifacts[f"cache/augmented_seed{seed}/metadata.json"] = core.sha256_file(
            ROOT / f"cache/augmented_seed{seed}/metadata.json"
        )
        artifacts[f"metrics/augmentation_parity_seed{seed}.csv"] = core.sha256_file(
            ROOT / f"metrics/augmentation_parity_seed{seed}.csv"
        )
    payload = {
        "phase": "prepare",
        "created_at": utc_now(),
        "train_rows": core.EXPECTED_SPLIT_COUNTS["train"],
        "val_rows": core.EXPECTED_SPLIT_COUNTS["val"],
        "test_accessed": False,
        "pose_sha256": POSE_SHA256,
        "scalers": metadata["scalers"],
        "artifacts": artifacts,
    }
    atomic_json(payload, PREPARE_MARKER)
    log_line(ROOT / "logs/prepare.log", "phase=prepare status=complete test_accessed=false")


def tensor_for_stream(array: np.ndarray, stream: str, device: torch.device) -> torch.Tensor:
    value = np.asarray(array, dtype=np.float32)
    if stream == "static_context":
        value = np.broadcast_to(value[:, None, :], (len(value), core.SEQ_LEN, 23)).copy()
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def probabilities(model: nn.Module, tensors: Mapping[str, torch.Tensor], n: int, batch_size: int = 512) -> np.ndarray:
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            inputs = {name: value[start : start + batch_size] for name, value in tensors.items()}
            output.append(torch.softmax(model(inputs), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(output)


def run_training(variant: str, seed: int) -> None:
    if variant not in core.VARIANTS or seed not in core.SEEDS:
        raise ValueError("Training identity is outside the frozen grid.")
    marker = training_marker(ROOT, variant, seed)
    if marker.is_file():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        authenticate_artifacts(ROOT, payload["artifacts"])
        log_line(ROOT / f"logs/train_{variant}_seed{seed}.log", "status=already_complete_authenticated")
        return
    if not PREPARE_MARKER.is_file():
        raise RuntimeError("Training requires authenticated preparation.")
    prepare_payload = json.loads(PREPARE_MARKER.read_text(encoding="utf-8"))
    authenticate_artifacts(ROOT, prepare_payload["artifacts"])
    if prepare_payload.get("test_accessed") is not False:
        raise RuntimeError("Training preparation must prove set03 was not accessed.")
    if not torch.cuda.is_available():
        raise RuntimeError("Exp050 training is GPU-only; CUDA is unavailable.")
    device = torch.device("cuda")
    base_rows, base_arrays, _ = load_cache(BASE_CACHE)
    augmented_rows, augmented_arrays, _ = load_cache(ROOT / f"cache/augmented_seed{seed}")
    val_indices = np.flatnonzero(base_rows["split"].eq("val").to_numpy())
    streams = core.VARIANT_STREAMS[variant]
    train_tensors = {name: tensor_for_stream(augmented_arrays[name], name, device) for name in streams}
    val_tensors = {name: tensor_for_stream(base_arrays[name][val_indices], name, device) for name in streams}
    train_labels = torch.as_tensor(augmented_rows["label"].to_numpy(dtype=np.int64), device=device)
    val_labels = base_rows.iloc[val_indices]["label"].to_numpy(dtype=np.int64)
    core.seed_everything(seed)
    model = core.MslstmPracticalModel(variant).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=core.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=torch.ones(2, device=device))
    rng = np.random.RandomState(seed)
    history: list[dict[str, object]] = []
    best_f1 = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    state_path = ROOT / f"checkpoints/{variant}_seed{seed}.state.pt"
    start_epoch = 1
    if state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["variant"] != variant or int(state["seed"]) != seed:
            raise ValueError("Resumable state identity drifted.")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        rng.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        history = list(state["history"])
        best_f1 = float(state["best_f1"])
        best_epoch = int(state["best_epoch"])
        best_state = state["best_state"]
        start_epoch = int(state["last_epoch"]) + 1
    log = ROOT / f"logs/train_{variant}_seed{seed}.log"
    log_line(log, f"phase=train status=starting variant={variant} seed={seed} device={device} start_epoch={start_epoch}")
    started = time.time()
    n_train = len(train_labels)
    for epoch in range(start_epoch, core.EPOCHS + 1):
        model.train()
        order = rng.permutation(n_train)
        loss_sum = 0.0
        correct = 0
        for start in range(0, n_train, core.BATCH_SIZE):
            index = torch.as_tensor(order[start : start + core.BATCH_SIZE], dtype=torch.long, device=device)
            inputs = {name: value.index_select(0, index) for name, value in train_tensors.items()}
            labels = train_labels.index_select(0, index)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(index)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
        val_probability = probabilities(model, val_tensors, len(val_indices))
        metrics = core.compute_metrics(val_labels, val_probability)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / n_train,
                "train_accuracy": correct / n_train,
                **{f"val_{name}": value for name, value in metrics.items()},
            }
        )
        if float(metrics["f1"]) > best_f1:
            best_f1 = float(metrics["f1"])
            best_epoch = epoch
            best_state = copy.deepcopy({name: value.detach().cpu() for name, value in model.state_dict().items()})
        atomic_torch_save(
            {
                "variant": variant,
                "seed": seed,
                "last_epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "numpy_rng_state": rng.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
                "history": history,
                "best_f1": best_f1,
                "best_epoch": best_epoch,
                "best_state": best_state,
            },
            state_path,
        )
        atomic_csv(pd.DataFrame(history), ROOT / f"logs/training_history_{variant}_seed{seed}.csv")
        log_line(
            log,
            f"epoch={epoch}/{core.EPOCHS} train_loss={loss_sum/n_train:.6f} "
            f"val_f1={float(metrics['f1']):.6f} val_pr_auc={float(metrics['pr_auc']):.6f} best_epoch={best_epoch}",
        )
    if best_state is None or len(history) != core.EPOCHS:
        raise RuntimeError("Frozen training did not complete 15 epochs.")
    model.load_state_dict(best_state)
    model.to(device)
    val_probability = probabilities(model, val_tensors, len(val_indices))
    checkpoint = ROOT / f"checkpoints/{variant}_seed{seed}.best.pt"
    atomic_torch_save(
        {
            "variant": variant,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_f1": best_f1,
            "model_state": best_state,
            "total_parameters": core.count_parameters(model),
            "streams": streams,
            "prepare_sha256": core.sha256_file(PREPARE_MARKER),
        },
        checkpoint,
    )
    prediction = base_rows.iloc[val_indices][
        ["sequence_id", "split", "set_id", "video_id", "ped_id", "start_frame", "end_frame", "event_frame", "time_to_event_frames", "label"]
    ].copy()
    prediction["variant"] = variant
    prediction["seed"] = seed
    prediction["probability"] = val_probability
    prediction["prediction"] = (val_probability >= core.FIXED_THRESHOLD).astype(np.int64)
    prediction_path = ROOT / f"predictions/validation_{variant}_seed{seed}.csv"
    atomic_csv(prediction, prediction_path)
    history_path = ROOT / f"logs/training_history_{variant}_seed{seed}.csv"
    state_path.unlink(missing_ok=True)
    marker_payload = {
        "phase": "train",
        "created_at": utc_now(),
        "variant": variant,
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_f1": best_f1,
        "training_seconds": time.time() - started,
        "artifacts": {
            str(checkpoint.relative_to(ROOT)): core.sha256_file(checkpoint),
            str(prediction_path.relative_to(ROOT)): core.sha256_file(prediction_path),
            str(history_path.relative_to(ROOT)): core.sha256_file(history_path),
        },
    }
    atomic_json(marker_payload, marker)
    log_line(log, f"phase=train status=complete best_epoch={best_epoch} best_val_f1={best_f1:.6f}")
    del model, train_tensors, val_tensors, train_labels
    torch.cuda.empty_cache()


def run_train_all() -> None:
    for variant, seed in required_training_identities():
        run_training(variant, seed)
    assert_training_complete(ROOT)
    atomic_json(
        {"phase": "train_all", "created_at": utc_now(), "run_count": 18},
        ROOT / "TRAINING_COMPLETE",
    )


def load_test_cache() -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, object]]:
    if TEST_CACHE.is_dir() and (TEST_CACHE / "metadata.json").is_file():
        return load_cache(TEST_CACHE)
    prepare = json.loads(PREPARE_MARKER.read_text(encoding="utf-8"))
    rows = select_phase_rows(load_phase_rows("evaluate"), phase="evaluate")
    assemble_base_cache(rows, TEST_CACHE, scalers=prepare["scalers"])
    return load_cache(TEST_CACHE)


def transition_table(predictions: Mapping[tuple[str, int], pd.DataFrame]) -> pd.DataFrame:
    comparisons = (("A", "B"), ("B", "C"), ("C", "D"), ("C", "E"), ("C", "F"), ("D", "E"), ("D", "F"), ("E", "F"))
    records = []
    for seed in core.SEEDS:
        for left, right in comparisons:
            a = predictions[(left, seed)]
            b = predictions[(right, seed)]
            if not a["sequence_id"].equals(b["sequence_id"]):
                raise RuntimeError("Prediction cohort ordering drifted.")
            a_correct = a["prediction"].to_numpy() == a["label"].to_numpy()
            b_correct = b["prediction"].to_numpy() == b["label"].to_numpy()
            records.append(
                {
                    "comparison": f"{left}_to_{right}",
                    "seed": seed,
                    "left_wrong_right_correct": int((~a_correct & b_correct).sum()),
                    "left_correct_right_wrong": int((a_correct & ~b_correct).sum()),
                    "both_wrong": int((~a_correct & ~b_correct).sum()),
                    "both_correct": int((a_correct & b_correct).sum()),
                }
            )
    return pd.DataFrame(records)


def run_evaluate() -> None:
    if EVALUATION_MARKER.is_file():
        payload = json.loads(EVALUATION_MARKER.read_text(encoding="utf-8"))
        authenticate_artifacts(ROOT, payload["artifacts"])
        log_line(ROOT / "logs/evaluate.log", "status=already_complete_authenticated")
        return
    training = assert_training_complete(ROOT)
    if not torch.cuda.is_available():
        raise RuntimeError("Exp050 final inference is GPU-only; CUDA is unavailable.")
    device = torch.device("cuda")
    test_rows, test_arrays, test_metadata = load_test_cache()
    if set(test_rows["split"].astype(str)) != {"test"} or set(test_rows["set_id"].astype(str)) != {"set03"}:
        raise RuntimeError("Final cache is not designated set03.")
    labels = test_rows["label"].to_numpy(dtype=np.int64)
    per_seed = []
    predictions: dict[tuple[str, int], pd.DataFrame] = {}
    efficiency = []
    for variant, seed in required_training_identities():
        checkpoint_path = ROOT / f"checkpoints/{variant}_seed{seed}.best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = core.MslstmPracticalModel(variant).to(device)
        model.load_state_dict(checkpoint["model_state"])
        streams = core.VARIANT_STREAMS[variant]
        tensors = {name: tensor_for_stream(test_arrays[name], name, device) for name in streams}
        torch.cuda.synchronize()
        started = time.perf_counter()
        probability = probabilities(model, tensors, len(test_rows))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        metric = core.compute_metrics(labels, probability)
        validation = next(item for item in training if item["variant"] == variant and int(item["seed"]) == seed)
        per_seed.append(
            {
                "variant": variant,
                "seed": seed,
                "streams": "+".join(streams),
                "best_epoch": int(checkpoint["best_epoch"]),
                "best_validation_f1": float(checkpoint["best_validation_f1"]),
                "total_parameters": int(checkpoint["total_parameters"]),
                "training_seconds": float(validation["training_seconds"]),
                **metric,
            }
        )
        frame = test_rows[
            ["sequence_id", "split", "set_id", "video_id", "ped_id", "start_frame", "end_frame", "event_frame", "time_to_event_frames", "label"]
        ].copy()
        frame["variant"] = variant
        frame["seed"] = seed
        frame["threshold"] = core.FIXED_THRESHOLD
        frame["probability"] = probability
        frame["prediction"] = (probability >= core.FIXED_THRESHOLD).astype(np.int64)
        frame["error"] = np.where(
            (frame["label"] == 0) & (frame["prediction"] == 1),
            "FP",
            np.where((frame["label"] == 1) & (frame["prediction"] == 0), "FN", np.where(frame["label"] == 1, "TP", "TN")),
        )
        prediction_path = ROOT / f"predictions/set03_{variant}_seed{seed}.csv"
        atomic_csv(frame, prediction_path)
        predictions[(variant, seed)] = frame
        efficiency.append(
            {
                "variant": variant,
                "seed": seed,
                "device": str(device),
                "total_parameters": int(checkpoint["total_parameters"]),
                "inference_seconds": elapsed,
                "seconds_per_sample": elapsed / len(test_rows),
                "pose_estimator_runtime_included": False,
            }
        )
        log_line(ROOT / "logs/evaluate.log", f"variant={variant} seed={seed} f1={metric['f1']:.6f} pr_auc={metric['pr_auc']:.6f}")
        del model, tensors
        torch.cuda.empty_cache()
    per_seed_frame = pd.DataFrame(per_seed)
    summary = per_seed_frame.groupby("variant", sort=False).agg(
        **{
            f"{metric}_{stat}": (metric, stat)
            for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "total_parameters", "training_seconds")
            for stat in ("mean", "std")
        }
    ).reset_index()
    comparisons = (("A", "B"), ("B", "C"), ("C", "D"), ("C", "E"), ("C", "F"), ("D", "E"), ("D", "F"), ("E", "F"), ("A", "E"), ("A", "F"))
    delta_records = []
    indexed = per_seed_frame.set_index(["variant", "seed"])
    for left, right in comparisons:
        for seed in core.SEEDS:
            record = {"comparison": f"{left}_to_{right}", "seed": seed}
            for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
                record[f"{metric}_delta"] = float(indexed.loc[(right, seed), metric] - indexed.loc[(left, seed), metric])
            delta_records.append(record)
    deltas = pd.DataFrame(delta_records)
    recovery = []
    for metric in ("f1", "pr_auc", "roc_auc"):
        pivot = per_seed_frame.pivot(index="seed", columns="variant", values=metric)
        gt_gain = pivot["B"] - pivot["C"]
        record: dict[str, object] = {
            "metric": metric,
            "gt_behavior_gain_mean": float(gt_gain.mean()),
            "raw_pose_recovery_mean": float((pivot["D"] - pivot["C"]).mean()),
            "biu_v2_recovery_mean": float((pivot["E"] - pivot["C"]).mean()),
            "biu_v1_recovery_mean": float((pivot["F"] - pivot["C"]).mean()),
        }
        for name, variant in (("raw_pose", "D"), ("biu_v2", "E"), ("biu_v1", "F")):
            fraction = np.where(gt_gain.to_numpy() > 0, (pivot[variant] - pivot["C"]) / gt_gain, np.nan)
            record[f"{name}_recovery_fraction_mean"] = float(np.nanmean(fraction)) if np.isfinite(fraction).any() else float("nan")
        recovery.append(record)
    metrics_root = ROOT / "metrics"
    output_tables = {
        "per_seed_results.csv": per_seed_frame,
        "fixed_0p5_results.csv": per_seed_frame,
        "summary_results.csv": summary,
        "paired_deltas.csv": deltas,
        "behavior_recovery.csv": pd.DataFrame(recovery),
        "efficiency.csv": pd.DataFrame(efficiency),
        "error_transitions_fixed_0p5.csv": transition_table(predictions),
    }
    for name, frame in output_tables.items():
        atomic_csv(frame, metrics_root / name)
    report_lines = [
        "# Exp050 Final Report",
        "",
        "Fresh controlled A--F results on PIE 16 observed frames / TTE30--60. Fixed threshold: 0.5.",
        "",
        frame_to_markdown(summary),
        "",
        "The set03 result is the designated Exp050 final evaluation. set03 had prior independent post-hoc exposure, but Exp050 was frozen before this evaluation.",
        "",
        "Pose extraction and YOLOP extraction runtimes are excluded because pinned historical caches were reused.",
    ]
    report_path = ROOT / "FINAL_REPORT.md"
    temporary_report = report_path.with_suffix(".md.tmp")
    temporary_report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    temporary_report.replace(report_path)
    artifacts = {f"metrics/{name}": core.sha256_file(metrics_root / name) for name in output_tables}
    artifacts["FINAL_REPORT.md"] = core.sha256_file(report_path)
    for variant, seed in required_training_identities():
        relative = f"predictions/set03_{variant}_seed{seed}.csv"
        artifacts[relative] = core.sha256_file(ROOT / relative)
    payload = {
        "phase": "evaluate",
        "created_at": utc_now(),
        "test_rows": len(test_rows),
        "test_sequence_id_sha256": test_metadata["sequence_id_sha256"],
        "artifacts": artifacts,
    }
    atomic_json(payload, EVALUATION_MARKER)
    log_line(ROOT / "logs/evaluate.log", "phase=evaluate status=complete")


def run_verify() -> None:
    assert_training_complete(ROOT)
    if not EVALUATION_MARKER.is_file():
        raise RuntimeError("Evaluation marker is missing.")
    evaluation = json.loads(EVALUATION_MARKER.read_text(encoding="utf-8"))
    authenticate_artifacts(ROOT, evaluation["artifacts"])
    COMPLETE_MARKER.write_text(
        json.dumps(
            {
                "experiment": 50,
                "completed_at": utc_now(),
                "training_runs": 18,
                "test_rows": int(evaluation["test_rows"]),
                "evaluation_marker_sha256": core.sha256_file(EVALUATION_MARKER),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Exp050 verification passed; EXPERIMENT_COMPLETE written.")


def gpu_smoke() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in Exp050 execution environment.")
    device = torch.device("cuda")
    model = core.MslstmPracticalModel("E").to(device)
    inputs = {
        name: torch.zeros((2, core.STREAM_DIMS[name]), device=device)
        if name.startswith("biu_")
        else torch.zeros((2, core.SEQ_LEN, core.STREAM_DIMS[name]), device=device)
        for name in core.VARIANT_STREAMS["E"]
    }
    loss = model(inputs).sum()
    loss.backward()
    print(json.dumps({"device": str(device), "gpu": torch.cuda.get_device_name(0), "loss": float(loss.item())}))


def status() -> None:
    completed = [f"{variant}{seed}" for variant, seed in required_training_identities() if training_marker(ROOT, variant, seed).is_file()]
    active_states = sorted(path.name for path in (ROOT / "checkpoints").glob("*.state.pt"))
    print(json.dumps(
        {
            "prepare_complete": PREPARE_MARKER.is_file(),
            "completed_training_runs": completed,
            "completed_training_count": len(completed),
            "active_states": active_states,
            "evaluation_complete": EVALUATION_MARKER.is_file(),
            "experiment_complete": COMPLETE_MARKER.is_file(),
        },
        indent=2,
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    train = subparsers.add_parser("train")
    train.add_argument("--variant", choices=core.VARIANTS, required=True)
    train.add_argument("--seed", type=int, choices=core.SEEDS, required=True)
    subparsers.add_parser("train-all")
    subparsers.add_parser("evaluate")
    subparsers.add_parser("verify")
    subparsers.add_parser("gpu-smoke")
    subparsers.add_parser("status")
    args = parser.parse_args()
    if args.command == "prepare":
        run_prepare()
    elif args.command == "train":
        run_training(args.variant, args.seed)
    elif args.command == "train-all":
        run_train_all()
    elif args.command == "evaluate":
        run_evaluate()
    elif args.command == "verify":
        run_verify()
    elif args.command == "gpu-smoke":
        gpu_smoke()
    elif args.command == "status":
        status()


if __name__ == "__main__":
    main()
