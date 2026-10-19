"""Amended PSI-native S187 versus R247 train/validation experiment."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
EXP113 = REPO / "experiments/113_psi_native_biu_xgb"
POSE = REPO / "data/psi_vitpose_s_all_tracks.jsonl"
POSE_SHA256 = "31d8f45a7f8b61803b2dde8f981ce012086ac9905455d7aaac743bc1ec2a3fc7"
EXCLUDED = {"video_0032", "video_0055", "video_0093"}
SEEDS = (42, 43, 44)


def guard(event, args):
    if event == "open" and isinstance(args[0], (str, bytes)):
        path = str(args[0]).replace("\\", "/").lower()
        if "/psi2.0_test/" in path or "/test_evaluation/" in path or "/cache/test/" in path:
            raise RuntimeError("Exp114 test data are closed")


sys.addaudithook(guard)


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


HELPER = module(
    "psi114_helpers",
    REPO / "experiments/098_jaad_matched_biu_xgb/scripts/run_experiment.py",
)
V2 = module("psi114_v2", REPO / "src/biu_v2_extractor.py")
write, record = HELPER.write, HELPER.record


def atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows() -> dict[str, pd.DataFrame]:
    train = pd.read_csv(EXP113 / "cache/train/rows.csv")
    val = pd.read_csv(EXP113 / "cache/val/rows.csv")
    train = train.loc[~train.video_id.astype(str).isin(EXCLUDED)].reset_index(drop=True)
    assert len(train) == 26295 and len(val) == 14922
    assert not set(train.video_id.astype(str)) & EXCLUDED
    assert not set(val.video_id.astype(str)) & EXCLUDED
    return {"train": train, "val": val}


def observed(row) -> list[int]:
    frames = list(map(int, json.loads(row.observed_frames_json)))
    if len(frames) != 15 or frames != list(range(frames[0], frames[0] + 15)):
        raise RuntimeError(f"invalid observation frames: {row.sequence_id}")
    if int(row.target_frame) != frames[-1] + 1 or int(row.target_frame) in frames:
        raise RuntimeError(f"invalid next-frame target: {row.sequence_id}")
    return frames


def load_pose(required: set[tuple[str, str, int]]):
    if sha256(POSE) != POSE_SHA256:
        raise RuntimeError("PSI pose cache hash changed")
    index = {}
    with POSE.open() as handle:
        for line in handle:
            item = json.loads(line)
            key = (str(item["video"]), str(item["pid"]), int(item["frame_idx"]))
            if key not in required:
                continue
            xy = np.asarray(item["keypoints"], dtype=np.float32)
            confidence = np.asarray(item["kp_conf"], dtype=np.float32)
            if key in index or xy.shape != (17, 2) or confidence.shape != (17,):
                raise RuntimeError(f"invalid pose record: {key}")
            if not np.isfinite(xy).all() or not np.isfinite(confidence).all():
                raise RuntimeError(f"non-finite pose record: {key}")
            index[key] = (xy, confidence)
    missing = required - set(index)
    if missing:
        raise RuntimeError(f"retained cohort unexpectedly lacks pose: {next(iter(missing))}")
    return index


def track_geometry(video: str) -> dict[str, tuple[dict[int, np.ndarray], list[int]]]:
    path = EXP113 / f"dataset/PSI2.0_TrainVal/annotations/cognitive_annotation_extended/{video}/pedestrian_intent.json"
    source = json.loads(path.read_text())
    if source.get("video_name") != video:
        raise RuntimeError(f"cognitive video identity mismatch: {video}")
    output = {}
    for pid, item in source["pedestrians"].items():
        frames = list(map(int, item["observed_frames"]))
        boxes = np.asarray(item["cv_annotations"]["bboxes"], dtype=np.float32)
        if boxes.shape != (len(frames), 4) or len(frames) != len(set(frames)):
            raise RuntimeError(f"malformed cognitive BBoxes: {video}/{pid}")
        mapping = {frame: box for frame, box in zip(frames, boxes)}
        output[str(pid)] = (mapping, frames)
    return output


def sequence_pid(row) -> str:
    pieces = str(row.sequence_id).split(":")
    if len(pieces) != 4 or pieces[0] != str(row.video_id):
        raise RuntimeError(f"unexpected sequence identity: {row.sequence_id}")
    # Native gap splitting may suffix the database sequence identity (for
    # example track_13-0); geometry remains keyed by the original track ID.
    return str(row.ped_id)


def half_values(coords: np.ndarray, confidence: np.ndarray, boxes: np.ndarray) -> dict[str, float]:
    extracted = V2.extract_biu_v2(
        coords, confidence, bbox_sequence=boxes, image_size=(1280, 720)
    )
    widths = boxes[:, 2] - boxes[:, 0]
    centers = (boxes[:, 0] + boxes[:, 2]) / 2.0
    areas = widths * (boxes[:, 3] - boxes[:, 1])
    values = {
        "lower_body_motion_magnitude_mean": float(extracted["lower_body_motion_magnitude_mean"]),
        "lead_ankle_forward_delta": float(extracted["lead_ankle_forward_delta"]),
        "torso_motion_magnitude_mean": float(extracted["torso_motion_magnitude_mean"]),
        "torso_center_x_delta": float(extracted["torso_center_x_delta"]),
        "bbox_center_dx_norm": float((centers[-1] - centers[0]) / widths.mean()),
        "bbox_scale_log_delta": float(math.log(max(float(areas[-1]), 1e-12) / max(float(areas[0]), 1e-12))),
        "pose_confidence_mean": float(confidence.mean()),
    }
    if not np.isfinite(list(values.values())).all():
        raise RuntimeError("FIRST7/LAST8 temporal values are non-finite")
    return values


TEMPORAL_BASES = (
    "lower_body_motion_magnitude_mean", "lead_ankle_forward_delta",
    "torso_motion_magnitude_mean", "torso_center_x_delta", "bbox_center_dx_norm",
    "bbox_scale_log_delta", "pose_confidence_mean",
)


def temporal_values(coords: np.ndarray, confidence: np.ndarray, boxes: np.ndarray) -> list[float]:
    if coords.shape != (15, 17, 2) or confidence.shape != (15, 17) or boxes.shape != (15, 4):
        raise RuntimeError("temporal input must be exact PSI15")
    early = half_values(coords[:7], confidence[:7], boxes[:7])
    late = half_values(coords[7:], confidence[7:], boxes[7:])
    return [value for base in TEMPORAL_BASES for value in (early[base], late[base], late[base] - early[base])]


def speed_values(speed: dict[int, float], frames: list[int]) -> list[float]:
    missing = set(frames) - set(speed)
    if missing:
        raise RuntimeError(f"missing official observed speed frame: {min(missing)}")
    values = np.asarray([speed[frame] for frame in frames], dtype=np.float64)
    if values.shape != (15,) or not np.isfinite(values).all():
        raise RuntimeError("speed input must be finite PSI15")
    return [float(values.mean()), float(values.std(ddof=0)), float(values[0]), float(values[-1]), float(values[-1] - values[0])]


def prepare_source() -> None:
    speed_manifest = json.loads((ROOT / "metrics/speed_recovery.json").read_text())
    if not speed_manifest.get("all_required_frames_have_numeric_speed"):
        raise RuntimeError("source preparation requires the complete speed gate")
    contract = json.loads((ROOT / "feature_contract.json").read_text())
    s_names = contract["S187"]
    static_names = s_names[:161]
    if any(name.startswith("temporal__") for name in static_names) or not s_names[161].startswith("temporal__"):
        raise RuntimeError("S187 feature-family boundary changed")
    old_contract = json.loads((EXP113 / "feature_contract.json").read_text())
    indices = {
        arm: {name: index for index, name in enumerate(old_contract[arm])}
        for arm in ("A", "B", "C")
    }
    source_map = []
    for name in static_names:
        if name in indices["A"]:
            source_map.append(("A", indices["A"][name]))
        elif f"v1__{name}" in indices["B"]:
            source_map.append(("B", indices["B"][f"v1__{name}"]))
        elif name in indices["C"]:
            source_map.append(("C", indices["C"][name]))
        else:
            raise RuntimeError(f"Exp113 source cache lacks frozen union feature: {name}")
    populations = rows()
    required = {
        (str(row.video_id), str(row.ped_id), frame)
        for frame in populations.values()
        for row in frame.itertuples(index=False)
        for frame in observed(row)
    }
    pose = load_pose(required)
    audits = []
    for split, frame in populations.items():
        original_rows = pd.read_csv(EXP113 / f"cache/{split}/rows.csv")
        keep = ~original_rows.video_id.astype(str).isin(EXCLUDED) if split == "train" else np.ones(len(original_rows), dtype=bool)
        arrays = {arm: np.load(EXP113 / f"cache/{split}/{arm}.npy")[keep] for arm in ("A", "B", "C")}
        geometry_by_video = {}
        speed_by_video = {}
        output = np.empty((len(frame), 187), dtype=np.float32)
        for position, row in enumerate(frame.itertuples(index=False)):
            video, pid = str(row.video_id), sequence_pid(row)
            if video not in geometry_by_video:
                geometry_by_video[video] = track_geometry(video)
            if video not in speed_by_video:
                compact = json.loads((ROOT / f"cache/speed/{video}.json").read_text())
                if compact["unit"] != "km/hr" or compact["video"] != video:
                    raise RuntimeError(f"speed identity/unit mismatch: {video}")
                speed_by_video[video] = {int(key): float(value) for key, value in compact["speed_km_hr"].items()}
            frames = observed(row)
            try:
                box_map = geometry_by_video[video][pid][0]
                boxes = np.stack([box_map[index] for index in frames])
            except KeyError as error:
                raise RuntimeError(f"missing exact observed BBox: {row.sequence_id}") from error
            keys = [(video, str(row.ped_id), index) for index in frames]
            coords = np.stack([pose[key][0] for key in keys])
            confidence = np.stack([pose[key][1] for key in keys])
            static = [arrays[arm][position, index] for arm, index in source_map]
            output[position] = np.asarray(static + temporal_values(coords, confidence, boxes) + speed_values(speed_by_video[video], frames), dtype=np.float32)
            if position % 2000 == 0:
                print(f"source {split}: {position}/{len(frame)}", flush=True)
        if output.shape != (len(frame), 187) or not np.isfinite(output).all():
            raise RuntimeError(f"malformed S187 matrix: {split}")
        directory = ROOT / f"cache/features/{split}"
        atomic_npy(directory / "S187.npy", output)
        write(directory / "rows.csv", frame)
        audits.append({"split": split, "windows": len(frame), "features": 187, "all_finite": True})
    write(ROOT / "metrics/source_feature_audit.json", {
        "status": "complete", "development_only": True, "test_accessed": False,
        "first_partition_frames": 7, "last_partition_frames": 8,
        "target_frame_used_as_feature": False, "speed_unit": "km/hr", "cohorts": audits,
    })


def road_modules():
    scripts = REPO / "experiments/075_pie_twinlitenetplus_road_context/scripts"
    sys.path.insert(0, str(scripts))
    try:
        # Both Exp075 modules import this exact local name as their trust anchor.
        module("contract", scripts / "contract.py")
        adapter = module("psi114_twinlite", scripts / "twinlite_adapter.py")
        context = module("psi114_road_context", scripts / "context.py")
    finally:
        sys.path.remove(str(scripts))
    return adapter, context


def decode_required(video_path: Path, required: set[int], batch_size: int = 1):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open PSI video: {video_path}")
    maximum = max(required)
    batch_frames, batch_images = [], []
    try:
        index = 0
        while index <= maximum:
            ok, image = capture.read()
            if not ok or image is None:
                raise RuntimeError(f"cannot decode PSI frame: {video_path}:{index}")
            if index in required:
                batch_frames.append(index)
                batch_images.append(image)
                if len(batch_images) == batch_size:
                    yield batch_frames, batch_images
                    batch_frames, batch_images = [], []
            index += 1
        if batch_images:
            yield batch_frames, batch_images
    finally:
        capture.release()


def extract_road() -> None:
    video_manifest = json.loads((ROOT / "metrics/video_recovery.json").read_text())
    if video_manifest.get("status") != "complete" or video_manifest.get("videos") != 140:
        raise RuntimeError("road extraction requires all 140 verified videos")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("primary TwinLite extraction requires CUDA")
    adapter, context = road_modules()
    model = adapter.TwinLiteExtractor(device="cuda", half=True)
    checkpoint = REPO / "experiments/075_pie_twinlitenetplus_road_context/checkpoints/large.pth"
    if sha256(checkpoint) != "5605d1a8ce762c1b6f7d23f4c2e7ec9384bfd6e6887b040b43e1f8902a817e22":
        raise RuntimeError("TwinLite checkpoint hash changed")
    populations = rows()
    all_rows = pd.concat([frame.assign(_split=split) for split, frame in populations.items()], ignore_index=True)
    reports = []
    for number, (video, frame) in enumerate(all_rows.groupby("video_id", sort=True), start=1):
        video = str(video)
        part = ROOT / f"cache/road_parts/{video}.npz"
        if part.is_file():
            with np.load(part, allow_pickle=False) as saved:
                if len(saved["sequence_id"]) != len(frame) or saved["road"].shape != (len(frame), 60):
                    raise RuntimeError(f"malformed resumed road part: {video}")
            print(f"road {number}/140 {video}: reused", flush=True)
            continue
        geometry = track_geometry(video)
        needed_keys = {}
        for row in frame.itertuples(index=False):
            pid = sequence_pid(row)
            for index in observed(row):
                needed_keys[(pid, index)] = geometry[pid][0][index]
        needed_frames = {index for _, index in needed_keys}
        keys_by_frame = {}
        for key in needed_keys:
            keys_by_frame.setdefault(key[1], []).append(key)
        static = {}
        drivable_zero = lane_zero = 0
        foreground = [[], []]
        start = time.perf_counter()
        for frame_ids, images in decode_required(ROOT / f"cache/videos/{video}.mp4", needed_frames):
            drivable, lane = model.predict(images)
            for index, road_mask, lane_mask in zip(frame_ids, drivable, lane):
                drivable_zero += int(not road_mask.any())
                lane_zero += int(not lane_mask.any())
                foreground[0].append(float(road_mask.mean()))
                foreground[1].append(float(lane_mask.mean()))
                for pid, frame_index in keys_by_frame[index]:
                    static[(pid, index)] = context.road_relative_static(
                        road_mask, lane_mask, needed_keys[(pid, frame_index)]
                    )
        ids, road_values = [], []
        for row in frame.itertuples(index=False):
            pid, frames = sequence_pid(row), observed(row)
            boxes = np.stack([geometry[pid][0][index] for index in frames])
            sequence = context._REPRESENTATIONS.assemble_road_relative_temporal(
                np.stack([static[(pid, index)] for index in frames]), boxes
            )
            early, late = sequence[:7].mean(axis=0), sequence[7:].mean(axis=0)
            road_values.append(np.column_stack((early, late, late - early)).reshape(-1))
            ids.append(str(row.sequence_id))
        values = np.asarray(road_values, dtype=np.float32)
        if values.shape != (len(frame), 60) or not np.isfinite(values).all():
            raise RuntimeError(f"malformed road features: {video}")
        part.parent.mkdir(parents=True, exist_ok=True)
        temporary = part.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, sequence_id=np.asarray(ids), road=values)
        os.replace(temporary, part)
        reports.append({
            "video": video, "windows": len(frame), "unique_frames": len(needed_frames),
            "seconds": time.perf_counter() - start, "drivable_zero_frames": drivable_zero,
            "lane_zero_frames": lane_zero, "drivable_foreground_mean": float(np.mean(foreground[0])),
            "lane_foreground_mean": float(np.mean(foreground[1])),
        })
        print(f"road {number}/140 {video}: {len(needed_frames)} unique frames", flush=True)
    # Assemble canonical split order only after every independently resumable part exists.
    for split, expected in populations.items():
        indexed = {}
        for video in sorted(expected.video_id.astype(str).unique()):
            with np.load(ROOT / f"cache/road_parts/{video}.npz", allow_pickle=False) as part:
                for identity, value in zip(part["sequence_id"].astype(str), part["road"]):
                    if identity in indexed:
                        raise RuntimeError(f"duplicate road identity: {identity}")
                    indexed[identity] = value
        if set(indexed) != set(expected.sequence_id.astype(str)):
            raise RuntimeError(f"road cohort mismatch: {split}")
        road = np.stack([indexed[identity] for identity in expected.sequence_id.astype(str)])
        source = np.load(ROOT / f"cache/features/{split}/S187.npy")
        combined = np.column_stack((source, road)).astype(np.float32)
        if combined.shape != (len(expected), 247) or not np.isfinite(combined).all():
            raise RuntimeError(f"malformed R247 matrix: {split}")
        atomic_npy(ROOT / f"cache/features/{split}/R247.npy", combined)
        np.testing.assert_array_equal(source, combined[:, :187])
    write(ROOT / "metrics/road_extraction.json", {
        "status": "complete", "development_only": True, "test_accessed": False,
        "device": torch.cuda.get_device_name(0), "precision": "fp16", "batch_size": 1,
        "batch_policy_source": "Exp075 batch parity was false; preserve selected batch size 1",
        "first_partition_frames": 7, "last_partition_frames": 8,
        "videos": 140, "new_video_reports": reports,
    })


def train() -> None:
    contract = json.loads((ROOT / "feature_contract.json").read_text())
    populations = rows()
    train_rows, val_rows = populations["train"], populations["val"]
    for split, expected in populations.items():
        saved = pd.read_csv(ROOT / f"cache/features/{split}/rows.csv")
        if not saved.sequence_id.astype(str).equals(expected.sequence_id.astype(str)):
            raise RuntimeError(f"prepared row order mismatch: {split}")
    y_train, y_val = train_rows.label.to_numpy(), val_rows.label.to_numpy()
    results = []
    for arm in ("S187", "R247"):
        xt = np.load(ROOT / f"cache/features/train/{arm}.npy")
        xv = np.load(ROOT / f"cache/features/val/{arm}.npy")
        if xt.shape != (26295, len(contract[arm])) or xv.shape != (14922, len(contract[arm])):
            raise RuntimeError(f"feature dimension mismatch: {arm}")
        for seed in SEEDS:
            model = HELPER.bbox_core.build_xgb(seed, y_train, n_estimators=250)
            start = time.perf_counter()
            model.fit(xt, y_train)
            (ROOT / "models").mkdir(exist_ok=True)
            joblib.dump(model, ROOT / f"models/{arm}_seed{seed}.joblib")
            probability = model.predict_proba(xv)[:, 1]
            prediction = (probability >= .5).astype(int)
            output = val_rows.copy()
            output["probability"], output["prediction"] = probability, prediction
            write(ROOT / f"predictions/validation_{arm}_seed{seed}.csv", output)
            scores = HELPER.core.compute_metrics(y_val, probability)
            scores["macro_f1"] = f1_score(y_val, prediction, average="macro")
            results.append({"arm": arm, "seed": seed, "fit_seconds": time.perf_counter() - start, **scores})
            print(f'{arm}{seed}: macro={scores["macro_f1"]:.6f} AP={scores["pr_auc"]:.6f}', flush=True)
    table = pd.DataFrame(results)
    columns = ["accuracy", "precision", "recall", "f1", "macro_f1", "pr_auc", "roc_auc", "fp", "fn"]
    summary = table.groupby("arm")[columns].agg(["mean", "std"])
    summary.columns = ["_".join(value) for value in summary.columns]
    left = table[table.arm == "S187"].set_index("seed")[columns]
    right = table[table.arm == "R247"].set_index("seed")[columns]
    delta = (right - left).reset_index()
    delta["comparison"] = "R247-S187"
    write(ROOT / "metrics/per_seed_results.csv", table)
    write(ROOT / "metrics/summary_results.csv", summary.reset_index())
    write(ROOT / "metrics/paired_deltas.csv", delta)
    outputs = [
        record(path) for folder in ("metrics", "models", "predictions")
        for path in sorted((ROOT / folder).glob("*")) if path.is_file()
    ]
    write(ROOT / "EXPERIMENT_COMPLETE.json", {
        "status": "complete", "amended_complete_input_cohort": True,
        "train_windows": 26295, "validation_windows": 14922,
        "development_only": True, "test_accessed": False, "fits": 6, "outputs": outputs,
    })


def verify() -> None:
    complete = json.loads((ROOT / "EXPERIMENT_COMPLETE.json").read_text())
    assert complete["fits"] == 6 and complete["test_accessed"] is False
    populations = rows()
    contract = json.loads((ROOT / "feature_contract.json").read_text())
    checks = {}
    for split, frame in populations.items():
        source = np.load(ROOT / f"cache/features/{split}/S187.npy")
        road = np.load(ROOT / f"cache/features/{split}/R247.npy")
        assert source.shape == (len(frame), 187) and road.shape == (len(frame), 247)
        np.testing.assert_array_equal(source, road[:, :187])
        assert np.isfinite(source).all() and np.isfinite(road).all()
        checks[f"{split}_matched_source_prefix"] = True
    result = pd.read_csv(ROOT / "metrics/per_seed_results.csv")
    assert len(result) == 6 and set(result.arm) == set(contract) and set(result.seed) == set(SEEDS)
    for arm in contract:
        for seed in SEEDS:
            prediction = pd.read_csv(ROOT / f"predictions/validation_{arm}_seed{seed}.csv")
            assert len(prediction) == 14922
            assert prediction.sequence_id.astype(str).equals(populations["val"].sequence_id.astype(str))
            model = joblib.load(ROOT / f"models/{arm}_seed{seed}.joblib")
            probability = model.predict_proba(np.load(ROOT / f"cache/features/val/{arm}.npy"))[:, 1]
            # CSV decimal parsing is authenticated at the stored float32 model
            # probability precision, matching Exp113's verification boundary.
            np.testing.assert_array_equal(
                prediction.probability.to_numpy(np.float32), probability
            )
    write(ROOT / "VERIFIED.json", {
        "status": "verified", "checks": checks, "train_windows": 26295,
        "validation_windows": 14922, "fits": 6, "test_accessed": False,
    })


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("prepare-source", "extract-road", "train", "verify"):
        raise SystemExit("usage: run.py {prepare-source|extract-road|train|verify}")
    {"prepare-source": prepare_source, "extract-road": extract_road, "train": train, "verify": verify}[sys.argv[1]]()
