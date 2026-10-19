"""Frozen no-refit S187/R247 PSI test evaluation."""

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
EXP114 = ROOT.parent
REPO = EXP114.parents[1]
EXP113_TEST = REPO / "experiments/113_psi_native_biu_xgb/test_evaluation"
POSE = REPO / "data/psi_vitpose_s_all_tracks.jsonl"
POSE_SHA256 = "31d8f45a7f8b61803b2dde8f981ce012086ac9905455d7aaac743bc1ec2a3fc7"
SEEDS = (42, 43, 44)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


HELPER = module("psi114_test_helper", REPO / "experiments/098_jaad_matched_biu_xgb/scripts/run_experiment.py")
V2 = module("psi114_test_v2", REPO / "src/biu_v2_extractor.py")
write, record = HELPER.write, HELPER.record


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_rows():
    frame = pd.read_csv(EXP113_TEST / "cache/rows.csv")
    assert len(frame) == 20547 and frame.sequence_id.is_unique and int(frame.label.sum()) == 15462
    return frame


def observed(row):
    frames = list(map(int, json.loads(row.observed_frames_json)))
    assert len(frames) == 15 and frames == list(range(frames[0], frames[0] + 15))
    assert int(row.target_frame) == frames[-1] + 1 and int(row.target_frame) not in frames
    return frames


def load_pose(required):
    assert sha256(POSE) == POSE_SHA256
    output = {}
    with POSE.open() as handle:
        for line in handle:
            item = json.loads(line)
            key = (str(item["video"]), str(item["pid"]), int(item["frame_idx"]))
            if key in required:
                assert key not in output
                output[key] = (np.asarray(item["keypoints"], float), np.asarray(item["kp_conf"], float))
    assert set(output) == required
    return output


def geometry(video):
    path = EXP113_TEST / f"dataset/PSI2.0_TrainVal/annotations/cognitive_annotation_extended/{video}/pedestrian_intent.json"
    source = json.loads(path.read_text())
    assert source["video_name"] == video
    result = {}
    for pid, item in source["pedestrians"].items():
        frames = list(map(int, item["observed_frames"]))
        boxes = np.asarray(item["cv_annotations"]["bboxes"], float)
        assert boxes.shape == (len(frames), 4)
        result[str(pid)] = {frame: box for frame, box in zip(frames, boxes)}
    return result


BASES = (
    "lower_body_motion_magnitude_mean", "lead_ankle_forward_delta",
    "torso_motion_magnitude_mean", "torso_center_x_delta", "bbox_center_dx_norm",
    "bbox_scale_log_delta", "pose_confidence_mean",
)


def half(coords, confidence, boxes):
    extracted = V2.extract_biu_v2(coords, confidence, bbox_sequence=boxes, image_size=(1280, 720))
    widths = boxes[:, 2] - boxes[:, 0]
    centers = (boxes[:, 0] + boxes[:, 2]) / 2
    areas = widths * (boxes[:, 3] - boxes[:, 1])
    result = {
        "lower_body_motion_magnitude_mean": float(extracted["lower_body_motion_magnitude_mean"]),
        "lead_ankle_forward_delta": float(extracted["lead_ankle_forward_delta"]),
        "torso_motion_magnitude_mean": float(extracted["torso_motion_magnitude_mean"]),
        "torso_center_x_delta": float(extracted["torso_center_x_delta"]),
        "bbox_center_dx_norm": float((centers[-1] - centers[0]) / widths.mean()),
        "bbox_scale_log_delta": float(math.log(max(float(areas[-1]), 1e-12) / max(float(areas[0]), 1e-12))),
        "pose_confidence_mean": float(confidence.mean()),
    }
    assert np.isfinite(list(result.values())).all()
    return result


def temporal(coords, confidence, boxes):
    early, late = half(coords[:7], confidence[:7], boxes[:7]), half(coords[7:], confidence[7:], boxes[7:])
    return [value for name in BASES for value in (early[name], late[name], late[name] - early[name])]


def ego(speed, frames):
    assert not set(frames) - set(speed)
    values = np.asarray([speed[frame] for frame in frames], float)
    assert values.shape == (15,) and np.isfinite(values).all()
    return [values.mean(), values.std(ddof=0), values[0], values[-1], values[-1] - values[0]]


def prepare_source():
    speed_gate = json.loads((ROOT / "metrics/speed_recovery.json").read_text())
    assert speed_gate["all_required_frames_have_numeric_speed"] and speed_gate["windows"] == 20547
    rows = test_rows()
    required = {(str(row.video_id), str(row.ped_id), frame) for row in rows.itertuples(index=False) for frame in observed(row)}
    poses = load_pose(required)
    old_contract = json.loads((REPO / "experiments/113_psi_native_biu_xgb/feature_contract.json").read_text())
    contract = json.loads((EXP114 / "feature_contract.json").read_text())
    static_names = contract["S187"][:161]
    indices = {arm: {name: index for index, name in enumerate(old_contract[arm])} for arm in ("A", "B", "C")}
    mapping = []
    for name in static_names:
        if name in indices["A"]:
            mapping.append(("A", indices["A"][name]))
        elif f"v1__{name}" in indices["B"]:
            mapping.append(("B", indices["B"][f"v1__{name}"]))
        else:
            mapping.append(("C", indices["C"][name]))
    arrays = {arm: np.load(EXP113_TEST / f"cache/{arm}.npy") for arm in ("A", "B", "C")}
    output = np.empty((len(rows), 187), np.float32)
    geometries, speeds = {}, {}
    for position, row in enumerate(rows.itertuples(index=False)):
        video, pid, frames = str(row.video_id), str(row.ped_id), observed(row)
        if video not in geometries:
            geometries[video] = geometry(video)
            compact = json.loads((ROOT / f"cache/speed/{video}.json").read_text())
            assert compact["video"] == video and compact["unit"] == "km/hr"
            speeds[video] = {int(key): float(value) for key, value in compact["speed_km_hr"].items()}
        boxes = np.stack([geometries[video][pid][frame] for frame in frames])
        coords = np.stack([poses[(video, pid, frame)][0] for frame in frames])
        confidence = np.stack([poses[(video, pid, frame)][1] for frame in frames])
        static = [arrays[arm][position, index] for arm, index in mapping]
        output[position] = np.asarray(static + temporal(coords, confidence, boxes) + ego(speeds[video], frames), np.float32)
        if position % 2000 == 0:
            print(f"source {position}/{len(rows)}", flush=True)
    assert output.shape == (20547, 187) and np.isfinite(output).all()
    (ROOT / "cache/features").mkdir(parents=True, exist_ok=True)
    np.save(ROOT / "cache/features/S187.npy", output)
    write(ROOT / "cache/features/rows.csv", rows)
    write(ROOT / "metrics/source_features.json", {"status": "complete", "windows": 20547,
          "features": 187, "first_frames": 7, "last_frames": 8, "target_used": False,
          "pose_complete_windows": int(rows.pose_complete.sum()), "test_authorized": True})


def road_modules():
    scripts = REPO / "experiments/075_pie_twinlitenetplus_road_context/scripts"
    sys.path.insert(0, str(scripts))
    module("contract", scripts / "contract.py")
    adapter = module("psi114_test_twin", scripts / "twinlite_adapter.py")
    context = module("psi114_test_context", scripts / "context.py")
    sys.path.remove(str(scripts))
    return adapter, context


def decode(path, required):
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    try:
        for index in range(max(required) + 1):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"decode failure: {path}:{index}")
            if index in required:
                yield index, image
    finally:
        capture.release()


def extract_road():
    import torch
    recovery = json.loads((ROOT / "metrics/video_recovery.json").read_text())
    assert recovery["status"] == "complete" and recovery["videos"] == 49
    assert torch.cuda.is_available()
    adapter, context = road_modules()
    model = adapter.TwinLiteExtractor(device="cuda", half=True)
    assert sha256(REPO / "experiments/075_pie_twinlitenetplus_road_context/checkpoints/large.pth") == "5605d1a8ce762c1b6f7d23f4c2e7ec9384bfd6e6887b040b43e1f8902a817e22"
    rows = test_rows()
    reports = []
    for number, (video, frame) in enumerate(rows.groupby("video_id", sort=True), start=1):
        video = str(video)
        part = ROOT / f"cache/road_parts/{video}.npz"
        if part.is_file():
            print(f"road {number}/49 {video}: reused", flush=True)
            continue
        geo = geometry(video)
        needed = {(str(row.ped_id), index): geo[str(row.ped_id)][index]
                  for row in frame.itertuples(index=False) for index in observed(row)}
        by_frame = {}
        for key in needed:
            by_frame.setdefault(key[1], []).append(key)
        static, road_zero, lane_zero, fractions = {}, 0, 0, [[], []]
        start = time.perf_counter()
        for index, image in decode(ROOT / f"cache/videos/{video}.mp4", set(by_frame)):
            road_masks, lane_masks = model.predict([image])
            road_mask, lane_mask = road_masks[0], lane_masks[0]
            road_zero += int(not road_mask.any()); lane_zero += int(not lane_mask.any())
            fractions[0].append(float(road_mask.mean())); fractions[1].append(float(lane_mask.mean()))
            for pid, frame_index in by_frame[index]:
                static[(pid, index)] = context.road_relative_static(road_mask, lane_mask, needed[(pid, frame_index)])
        ids, values = [], []
        for row in frame.itertuples(index=False):
            pid, frames = str(row.ped_id), observed(row)
            boxes = np.stack([geo[pid][index] for index in frames])
            sequence = context._REPRESENTATIONS.assemble_road_relative_temporal(
                np.stack([static[(pid, index)] for index in frames]), boxes)
            early, late = sequence[:7].mean(0), sequence[7:].mean(0)
            values.append(np.column_stack((early, late, late - early)).reshape(-1)); ids.append(str(row.sequence_id))
        values = np.asarray(values, np.float32)
        assert values.shape == (len(frame), 60) and np.isfinite(values).all()
        part.parent.mkdir(parents=True, exist_ok=True)
        temporary = part.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, sequence_id=np.asarray(ids), road=values)
        os.replace(temporary, part)
        reports.append({"video": video, "windows": len(frame), "unique_frames": len(by_frame),
                        "seconds": time.perf_counter() - start, "road_zero": road_zero,
                        "lane_zero": lane_zero, "road_foreground_mean": float(np.mean(fractions[0])),
                        "lane_foreground_mean": float(np.mean(fractions[1]))})
        print(f"road {number}/49 {video}: {len(by_frame)} frames", flush=True)
    indexed = {}
    for video in sorted(rows.video_id.astype(str).unique()):
        with np.load(ROOT / f"cache/road_parts/{video}.npz", allow_pickle=False) as part:
            indexed.update(zip(part["sequence_id"].astype(str), part["road"]))
    assert set(indexed) == set(rows.sequence_id.astype(str))
    source = np.load(ROOT / "cache/features/S187.npy")
    road = np.stack([indexed[key] for key in rows.sequence_id.astype(str)]).astype(np.float32)
    combined = np.column_stack((source, road)).astype(np.float32)
    assert combined.shape == (20547, 247) and np.isfinite(combined).all()
    np.testing.assert_array_equal(source, combined[:, :187])
    np.save(ROOT / "cache/features/R247.npy", combined)
    write(ROOT / "metrics/road_extraction.json", {"status": "complete", "videos": 49,
          "unique_frames": 13144, "batch_size": 1, "precision": "fp16",
          "device": torch.cuda.get_device_name(0), "reports": reports, "test_authorized": True})


def score(y, probability):
    result = HELPER.core.compute_metrics(y, probability)
    result["macro_f1"] = f1_score(y, probability >= .5, average="macro")
    return result


def evaluate():
    frozen = json.loads((ROOT / "freeze.json").read_text())
    for item in frozen["models"] + frozen["sources"]:
        assert record(item["path"]) == item
    rows = test_rows(); y = rows.label.to_numpy()
    source = np.load(ROOT / "cache/features/S187.npy"); road = np.load(ROOT / "cache/features/R247.npy")
    np.testing.assert_array_equal(source, road[:, :187])
    results, probabilities, transitions = [], {}, []
    for arm, matrix in (("S187", source), ("R247", road)):
        for seed in SEEDS:
            model = joblib.load(EXP114 / f"models/{arm}_seed{seed}.joblib")
            probability = model.predict_proba(matrix)[:, 1]
            probabilities[(arm, seed)] = probability
            output = rows.copy(); output["probability"] = probability; output["prediction"] = (probability >= .5).astype(int)
            write(ROOT / f"predictions/test_{arm}_seed{seed}.csv", output)
            results.append({"arm": arm, "seed": seed, **score(y, probability)})
    table = pd.DataFrame(results)
    columns = ["accuracy", "precision", "recall", "f1", "macro_f1", "pr_auc", "roc_auc", "fp", "fn", "fp_ge_0p9_count", "fn_le_0p1_count"]
    summary = table.groupby("arm")[columns].agg(["mean", "std"]); summary.columns = ["_".join(value) for value in summary.columns]
    delta = table[table.arm == "R247"].set_index("seed")[columns] - table[table.arm == "S187"].set_index("seed")[columns]
    delta["comparison"] = "R247-S187"
    for seed in SEEDS:
        a, b = probabilities[("S187", seed)] >= .5, probabilities[("R247", seed)] >= .5
        transitions.append({"seed": seed, "wrong_to_correct": int(((a != y) & (b == y)).sum()),
                            "correct_to_wrong": int(((a == y) & (b != y)).sum()),
                            "fp_fixed": int(((y == 0) & a & ~b).sum()), "fn_fixed": int(((y == 1) & ~a & b).sum()),
                            "fp_added": int(((y == 0) & ~a & b).sum()), "fn_added": int(((y == 1) & a & ~b).sum())})
    write(ROOT / "metrics/per_seed_results.csv", table); write(ROOT / "metrics/summary_results.csv", summary.reset_index())
    write(ROOT / "metrics/paired_deltas.csv", delta.reset_index()); write(ROOT / "metrics/error_transitions.csv", pd.DataFrame(transitions))
    write(ROOT / "metrics/cohort.json", {"windows": len(rows), "videos": int(rows.video_id.nunique()),
          "tracks": int(rows.track_id.nunique()), "positives": int(y.sum()), "negatives": int((1-y).sum()),
          "pose_complete": int(rows.pose_complete.sum()), "annotation_vote_ties": int(rows.annotation_vote.eq(.5).sum())})
    write(ROOT / "metrics/always_cross.json", score(y, np.ones(len(y))))
    write(ROOT / "EXPERIMENT_COMPLETE.json", {"status": "complete", "no_refit": True, "test_authorized": True,
          "models_evaluated": 6, "windows": len(rows), "outputs": [record(path) for folder in ("metrics", "predictions") for path in sorted((ROOT / folder).glob("*")) if path.is_file()]})


def verify():
    complete = json.loads((ROOT / "EXPERIMENT_COMPLETE.json").read_text())
    assert complete["no_refit"] and complete["models_evaluated"] == 6
    rows = test_rows(); y = rows.label.to_numpy(); table = pd.read_csv(ROOT / "metrics/per_seed_results.csv").set_index(["arm", "seed"])
    source = np.load(ROOT / "cache/features/S187.npy"); road = np.load(ROOT / "cache/features/R247.npy")
    np.testing.assert_array_equal(source, road[:, :187])
    for arm, matrix in (("S187", source), ("R247", road)):
        for seed in SEEDS:
            model = joblib.load(EXP114 / f"models/{arm}_seed{seed}.joblib"); probability = model.predict_proba(matrix)[:, 1]
            saved = pd.read_csv(ROOT / f"predictions/test_{arm}_seed{seed}.csv")
            assert saved.sequence_id.astype(str).equals(rows.sequence_id.astype(str))
            np.testing.assert_array_equal(saved.probability.to_numpy(np.float32), probability)
            metrics = score(y, probability)
            for name in ("accuracy", "precision", "recall", "f1", "macro_f1", "pr_auc", "roc_auc", "fp", "fn"):
                np.testing.assert_allclose(table.loc[(arm, seed), name], metrics[name], rtol=0, atol=1e-12)
    write(ROOT / "VERIFIED.json", {"status": "verified", "saved_probabilities": 6,
          "metrics_reproduced": True, "matched_source_prefix": True, "no_refit": True, "windows": 20547})


if __name__ == "__main__":
    action = sys.argv[1]
    {"prepare-source": prepare_source, "extract-road": extract_road, "evaluate": evaluate, "verify": verify}[action]()
