"""Exact historical MSLSTM YOLOP feature definition and lazy inference helpers."""

from __future__ import annotations

import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


YOLOP_ROOT = Path("/home/abdelrahman/Hadwa/MSLSTM-PID/YOLOP")
YOLOP_WEIGHTS = YOLOP_ROOT / "weights" / "End-to-end.pth"
YOLOP_REPOSITORY_COMMIT = "8d8f68df318c71f01d6f813c024df646c7d1978f"
YOLOP_WEIGHTS_SHA256 = "620e7495328cfc8e04807c14cbb8f8df9916953359d190cbf494b2d049f03b0c"
YOLOP_NOTEBOOK = Path("/home/abdelrahman/Hadwa/MSLSTM-PID/Offline Extraction/yolop-extraction.ipynb")
YOLOP_NOTEBOOK_SHA256 = "2310285c26a88e9acc5aa019e1d301e9435b58bd8b69e7c2d9292951575f8b2f"
PIE_VIDEO_ROOT = Path("/home/abdelrahman/Hadwa/PIE_Dataset")
PIE_ANNOTATION_ROOT = Path("/home/abdelrahman/Hadwa/MSLSTM-PID/PIE/annotations/annotations")
PIE_YOLOP_CACHE_ROOT = Path("/home/abdelrahman/Hadwa/MSLSTM-PID/Extracted YOLOP")
JAAD_IMAGE_ROOT = Path("/home/abdelrahman/Hadwa/datasets/JAAD/images")
YOLOP_IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
GRID_SIZE = 3
GRID_SCALE = 1.0
RADIUS_METERS = 15.0
PIXELS_PER_METER = 30
FEATURE_DIMENSION = 20
VEHICLE_COCO_CLASSES = frozenset((2, 3, 5, 7))
PEDESTRIAN_COCO_CLASS = 0
HISTORICAL_PARITY_CASES = (
    ("set03", "video_0001", 1464, "3_1_220"),
    ("set03", "video_0001", 1465, "3_1_220"),
    ("set03", "video_0001", 1466, "3_1_220"),
)


def grid_features(
    mask: np.ndarray | None,
    cx: float,
    cy: float,
    pedestrian_height: float,
    *,
    grid_size: int = GRID_SIZE,
    scale: float = GRID_SCALE,
) -> np.ndarray:
    """Reproduce the historical notebook's pedestrian-centred mask grid."""
    if mask is None or pedestrian_height <= 0:
        return np.zeros(grid_size * grid_size, dtype=np.float32)
    cell = max(1, int(pedestrian_height * scale))
    half = (grid_size * cell) // 2
    height, width = mask.shape[:2]
    output = np.zeros((grid_size, grid_size), dtype=np.float32)
    cx, cy = int(cx), int(cy)
    for row in range(grid_size):
        for column in range(grid_size):
            x1 = int(np.clip(cx - half + column * cell, 0, width))
            y1 = int(np.clip(cy - half + row * cell, 0, height))
            x2 = int(np.clip(x1 + cell, 0, width))
            y2 = int(np.clip(y1 + cell, 0, height))
            if x2 > x1 and y2 > y1:
                mean_value = mask[y1:y2, x1:x2].mean()
                output[row, column] = mean_value / 255.0 if mean_value > 0 else 0.0
    return output.reshape(-1)


def nearby_counts(
    detections: np.ndarray | None,
    cx: float,
    cy: float,
    radius_pixels: float,
) -> np.ndarray:
    """Reproduce historical normalized nearby vehicle/pedestrian counts."""
    if detections is None or not isinstance(detections, np.ndarray) or not len(detections):
        return np.zeros(2, dtype=np.float32)
    vehicles = 0
    pedestrians = 0
    for detection in detections:
        x1, y1, x2, y2 = detection[:4]
        class_id = int(detection[5])
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        distance_squared = (float(center_x) - float(cx)) ** 2 + (
            float(center_y) - float(cy)
        ) ** 2
        if distance_squared < float(radius_pixels) ** 2:
            if class_id in VEHICLE_COCO_CLASSES:
                vehicles += 1
            elif class_id == PEDESTRIAN_COCO_CLASS:
                pedestrians += 1
    maximum = 10.0
    return np.asarray(
        [min(vehicles, maximum) / maximum, min(pedestrians, maximum) / maximum],
        dtype=np.float32,
    )


def compose_features(
    drivable_area_mask: np.ndarray | None,
    lane_line_mask: np.ndarray | None,
    detections: np.ndarray | None,
    bbox: tuple[float, float, float, float] | list[float],
) -> np.ndarray:
    """Build the frozen 9 DA + 9 lane + 2 object feature vector."""
    x1, y1, x2, y2 = bbox
    pedestrian_cx = int((x1 + x2) / 2)
    pedestrian_cy = int(y2)
    pedestrian_height = int(y2 - y1)
    result = np.concatenate(
        (
            grid_features(drivable_area_mask, pedestrian_cx, pedestrian_cy, pedestrian_height),
            grid_features(lane_line_mask, pedestrian_cx, pedestrian_cy, pedestrian_height),
            nearby_counts(
                detections,
                pedestrian_cx,
                pedestrian_cy,
                RADIUS_METERS * PIXELS_PER_METER,
            ),
        )
    ).astype(np.float32, copy=False)
    if result.shape != (FEATURE_DIMENSION,):
        raise RuntimeError(f"Unexpected YOLOP feature shape: {result.shape}")
    return result


def historical_letterbox(
    image: np.ndarray,
    new_shape: int | tuple[int, int] = YOLOP_IMAGE_SIZE,
    color: tuple[int, int, int] = (114, 114, 114),
    *,
    auto: bool = False,
    scale_fill: bool = False,
    scale_up: bool = True,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Exact local fallback used by the historical extraction notebook."""
    import cv2

    shape = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    ratio_value = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:
        ratio_value = min(ratio_value, 1.0)
    ratio = (ratio_value, ratio_value)
    new_unpadded = (int(round(shape[1] * ratio_value)), int(round(shape[0] * ratio_value)))
    dw = new_shape[1] - new_unpadded[0]
    dh = new_shape[0] - new_unpadded[1]
    if auto:
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpadded = (new_shape[1], new_shape[0])
        ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpadded:
        image = cv2.resize(image, new_unpadded, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return image, ratio, (dw, dh)


class YolopExtractor:
    """Lazy wrapper around the pinned historical YOLOP checkpoint."""

    def __init__(self, device: str = "cuda:0"):
        import torch

        from contracts import sha256_file

        if sha256_file(YOLOP_WEIGHTS) != YOLOP_WEIGHTS_SHA256:
            raise ValueError(f"YOLOP weight hash mismatch: {YOLOP_WEIGHTS}")
        if sha256_file(YOLOP_NOTEBOOK) != YOLOP_NOTEBOOK_SHA256:
            raise ValueError(f"YOLOP notebook hash mismatch: {YOLOP_NOTEBOOK}")
        if str(YOLOP_ROOT) not in sys.path:
            sys.path.insert(0, str(YOLOP_ROOT))
        from lib.config import cfg
        from lib.models import get_net

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not visible to PyTorch.")
        if self.device.type == "cuda":
            # RTX TF32 produced small deviations from the historical cache.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        model = get_net(cfg)
        checkpoint = torch.load(YOLOP_WEIGHTS, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
        incompatibility = model.load_state_dict(state_dict, strict=True)
        if incompatibility.missing_keys or incompatibility.unexpected_keys:
            raise RuntimeError("Pinned YOLOP checkpoint did not load strictly.")
        self.model = model.to(self.device).eval()
        self.parameter_count = sum(parameter.numel() for parameter in model.parameters())

    def infer_frame(
        self, image: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
        """Return detections and the two historical uint8 sigmoid masks."""
        import torch

        from lib.core.general import non_max_suppression, scale_coords

        image_original = image
        prepared, _, _ = historical_letterbox(image_original, YOLOP_IMAGE_SIZE, auto=False)
        prepared = prepared[:, :, ::-1].transpose(2, 0, 1)
        tensor = torch.from_numpy(np.ascontiguousarray(prepared)).to(self.device)
        tensor = tensor.float().div(255.0).unsqueeze(0)
        with torch.inference_mode():
            detection_output, drivable_output, lane_output = self.model(tensor)
            raw_detection = (
                detection_output[0]
                if isinstance(detection_output, (list, tuple))
                else detection_output
            )
            detections = non_max_suppression(
                raw_detection,
                CONFIDENCE_THRESHOLD,
                IOU_THRESHOLD,
                classes=None,
                agnostic=False,
            )[0]
            detection_array = None
            if detections is not None and len(detections):
                detections = detections.clone()
                scale_coords(tensor.shape[2:], detections[:, :4], image_original.shape)
                detection_array = detections.cpu().numpy()
            target_size = image_original.shape[:2]
            drivable_output = torch.nn.functional.interpolate(
                drivable_output, size=target_size, mode="bilinear", align_corners=False
            )
            lane_output = torch.nn.functional.interpolate(
                lane_output, size=target_size, mode="bilinear", align_corners=False
            )
            drivable_mask = (
                drivable_output[0][0].sigmoid().cpu().numpy() * 255
            ).astype(np.uint8)
            lane_mask = (lane_output[0][0].sigmoid().cpu().numpy() * 255).astype(np.uint8)
        return detection_array, drivable_mask, lane_mask


def _historical_pie_bbox(set_id: str, video_id: str, frame: int, ped_id: str) -> list[float]:
    path = PIE_ANNOTATION_ROOT / set_id / f"{video_id}_annt.xml"
    root = ET.parse(path).getroot()
    for track in root.findall(".//track"):
        if track.get("label", "").lower() != "pedestrian":
            continue
        for box in track.findall("./box"):
            identity = box.find('./attribute[@name="id"]')
            if (
                identity is not None
                and identity.text == ped_id
                and int(box.get("frame", -1)) == frame
                and box.get("outside") != "1"
            ):
                return [float(box.get(key)) for key in ("xtl", "ytl", "xbr", "ybr")]
    raise KeyError(f"Missing historical PIE BBox: {set_id}:{video_id}:{frame}:{ped_id}")


def verify_historical_parity(device: str = "cuda:0") -> list[dict[str, object]]:
    """Compare predeclared PIE cases against the saved historical cache."""
    import cv2
    import pickle

    extractor = YolopExtractor(device)
    reports: list[dict[str, object]] = []
    captures: dict[tuple[str, str], object] = {}
    caches: dict[tuple[str, str], Mapping[int, object]] = {}
    try:
        for set_id, video_id, frame, ped_id in HISTORICAL_PARITY_CASES:
            video_key = (set_id, video_id)
            if video_key not in captures:
                capture = cv2.VideoCapture(str(PIE_VIDEO_ROOT / set_id / f"{video_id}.mp4"))
                if not capture.isOpened():
                    raise FileNotFoundError(f"Cannot open historical PIE video: {video_key}")
                captures[video_key] = capture
                cache_path = PIE_YOLOP_CACHE_ROOT / f"{set_id}_{video_id}_yolop_features.pkl"
                with cache_path.open("rb") as handle:
                    caches[video_key] = pickle.load(handle)[f"{set_id}_{video_id}"]
            capture = captures[video_key]
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot read historical PIE frame: {video_key}:{frame}")
            detections, drivable_mask, lane_mask = extractor.infer_frame(image)
            actual = compose_features(
                drivable_mask,
                lane_mask,
                detections,
                _historical_pie_bbox(set_id, video_id, frame, ped_id),
            )
            expected = np.asarray(caches[video_key][frame][ped_id], dtype=np.float32)
            maximum_error = float(np.max(np.abs(actual - expected)))
            passed = bool(np.allclose(actual, expected, atol=1e-6, rtol=0.0))
            reports.append(
                {
                    "set_id": set_id,
                    "video_id": video_id,
                    "frame": frame,
                    "ped_id": ped_id,
                    "max_abs_error": maximum_error,
                    "passed_atol_1e_6": passed,
                }
            )
            if not passed:
                raise ValueError(f"Historical YOLOP parity failed: {reports[-1]}")
    finally:
        for capture in captures.values():
            capture.release()
    return reports


def _required_bbox_records(manifest, database) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    """Deduplicate exact observation-frame BBoxes without reading target labels."""
    required: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in manifest.itertuples(index=False):
        for frame in row.observation_frames:
            required[str(row.video_id)][int(frame)].add(str(row.ped_id))
    result: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for video_id, frames in required.items():
        output_frames: dict[int, dict[str, np.ndarray]] = {}
        for frame, ped_ids in frames.items():
            output_frames[frame] = {}
            for ped_id in sorted(ped_ids):
                pedestrian = database[video_id]["ped_annotations"][ped_id]
                indices = {int(value): index for index, value in enumerate(pedestrian["frames"])}
                index = indices.get(frame)
                if index is None:
                    raise KeyError(f"Missing JAAD BBox identity: {video_id}:{ped_id}:{frame}")
                bbox = np.asarray(pedestrian["bbox"][index], dtype=np.float32)
                if bbox.shape != (4,) or not np.isfinite(bbox).all():
                    raise ValueError(f"Malformed JAAD BBox: {video_id}:{ped_id}:{frame}")
                output_frames[frame][ped_id] = bbox
        result[video_id] = output_frames
    return result


def build_yolop_cache(
    manifest,
    database,
    output_root: Path,
    *,
    device: str = "cuda:0",
) -> dict[str, object]:
    """Build resumable per-video label-free JAAD YOLOP feature parts."""
    import cv2

    output_root = Path(output_root)
    parts_root = output_root / "yolop_parts"
    parts_root.mkdir(parents=True, exist_ok=True)
    required = _required_bbox_records(manifest, database)
    expected_records = sum(
        len(pedestrians)
        for frames in required.values()
        for pedestrians in frames.values()
    )
    expected_frames = sum(len(frames) for frames in required.values())
    parity = verify_historical_parity(device)
    extractor = YolopExtractor(device)
    completed_frames = 0
    completed_records = 0
    started = time.time()
    for video_index, video_id in enumerate(sorted(required), start=1):
        destination = parts_root / f"{video_id}.npz"
        video_frames = required[video_id]
        video_record_count = sum(len(pedestrians) for pedestrians in video_frames.values())
        if destination.exists():
            with np.load(destination, allow_pickle=False) as existing:
                if len(existing["features"]) != video_record_count:
                    raise ValueError(f"Incomplete existing YOLOP part: {destination}")
                if existing["features"].shape[1:] != (FEATURE_DIMENSION,):
                    raise ValueError(f"Malformed existing YOLOP part: {destination}")
            completed_frames += len(video_frames)
            completed_records += video_record_count
            print(
                f"yolop_progress={video_index}/{len(required)} frames={completed_frames}/{expected_frames} "
                f"records={completed_records}/{expected_records} resumed=1",
                flush=True,
            )
            continue
        keys_video: list[str] = []
        keys_ped: list[str] = []
        keys_frame: list[int] = []
        vectors: list[np.ndarray] = []
        for frame in sorted(video_frames):
            image_path = JAAD_IMAGE_ROOT / video_id / f"{frame:05d}.png"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if image is None:
                raise FileNotFoundError(f"Missing exact JAAD source frame: {image_path}")
            detections, drivable_mask, lane_mask = extractor.infer_frame(image)
            for ped_id, bbox in sorted(video_frames[frame].items()):
                vector = compose_features(drivable_mask, lane_mask, detections, bbox)
                if not np.isfinite(vector).all() or np.any(vector < 0) or np.any(vector > 1):
                    raise ValueError(f"Invalid YOLOP vector: {video_id}:{ped_id}:{frame}")
                keys_video.append(video_id)
                keys_ped.append(ped_id)
                keys_frame.append(frame)
                vectors.append(vector)
        temporary = destination.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                video_id=np.asarray(keys_video),
                ped_id=np.asarray(keys_ped),
                frame=np.asarray(keys_frame, dtype=np.int64),
                features=np.stack(vectors).astype(np.float32),
            )
        os.replace(temporary, destination)
        completed_frames += len(video_frames)
        completed_records += len(vectors)
        print(
            f"yolop_progress={video_index}/{len(required)} frames={completed_frames}/{expected_frames} "
            f"records={completed_records}/{expected_records} resumed=0",
            flush=True,
        )
    report = {
        "phase": "label_sealed_yolop_extraction",
        "target_labels_attached": False,
        "probabilities_computed": False,
        "videos": len(required),
        "unique_frames": expected_frames,
        "pedestrian_frame_records": expected_records,
        "feature_dimension": FEATURE_DIMENSION,
        "device": str(device),
        "tf32_enabled": False,
        "deterministic_cudnn": True,
        "model_parameters": extractor.parameter_count,
        "weights_path": str(YOLOP_WEIGHTS),
        "weights_sha256": YOLOP_WEIGHTS_SHA256,
        "notebook_path": str(YOLOP_NOTEBOOK),
        "notebook_sha256": YOLOP_NOTEBOOK_SHA256,
        "repository_commit": YOLOP_REPOSITORY_COMMIT,
        "historical_parity": parity,
        "elapsed_seconds": time.time() - started,
    }
    report_path = output_root / "yolop_extraction_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


class YolopStore:
    """Fail-closed exact-key reader for completed per-video JAAD parts."""

    def __init__(self, parts_root: Path):
        self.parts_root = Path(parts_root)
        if not self.parts_root.is_dir():
            raise FileNotFoundError(f"Missing JAAD YOLOP parts directory: {self.parts_root}")
        self._videos: dict[str, dict[tuple[str, int], np.ndarray]] = {}

    def _load_video(self, video_id: str) -> dict[tuple[str, int], np.ndarray]:
        video_id = str(video_id)
        if video_id in self._videos:
            return self._videos[video_id]
        path = self.parts_root / f"{video_id}.npz"
        if not path.is_file():
            raise KeyError(f"Missing exact JAAD YOLOP video part: {video_id}")
        with np.load(path, allow_pickle=False) as payload:
            videos = payload["video_id"].astype(str)
            pedestrians = payload["ped_id"].astype(str)
            frames = payload["frame"].astype(np.int64)
            features = payload["features"].astype(np.float32)
        count = len(features)
        if not (len(videos) == len(pedestrians) == len(frames) == count):
            raise ValueError(f"Misaligned JAAD YOLOP arrays: {path}")
        if features.shape != (count, FEATURE_DIMENSION) or not np.isfinite(features).all():
            raise ValueError(f"Malformed JAAD YOLOP features: {path}")
        if np.any(features < 0) or np.any(features > 1) or np.any(videos != video_id):
            raise ValueError(f"Invalid JAAD YOLOP content: {path}")
        records: dict[tuple[str, int], np.ndarray] = {}
        for index in range(count):
            key = (str(pedestrians[index]), int(frames[index]))
            if key in records:
                raise ValueError(f"Duplicate JAAD YOLOP identity in {path}: {key}")
            records[key] = features[index]
        self._videos[video_id] = records
        return records

    def get_window(
        self, video_id: str, ped_id: str, frames: Sequence[int]
    ) -> np.ndarray:
        records = self._load_video(str(video_id))
        identities = [(str(ped_id), int(frame)) for frame in frames]
        missing = [identity for identity in identities if identity not in records]
        if missing:
            raise KeyError(
                f"Missing exact JAAD YOLOP pedestrian/frame identities for {video_id}: {missing[:3]}"
            )
        return np.stack([records[identity] for identity in identities]).astype(
            np.float32, copy=False
        )
