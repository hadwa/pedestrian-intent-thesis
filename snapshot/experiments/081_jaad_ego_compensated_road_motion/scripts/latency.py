"""Component-scoped 16-frame latency measurements for Exp081."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping
from xml.etree import ElementTree

import cv2
import joblib
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


def _load_sibling(filename: str, name: str):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Exp081 module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


COMPONENTS = (
    "twinlite_masks",
    "sparse_motion",
    "raw_road",
    "compensated_road",
    "cached_xgboost",
    "incremental_pipeline",
)


def required_component_schema() -> tuple[str, ...]:
    return COMPONENTS


def _statistics(samples_ms: list[float]) -> dict[str, float]:
    values = np.asarray(samples_ms, dtype=float)
    if values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all():
        raise ValueError("Exp081 latency samples must be finite and nonempty")
    return {
        "mean_ms": float(values.mean()),
        "std_ms": float(values.std(ddof=0)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def benchmark_callable(
    component: str,
    function: Callable[[], Any],
    *,
    warmups: int,
    repetitions: int,
    scope: str,
) -> dict[str, Any]:
    if component not in COMPONENTS or warmups < 0 or repetitions <= 0:
        raise ValueError("Exp081 latency benchmark settings are invalid")
    for _ in range(warmups):
        function()
    samples: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return {
        "component": component,
        "warmups": warmups,
        "repetitions": repetitions,
        "scope": scope,
        **_statistics(samples),
    }


def benchmark_training_window(root: Path | str, extractor: Any) -> pd.DataFrame:
    """Benchmark one lexicographically fixed JAAD training window."""
    root = Path(root)
    contract = _load_sibling("contract.py", "exp081_contract_for_latency")
    data = _load_sibling("jaad_data.py", "exp081_data_for_latency")
    extraction = _load_sibling("extract.py", "exp081_extract_for_latency")
    motion = _load_sibling("motion.py", "exp081_motion_for_latency")
    road = _load_sibling("road_context.py", "exp081_road_for_latency")
    model = _load_sibling("model.py", "exp081_model_for_latency")
    registration = contract.load_registration()
    anchor = data.load_anchor("train").sort_values("sequence_id", kind="stable").reset_index(drop=True)
    identity = extraction.load_identity_rows("train").sort_values("sequence_id", kind="stable").reset_index(drop=True)
    if str(anchor.loc[0, "sequence_id"]) != str(identity.loc[0, "sequence_id"]):
        raise RuntimeError("Exp081 latency identity differs from anchor")
    row = identity.iloc[0]
    frame_ids = tuple(range(int(row.start_frame), int(row.end_frame) + 1))
    frames_by_id = data.decode_frames(str(row.video_id), frame_ids)
    frames = [frames_by_id[index] for index in frame_ids]
    all_boxes_by_id = data.load_all_pedestrian_boxes(str(row.video_id), frame_ids)
    all_boxes = [all_boxes_by_id[index] for index in frame_ids]
    target_boxes = np.asarray(json.loads(str(row.bbox_sequence_json)), dtype=np.float64)
    road_masks, lane_masks = extractor.predict(frames)
    movement = motion.estimate_window_motion(
        frames, all_boxes, registration["motion_estimator"]
    )
    raw_sequence = road.uncompensated_sequence(road_masks, lane_masks, target_boxes)
    compensated_sequence = road.compensated_sequence(
        road_masks, lane_masks, target_boxes, movement.to_first
    )
    context = extraction.load_context(
        root, "train", anchor["sequence_id"].astype(str).to_numpy()
    )
    variants = model.assemble_variants(anchor, context)
    d_frame = variants["D"].iloc[[0]]
    d_columns = model.feature_columns(d_frame)
    estimator = joblib.load(root / "models/variant_D_seed42.joblib")

    def twinlite_call():
        return extractor.predict(frames)

    def motion_call():
        return motion.estimate_window_motion(
            frames, all_boxes, registration["motion_estimator"]
        )

    def raw_road_call():
        return road.summarize(
            road.uncompensated_sequence(road_masks, lane_masks, target_boxes)
        )

    def compensated_road_call():
        return road.summarize(
            road.compensated_sequence(
                road_masks, lane_masks, target_boxes, movement.to_first
            )
        )

    def xgboost_call():
        return estimator.predict_proba(d_frame.loc[:, d_columns])

    anchor_values = anchor.loc[[0], model.anchor_columns()].reset_index(drop=True)

    def incremental_call():
        current_road, current_lane = extractor.predict(frames)
        current_motion = motion.estimate_window_motion(
            frames, all_boxes, registration["motion_estimator"]
        )
        current_summary = road.summarize(
            road.compensated_sequence(
                current_road, current_lane, target_boxes, current_motion.to_first
            )
        )
        values = pd.concat(
            (
                anchor_values,
                pd.DataFrame([current_summary], columns=model.compensated_road_columns()),
            ),
            axis=1,
        )
        return estimator.predict_proba(values.loc[:, d_columns])

    calls: Mapping[str, tuple[Callable[[], Any], str]] = {
        "twinlite_masks": (twinlite_call, "16 predecoded JAAD frames; CUDA TwinLiteNet+ Large"),
        "sparse_motion": (motion_call, "15 transitions; all-pedestrian masks; CPU OpenCV"),
        "raw_road": (raw_road_call, "cached TwinLite masks to uncompensated 60-D summary"),
        "compensated_road": (compensated_road_call, "cached masks and motion to compensated 60-D summary"),
        "cached_xgboost": (xgboost_call, "cached 242-D Variant D row; one predictor"),
        "incremental_pipeline": (
            incremental_call,
            "TwinLite + motion + compensated road + XGBoost; frame decode and ViTPose/anchor extraction excluded",
        ),
    }
    rows = []
    for component in COMPONENTS:
        settings = registration["latency"][component]
        function, scope = calls[component]
        rows.append(
            benchmark_callable(
                component,
                function,
                warmups=int(settings["warmups"]),
                repetitions=int(settings["repetitions"]),
                scope=scope,
            )
        )
    return pd.DataFrame(rows)


def _pie_boxes(
    annotation_path: Path, frame_ids: tuple[int, ...]
) -> dict[int, np.ndarray]:
    requested = set(frame_ids)
    values: dict[int, list[list[float]]] = {frame_id: [] for frame_id in frame_ids}
    for track in ElementTree.parse(annotation_path).getroot().findall("track"):
        if track.get("label") != "pedestrian":
            continue
        for box in track.findall("box"):
            frame_id = int(box.get("frame", "-1"))
            if frame_id not in requested or box.get("outside", "0") == "1":
                continue
            coordinates = [
                float(box.get(name, "nan"))
                for name in ("xtl", "ytl", "xbr", "ybr")
            ]
            if not np.isfinite(coordinates).all():
                raise RuntimeError(f"Invalid PIE pedestrian box in {annotation_path}")
            values[frame_id].append(coordinates)
    return {
        frame_id: np.asarray(boxes, dtype=np.float64).reshape((-1, 4))
        for frame_id, boxes in values.items()
    }


def _decode_pie_window(video_path: Path, frame_ids: tuple[int, ...]) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)
    images = []
    try:
        for one_based_frame in frame_ids:
            zero_based_frame = one_based_frame - 1
            if zero_based_frame < 0 or not capture.set(
                cv2.CAP_PROP_POS_FRAMES, float(zero_based_frame)
            ):
                raise RuntimeError(f"Cannot seek PIE frame {video_path}:{one_based_frame}")
            position = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
            if position != zero_based_frame:
                raise RuntimeError(
                    f"PIE frame seek drift: {position} != {zero_based_frame}"
                )
            ok, image = capture.read()
            if not ok or image is None:
                raise RuntimeError(f"Cannot decode PIE frame {video_path}:{one_based_frame}")
            images.append(image)
    finally:
        capture.release()
    return images


def pie_visual_proxy_telemetry_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe proxy/OBD associations on frozen PIE development windows only."""
    contract = _load_sibling("contract.py", "exp081_contract_for_pie_audit")
    motion = _load_sibling("motion.py", "exp081_motion_for_pie_audit")
    model = _load_sibling("model.py", "exp081_model_for_pie_audit")
    registration = contract.load_registration()
    settings = registration["pie_proxy_audit"]
    columns = [
        "sequence_id",
        "set_id",
        "video_id",
        "start_frame",
        "end_frame",
        "obd_speed_mean",
        "obd_speed_std",
        "obd_speed_delta",
    ]
    samples = []
    for split, source_name in (
        ("train", "exp076_prepared_train"),
        ("validation", "exp076_prepared_validation"),
    ):
        frame = pd.read_csv(contract.resolve_source(source_name), usecols=columns)
        frame = frame.sort_values("sequence_id", kind="stable").head(
            int(settings["windows_per_split"])
        )
        frame.insert(0, "source_split", split)
        samples.append(frame)
    selected = pd.concat(samples, ignore_index=True)
    allowed_sets = set(settings["allowed_sets"])
    if not set(selected["set_id"].astype(str)).issubset(allowed_sets):
        raise RuntimeError("Exp081 PIE proxy audit selected a forbidden cohort")
    if selected["set_id"].astype(str).eq("set03").any():
        raise RuntimeError("Exp081 PIE proxy audit forbids set03")

    video_root = Path("/home/abdelrahman/Hadwa/PIE_Dataset")
    annotation_root = Path(
        "/home/abdelrahman/Hadwa/MSLSTM-PID/PIE/annotations/annotations"
    )
    proxy_rows = []
    for row in selected.itertuples(index=False):
        frame_ids = tuple(range(int(row.start_frame), int(row.end_frame) + 1))
        if len(frame_ids) != 16:
            raise RuntimeError("Exp081 PIE proxy audit window changed")
        video_path = video_root / str(row.set_id) / f"{row.video_id}.mp4"
        annotation_path = annotation_root / str(row.set_id) / f"{row.video_id}_annt.xml"
        frames = _decode_pie_window(video_path, frame_ids)
        boxes_by_frame = _pie_boxes(annotation_path, frame_ids)
        movement = motion.estimate_window_motion(
            frames,
            [boxes_by_frame[frame_id] for frame_id in frame_ids],
            registration["motion_estimator"],
        )
        proxy_rows.append(
            {
                "source_split": str(row.source_split),
                "sequence_id": str(row.sequence_id),
                "set_id": str(row.set_id),
                **{
                    name: float(value)
                    for name, value in zip(model.proxy_columns(), movement.proxy)
                },
                "obd_speed_mean": float(row.obd_speed_mean),
                "obd_speed_std": float(row.obd_speed_std),
                "obd_speed_delta": float(row.obd_speed_delta),
            }
        )
    sample_frame = pd.DataFrame(proxy_rows)
    association_rows = []
    for proxy_name in model.proxy_columns():
        for telemetry_name in (
            "obd_speed_mean",
            "obd_speed_std",
            "obd_speed_delta",
        ):
            correlation = sample_frame[[proxy_name, telemetry_name]].corr(
                method="spearman"
            ).iloc[0, 1]
            association_rows.append(
                {
                    "proxy_feature": proxy_name,
                    "telemetry_feature": telemetry_name,
                    "spearman_correlation": float(correlation)
                    if np.isfinite(correlation)
                    else np.nan,
                    "windows": len(sample_frame),
                    "sets": ",".join(sorted(sample_frame["set_id"].unique())),
                    "contains_set03": False,
                    "fitted_mapping": False,
                }
            )
    return pd.DataFrame(association_rows), sample_frame
