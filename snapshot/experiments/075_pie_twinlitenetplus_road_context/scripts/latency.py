"""Component and online incremental latency protocols for Exp075."""

from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np

import context


INCLUDED_STAGES = [
    "twinlite_masks",
    "road_relative_20d",
    "road_summary_60d",
    "xgboost_predict",
]
EXCLUDED_STAGES = ["frame_video_decoding", "vitpose_keypoint_extraction"]


def _statistics(elapsed_ms: Sequence[float]) -> dict[str, float]:
    values = np.asarray(elapsed_ms, dtype=float)
    return {
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
    }


def _run_online_once(
    extractor,
    images: Sequence[np.ndarray],
    bboxes: np.ndarray,
    base_features: np.ndarray,
    predictor,
    batch_size: int,
) -> float:
    drivable: list[np.ndarray] = []
    lane: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        road_masks, lane_masks = extractor.predict(images[start : start + batch_size])
        drivable.extend(road_masks)
        lane.extend(lane_masks)
    road = context.road_relative_sequence(
        np.stack(drivable), np.stack(lane), np.asarray(bboxes, dtype=np.float32)
    )
    summary = context.summarize_road(road)
    row = np.concatenate((np.asarray(base_features, dtype=np.float32), summary))[None]
    if row.shape != (1, 247):
        raise RuntimeError("online Exp075 row must contain exactly 247 features")
    probability = float(predictor.predict_proba(row)[0, 1])
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise RuntimeError("online predictor returned an invalid probability")
    return probability


def benchmark_online_incremental(
    extractor,
    images: Sequence[np.ndarray],
    bboxes: np.ndarray,
    base_features: np.ndarray,
    predictor,
    *,
    batch_sizes: Sequence[int],
    warmups: int,
    repetitions: int,
    synchronize: Callable[[], None] | None = None,
) -> list[dict[str, object]]:
    """Time already-decoded RGB plus existing Exp072 cues through prediction."""

    if len(images) != 16 or np.asarray(bboxes).shape != (16, 4):
        raise ValueError("online latency requires one exact 16-frame observation")
    if np.asarray(base_features).shape != (187,):
        raise ValueError("online latency requires exactly 187 existing Exp072 cues")
    if warmups < 0 or repetitions <= 0:
        raise ValueError("latency warmups/repetitions are invalid")
    sync = synchronize or (lambda: None)
    output = []
    for batch_size in map(int, batch_sizes):
        if batch_size <= 0 or 16 % batch_size:
            raise ValueError("online batch size must divide 16")
        probability = float("nan")
        for _ in range(int(warmups)):
            probability = _run_online_once(
                extractor, images, bboxes, base_features, predictor, batch_size
            )
        elapsed = []
        for _ in range(int(repetitions)):
            sync()
            started = time.perf_counter_ns()
            probability = _run_online_once(
                extractor, images, bboxes, base_features, predictor, batch_size
            )
            sync()
            elapsed.append((time.perf_counter_ns() - started) / 1_000_000.0)
        output.append(
            {
                "component": "online_incremental_window",
                "batch_size": batch_size,
                "observation_frames": 16,
                "warmups": int(warmups),
                "repetitions": int(repetitions),
                "included_stages": INCLUDED_STAGES.copy(),
                "excluded_stages": EXCLUDED_STAGES.copy(),
                "last_probability": probability,
                **_statistics(elapsed),
            }
        )
        output[-1]["per_16_frame_window_mean_ms"] = output[-1]["mean_ms"]
    return output


def benchmark_cached_predictor(
    predictor,
    rows: np.ndarray,
    *,
    batch_size: int,
    warmups: int,
    repetitions: int,
) -> dict[str, object]:
    values = np.asarray(rows, dtype=np.float32)
    if values.shape != (batch_size, 247):
        raise ValueError("cached predictor rows must have shape batch_size x 247")
    for _ in range(int(warmups)):
        predictor.predict_proba(values)
    elapsed = []
    for _ in range(int(repetitions)):
        started = time.perf_counter_ns()
        predictor.predict_proba(values)
        elapsed.append((time.perf_counter_ns() - started) / 1_000_000.0)
    statistics = _statistics(elapsed)
    output = {
        "component": "cached_247d_predictor",
        "batch_size": int(batch_size),
        "observation_frames": 16,
        "warmups": int(warmups),
        "repetitions": int(repetitions),
        **statistics,
        "per_sample_mean_ms": statistics["mean_ms"] / batch_size,
    }
    output["per_16_frame_window_mean_ms"] = output["per_sample_mean_ms"]
    return output


def benchmark_twinlite(
    extractor,
    images: Sequence[np.ndarray],
    *,
    inference_batch_size: int,
    warmups: int,
    repetitions: int,
    synchronize: Callable[[], None] | None = None,
) -> dict[str, object]:
    if (
        len(images) != 16
        or inference_batch_size <= 0
        or 16 % inference_batch_size
    ):
        raise ValueError("TwinLite latency requires one exact 16-frame window")
    sync = synchronize or (lambda: None)

    def run_window() -> None:
        for start in range(0, 16, inference_batch_size):
            extractor.predict(images[start : start + inference_batch_size])

    for _ in range(int(warmups)):
        run_window()
    elapsed = []
    for _ in range(int(repetitions)):
        sync()
        started = time.perf_counter_ns()
        run_window()
        sync()
        elapsed.append((time.perf_counter_ns() - started) / 1_000_000.0)
    statistics = _statistics(elapsed)
    return {
        "component": "twinlitenet_masks",
        "batch_size": int(inference_batch_size),
        "observation_frames": 16,
        "warmups": int(warmups),
        "repetitions": int(repetitions),
        **statistics,
        "per_16_frame_window_mean_ms": statistics["mean_ms"],
        "per_frame_mean_ms": statistics["mean_ms"] / 16,
    }


def benchmark_road_components(
    drivable: np.ndarray,
    lane: np.ndarray,
    bboxes: np.ndarray,
    *,
    warmups: int,
    repetitions: int,
) -> list[dict[str, object]]:
    inputs = (
        np.asarray(drivable, dtype=bool),
        np.asarray(lane, dtype=bool),
        np.asarray(bboxes, dtype=np.float32),
    )
    if inputs[0].shape[0] != 16 or inputs[1].shape != inputs[0].shape or inputs[2].shape != (16, 4):
        raise ValueError("road latency requires exact 16-frame masks and BBoxes")
    for _ in range(int(warmups)):
        context.road_relative_sequence(*inputs)
    construction = []
    road = None
    for _ in range(int(repetitions)):
        started = time.perf_counter_ns()
        road = context.road_relative_sequence(*inputs)
        construction.append((time.perf_counter_ns() - started) / 1_000_000.0)
    assert road is not None
    for _ in range(int(warmups)):
        context.summarize_road(road)
    summary = []
    for _ in range(int(repetitions)):
        started = time.perf_counter_ns()
        context.summarize_road(road)
        summary.append((time.perf_counter_ns() - started) / 1_000_000.0)
    base = {"batch_size": 1, "warmups": int(warmups), "repetitions": int(repetitions)}
    output = [
        {"component": "road_relative_20d", **base, **_statistics(construction)},
        {"component": "road_summary_60d", **base, **_statistics(summary)},
    ]
    for record in output:
        record["observation_frames"] = 16
        record["per_16_frame_window_mean_ms"] = record["mean_ms"]
    return output
