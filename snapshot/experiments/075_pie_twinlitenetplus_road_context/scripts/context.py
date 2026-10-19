"""Authenticated Exp063 road equations and frozen Exp075 summaries."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import contract


_REPRESENTATIONS_PATH = contract.resolve_source("exp063_representations")
contract.resolve_source("exp063_contracts")
if str(contract.ACTIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(contract.ACTIVE_ROOT))
_REPRESENTATIONS = importlib.import_module(
    "experiments.063_pie_corrected_yolop_road_relative_context.scripts.representations"
)
if _REPRESENTATIONS_PATH.resolve() != __import__(
    "pathlib"
).Path(_REPRESENTATIONS.__file__).resolve():
    raise RuntimeError("authenticated Exp063 representation import drifted")

ROAD_CHANNELS = tuple(contract.load_registration()["road_context"]["channels"])
if tuple(_REPRESENTATIONS.ROAD_RELATIVE_CHANNELS) != ROAD_CHANNELS:
    raise RuntimeError("Exp063 road-relative channel order drifted")


def road_relative_sequence(
    drivable_masks: np.ndarray,
    lane_masks: np.ndarray,
    bboxes: np.ndarray,
) -> np.ndarray:
    values = _REPRESENTATIONS.road_relative_sequence(
        drivable_masks, lane_masks, bboxes
    )
    output = np.asarray(values, dtype=np.float32)
    if output.shape != (16, 20) or not np.isfinite(output).all():
        raise RuntimeError("Exp063 road-relative output must be finite 16x20")
    return output


def road_relative_static(
    drivable_mask: np.ndarray,
    lane_mask: np.ndarray,
    bbox: np.ndarray,
) -> np.ndarray:
    """Compute the frozen 13-D per-frame geometry without retaining masks."""

    values = _REPRESENTATIONS.road_relative_sequence(
        [drivable_mask], [lane_mask], np.asarray(bbox, dtype=np.float32)[None, :]
    )[0, :13]
    output = np.asarray(values, dtype=np.float32)
    if output.shape != (13,) or not np.isfinite(output).all():
        raise RuntimeError("Exp063 static road output must be finite 13-D")
    return output


def assemble_road_relative(
    static_features: np.ndarray,
    bboxes: np.ndarray,
) -> np.ndarray:
    """Apply the unchanged frozen temporal equations to compact frame state."""

    values = _REPRESENTATIONS.assemble_road_relative_temporal(
        static_features, bboxes
    )
    output = np.asarray(values, dtype=np.float32)
    if output.shape != (16, 20) or not np.isfinite(output).all():
        raise RuntimeError("Exp063 compact road output must be finite 16x20")
    return output


def context_columns() -> tuple[str, ...]:
    return tuple(
        name
        for channel in ROAD_CHANNELS
        for name in (
            f"road__{channel}__early_mean",
            f"road__{channel}__late_mean",
            f"road__{channel}__late_minus_early",
        )
    )


def summarize_road(sequence: np.ndarray) -> np.ndarray:
    values = np.asarray(sequence, dtype=np.float32)
    if values.shape != (16, 20) or not np.isfinite(values).all():
        raise ValueError("road-relative sequence must be finite 16x20")
    early = values[:8].mean(axis=0)
    late = values[8:].mean(axis=0)
    return np.column_stack((early, late, late - early)).reshape(-1)


def load_context(
    root: Path,
    split: str,
    expected_ids: np.ndarray,
    *,
    manifest_name: str = "extraction_manifest.json",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Authenticate extracted parts and return exact ordered 60-D summaries."""

    root = Path(root)
    manifest_path = root / "metrics" / manifest_name
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError("missing Exp075 extraction manifest") from error
    expected_set03 = str(split) == "test"
    if manifest.get("target_labels_passed_to_perception") is not False:
        raise RuntimeError("Exp075 context extraction is not label sealed")
    if manifest.get("set03_accessed") is not expected_set03:
        raise RuntimeError("Exp075 context manifest has wrong set03 access state")
    id_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    records = [record for record in manifest.get("parts", []) if record["split"] == split]
    if not records:
        raise RuntimeError(f"missing Exp075 context parts for {split}")
    for record in records:
        path = Path(record["path"])
        if not path.is_absolute():
            path = contract.ACTIVE_ROOT / path
        expected = {"sha256": record["sha256"], "size_bytes": record["size_bytes"]}
        if not path.is_file() or contract.artifact_record(path) != expected:
            raise RuntimeError(f"Exp075 context part authentication failed: {path}")
        with np.load(path, allow_pickle=False) as bundle:
            ids = np.asarray(bundle["sequence_id"]).astype(str)
            values = np.asarray(bundle["road_relative"], dtype=np.float32)
        if values.shape != (len(ids), 16, 20) or not np.isfinite(values).all():
            raise RuntimeError(f"malformed Exp075 context part: {path}")
        if len(ids) != int(record["rows"]):
            raise RuntimeError(f"Exp075 context part row count drifted: {path}")
        id_parts.append(ids)
        value_parts.append(values)
    ids = np.concatenate(id_parts)
    expected_ids = np.asarray(expected_ids).astype(str)
    if len(ids) != len(expected_ids):
        raise RuntimeError("Exp075 context population mismatch")
    if not np.array_equal(ids, expected_ids):
        raise RuntimeError("Exp075 context sequence order mismatch")
    summaries = np.stack([summarize_road(values) for values in np.concatenate(value_parts)])
    frame = pd.DataFrame(summaries, columns=context_columns())
    frame.insert(0, "sequence_id", ids)
    return frame, {
        "split": str(split),
        "rows": len(frame),
        "parts": len(records),
        "all_finite": bool(np.isfinite(summaries).all()),
        "set03_accessed": expected_set03,
        "target_labels_passed_to_perception": False,
    }


def compare_development_coverage(
    exp075: pd.DataFrame,
    exp074: dict[str, object],
) -> pd.DataFrame:
    """Compare like-for-like combined development coverage to original YOLOP."""

    required = {"split", "metric", "value"}
    if required.difference(exp075.columns):
        raise ValueError("Exp075 coverage table is incomplete")
    indexed = exp075.set_index(["split", "metric"])["value"]
    weights = {
        split: float(indexed.loc[(split, "windows")])
        for split in ("train", "validation")
    }
    denominator = sum(weights.values())

    def combined(metric: str) -> float:
        return sum(
            weights[split] * float(indexed.loc[(split, metric)])
            for split in ("train", "validation")
        ) / denominator

    rows = []
    for metric in (
        "road_geometry_valid_frame_fraction",
        "lane_geometry_valid_frame_fraction",
    ):
        old = float(exp074[metric])
        new = combined(metric)
        rows.append(
            {
                "metric": metric,
                "exp074": old,
                "exp075": new,
                "delta_exp075_minus_exp074": new - old,
            }
        )
    old_nonzero = dict(exp074["context_nonzero_fraction"])
    for index, column in enumerate(context_columns()):
        if column not in old_nonzero:
            continue
        new = combined(f"summary_{index:02d}_nonzero_fraction")
        old = float(old_nonzero[column])
        rows.append(
            {
                "metric": column,
                "exp074": old,
                "exp075": new,
                "delta_exp075_minus_exp074": new - old,
            }
        )
    return pd.DataFrame(rows)
