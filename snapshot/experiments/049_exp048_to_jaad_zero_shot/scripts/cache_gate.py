"""Immutable label-sealed feature cache and pre-inference authentication gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from contracts import OBSERVATION_LENGTH, sha256_file
from features import JAAD_POSE_PATH, PoseStore, build_non_yolop_features
from jaad_protocol import manifest_for_csv
from yolop import YolopStore


ARRAY_SPECS: dict[str, tuple[tuple[int, ...], str]] = {
    "bbox": ((OBSERVATION_LENGTH, 4), "float32"),
    "ped_action": ((OBSERVATION_LENGTH, 1), "float32"),
    "ped_look": ((OBSERVATION_LENGTH, 1), "float32"),
    "ego_speed": ((OBSERVATION_LENGTH, 1), "float32"),
    "ego_acc": ((OBSERVATION_LENGTH, 2), "float32"),
    "static_context": ((23,), "float32"),
    "yolop": ((OBSERVATION_LENGTH, 20), "float32"),
    "exp012_pose": ((OBSERVATION_LENGTH, 34), "float32"),
    "biu_v2": ((63,), "float32"),
    "biu_v1": ((72,), "float32"),
}


def sequence_id_sha256(sequence_ids: Sequence[str]) -> str:
    payload = "".join(f"{str(sequence_id)}\n" for sequence_id in sequence_ids)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_cache_identity(metadata: Mapping[str, object], sequence_ids: Sequence[str]) -> None:
    if bool(metadata.get("contains_labels", True)):
        raise ValueError("Feature cache must not contain test labels.")
    if int(metadata.get("row_count", -1)) != len(sequence_ids):
        raise ValueError("Feature cache row count drifted.")
    if str(metadata.get("sequence_id_sha256", "")) != sequence_id_sha256(sequence_ids):
        raise ValueError("Feature cache sequence identity drifted.")


def _open_arrays(root: Path, row_count: int, mode: str) -> dict[str, np.memmap]:
    arrays: dict[str, np.memmap] = {}
    for name, (tail_shape, dtype) in ARRAY_SPECS.items():
        path = root / f"{name}.npy"
        shape = (row_count, *tail_shape)
        if mode == "w+":
            arrays[name] = np.lib.format.open_memmap(path, mode=mode, dtype=dtype, shape=shape)
        else:
            array = np.load(path, mmap_mode=mode, allow_pickle=False)
            if array.shape != shape or str(array.dtype) != dtype:
                raise ValueError(f"Feature cache array contract drifted: {path}")
            arrays[name] = array
    return arrays


def build_feature_cache(
    manifest,
    database,
    pose_store: PoseStore,
    yolop_store: YolopStore,
    output_root: Path,
) -> dict[str, object]:
    """Build or resume one shared unlabeled A--F feature cache."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if "label" in manifest.columns:
        raise ValueError("Feature cache construction received test labels.")
    sequence_ids = manifest["sequence_id"].astype(str).tolist()
    identity_sha = sequence_id_sha256(sequence_ids)
    state_path = output_root / "build_state.json"
    manifest_path = output_root / "manifest.csv"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        validate_cache_identity(state, sequence_ids)
        next_index = int(state["next_index"])
        arrays = _open_arrays(output_root, len(manifest), "r+")
    else:
        if any((output_root / f"{name}.npy").exists() for name in ARRAY_SPECS):
            raise FileExistsError("Untracked partial feature arrays exist without build_state.json.")
        rows = manifest.reset_index(drop=True).copy()
        rows.insert(0, "cache_index", np.arange(len(rows), dtype=np.int64))
        manifest_for_csv(rows).to_csv(manifest_path, index=False)
        arrays = _open_arrays(output_root, len(rows), "w+")
        next_index = 0
        state = {
            "phase": "label_sealed_feature_cache_build",
            "row_count": len(rows),
            "contains_labels": False,
            "sequence_id_sha256": identity_sha,
            "next_index": 0,
        }
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for index in range(next_index, len(manifest)):
        row = manifest.iloc[index]
        bundle = build_non_yolop_features(row, database, pose_store)
        yolop = yolop_store.get_window(
            str(row["video_id"]), str(row["ped_id"]), row["observation_frames"]
        )
        values = {
            "bbox": bundle.bbox,
            "ped_action": bundle.ped_action,
            "ped_look": bundle.ped_look,
            "ego_speed": bundle.ego_speed,
            "ego_acc": bundle.ego_acc,
            "static_context": bundle.static_context,
            "yolop": yolop,
            "exp012_pose": bundle.pose,
            "biu_v2": bundle.biu_v2,
            "biu_v1": bundle.biu_v1,
        }
        for name, value in values.items():
            array = np.asarray(value, dtype=np.float32)
            expected_shape = ARRAY_SPECS[name][0]
            if array.shape != expected_shape or not np.isfinite(array).all():
                raise ValueError(f"Invalid {name} at cache row {index}: {array.shape}")
            arrays[name][index] = array
        if (index + 1) % 500 == 0 or index + 1 == len(manifest):
            for array in arrays.values():
                array.flush()
            state["next_index"] = index + 1
            state_path.write_text(
                json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"feature_cache_progress={index + 1}/{len(manifest)}", flush=True)
    artifacts = {"manifest.csv": sha256_file(manifest_path)}
    for name in ARRAY_SPECS:
        artifacts[f"{name}.npy"] = sha256_file(output_root / f"{name}.npy")
    metadata = {
        "phase": "immutable_label_sealed_feature_cache",
        "row_count": len(manifest),
        "contains_labels": False,
        "probabilities_computed": False,
        "sequence_id_sha256": identity_sha,
        "array_specs": {
            name: {"tail_shape": list(shape), "dtype": dtype}
            for name, (shape, dtype) in ARRAY_SPECS.items()
        },
        "artifacts": artifacts,
    }
    metadata_path = output_root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    state["phase"] = "complete"
    state["metadata_sha256"] = sha256_file(metadata_path)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def authenticate_feature_cache(
    output_root: Path, sequence_ids: Sequence[str]
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    output_root = Path(output_root)
    metadata_path = output_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_cache_identity(metadata, sequence_ids)
    for relative, expected_sha in metadata["artifacts"].items():
        path = output_root / relative
        if not path.is_file() or sha256_file(path) != str(expected_sha):
            raise ValueError(f"Feature cache artifact hash mismatch: {path}")
    arrays = _open_arrays(output_root, len(sequence_ids), "r")
    return metadata, arrays


def create_audit_marker(
    experiment_root: Path,
    sequence_ids: Sequence[str],
    checkpoint_records: Sequence[object],
) -> dict[str, object]:
    """Authenticate every pre-inference artifact and create the final seal."""
    experiment_root = Path(experiment_root)
    predictions_root = experiment_root / "predictions"
    if predictions_root.exists() and any(path.is_file() for path in predictions_root.rglob("*")):
        raise ValueError("Predictions exist before the pre-inference audit gate.")
    cache_root = experiment_root / "cache" / "test_features"
    metadata, _ = authenticate_feature_cache(cache_root, sequence_ids)
    marker_paths = (
        experiment_root / "cache" / "POSE_EXTRACTION_COMPLETE",
        experiment_root / "cache" / "FEATURE_AUDIT_COMPLETE",
        experiment_root / "cache" / "YOLOP_EXTRACTION_COMPLETE",
        experiment_root / "cache" / "FEATURE_CACHE_COMPLETE",
    )
    missing_markers = [str(path) for path in marker_paths if not path.is_file()]
    if missing_markers:
        raise FileNotFoundError(f"Missing pre-inference completion markers: {missing_markers}")
    compact_paths = (
        experiment_root / "metrics" / "protocol_audit_trainval.json",
        experiment_root / "metrics" / "feature_audit_test.json",
        experiment_root / "metrics" / "yolop_historical_parity.json",
        experiment_root / "metrics" / "yolop_extraction_report.json",
        cache_root / "metadata.json",
        JAAD_POSE_PATH,
        *marker_paths,
    )
    missing_compact = [str(path) for path in compact_paths if not path.is_file()]
    if missing_compact:
        raise FileNotFoundError(f"Missing pre-inference artifacts: {missing_compact}")
    checkpoint_rows = []
    identities = set()
    for record in checkpoint_records:
        identity = (str(record.variant), int(record.seed))
        if identity in identities:
            raise ValueError(f"Duplicate checkpoint identity at audit: {identity}")
        identities.add(identity)
        if sha256_file(record.checkpoint_path) != str(record.checkpoint_sha256):
            raise ValueError(f"Checkpoint drifted before audit: {identity}")
        checkpoint_rows.append(
            {
                "variant": identity[0],
                "seed": identity[1],
                "checkpoint_path": str(record.checkpoint_path),
                "checkpoint_sha256": str(record.checkpoint_sha256),
                "marker_path": str(record.marker_path),
                "marker_sha256": sha256_file(record.marker_path),
                "total_parameters": int(record.total_parameters),
            }
        )
    if len(checkpoint_rows) != 18:
        raise ValueError("Pre-inference audit requires exactly 18 checkpoints.")
    provenance_path = experiment_root / "metrics" / "checkpoint_provenance.json"
    provenance_path.write_text(
        json.dumps(checkpoint_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifact_hashes = {str(path): sha256_file(path) for path in compact_paths}
    artifact_hashes[str(provenance_path)] = sha256_file(provenance_path)
    payload = {
        "phase": "pre_inference_audit",
        "complete": True,
        "contains_test_labels": False,
        "probabilities_computed": False,
        "threshold_selected_on_jaad": False,
        "row_count": int(metadata["row_count"]),
        "sequence_id_sha256": str(metadata["sequence_id_sha256"]),
        "checkpoint_count": len(checkpoint_rows),
        "artifacts": artifact_hashes,
    }
    marker_path = experiment_root / "cache" / "AUDIT_COMPLETE"
    marker_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def authenticate_audit_marker(experiment_root: Path) -> dict[str, object]:
    experiment_root = Path(experiment_root)
    marker_path = experiment_root / "cache" / "AUDIT_COMPLETE"
    if not marker_path.is_file():
        raise FileNotFoundError(f"Missing AUDIT_COMPLETE marker: {marker_path}")
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    if payload.get("phase") != "pre_inference_audit" or payload.get("complete") is not True:
        raise ValueError("AUDIT_COMPLETE phase/status is invalid.")
    if payload.get("contains_test_labels") is not False or payload.get("probabilities_computed") is not False:
        raise ValueError("AUDIT_COMPLETE violates the frozen label/probability seal.")
    for raw_path, expected_sha in payload.get("artifacts", {}).items():
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != str(expected_sha):
            raise ValueError(f"Pre-inference artifact hash mismatch: {path}")
    return payload
