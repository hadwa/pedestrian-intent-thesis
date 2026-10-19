"""Frozen Exp082-D feature assembly for Exp083."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CONTRACT = _load(SCRIPT_DIR / "contract.py", "exp083_contract_for_features")
CONTEXT = _load(
    SCRIPT_DIR / "context_extraction.py", "exp083_context_for_features"
)
PARENT_MODEL = _load(
    CONTRACT.resolve_source("exp081_model"), "exp083_authenticated_exp081_model"
)
COVERAGE = _load(SCRIPT_DIR / "coverage.py", "exp083_coverage_for_features")

ANCHOR_COLUMNS = tuple(PARENT_MODEL.anchor_columns())
ROAD_COLUMNS = tuple(PARENT_MODEL.compensated_road_columns())
PREDICTOR_COLUMNS = ANCHOR_COLUMNS + ROAD_COLUMNS
if not (
    len(ANCHOR_COLUMNS) == 182
    and len(ROAD_COLUMNS) == 60
    and len(PREDICTOR_COLUMNS) == 242
    and len(set(PREDICTOR_COLUMNS)) == 242
):
    raise RuntimeError("Exp083 frozen Variant D feature ABI changed")


def anchor_columns() -> tuple[str, ...]:
    return ANCHOR_COLUMNS


def road_columns() -> tuple[str, ...]:
    return ROAD_COLUMNS


def predictor_columns() -> tuple[str, ...]:
    return PREDICTOR_COLUMNS


def fit_training_medians(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or not len(matrix) or np.isinf(matrix).any():
        raise ValueError("Exp083 training imputer requires a finite-or-NaN matrix")
    medians = np.nanmedian(matrix, axis=0)
    if medians.shape != (matrix.shape[1],) or not np.isfinite(medians).all():
        raise RuntimeError("Exp083 training contains an all-missing feature")
    return medians


def apply_training_medians(values: np.ndarray, medians: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float).copy()
    frozen = np.asarray(medians, dtype=float)
    if (
        matrix.ndim != 2
        or frozen.shape != (matrix.shape[1],)
        or not np.isfinite(frozen).all()
        or np.isinf(matrix).any()
    ):
        raise ValueError("Exp083 imputer matrix or frozen medians are invalid")
    missing = np.isnan(matrix)
    if missing.any():
        matrix[missing] = np.broadcast_to(frozen, matrix.shape)[missing]
    if not np.isfinite(matrix).all():
        raise RuntimeError("Exp083 imputation left non-finite predictors")
    return matrix


def _pose_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return np.array_equal(left["keypoints"], right["keypoints"]) and np.array_equal(
        left["kp_conf"], right["kp_conf"]
    )


def load_pose_index(
    needed: set[tuple[str, str, str, int]],
) -> tuple[dict[tuple[str, str, str, int], dict[str, np.ndarray]], dict[str, Any]]:
    output = {}
    contributions = {}
    duplicates = 0
    for source in COVERAGE.POSE_SOURCES:
        path = Path(source["path"])
        if CONTRACT.sha256_file(path) != source["sha256"]:
            raise RuntimeError(f"Exp083 pose source authentication failed: {path}")
        added = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                item = json.loads(line)
                key = (
                    str(item["set"]),
                    str(item["video"]),
                    str(item["pid"]),
                    int(item["frame_idx"]),
                )
                if key not in needed:
                    continue
                record = {
                    "keypoints": np.asarray(item["keypoints"], dtype=float),
                    "kp_conf": np.asarray(item["kp_conf"], dtype=float),
                }
                if record["keypoints"].shape != (17, 2) or record["kp_conf"].shape != (
                    17,
                ):
                    raise ValueError(f"invalid COCO-17 pose at {path}:{line_number}")
                if key in output:
                    if not _pose_equal(output[key], record):
                        raise ValueError(f"conflicting exact pose key: {key}")
                    duplicates += 1
                    continue
                output[key] = record
                added += 1
        contributions[str(path)] = added
    missing = needed.difference(output)
    if missing:
        raise ValueError(f"missing exact pose frame: {sorted(missing)[0]}")
    return output, {
        "required": len(needed),
        "loaded": len(output),
        "duplicate_identical": duplicates,
        "source_contributions": contributions,
    }


def _needed_pose_keys(manifest: pd.DataFrame) -> set[tuple[str, str, str, int]]:
    return {
        ("jaad", str(row.video_id), str(row.ped_id), frame_id)
        for row in manifest.itertuples(index=False)
        for frame_id in CONTEXT._observed_frames(row.observed_frames_json)
    }


def build_anchor_182(
    manifest: pd.DataFrame,
    *,
    pose_index: Mapping[tuple[str, str, str, int], Mapping[str, Any]],
) -> pd.DataFrame:
    needed = _needed_pose_keys(manifest)
    missing = needed.difference(pose_index)
    if missing:
        raise ValueError(f"missing exact pose frame: {sorted(missing)[0]}")
    feature_contract = PARENT_MODEL.TWO_STAGE_CORE.load_joint_contract(
        PARENT_MODEL.DEFINITIONS_PATH
    )
    rows = []
    source_ids = CONTEXT.source_sequence_ids(manifest)
    for source_id, row in zip(source_ids, manifest.itertuples(index=False)):
        frame_ids = CONTEXT._observed_frames(row.observed_frames_json)
        keys = [
            ("jaad", str(row.video_id), str(row.ped_id), frame_id)
            for frame_id in frame_ids
        ]
        coords = np.asarray([pose_index[key]["keypoints"] for key in keys], dtype=float)
        confidence = np.asarray(
            [pose_index[key]["kp_conf"] for key in keys], dtype=float
        )
        boxes = np.asarray(json.loads(str(row.bbox_sequence_json)), dtype=float)
        if coords.shape != (16, 17, 2) or confidence.shape != (16, 17):
            raise ValueError(f"invalid exact pose window: {row.sequence_id}")
        if boxes.shape != (16, 4) or not np.isfinite(boxes).all():
            raise ValueError(f"invalid exact BBox window: {row.sequence_id}")
        views = PARENT_MODEL.TWO_STAGE_CORE.extract_window_views(
            coords, confidence, boxes, contract=feature_contract
        )
        portable = views["portable"]
        if tuple(portable) != ANCHOR_COLUMNS:
            raise RuntimeError("Exp083 anchor feature ordering changed")
        rows.append(
            {
                "sequence_id": str(row.sequence_id),
                "source_sequence_id": str(source_id),
                "label": int(row.label),
                **portable,
            }
        )
    output = pd.DataFrame(rows)
    if output.shape != (len(manifest), 3 + len(ANCHOR_COLUMNS)):
        raise RuntimeError("Exp083 anchor shape changed")
    if output["sequence_id"].duplicated().any():
        raise RuntimeError("Exp083 anchor identity changed")
    return output


def assemble_variant_d(anchor: pd.DataFrame, road: pd.DataFrame) -> pd.DataFrame:
    anchor_required = {"sequence_id", "source_sequence_id", "label", *ANCHOR_COLUMNS}
    road_required = {"source_sequence_id", *ROAD_COLUMNS}
    if anchor_required.difference(anchor.columns):
        raise ValueError("Exp083 anchor is incomplete")
    if road_required.difference(road.columns):
        raise ValueError("Exp083 road context is incomplete")
    if "label" in road.columns:
        raise ValueError("Exp083 road context must be label-free")
    checked_anchor = anchor.loc[
        :, ["sequence_id", "source_sequence_id", "label", *ANCHOR_COLUMNS]
    ].copy()
    checked_road = road.loc[:, ["source_sequence_id", *ROAD_COLUMNS]].copy()
    if checked_road["source_sequence_id"].astype(str).duplicated().any():
        raise ValueError("Exp083 road context identities are not unique")
    merged = checked_anchor.merge(
        checked_road,
        on="source_sequence_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    np.testing.assert_array_equal(
        merged["sequence_id"].astype(str), checked_anchor["sequence_id"].astype(str)
    )
    if merged.loc[:, ROAD_COLUMNS].isna().any().any():
        raise RuntimeError("Exp083 road context join is incomplete")
    result = merged.loc[
        :, ["sequence_id", "source_sequence_id", "label", *PREDICTOR_COLUMNS]
    ]
    if result.shape != (len(anchor), 3 + len(PREDICTOR_COLUMNS)):
        raise RuntimeError("Exp083 Variant D shape changed")
    return result


def assert_shared_row_parity(
    actual: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    shared = actual.merge(
        reference,
        on="source_sequence_id",
        how="inner",
        suffixes=("__actual", "__reference"),
        validate="one_to_one",
    )
    if shared.empty:
        raise RuntimeError("Exp083 parity has no shared exact rows")
    differences = []
    nan_mismatches = 0
    for column in columns:
        left = shared[f"{column}__actual"].to_numpy(float)
        right = shared[f"{column}__reference"].to_numpy(float)
        nan_mismatches += int(np.logical_xor(np.isnan(left), np.isnan(right)).sum())
        finite = np.isfinite(left) & np.isfinite(right)
        if finite.any():
            differences.append(np.abs(left[finite] - right[finite]))
    maximum = float(np.concatenate(differences).max()) if differences else 0.0
    if nan_mismatches or maximum > tolerance:
        raise RuntimeError("Exp083 shared-row feature parity failed")
    return {
        "shared_rows": len(shared),
        "columns": len(columns),
        "max_abs_difference": maximum,
        "nan_mismatches": nan_mismatches,
        "tolerance": tolerance,
        "status": "pass",
    }


def _load_context_roots(
    roots: list[Path], expected_ids: set[str]
) -> pd.DataFrame:
    ids = []
    values = []
    for root in roots:
        for path in sorted(root.rglob("*.npz")):
            with np.load(path, allow_pickle=False) as archive:
                part_ids = archive["sequence_id"].astype(str)
                selected = np.asarray(
                    [sequence_id in expected_ids for sequence_id in part_ids],
                    dtype=bool,
                )
                if selected.any():
                    ids.extend(part_ids[selected].tolist())
                    values.append(
                        np.asarray(
                            archive["road_compensated_60d"][selected],
                            dtype=np.float32,
                        )
                    )
    if not ids:
        return pd.DataFrame(columns=["source_sequence_id", *ROAD_COLUMNS])
    matrix = np.concatenate(values, axis=0)
    result = pd.DataFrame(matrix, columns=ROAD_COLUMNS)
    result.insert(0, "source_sequence_id", ids)
    if result["source_sequence_id"].duplicated().any():
        raise RuntimeError("Exp083 context source identities overlap")
    return result


def load_development_context(split: str, expected_ids: set[str]) -> pd.DataFrame:
    parent_split = "validation" if split == "val" else split
    roots = [
        CONTEXT.PARENT_DEVELOPMENT_CACHE / parent_split,
        CONTRACT.EXPERIMENT_ROOT / "cache/context_parts/development" / split,
    ]
    result = _load_context_roots(roots, expected_ids)
    actual = set(result["source_sequence_id"].astype(str))
    if actual != expected_ids:
        missing = sorted(expected_ids.difference(actual))
        raise RuntimeError(f"Exp083 development context is incomplete: {missing[:1]}")
    return result


def load_test_context(expected_ids: set[str]) -> pd.DataFrame:
    roots = [
        CONTEXT.PARENT_TEST_CACHE,
        CONTRACT.EXPERIMENT_ROOT / "cache/context_parts/test",
    ]
    result = _load_context_roots(roots, expected_ids)
    actual = set(result["source_sequence_id"].astype(str))
    if actual != expected_ids:
        missing = sorted(expected_ids.difference(actual))
        raise RuntimeError(f"Exp083 test context is incomplete: {missing[:1]}")
    return result


def _subset_anchor(
    all_anchor: pd.DataFrame,
    target_manifest: pd.DataFrame,
) -> pd.DataFrame:
    identities = pd.DataFrame(
        {
            "sequence_id": target_manifest["sequence_id"].astype(str),
            "source_sequence_id": CONTEXT.source_sequence_ids(target_manifest),
            "label": target_manifest["label"].astype(int),
        }
    )
    source = all_anchor.loc[:, ["source_sequence_id", *ANCHOR_COLUMNS]].copy()
    result = identities.merge(
        source,
        on="source_sequence_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if result.loc[:, ANCHOR_COLUMNS].isna().all(axis=1).any():
        raise RuntimeError("Exp083 JAAD_beh is not an exact JAAD_all subset")
    return result


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_feature_cache(
    path: Path,
    frame: pd.DataFrame,
    track_ids: np.ndarray,
    medians: np.ndarray,
) -> None:
    matrix = frame.loc[:, PREDICTOR_COLUMNS].to_numpy(float)
    transformed = apply_training_medians(matrix, medians).astype(np.float32)
    _atomic_npz(
        path,
        sequence_id=np.asarray(frame["sequence_id"].astype(str).tolist(), dtype=np.str_),
        source_sequence_id=np.asarray(
            frame["source_sequence_id"].astype(str).tolist(), dtype=np.str_
        ),
        track_id=np.asarray(np.asarray(track_ids).astype(str).tolist(), dtype=np.str_),
        label=frame["label"].to_numpy(np.int8),
        X=transformed,
        predictor_columns=np.asarray(PREDICTOR_COLUMNS),
    )


def run_prepare_development(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    marker = CONTRACT.authenticate_phase(root, "extract-development-context")
    if marker.get("quality_status") != "pass" or marker.get("test_accessed") is not False:
        raise RuntimeError("Exp083 development context phase did not pass")
    if (root / "cache/jaad_beh_test_features.npz").exists() or (
        root / "cache/jaad_all_test_features.npz"
    ).exists():
        raise RuntimeError("Exp083 test feature cache exists before development freeze")

    manifests = {
        (cohort, split): pd.read_csv(
            root / "manifests" / f"{cohort}_{split}.csv"
        )
        for cohort in ("jaad_beh", "jaad_all")
        for split in ("train", "val")
    }
    all_development = pd.concat(
        [manifests[("jaad_all", "train")], manifests[("jaad_all", "val")]],
        ignore_index=True,
    )
    needed = _needed_pose_keys(all_development)
    pose_index, pose_audit = load_pose_index(needed)

    assembled: dict[tuple[str, str], pd.DataFrame] = {}
    anchor_parity = {}
    road_parity = {}
    for split in ("train", "val"):
        all_manifest = manifests[("jaad_all", split)]
        all_anchor = build_anchor_182(all_manifest, pose_index=pose_index)
        expected_ids = set(all_anchor["source_sequence_id"].astype(str))
        context = load_development_context(split, expected_ids)
        assembled[("jaad_all", split)] = assemble_variant_d(all_anchor, context)

        behavior_anchor = _subset_anchor(
            all_anchor, manifests[("jaad_beh", split)]
        )
        assembled[("jaad_beh", split)] = assemble_variant_d(
            behavior_anchor, context
        )

        source_name = (
            "exp077_prepared_validation" if split == "val" else "exp077_prepared_train"
        )
        reference = pd.read_csv(
            PARENT_MODEL.CONTRACT.resolve_source(source_name),
            usecols=["sequence_id", *ANCHOR_COLUMNS],
        ).rename(columns={"sequence_id": "source_sequence_id"})
        anchor_parity[split] = assert_shared_row_parity(
            all_anchor,
            reference,
            columns=ANCHOR_COLUMNS,
        )

        parent_split = "validation" if split == "val" else split
        parent_context = _load_context_roots(
            [CONTEXT.PARENT_DEVELOPMENT_CACHE / parent_split], expected_ids
        )
        road_parity[split] = assert_shared_row_parity(
            context,
            parent_context,
            columns=ROAD_COLUMNS,
        )

    outputs = []
    imputer_records = {}
    for cohort in ("jaad_beh", "jaad_all"):
        training = assembled[(cohort, "train")]
        medians = fit_training_medians(
            training.loc[:, PREDICTOR_COLUMNS].to_numpy(float)
        )
        imputer_path = root / "metrics" / f"{cohort}_training_imputer.npz"
        _atomic_npz(
            imputer_path,
            medians=medians,
            predictor_columns=np.asarray(PREDICTOR_COLUMNS),
        )
        outputs.append(imputer_path)
        imputer_records[cohort] = CONTRACT.artifact_record(imputer_path)
        for split in ("train", "val"):
            cache_path = root / "cache" / f"{cohort}_{split}_features.npz"
            _write_feature_cache(
                cache_path,
                assembled[(cohort, split)],
                manifests[(cohort, split)]["track_id"].astype(str).to_numpy(),
                medians,
            )
            outputs.append(cache_path)

    feature_contract_path = root / "metrics/feature_contract.json"
    parity_path = root / "metrics/feature_parity.json"
    CONTRACT.atomic_write_json(
        feature_contract_path,
        {
            "anchor_columns": list(ANCHOR_COLUMNS),
            "anchor_dimension": len(ANCHOR_COLUMNS),
            "road_columns": list(ROAD_COLUMNS),
            "road_dimension": len(ROAD_COLUMNS),
            "predictor_columns": list(PREDICTOR_COLUMNS),
            "predictor_dimension": len(PREDICTOR_COLUMNS),
            "imputation": "per-cohort training median applied unchanged",
            "imputers": imputer_records,
            "test_accessed": False,
        },
    )
    CONTRACT.atomic_write_json(
        parity_path,
        {
            "pose": pose_audit,
            "anchor_shared_exp077": anchor_parity,
            "road_shared_exp081": road_parity,
            "test_accessed": False,
        },
    )
    outputs.extend([feature_contract_path, parity_path])
    phase = CONTRACT.publish_phase(
        root,
        "prepare-development",
        outputs,
        test_accessed=False,
        test_features_accessed=False,
        test_metrics_accessed=False,
        cohorts=["jaad_beh", "jaad_all"],
        dimensions=242,
    )
    return {
        "marker": str(phase),
        "rows": {
            f"{cohort}_{split}": len(assembled[(cohort, split)])
            for cohort in ("jaad_beh", "jaad_all")
            for split in ("train", "val")
        },
    }


def _load_frozen_imputer(root: Path, cohort: str) -> np.ndarray:
    path = root / "metrics" / f"{cohort}_training_imputer.npz"
    with np.load(path, allow_pickle=False) as archive:
        columns = tuple(archive["predictor_columns"].astype(str))
        medians = np.asarray(archive["medians"], dtype=float)
    if columns != PREDICTOR_COLUMNS or medians.shape != (242,):
        raise RuntimeError("Exp083 frozen training imputer changed")
    return medians


def run_prepare_test(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    extraction = CONTRACT.authenticate_phase(root, "extract-test-context")
    CONTRACT.authenticate_phase(root, "prepare-development")
    CONTRACT.authenticate_phase(root, "train-validate")
    if (
        extraction.get("test_accessed") is not True
        or extraction.get("test_metrics_accessed") is not False
    ):
        raise RuntimeError("Exp083 test context authorization changed")

    manifests = {
        cohort: pd.read_csv(root / "manifests" / f"{cohort}_test.csv")
        for cohort in ("jaad_beh", "jaad_all")
    }
    needed = _needed_pose_keys(manifests["jaad_all"])
    pose_index, pose_audit = load_pose_index(needed)
    all_anchor = build_anchor_182(manifests["jaad_all"], pose_index=pose_index)
    expected_ids = set(all_anchor["source_sequence_id"].astype(str))
    context = load_test_context(expected_ids)
    assembled = {
        "jaad_all": assemble_variant_d(all_anchor, context),
        "jaad_beh": assemble_variant_d(
            _subset_anchor(all_anchor, manifests["jaad_beh"]), context
        ),
    }
    outputs = []
    for cohort, frame in assembled.items():
        medians = _load_frozen_imputer(root, cohort)
        cache_path = root / "cache" / f"{cohort}_test_features.npz"
        _write_feature_cache(
            cache_path,
            frame,
            manifests[cohort]["track_id"].astype(str).to_numpy(),
            medians,
        )
        outputs.append(cache_path)
    audit_path = root / "metrics/test_feature_audit.json"
    CONTRACT.atomic_write_json(
        audit_path,
        {
            "pose": pose_audit,
            "rows": {cohort: len(frame) for cohort, frame in assembled.items()},
            "predictor_dimension": 242,
            "training_imputers_reused": True,
            "test_metrics_accessed": False,
        },
    )
    outputs.append(audit_path)
    phase = CONTRACT.publish_phase(
        root,
        "prepare-test",
        outputs,
        test_accessed=True,
        test_features_accessed=True,
        test_metrics_accessed=False,
        rows={cohort: len(frame) for cohort, frame in assembled.items()},
    )
    return {
        "marker": str(phase),
        "rows": {cohort: len(frame) for cohort, frame in assembled.items()},
    }
