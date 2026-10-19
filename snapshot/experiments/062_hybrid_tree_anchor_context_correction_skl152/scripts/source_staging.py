"""One-time mixed-source isolation for Exp062 train/validation evidence."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from .contracts import (
    ExperimentConfig,
    canonical_json_sha256,
    experiment_config_sha256,
    sha256_file,
)
from .data import (
    EXP012_BUILDER,
    EXP012_PREDICTIONS,
    EXP055_TRAINVAL,
    TrainValData,
    _authenticate_exp012_definition,
    _load_exp012_builder,
    load_authenticated_trainval,
    read_historical_validation_prefix,
)


STAGING_MARKER = "SOURCE_STAGING.json"
STAGED_DIRECTORY = "staged_sources"


def _required_derived_names(config: ExperimentConfig) -> set[str]:
    return {
        "rows.csv",
        "summary_v1.npy",
        "summary_v2_core.npy",
        "ego.npy",
        "yolop.npy",
        "stage-sources.log",
        *(
            f"validation_reference_{anchor}_seed{seed}.csv"
            for anchor in config.anchor_families
            for seed in config.seeds
        ),
    }


@dataclass(frozen=True)
class StageBuildResult:
    """All mixed-source evidence returned by the one authorized build callback."""

    data: TrainValData
    validation_references: Mapping[tuple[str, int], pd.DataFrame]
    original_sources: Mapping[str, Path]
    extraction_rules: Mapping[str, str]


def _atomic_json(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        frame.to_csv(handle, index=False)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npy(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        np.save(handle, np.asarray(array), allow_pickle=False)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    return canonical_json_sha256(
        {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "bytes_sha256": __import__("hashlib").sha256(value.tobytes()).hexdigest(),
        }
    )


def _validate_build_result(result: StageBuildResult, config: ExperimentConfig) -> None:
    data = result.data
    total = config.train_rows + config.val_rows
    if (
        len(data.rows) != total
        or data.summary_v1.shape != (total, config.rf_feature_dim)
        or data.summary_v2_core.shape != (total, config.xgb_feature_dim)
        or data.ego.shape != (total, *config.ego_shape)
        or data.yolop.shape != (total, *config.yolop_shape)
        or data.labels.shape != (total,)
        or data.train_mask.shape != (total,)
        or data.val_mask.shape != (total,)
        or int(data.train_mask.sum()) != config.train_rows
        or int(data.val_mask.sum()) != config.val_rows
    ):
        raise ValueError("source staging received an invalid train/validation population")
    if data.rows["sequence_id"].astype(str).duplicated().any():
        raise ValueError("source staging requires unique ordered sequence IDs")
    if (
        not data.rows.loc[data.train_mask, "split"].astype(str).eq("train").all()
        or not data.rows.loc[data.val_mask, "split"].astype(str).eq("val").all()
        or not np.array_equal(
            data.rows["cache_index"].to_numpy(dtype=np.int64), np.arange(total)
        )
        or not np.array_equal(data.rows["label"].to_numpy(dtype=np.int64), data.labels)
    ):
        raise ValueError("source staging row order, split, or labels drifted")
    starts = data.rows["start_frame"].to_numpy(dtype=np.int64)
    ends = data.rows["end_frame"].to_numpy(dtype=np.int64)
    tte = data.rows["time_to_event_frames"].to_numpy(dtype=np.int64)
    if (
        not data.rows["observation_length"].eq(config.observed_frames).all()
        or not data.rows["frame_step"].eq(1).all()
        or not data.rows["frame_sampling"].astype(str).eq("consecutive").all()
        or not np.array_equal(ends - starts, np.full(total, config.observed_frames - 1))
        or not ((30 <= tte) & (tte <= 60)).all()
        or not np.array_equal(
            data.rows["event_frame"].to_numpy(dtype=np.int64) - ends, tte
        )
    ):
        raise ValueError("source staging temporal invariants failed")
    expected_references = {
        (anchor, seed) for anchor in config.anchor_families for seed in config.seeds
    }
    if set(result.validation_references) != expected_references:
        raise ValueError("source staging requires all six validation parity references")
    expected_validation = data.rows.loc[
        data.val_mask, ["sequence_id", "split", "label"]
    ].reset_index(drop=True)
    for identity, frame in result.validation_references.items():
        if (
            frame.columns.tolist()
            != ["sequence_id", "split", "label", "probability"]
            or len(frame) != config.val_rows
            or not frame["split"].astype(str).eq("val").all()
            or not np.array_equal(
                frame["sequence_id"].astype(str).to_numpy(),
                expected_validation["sequence_id"].astype(str).to_numpy(),
            )
            or not np.array_equal(
                frame["label"].to_numpy(dtype=np.int64),
                expected_validation["label"].to_numpy(dtype=np.int64),
            )
        ):
            raise ValueError(f"staged validation reference is malformed: {identity}")


def _production_build(config: ExperimentConfig) -> StageBuildResult:
    """Open mixed historical containers only inside the explicit staging command."""
    data = load_authenticated_trainval(config)
    expected_validation = data.rows.loc[
        data.val_mask, ["sequence_id", "split", "label"]
    ].reset_index(drop=True)
    references: dict[tuple[str, int], pd.DataFrame] = {}
    prediction_sources: dict[str, Path] = {}
    for anchor in config.anchor_families:
        feature_set, model = (
            ("biu_v1_plus_bbox", "rf")
            if anchor == "rf_v1_bbox"
            else ("biu_v2_core_plus_bbox", "xgb")
        )
        for seed in config.seeds:
            source = EXP012_PREDICTIONS / f"{feature_set}__{model}__seed{seed}.csv"
            references[(anchor, seed)] = read_historical_validation_prefix(
                source, anchor, seed, expected_validation
            )
            prediction_sources[f"historical_prediction_{anchor}_seed{seed}"] = source

    definition_path, _ = _authenticate_exp012_definition(config)
    builder = _load_exp012_builder()
    original_sources: dict[str, Path] = {
        "trainval_rows": EXP055_TRAINVAL / "rows.csv",
        "trainval_ego": EXP055_TRAINVAL / "ego.npy",
        "trainval_yolop": EXP055_TRAINVAL / "yolop.npy",
        "exp012_builder": EXP012_BUILDER,
        "exp012_feature_definition": definition_path,
        "exp012_keypoints_mixed_container": Path(builder.KEYPOINTS_JSONL),
        "exp012_feature_dictionary": Path(builder.DICT_PATH),
        **prediction_sources,
    }
    if "source_annotation_path" in data.rows:
        for index, source in enumerate(
            sorted({str(value) for value in data.rows["source_annotation_path"]})
        ):
            original_sources[f"pie_annotation_{index:04d}"] = Path(source)
    return StageBuildResult(
        data=data,
        validation_references=references,
        original_sources=original_sources,
        extraction_rules={
            "anchor_matrices": (
                "Task 2 Exp012 reconstruction; exact needed train/validation keys; "
                "SPLITS=(train,val); ordered authenticated rows"
            ),
            "historical_predictions": (
                "read exactly first 3619 rows; require split=val; exact ordered IDs/labels"
            ),
            "contexts": "authenticated Exp055 trainval cache; train/validation rows only",
        },
    )


def stage_sources(
    root: Path,
    *,
    config: ExperimentConfig | None = None,
    build: Callable[[], StageBuildResult] | None = None,
) -> dict[str, object]:
    """Create the immutable train/validation-only source boundary exactly once."""
    root = Path(root).resolve()
    config = config or ExperimentConfig()
    marker_path = root / STAGING_MARKER
    staged_root = root / STAGED_DIRECTORY
    log_path = root / "logs" / "stage-sources.log"
    if marker_path.exists() or staged_root.exists() or log_path.exists():
        raise FileExistsError("Exp062 source staging already exists; refusing restaging")
    result = (build or (lambda: _production_build(config)))()
    _validate_build_result(result, config)
    staged_root.mkdir(parents=True, exist_ok=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("Exp062 stage-sources started\n", encoding="utf-8")

    paths = {
        "rows.csv": staged_root / "rows.csv",
        "summary_v1.npy": staged_root / "summary_v1.npy",
        "summary_v2_core.npy": staged_root / "summary_v2_core.npy",
        "ego.npy": staged_root / "ego.npy",
        "yolop.npy": staged_root / "yolop.npy",
    }
    _atomic_csv(result.data.rows, paths["rows.csv"])
    _atomic_npy(result.data.summary_v1, paths["summary_v1.npy"])
    _atomic_npy(result.data.summary_v2_core, paths["summary_v2_core.npy"])
    _atomic_npy(result.data.ego, paths["ego.npy"])
    _atomic_npy(result.data.yolop, paths["yolop.npy"])
    for (anchor, seed), reference in result.validation_references.items():
        name = f"validation_reference_{anchor}_seed{seed}.csv"
        paths[name] = staged_root / name
        _atomic_csv(reference, paths[name])
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("Exp062 stage-sources completed\n")
    paths["stage-sources.log"] = log_path

    original_sources = {}
    for name, source in sorted(result.original_sources.items()):
        source = Path(source).resolve()
        if not source.is_file():
            raise RuntimeError(f"source staging original source is missing: {name}")
        original_sources[name] = {"path": str(source), "sha256": sha256_file(source)}
    derived = {
        name: {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
        }
        for name, path in sorted(paths.items())
    }
    ordered_ids = result.data.rows["sequence_id"].astype(str).tolist()
    labels = result.data.labels.astype(np.int64)
    payload: dict[str, object] = {
        "schema": "exp062-source-staging-v1",
        "config_sha256": experiment_config_sha256(config),
        "extraction_rules": dict(result.extraction_rules),
        "original_sources": original_sources,
        "ordered_sequence_ids_sha256": canonical_json_sha256(ordered_ids),
        "ordered_labels_sha256": _array_sha256(labels),
        "train_rows": config.train_rows,
        "validation_rows": config.val_rows,
        "derived_artifacts": derived,
    }
    _atomic_json(payload, marker_path)
    return payload


def verify_source_staging(
    root: Path, *, config: ExperimentConfig | None = None
) -> dict[str, object]:
    """Authenticate only isolated derived files; never resolve original sources."""
    root = Path(root).resolve()
    config = config or ExperimentConfig()
    marker_path = root / STAGING_MARKER
    if marker_path.is_symlink():
        raise RuntimeError("Exp062 source staging marker must not be a symlink")
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Exp062 source staging marker is missing or invalid") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "exp062-source-staging-v1"
        or payload.get("config_sha256") != experiment_config_sha256(config)
    ):
        raise RuntimeError("Exp062 source staging identity mismatch")
    derived = payload.get("derived_artifacts")
    if not isinstance(derived, dict) or set(derived) != _required_derived_names(config):
        raise RuntimeError("Exp062 staged derived artifact manifest is malformed")
    for name, evidence in derived.items():
        if not isinstance(evidence, dict):
            raise RuntimeError("Exp062 staged derived artifact evidence is malformed")
        expected = (
            root / "logs" / name
            if name == "stage-sources.log"
            else root / STAGED_DIRECTORY / name
        )
        actual = Path(os.path.abspath(root / str(evidence.get("path", ""))))
        expected_absolute = Path(os.path.abspath(expected))
        current = root
        relative = expected.relative_to(root)
        has_symlink = False
        for component in relative.parts:
            current = current / component
            if current.is_symlink():
                has_symlink = True
                break
        if (
            has_symlink
            or actual != expected_absolute
            or not actual.is_file()
            or sha256_file(actual) != evidence.get("sha256")
        ):
            raise RuntimeError(f"derived artifact hash mismatch: {name}")
    return payload


def load_staged_trainval(
    root: Path, *, config: ExperimentConfig | None = None
) -> TrainValData:
    """Load only the immutable staged train/validation population."""
    root = Path(root).resolve()
    config = config or ExperimentConfig()
    manifest = verify_source_staging(root, config=config)
    staged = root / STAGED_DIRECTORY
    rows = pd.read_csv(staged / "rows.csv")
    labels = rows["label"].to_numpy(dtype=np.int64)
    train_mask = rows["split"].astype(str).eq("train").to_numpy()
    val_mask = rows["split"].astype(str).eq("val").to_numpy()
    data = TrainValData(
        rows=rows,
        summary_v1=np.load(staged / "summary_v1.npy", allow_pickle=False),
        summary_v2_core=np.load(staged / "summary_v2_core.npy", allow_pickle=False),
        ego=np.load(staged / "ego.npy", allow_pickle=False),
        yolop=np.load(staged / "yolop.npy", allow_pickle=False),
        labels=labels,
        train_mask=train_mask,
        val_mask=val_mask,
    )
    _validate_build_result(
        StageBuildResult(
            data=data,
            validation_references={
                (anchor, seed): read_staged_validation_reference(
                    root, anchor, seed, config=config, verify=False
                )
                for anchor in config.anchor_families
                for seed in config.seeds
            },
            original_sources={},
            extraction_rules={},
        ),
        config,
    )
    if manifest.get("ordered_sequence_ids_sha256") != canonical_json_sha256(
        rows["sequence_id"].astype(str).tolist()
    ) or manifest.get("ordered_labels_sha256") != _array_sha256(labels):
        raise RuntimeError("staged row identity or labels drifted")
    return data


def read_staged_validation_reference(
    root: Path,
    anchor_name: str,
    seed: int,
    *,
    config: ExperimentConfig | None = None,
    verify: bool = True,
) -> pd.DataFrame:
    """Read one isolated validation-only parity reference."""
    root = Path(root).resolve()
    config = config or ExperimentConfig()
    if anchor_name not in config.anchor_families or seed not in config.seeds:
        raise ValueError("staged validation reference identity is invalid")
    if verify:
        verify_source_staging(root, config=config)
    path = (
        root
        / STAGED_DIRECTORY
        / f"validation_reference_{anchor_name}_seed{seed}.csv"
    )
    return pd.read_csv(path, float_precision="round_trip")


__all__ = [
    "STAGING_MARKER",
    "STAGED_DIRECTORY",
    "StageBuildResult",
    "stage_sources",
    "verify_source_staging",
    "load_staged_trainval",
    "read_staged_validation_reference",
]
