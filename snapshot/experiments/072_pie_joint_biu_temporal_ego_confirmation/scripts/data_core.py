"""Authenticated cohort and prediction boundaries for Exp072."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ACTIVE_ROOT = Path(__file__).resolve().parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import contract  # noqa: E402


REGISTRATION = contract.EXPECTED_REGISTRATION
IDENTITY_COLUMNS = (
    "sequence_id",
    "set_id",
    "video_id",
    "ped_id",
    "track_id",
    "start_frame",
    "end_frame",
    "label",
    "observation_length",
    "frame_step",
    "frame_sampling",
    "time_to_event_frames",
)


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ACTIVE_ROOT / value


def authenticate_parent(path: Path, expected_sha256: str) -> Path:
    resolved = Path(path).resolve(strict=True)
    if contract.sha256_file(resolved) != str(expected_sha256):
        raise RuntimeError(f"parent artifact hash mismatch: {resolved}")
    return resolved


def _source(name: str) -> Path:
    record = REGISTRATION["sources"][name]
    return authenticate_parent(_resolve(record["path"]), record["sha256"])


def _authenticate_completion(name: str) -> dict[str, Any]:
    marker = _source(name)
    completion = json.loads(marker.read_text(encoding="utf-8"))
    outputs = completion.get("outputs")
    if completion.get("status") != "complete" or not isinstance(outputs, dict) or not outputs:
        raise RuntimeError(f"invalid parent completion: {marker}")
    for relative, expected in outputs.items():
        artifact = marker.parent / relative
        if not artifact.is_file() or contract.artifact_record(artifact) != expected:
            raise RuntimeError(f"parent completion output mismatch: {artifact}")
    return completion


def _authenticate_parent_scripts() -> None:
    for name, expected in REGISTRATION["parent_script_sha256"].items():
        authenticate_parent(_resolve(REGISTRATION["parent_scripts"][name]), expected)


def feature_columns() -> tuple[str, ...]:
    path = _source("feature_contract")
    payload = json.loads(path.read_text(encoding="utf-8"))
    columns = tuple(map(str, payload["arms"][REGISTRATION["arm"]]))
    if len(columns) != REGISTRATION["feature_count"] or len(columns) != len(set(columns)):
        raise RuntimeError("Exp072 feature contract is not exactly 187 unique columns")
    return columns


def _validate_observation_frame(frame: pd.DataFrame, *, name: str, expected_sets: set[str], rows: int) -> None:
    missing = set(IDENTITY_COLUMNS).difference(frame.columns)
    if missing:
        raise RuntimeError(f"{name} missing identity columns: {sorted(missing)}")
    if len(frame) != rows or frame["sequence_id"].duplicated().any():
        raise RuntimeError(f"{name} row identity changed")
    if set(frame["set_id"].astype(str)) != expected_sets:
        raise RuntimeError(f"{name} set membership changed")
    if not set(frame["label"].astype(int).unique()).issubset({0, 1}):
        raise RuntimeError(f"{name} labels are not binary")
    exact = (
        frame["observation_length"].astype(int).eq(16)
        & frame["frame_step"].astype(int).eq(1)
        & frame["frame_sampling"].astype(str).eq("consecutive")
        & (frame["end_frame"].astype(int) - frame["start_frame"].astype(int)).eq(15)
        & frame["time_to_event_frames"].astype(int).between(30, 60, inclusive="both")
    )
    if not exact.all():
        raise RuntimeError(f"{name} observation or TTE contract changed")


def _authoritative_split(split_name: Literal["train", "validation", "test"]) -> pd.DataFrame:
    source_name = {"train": "split_train", "validation": "split_validation", "test": "split_test"}[split_name]
    frame = pd.read_csv(_source(source_name))
    allowed = set(REGISTRATION["sets"][split_name])
    result = frame.loc[frame["set_id"].astype(str).isin(allowed)].copy()
    result = result.sort_values("sequence_id", kind="stable").reset_index(drop=True)
    _validate_observation_frame(
        result,
        name=f"authoritative {split_name}",
        expected_sets=allowed,
        rows=int(REGISTRATION["row_counts"][split_name]),
    )
    return result


def _assert_split_parity(actual: pd.DataFrame, expected: pd.DataFrame, *, name: str) -> None:
    left = actual.loc[:, IDENTITY_COLUMNS].sort_values("sequence_id", kind="stable").reset_index(drop=True)
    right = expected.loc[:, IDENTITY_COLUMNS].sort_values("sequence_id", kind="stable").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_like=False)
    except AssertionError as error:
        raise RuntimeError(f"{name} differs from the authoritative split") from error


def _summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "positive_count": int(frame["label"].astype(int).sum()),
        "tracks": int(frame["track_id"].astype(str).nunique()),
        "videos": int(frame[["set_id", "video_id"]].astype(str).drop_duplicates().shape[0]),
        "sets": {
            str(set_id): {
                "rows": int(len(group)),
                "positive_count": int(group["label"].astype(int).sum()),
            }
            for set_id, group in frame.groupby("set_id", sort=True)
        },
    }


def assemble_development() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    contract.load_registration()
    _authenticate_parent_scripts()
    _authenticate_completion("exp037_prepare_completion")
    _authenticate_completion("exp039_confirm_completion")
    development = pd.read_csv(_source("prepared_development"))
    set06 = pd.read_csv(_source("confirmation_features"))
    columns = feature_columns()

    train = pd.concat(
        [development.loc[development["set_id"].astype(str).isin(("set01", "set02"))], set06],
        ignore_index=True,
    ).sort_values("sequence_id", kind="stable").reset_index(drop=True)
    validation = (
        development.loc[development["set_id"].astype(str).isin(("set04", "set05"))]
        .sort_values("sequence_id", kind="stable")
        .reset_index(drop=True)
    )
    _validate_observation_frame(train, name="train", expected_sets=set(REGISTRATION["sets"]["train"]), rows=2366)
    _validate_observation_frame(
        validation,
        name="validation",
        expected_sets=set(REGISTRATION["sets"]["validation"]),
        rows=3619,
    )
    if set(train["sequence_id"].astype(str)).intersection(validation["sequence_id"].astype(str)):
        raise RuntimeError("train and validation sequence IDs overlap")
    required = set(columns)
    if required.difference(train.columns) or required.difference(validation.columns):
        raise RuntimeError("development matrices are missing frozen features")
    if not np.isfinite(train.loc[:, columns].to_numpy(float)).all():
        raise RuntimeError("train features contain non-finite values")
    if not np.isfinite(validation.loc[:, columns].to_numpy(float)).all():
        raise RuntimeError("validation features contain non-finite values")

    expected_train = _authoritative_split("train")
    expected_validation = _authoritative_split("validation")
    _assert_split_parity(train, expected_train, name="train")
    _assert_split_parity(validation, expected_validation, name="validation")
    audit = {
        "train": _summary(train),
        "validation": _summary(validation),
        "feature_count": len(columns),
        "feature_columns": list(columns),
        "train_validation_overlap": 0,
        "set03_materialized": False,
    }
    return train, validation, audit


def load_anchor(split: Literal["val", "test"]) -> pd.DataFrame:
    if split not in ("val", "test"):
        raise ValueError("anchor split must be val or test")
    expected_name: Literal["validation", "test"] = "validation" if split == "val" else "test"
    expected = _authoritative_split(expected_name).set_index("sequence_id")
    parts = []
    for seed in REGISTRATION["seeds"]:
        record = REGISTRATION["anchor_predictions"][str(seed)]
        source = authenticate_parent(_resolve(record["path"]), record["sha256"])
        frame = pd.read_csv(source)
        frame = frame.loc[frame["split"].astype(str).eq(split)].copy()
        if len(frame) != len(expected) or frame["sequence_id"].duplicated().any():
            raise RuntimeError(f"Exp012 anchor {split} cohort changed for seed {seed}")
        if set(frame["sequence_id"].astype(str)) != set(expected.index.astype(str)):
            raise RuntimeError(f"Exp012 anchor {split} IDs changed for seed {seed}")
        labels = frame.set_index("sequence_id")["label"].astype(int).sort_index()
        expected_labels = expected["label"].astype(int).sort_index()
        if not labels.equals(expected_labels):
            raise RuntimeError(f"Exp012 anchor {split} labels changed for seed {seed}")
        frame = frame.rename(columns={"prob": "anchor_probability"})
        frame["seed"] = int(seed)
        parts.append(frame.loc[:, ["seed", "sequence_id", "split", "label", "anchor_probability"]])
    output = pd.concat(parts, ignore_index=True).sort_values(["seed", "sequence_id"], kind="stable").reset_index(drop=True)
    if output.duplicated(["seed", "sequence_id"]).any() or not output["anchor_probability"].between(0, 1).all():
        raise RuntimeError("Exp012 anchor probabilities are invalid")
    return output


def align_predictions(candidate: pd.DataFrame, anchor: pd.DataFrame) -> pd.DataFrame:
    required = {"seed", "sequence_id", "label", "probability"}
    if required.difference(candidate.columns):
        raise ValueError("candidate predictions are missing required columns")
    if candidate.duplicated(["seed", "sequence_id"]).any():
        raise ValueError("candidate predictions contain duplicate rows")
    order = list(zip(candidate["seed"].astype(int), candidate["sequence_id"].astype(str)))
    merged = candidate.merge(
        anchor.loc[:, ["seed", "sequence_id", "label", "anchor_probability"]],
        on=["seed", "sequence_id", "label"],
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if len(merged) != len(candidate) or list(zip(merged["seed"].astype(int), merged["sequence_id"].astype(str))) != order:
        raise RuntimeError("candidate and anchor predictions do not align exactly")
    if not merged["probability"].astype(float).between(0, 1).all():
        raise RuntimeError("candidate probabilities are invalid")
    return merged


def _authenticated_passing_gate(gate_path: Path) -> dict[str, Any]:
    gate_path = Path(gate_path)
    marker = gate_path.parent / ".validate.complete.json"
    try:
        completion = json.loads(marker.read_text(encoding="utf-8"))
        outputs = completion["outputs"]
        expected = outputs[gate_path.name]
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("set03 requires an authenticated passing validation gate") from error
    if contract.artifact_record(gate_path) != expected:
        raise RuntimeError("set03 requires an authenticated passing validation gate")
    if completion.get("status") != "complete" or completion.get("promotion_status") != "advance":
        raise RuntimeError("set03 requires an authenticated passing validation gate")
    if gate.get("status") != "advance" or not all(gate.get("checks", {}).values()):
        raise RuntimeError("set03 requires an authenticated passing validation gate")
    return gate


def _load_parent_runner():
    _authenticate_parent_scripts()
    path = _resolve(REGISTRATION["parent_scripts"]["run_experiment.py"])
    parent_dir = path.parent
    saved_modules = {name: sys.modules.pop(name, None) for name in ("feature_core", "model_core")}
    sys.path.insert(0, str(parent_dir))
    try:
        spec = importlib.util.spec_from_file_location("exp072_pinned_exp037_runner", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot import pinned Exp037 runner")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(parent_dir))
        for name, previous in saved_modules.items():
            sys.modules.pop(name, None)
            if previous is not None:
                sys.modules[name] = previous
    return module


def materialize_test_after_gate(gate_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    _authenticated_passing_gate(gate_path)
    _source("keypoints")
    _source("split_test")
    parent = _load_parent_runner()
    test, arms = parent._materialize_features(("set03",))
    columns = feature_columns()
    if tuple(arms[REGISTRATION["arm"]]) != columns:
        raise RuntimeError("set03 parent feature order changed")
    test = test.sort_values("sequence_id", kind="stable").reset_index(drop=True)
    _validate_observation_frame(test, name="test", expected_sets={"set03"}, rows=3760)
    _assert_split_parity(test, _authoritative_split("test"), name="test")
    if not np.isfinite(test.loc[:, columns].to_numpy(float)).all():
        raise RuntimeError("test features contain non-finite values")
    return test, parent._source_manifest(("set03",))
