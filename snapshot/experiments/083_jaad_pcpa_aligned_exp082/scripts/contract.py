"""Immutable Exp083 registration and authenticated phase markers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOT = Path(__file__).resolve().parents[3]
REGISTRATION_PATH = EXPERIMENT_ROOT / "pre_registration.json"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


EXPECTED_REGISTRATION = _read_json(REGISTRATION_PATH)


def load_registration(path: Path | str | None = None) -> dict[str, Any]:
    registration = _read_json(REGISTRATION_PATH if path is None else Path(path))
    if registration != EXPECTED_REGISTRATION:
        raise ValueError("Exp083 registration differs from the frozen contract")
    return registration


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_record(path: Path | str) -> dict[str, Any]:
    artifact = Path(path).resolve()
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    return {
        "path": str(artifact),
        "sha256": sha256_file(artifact),
        "size_bytes": artifact.stat().st_size,
    }


def resolve_source(name: str) -> Path:
    source = load_registration()["sources"][name]
    path = Path(source["path"])
    if not path.is_absolute():
        path = ACTIVE_ROOT / path
    actual = artifact_record(path)
    if actual["sha256"] != source["sha256"] or actual["size_bytes"] != source["size_bytes"]:
        raise RuntimeError(f"Exp083 source authentication failed: {name}")
    return path.resolve()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path | str, value: str) -> None:
    _atomic_write(Path(path), value.encode("utf-8"))


def atomic_write_json(path: Path | str, value: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def phase_marker_path(root: Path | str, phase: str) -> Path:
    return Path(root) / "metrics" / f".{phase}.complete.json"


def publish_phase(
    root: Path | str,
    phase: str,
    outputs: Iterable[Path | str],
    *,
    test_accessed: bool,
    **metadata: Any,
) -> Path:
    payload = {
        "phase": phase,
        "status": "complete",
        "test_accessed": bool(test_accessed),
        "outputs": [artifact_record(path) for path in outputs],
        **metadata,
    }
    marker = phase_marker_path(root, phase)
    atomic_write_json(marker, payload)
    return marker


def authenticate_phase(root: Path | str, phase: str) -> dict[str, Any]:
    try:
        payload = _read_json(phase_marker_path(root, phase))
        if payload.get("phase") != phase or payload.get("status") != "complete":
            raise RuntimeError("invalid marker state")
        for expected in payload["outputs"]:
            actual = artifact_record(expected["path"])
            if actual["sha256"] != expected["sha256"] or actual["size_bytes"] != expected["size_bytes"]:
                raise RuntimeError(f"changed output: {expected['path']}")
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"Exp083 phase authentication failed: {phase}") from exc
    return payload
