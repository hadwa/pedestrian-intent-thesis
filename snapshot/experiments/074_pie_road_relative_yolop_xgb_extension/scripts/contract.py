"""Immutable registration and atomic artifact helpers for Exp074."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOT = Path(__file__).resolve().parents[3]
REGISTRATION_PATH = EXPERIMENT_ROOT / "pre_registration.json"
EXPECTED_REGISTRATION = json.loads(REGISTRATION_PATH.read_text(encoding="utf-8"))


def load_registration(path: Path = REGISTRATION_PATH) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value != EXPECTED_REGISTRATION:
        raise ValueError("Exp074 registration differs from the frozen contract")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path) -> dict[str, int | str]:
    return {"sha256": sha256_file(path), "size_bytes": Path(path).stat().st_size}


def resolve_source(name: str) -> Path:
    record = EXPECTED_REGISTRATION["sources"][name]
    path = ACTIVE_ROOT / record["path"]
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"Exp074 source authentication failed: {name}")
    return path


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(Path(path), (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def atomic_csv(path: Path, value: pd.DataFrame) -> None:
    _atomic_bytes(Path(path), value.to_csv(index=False).encode())


def atomic_text(path: Path, value: str) -> None:
    _atomic_bytes(Path(path), value.encode())
