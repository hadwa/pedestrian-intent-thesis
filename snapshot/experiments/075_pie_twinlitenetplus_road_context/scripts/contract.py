"""Immutable registration and atomic artifact helpers for Exp075."""

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
        raise ValueError("Exp075 registration differs from the frozen contract")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path) -> dict[str, int | str]:
    source = Path(path)
    return {"sha256": sha256_file(source), "size_bytes": source.stat().st_size}


def resolve_source(name: str) -> Path:
    record = EXPECTED_REGISTRATION["sources"][name]
    path = ACTIVE_ROOT / record["path"]
    expected = {"sha256": record["sha256"], "size_bytes": record["size_bytes"]}
    if not path.is_file() or artifact_record(path) != expected:
        raise RuntimeError(f"Exp075 source authentication failed: {name}")
    return path


def _atomic_bytes(path: Path, value: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(Path(path), (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def atomic_csv(path: Path, value: pd.DataFrame) -> None:
    _atomic_bytes(Path(path), value.to_csv(index=False).encode())


def atomic_text(path: Path, value: str) -> None:
    _atomic_bytes(Path(path), value.encode())

