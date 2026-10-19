"""Immutable contract and artifact helpers for Exp072."""

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
PREREGISTRATION_PATH = EXPERIMENT_ROOT / "pre_registration.json"
EXPECTED_REGISTRATION_SHA256 = "0c10511f8af2fc13c03d048850896b56dae86a6cdeb270c749fa07a515083d94"

EXPECTED_REGISTRATION: dict[str, Any] = json.loads(
    (EXPERIMENT_ROOT / "pre_registration.json").read_text(encoding="utf-8")
)


def load_registration(path: Path = PREREGISTRATION_PATH) -> dict[str, Any]:
    path = Path(path)
    if path.resolve() == PREREGISTRATION_PATH.resolve() and sha256_file(path) != EXPECTED_REGISTRATION_SHA256:
        raise ValueError("Exp072 preregistration hash differs from the frozen contract")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value != EXPECTED_REGISTRATION:
        raise ValueError("Exp072 preregistration differs from the frozen contract")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path) -> dict[str, int | str]:
    resolved = Path(path).resolve(strict=True)
    return {"sha256": sha256_file(resolved), "size_bytes": resolved.stat().st_size}


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_bytes(Path(path), payload.encode("utf-8"))


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_bytes(Path(path), frame.to_csv(index=False).encode("utf-8"))


def atomic_text(path: Path, value: str) -> None:
    _atomic_bytes(Path(path), value.encode("utf-8"))
