"""Exact interpreter and package contract for Exp062 public execution."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSIONS = {
    "python": "3.10.20",
    "numpy": "1.26.3",
    "scipy": "1.13.1",
    "scikit_learn": "1.5.2",
    "joblib": "1.5.1",
    "threadpoolctl": "3.6.0",
    "pandas": "2.3.3",
    "xgboost": "3.2.0",
    "torch": "2.11.0+cu128",
    "pytest": "9.1.1",
}
PACKAGE_DISTRIBUTIONS = {
    "numpy": "numpy",
    "scipy": "scipy",
    "scikit_learn": "scikit-learn",
    "joblib": "joblib",
    "threadpoolctl": "threadpoolctl",
    "pandas": "pandas",
    "xgboost": "xgboost",
    "torch": "torch",
    "pytest": "pytest",
}


class RuntimeContractError(RuntimeError):
    """Raised before CLI dispatch when the Exp062 runtime has drifted."""


@dataclass(frozen=True)
class RuntimeSnapshot:
    python: str
    executable: str
    prefix: str
    numpy: str
    scipy: str
    scikit_learn: str
    joblib: str
    threadpoolctl: str
    pandas: str
    xgboost: str
    torch: str
    pytest: str


def capture_runtime_snapshot() -> RuntimeSnapshot:
    """Read package metadata without importing experiment data/model modules."""
    packages = {
        field: version(distribution)
        for field, distribution in PACKAGE_DISTRIBUTIONS.items()
    }
    return RuntimeSnapshot(
        python=".".join(str(value) for value in sys.version_info[:3]),
        executable=sys.executable,
        prefix=sys.prefix,
        **packages,
    )


def assert_runtime_contract(snapshot: RuntimeSnapshot | None = None) -> None:
    """Reject any runtime other than the experiment-local pinned environment."""
    observed = capture_runtime_snapshot() if snapshot is None else snapshot
    expected_prefix = EXPERIMENT_ROOT / ".venv"
    expected_executable = expected_prefix / "bin" / "python"
    mismatches: list[str] = []

    for field, expected in EXPECTED_VERSIONS.items():
        actual = getattr(observed, field)
        if actual != expected:
            label = "scikit-learn" if field == "scikit_learn" else field
            mismatches.append(f"{label}: expected {expected}, observed {actual}")
    if Path(observed.executable).resolve() != expected_executable.resolve():
        mismatches.append(
            f"executable: expected {expected_executable}, observed {observed.executable}"
        )
    if Path(observed.prefix).resolve() != expected_prefix.resolve():
        mismatches.append(f"prefix: expected {expected_prefix}, observed {observed.prefix}")

    if mismatches:
        raise RuntimeContractError("Exp062 runtime mismatch: " + "; ".join(mismatches))


__all__ = [
    "EXPECTED_VERSIONS",
    "RuntimeContractError",
    "RuntimeSnapshot",
    "assert_runtime_contract",
    "capture_runtime_snapshot",
]
