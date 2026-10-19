"""Fail-closed runtime contracts for the Exp062 execution boundary."""

from __future__ import annotations

import importlib
import sys
from dataclasses import replace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _runtime_module():
    return importlib.import_module("scripts.runtime")


def test_checked_runtime_matches_the_experiment_contract() -> None:
    """Catches an incomplete lock or a runtime snapshot that omits a dependency."""
    runtime = _runtime_module()

    snapshot = runtime.capture_runtime_snapshot()

    assert snapshot.python == "3.10.20"
    assert snapshot.numpy == "1.26.3"
    assert snapshot.scipy == "1.13.1"
    assert snapshot.scikit_learn == "1.5.2"
    assert snapshot.joblib == "1.5.1"
    assert snapshot.threadpoolctl == "3.6.0"
    assert snapshot.pandas == "2.3.3"
    assert snapshot.xgboost == "3.2.0"
    assert snapshot.torch == "2.11.0+cu128"
    assert snapshot.pytest == "9.1.1"
    runtime.assert_runtime_contract(snapshot)


@pytest.mark.parametrize(
    ("field", "wrong_value", "message"),
    [
        ("scikit_learn", "1.7.2", "scikit-learn"),
        ("numpy", "2.2.6", "numpy"),
        ("python", "3.11.0", "python"),
        ("executable", "/usr/bin/python3", "executable"),
        ("prefix", "/tmp/wrong-runtime", "prefix"),
    ],
)
def test_runtime_contract_rejects_scientific_or_interpreter_drift(
    field: str, wrong_value: str, message: str
) -> None:
    """Catches acceptance of every explicitly prohibited runtime mismatch."""
    runtime = _runtime_module()
    snapshot = replace(runtime.capture_runtime_snapshot(), **{field: wrong_value})

    with pytest.raises(runtime.RuntimeContractError, match=message):
        runtime.assert_runtime_contract(snapshot)


def test_public_cli_has_no_runtime_bypass_flag() -> None:
    """Catches a production escape hatch around the runtime contract."""
    runner = importlib.import_module("scripts.runner")

    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(["--skip-runtime-check", "status"])


def test_runtime_rejection_precedes_phase_source_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches source authorization or dispatch before the runtime is accepted."""
    runner = importlib.import_module("scripts.runner")
    source_accessed = False

    def source_sentinel(*_args: object, **_kwargs: object) -> None:
        nonlocal source_accessed
        source_accessed = True
        raise AssertionError("source sentinel fired before runtime rejection")

    def reject_runtime() -> None:
        raise RuntimeError("injected runtime mismatch")

    monkeypatch.setattr(runner, "prepare", source_sentinel)

    with pytest.raises(RuntimeError, match="injected runtime mismatch"):
        runner.main(["prepare"], _runtime_checker=reject_runtime)

    assert not source_accessed


def test_main_uses_real_runtime_checker_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches a runner that exposes a checker but does not enforce it."""
    runner = importlib.import_module("scripts.runner")

    def reject_runtime() -> None:
        raise RuntimeError("default runtime checker called")

    monkeypatch.setattr(runner, "assert_runtime_contract", reject_runtime)
    with pytest.raises(RuntimeError, match="default runtime checker called"):
        runner.main(["status"])
