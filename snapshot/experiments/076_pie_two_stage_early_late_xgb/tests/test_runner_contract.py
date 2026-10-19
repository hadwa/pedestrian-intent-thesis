from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_experiment.py"


def test_runner_resolves_the_exp037_contract_pinned_by_exp072() -> None:
    spec = importlib.util.spec_from_file_location("exp076_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    contract, full_columns = module._contract()

    assert len(contract["portable"]) == 182
    assert len(full_columns) == 187
    assert module.FEATURE_CONTRACT.is_file()


def test_report_markdown_does_not_require_optional_tabulate() -> None:
    spec = importlib.util.spec_from_file_location("exp076_runner_markdown", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    rendered = module.markdown_table(module.pd.DataFrame({"model": ["a"], "f1": [0.5]}))

    assert "| model | f1 |" in rendered
    assert "| a | 0.5 |" in rendered
