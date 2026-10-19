#!/usr/bin/env python3
"""Resume Exp047 evaluation after the frozen runner's missing import failure."""

from __future__ import annotations

from pathlib import Path

import run_experiment as runner
from exp047_core import TEST_SETS


def prepare_runner():
    """Inject the frozen constant and bind evaluation provenance to this launcher."""
    runner.TEST_SETS = TEST_SETS
    runner.__file__ = str(Path(__file__).resolve())
    return runner


def main() -> None:
    prepared = prepare_runner()
    prepared.main()


if __name__ == "__main__":
    main()
