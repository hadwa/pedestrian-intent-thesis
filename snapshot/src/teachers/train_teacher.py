"""Train one teacher configuration through the MSLSTM adapter."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from src.teachers.mslstm_adapter import run_mslstm_training


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one MSLSTM teacher config.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = run_mslstm_training(args.config)
    print(f"Teacher checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
