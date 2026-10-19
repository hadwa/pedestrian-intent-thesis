"""Frozen scientific and filesystem contracts for Exp049."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


EXPERIMENT_NUMBER = 49
OBSERVATION_LENGTH = 30
PREDICTION_OFFSET = 1
FIXED_THRESHOLD = 0.5
VARIANTS = ("A", "B", "C", "D", "E", "F")
SEEDS = (42, 43, 44)

VARIANT_STREAMS = {
    "A": (
        "bbox",
        "ped_action",
        "ped_look",
        "ego_speed",
        "ego_acc",
        "static_context",
        "yolop",
    ),
    "B": ("bbox", "ped_action", "ped_look", "ego_speed", "ego_acc", "yolop"),
    "C": ("bbox", "ego_speed", "ego_acc", "yolop"),
    "D": ("bbox", "ego_speed", "ego_acc", "yolop", "pose"),
    "E": ("bbox", "ego_speed", "ego_acc", "yolop", "biu_v2"),
    "F": ("bbox", "ego_speed", "ego_acc", "yolop", "biu_v1"),
}

EXPECTED_PARAMETERS = {
    "A": 15_436_304,
    "B": 13_218_062,
    "C": 8_871_690,
    "D": 11_112_460,
    "E": 8_945_931,
    "F": 8_947_083,
}

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "049_exp048_to_jaad_zero_shot"
EXP048_ROOT = REPO_ROOT / "experiments" / "048_mslstm_practical_behavior_replacement"
EXP048_CHECKPOINT_ROOT = EXP048_ROOT / "checkpoints"


@dataclass(frozen=True)
class VariantSpec:
    name: str
    streams: tuple[str, ...]


@dataclass(frozen=True)
class CheckpointRecord:
    variant: str
    seed: int
    checkpoint_path: Path
    marker_path: Path
    checkpoint_sha256: str
    total_parameters: int


VARIANT_SPECS = tuple(VariantSpec(name, VARIANT_STREAMS[name]) for name in VARIANTS)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
