"""Local, test-sealed preflight for the frozen Exp114 sources and contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
EXP113 = REPO / "experiments/113_psi_native_biu_xgb"
EXP037 = REPO / "experiments/037_pie_joint_biu_temporal_ego"
POSE = REPO / "data/psi_vitpose_s_all_tracks.jsonl"
EXPECTED_POSE_SHA256 = "31d8f45a7f8b61803b2dde8f981ce012086ac9905455d7aaac743bc1ec2a3fc7"
MISSING_PSI2_SPEED_VIDEOS = ("video_0032", "video_0055", "video_0093")
SEEDS = (42, 43, 44)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def load_feature_contract() -> dict[str, list[str]]:
    spec = importlib.util.spec_from_file_location(
        "psi114_feature_core", EXP037 / "scripts/feature_core.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen Exp037 feature code")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    definitions = REPO / "experiments/012_pie_biu_v2_30_60_benchmark/metrics/pie_biu_v2_feature_set_definitions.csv"
    source = module.load_exp012_feature_contract(definitions)
    arms = module.build_arm_columns(source)
    source_columns = list(arms["joint_temporal_ego_xgb"])
    road_channels = (
        "road_foot_inside", "road_signed_boundary_distance_h",
        "road_nearest_boundary_dx_h", "road_nearest_boundary_dy_h",
        "road_boundary_tangent_x", "road_boundary_tangent_y",
        "road_boundary_valid", "lane_nearest_distance_h",
        "lane_nearest_dx_h", "lane_nearest_dy_h", "lane_tangent_x",
        "lane_tangent_y", "lane_geometry_valid", "foot_velocity_x_h",
        "foot_velocity_y_h", "foot_speed_h", "motion_parallel_to_road",
        "motion_toward_drivable_area", "motion_parallel_to_lane",
        "road_signed_distance_change_h",
    )
    road_columns = [
        f"road__{channel}__{stat}"
        for channel in road_channels
        for stat in ("early_mean", "late_mean", "late_minus_early")
    ]
    assert len(source_columns) == 187 and len(set(source_columns)) == 187
    assert len(road_columns) == 60 and not set(source_columns) & set(road_columns)
    return {"S187": source_columns, "R247": source_columns + road_columns}


def main() -> int:
    # Deliberately read only Exp113 train/validation caches; there is no test path.
    rows = {
        split: pd.read_csv(EXP113 / f"cache/{split}/rows.csv")
        for split in ("train", "val")
    }
    expected = {"train": 26880, "val": 14922}
    for split, frame in rows.items():
        assert len(frame) == expected[split] and frame["sequence_id"].is_unique
    assert sha256(POSE) == EXPECTED_POSE_SHA256
    live = json.loads((ROOT / "live_zip_directory_audit.json").read_text())
    assert live["test_members_read"] is False
    assert tuple(live["cv_annotation"]["missing_selected_videos"]) == MISSING_PSI2_SPEED_VIDEOS
    affected = {
        split: {
            video: int((frame["video_id"].astype(str) == video).sum())
            for video in MISSING_PSI2_SPEED_VIDEOS
        }
        for split, frame in rows.items()
    }
    assert sum(affected["train"].values()) == 585
    assert sum(affected["val"].values()) == 0
    contract = load_feature_contract()
    (ROOT / "feature_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "status": "blocked",
        "reason": "official PSI2 frame speed is unavailable for full frozen cohort",
        "development_only": True,
        "test_accessed": False,
        "cohort_rows": expected,
        "missing_speed_videos": list(MISSING_PSI2_SPEED_VIDEOS),
        "affected_windows": affected,
        "affected_train_windows_total": 585,
        "feature_dimensions": {name: len(columns) for name, columns in contract.items()},
        "planned_arms": ["S187", "R247"],
        "planned_seeds": list(SEEDS),
        "planned_fits": 6,
        "fits_run": 0,
        "predictions_run": 0,
        "videos_downloaded": 0,
        "cv_annotations_downloaded": 0,
        "road_extraction_run": False,
        "pose_reextraction_run": False,
        "sources": [
            record(EXP113 / "freeze.json"),
            record(EXP113 / "feature_contract.json"),
            record(EXP113 / "cache/train/rows.csv"),
            record(EXP113 / "cache/val/rows.csv"),
            record(EXP037 / "scripts/feature_core.py"),
            record(REPO / "experiments/012_pie_biu_v2_30_60_benchmark/metrics/pie_biu_v2_feature_set_definitions.csv"),
            record(REPO / "experiments/075_pie_twinlitenetplus_road_context/checkpoints/large.pth"),
            record(REPO / "experiments/075_pie_twinlitenetplus_road_context/scripts/twinlite_adapter.py"),
            record(REPO / "experiments/063_pie_corrected_yolop_road_relative_context/scripts/representations.py"),
            record(POSE),
            record(ROOT / "DESIGN.md"),
            record(ROOT / "live_zip_directory_audit.json"),
        ],
    }
    (ROOT / "metrics/source_preflight.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    blocker = {
        "status": "blocked_before_preparation",
        "source_preflight": record(ROOT / "metrics/source_preflight.json"),
        "design": record(ROOT / "DESIGN.md"),
        "test_accessed": False,
        "fits_run": 0,
        "resolution": "supply and authenticate exact official frame speed for the full PSI2 cohort",
    }
    (ROOT / "BLOCKED.json").write_text(
        json.dumps(blocker, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print("BLOCKED: exact official PSI2 speed is absent for 585 training windows", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
