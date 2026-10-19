#!/usr/bin/env python3
"""Command-line entry point for the frozen Exp049 transfer evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = Path(__file__).resolve().parent
for candidate in (REPO_ROOT, SCRIPT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cache_gate import build_feature_cache, create_audit_marker  # noqa: E402
from checkpoints import authenticate_exp048_checkpoints, load_frozen_model  # noqa: E402
from contracts import EXPERIMENT_ROOT, sha256_file  # noqa: E402
from evaluation import finalize_saved_predictions, run_evaluation  # noqa: E402
from features import PoseStore, build_non_yolop_features  # noqa: E402
from jaad_protocol import (  # noqa: E402
    attach_target_labels,
    audit_manifest,
    build_observation_manifest,
    load_selected_jaad_database,
    read_official_split,
)
from yolop import YolopStore, build_yolop_cache, verify_historical_parity  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="Authenticate frozen checkpoints without JAAD access.")
    protocol = subparsers.add_parser(
        "audit-protocol", help="Audit train/validation protocol without test-label access."
    )
    protocol.add_argument("--splits", nargs="+", choices=("train", "val"), required=True)
    feature_audit = subparsers.add_parser(
        "audit-features", help="Audit label-sealed non-YOLOP test features."
    )
    feature_audit.add_argument("--exclude-yolop", action="store_true", required=True)
    parity = subparsers.add_parser(
        "verify-yolop-parity", help="Verify pinned YOLOP against historical PIE vectors."
    )
    parity.add_argument("--device", default="cuda:0")
    prepare_yolop = subparsers.add_parser(
        "prepare-yolop", help="Build the label-sealed JAAD YOLOP cache."
    )
    prepare_yolop.add_argument("--device", default="cuda:0")
    subparsers.add_parser(
        "prepare-cache", help="Build the immutable shared label-sealed A--F feature cache."
    )
    subparsers.add_parser(
        "audit", help="Authenticate all pre-inference artifacts and create AUDIT_COMPLETE."
    )
    evaluate = subparsers.add_parser(
        "evaluate", help="Run all 18 frozen zero-shot checkpoint inferences."
    )
    evaluate.add_argument("--device", default="cuda:0")
    subparsers.add_parser(
        "finalize", help="Regenerate compact reports from authenticated saved predictions only."
    )
    return parser


def status() -> dict[str, object]:
    records = authenticate_exp048_checkpoints()
    for record in records:
        load_frozen_model(record, "cpu")
    predictions_root = EXPERIMENT_ROOT / "predictions"
    return {
        "authenticated_checkpoints": len(records),
        "expected_checkpoints": 18,
        "identities": [f"{record.variant}{record.seed}" for record in records],
        "status_command_reads_jaad": False,
        "final_test_evaluation_completed": (EXPERIMENT_ROOT / "EXPERIMENT_COMPLETE").is_file(),
        "pose_extraction_complete": (EXPERIMENT_ROOT / "cache" / "POSE_EXTRACTION_COMPLETE").is_file(),
        "feature_audit_complete": (EXPERIMENT_ROOT / "cache" / "FEATURE_AUDIT_COMPLETE").is_file(),
        "yolop_extraction_complete": (EXPERIMENT_ROOT / "cache" / "YOLOP_EXTRACTION_COMPLETE").is_file(),
        "feature_cache_complete": (EXPERIMENT_ROOT / "cache" / "FEATURE_CACHE_COMPLETE").is_file(),
        "audit_complete_created": (EXPERIMENT_ROOT / "cache" / "AUDIT_COMPLETE").is_file(),
        "prediction_files": len(list(predictions_root.glob("jaad_test_*.csv"))) if predictions_root.exists() else 0,
        "experiment_complete": (EXPERIMENT_ROOT / "EXPERIMENT_COMPLETE").is_file(),
    }


def audit_protocol(splits: list[str]) -> dict[str, object]:
    reports: dict[str, object] = {}
    for split in splits:
        video_ids = read_official_split(split)
        database = load_selected_jaad_database(video_ids)
        manifest = build_observation_manifest(database, video_ids, split)
        labeled = attach_target_labels(manifest, database)
        reports[split] = audit_manifest(labeled, video_ids)
    payload = {
        "protocol": "30_contiguous_observation_frames_next_raw_frame_target",
        "splits_parsed": list(splits),
        "official_test_split_labels_parsed": False,
        "reports": reports,
    }
    output = EXPERIMENT_ROOT / "metrics" / "protocol_audit_trainval.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def audit_features(*, exclude_yolop: bool) -> dict[str, object]:
    if not exclude_yolop:
        raise ValueError("The pre-YOLOP feature audit requires --exclude-yolop.")
    video_ids = read_official_split("test")
    database = load_selected_jaad_database(video_ids, include_cross=False)
    manifest = build_observation_manifest(database, video_ids, "test")
    if "label" in manifest.columns:
        raise RuntimeError("Feature audit manifest must remain label-sealed.")
    if any(
        "cross" in pedestrian.get("behavior", {})
        for video in database.values()
        for pedestrian in video["ped_annotations"].values()
    ):
        raise RuntimeError("Feature audit database retained crossing labels.")

    pose_store = PoseStore()
    directions: Counter[str] = Counter()
    pose_valid = 0
    joint_fractions: list[float] = []
    action_missing = 0
    look_missing = 0
    nonfinite_windows = 0
    ego_speed_values: list[np.ndarray] = []
    arrays = (
        "bbox",
        "ped_action",
        "ped_look",
        "ego_speed",
        "ego_acc",
        "static_context",
        "pose",
        "biu_v2",
        "biu_v1",
    )
    for index, row in enumerate(manifest.itertuples(index=False), start=1):
        bundle = build_non_yolop_features(row, database, pose_store)
        nonfinite_windows += int(
            any(not np.isfinite(np.asarray(getattr(bundle, name))).all() for name in arrays)
        )
        pose_valid += int(bundle.pose_valid)
        joint_fractions.append(float(bundle.valid_joint_fraction))
        directions[bundle.biu_v2_direction_source] += 1
        action_missing += int(bundle.action_missing_frames)
        look_missing += int(bundle.look_missing_frames)
        ego_speed_values.append(bundle.ego_speed.reshape(-1))
        if index % 1000 == 0:
            print(f"feature_audit_progress={index}/{len(manifest)}", file=sys.stderr, flush=True)

    joint = np.asarray(joint_fractions, dtype=np.float64)
    ego = np.concatenate(ego_speed_values).astype(np.float64)
    payload = {
        "phase": "label_sealed_non_yolop_feature_audit",
        "windows": int(len(manifest)),
        "official_test_videos": int(len(video_ids)),
        "target_labels_attached": False,
        "cross_values_present_in_database": False,
        "probabilities_computed": False,
        "yolop_included": False,
        "pose_valid_windows": int(pose_valid),
        "pose_invalid_windows": int(len(manifest) - pose_valid),
        "valid_joint_fraction_mean": float(joint.mean()),
        "valid_joint_fraction_median": float(np.median(joint)),
        "all_zero_pose_windows": int(np.sum(joint == 0.0)),
        "nonfinite_feature_windows": int(nonfinite_windows),
        "action_missing_observation_frames": int(action_missing),
        "look_missing_observation_frames": int(look_missing),
        "biu_v2_direction_sources": dict(sorted(directions.items())),
        "biu_v2_bbox_direction_fallback_windows": int(directions.get("bbox_center_x", 0)),
        "ego_speed_min": float(ego.min()),
        "ego_speed_max": float(ego.max()),
        "ego_speed_mean": float(ego.mean()),
    }
    output = EXPERIMENT_ROOT / "metrics" / "feature_audit_test.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def verify_yolop(device: str) -> dict[str, object]:
    cases = verify_historical_parity(device)
    payload = {
        "device": device,
        "atol": 1e-6,
        "rtol": 0.0,
        "cases": cases,
        "all_passed": all(bool(case["passed_atol_1e_6"]) for case in cases),
    }
    output = EXPERIMENT_ROOT / "metrics" / "yolop_historical_parity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def prepare_yolop(device: str) -> dict[str, object]:
    video_ids = read_official_split("test")
    database = load_selected_jaad_database(video_ids, include_cross=False)
    manifest = build_observation_manifest(database, video_ids, "test")
    if "label" in manifest.columns:
        raise RuntimeError("YOLOP preparation must remain label-sealed.")
    if any(
        "cross" in pedestrian.get("behavior", {})
        for video in database.values()
        for pedestrian in video["ped_annotations"].values()
    ):
        raise RuntimeError("YOLOP preparation database retained crossing labels.")
    cache_root = EXPERIMENT_ROOT / "cache" / "yolop"
    payload = build_yolop_cache(manifest, database, cache_root, device=device)
    report_source = cache_root / "yolop_extraction_report.json"
    report_output = EXPERIMENT_ROOT / "metrics" / "yolop_extraction_report.json"
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_bytes(report_source.read_bytes())
    parts = sorted((cache_root / "yolop_parts").glob("*.npz"))
    if len(parts) != int(payload["videos"]):
        raise RuntimeError("YOLOP cache part count does not match the audited video count.")
    marker = {
        "phase": "yolop_extraction",
        "complete": True,
        "target_labels_attached": False,
        "probabilities_computed": False,
        "report_path": str(report_output),
        "report_sha256": sha256_file(report_output),
        "part_count": len(parts),
        "part_sha256": {part.name: sha256_file(part) for part in parts},
    }
    marker_path = EXPERIMENT_ROOT / "cache" / "YOLOP_EXTRACTION_COMPLETE"
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _authenticate_yolop_marker() -> dict[str, object]:
    marker_path = EXPERIMENT_ROOT / "cache" / "YOLOP_EXTRACTION_COMPLETE"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("phase") != "yolop_extraction" or marker.get("complete") is not True:
        raise ValueError("YOLOP extraction marker is not complete.")
    if marker.get("target_labels_attached") is not False or marker.get("probabilities_computed") is not False:
        raise ValueError("YOLOP extraction marker violates the label/probability seal.")
    report = Path(str(marker["report_path"]))
    if sha256_file(report) != str(marker["report_sha256"]):
        raise ValueError("YOLOP extraction report hash mismatch.")
    parts_root = EXPERIMENT_ROOT / "cache" / "yolop" / "yolop_parts"
    actual_parts = sorted(parts_root.glob("*.npz"))
    expected = marker.get("part_sha256", {})
    if len(actual_parts) != int(marker["part_count"]) or set(expected) != {
        path.name for path in actual_parts
    }:
        raise ValueError("YOLOP cache part population drifted.")
    for path in actual_parts:
        if sha256_file(path) != str(expected[path.name]):
            raise ValueError(f"YOLOP cache part hash mismatch: {path}")
    return marker


def prepare_feature_cache() -> dict[str, object]:
    _authenticate_yolop_marker()
    video_ids = read_official_split("test")
    database = load_selected_jaad_database(video_ids, include_cross=False)
    manifest = build_observation_manifest(database, video_ids, "test")
    if "label" in manifest.columns:
        raise RuntimeError("Feature cache manifest must remain label-sealed.")
    pose_store = PoseStore()
    yolop_store = YolopStore(EXPERIMENT_ROOT / "cache" / "yolop" / "yolop_parts")
    cache_root = EXPERIMENT_ROOT / "cache" / "test_features"
    payload = build_feature_cache(manifest, database, pose_store, yolop_store, cache_root)
    marker = {
        "phase": "feature_cache",
        "complete": True,
        "contains_labels": False,
        "probabilities_computed": False,
        "metadata_path": str(cache_root / "metadata.json"),
        "metadata_sha256": sha256_file(cache_root / "metadata.json"),
    }
    marker_path = EXPERIMENT_ROOT / "cache" / "FEATURE_CACHE_COMPLETE"
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def complete_audit() -> dict[str, object]:
    video_ids = read_official_split("test")
    database = load_selected_jaad_database(video_ids, include_cross=False)
    manifest = build_observation_manifest(database, video_ids, "test")
    if "label" in manifest.columns:
        raise RuntimeError("Pre-inference audit manifest must remain label-sealed.")
    records = authenticate_exp048_checkpoints()
    return create_audit_marker(
        EXPERIMENT_ROOT, manifest["sequence_id"].astype(str).tolist(), records
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "status":
        print(json.dumps(status(), indent=2, sort_keys=True))
    elif args.command == "audit-protocol":
        print(json.dumps(audit_protocol(args.splits), indent=2, sort_keys=True))
    elif args.command == "audit-features":
        print(
            json.dumps(
                audit_features(exclude_yolop=args.exclude_yolop),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "verify-yolop-parity":
        print(json.dumps(verify_yolop(args.device), indent=2, sort_keys=True))
    elif args.command == "prepare-yolop":
        print(json.dumps(prepare_yolop(args.device), indent=2, sort_keys=True))
    elif args.command == "prepare-cache":
        print(json.dumps(prepare_feature_cache(), indent=2, sort_keys=True))
    elif args.command == "audit":
        print(json.dumps(complete_audit(), indent=2, sort_keys=True))
    elif args.command == "evaluate":
        print(json.dumps(run_evaluation(args.device), indent=2, sort_keys=True))
    elif args.command == "finalize":
        print(json.dumps(finalize_saved_predictions(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
