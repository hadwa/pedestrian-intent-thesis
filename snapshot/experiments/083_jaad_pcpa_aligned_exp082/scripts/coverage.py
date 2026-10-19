"""Exact-source coverage checks for Exp083."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
JAAD_IMAGES_ROOT = Path("/home/abdelrahman/Hadwa/datasets/JAAD/images")
POSE_SOURCES = (
    {
        "path": REPO_ROOT
        / "experiments/049_exp048_to_jaad_zero_shot/cache/"
        "jaad_keypoints_vitpose_s_exp049_extended.jsonl",
        "sha256": "9eae258cd7fed3b1278aa86f6ab5c508fb759a24c7912d5c57677506ae86e4db",
    },
    {
        "path": REPO_ROOT
        / "experiments/079_jaad_api_protocol_two_stage_xgb/cache/"
        "jaad_api_protocol_missing_pose.jsonl",
        "sha256": "d0c0fd771eff1dadd7c499a680cef704409e6a07a72ca31f3f3b39d6a86fe038",
    },
)
ROAD_CACHE_ROOTS = (
    REPO_ROOT
    / "experiments/081_jaad_ego_compensated_road_motion/cache/development_parts",
    REPO_ROOT
    / "experiments/082_jaad_compensated_road_followup_evaluation/cache/test_parts",
)
COHORTS = ("jaad_beh", "jaad_all")
SPLITS = ("train", "val", "test")


def _load_sibling(filename: str, name: str):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_sibling("contract.py", "exp083_contract_for_coverage")


def _joined_unique(values: Iterable[object]) -> str:
    return "|".join(sorted({str(value) for value in values}))


def required_frame_keys(
    manifests: Mapping[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    """Expand manifests and deduplicate exact pedestrian-frame source keys."""
    rows = []
    for (cohort, split), manifest in manifests.items():
        required = {
            "sequence_id",
            "video_id",
            "ped_id",
            "observed_frames_json",
        }
        missing = required.difference(manifest.columns)
        if missing:
            raise RuntimeError(f"manifest missing columns: {sorted(missing)}")
        for row in manifest.itertuples(index=False):
            frame_ids = json.loads(row.observed_frames_json)
            for frame_id in frame_ids:
                rows.append(
                    {
                        "video_id": str(row.video_id),
                        "ped_id": str(row.ped_id),
                        "frame_id": int(frame_id),
                        "cohort": str(cohort),
                        "split": str(split),
                        "sequence_id": str(row.sequence_id),
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=[
                "video_id",
                "ped_id",
                "frame_id",
                "cohorts",
                "splits",
                "sequence_count",
            ]
        )
    expanded = pd.DataFrame(rows)
    return (
        expanded.groupby(["video_id", "ped_id", "frame_id"], sort=True)
        .agg(
            cohorts=("cohort", _joined_unique),
            splits=("split", _joined_unique),
            sequence_count=("sequence_id", "nunique"),
        )
        .reset_index()
    )


def exact_missing_keys(
    required: pd.DataFrame,
    available: pd.DataFrame,
    *,
    key_columns: list[str],
) -> pd.DataFrame:
    """Return required keys absent from available; no fuzzy matching is allowed."""
    for name, frame in (("required", required), ("available", available)):
        missing = set(key_columns).difference(frame.columns)
        if missing:
            raise RuntimeError(f"{name} keys missing columns: {sorted(missing)}")
    available_keys = available[key_columns].drop_duplicates()
    compared = required[key_columns].drop_duplicates().merge(
        available_keys.assign(_present=True),
        on=key_columns,
        how="left",
        sort=False,
    )
    return compared.loc[compared["_present"].isna(), key_columns].reset_index(
        drop=True
    )


def coverage_gate_status(missing: pd.DataFrame) -> str:
    if len(missing):
        return "blocked_new_extraction_required"
    return "coverage_complete"


def require_protocol_audit(root: Path | str) -> dict:
    """Authenticate the frozen protocol outputs before source inspection."""
    try:
        return contract.authenticate_phase(Path(root), "protocol-audit")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "authenticated protocol-audit marker required before coverage audit"
        ) from exc


def load_manifests(root: Path | str) -> dict[tuple[str, str], pd.DataFrame]:
    root = Path(root)
    manifests = {}
    for cohort in COHORTS:
        for split in SPLITS:
            path = root / "manifests" / f"{cohort}_{split}.csv"
            manifests[(cohort, split)] = pd.read_csv(path)
    return manifests


def load_pose_inventory() -> tuple[pd.DataFrame, list[dict]]:
    """Load only exact pose identities while validating frozen source bytes."""
    rows = []
    records = []
    for source in POSE_SOURCES:
        path = Path(source["path"])
        actual_sha = contract.sha256_file(path)
        if actual_sha != source["sha256"]:
            raise RuntimeError(f"pose source authentication failed: {path}")
        source_rows = 0
        invalid_coco17_rows = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                item = json.loads(line)
                if item.get("set") != "jaad":
                    continue
                keypoints = item.get("keypoints")
                confidences = item.get("kp_conf")
                valid = (
                    isinstance(keypoints, list)
                    and len(keypoints) == 17
                    and all(isinstance(point, list) and len(point) == 2 for point in keypoints)
                    and isinstance(confidences, list)
                    and len(confidences) == 17
                )
                if not valid:
                    invalid_coco17_rows += 1
                rows.append(
                    {
                        "video_id": str(item["video"]),
                        "ped_id": str(item["pid"]),
                        "frame_id": int(item["frame_idx"]),
                        "pose_source": str(path),
                        "pose_line_number": line_number,
                        "coco17_valid": valid,
                    }
                )
                source_rows += 1
        records.append(
            {
                "path": str(path),
                "sha256": actual_sha,
                "size_bytes": path.stat().st_size,
                "rows": source_rows,
                "invalid_coco17_rows": invalid_coco17_rows,
            }
        )
    inventory = pd.DataFrame(rows)
    duplicate = inventory.duplicated(["video_id", "ped_id", "frame_id"], keep=False)
    if duplicate.any():
        sample = inventory.loc[duplicate].head(3).to_dict(orient="records")
        raise RuntimeError(f"pose source identities overlap: {sample}")
    if not inventory["coco17_valid"].all():
        raise RuntimeError("pose inventory contains non-COCO17 records")
    return inventory, records


def load_road_sequence_inventory() -> tuple[pd.DataFrame, list[dict]]:
    """Read only cached exact sequence identities, not feature tensors."""
    rows = []
    sources = []
    for root in ROAD_CACHE_ROOTS:
        for path in sorted(root.rglob("*.npz")):
            with np.load(path, allow_pickle=False) as archive:
                if "sequence_id" not in archive.files:
                    raise RuntimeError(f"road cache lacks sequence_id: {path}")
                sequence_ids = archive["sequence_id"].astype(str).tolist()
            rows.extend(
                {"source_sequence_id": sequence_id, "cache_path": str(path)}
                for sequence_id in sequence_ids
            )
            sources.append(
                {
                    "path": str(path),
                    "sha256": contract.sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "sequence_rows": len(sequence_ids),
                }
            )
    inventory = pd.DataFrame(rows)
    if inventory["source_sequence_id"].duplicated().any():
        raise RuntimeError("road cache sequence IDs are not unique")
    return inventory, sources


def _source_sequence_ids(manifest: pd.DataFrame) -> pd.Series:
    return (
        manifest["video_id"].astype(str)
        + ":"
        + manifest["ped_id"].astype(str)
        + ":"
        + manifest["start_frame"].astype(int).astype(str)
        + ":"
        + manifest["end_frame"].astype(int).astype(str)
        + ":"
        + manifest["event_frame"].astype(int).astype(str)
        + ":"
        + manifest["label"].astype(int).astype(str)
    )


def _coverage_row(
    cohort: str,
    split: str,
    input_name: str,
    unit: str,
    required: int,
    missing: int,
) -> dict:
    present = required - missing
    return {
        "cohort": cohort,
        "split": split,
        "input": input_name,
        "unit": unit,
        "required": required,
        "present": present,
        "missing": missing,
        "coverage_rate": present / required if required else 1.0,
    }


def _missing_records(
    missing: pd.DataFrame,
    *,
    cohort: str,
    split: str,
    input_name: str,
    unit: str,
) -> list[dict]:
    records = []
    for record in missing.to_dict(orient="records"):
        records.append(
            {
                "cohort": cohort,
                "split": split,
                "input": input_name,
                "unit": unit,
                **record,
            }
        )
    return records


def audit_pose_coverage(
    required: pd.DataFrame,
    available: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    keys = ["video_id", "ped_id", "frame_id"]
    missing = exact_missing_keys(required, available, key_columns=keys)
    return {
        "required": len(required[keys].drop_duplicates()),
        "missing": len(missing),
    }, missing


def audit_rgb_coverage(required: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    frame_keys = required[["video_id", "frame_id"]].drop_duplicates().copy()
    paths = frame_keys.apply(
        lambda row: JAAD_IMAGES_ROOT
        / str(row["video_id"])
        / f"{int(row['frame_id']):05d}.png",
        axis=1,
    )
    missing = frame_keys.loc[~paths.map(Path.is_file)].reset_index(drop=True)
    return {"required": len(frame_keys), "missing": len(missing)}, missing


def audit_context_coverage(
    manifest: pd.DataFrame,
    available: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    required = manifest[["sequence_id"]].copy()
    required["source_sequence_id"] = _source_sequence_ids(manifest)
    available_ids = set(available["source_sequence_id"].astype(str))
    missing = required.loc[
        ~required["source_sequence_id"].isin(available_ids)
    ].reset_index(drop=True)
    return {"required": len(required), "missing": len(missing)}, missing


def audit_rgb_and_context_coverage(
    required: pd.DataFrame,
    manifest: pd.DataFrame,
    road_available: pd.DataFrame,
) -> dict[str, tuple[dict, pd.DataFrame]]:
    return {
        "rgb": audit_rgb_coverage(required),
        "context": audit_context_coverage(manifest, road_available),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact table without pandas' optional tabulate dependency."""
    columns = [str(column) for column in frame.columns]

    def render(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def _render_audit(
    summary: pd.DataFrame,
    pose_sources: list[dict],
    road_sources: list[dict],
    gate_status: str,
    unique_missing_context_sequences: int,
    unique_missing_context_rgb_frames: int,
) -> str:
    lines = [
        "# Exp083 Input Coverage Audit",
        "",
        f"**Gate status:** `{gate_status}`",
        "",
        "This is an identity/availability audit only. It did not decode test RGB, "
        "assemble test features, train a model, or compute test metrics.",
        "",
        "## Pose Sources",
        "",
    ]
    for source in pose_sources:
        lines.append(
            f"- `{source['path']}`: SHA-256 `{source['sha256']}`, "
            f"{source['rows']:,} COCO-17 rows, "
            f"{source['invalid_coco17_rows']} invalid rows."
        )
    lines.extend(
        [
            "",
            "Pose identity is exact `(video_id, ped_id, frame_id)`; no suffix "
            "aliasing or nearby-frame substitution is permitted.",
            "",
            "## Coverage",
            "",
            markdown_table(summary),
            "",
            "## Stop Decision",
            "",
            f"Existing compact Exp081/082 caches leave "
            f"{unique_missing_context_sequences:,} unique official JAAD_all "
            "sequences requiring fresh motion/TwinLiteNet+ context extraction. "
            f"Their observation windows contain "
            f"{unique_missing_context_rgb_frames:,} unique RGB frames.",
            "The runner therefore stops before extraction, feature assembly, or "
            "training, as frozen in the Exp083 design.",
            "",
            f"The road identity inventory contains {len(road_sources)} per-video "
            "NPZ artifacts. Only their `sequence_id` arrays were read during this "
            "coverage gate; cached test feature tensors were not accessed.",
            "",
        ]
    )
    return "\n".join(lines)


def run_coverage_audit(root: Path | str) -> dict:
    root = Path(root)
    require_protocol_audit(root)
    manifests = load_manifests(root)
    pose_available, pose_sources = load_pose_inventory()
    road_available, road_sources = load_road_sequence_inventory()

    summary_rows = []
    missing_rows = []
    missing_context_ids = set()
    for cohort in COHORTS:
        for split in SPLITS:
            manifest = manifests[(cohort, split)]
            required_frames = required_frame_keys({(cohort, split): manifest})

            pose_stats, pose_missing = audit_pose_coverage(
                required_frames, pose_available
            )
            summary_rows.append(
                _coverage_row(
                    cohort,
                    split,
                    "vitpose_s_coco17",
                    "pedestrian_frame",
                    pose_stats["required"],
                    pose_stats["missing"],
                )
            )
            missing_rows.extend(
                _missing_records(
                    pose_missing,
                    cohort=cohort,
                    split=split,
                    input_name="vitpose_s_coco17",
                    unit="pedestrian_frame",
                )
            )

            rgb_context = audit_rgb_and_context_coverage(
                required_frames, manifest, road_available
            )
            rgb_stats, rgb_missing = rgb_context["rgb"]
            summary_rows.append(
                _coverage_row(
                    cohort,
                    split,
                    "rgb_frame",
                    "frame",
                    rgb_stats["required"],
                    rgb_stats["missing"],
                )
            )
            missing_rows.extend(
                _missing_records(
                    rgb_missing,
                    cohort=cohort,
                    split=split,
                    input_name="rgb_frame",
                    unit="frame",
                )
            )

            context_stats, context_missing = rgb_context["context"]
            missing_context_ids.update(context_missing["source_sequence_id"])
            for input_name in ("motion_proxy_15d", "road_context_60d"):
                summary_rows.append(
                    _coverage_row(
                        cohort,
                        split,
                        input_name,
                        "sequence",
                        context_stats["required"],
                        context_stats["missing"],
                    )
                )
                missing_rows.extend(
                    _missing_records(
                        context_missing,
                        cohort=cohort,
                        split=split,
                        input_name=input_name,
                        unit="sequence",
                    )
                )

    summary = pd.DataFrame(summary_rows)
    missing = pd.DataFrame(missing_rows)
    blocking_missing = summary.loc[
        summary["input"].isin(
            [
                "vitpose_s_coco17",
                "rgb_frame",
                "motion_proxy_15d",
                "road_context_60d",
            ]
        ),
        "missing",
    ].sum()
    gate_status = (
        "blocked_new_extraction_required" if blocking_missing else "coverage_complete"
    )
    all_manifest = pd.concat(
        [manifests[("jaad_all", split)] for split in SPLITS], ignore_index=True
    )
    all_manifest = all_manifest.assign(
        source_sequence_id=_source_sequence_ids(all_manifest)
    )
    missing_context_manifest = all_manifest.loc[
        all_manifest["source_sequence_id"].isin(missing_context_ids)
    ]
    missing_context_frames = required_frame_keys(
        {("jaad_all", "all_splits"): missing_context_manifest}
    )
    unique_missing_context_rgb_frames = len(
        missing_context_frames[["video_id", "frame_id"]].drop_duplicates()
    )

    summary_path = root / "metrics/input_coverage.csv"
    missing_path = root / "metrics/missing_input_frames.csv"
    report_path = root / "INPUT_COVERAGE_AUDIT.md"
    contract.atomic_write_text(summary_path, summary.to_csv(index=False))
    contract.atomic_write_text(missing_path, missing.to_csv(index=False))
    contract.atomic_write_text(
        report_path,
        _render_audit(
            summary,
            pose_sources,
            road_sources,
            gate_status,
            len(missing_context_ids),
            unique_missing_context_rgb_frames,
        ),
    )
    marker = contract.publish_phase(
        root,
        "coverage-audit",
        [summary_path, missing_path, report_path],
        test_accessed=False,
        test_identity_audited=True,
        test_features_accessed=False,
        test_metrics_accessed=False,
        gate_status=gate_status,
        unique_missing_context_sequences=len(missing_context_ids),
        unique_missing_context_rgb_frames=unique_missing_context_rgb_frames,
        pose_sources=pose_sources,
    )
    return {
        "marker": str(marker),
        "gate_status": gate_status,
        "unique_missing_context_sequences": len(missing_context_ids),
        "unique_missing_context_rgb_frames": unique_missing_context_rgb_frames,
    }
