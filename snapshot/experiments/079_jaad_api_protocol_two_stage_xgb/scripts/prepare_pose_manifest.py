"""Prepare the exact missing-pose manifest for Exp079."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pose_support import build_missing_manifest, load_pose_cache_keys


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOT = Path(__file__).resolve().parents[3]
WINDOWS = ACTIVE_ROOT / "experiments/031_jaad_api_label_protocol_sensitivity/metrics/protocol_a_window_audit.csv"
EXISTING_CACHES = (
    ACTIVE_ROOT / "experiments/049_exp048_to_jaad_zero_shot/cache/jaad_keypoints_vitpose_s_exp049_extended.jsonl",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, str | int]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256(resolved), "size_bytes": resolved.stat().st_size}


def main() -> None:
    manifest, coverage = build_missing_manifest(WINDOWS, EXISTING_CACHES)
    manifest_path = ROOT / "cache/required_missing_pose_rows.csv"
    coverage_path = ROOT / "metrics/pre_extraction_pose_coverage.json"
    provenance_path = ROOT / "metrics/pose_source_provenance.json"
    manifest.to_csv(manifest_path, index=False)
    coverage_path.write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n")
    _, source_counts = load_pose_cache_keys(EXISTING_CACHES, verify_payload=True)
    provenance = {
        "window_manifest": artifact(WINDOWS),
        "existing_pose_caches": [artifact(path) for path in EXISTING_CACHES],
        "existing_cache_unique_key_contributions": source_counts,
        "missing_manifest": artifact(manifest_path),
        "identity_policy": "exact (set,video,pid,frame_idx); no b-suffix aliasing",
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"missing_rows": len(manifest), "coverage": coverage}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
