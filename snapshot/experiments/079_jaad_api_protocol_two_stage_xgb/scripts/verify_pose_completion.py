"""Verify Exp079 pose output and full Protocol A cache coverage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pose_support import build_missing_manifest, validate_pose_output


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOT = Path(__file__).resolve().parents[3]
WINDOWS = ACTIVE_ROOT / "experiments/031_jaad_api_label_protocol_sensitivity/metrics/protocol_a_window_audit.csv"
MANIFEST = ROOT / "cache/required_missing_pose_rows.csv"
OUTPUT = ROOT / "cache/jaad_api_protocol_missing_pose.jsonl"
CACHES = (
    ACTIVE_ROOT / "experiments/049_exp048_to_jaad_zero_shot/cache/jaad_keypoints_vitpose_s_exp049_extended.jsonl",
    OUTPUT,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    output_audit = validate_pose_output(OUTPUT, MANIFEST)
    remaining, coverage = build_missing_manifest(WINDOWS, CACHES)
    if not remaining.empty:
        raise RuntimeError(f"pose completion left {len(remaining)} exact identities missing")
    audit = {
        "output": output_audit,
        "full_protocol_coverage": coverage,
        "pose_output_path": str(OUTPUT.resolve()),
        "pose_output_sha256": sha256(OUTPUT),
        "pose_output_size_bytes": OUTPUT.stat().st_size,
    }
    path = ROOT / "metrics/post_extraction_pose_coverage.json"
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    (ROOT / "POSE_EXTRACTION_COMPLETE").write_text("complete\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
