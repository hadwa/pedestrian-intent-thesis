"""Freeze authorized test inputs and reproduce development predictions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP114 = ROOT.parent
EXP113_TEST = EXP114.parent / "113_psi_native_biu_xgb/test_evaluation"
SEEDS = (42, 43, 44)


def record(path: Path) -> dict[str, object]:
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main() -> None:
    gate = json.loads((ROOT / "archive_gate.json").read_text())
    assert gate["cv_annotation"]["member_count"] == 49 and not gate["cv_annotation"]["missing_videos"]
    assert gate["rgb"]["member_count"] == 49 and not gate["rgb"]["missing_videos"]
    split = json.loads((EXP113_TEST / "official_split.json").read_text())
    assert len(split["test"]) == 49 and "video_0192" not in split["test"]
    rows = pd.read_csv(EXP113_TEST / "cache/rows.csv")
    assert len(rows) == 20547 and rows.sequence_id.is_unique
    assert set(rows.video_id.astype(str)) == set(split["test"])
    assert int(rows.label.sum()) == 15462 and rows.track_id.nunique() == 96
    required = {
        (str(row.video_id), int(frame))
        for row in rows.itertuples(index=False)
        for frame in json.loads(row.observed_frames_json)
    }
    model_records = []
    for arm in ("S187", "R247"):
        validation = np.load(EXP114 / f"cache/features/val/{arm}.npy")
        validation_rows = pd.read_csv(EXP114 / "cache/features/val/rows.csv")
        for seed in SEEDS:
            path = EXP114 / f"models/{arm}_seed{seed}.joblib"
            model = joblib.load(path)
            probability = model.predict_proba(validation)[:, 1]
            saved = pd.read_csv(EXP114 / f"predictions/validation_{arm}_seed{seed}.csv")
            assert saved.sequence_id.astype(str).equals(validation_rows.sequence_id.astype(str))
            np.testing.assert_array_equal(saved.probability.to_numpy(np.float32), probability)
            model_records.append(record(path))
    sources = [
        ROOT / "DESIGN.md", ROOT / "archive_gate.json", EXP114 / "APPROVED_COHORT_AMENDMENT.md",
        EXP114 / "feature_contract.json", EXP114 / "run.py", EXP114 / "AMENDED_COMPLETE.json",
        EXP113_TEST / "official_split.json", EXP113_TEST / "cache/rows.csv",
        EXP113_TEST / "cache/A.npy", EXP113_TEST / "cache/B.npy", EXP113_TEST / "cache/C.npy",
        EXP114 / "scripts/recover_sources.py",
        ROOT / "scripts/preflight.py", ROOT / "scripts/recover.py", ROOT / "evaluate.py",
        ROOT / "verify_results.py",
        EXP114 / "../075_pie_twinlitenetplus_road_context/pre_registration.json",
        EXP114 / "../075_pie_twinlitenetplus_road_context/scripts/contract.py",
        EXP114 / "../075_pie_twinlitenetplus_road_context/scripts/context.py",
        EXP114 / "../075_pie_twinlitenetplus_road_context/scripts/twinlite_adapter.py",
        EXP114 / "../075_pie_twinlitenetplus_road_context/scripts/model.py",
        EXP114 / "../075_pie_twinlitenetplus_road_context/checkpoints/large.pth",
        EXP114.parent.parent / "external/TwinLiteNetPlus/model/model.py",
        EXP114.parent.parent / "external/TwinLiteNetPlus/model/config.py",
        EXP114 / "../063_pie_corrected_yolop_road_relative_context/scripts/representations.py",
        EXP114.parent.parent / "data/psi_vitpose_s_all_tracks.jsonl",
    ]
    output = {
        "status": "frozen_ready_for_speed_recovery",
        "test_authorized": True,
        "no_refit": True,
        "threshold": 0.5,
        "windows": len(rows),
        "videos": 49,
        "tracks": int(rows.track_id.nunique()),
        "positives": int(rows.label.sum()),
        "unique_observed_video_frames": len(required),
        "development_predictions_reproduced": 6,
        "models": model_records,
        "sources": [record(path) for path in sources],
    }
    (ROOT / "freeze.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
