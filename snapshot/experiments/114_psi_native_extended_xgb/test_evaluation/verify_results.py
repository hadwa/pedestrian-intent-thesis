"""Independent saved-artifact verification for the authorized Exp114 test evaluation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parent
EXP114 = ROOT.parent
REPO = EXP114.parents[1]
SEEDS = (42, 43, 44)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


HELPER = load_module(
    "psi114_test_independent_helper",
    REPO / "experiments/098_jaad_matched_biu_xgb/scripts/run_experiment.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual: float, expected: float) -> None:
    assert np.isclose(actual, expected, rtol=1e-10, atol=1e-10), (actual, expected)


def main() -> None:
    freeze = json.loads((ROOT / "freeze.json").read_text())
    frozen_models = {Path(item["path"]).name: item["sha256"] for item in freeze["models"]}
    rows = pd.read_csv(ROOT / "cache/features/rows.csv")
    assert len(rows) == 20547 and rows.sequence_id.is_unique
    assert int(rows.label.sum()) == 15462 and rows.track_id.nunique() == 96
    source = np.load(ROOT / "cache/features/S187.npy")
    road = np.load(ROOT / "cache/features/R247.npy")
    assert source.shape == (20547, 187) and road.shape == (20547, 247)
    np.testing.assert_array_equal(source, road[:, :187])
    table = pd.read_csv(ROOT / "metrics/per_seed_results.csv")
    verified = []
    for arm, matrix in (("S187", source), ("R247", road)):
        for seed in SEEDS:
            model_path = EXP114 / f"models/{arm}_seed{seed}.joblib"
            assert sha256(model_path) == frozen_models[model_path.name]
            model = joblib.load(model_path)
            probability = model.predict_proba(matrix)[:, 1]
            saved = pd.read_csv(ROOT / f"predictions/test_{arm}_seed{seed}.csv")
            assert saved.sequence_id.astype(str).equals(rows.sequence_id.astype(str))
            np.testing.assert_array_equal(saved.probability.to_numpy(np.float32), probability.astype(np.float32))
            metrics = HELPER.core.compute_metrics(rows.label.to_numpy(int), probability)
            metrics["macro_f1"] = f1_score(
                rows.label.to_numpy(int), probability >= 0.5, average="macro"
            )
            recorded = table[(table.arm == arm) & (table.seed == seed)].iloc[0]
            for name, value in metrics.items():
                close(float(recorded[name]), float(value))
            verified.append({"arm": arm, "seed": seed, "model_sha256": sha256(model_path)})
    output = {
        "status": "PASS",
        "test_authorized": True,
        "no_refit": True,
        "windows": 20547,
        "videos": int(rows.video_id.nunique()),
        "tracks": int(rows.track_id.nunique()),
        "positives": int(rows.label.sum()),
        "source_prefix_exact": True,
        "saved_probability_files_reproduced": 6,
        "saved_metric_rows_reproduced": 6,
        "models": verified,
    }
    (ROOT / "INDEPENDENT_VERIFICATION.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
