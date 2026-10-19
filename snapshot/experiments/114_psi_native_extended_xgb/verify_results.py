"""Independent verification of saved amended Exp114 development artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
EXP113 = ROOT.parent / "113_psi_native_biu_xgb"
SEEDS = (42, 43, 44)


def main() -> None:
    train = pd.read_csv(ROOT / "cache/features/train/rows.csv")
    val = pd.read_csv(ROOT / "cache/features/val/rows.csv")
    assert len(train) == 26295 and len(val) == 14922
    assert set(train.video_id.astype(str)).isdisjoint({"video_0032", "video_0055", "video_0093"})
    assert val.sequence_id.astype(str).equals(pd.read_csv(EXP113 / "cache/val/rows.csv").sequence_id.astype(str))
    source_train = np.load(ROOT / "cache/features/train/S187.npy")
    source_val = np.load(ROOT / "cache/features/val/S187.npy")
    road_train = np.load(ROOT / "cache/features/train/R247.npy")
    road_val = np.load(ROOT / "cache/features/val/R247.npy")
    np.testing.assert_array_equal(source_train, road_train[:, :187])
    np.testing.assert_array_equal(source_val, road_val[:, :187])
    assert source_train.shape == (26295, 187) and road_train.shape == (26295, 247)
    assert source_val.shape == (14922, 187) and road_val.shape == (14922, 247)
    table = pd.read_csv(ROOT / "metrics/per_seed_results.csv").set_index(["arm", "seed"])
    expected_params = {
        "n_estimators": 250, "max_depth": 4, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1,
        "objective": "binary:logistic", "eval_metric": "logloss", "n_jobs": 4,
    }
    y = val.label.to_numpy(int)
    for arm, matrix, training in (("S187", source_val, source_train), ("R247", road_val, road_train)):
        for seed in SEEDS:
            model = joblib.load(ROOT / f"models/{arm}_seed{seed}.joblib")
            parameters = model.named_steps["model"].get_params()
            assert all(parameters[key] == value for key, value in expected_params.items())
            assert parameters["random_state"] == seed
            np.testing.assert_allclose(model.named_steps["imputer"].statistics_, np.median(training, axis=0), rtol=0, atol=0)
            probability = model.predict_proba(matrix)[:, 1]
            prediction = probability >= 0.5
            saved = pd.read_csv(ROOT / f"predictions/validation_{arm}_seed{seed}.csv")
            assert saved.sequence_id.astype(str).equals(val.sequence_id.astype(str))
            np.testing.assert_array_equal(saved.probability.to_numpy(np.float32), probability)
            metrics = {
                "accuracy": accuracy_score(y, prediction),
                "precision": precision_score(y, prediction),
                "recall": recall_score(y, prediction),
                "f1": f1_score(y, prediction),
                "macro_f1": f1_score(y, prediction, average="macro"),
                "pr_auc": average_precision_score(y, probability),
                "roc_auc": roc_auc_score(y, probability),
            }
            for name, value in metrics.items():
                np.testing.assert_allclose(table.loc[(arm, seed), name], value, rtol=0, atol=1e-12)
    result = {
        "status": "verified",
        "independent_metric_recomputation": True,
        "saved_model_probability_reproduction": True,
        "training_only_imputer_medians": True,
        "matched_source_prefix": True,
        "train_windows": len(train),
        "validation_windows": len(val),
        "fits": 6,
        "test_accessed": False,
    }
    (ROOT / "INDEPENDENT_VERIFICATION.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
