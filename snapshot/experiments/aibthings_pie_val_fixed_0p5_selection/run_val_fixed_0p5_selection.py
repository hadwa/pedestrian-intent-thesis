"""Validation-based fixed-0.5 model selection for PIE, significance tests, and naive baselines.

Read-only inputs:
- PIE saved val/test probability CSVs under
  /home/abdelrahman/Hadwa/BIU-PID-Distillation/results/wacv_pcpa_30_60_overlap06/
- JAAD saved val/test probability CSVs under
  /home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/aibthings_jaad_all_models_fixed_0p5_30_60/probabilities/

Outputs (this folder only):
- pie_val_fixed_0p5_all_models.csv          : val + test fixed-0.5 metrics for every PIE feature set / model
- pie_selected_by_val_fixed_0p5.csv         : per feature set, model with highest val fixed-0.5 F1 + its test metrics
- significance_tests.csv                    : McNemar exact tests + paired bootstrap deltas (PIE and JAAD)
- naive_baselines.csv                       : all-positive / majority-class baselines on PIE and JAAD test splits
"""

import itertools
import json
import math
import os

import numpy as np
import pandas as pd
from scipy import stats

PIE_DIR = "/home/abdelrahman/Hadwa/BIU-PID-Distillation/results/wacv_pcpa_30_60_overlap06"
JAAD_PROB_DIR = "/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/aibthings_jaad_all_models_fixed_0p5_30_60/probabilities"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURE_SETS = ["biu", "kp", "bbox", "biu_bbox", "kp_bbox", "biu_kp", "biu_kp_bbox"]
MODELS = ["lightgbm", "logistic_regression", "random_forest", "xgboost"]
THRESH = 0.5
RNG_SEED = 12345
N_BOOT = 10000


def pie_path(fs, split, model):
    return os.path.join(
        PIE_DIR,
        f"student_{fs}_supervised_wacv_pcpa_30_60_overlap06_{split}_predictions_{model}.csv",
    )


def metrics_from(y, p, thresh=THRESH):
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    pred = (p >= thresh).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def load_pie(fs, split, model):
    df = pd.read_csv(pie_path(fs, split, model))
    df = df.sort_values("sequence_id").reset_index(drop=True)
    return df["sequence_id"].to_numpy(), df["label"].to_numpy().astype(int), df["student_prob"].to_numpy()


rows = []
for fs, model in itertools.product(FEATURE_SETS, MODELS):
    vp = pie_path(fs, "val", model)
    tp_ = pie_path(fs, "test", model)
    if not (os.path.exists(vp) and os.path.exists(tp_)):
        print(f"MISSING: {fs} {model}")
        continue
    _, yv, pv = load_pie(fs, "val", model)
    _, yt, pt = load_pie(fs, "test", model)
    mv = metrics_from(yv, pv)
    mt = metrics_from(yt, pt)
    row = {"feature_set": fs, "model": model, "val_rows": len(yv), "test_rows": len(yt)}
    row.update({f"val_{k}": v for k, v in mv.items()})
    row.update({f"test_{k}": v for k, v in mt.items()})
    rows.append(row)

all_df = pd.DataFrame(rows)
all_df.to_csv(os.path.join(OUT_DIR, "pie_val_fixed_0p5_all_models.csv"), index=False)

sel = (
    all_df.sort_values(["feature_set", "val_f1"], ascending=[True, False])
    .groupby("feature_set", as_index=False)
    .first()
)
sel["selection_rule"] = "highest validation F1 at fixed threshold 0.5"
sel.to_csv(os.path.join(OUT_DIR, "pie_selected_by_val_fixed_0p5.csv"), index=False)
print("=== PIE selected by val fixed-0.5 F1 ===")
print(
    sel[
        ["feature_set", "model", "val_f1", "test_accuracy", "test_roc_auc", "test_pr_auc", "test_f1", "test_precision", "test_recall"]
    ].to_string(index=False)
)


def mcnemar_exact(y, pred_a, pred_b):
    correct_a = pred_a == y
    correct_b = pred_b == y
    b = int((correct_a & ~correct_b).sum())  # A right, B wrong
    c = int((~correct_a & correct_b).sum())  # A wrong, B right
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = 2 * sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) * (0.5 ** n)
    return b, c, min(1.0, p)


def paired_bootstrap(y, pa, pb, n_boot=N_BOOT, seed=RNG_SEED):
    from sklearn.metrics import f1_score, accuracy_score

    rng = np.random.default_rng(seed)
    n = len(y)
    d_f1, d_acc = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        qa = (pa[idx] >= THRESH).astype(int)
        qb = (pb[idx] >= THRESH).astype(int)
        d_f1.append(f1_score(yy, qb, zero_division=0) - f1_score(yy, qa, zero_division=0))
        d_acc.append(accuracy_score(yy, qb) - accuracy_score(yy, qa))
    d_f1 = np.array(d_f1)
    d_acc = np.array(d_acc)
    return {
        "delta_f1_mean": float(d_f1.mean()),
        "delta_f1_ci_low": float(np.percentile(d_f1, 2.5)),
        "delta_f1_ci_high": float(np.percentile(d_f1, 97.5)),
        "delta_acc_mean": float(d_acc.mean()),
        "delta_acc_ci_low": float(np.percentile(d_acc, 2.5)),
        "delta_acc_ci_high": float(np.percentile(d_acc, 97.5)),
    }


sig_rows = []

# PIE: BBox/RF vs BIU+BBox/RF (both validation-selected)
ids_a, y_a, p_a = load_pie("bbox", "test", "random_forest")
ids_b, y_b, p_b = load_pie("biu_bbox", "test", "random_forest")
assert (ids_a == ids_b).all() and (y_a == y_b).all()
pred_a = (p_a >= THRESH).astype(int)
pred_b = (p_b >= THRESH).astype(int)
b, c, pval = mcnemar_exact(y_a, pred_a, pred_b)
boot = paired_bootstrap(y_a, p_a, p_b)
sig_rows.append(
    {
        "dataset": "PIE",
        "comparison": "BBox/RF vs BIU+BBox/RF (fixed 0.5)",
        "n_test": len(y_a),
        "a_right_b_wrong": b,
        "a_wrong_b_right": c,
        "mcnemar_exact_p": pval,
        **boot,
    }
)


def load_jaad(fs, split, model):
    df = pd.read_csv(os.path.join(JAAD_PROB_DIR, fs, f"{split}_{model}.csv"))
    idc = "sequence_id" if "sequence_id" in df.columns else df.columns[0]
    prob_col = [c for c in df.columns if "prob" in c][0]
    label_col = [c for c in df.columns if "label" in c or c == "y_true"][0]
    df = df.sort_values(idc).reset_index(drop=True)
    return df[idc].to_numpy(), df[label_col].to_numpy().astype(int), df[prob_col].to_numpy()


# JAAD: BBox/RF vs BIU+BBox/LR (the paper's validation-selected models)
try:
    ja, ya, pa = load_jaad("bbox", "test", "random_forest")
    jb, yb, pb = load_jaad("biu_bbox", "test", "logistic_regression")
    assert (ja == jb).all() and (ya == yb).all()
    qa = (pa >= THRESH).astype(int)
    qb = (pb >= THRESH).astype(int)
    b, c, pval = mcnemar_exact(ya, qa, qb)
    boot = paired_bootstrap(ya, pa, pb)
    sig_rows.append(
        {
            "dataset": "JAAD",
            "comparison": "BBox/RF vs BIU+BBox/LR (fixed 0.5)",
            "n_test": len(ya),
            "a_right_b_wrong": b,
            "a_wrong_b_right": c,
            "mcnemar_exact_p": pval,
            **boot,
        }
    )
    jc, yc, pc = load_jaad("biu_kp_bbox", "test", "lightgbm")
    assert (ja == jc).all()
    qc = (pc >= THRESH).astype(int)
    b, c, pval = mcnemar_exact(ya, qa, qc)
    boot = paired_bootstrap(ya, pa, pc)
    sig_rows.append(
        {
            "dataset": "JAAD",
            "comparison": "BBox/RF vs BIU+KP+BBox/LGBM (fixed 0.5)",
            "n_test": len(ya),
            "a_right_b_wrong": b,
            "a_wrong_b_right": c,
            "mcnemar_exact_p": pval,
            **boot,
        }
    )
except Exception as e:
    print("JAAD significance failed:", e)

sig_df = pd.DataFrame(sig_rows)
sig_df.to_csv(os.path.join(OUT_DIR, "significance_tests.csv"), index=False)
print("\n=== Significance tests ===")
print(sig_df.to_string(index=False))

# Naive baselines
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

base_rows = []
for name, (y,) in {"PIE": (y_a,), "JAAD": (ya,)}.items():
    for label_choice, tag in [(1, "all_positive"), (int(pd.Series(y).mode()[0]), "majority_class")]:
        pred = np.full_like(y, label_choice)
        base_rows.append(
            {
                "dataset": name,
                "baseline": tag,
                "predicted_class": label_choice,
                "n_test": len(y),
                "positive_ratio": float(y.mean()),
                "accuracy": accuracy_score(y, pred),
                "f1": f1_score(y, pred, zero_division=0),
                "precision": precision_score(y, pred, zero_division=0),
                "recall": recall_score(y, pred, zero_division=0),
            }
        )
base_df = pd.DataFrame(base_rows)
base_df.to_csv(os.path.join(OUT_DIR, "naive_baselines.csv"), index=False)
print("\n=== Naive baselines ===")
print(base_df.to_string(index=False))
