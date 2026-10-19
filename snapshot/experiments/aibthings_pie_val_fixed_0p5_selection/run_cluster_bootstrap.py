"""Track-level (cluster) bootstrap for the paper's paired comparisons.

Overlapping windows within a pedestrian track are correlated, so window-level
McNemar/bootstrap are anti-conservative. Here tracks are resampled with
replacement (10k iterations, seed 12345) and metric deltas are computed on the
union of the sampled tracks' windows.

Outputs: cluster_bootstrap_tests.csv (this folder).
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PIE_DIR = "/home/abdelrahman/Hadwa/BIU-PID-Distillation/results/wacv_pcpa_30_60_overlap06"
JAAD_PROB_DIR = "/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/aibthings_jaad_all_models_fixed_0p5_30_60/probabilities"
THRESH = 0.5
N_BOOT = 10000
SEED = 12345


def load_pie(fs, model):
    df = pd.read_csv(os.path.join(
        PIE_DIR, f"student_{fs}_supervised_wacv_pcpa_30_60_overlap06_test_predictions_{model}.csv"))
    df = df.sort_values("sequence_id").reset_index(drop=True)
    track = df["sequence_id"].str.split("_").str[:6].str.join("_")
    return df["sequence_id"].to_numpy(), track.to_numpy(), df["label"].to_numpy().astype(int), df["student_prob"].to_numpy()


def load_jaad(fs, model):
    df = pd.read_csv(os.path.join(JAAD_PROB_DIR, fs, f"test_{model}.csv"))
    df = df.sort_values("sequence_id").reset_index(drop=True)
    track = df["sequence_id"].str.split(":").str[:2].str.join(":")
    return df["sequence_id"].to_numpy(), track.to_numpy(), df["label"].to_numpy().astype(int), df["student_prob"].to_numpy()


def cluster_bootstrap(track, y, pa, pb):
    rng = np.random.default_rng(SEED)
    tracks = np.unique(track)
    idx_by_track = {t: np.flatnonzero(track == t) for t in tracks}
    qa = (pa >= THRESH).astype(int)
    qb = (pb >= THRESH).astype(int)
    d_f1, d_acc = [], []
    for _ in range(N_BOOT):
        sample_tracks = rng.choice(tracks, size=len(tracks), replace=True)
        idx = np.concatenate([idx_by_track[t] for t in sample_tracks])
        yy, aa, bb = y[idx], qa[idx], qb[idx]
        d_f1.append(f1_score(yy, bb, zero_division=0) - f1_score(yy, aa, zero_division=0))
        d_acc.append(accuracy_score(yy, bb) - accuracy_score(yy, aa))
    d_f1, d_acc = np.array(d_f1), np.array(d_acc)

    def summ(d):
        return {
            "mean": float(d.mean()),
            "ci_low": float(np.percentile(d, 2.5)),
            "ci_high": float(np.percentile(d, 97.5)),
            "p_two_sided_sign": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
        }

    return {"n_tracks": len(tracks), "delta_f1": summ(d_f1), "delta_acc": summ(d_acc)}


def flat(dataset, comparison, r):
    return {
        "dataset": dataset, "comparison": comparison, "n_tracks": r["n_tracks"],
        "delta_f1_mean": r["delta_f1"]["mean"], "delta_f1_ci_low": r["delta_f1"]["ci_low"],
        "delta_f1_ci_high": r["delta_f1"]["ci_high"], "delta_f1_p": r["delta_f1"]["p_two_sided_sign"],
        "delta_acc_mean": r["delta_acc"]["mean"], "delta_acc_ci_low": r["delta_acc"]["ci_low"],
        "delta_acc_ci_high": r["delta_acc"]["ci_high"], "delta_acc_p": r["delta_acc"]["p_two_sided_sign"],
    }


sa, ta, ya, pa = load_jaad("bbox", "random_forest")
sb, tb, yb, pb = load_jaad("biu_bbox", "logistic_regression")
assert (sa == sb).all() and (ya == yb).all()
rows = [flat("JAAD", "BBox/RF vs BIU+BBox/LR", cluster_bootstrap(ta, ya, pa, pb))]

sc, tc, yc, pc = load_jaad("biu_kp_bbox", "lightgbm")
assert (sa == sc).all()
rows.append(flat("JAAD", "BBox/RF vs BIU+KP+BBox/LGBM", cluster_bootstrap(ta, ya, pa, pc)))

sa2, ta2, ya2, pa2 = load_pie("bbox", "random_forest")
sb2, tb2, yb2, pb2 = load_pie("biu_bbox", "random_forest")
assert (sa2 == sb2).all() and (ya2 == yb2).all()
rows.append(flat("PIE", "BBox/RF vs BIU+BBox/RF", cluster_bootstrap(ta2, ya2, pa2, pb2)))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "cluster_bootstrap_tests.csv"), index=False)
print(df.to_string(index=False))
