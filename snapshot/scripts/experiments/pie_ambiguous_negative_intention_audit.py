#!/usr/bin/env python
"""PIE ambiguous-negative intention audit (experiment 015).

Question: can PIE's human-annotated ``intention_prob`` (annotations_attributes
XMLs) define "ambiguous negatives" on PIE, analogous to JAAD's
``crossing_attr == 0`` tracks, and do the main models' false positives
concentrate on them?

Inputs (read-only):
- PIE attribute XMLs:
  /home/abdelrahman/Hadwa/MSLSTM-PID/PIE/annotations/annotations_attributes/set*/video_*_attributes.xml
  (per-pedestrian ``intention_prob`` and ``crossing`` attributes)
- Archived PIE 30-60 benchmark test predictions (fixed benchmark, NOT rerun):
  /home/abdelrahman/Hadwa/BIU-PID-Distillation/results/wacv_pcpa_30_60_overlap06/
  student_{fs}_supervised_wacv_pcpa_30_60_overlap06_test_predictions_{model}.csv

Outputs -> experiments/015_pie_ambiguous_negative_intention_audit/

This is a DIAGNOSTIC / sensitivity analysis at fixed threshold 0.5.
The "remove ambiguous negatives" numbers are a sensitivity analysis,
not a benchmark. No thresholds or models are selected on test data.
"""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import glob
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

PIE_ATTR_DIR = "/home/abdelrahman/Hadwa/MSLSTM-PID/PIE/annotations/annotations_attributes"
PRED_DIR = "/home/abdelrahman/Hadwa/BIU-PID-Distillation/results/wacv_pcpa_30_60_overlap06"
OUT_DIR = (
    "/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/"
    "experiments/015_pie_ambiguous_negative_intention_audit"
)
FEATURE_SETS = ["bbox", "biu_bbox"]
MODEL = "random_forest"
THRESHOLD = 0.5

os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: parse intention_prob + crossing per pedestrian from attribute XMLs.
# ---------------------------------------------------------------------------
def load_pie_attributes():
    rows = []
    xml_paths = sorted(
        p
        for p in glob.glob(os.path.join(PIE_ATTR_DIR, "set*", "video_*_attributes.xml"))
        if p.endswith(".xml")  # skip Windows Zone.Identifier files
    )
    if not xml_paths:
        raise SystemExit(f"No attribute XMLs found under {PIE_ATTR_DIR}")
    for path in xml_paths:
        root = ET.parse(path).getroot()
        for ped in root.iter("pedestrian"):
            attrs = ped.attrib
            if "intention_prob" not in attrs:
                raise SystemExit(
                    f"'intention_prob' missing in {path}; attributes present: "
                    f"{sorted(attrs)}"
                )
            rows.append(
                {
                    "ped_id": attrs["id"],
                    "intention_prob": float(attrs["intention_prob"]),
                    "crossing": int(attrs["crossing"]),
                    "source_xml": path,
                }
            )
    df = pd.DataFrame(rows)
    dup = df["ped_id"].duplicated()
    if dup.any():
        raise SystemExit(f"Duplicate ped ids in XMLs: {df.loc[dup, 'ped_id'].tolist()[:10]}")
    return df


attrs_df = load_pie_attributes()
attrs_df[["ped_id", "intention_prob", "crossing"]].to_csv(
    os.path.join(OUT_DIR, "pie_track_intention_attributes.csv"), index=False
)

# Histogram table of intention_prob by crossing value (all annotated tracks).
bins = np.linspace(0.0, 1.0, 11)
hist_rows = []
for crossing_val, grp in attrs_df.groupby("crossing"):
    counts, edges = np.histogram(grp["intention_prob"], bins=bins)
    for lo, hi, c in zip(edges[:-1], edges[1:], counts):
        hist_rows.append(
            {
                "crossing": crossing_val,
                "bin_lo": round(lo, 1),
                "bin_hi": round(hi, 1),
                "n_tracks": int(c),
            }
        )
hist_df = pd.DataFrame(hist_rows)
hist_df.to_csv(os.path.join(OUT_DIR, "intention_prob_histogram_by_crossing.csv"), index=False)


# ---------------------------------------------------------------------------
# Step 2: load test predictions and join on pedestrian id.
# sequence_id example: set03_video_0001_3_1_220_1515_1530_1590_0
# tokens: [set03, video, 0001, 3, 1, 220, obs_start, obs_end, tte_frame, win_idx]
# -> track key = first 6 tokens; ped id = tokens 3:6 joined ("3_1_220"),
#    matching the XML id format "<set>_<video>_<ped>" (e.g. "4_7_1356").
# ---------------------------------------------------------------------------
def load_predictions(feature_set):
    path = os.path.join(
        PRED_DIR,
        f"student_{feature_set}_supervised_wacv_pcpa_30_60_overlap06_"
        f"test_predictions_{MODEL}.csv",
    )
    df = pd.read_csv(path)
    toks = df["sequence_id"].str.split("_")
    df["track_key"] = toks.str[:6].str.join("_")
    df["ped_id"] = toks.str[3:6].str.join("_")
    merged = df.merge(
        attrs_df[["ped_id", "intention_prob", "crossing"]], on="ped_id", how="left"
    )
    unmatched = merged["intention_prob"].isna()
    if unmatched.any():
        raise SystemExit(
            f"{feature_set}: {unmatched.sum()} test windows have ped ids not found "
            f"in attribute XMLs, e.g. {merged.loc[unmatched, 'ped_id'].unique()[:10]}"
        )
    return merged, path


# ---------------------------------------------------------------------------
# Step 3: negative subgroups among label==0 windows by track intention_prob.
# ---------------------------------------------------------------------------
def subgroup_of(p):
    if p <= 0.33:
        return "clean_negative"
    if p < 0.66:
        return "medium"
    return "ambiguous_negative"


SUBGROUPS = ["clean_negative", "medium", "ambiguous_negative"]


def metrics_at_threshold(y, prob, thr=THRESHOLD):
    pred = (prob >= thr).astype(int)
    return {
        "f1": f1_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "pr_auc": average_precision_score(y, prob),
        "n_windows": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
    }


count_rows, fpr_rows, sens_rows = [], [], []
plot_data = {}
pred_paths = {}

for fs in FEATURE_SETS:
    df, path = load_predictions(fs)
    pred_paths[fs] = path
    df["subgroup"] = df["intention_prob"].apply(subgroup_of)

    neg = df[df["label"] == 0].copy()
    if fs == FEATURE_SETS[0]:
        # Subgroup counts (windows + tracks) — identical across feature sets
        # (same benchmark windows), computed once from the first one and
        # cross-checked against the second below.
        for sg in SUBGROUPS:
            sub = neg[neg["subgroup"] == sg]
            count_rows.append(
                {
                    "subgroup": sg,
                    "n_neg_windows": int(len(sub)),
                    "n_neg_tracks": int(sub["track_key"].nunique()),
                    "n_neg_ped_ids": int(sub["ped_id"].nunique()),
                }
            )
        counts_ref = neg.groupby("subgroup").size()
    else:
        assert neg.groupby("subgroup").size().equals(counts_ref), (
            "Negative subgroup window counts differ across feature sets"
        )

    # Step 4a: per-subgroup FPR + mean predicted prob at fixed 0.5.
    fs_plot = {}
    for sg in SUBGROUPS:
        sub = neg[neg["subgroup"] == sg]
        fp = int((sub["student_prob"] >= THRESHOLD).sum())
        fpr = fp / len(sub) if len(sub) else float("nan")
        fpr_rows.append(
            {
                "feature_set": fs,
                "model": MODEL,
                "threshold": THRESHOLD,
                "subgroup": sg,
                "n_neg_windows": int(len(sub)),
                "false_positives": fp,
                "fpr": fpr,
                "mean_predicted_prob": float(sub["student_prob"].mean()),
            }
        )
        fs_plot[sg] = {"fpr": fpr, "mean_prob": float(sub["student_prob"].mean())}
    plot_data[fs] = fs_plot

    # Step 4b: sensitivity analysis — remove ambiguous negatives (label==0
    # windows whose track intention_prob >= 0.66) and recompute metrics.
    # This is a sensitivity analysis, NOT a benchmark.
    full = metrics_at_threshold(df["label"].values, df["student_prob"].values)
    keep = ~((df["label"] == 0) & (df["subgroup"] == "ambiguous_negative"))
    red = df[keep]
    reduced = metrics_at_threshold(red["label"].values, red["student_prob"].values)
    for variant, m in [("full_test", full), ("no_ambiguous_negatives", reduced)]:
        sens_rows.append({"feature_set": fs, "model": MODEL, "variant": variant, **m})
    sens_rows.append(
        {
            "feature_set": fs,
            "model": MODEL,
            "variant": "delta (no_ambig - full)",
            "f1": reduced["f1"] - full["f1"],
            "precision": reduced["precision"] - full["precision"],
            "recall": reduced["recall"] - full["recall"],
            "pr_auc": reduced["pr_auc"] - full["pr_auc"],
            "n_windows": reduced["n_windows"] - full["n_windows"],
            "n_pos": reduced["n_pos"] - full["n_pos"],
            "n_neg": reduced["n_neg"] - full["n_neg"],
        }
    )

counts_df = pd.DataFrame(count_rows)
counts_df.to_csv(os.path.join(OUT_DIR, "negative_subgroup_counts.csv"), index=False)
fpr_df = pd.DataFrame(fpr_rows)
fpr_df.to_csv(os.path.join(OUT_DIR, "per_subgroup_fpr.csv"), index=False)
sens_df = pd.DataFrame(sens_rows)
sens_df.to_csv(
    os.path.join(OUT_DIR, "sensitivity_remove_ambiguous_negatives.csv"), index=False
)

# ---------------------------------------------------------------------------
# Figure: per-subgroup FPR and mean predicted prob, both feature sets.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
x = np.arange(len(SUBGROUPS))
w = 0.35
colors = {"bbox": "#4878a8", "biu_bbox": "#c46a4a"}
for ax, key, title in [
    (axes[0], "fpr", f"FPR at threshold {THRESHOLD} (label==0 windows)"),
    (axes[1], "mean_prob", "Mean predicted crossing prob (label==0 windows)"),
]:
    for i, fs in enumerate(FEATURE_SETS):
        vals = [plot_data[fs][sg][key] for sg in SUBGROUPS]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, label=fs, color=colors[fs])
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["clean\n(<=0.33)", "medium\n(0.33-0.66)", "ambiguous\n(>=0.66)"], fontsize=9
    )
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
fig.suptitle(
    "PIE 30-60 test negatives by human intention_prob (random_forest, fixed 0.5)",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "per_subgroup_fpr_and_mean_prob.png"), dpi=150)

# ---------------------------------------------------------------------------
# Console summary (numbers quoted in report.md come from the CSVs above).
# ---------------------------------------------------------------------------
print("Attribute XMLs parsed:", attrs_df["source_xml"].nunique(), "files,")
print("annotated pedestrian tracks:", len(attrs_df))
print("\nintention_prob summary by crossing value (all annotated tracks):")
print(
    attrs_df.groupby("crossing")["intention_prob"]
    .describe()[["count", "mean", "min", "25%", "50%", "75%", "max"]]
    .to_string()
)
print("\nPrediction files used:")
for fs, p in pred_paths.items():
    print(" ", p)
print("\nNegative subgroup counts:\n", counts_df.to_string(index=False))
print("\nPer-subgroup FPR:\n", fpr_df.to_string(index=False))
print("\nSensitivity (remove ambiguous negatives):\n", sens_df.to_string(index=False))
