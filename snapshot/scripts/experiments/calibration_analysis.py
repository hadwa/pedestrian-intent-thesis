#!/usr/bin/env python3
"""Calibration analysis (experiment 013) from already-saved prediction probabilities.

Computes ECE (15 equal-width bins), Maximum Calibration Error, Brier score and
reliability-curve data for every saved (dataset, feature_set, model) prediction
file, on the TEST split. As a secondary analysis, fits isotonic regression on
the VALIDATION probabilities only and reports test ECE before/after.

No models are retrained. No test data is used for any fitting or selection.

Inputs (read only):
- JAAD 30-60: experiments/aibthings_jaad_all_models_fixed_0p5_30_60/probabilities/
    {feature_set}/{val,test}_{model}.csv   (columns: sequence_id, label, student_prob, ...)
- PIE 30-60 (archive repo, read-only): results/wacv_pcpa_30_60_overlap06/
    student_{feature_set}_supervised_wacv_pcpa_30_60_overlap06_{val,test}_predictions_{model}.csv

Outputs (written under experiments/013_calibration_analysis/):
- calibration_metrics.csv   tidy per-(dataset, feature_set, model) metrics
- reliability_curves.csv    per-bin confidence/accuracy/count (raw and isotonic)
- figures/jaad_reliability.png, figures/pie_reliability.png (grayscale-safe)

Usage:
    MPLCONFIGDIR=/tmp/mplconfig python3 scripts/experiments/calibration_analysis.py
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

REPO = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean")
ARCHIVE = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation")

JAAD_DIR = REPO / "experiments/aibthings_jaad_all_models_fixed_0p5_30_60/probabilities"
PIE_DIR = ARCHIVE / "results/wacv_pcpa_30_60_overlap06"

OUT_DIR = REPO / "experiments/013_calibration_analysis"
FIG_DIR = OUT_DIR / "figures"

N_BINS = 15

# Paper-selected / highlighted (dataset, feature_set, model) combinations.
HIGHLIGHTED = {
    ("jaad", "bbox", "random_forest"),
    ("jaad", "biu_bbox", "logistic_regression"),
    ("jaad", "biu_kp_bbox", "lightgbm"),
    ("pie", "bbox", "random_forest"),
    ("pie", "biu_bbox", "random_forest"),
}


def calibration_stats(y_true, y_prob, n_bins=N_BINS):
    """ECE / MCE (equal-width bins) and per-bin reliability data."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize with right-closed final bin: values == 1.0 go to the last bin.
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)

    rows = []
    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count > 0:
            conf = float(y_prob[mask].mean())
            acc = float(y_true[mask].mean())
            gap = abs(acc - conf)
            ece += (count / n) * gap
            mce = max(mce, gap)
        else:
            conf, acc = np.nan, np.nan
        rows.append(
            {
                "bin_lower": edges[b],
                "bin_upper": edges[b + 1],
                "bin_confidence": conf,
                "bin_accuracy": acc,
                "bin_count": count,
            }
        )
    return ece, mce, rows


def brier(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def load_predictions(dataset):
    """Yield (feature_set, model, val_df, test_df); dfs have label + student_prob."""
    if dataset == "jaad":
        for fs_dir in sorted(p for p in JAAD_DIR.iterdir() if p.is_dir()):
            fs = fs_dir.name
            models = sorted(
                f.name[len("test_") : -len(".csv")]
                for f in fs_dir.glob("test_*.csv")
            )
            for model in models:
                test = pd.read_csv(fs_dir / f"test_{model}.csv")
                val_path = fs_dir / f"val_{model}.csv"
                val = pd.read_csv(val_path) if val_path.exists() else None
                yield fs, model, val, test
    elif dataset == "pie":
        prefix = "student_"
        mid = "_supervised_wacv_pcpa_30_60_overlap06_test_predictions_"
        combos = []
        for f in sorted(PIE_DIR.glob(f"{prefix}*{mid}*.csv")):
            name = f.name[len(prefix) : -len(".csv")]
            fs, model = name.split(mid)
            combos.append((fs, model, f))
        for fs, model, test_path in combos:
            test = pd.read_csv(test_path)
            val_path = PIE_DIR / (
                f"student_{fs}_supervised_wacv_pcpa_30_60_overlap06_"
                f"val_predictions_{model}.csv"
            )
            val = pd.read_csv(val_path) if val_path.exists() else None
            yield fs, model, val, test
    else:
        raise ValueError(dataset)


def main():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    curve_rows = []
    # keep raw test probs of highlighted combos for plotting
    plot_data = {}

    for dataset in ("jaad", "pie"):
        for fs, model, val, test in load_predictions(dataset):
            for df, name in ((test, "test"), (val, "val")):
                if df is not None:
                    assert {"label", "student_prob"} <= set(df.columns), (
                        f"missing columns in {dataset}/{fs}/{model} {name}"
                    )
            y = test["label"].to_numpy()
            p = test["student_prob"].to_numpy()
            ece, mce, bins = calibration_stats(y, p)
            row = {
                "dataset": dataset,
                "feature_set": fs,
                "model": model,
                "split": "test",
                "n_test": len(y),
                "test_base_rate": float(np.mean(y)),
                "ece": ece,
                "mce": mce,
                "brier": brier(y, p),
                "highlighted": (dataset, fs, model) in HIGHLIGHTED,
            }
            for b in bins:
                curve_rows.append(
                    {"dataset": dataset, "feature_set": fs, "model": model,
                     "variant": "raw", **b}
                )

            # Secondary analysis: isotonic recalibration fit on VALIDATION only.
            if val is not None and len(val) > 0:
                iso = IsotonicRegression(
                    y_min=0.0, y_max=1.0, out_of_bounds="clip"
                )
                iso.fit(
                    val["student_prob"].to_numpy(),
                    val["label"].to_numpy().astype(float),
                )
                p_iso = iso.predict(p)
                ece_i, mce_i, bins_i = calibration_stats(y, p_iso)
                row.update(
                    {
                        "n_val": len(val),
                        "ece_isotonic": ece_i,
                        "mce_isotonic": mce_i,
                        "brier_isotonic": brier(y, p_iso),
                    }
                )
                for b in bins_i:
                    curve_rows.append(
                        {"dataset": dataset, "feature_set": fs, "model": model,
                         "variant": "isotonic_val_fit", **b}
                    )
            else:
                row.update(
                    {"n_val": 0, "ece_isotonic": np.nan,
                     "mce_isotonic": np.nan, "brier_isotonic": np.nan}
                )

            metric_rows.append(row)
            if (dataset, fs, model) in HIGHLIGHTED:
                plot_data[(dataset, fs, model)] = (row["ece"], bins)

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["dataset", "feature_set", "model"]
    )
    metrics.to_csv(OUT_DIR / "calibration_metrics.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(
        OUT_DIR / "reliability_curves.csv", index=False
    )

    # ---- Figures: one grayscale-safe reliability diagram per dataset ----
    styles = [
        {"color": "0.0", "marker": "o", "linestyle": "-"},
        {"color": "0.45", "marker": "s", "linestyle": "--"},
        {"color": "0.7", "marker": "^", "linestyle": "-."},
    ]
    for dataset, title in (("jaad", "JAAD 30-60 (test)"),
                           ("pie", "PIE 30-60 (test)")):
        combos = [k for k in plot_data if k[0] == dataset]
        combos.sort()
        fig, (ax, axh) = plt.subplots(
            2, 1, figsize=(5.2, 6.2), sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12},
        )
        ax.plot([0, 1], [0, 1], color="0.8", linestyle=":", linewidth=1,
                zorder=1, label="Perfect calibration")
        bar_w = (1.0 / N_BINS) / (len(combos) + 1)
        hatches = ["", "///", "..."]
        for i, key in enumerate(combos):
            _, fs, model = key
            ece, bins = plot_data[key]
            bdf = pd.DataFrame(bins).dropna(subset=["bin_confidence"])
            label = f"{fs} / {model} (ECE={ece:.3f})"
            st = styles[i % len(styles)]
            ax.plot(bdf["bin_confidence"], bdf["bin_accuracy"],
                    linewidth=1.6, markersize=5, zorder=3, label=label, **st)
            centers = (bdf["bin_lower"] + bdf["bin_upper"]) / 2
            axh.bar(centers + (i - (len(combos) - 1) / 2) * bar_w,
                    bdf["bin_count"], width=bar_w,
                    color=st["color"], edgecolor="white", linewidth=0.4,
                    hatch=hatches[i % len(hatches)])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Empirical accuracy (bin)")
        ax.set_title(f"Reliability diagram, {title}", fontsize=11)
        ax.legend(fontsize=8, loc="upper left", frameon=False)
        ax.grid(True, color="0.92", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        axh.set_ylabel("Count")
        axh.set_xlabel("Predicted probability (bin confidence)")
        axh.grid(True, axis="y", color="0.92", linewidth=0.6)
        axh.set_axisbelow(True)
        for a in (ax, axh):
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
        fig.savefig(FIG_DIR / f"{dataset}_reliability.png", dpi=200,
                    bbox_inches="tight")
        plt.close(fig)

    # console summary of highlighted models
    hl = metrics[metrics["highlighted"]]
    cols = ["dataset", "feature_set", "model", "n_test", "ece",
            "ece_isotonic", "mce", "brier", "brier_isotonic"]
    print(hl[cols].to_string(index=False))
    print(f"\nWrote {OUT_DIR / 'calibration_metrics.csv'}")
    print(f"Wrote {OUT_DIR / 'reliability_curves.csv'}")
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
