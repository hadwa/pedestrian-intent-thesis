# summarize_runs.py
import argparse, json
from pathlib import Path
import pandas as pd

def safe_get(d, *keys, default=None):
    for k in keys:
        if d is None: return default
        d = d.get(k) if isinstance(d, dict) else default
    return d if d is not None else default

def collect_run(run_dir: Path):
    """Return a dict with metrics for one model run directory."""
    om_path = run_dir / "overall_metrics.json"
    rt_path = run_dir / "runtimes.csv"
    seq_meta = run_dir / "seq_meta.txt"

    if not om_path.exists() or not rt_path.exists():
        return None

    with om_path.open("r", encoding="utf-8") as f:
        om = json.load(f)

    # overall report holds per-class metrics; class "1" = positive (cross)
    rep = om.get("overall_report", {})
    f1_pos   = safe_get(rep, "1", "f1-score", default=None)
    prec_pos = safe_get(rep, "1", "precision", default=None)
    rec_pos  = safe_get(rep, "1", "recall", default=None)
    f1_macro = safe_get(rep, "macro avg", "f1-score", default=None)
    f1_weighted = safe_get(rep, "weighted avg", "f1-score", default=None)

    # AUCs
    roc_auc = om.get("roc_auc", None)
    pr_auc  = om.get("pr_auc", None)

    # horizon/context from overall_metrics (written by the factory script)
    H = om.get("H", None)
    W = om.get("W", None)
    skip = om.get("skip", None)
    target_mode = om.get("target_mode", None)
    threshold = om.get("threshold", None)

    # read runtimes
    rt_df = pd.read_csv(rt_path)
    train_mean = rt_df["train_time_sec"].mean() if "train_time_sec" in rt_df else None
    infer_mean = rt_df["infer_time_sec"].mean() if "infer_time_sec" in rt_df else None
    mss_mean   = rt_df["ms_per_sample"].mean()   if "ms_per_sample"   in rt_df else None
    mss_std    = rt_df["ms_per_sample"].std()    if "ms_per_sample"   in rt_df else None
    folds      = len(rt_df)

    # model name = last dir (rf/xgb/lgbm/cat) if you followed factory layout
    model = run_dir.name
    # parent may hold an experiment group; keep it to distinguish horizons, etc.
    group = run_dir.parent.name

    # Try to read seq_meta for extra reproducibility info
    meta_txt = seq_meta.read_text(encoding="utf-8") if seq_meta.exists() else ""
    # Compute actual horizon in seconds if we can (fps=30 common; factory used "skip" as sampling stride)
    horizon_sec = None
    if H is not None and skip is not None:
        horizon_sec = (H * skip) / 30.0

    return {
        "group": group,
        "model": model,
        "W": W, "H": H, "skip": skip,
        "horizon_sec": horizon_sec,
        "target_mode": target_mode,
        "threshold": threshold,
        "F1_pos": f1_pos,
        "Precision_pos": prec_pos,
        "Recall_pos": rec_pos,
        "F1_macro": f1_macro,
        "F1_weighted": f1_weighted,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "folds": folds,
        "train_time_sec_mean": train_mean,
        "infer_time_sec_mean": infer_mean,
        "ms_per_sample_mean": mss_mean,
        "ms_per_sample_std": mss_std,
        "run_dir": str(run_dir),
        "meta": meta_txt.strip()
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="../results/factory_runs",
                    help="Folder that contains subfolders per model (rf/xgb/lgbm/cat)")
    ap.add_argument("--out_csv", type=str, default="summary.csv")
    ap.add_argument("--sort_by", type=str, default="PR_AUC",
                    choices=["PR_AUC","ROC_AUC","F1_pos","F1_macro","ms_per_sample_mean"])
    ap.add_argument("--descending", action="store_true", help="sort descending")
    args = ap.parse_args()

    base = Path(args.base_dir)
    if not base.exists():
        raise SystemExit(f"Base dir not found: {base}")

    rows = []
    # Look one level deep (…/factory_runs/<model>/). If you have deeper nesting, walk recursively:
    for run_dir in base.rglob("overall_metrics.json"):
        rd = run_dir.parent
        item = collect_run(rd)
        if item:
            rows.append(item)

    if not rows:
        raise SystemExit(f"No runs found under {base}")

    df = pd.DataFrame(rows)

    # Round a few columns for readability
    round_cols = ["F1_pos","Precision_pos","Recall_pos","F1_macro","F1_weighted",
                  "ROC_AUC","PR_AUC","horizon_sec","ms_per_sample_mean","ms_per_sample_std",
                  "train_time_sec_mean","infer_time_sec_mean"]
    for c in round_cols:
        if c in df.columns:
            df[c] = df[c].astype(float).round(4)

    # Sort and save
    df = df.sort_values(args.sort_by, ascending=not args.descending)
    out_csv = Path(args.out_csv)
    df.to_csv(out_csv, index=False)

    # Pretty print condensed view
    cols_to_show = ["group","model","W","H","skip","horizon_sec",
                    "F1_pos","PR_AUC","ROC_AUC",
                    "ms_per_sample_mean","ms_per_sample_std","folds"]
    print("\n=== Summary ===")
    print(df[cols_to_show].to_string(index=False))
    print(f"\nSaved: {out_csv.resolve()}")

if __name__ == "__main__":
    main()
