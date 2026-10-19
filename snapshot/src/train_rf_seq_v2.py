# train_rf_biu_min.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance
import joblib


from seq_builder import load_csv, build_sequences
csv_path = "train_11.csv"

def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("csv_path", type=str)
    ap.add_argument("--W", type=int, default=10, help="history window (frames kept)")
    ap.add_argument("--H", type=int, default=5, help="horizon (frames to look ahead)")
    ap.add_argument("--skip", type=int, default=2, help="skip frames in between (2 => every 3rd frame)")
    ap.add_argument("--target_mode", type=str, default="any",
                    choices=["any","all","majority","next"],
                    help="how to derive label from next H frames")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--outdir", type=str, default="../results/exp_BIU10")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.5, help="decision threshold on P(y=1)")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Build sequences
    df = load_csv(csv_path)
    X, y, groups, feat_names, meta = build_sequences(
        df, W=args.W, H=args.H, skip=args.skip, target_mode=args.target_mode
    )
    print(f"Dataset: X={X.shape}, positives={y.sum()}/{len(y)} ({y.mean()*100:.2f}%), "
          f"groups={len(np.unique(groups))}")

    # Group-aware CV (leave-track-out)
    n_groups = len(np.unique(groups))
    n_splits = min(args.splits, max(2, n_groups))
    gkf = GroupKFold(n_splits=n_splits)

    y_pred_all = np.zeros_like(y)
    y_proba_all = np.zeros_like(y, dtype=float)

    fold = 1
    for tr, te in gkf.split(X, y, groups):
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            class_weight="balanced",
            random_state=args.random_state + fold
        )
        clf.fit(X[tr], y[tr])

        proba = clf.predict_proba(X[te])[:, 1]
        pred  = (proba >= args.threshold).astype(int)

        print(f"\n=== Fold {fold} ===")
        print(classification_report(y[te], pred, digits=3))
        try:
            print(f"ROC-AUC: {roc_auc_score(y[te], proba):.4f} | "
                  f"PR-AUC: {average_precision_score(y[te], proba):.4f}")
        except ValueError:
            print("ROC/PR not computed (only one class present in this fold).")

        y_pred_all[te]  = pred
        y_proba_all[te] = proba
        fold += 1

    print("\n=== Overall (OOGroup CV) ===")
    print(classification_report(y, y_pred_all, digits=3))
    try:
        print(f"ROC-AUC: {roc_auc_score(y, y_proba_all):.4f} | "
              f"PR-AUC: {average_precision_score(y, y_proba_all):.4f}")
    except ValueError:
        pass

    # Train final model on ALL data
    final_clf = RandomForestClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced",
        random_state=args.random_state
    )
    final_clf.fit(X, y)

    # Save model & metadata
    joblib.dump(final_clf, outdir / "rf_model.joblib")
    pd.Series(feat_names, name="feature").to_csv(outdir / "feature_columns.csv", index=False)
    (outdir / "seq_meta.txt").write_text(
        "\n".join([
            f"W={args.W}", f"H={args.H}", f"skip={args.skip}",
            f"target_mode={args.target_mode}", f"threshold={args.threshold}",
            f"random_state={args.random_state}", f"csv_path={Path(csv_path).resolve()}"
        ]), encoding="utf-8"
    )

    
    
    # --- Fold-wise permutation importance averaged & aggregated by BIU ---
    print("\nFold-wise permutation importance (averaged across CV folds)…")
    
    # Map feature index -> BIU name and lag
    def parse_feat(name: str):
        # "feat[t-<lag>].<BIU_NAME>"
        lag = name.split("]")[0].split("t-")[-1]
        biu = name.split("].")[-1]
        return lag, biu
    feat_lag = [parse_feat(n) for n in feat_names]
    biu_names = sorted({biu for _, biu in feat_lag})
    biu_to_idx = {biu: [i for i,(lag,b) in enumerate(feat_lag) if b == biu] for biu in biu_names}
    
    gkf = GroupKFold(n_splits=n_splits)
    fold_imps = []
    
    fold = 1
    for tr, te in gkf.split(X, y, groups):
        clf = RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, class_weight="balanced", random_state=args.random_state + 100 + fold
        )
        clf.fit(X[tr], y[tr])

        # Subsample the test fold to keep it light (optional)
        te_idx = te
        if len(te_idx) > 6000:
            rng = np.random.RandomState(args.random_state + fold)
            te_idx = rng.choice(te_idx, size=6000, replace=False)

        pi = permutation_importance(
            clf, X[te_idx], y[te_idx],
            n_repeats=3, random_state=args.random_state + fold, n_jobs=1  # n_jobs=1 avoids nested parallelism
        )
        # Aggregate by BIU (sum importances of all lags belonging to that BIU)
        agg = {biu: float(np.maximum(0, pi.importances_mean[biu_to_idx[biu]]).sum())
            for biu in biu_names}
        fold_imps.append(pd.Series(agg, name=f"fold{fold}"))
        fold += 1
        
    imp_df = pd.concat(fold_imps, axis=1).fillna(0.0)
    imp_df["mean_importance"] = imp_df.mean(axis=1)
    imp_df = imp_df.sort_values("mean_importance", ascending=False)    
    
    imp_df.to_csv(outdir / "importance_by_biu_cv.csv")
    print("\nTop BIUs by CV-averaged permutation importance:")
    print(imp_df["mean_importance"].sort_values(ascending=False).head(10))
    
if __name__ == "__main__":
    main()
