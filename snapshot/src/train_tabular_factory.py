import argparse, json, time, sys
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
import joblib

# Optional libs
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

# Your helpers
from seq_builder import load_csv, build_sequences

CSV_PATH = "train_11.csv"

def parse_feat(name: str):
    # expects "feat[t-<lag>].<BIU_NAME>"
    lag = name.split("]")[0].split("t-")[-1]
    biu = name.split("].")[-1]
    return lag, biu

def build_model(model_name, args, pos_weight):
    """Return (estimator, fit_kwargs, predict_proba_fn, model_label)."""
    model_name = model_name.lower()
    fit_kwargs = {}
    label = model_name

    if model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
            min_samples_leaf=args.rf_min_leaf,
            n_jobs=-1,
            class_weight="balanced",
            random_state=args.random_state
        )
        def _predict_proba(m, X): return m.predict_proba(X)[:, 1]
        return clf, fit_kwargs, _predict_proba, label

    if model_name == "xgb":
        if xgb is None:
            sys.exit("xgboost not installed. pip install xgboost")
        clf = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method=args.xgb_tree_method,  # auto | hist | gpu_hist
            n_estimators=args.xgb_n_estimators,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_lr,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample,
            min_child_weight=args.xgb_min_child_weight,
            reg_lambda=args.xgb_reg_lambda,
            n_jobs=-1,
            random_state=args.random_state,
            scale_pos_weight=pos_weight
        )
        def _predict_proba(m, X): return m.predict_proba(X)[:, 1]
        return clf, fit_kwargs, _predict_proba, f"xgb_{args.xgb_tree_method}"

    if model_name == "lgbm":
        if lgb is None:
            sys.exit("lightgbm not installed. pip install lightgbm")
        clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=args.lgb_n_estimators,
            learning_rate=args.lgb_lr,
            max_depth=args.lgb_max_depth,
            num_leaves=args.lgb_num_leaves,
            subsample=args.lgb_bagging_fraction,
            colsample_bytree=args.lgb_feature_fraction,
            reg_lambda=args.lgb_lambda_l2,
            min_child_samples=args.lgb_min_data_in_leaf,
            n_jobs=-1,
            random_state=args.random_state,
            # handles imbalance similarly to xgb:
            scale_pos_weight=pos_weight,
            # metric will be set in fit via eval_metric
        )
        # LightGBM sklearn wrapper uses 'eval_metric' in fit
        fit_kwargs["eval_metric"] = "aucpr"
        def _predict_proba(m, X): return m.predict_proba(X)[:, 1]
        return clf, fit_kwargs, _predict_proba, "lgbm"

    if model_name == "cat":
        if CatBoostClassifier is None:
            sys.exit("catboost not installed. pip install catboost")
        # For numeric-only features this works well; class_weights combats imbalance
        clf = CatBoostClassifier(
            iterations=args.cat_iterations,
            depth=args.cat_depth,
            learning_rate=args.cat_lr,
            l2_leaf_reg=args.cat_l2_leaf_reg,
            loss_function="Logloss",
            eval_metric="PRAUC",
            random_seed=args.random_state,
            verbose=False,
            class_weights=[1.0, pos_weight]  # [w0, w1]
        )
        def _predict_proba(m, X): return m.predict_proba(X)[:, 1]
        return clf, fit_kwargs, _predict_proba, "cat"

    sys.exit(f"Unknown --model {model_name}. Use rf|xgb|lgbm|cat.")

def main():
    ap = argparse.ArgumentParser()
    # Windowing/labels (same as your RF script)
    ap.add_argument("--W", type=int, default=10, help="history window (kept frames)")
    ap.add_argument("--H", type=int, default=5, help="horizon (future sampled steps)")
    ap.add_argument("--skip", type=int, default=5, help="sampling stride (native 30fps / skip)")
    ap.add_argument("--target_mode", type=str, default="any",
                    choices=["any","all","majority","next"])
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)

    # Model selector
    ap.add_argument("--model", type=str, default="xgb",
                    choices=["rf","xgb","lgbm","cat"])

    # RF params
    ap.add_argument("--rf_n_estimators", type=int, default=800)
    ap.add_argument("--rf_max_depth", type=int, default=None)
    ap.add_argument("--rf_min_leaf", type=int, default=2)

    # XGB params
    ap.add_argument("--xgb_tree_method", type=str, default="hist",
                    choices=["auto","hist","gpu_hist"])
    ap.add_argument("--xgb_n_estimators", type=int, default=2000)
    ap.add_argument("--xgb_max_depth", type=int, default=6)
    ap.add_argument("--xgb_lr", type=float, default=0.05)
    ap.add_argument("--xgb_subsample", type=float, default=0.9)
    ap.add_argument("--xgb_colsample", type=float, default=0.8)
    ap.add_argument("--xgb_min_child_weight", type=float, default=1.0)
    ap.add_argument("--xgb_reg_lambda", type=float, default=1.0)

    # LGBM params
    ap.add_argument("--lgb_n_estimators", type=int, default=2000)
    ap.add_argument("--lgb_lr", type=float, default=0.05)
    ap.add_argument("--lgb_max_depth", type=int, default=-1)
    ap.add_argument("--lgb_num_leaves", type=int, default=63)
    ap.add_argument("--lgb_feature_fraction", type=float, default=0.8)
    ap.add_argument("--lgb_bagging_fraction", type=float, default=0.9)
    ap.add_argument("--lgb_min_data_in_leaf", type=int, default=100)
    ap.add_argument("--lgb_lambda_l2", type=float, default=0.0)

    # CatBoost params
    ap.add_argument("--cat_iterations", type=int, default=2000)
    ap.add_argument("--cat_depth", type=int, default=8)
    ap.add_argument("--cat_lr", type=float, default=0.05)
    ap.add_argument("--cat_l2_leaf_reg", type=float, default=3.0)

    # Misc
    ap.add_argument("--outdir", type=str, default="../results/factory_runs")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--early_stopping_rounds", type=int, default=50)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--do_perm_imp", action="store_true",
                    help="compute permutation importance aggregated by BIU (slow)")
    ap.add_argument("--rf_fit_full_fold", action="store_true",
                help="If set, train RF on the entire training fold (no val split, no threshold tuning).")
    ap.add_argument("--tune_threshold", action="store_true",
                help="tune decision threshold on the fold's validation set for max-F1")
    ap.add_argument("--calibration", type=str, default="none",
                choices=["none","isotonic","platt"],
                help="post-hoc probability calibration on the validation set")


    args = ap.parse_args()

    outdir = Path(args.outdir) / args.model
    outdir.mkdir(parents=True, exist_ok=True)
    
    
    # ====== Helper functions ======
    from sklearn.metrics import precision_recall_curve
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    def find_best_f1_threshold(y_true, p):
        # Use PR curve to scan thresholds; maximize F1
        prec, rec, thr = precision_recall_curve(y_true, p)
        # Drop last point (threshold array is len-1 of prec/rec)
        prec, rec = prec[:-1], rec[:-1]
        f1 = (2*prec*rec) / np.maximum(prec+rec, 1e-9)
        i = int(np.nanargmax(f1))
        return float(thr[i]), float(f1[i]), float(prec[i]), float(rec[i])

    def calibrate_probs(y_val, p_val, method="none"):
        if method == "none":
            return lambda z: z
        if method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_val, y_val)
            return lambda z: iso.predict(z)
        if method == "platt":
            # Platt scaling on logit of p
            z = np.clip(p_val, 1e-6, 1-1e-6)
            logit = np.log(z/(1-z)).reshape(-1,1)
            lr = LogisticRegression(C=1000.0, solver="lbfgs", max_iter=1000)
            lr.fit(logit, y_val)
            return lambda zz: lr.predict_proba(np.log(np.clip(zz,1e-6,1-1e-6)/(1-np.clip(zz,1e-6,1-1e-6))).reshape(-1,1))[:,1]
    

    # === Load & build sequences ===
    df = load_csv(CSV_PATH)
    X, y, groups, feat_names, meta = build_sequences(
        df, W=args.W, H=args.H, skip=args.skip, target_mode=args.target_mode
    )
    print(f"Dataset: X={X.shape}, positives={y.sum()}/{len(y)} ({y.mean()*100:.2f}%), "
          f"groups={len(np.unique(groups))}")

    # Save features & meta
    pd.Series(feat_names, name="feature").to_csv(outdir / "feature_columns.csv", index=False)
    (outdir / "seq_meta.txt").write_text(
        "\n".join([
            f"W={args.W}", f"H={args.H}", f"skip={args.skip}",
            f"target_mode={args.target_mode}", f"threshold={args.threshold}",
            f"random_state={args.random_state}", f"csv_path={Path(CSV_PATH).resolve()}",
            f"model={args.model}"
        ]), encoding="utf-8"
    )

    # CV
    n_groups = len(np.unique(groups))
    n_splits = min(args.splits, max(2, n_groups))
    gkf = GroupKFold(n_splits=n_splits)

    y_pred_all = np.zeros_like(y)
    y_proba_all = np.zeros_like(y, dtype=float)
    fold_reports, all_pred_rows, runtime_rows = [], [], []

    fold = 1
    for tr, te in gkf.split(X, y, groups):
        # --- decide if we need a validation split ---
        need_val = (args.model in ("xgb","lgbm","cat"))  # boosted models use early stopping
        # RF only needs a val split if you plan to tune threshold or calibrate
        if args.model == "rf":
            need_val = (not args.rf_fit_full_fold) and (args.tune_threshold or args.calibration != "none")
        if need_val:    
        # small val split from training fold (for early stopping when available)
            X_tr, X_val, y_tr, y_val = train_test_split(
                 X[tr], y[tr], test_size=args.val_size, stratify=y[tr],
                 random_state=args.random_state + 10*fold
        )
        else:
            X_tr, y_tr = X[tr], y[tr]
            X_val, y_val = None, None

        pos = max(1, int(y_tr.sum()))
        neg = int(len(y_tr) - pos)
        pos_weight = neg / pos

        clf, fit_kwargs, predict_proba_fn, label = build_model(args.model, args, pos_weight)
        
        if args.model == "rf" and args.rf_fit_full_fold:
        # match your old script's per-fold seed
            clf.set_params(random_state=args.random_state + fold)
            X_fit, y_fit = X[tr], y[tr]     # full training fold
        else:
            X_fit, y_fit = X_tr, y_tr

        # add early stopping if lib supports eval_set
        supports_early_stop = hasattr(clf, "fit") and "eval_set" in clf.fit.__code__.co_varnames
        t0 = time.perf_counter()
        if supports_early_stop and args.model in ("xgb","lgbm","cat"):
            fit_kwargs_local = dict(fit_kwargs)  # copy
            if args.model == "lgbm":
                fit_kwargs_local["eval_set"] = [(X_val, y_val)]
                fit_kwargs_local["callbacks"] = [lgb.early_stopping(args.early_stopping_rounds, verbose=False)]
            elif args.model == "xgb":
                fit_kwargs_local["eval_set"] = [(X_val, y_val)]
                fit_kwargs_local["verbose"] = False
                fit_kwargs_local["early_stopping_rounds"] = args.early_stopping_rounds
            elif args.model == "cat":
                fit_kwargs_local["eval_set"] = [(X_val, y_val)]
                fit_kwargs_local["early_stopping_rounds"] = args.early_stopping_rounds
                fit_kwargs_local["verbose"] = False
            clf.fit(X_tr, y_tr, **fit_kwargs_local)
        else:
            clf.fit(X_tr, y_tr)
        train_time = time.perf_counter() - t0
        
        # --- validation-based calibration / threshold tuning (optional) ---
        # If you trained RF on full fold, we **should not** tune threshold on a val subset
        # (it would leak because model saw that data). So just use args.threshold.
        if args.model == "rf" and args.rf_fit_full_fold:
            cal_map = lambda z: z
            thr_used = args.threshold
        else:
            # safe: tune on X_val (if present), else fall back to fixed threshold
            p_val = predict_proba_fn(clf, X_val) if X_val is not None else None
            cal_map = calibrate_probs(y_val, p_val, method=args.calibration) if (X_val is not None) else (lambda z: z)
            if args.tune_threshold and X_val is not None:
                thr_used, _, _, _ = find_best_f1_threshold(y_val, cal_map(p_val))
            else:
                thr_used = args.threshold

        # inference timing
        t1 = time.perf_counter()
        proba = predict_proba_fn(clf, X[te])
        infer_time = time.perf_counter() - t1
        ms_per_sample = (infer_time / len(te)) * 1000.0

        pred = (proba >= args.threshold).astype(int)

        # metrics
        rep = classification_report(y[te], pred, digits=3, output_dict=True)
        try:
            roc = roc_auc_score(y[te], proba)
            pr  = average_precision_score(y[te], proba)
        except ValueError:
            roc, pr = None, None

        print(f"\n=== Fold {fold} ({args.model}) ===")
        print(pd.DataFrame(rep).T.round(3))
        print(f"ROC-AUC: {None if roc is None else f'{roc:.4f}'} | "
              f"PR-AUC: {None if pr  is None else f'{pr:.4f}'} | "
              f"train {train_time:.3f}s | infer {infer_time:.3f}s "
              f"({ms_per_sample:.3f} ms/sample)")

        # collect
        y_pred_all[te] = pred
        y_proba_all[te] = proba
        fold_reports.append({
            "fold": fold,
            "report": rep,
            "roc_auc": roc, "pr_auc": pr,
            "train_time_sec": train_time,
            "infer_time_sec": infer_time,
            "ms_per_sample": ms_per_sample
        })
        all_pred_rows.append(pd.DataFrame({
            "index": te,
            "group": groups[te],
            "y_true": y[te],
            "y_proba": proba,
            "y_pred": pred,
            "fold": fold
        }))
        runtime_rows.append([fold, train_time, infer_time, ms_per_sample])
        fold += 1

    # overall
    overall_report = classification_report(y, y_pred_all, digits=3, output_dict=True)
    try:
        overall_roc = roc_auc_score(y, y_proba_all)
        overall_pr  = average_precision_score(y, y_proba_all)
    except ValueError:
        overall_roc, overall_pr = None, None

    # save metrics & predictions
    with open(outdir / "fold_metrics.json", "w", encoding="utf-8") as f:
        json.dump(fold_reports, f, indent=2)
    with open(outdir / "overall_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "overall_report": overall_report,
            "roc_auc": overall_roc,
            "pr_auc": overall_pr,
            "threshold": args.threshold,
            "W": args.W, "H": args.H, "skip": args.skip, "target_mode": args.target_mode
        }, f, indent=2)
    pd.concat(all_pred_rows, ignore_index=True).to_csv(outdir / "cv_predictions.csv", index=False)
    pd.DataFrame(runtime_rows, columns=["fold","train_time_sec","infer_time_sec","ms_per_sample"])\
        .to_csv(outdir / "runtimes.csv", index=False)

    # train final model on all data (with small val for early stopping if possible)
    X_tr_all, X_val_all, y_tr_all, y_val_all = train_test_split(
        X, y, test_size=args.val_size, stratify=y, random_state=args.random_state + 777
    )
    pos_all = max(1, int(y_tr_all.sum())); neg_all = int(len(y_tr_all) - pos_all)
    pos_weight_all = neg_all / pos_all
    clf_all, fit_kwargs_all, predict_proba_fn_all, label_all = build_model(args.model, args, pos_weight_all)

    t0 = time.perf_counter()
    supports_early_stop = hasattr(clf_all, "fit") and "eval_set" in clf_all.fit.__code__.co_varnames
    if supports_early_stop and args.model in ("xgb","lgbm","cat"):
        if args.model == "lgbm":
            clf_all.fit(X_tr_all, y_tr_all,
                        eval_set=[(X_val_all, y_val_all)],
                        eval_metric="aucpr",
                        callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False)])
        elif args.model == "xgb":
            clf_all.fit(X_tr_all, y_tr_all,
                        eval_set=[(X_val_all, y_val_all)],
                        verbose=False,
                        early_stopping_rounds=args.early_stopping_rounds)
        elif args.model == "cat":
            clf_all.fit(X_tr_all, y_tr_all,
                        eval_set=[(X_val_all, y_val_all)],
                        early_stopping_rounds=args.early_stopping_rounds,
                        verbose=False)
    else:
        clf_all.fit(X_tr_all, y_tr_all)
    train_all_time = time.perf_counter() - t0

    joblib.dump(clf_all, outdir / f"{args.model}_model.joblib")
    (outdir / "final_train_time_sec.txt").write_text(f"{train_all_time:.6f}", encoding="utf-8")

    # Optional: permutation importance aggregated by BIU
    if args.do_perm_imp:
        print("\nComputing permutation importance aggregated by BIU…")
        feat_lag = [parse_feat(n) for n in feat_names]
        biu_names = sorted({biu for _, biu in feat_lag})
        biu_to_idx = {biu: [i for i,(lag,b) in enumerate(feat_lag) if b == biu] for biu in biu_names}

        # Use a fresh fold again (small models keep memory sane)
        gkf2 = GroupKFold(n_splits=n_splits)
        fold_imps = []
        rnd = args.random_state + 888
        for k, (tr2, te2) in enumerate(gkf2.split(X, y, groups), start=1):
            # small val for early stopping where available
            X_tr2, X_val2, y_tr2, y_val2 = train_test_split(X[tr2], y[tr2], test_size=args.val_size,
                                                            stratify=y[tr2], random_state=rnd + k)
            pos2 = max(1, int(y_tr2.sum())); neg2 = int(len(y_tr2) - pos2)
            pos_weight2 = neg2 / pos2
            clf2, fit_kwargs2, predict_proba_fn2, _ = build_model(args.model, args, pos_weight2)
            if hasattr(clf2, "fit") and "eval_set" in clf2.fit.__code__.co_varnames and args.model in ("xgb","lgbm","cat"):
                if args.model == "lgbm":
                    clf2.fit(X_tr2, y_tr2,
                             eval_set=[(X_val2, y_val2)],
                             eval_metric="aucpr",
                             callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False)])
                elif args.model == "xgb":
                    clf2.fit(X_tr2, y_tr2,
                             eval_set=[(X_val2, y_val2)],
                             verbose=False,
                             early_stopping_rounds=args.early_stopping_rounds)
                elif args.model == "cat":
                    clf2.fit(X_tr2, y_tr2,
                             eval_set=[(X_val2, y_val2)],
                             early_stopping_rounds=args.early_stopping_rounds,
                             verbose=False)
            else:
                clf2.fit(X_tr2, y_tr2)

            te_idx = te2
            if len(te_idx) > 6000:
                rng = np.random.RandomState(rnd + k)
                te_idx = rng.choice(te_idx, size=6000, replace=False)

            pi = permutation_importance(
                clf2, X[te_idx], y[te_idx],
                n_repeats=3, random_state=rnd + k, n_jobs=1,
                scoring="average_precision"
            )
            agg = {biu: float(np.maximum(0, pi.importances_mean[biu_to_idx[biu]]).sum())
                   for biu in biu_names}
            fold_imps.append(pd.Series(agg, name=f"fold{k}"))

        imp_df = pd.concat(fold_imps, axis=1).fillna(0.0)
        imp_df["mean_importance"] = imp_df.mean(axis=1)
        imp_df.sort_values("mean_importance", ascending=False)\
              .to_csv(outdir / "importance_by_biu_cv.csv")

if __name__ == "__main__":
    main()