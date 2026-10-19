import argparse, json, time, sys
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
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

from seq_builder import load_csv, build_sequences

# CSV_PATH = "train_10_enriched.csv"

class XGBNative:
    """
    Minimal wrapper around xgboost.train so it behaves like an estimator with
    .fit(X,y, eval_set=[(X_val,y_val)]) and .predict_proba(X).
    """
    def __init__(self, params, num_boost_round=2000, early_stopping_rounds=50):
        import xgboost as xgb  # local import to bind the right package
        self.xgb = xgb
        self.params = params
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.booster = None

    def fit(self, X, y, eval_set=None, verbose=False):
        dtrain = self.xgb.DMatrix(X, label=y)
        evals = []
        if eval_set:
            Xv, yv = eval_set[0]
            dval = self.xgb.DMatrix(Xv, label=yv)
            evals = [(dval, "validation")]
        self.booster = self.xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds if evals else None,
            verbose_eval=False,
        )
        return self

    def predict_proba(self, X):
        dm = self.xgb.DMatrix(X)
        # Use best_iteration if available; fall back to full model.
        kwargs = {}
        best_iter = getattr(self.booster, "best_iteration", None)
        if best_iter is not None:
            # predict using trees [0, best_iter]
            kwargs["iteration_range"] = (0, int(best_iter) + 1)
        p = self.booster.predict(dm, **kwargs).ravel()
        # return 2-col array like sklearn: [P(class0), P(class1)]
        return np.column_stack([1.0 - p, p])
    
    # optional: make pickling robust even if you later add attrs
    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("xgb", None)  # in case it ever sneaks back
        return d


def parse_feat(name: str):
    lag = name.split("]")[0].split("t-")[-1]
    biu = name.split("].")[-1]
    return lag, biu

def find_best_f1_threshold(y_true, p):
    prec, rec, thr = precision_recall_curve(y_true, p)
    prec, rec = prec[:-1], rec[:-1]
    f1 = (2*prec*rec) / np.maximum(prec+rec, 1e-9)
    i = int(np.nanargmax(f1))
    return float(thr[i]), float(f1[i]), float(prec[i]), float(rec[i])

def calibrate_probs(y_val, p_val, method="none"):
    if method == "none" or y_val is None or p_val is None:
        return lambda z: z
    p_val = np.asarray(p_val).ravel()
    y_val = np.asarray(y_val).ravel()
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_val, y_val)
        return lambda z: iso.predict(np.asarray(z).ravel())
    if method == "platt":
        z = np.clip(p_val, 1e-6, 1-1e-6)
        logit = np.log(z/(1-z)).reshape(-1,1)
        lr = LogisticRegression(C=1000.0, solver="lbfgs", max_iter=1000)
        lr.fit(logit, y_val)
        def map_fn(zz):
            zz = np.clip(np.asarray(zz).ravel(), 1e-6, 1-1e-6)
            logit_zz = np.log(zz/(1-zz)).reshape(-1,1)
            return lr.predict_proba(logit_zz)[:,1]
        return map_fn
    return lambda z: z

def build_model(model_name, args, pos_weight):
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
       # Use native xgboost.train. Early stopping is handled inside XGBNative.
        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": args.xgb_tree_method,  # "hist" or "gpu_hist"
            "eta": args.xgb_lr,
            "max_depth": args.xgb_max_depth,
            "subsample": args.xgb_subsample,
            "colsample_bytree": args.xgb_colsample,
            "min_child_weight": args.xgb_min_child_weight,
            "lambda": args.xgb_reg_lambda,
            "nthread": -1,
            "seed": args.random_state,
            "scale_pos_weight": pos_weight,
        }
        clf = XGBNative(
            params=params,
            num_boost_round=args.xgb_n_estimators,
            early_stopping_rounds=args.early_stopping_rounds
        )
        def _predict_proba(m, X): return m.predict_proba(X)[:, 1]
        return clf, {}, _predict_proba, f"xgb_{args.xgb_tree_method}"

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
            scale_pos_weight=pos_weight
        )
        # fit_kwargs["eval_metric"] = "aucpr"
        def _predict_proba(m, X): return m.predict_proba(X)[:, 1]
        return clf, fit_kwargs, _predict_proba, "lgbm"

    if model_name == "cat":
        if CatBoostClassifier is None:
            sys.exit("catboost not installed. pip install catboost")
        clf = CatBoostClassifier(
            iterations=args.cat_iterations,
            depth=args.cat_depth,
            learning_rate=args.cat_lr,
            l2_leaf_reg=args.cat_l2_leaf_reg,
            loss_function="Logloss",
            eval_metric="PRAUC",
            random_seed=args.random_state,
            verbose=False,
            class_weights=[1.0, pos_weight]
        )
        def _predict_proba(m, X): return m.predict_proba(X)[:, 1]
        return clf, fit_kwargs, _predict_proba, "cat"

    sys.exit(f"Unknown --model {model_name}. Use rf|xgb|lgbm|cat.")

def main():
    import xgboost as xgb, lightgbm as lgb, sklearn
    print("Runtime versions → xgboost:", xgb.__version__, "| lightgbm:", lgb.__version__, "| sklearn:", sklearn.__version__)

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="train_10_enriched.csv",
                help="Path to features CSV (use enriched file with temporal features)")
    # windowing/labels

    ap.add_argument("--W", type=int, default=10)
    ap.add_argument("--H", type=int, default=5)
    ap.add_argument("--skip", type=int, default=5)
    ap.add_argument("--target_mode", type=str, default="any",
                    choices=["any","all","majority","next"])
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)

    # model selector
    ap.add_argument("--model", type=str, default="xgb",
                    choices=["rf","xgb","lgbm","cat"])

    # RF params
    ap.add_argument("--rf_n_estimators", type=int, default=800)
    ap.add_argument("--rf_max_depth", type=int, default=None)
    ap.add_argument("--rf_min_leaf", type=int, default=2)
    ap.add_argument("--rf_fit_full_fold", action="store_true",
                    help="Train RF on the entire training fold (no inner val split).")

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

    # extras
    ap.add_argument("--outdir", type=str, default="../results/factory_runs")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--early_stopping_rounds", type=int, default=50)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--do_perm_imp", action="store_true",
                    help="compute permutation importance aggregated by BIU (slow)")

    # NEW FLAGS (this is what your error complained about)
    ap.add_argument("--tune_threshold", action="store_true",
                    help="tune decision threshold on the fold's validation split for max-F1")
    ap.add_argument("--calibration", type=str, default="none",
                    choices=["none","isotonic","platt"],
                    help="post-hoc probability calibration on the validation split")

    args = ap.parse_args()

    outdir = Path(args.outdir) / args.model
    outdir.mkdir(parents=True, exist_ok=True)

    # load & build sequences
    df = load_csv(args.csv)
    X, y, groups, feat_names, meta = build_sequences(
        df, W=args.W, H=args.H, skip=args.skip, target_mode=args.target_mode
    )
    print(f"Dataset: X={X.shape}, positives={y.sum()}/{len(y)} ({y.mean()*100:.2f}%), "
          f"groups={len(np.unique(groups))}")

    pd.Series(feat_names, name="feature").to_csv(outdir / "feature_columns.csv", index=False)
    (outdir / "seq_meta.txt").write_text(
        "\n".join([
            f"W={args.W}", f"H={args.H}", f"skip={args.skip}",
            f"target_mode={args.target_mode}", f"threshold={args.threshold}",
            f"random_state={args.random_state}", f"csv_path={Path(args.csv).resolve()}",
            f"model={args.model}", f"calibration={args.calibration}",
            f"tune_threshold={args.tune_threshold}"
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
        # need validation?
        need_val = (args.model in ("xgb","lgbm","cat"))
        if args.model == "rf":
            need_val = (not args.rf_fit_full_fold) and (args.tune_threshold or args.calibration != "none")

        if need_val:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X[tr], y[tr], test_size=args.val_size, stratify=y[tr],
                random_state=args.random_state + 10*fold
            )
            y_train_fold = y_tr
        else:
            X_tr, y_tr = X[tr], y[tr]
            X_val, y_val = None, None
            y_train_fold = y[tr]

        pos = int(y_train_fold.sum()); neg = int(len(y_train_fold) - pos)
        pos_weight = (neg / max(pos, 1))

        clf, fit_kwargs, predict_proba_fn, label = build_model(args.model, args, pos_weight)

        # RF full fold like your original script
        if args.model == "rf" and args.rf_fit_full_fold:
            clf.set_params(random_state=args.random_state + fold)
            X_fit, y_fit = X[tr], y[tr]
        else:
            X_fit, y_fit = X_tr, y_tr

        # fit
        t0 = time.perf_counter()
        if args.model in ("xgb","lgbm","cat") and X_val is not None:
            
            
            if args.model == "lgbm":
                clf.fit(X_fit, y_fit,
                        eval_set=[(X_val, y_val)],
                        eval_metric="aucpr",
                        callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False)],
                        **fit_kwargs)
            elif args.model == "xgb":
                
                # XGB branch
                clf.fit(
                    X_fit, y_fit,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
            elif args.model == "cat":
                clf.fit(X_fit, y_fit,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=args.early_stopping_rounds,
                        verbose=False,
                        **fit_kwargs)
        else:
            clf.fit(X_fit, y_fit)
        train_time = time.perf_counter() - t0

        # calibration & threshold
        if args.model == "rf" and args.rf_fit_full_fold:
            cal_map = lambda z: z
            thr_used = args.threshold
        else:
            p_val = predict_proba_fn(clf, X_val) if X_val is not None else None
            cal_map = calibrate_probs(y_val, p_val, method=args.calibration) if (X_val is not None) else (lambda z: z)
            if args.tune_threshold and X_val is not None:
                thr_used, _, _, _ = find_best_f1_threshold(y_val, cal_map(p_val))
            else:
                thr_used = args.threshold

        # test inference
        t1 = time.perf_counter()
        proba = predict_proba_fn(clf, X[te])
        proba = cal_map(proba)
        infer_time = time.perf_counter() - t1
        ms_per_sample = (infer_time / len(te)) * 1000.0

        pred = (proba >= thr_used).astype(int)

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
              f"train {train_time:.3f}s | infer {infer_time:.3f}s ({ms_per_sample:.3f} ms/sample)")

        y_pred_all[te] = pred
        y_proba_all[te] = proba
        fold_reports.append({
            "fold": fold,
            "report": rep,
            "roc_auc": roc, "pr_auc": pr,
            "train_time_sec": train_time,
            "infer_time_sec": infer_time,
            "ms_per_sample": ms_per_sample,
            "threshold_used": thr_used,
            "calibration": args.calibration
        })
        all_pred_rows.append(pd.DataFrame({
            "index": te, "group": groups[te],
            "y_true": y[te], "y_proba": proba, "y_pred": pred, "fold": fold
        }))
        runtime_rows.append([fold, train_time, infer_time, ms_per_sample])
        fold += 1

    overall_report = classification_report(y, y_pred_all, digits=3, output_dict=True)
    try:
        overall_roc = roc_auc_score(y, y_proba_all)
        overall_pr  = average_precision_score(y, y_proba_all)
    except ValueError:
        overall_roc, overall_pr = None, None

    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "fold_metrics.json", "w", encoding="utf-8") as f:
        json.dump(fold_reports, f, indent=2)
    with open(outdir / "overall_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "overall_report": overall_report,
            "roc_auc": overall_roc,
            "pr_auc": overall_pr,
            "threshold": args.threshold,
            "W": args.W, "H": args.H, "skip": args.skip, "target_mode": args.target_mode,
            "tuned_threshold": bool(args.tune_threshold),
            "calibration": args.calibration
        }, f, indent=2)

    pd.concat(all_pred_rows, ignore_index=True).to_csv(outdir / "cv_predictions.csv", index=False)
    pd.DataFrame(runtime_rows, columns=["fold","train_time_sec","infer_time_sec","ms_per_sample"])\
        .to_csv(outdir / "runtimes.csv", index=False)

    # final model (train on all with small val for boosted models)
    X_tr_all, X_val_all, y_tr_all, y_val_all = train_test_split(
        X, y, test_size=args.val_size, stratify=y, random_state=args.random_state + 777
    )
    pos_all = int(y_tr_all.sum()); neg_all = int(len(y_tr_all) - pos_all)
    pos_weight_all = (neg_all / max(pos_all,1))
    clf_all, fit_kwargs_all, predict_proba_fn_all, label_all = build_model(args.model, args, pos_weight_all)

    t0 = time.perf_counter()
    if args.model in ("xgb","lgbm","cat"):
        if args.model == "lgbm":
            clf_all.fit(X_tr_all, y_tr_all,
                        eval_set=[(X_val_all, y_val_all)],
                        eval_metric="aucpr",
                        callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False)],
                        **fit_kwargs_all)
        elif args.model == "xgb":
            # XGB branch
                clf_all.fit(
                    X_tr_all, y_tr_all,
                    eval_set=[(X_val_all, y_val_all)],
                    verbose=False
                )
        elif args.model == "cat":
            clf_all.fit(X_tr_all, y_tr_all,
                        eval_set=[(X_val_all, y_val_all)],
                        early_stopping_rounds=args.early_stopping_rounds,
                        verbose=False,
                        **fit_kwargs_all)
    else:
        clf_all.fit(X, y)
    train_all_time = time.perf_counter() - t0
    joblib.dump(clf_all, outdir / f"{args.model}_model.joblib")
    (outdir / "final_train_time_sec.txt").write_text(f"{train_all_time:.6f}", encoding="utf-8")

    # optional permutation importance aggregated by BIU
    if args.do_perm_imp:
        print("\nComputing permutation importance aggregated by BIU…")
        feat_lag = [parse_feat(n) for n in feat_names]
        biu_names = sorted({biu for _, biu in feat_lag})
        biu_to_idx = {biu: [i for i,(lag,b) in enumerate(feat_lag) if b == biu] for biu in biu_names}
        gkf2 = GroupKFold(n_splits=n_splits)
        fold_imps = []
        rnd = args.random_state + 888
        for k, (tr2, te2) in enumerate(gkf2.split(X, y, groups), start=1):
            Xt, Xv, yt, yv = train_test_split(X[tr2], y[tr2], test_size=args.val_size,
                                              stratify=y[tr2], random_state=rnd + k)
            pos2 = int(yt.sum()); neg2 = int(len(yt) - pos2)
            pw2  = (neg2 / max(pos2,1))
            clf2, fkw2, ppf2, _ = build_model(args.model, args, pw2)
            if args.model in ("xgb","lgbm","cat"):
                if args.model == "lgbm":
                    clf2.fit(Xt, yt, eval_set=[(Xv, yv)], eval_metric="aucpr",
                             callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False)], **fkw2)
                elif args.model == "xgb":
                    clf2.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
                elif args.model == "cat":
                    clf2.fit(Xt, yt, eval_set=[(Xv, yv)], early_stopping_rounds=args.early_stopping_rounds,
                             verbose=False, **fkw2)
            else:
                clf2.fit(Xt, yt)

            te_idx = te2
            if len(te_idx) > 6000:
                rng = np.random.RandomState(rnd + k)
                te_idx = rng.choice(te_idx, size=6000, replace=False)

            pi = permutation_importance(
                clf2, X[te_idx], y[te_idx], n_repeats=3,
                random_state=rnd + k, n_jobs=1, scoring="average_precision"
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
