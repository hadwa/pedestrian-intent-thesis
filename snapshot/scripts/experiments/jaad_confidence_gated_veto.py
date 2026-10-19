"""Experiment 020: confidence-gated Stage-2 veto for the JAAD staged pipeline.

Question: does making the staged BBox->BIU-v2 pipeline's Stage-2 veto
CONFIDENCE-AWARE improve it? Gated rule: BIU v2 may only veto a Stage-1
positive when the stage-2 canonical window's skeleton confidence is >= a gate
g; low-confidence windows keep the Stage-1 decision.

Design (all selection on validation only; test evaluated once per config):
- Reuses experiment 011's machinery verbatim (imported module): track-level
  canonical windows (exp-004 rule), seed-ensemble probs (42/43/44), exp-004
  three-group assignments, recall-target thresholds on validation, track
  bootstrap 10k seed 12345.
- Parity check: the ungated staged operating points are recomputed and must
  match experiments/011_jaad_staged_bbox_biu_v2/metrics/staged_operating_points.csv
  exactly before any gating is applied.
- Confidence signals (two variants), from experiment 014's window aggregates
  (experiments/014_confidence_aware_biu_audit/metrics/window_confidence_<h>.csv):
  skeleton_conf_mean and lower_body_kp_conf_mean of the stage-2 canonical window.
- Gate grid: train+val quantiles {0.1, 0.25, 0.5, 0.75} of the signal over all
  train+val windows of the stage-2 horizon (exp-014 convention; no test data).
  g chosen per (config, recall target, signal) to maximize VALIDATION F1 at the
  config's 011 thresholds; ties broken toward the SMALLEST quantile (weakest
  gate, i.e. deviate from ungated 011 only when validation shows a gain).
- Oracle-style DIAGNOSTIC (clearly labeled, not a result): the same gate grid
  swept on TEST to show the headroom of the gate regardless of the small
  validation set.
- Missing stage-2 window (no BIU-v2 features) keeps the Stage-1 decision, as in
  011. A present stage-2 window with missing confidence would be treated as
  low-confidence (no veto); the join check counts such tracks (expected 0).

Outputs (experiments/020_jaad_confidence_gated_veto/):
- metrics/parity_check_vs_011.csv       (ungated reproduction vs saved 011 rows)
- metrics/gate_quantile_values.csv      (horizon x signal x quantile -> g)
- metrics/val_gate_selection.csv        (validation F1 landscape + chosen g)
- metrics/gated_test_results.csv        (gated vs ungated test metrics per config/signal)
- metrics/decision_change_table.csv     (counts of gate-changed tracks by group/correctness)
- metrics/decision_change_tracks.csv    (per-track detail of every changed decision)
- metrics/gated_bootstrap.csv           (track bootstrap gated vs ungated, 10k seed 12345)
- metrics/test_gate_sweep_diagnostic.csv (DIAGNOSTIC test sweep over the gate grid)
- figures/gate_sweep_f1_ambfpr.png      (F1 and amb-neg FPR vs gate quantile)
"""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

CLEAN_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean")
SCRIPTS_DIR = CLEAN_ROOT / "scripts/experiments"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import jaad_staged_bbox_biu_v2 as staged  # noqa: E402  (exp 011 module, reused verbatim)

OUT_ROOT = CLEAN_ROOT / "experiments/020_jaad_confidence_gated_veto"
METRICS_DIR = OUT_ROOT / "metrics"
FIGURES_DIR = OUT_ROOT / "figures"

CONF_DIR = CLEAN_ROOT / "experiments/014_confidence_aware_biu_audit/metrics"
OPS_011 = CLEAN_ROOT / "experiments/011_jaad_staged_bbox_biu_v2/metrics/staged_operating_points.csv"

STAGE1 = ("30_60", "bbox_only")
STAGE2_HORIZONS = ["15_30", "5_15"]
STAGE2_SETS = ["biu_v2_core", "biu_v2_core_plus_bbox"]
RECALL_TARGETS = [0.90, 0.95]
SIGNALS = ["skeleton_conf_mean", "lower_body_kp_conf_mean"]
GATE_QUANTILES = [0.10, 0.25, 0.50, 0.75]
METRIC_KEYS = ["f1", "precision", "recall", "amb_neg_fpr", "clean_neg_fpr"]


def load_confidence(horizon: str) -> pd.DataFrame:
    df = pd.read_csv(CONF_DIR / f"window_confidence_{horizon}.csv")
    return df[["sequence_id", "split"] + SIGNALS]


def gate_grid(conf: pd.DataFrame) -> dict[str, dict[float, float]]:
    """Train+val quantiles of each signal (all windows of the horizon; no test)."""
    tv = conf[conf["split"].isin(["train", "val"])]
    return {sig: {q: float(tv[sig].quantile(q)) for q in GATE_QUANTILES} for sig in SIGNALS}


def gated_staged_predict(s1: pd.DataFrame, s2: pd.DataFrame, conf_map: pd.Series,
                         t1: float, t2: float, g: float) -> pd.DataFrame:
    """Staged prediction where the Stage-2 veto applies only if window conf >= g.

    - Track missing from stage 2 entirely: keep Stage-1 decision (as in 011).
    - Stage-2 window present, conf >= g: normal veto (final = s1 AND s2).
    - Stage-2 window present, conf < g (or conf NaN): veto disabled, keep Stage-1.
    """
    joined = s1[["label", "group", "prob"]].rename(columns={"prob": "p1"}).join(
        s2[["prob", "sequence_id"]].rename(columns={"prob": "p2", "sequence_id": "s2_sequence_id"}),
        how="left",
    )
    joined["conf"] = joined["s2_sequence_id"].map(conf_map)
    stage1 = (joined["p1"] >= t1).astype(int)
    veto_allowed = joined["p2"].notna() & (joined["conf"] >= g)  # NaN conf -> False -> no veto
    stage2 = np.where(veto_allowed, (joined["p2"] >= t2).astype(int), 1)
    joined["pred"] = (stage1 & stage2).astype(int)
    joined["veto_allowed"] = veto_allowed.to_numpy()
    return joined


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    track_table = staged.build_track_table()

    # model choice per (horizon, feature_set): recompute with 011's rule and verify
    # against the saved 011 selection file.
    sel_011 = pd.read_csv(CLEAN_ROOT / "experiments/011_jaad_staged_bbox_biu_v2/metrics/staged_model_selection.csv")
    chosen: dict[tuple[str, str], str] = {}
    needed = [STAGE1] + list(product(STAGE2_HORIZONS, STAGE2_SETS))
    for h, fs in needed:
        m = staged.pick_model(track_table, h, fs)
        saved = sel_011[(sel_011["horizon_label"] == h) & (sel_011["feature_set"] == fs)]["chosen_model"].iloc[0]
        assert m == saved, f"model selection mismatch for {h}/{fs}: recomputed {m} vs saved {saved}"
        chosen[(h, fs)] = m

    s1_m = chosen[STAGE1]
    s1_val = staged.get_split(track_table, STAGE1[0], STAGE1[1], s1_m, "val")
    s1_test = staged.get_split(track_table, STAGE1[0], STAGE1[1], s1_m, "test")

    ops_011 = pd.read_csv(OPS_011)

    # confidence tables + gate grids per stage-2 horizon
    conf_tables = {h: load_confidence(h) for h in STAGE2_HORIZONS}
    grids = {h: gate_grid(conf_tables[h]) for h in STAGE2_HORIZONS}
    grid_rows = [
        {"horizon_label": h, "signal": sig, "quantile": q, "gate_value": g}
        for h in STAGE2_HORIZONS for sig, qs in grids[h].items() for q, g in qs.items()
    ]
    pd.DataFrame(grid_rows).to_csv(METRICS_DIR / "gate_quantile_values.csv", index=False)

    parity_rows, val_rows, result_rows, change_rows, change_track_rows, boot_rows, sweep_rows = \
        [], [], [], [], [], [], []

    for s2_h, s2_fs in product(STAGE2_HORIZONS, STAGE2_SETS):
        s2_m = chosen[(s2_h, s2_fs)]
        s2_val = staged.get_split(track_table, s2_h, s2_fs, s2_m, "val")
        s2_test = staged.get_split(track_table, s2_h, s2_fs, s2_m, "test")
        cfg = f"staged|bbox30_60->{s2_fs}@{s2_h}"
        conf = conf_tables[s2_h]
        conf_maps = {sig: conf.set_index("sequence_id")[sig] for sig in SIGNALS}

        for target in RECALL_TARGETS:
            # ---- thresholds exactly as in 011 (validation only, conditional stage-2)
            t1 = staged.threshold_for_recall(s1_val["label"].to_numpy(), s1_val["prob"].to_numpy(), target)
            passed = s1_val[s1_val["prob"] >= t1]
            s2_val_passed = s2_val.loc[s2_val.index.intersection(passed.index)]
            if s2_val_passed.empty or s2_val_passed["label"].nunique() < 2:
                t2 = 0.5
            else:
                t2 = staged.threshold_for_recall(
                    s2_val_passed["label"].to_numpy(), s2_val_passed["prob"].to_numpy(), target)

            # ---- parity check vs saved 011 operating points (ungated)
            ungated = staged.staged_predict(s1_test, s2_test, t1, t2)
            y = ungated["label"].to_numpy()
            grp = ungated["group"].to_numpy()
            pred_u = ungated["pred"].to_numpy()
            m_u = staged.track_metrics(y, pred_u, grp)
            saved = ops_011[(ops_011["config"] == cfg) & (ops_011["recall_target"] == target)].iloc[0]
            prow = {"config": cfg, "recall_target": target,
                    "t1_repro": t1, "t1_saved": saved["t1"],
                    "t2_repro": t2, "t2_saved": saved["t2"]}
            max_diff = max(abs(t1 - saved["t1"]), abs(t2 - saved["t2"]))
            for k in METRIC_KEYS:
                prow[f"{k}_repro"] = m_u[k]
                prow[f"{k}_saved"] = saved[k]
                max_diff = max(max_diff, abs(m_u[k] - saved[k]))
            prow["max_abs_diff"] = max_diff
            prow["parity_ok"] = bool(max_diff < 1e-9)
            parity_rows.append(prow)
            assert prow["parity_ok"], f"parity FAILED for {cfg} @ recall {target}: max diff {max_diff}"

            for sig in SIGNALS:
                cmap = conf_maps[sig]

                # ---- gate selection on VALIDATION (config thresholds fixed)
                yv = s1_val["label"].to_numpy()
                ungated_val = staged.staged_predict(s1_val, s2_val, t1, t2)
                f1_val_ungated = f1_score(yv, ungated_val["pred"].to_numpy(), zero_division=0)
                best_q, best_f1 = None, -1.0
                landscape = []
                for q in GATE_QUANTILES:
                    g = grids[s2_h][sig][q]
                    jv = gated_staged_predict(s1_val, s2_val, cmap, t1, t2, g)
                    f1v = f1_score(yv, jv["pred"].to_numpy(), zero_division=0)
                    landscape.append((q, g, f1v))
                    if f1v > best_f1:  # strict >: ties keep the SMALLEST quantile (weakest gate)
                        best_f1, best_q = f1v, q
                for q, g, f1v in landscape:
                    val_rows.append({"config": cfg, "recall_target": target, "signal": sig,
                                     "quantile": q, "gate_value": g, "val_f1_gated": f1v,
                                     "val_f1_ungated": f1_val_ungated,
                                     "chosen": bool(q == best_q)})
                g_star = grids[s2_h][sig][best_q]

                # ---- test evaluation at the chosen gate (once)
                gated = gated_staged_predict(s1_test, s2_test, cmap, t1, t2, g_star)
                pred_g = gated["pred"].to_numpy()
                m_g = staged.track_metrics(y, pred_g, grp)
                n_conf_nan_with_p2 = int((gated["p2"].notna() & gated["conf"].isna()).sum())
                row = {"config": cfg, "recall_target": target, "signal": sig,
                       "chosen_quantile": best_q, "gate_value": g_star, "t1": t1, "t2": t2,
                       "n_conf_missing_with_stage2": n_conf_nan_with_p2}
                for k in METRIC_KEYS:
                    row[f"{k}_gated"] = m_g[k]
                    row[f"{k}_ungated"] = m_u[k]
                    row[f"delta_{k}"] = m_g[k] - m_u[k]
                row.update({k: m_g[k] for k in ["n_tracks", "n_amb", "n_clean", "n_pos"]})

                # ---- decision-change accounting (gate can only flip veto -> positive)
                changed = gated[pred_g != pred_u]
                assert (changed["pred"] == 1).all(), "gate produced a 0->1-only rule violation"
                n_rec_tp = int(((changed["label"] == 1)).sum())
                n_fp_amb = int((changed["group"] == "ambiguous_negative").sum())
                n_fp_cln = int((changed["group"] == "clean_negative").sum())
                row.update({"n_changed": len(changed), "n_recovered_tp": n_rec_tp,
                            "n_new_fp_ambiguous": n_fp_amb, "n_new_fp_clean": n_fp_cln,
                            "n_unchanged": int(len(gated) - len(changed))})
                result_rows.append(row)
                change_rows.append({"config": cfg, "recall_target": target, "signal": sig,
                                    "chosen_quantile": best_q, "gate_value": g_star,
                                    "n_changed": len(changed), "n_recovered_tp": n_rec_tp,
                                    "n_new_fp_ambiguous": n_fp_amb, "n_new_fp_clean": n_fp_cln,
                                    "n_unchanged": int(len(gated) - len(changed))})
                for tk, r in changed.iterrows():
                    change_track_rows.append({
                        "config": cfg, "recall_target": target, "signal": sig,
                        "gate_value": g_star, "track_key": tk, "label": int(r["label"]),
                        "group": r["group"], "p1": r["p1"], "p2": r["p2"], "conf": r["conf"],
                        "change": "recovered_tp" if r["label"] == 1 else "new_fp"})

                # ---- track bootstrap gated vs ungated (10k, seed 12345)
                boot = staged.bootstrap_delta(y, grp, pred_u, pred_g)
                boot_rows.append({"config": cfg, "recall_target": target, "signal": sig,
                                  "chosen_quantile": best_q, "gate_value": g_star,
                                  "comparison": "gated_minus_ungated", **boot})

                # ---- DIAGNOSTIC: full test sweep over the gate grid (labeled diagnostic)
                sweep_rows.append({"config": cfg, "recall_target": target, "signal": sig,
                                   "quantile": 0.0, "gate_value": float("-inf"),
                                   "note": "ungated_reference",
                                   **{k: m_u[k] for k in METRIC_KEYS},
                                   "n_changed_vs_ungated": 0})
                for q in GATE_QUANTILES:
                    g = grids[s2_h][sig][q]
                    js = gated_staged_predict(s1_test, s2_test, cmap, t1, t2, g)
                    ms = staged.track_metrics(y, js["pred"].to_numpy(), grp)
                    sweep_rows.append({"config": cfg, "recall_target": target, "signal": sig,
                                       "quantile": q, "gate_value": g,
                                       "note": "diagnostic_test_sweep",
                                       **{k: ms[k] for k in METRIC_KEYS},
                                       "n_changed_vs_ungated": int((js["pred"].to_numpy() != pred_u).sum())})

    pd.DataFrame(parity_rows).to_csv(METRICS_DIR / "parity_check_vs_011.csv", index=False)
    pd.DataFrame(val_rows).to_csv(METRICS_DIR / "val_gate_selection.csv", index=False)
    pd.DataFrame(result_rows).to_csv(METRICS_DIR / "gated_test_results.csv", index=False)
    pd.DataFrame(change_rows).to_csv(METRICS_DIR / "decision_change_table.csv", index=False)
    pd.DataFrame(change_track_rows).to_csv(METRICS_DIR / "decision_change_tracks.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(METRICS_DIR / "gated_bootstrap.csv", index=False)
    pd.DataFrame(sweep_rows).to_csv(METRICS_DIR / "test_gate_sweep_diagnostic.csv", index=False)

    make_figure(pd.DataFrame(sweep_rows))
    print(json.dumps({
        "parity_all_ok": bool(all(r["parity_ok"] for r in parity_rows)),
        "n_result_rows": len(result_rows),
        "n_sweep_rows": len(sweep_rows),
        "n_changed_total_at_chosen_gates": int(sum(r["n_changed"] for r in change_rows)),
    }, indent=2))


def make_figure(sweep: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = sweep["config"].unique()
    fig, axes = plt.subplots(len(configs), len(RECALL_TARGETS),
                             figsize=(11, 3.0 * len(configs)), sharex=True)
    colors = {"skeleton_conf_mean": "#1f77b4", "lower_body_kp_conf_mean": "#d62728"}
    for i, cfg in enumerate(configs):
        for j, target in enumerate(RECALL_TARGETS):
            ax = axes[i][j] if len(configs) > 1 else axes[j]
            for sig in SIGNALS:
                sub = sweep[(sweep["config"] == cfg) & (sweep["recall_target"] == target)
                            & (sweep["signal"] == sig)].sort_values("quantile")
                ax.plot(sub["quantile"], sub["f1"], "o-", color=colors[sig],
                        label=f"F1 ({sig})", lw=1.5, ms=4)
                ax.plot(sub["quantile"], sub["amb_neg_fpr"], "s--", color=colors[sig],
                        label=f"amb-neg FPR ({sig})", lw=1.2, ms=4, alpha=0.7)
            ax.set_title(f"{cfg.replace('staged|bbox30_60->', '')} | recall target {target}", fontsize=9)
            ax.set_xticks([0.0] + GATE_QUANTILES)
            ax.set_xticklabels(["ungated", "0.10", "0.25", "0.50", "0.75"], fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_ylim(0, 1)
            if i == len(configs) - 1:
                ax.set_xlabel("gate quantile (train+val)")
            if j == 0:
                ax.set_ylabel("F1 / amb-neg FPR")
    handles, labels = (axes[0][0] if len(configs) > 1 else axes[0]).get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle("Exp 020 DIAGNOSTIC: test F1 and ambiguous-negative FPR vs veto-gate quantile\n"
                 "(gate disables the Stage-2 veto on windows below the confidence quantile)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(FIGURES_DIR / "gate_sweep_f1_ambfpr.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
