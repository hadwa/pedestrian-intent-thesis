#!/usr/bin/env python3
"""
Experiment 017: PSI keypoint fidelity study (measurement-agreement, NOT a benchmark).

Quantifies how faithful ViTPose-S keypoints -- and the BIU v1 cues computed from
them -- are to human-annotated ground-truth keypoints on the PSI dataset
(29 tracks, PSI2.0_TrainVal).

Inputs (both JSONL, one record per (set, video, pid, frame_idx), COCO-17 joints):
  data/psi_filtered_tracks_export_29_tracks.jsonl  -- ground-truth annotations
      (unannotated joints are [None, None]; kp_conf all 1.0)
  data/psi_vitpose_s_all_tracks.jsonl              -- ViTPose-S estimates
      (keypoints 17x2, kp_conf 17, skeleton_conf; streamed, ~88 MB)

Outputs: experiments/017_psi_keypoint_fidelity_study/{metrics/*.csv, figures/*.png}

Run:
  MPLCONFIGDIR=/tmp/mplconfig python3 scripts/experiments/psi_keypoint_fidelity_study.py

BIU cues are computed with src/biu_extractor.py::extract_biu_features (reused
unchanged). The confidence gate 0.2 mirrors LOW_CONFIDENCE_THRESHOLD in the
archive pipeline (/home/abdelrahman/Hadwa/BIU-PID-Distillation/src/features/
biu_features.py, see experiments/014_confidence_aware_biu_audit/report.md).
"""

import json
import os
import sys
import warnings

# frames with zero GT-annotated joints legitimately yield empty-slice means (NaN)
warnings.filterwarnings("ignore", message="Mean of empty slice")
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean"
GT_PATH = os.path.join(REPO, "data", "psi_filtered_tracks_export_29_tracks.jsonl")
VP_PATH = os.path.join(REPO, "data", "psi_vitpose_s_all_tracks.jsonl")
OUT = os.path.join(REPO, "experiments", "017_psi_keypoint_fidelity_study")
METRICS = os.path.join(OUT, "metrics")
FIGURES = os.path.join(OUT, "figures")
os.makedirs(METRICS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

sys.path.insert(0, os.path.join(REPO, "src"))
from biu_extractor import extract_biu_features  # noqa: E402

# COCO-17 joint order (assumed for both files; verified empirically below).
JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
JOINT_GROUPS = {"head": list(range(0, 5)), "arms": list(range(5, 11)),
                "lower_body": list(range(11, 17))}
LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

# Archive BIU v1 pipeline low-confidence gate (LOW_CONFIDENCE_THRESHOLD).
LOW_CONF_GATE = 0.2
MIN_SCALE_JOINTS = 4   # min annotated GT joints for a bbox-extent scale
MIN_SCALE_PX = 5.0     # discard degenerate scales below this many pixels

# Joints each BIU v1 cue needs, incl. those used only for normalization.
CUE_REQUIRED = {
    "torso_lean_angle":    ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    "stride_length":       ["left_ankle", "right_ankle", "left_shoulder", "right_shoulder"],
    "head_yaw_angle":      ["nose", "left_shoulder", "right_shoulder"],
    "lead_foot_elevation": ["left_ankle", "right_ankle", "left_hip", "right_hip",
                            "left_shoulder", "right_shoulder"],
    "lead_foot_lifted":    ["left_ankle", "right_ankle", "left_hip", "right_hip",
                            "left_shoulder", "right_shoulder"],
    "arm_swing_asymmetry": ["left_wrist", "right_wrist", "left_shoulder", "right_shoulder"],
    "gaze_to_car_angle":   ["nose", "left_ear", "right_ear"],
    "body_yaw_angle":      ["left_shoulder", "right_shoulder"],
    "face_yaw_angle":      ["nose", "left_hip", "right_hip"],
    "norm_shoulder_width": ["left_shoulder", "right_shoulder", "nose", "left_ankle"],
}
CUES = list(CUE_REQUIRED.keys())
BINARY_CUES = {"lead_foot_lifted"}
NAME2IDX = {n: i for i, n in enumerate(JOINT_NAMES)}


def load_data():
    gt = {}
    with open(GT_PATH) as f:
        for line in f:
            r = json.loads(line)
            gt[(r["set"], r["video"], r["pid"], r["frame_idx"])] = r
    n_vp_total = 0
    matched = []
    with open(VP_PATH) as f:  # stream the 88 MB file
        for line in f:
            r = json.loads(line)
            n_vp_total += 1
            key = (r["set"], r["video"], r["pid"], r["frame_idx"])
            g = gt.get(key)
            if g is not None:
                matched.append((key, g, r))
    matched.sort(key=lambda t: t[0])
    return gt, matched, n_vp_total


def to_arrays(matched):
    n = len(matched)
    gt_kp = np.full((n, 17, 2), np.nan)
    est_kp = np.full((n, 17, 2), np.nan)
    est_conf = np.full((n, 17), np.nan)
    skel_conf = np.full(n, np.nan)
    labels = np.full(n, -1, dtype=int)
    track_ids, frame_idx = [], np.zeros(n, dtype=int)
    for i, (key, g, r) in enumerate(matched):
        for j in range(17):
            gj = g["keypoints"][j]
            if gj[0] is not None:
                gt_kp[i, j] = gj
            est_kp[i, j] = r["keypoints"][j]
        est_conf[i] = r["kp_conf"]
        skel_conf[i] = r["skeleton_conf"]
        labels[i] = int(g.get("label", -1))
        track_ids.append(key[:3])
        frame_idx[i] = key[3]
    return gt_kp, est_kp, est_conf, skel_conf, labels, track_ids, frame_idx


def person_scale(gt_kp):
    """Per-frame GT person scale: bbox extent = max(x-range, y-range) over
    annotated GT joints; NaN if < MIN_SCALE_JOINTS joints or < MIN_SCALE_PX px.
    Also returns shoulder-hip scale (mid-shoulder to mid-hip distance) where
    all four joints are annotated, for a consistency check."""
    n = gt_kp.shape[0]
    scale = np.full(n, np.nan)
    sh_scale = np.full(n, np.nan)
    for i in range(n):
        pts = gt_kp[i][~np.isnan(gt_kp[i, :, 0])]
        if len(pts) >= MIN_SCALE_JOINTS:
            ext = max(pts[:, 0].max() - pts[:, 0].min(),
                      pts[:, 1].max() - pts[:, 1].min())
            if ext >= MIN_SCALE_PX:
                scale[i] = ext
        ls, rs = gt_kp[i, 5], gt_kp[i, 6]
        lh, rh = gt_kp[i, 11], gt_kp[i, 12]
        if not (np.isnan(ls[0]) or np.isnan(rs[0]) or np.isnan(lh[0]) or np.isnan(rh[0])):
            sh_scale[i] = np.linalg.norm((ls + rs) / 2 - (lh + rh) / 2)
    return scale, sh_scale


def safe_corr(a, b, kind):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3 or np.nanstd(a[m]) == 0 or np.nanstd(b[m]) == 0:
        return np.nan, int(m.sum())
    f = pearsonr if kind == "pearson" else spearmanr
    return float(f(a[m], b[m])[0]), int(m.sum())


def compute_cues(kp):
    """Per-frame BIU v1 cues via src/biu_extractor.py. Rows with NaN inputs can
    produce garbage (the extractor's _unit() maps NaN vectors to zeros), so
    validity is enforced downstream via CUE_REQUIRED masks -- never via NaN."""
    n = kp.shape[0]
    out = {c: np.full(n, np.nan) for c in CUES}
    for i in range(n):
        jd = {name: np.array(kp[i, j], float) for name, j in NAME2IDX.items()}
        try:
            feats = extract_biu_features(jd)
        except Exception:
            continue
        for c in CUES:
            v = feats.get(c, np.nan)
            out[c][i] = float(v) if v is not None else np.nan
    return out


def main():
    rng_report = {}

    # ---------------- 1. matching + identity check ----------------
    gt, matched, n_vp_total = load_data()
    gt_kp, est_kp, est_conf, skel_conf, labels, track_ids, frame_idx = to_arrays(matched)
    n = len(matched)
    tracks = sorted(set(track_ids))
    track_arr = np.array(["/".join(t) for t in track_ids])

    gt_annotated = ~np.isnan(gt_kp[:, :, 0])          # (n, 17)
    n_gt_joints = int(gt_annotated.sum())
    diffs = est_kp - gt_kp                            # NaN where GT missing
    identical = (np.abs(diffs[:, :, 0]) == 0) & (np.abs(diffs[:, :, 1]) == 0) & gt_annotated
    dx = diffs[:, :, 0][gt_annotated]
    dy = diffs[:, :, 1][gt_annotated]

    per_track = (pd.DataFrame({"track": track_arr, "label": labels})
                 .groupby("track").agg(n_frames=("track", "size"),
                                       label=("label", "first")).reset_index())
    per_track.to_csv(os.path.join(METRICS, "per_track_frames.csv"), index=False)

    match_stats = pd.DataFrame([{
        "n_gt_records": len(gt),
        "n_vitpose_records_total": n_vp_total,
        "n_matched_frames": n,
        "n_tracks": len(tracks),
        "n_tracks_crossing": int((per_track["label"] == 1).sum()),
        "n_tracks_noncrossing": int((per_track["label"] == 0).sum()),
        "frames_per_track_min": int(per_track["n_frames"].min()),
        "frames_per_track_median": float(per_track["n_frames"].median()),
        "frames_per_track_max": int(per_track["n_frames"].max()),
        "frac_gt_joints_annotated": n_gt_joints / (n * 17),
        "n_gt_annotated_joints": n_gt_joints,
        "n_exactly_identical_joints": int(identical.sum()),
        "frac_exactly_identical_joints": float(identical.sum()) / n_gt_joints,
        "coord_diff_dx_median": float(np.median(dx)),
        "coord_diff_dy_median": float(np.median(dy)),
        "coord_diff_abs_dx_p50": float(np.median(np.abs(dx))),
        "coord_diff_abs_dy_p50": float(np.median(np.abs(dy))),
        "coord_diff_abs_dx_p90": float(np.percentile(np.abs(dx), 90)),
        "coord_diff_abs_dy_p90": float(np.percentile(np.abs(dy), 90)),
    }])
    match_stats.to_csv(os.path.join(METRICS, "match_stats.csv"), index=False)
    rng_report["match"] = match_stats.iloc[0].to_dict()

    # ---------------- 2. per-joint errors (pixel + normalized) ----------------
    scale, sh_scale = person_scale(gt_kp)
    both = ~(np.isnan(scale) | np.isnan(sh_scale))
    scale_check = pd.DataFrame([{
        "n_frames_with_bbox_scale": int((~np.isnan(scale)).sum()),
        "n_frames_with_shoulder_hip_scale": int((~np.isnan(sh_scale)).sum()),
        "bbox_scale_median_px": float(np.nanmedian(scale)),
        "shoulder_hip_scale_median_px": float(np.nanmedian(sh_scale)),
        "pearson_bbox_vs_shoulder_hip": safe_corr(scale[both], sh_scale[both], "pearson")[0],
        "median_ratio_shoulder_hip_over_bbox": float(np.nanmedian(sh_scale[both] / scale[both])),
    }])
    scale_check.to_csv(os.path.join(METRICS, "scale_definition_check.csv"), index=False)

    err_px = np.sqrt((diffs ** 2).sum(axis=2))        # (n, 17), NaN where GT missing
    err_norm = err_px / scale[:, None]                # NaN where no scale

    rows = []
    for j, name in enumerate(JOINT_NAMES):
        e = err_norm[:, j]
        ep = err_px[:, j]
        m = ~np.isnan(e)
        grp = next(g for g, idxs in JOINT_GROUPS.items() if j in idxs)
        rows.append({
            "joint_idx": j, "joint": name, "group": grp,
            "n_frames": int(m.sum()),
            "px_err_median": float(np.nanmedian(ep)),
            "px_err_mean": float(np.nanmean(ep)),
            "norm_err_median": float(np.nanmedian(e)),
            "norm_err_mean": float(np.nanmean(e)),
            "pck_0.1": float(np.mean(e[m] < 0.1)) if m.any() else np.nan,
            "pck_0.2": float(np.mean(e[m] < 0.2)) if m.any() else np.nan,
        })
    per_joint = pd.DataFrame(rows)
    per_joint.to_csv(os.path.join(METRICS, "per_joint_errors.csv"), index=False)

    grp_rows = []
    for g, idxs in JOINT_GROUPS.items():
        e = err_norm[:, idxs].ravel()
        e = e[~np.isnan(e)]
        grp_rows.append({"group": g, "n_joint_obs": len(e),
                         "norm_err_median": float(np.median(e)),
                         "norm_err_mean": float(np.mean(e)),
                         "pck_0.1": float(np.mean(e < 0.1)),
                         "pck_0.2": float(np.mean(e < 0.2))})
    per_group = pd.DataFrame(grp_rows)
    per_group.to_csv(os.path.join(METRICS, "per_group_errors.csv"), index=False)
    rng_report["groups"] = per_group

    # Joint-order sanity: L/R swap test.
    swap_rows = []
    for jl, jr in LR_PAIRS:
        m = gt_annotated[:, jl] & gt_annotated[:, jr] & ~np.isnan(scale)
        if m.sum() < 10:
            continue
        e_norm_order = (np.linalg.norm(est_kp[m, jl] - gt_kp[m, jl], axis=1)
                        + np.linalg.norm(est_kp[m, jr] - gt_kp[m, jr], axis=1)) / 2 / scale[m]
        e_swapped = (np.linalg.norm(est_kp[m, jr] - gt_kp[m, jl], axis=1)
                     + np.linalg.norm(est_kp[m, jl] - gt_kp[m, jr], axis=1)) / 2 / scale[m]
        swap_rows.append({
            "pair": f"{JOINT_NAMES[jl]}/{JOINT_NAMES[jr]}", "n_frames": int(m.sum()),
            "norm_err_mean_normal_order": float(e_norm_order.mean()),
            "norm_err_mean_lr_swapped": float(e_swapped.mean()),
            "swap_better": bool(e_swapped.mean() < e_norm_order.mean()),
        })
    swap_df = pd.DataFrame(swap_rows)
    swap_df.to_csv(os.path.join(METRICS, "joint_order_swap_test.csv"), index=False)
    rng_report["swap"] = swap_df

    # ---------------- 3. confidence vs error ----------------
    conf_rows = []
    for j, name in enumerate(JOINT_NAMES):
        r, nn = safe_corr(est_conf[:, j], err_norm[:, j], "spearman")
        conf_rows.append({"level": "joint", "name": name, "n": nn,
                          "spearman_conf_vs_norm_err": r})
    pooled_c = est_conf[gt_annotated & ~np.isnan(err_norm)]
    pooled_e = err_norm[gt_annotated & ~np.isnan(err_norm)]
    r, nn = safe_corr(pooled_c, pooled_e, "spearman")
    conf_rows.append({"level": "pooled_all_joints", "name": "all", "n": nn,
                      "spearman_conf_vs_norm_err": r})
    frame_err = np.nanmean(err_norm, axis=1)
    r, nn = safe_corr(skel_conf, frame_err, "spearman")
    conf_rows.append({"level": "frame", "name": "skeleton_conf_vs_mean_frame_err",
                      "n": nn, "spearman_conf_vs_norm_err": r})
    lb = JOINT_GROUPS["lower_body"]
    lb_conf = est_conf[:, lb].mean(axis=1)
    with np.errstate(all="ignore"):
        lb_err = np.nanmean(err_norm[:, lb], axis=1)
    r, nn = safe_corr(lb_conf, lb_err, "spearman")
    conf_rows.append({"level": "frame", "name": "lower_body_kp_conf_vs_lower_body_err",
                      "n": nn, "spearman_conf_vs_norm_err": r})
    conf_df = pd.DataFrame(conf_rows)
    conf_df.to_csv(os.path.join(METRICS, "confidence_vs_error.csv"), index=False)
    rng_report["conf"] = conf_df

    # ---------------- 4. BIU cue fidelity ----------------
    cues_gt = compute_cues(gt_kp)
    cues_est = compute_cues(est_kp)

    cue_rows = []
    for c in CUES:
        req = [NAME2IDX[nm] for nm in CUE_REQUIRED[c]]
        valid = gt_annotated[:, req].all(axis=1)
        valid &= ~np.isnan(cues_gt[c]) & ~np.isnan(cues_est[c])
        gated = valid & (est_conf[:, req].min(axis=1) >= LOW_CONF_GATE)
        row = {"cue": c, "binary": c in BINARY_CUES,
               "n_frames": int(valid.sum()), "n_frames_gated": int(gated.sum())}
        for tag, m in [("", valid), ("_gated", gated)]:
            a, b = cues_gt[c][m], cues_est[c][m]
            row[f"pearson{tag}"] = safe_corr(a, b, "pearson")[0]
            row[f"spearman{tag}"] = safe_corr(a, b, "spearman")[0]
            row[f"mae{tag}"] = float(np.mean(np.abs(b - a))) if m.any() else np.nan
            row[f"bias_est_minus_gt{tag}"] = float(np.mean(b - a)) if m.any() else np.nan
            if c in BINARY_CUES and m.any():
                row[f"agreement{tag}"] = float(np.mean(a == b))
        row["gt_std"] = float(np.std(cues_gt[c][valid])) if valid.any() else np.nan
        cue_rows.append(row)
    cue_df = pd.DataFrame(cue_rows).sort_values("spearman", ascending=False)
    cue_df.to_csv(os.path.join(METRICS, "per_cue_fidelity.csv"), index=False)
    rng_report["cues"] = cue_df

    # ---------------- 4b. motion deltas (BIU-v2-style) vs absolute ----------------
    # Quantities from both sources, normalized by the same GT scale:
    #   hip_center_x, ankle_center_x (positions; deltas = x-velocities over
    #   consecutive matched frames within a track), ankle_gap_y (lead-foot
    #   elevation proxy), plus delta versions of two v1 cues (torso lean, stride).
    def q_hip_cx(kp):
        return (kp[:, 11, 0] + kp[:, 12, 0]) / 2
    def q_ankle_cx(kp):
        return (kp[:, 15, 0] + kp[:, 16, 0]) / 2
    def q_ankle_gap_y(kp):
        return np.abs(kp[:, 15, 1] - kp[:, 16, 1])

    motion_specs = {
        "hip_center_x":  (q_hip_cx,  [11, 12], True),
        "ankle_center_x": (q_ankle_cx, [15, 16], True),
        "ankle_gap_y":   (q_ankle_gap_y, [15, 16], True),
        "torso_lean_angle": (None, [NAME2IDX[nm] for nm in CUE_REQUIRED["torso_lean_angle"]], False),
        "stride_length": (None, [NAME2IDX[nm] for nm in CUE_REQUIRED["stride_length"]], False),
    }
    # consecutive matched frame pairs within a track
    order = np.lexsort((frame_idx, track_arr))
    consec = []
    for a, b in zip(order[:-1], order[1:]):
        if track_arr[a] == track_arr[b] and frame_idx[b] - frame_idx[a] == 1:
            consec.append((a, b))
    consec = np.array(consec)
    ia, ib = consec[:, 0], consec[:, 1]

    mot_rows = []
    for qname, (fn, req, norm_by_scale) in motion_specs.items():
        if fn is not None:
            g_abs, e_abs = fn(gt_kp), fn(est_kp)
            if norm_by_scale:
                g_abs, e_abs = g_abs / scale, e_abs / scale
        else:  # reuse the v1 cue values (already scale-free)
            g_abs, e_abs = cues_gt[qname], cues_est[qname]
        valid = gt_annotated[:, req].all(axis=1) & ~np.isnan(g_abs) & ~np.isnan(e_abs)
        if norm_by_scale:
            valid &= ~np.isnan(scale)
        # absolute agreement
        m = valid
        p_abs = safe_corr(g_abs[m], e_abs[m], "pearson")[0]
        s_abs = safe_corr(g_abs[m], e_abs[m], "spearman")[0]
        # delta agreement over consecutive valid pairs
        pm = valid[ia] & valid[ib]
        g_d = g_abs[ib][pm] - g_abs[ia][pm]
        e_d = e_abs[ib][pm] - e_abs[ia][pm]
        p_d = safe_corr(g_d, e_d, "pearson")[0]
        s_d = safe_corr(g_d, e_d, "spearman")[0]
        mot_rows.append({
            "quantity": qname, "kind": "v2_motion" if fn is not None else "v1_cue",
            "n_frames_abs": int(m.sum()), "n_pairs_delta": int(pm.sum()),
            "pearson_absolute": p_abs, "spearman_absolute": s_abs,
            "pearson_delta": p_d, "spearman_delta": s_d,
            "delta_more_robust_pearson": bool(p_d > p_abs) if not (np.isnan(p_d) or np.isnan(p_abs)) else None,
            "mae_delta": float(np.mean(np.abs(e_d - g_d))) if pm.any() else np.nan,
            "gt_delta_std": float(np.std(g_d)) if pm.any() else np.nan,
        })
    mot_df = pd.DataFrame(mot_rows)
    mot_df.to_csv(os.path.join(METRICS, "motion_delta_fidelity.csv"), index=False)
    rng_report["motion"] = mot_df

    # ---------------- 5. secondary: group-difference sign agreement ----------------
    # 25 crossing vs 4 non-crossing tracks -- effect DIRECTIONS only.
    sign_rows = []
    tr_lab = dict(zip(per_track["track"], per_track["label"]))
    for c in CUES:
        req = [NAME2IDX[nm] for nm in CUE_REQUIRED[c]]
        valid = gt_annotated[:, req].all(axis=1) & ~np.isnan(cues_gt[c]) & ~np.isnan(cues_est[c])
        df = pd.DataFrame({"track": track_arr[valid], "g": cues_gt[c][valid],
                           "e": cues_est[c][valid]})
        tm = df.groupby("track").mean(numeric_only=True)
        tm["label"] = [tr_lab[t] for t in tm.index]
        if tm["label"].nunique() < 2:
            continue
        d_gt = tm.loc[tm.label == 1, "g"].mean() - tm.loc[tm.label == 0, "g"].mean()
        d_est = tm.loc[tm.label == 1, "e"].mean() - tm.loc[tm.label == 0, "e"].mean()
        sign_rows.append({
            "cue": c,
            "n_tracks_crossing": int((tm.label == 1).sum()),
            "n_tracks_noncrossing": int((tm.label == 0).sum()),
            "diff_crossing_minus_non_GT": float(d_gt),
            "diff_crossing_minus_non_ViTPose": float(d_est),
            "sign_agrees": bool(np.sign(d_gt) == np.sign(d_est)),
        })
    sign_df = pd.DataFrame(sign_rows)
    sign_df.to_csv(os.path.join(METRICS, "label_group_sign_agreement.csv"), index=False)
    rng_report["sign"] = sign_df

    # ---------------- figures (grayscale-safe) ----------------
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#888",
                         "axes.linewidth": 0.8, "figure.dpi": 150})
    grp_style = {"head": ("#cccccc", ""), "arms": ("#999999", "//"),
                 "lower_body": ("#4d4d4d", "")}

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(17)
    for g, idxs in JOINT_GROUPS.items():
        col, hatch = grp_style[g]
        vals = per_joint.set_index("joint_idx").loc[idxs, "norm_err_median"]
        ax.bar(x[idxs], vals, width=0.72, color=col, hatch=hatch,
               edgecolor="white", linewidth=0.6,
               label=g.replace("_", " "))
    ax.axhline(0.1, color="#555555", lw=0.8, ls="--")
    ax.text(16.4, 0.102, "PCK 0.1 threshold", ha="right", va="bottom",
            fontsize=7.5, color="#555555")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("left_", "L ").replace("right_", "R ")
                        for n in JOINT_NAMES], rotation=45, ha="right")
    ax.set_ylabel("median normalized error\n(fraction of GT person scale)")
    ax.set_title("ViTPose-S per-joint error vs PSI human keypoints (29 tracks)")
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "per_joint_norm_error.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    cdf = cue_df.sort_values("spearman", ascending=True)
    y = np.arange(len(cdf))
    ax.barh(y - 0.19, cdf["spearman"], height=0.36, color="#4d4d4d",
            edgecolor="white", linewidth=0.6, label="ungated")
    ax.barh(y + 0.19, cdf["spearman_gated"], height=0.36, color="#b3b3b3",
            hatch="//", edgecolor="white", linewidth=0.6,
            label="kp_conf >= 0.2 gate")
    ax.set_yticks(y)
    ax.set_yticklabels(cdf["cue"])
    ax.set_xlabel("Spearman correlation (GT-derived vs ViTPose-derived cue)")
    ax.set_title("BIU v1 cue fidelity under pose estimation (PSI)")
    ax.axvline(0, color="#888888", lw=0.8)
    ax.grid(axis="x", color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "per_cue_spearman.png"))
    plt.close(fig)

    # ---------------- console summary ----------------
    pd.set_option("display.width", 200)
    print("=== match stats ===")
    for k, v in rng_report["match"].items():
        print(f"  {k}: {v}")
    print("\n=== per-group normalized error ===")
    print(rng_report["groups"].to_string(index=False))
    print("\n=== L/R swap test ===")
    print(rng_report["swap"].to_string(index=False))
    print("\n=== confidence vs error (Spearman) ===")
    print(rng_report["conf"].to_string(index=False))
    print("\n=== per-cue fidelity ===")
    cols = ["cue", "n_frames", "n_frames_gated", "pearson", "spearman",
            "pearson_gated", "spearman_gated", "mae", "bias_est_minus_gt", "gt_std"]
    print(rng_report["cues"][cols].to_string(index=False))
    print("\n=== motion delta vs absolute ===")
    print(rng_report["motion"].to_string(index=False))
    print("\n=== label group sign agreement (25 vs 4 tracks; directions only) ===")
    print(rng_report["sign"].to_string(index=False))


if __name__ == "__main__":
    main()
