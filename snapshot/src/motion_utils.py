# motion_utils.py

import numpy as np
import pandas as pd

def add_temporal_features(
    df: pd.DataFrame,
    fps: float = 12.0,
    group_cols=("pid", "video"),
    sort_cols=("pid", "video", "frame_idx"),
):
    """
    Adds per-trajectory deltas/velocities/accelerations, rolling stats, and angle unwraps.
    Assumes df already contains per-frame BIUs + meta (frame_idx, pid, video, label).
    All group-ops use .transform(...) to keep indexes aligned and avoid insert errors.
    """
    req_meta = {"frame_idx", "pid", "video", "label"}
    missing = req_meta - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required meta columns: {missing}")

    df = df.sort_values(list(sort_cols)).reset_index(drop=True)
    g  = df.groupby(list(group_cols), group_keys=False)

    # which columns to derive from
    ang_cols = [c for c in ["torso_lean_angle","head_yaw_angle","body_yaw_angle","face_yaw_angle","gaze_to_car_angle"] if c in df.columns]
    lin_cols = [c for c in ["stride_length","lead_foot_elevation","arm_swing_asymmetry"] if c in df.columns]

    # 1) unwrap angles per trajectory (use transform to preserve index)
    for c in ang_cols:
        df[c + "_uw"] = g[c].transform(lambda s: np.unwrap(s.to_numpy()))

    # 2) linear deltas / velocities / accelerations (all via transform)
    for c in lin_cols:
        d1  = g[c].transform(lambda s: s.diff())
        d3  = g[c].transform(lambda s: s.diff(3))
        vel = d1 * fps
        # acc = g[vel.name].transform(lambda s: s.diff()) * fps  # diff of vel per group
        acc = g[c].transform(lambda s: s.diff().diff()) * (fps**2)  # ✅ true acceleration
        df[c + "_d1"]  = d1
        # df[c + "_d3"]  = d3
        df[c + "_d3"]  = g[c].transform(lambda s: s.diff(3))
        df[c + "_vel"] = vel
        df[c + "_acc"] = acc

    # 3) angular rates / accelerations from unwrapped angles
    for c in ang_cols:
        vel = g[c + "_uw"].transform(lambda s: s.diff()) * fps
        # acc = g[vel.name].transform(lambda s: s.diff()) * fps
        acc = g[c + "_uw"].transform(lambda s: s.diff().diff()) * (fps**2)
        df[c + "_vel"] = vel
        df[c + "_acc"] = acc

    # 4) short rolling stats (~0.5s) for gait cues; transform keeps index aligned
    win = max(1, int(round(fps * 0.5)))
    for c in ["stride_length","lead_foot_elevation"]:
        if c in df.columns:
            df[c + "_sma"] = g[c].transform(lambda s: s.rolling(win, min_periods=1).mean())
            df[c + "_std"] = g[c].transform(lambda s: s.rolling(win, min_periods=1).std()).fillna(0.0)

    # 5) simple interactions / embeddings
    if {"stride_length","lead_foot_elevation"} <= set(df.columns):
        df["gait_intensity"] = df["stride_length"] * df["lead_foot_elevation"]
    if {"head_yaw_angle","body_yaw_angle"} <= set(df.columns):
        df["yaw_alignment_cos"]  = np.cos(df["head_yaw_angle"] - df["body_yaw_angle"])
    if "gaze_to_car_angle" in df.columns:
        df["gaze_alignment_cos"] = np.cos(df["gaze_to_car_angle"])

    # 6) clip obvious outliers
    if "arm_swing_asymmetry" in df.columns:
        df["arm_swing_asymmetry"] = df["arm_swing_asymmetry"].clip(lower=0.0, upper=2.0)

    # 7) fill NaNs only for the newly created temporal columns
    new_cols = [c for c in df.columns if c.endswith(("_d1","_d3","_vel","_acc","_sma","_std"))] + \
               [c + "_uw" for c in ang_cols] + \
               ["gait_intensity","yaw_alignment_cos","gaze_alignment_cos"]
    for c in new_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    return df

def verify_temporal_hygiene(df: pd.DataFrame, group_cols=("pid","video")):
    df = df.sort_values(list(group_cols)+["frame_idx"]).reset_index(drop=True)
    key = df[group_cols[0]].astype(str) + "|" + df[group_cols[1]].astype(str)

    # frame order
    assert (df.groupby(key)["frame_idx"].diff().fillna(0) >= 0).all(), "frame_idx not sorted within tracks"

    # first row of each track should have ~0 diffs/vel/acc
    deriv_cols = [c for c in df.columns if c.endswith(("_d1","_d3","_vel","_acc"))]
    if deriv_cols:
        first_idx = df.groupby(key).head(1).index
        leakage = (df.loc[first_idx, deriv_cols].abs() > 1e-9).any(axis=None)
        print("cross-track leakage:", bool(leakage))

    # unwrap sanity
    for c in ["torso_lean_angle","head_yaw_angle","body_yaw_angle","face_yaw_angle","gaze_to_car_angle"]:
        if c in df.columns and (c + "_uw") in df.columns:
            raw_jumps = (df.groupby(key)[c].diff().abs() > np.pi).sum()
            uw_jumps  = (df.groupby(key)[c + "_uw"].diff().abs() > np.pi).sum()
            print(f"{c}: raw jumps>{np.pi} = {int(raw_jumps)}, unwrapped = {int(uw_jumps)}")
