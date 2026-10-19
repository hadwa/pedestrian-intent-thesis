# # import pandas as pd
# # import numpy as np

# # #  this was hybrid_biu_keypoints
# # df_hybrid = pd.read_csv("../data/hybrid_biu_keypoints_v2.csv")
# # df_pid = pd.read_csv("frame_pid_map.csv")
# # df_pid["frame"] = df_pid["frame"].str.replace(r"^[^/]+/", "", regex=True)
# # df_merged = pd.merge(df_hybrid, df_pid, on="frame", how="inner")

# # # Sort by person & frame
# # df_merged = df_merged.sort_values(["pid", "frame"]).reset_index(drop=True)

# # # Build sequences
# # seq_len = 10
# # frame_spacing = 5            # Use every 5th frame to simulate time gaps
# # future_offset = 5            # Predict label 5 frames after last input frame 
# # X, y, p = [], [], []

# # for pid, group in df_merged.groupby("pid"):
# #     group = group.reset_index(drop=True)
# #     for i in range(len(group)):
# #         idx_seq = [i + j * frame_spacing for j in range(seq_len)]
# #         idx_label = i + seq_len * frame_spacing + future_offset - 1

# #     # Check bounds
# #         if idx_label >= len(group) or any(j >= len(group) for j in idx_seq):
# #             continue
# #         chunk = group.iloc[idx_seq]
# #         label_next = group.iloc[idx_label]["label"]
# #         features = chunk.drop(columns=["frame", "label", "pid"]).values
# #         X.append(features)
# #         y.append(label_next)
# #         p.append(pid)


# # X_seq = np.stack(X)
# # y_seq = np.array(y)
# # p_seq = np.array(p)

# # # Save
# # np.save("X_seq.npy", X_seq)
# # np.save("y_seq.npy", y_seq)
# # np.save("p_seq.npy", p_seq)

# # print(f"✅ Predictive labeling: Saved {len(X_seq)} sequences from {df_merged['pid'].nunique()} pedestrians.")
# # print("Shape:", X_seq.shape, y_seq.shape)
# # print("Label balance:", np.bincount(y_seq))

# # seq_builder.py  ── non-overlapping windows, numeric tensors
# import pandas as pd
# import numpy as np
# from pathlib import Path

# # ----------------------------------------------------------------------
# # 0.  configurable parameters  (feel free to change)
# # ----------------------------------------------------------------------
# SEQ_LEN        = 30            # history length  (frames in the raw video)
# FRAME_SPACING  = 1             # keep every k-th frame inside the window
# FUTURE_OFFSET  = 30            # predict intent 1 s after the last input frame
# CSV_IN         = "../data/hybrid_biu_keypoints_v2.csv"
# PID_MAP_IN     = "frame_pid_map.csv"
# CACHE_DIR      = Path("../src")            # everything saved here
# CACHE_DIR.mkdir(exist_ok=True)

# # ----------------------------------------------------------------------
# # 1.  load & merge
# # ----------------------------------------------------------------------
# df_hybrid = pd.read_csv(CSV_IN)
# df_pid    = pd.read_csv(PID_MAP_IN)
# df_pid["frame"] = df_pid["frame"].str.replace(r"^[^/]+/", "", regex=True)

# df = (df_hybrid
#       .merge(df_pid, on="frame", how="inner")
#       .sort_values(["pid", "frame"])
#       .reset_index(drop=True))

# # ----------------------------------------------------------------------
# # 2.  build sequences  (non-overlapping)
# # ----------------------------------------------------------------------
# X, y, p = [], [], []

# step = SEQ_LEN * FRAME_SPACING            # jump this many rows each window

# for pid, g in df.groupby("pid"):
#     g = g.reset_index(drop=True)
#     for start in range(0, len(g), step):          # NON-OVERLAPPING
#         idx_seq   = [start + j*FRAME_SPACING for j in range(SEQ_LEN)]
#         idx_label = start + step + FUTURE_OFFSET - 1   # horizon

#         # bounds check
#         if idx_label >= len(g) or idx_seq[-1] >= len(g):
#             continue

#         chunk   = g.iloc[idx_seq]
#         features = chunk.drop(columns=["frame", "label", "pid"]).values \
#                        .astype(np.float32)        # numeric!
#         X.append(features)
#         y.append(g.iloc[idx_label]["label"])
#         p.append(pid)

# X_seq = np.stack(X, dtype=np.float32)     # shape: [N, SEQ_LEN, feat_dim]
# y_seq = np.asarray(y, dtype=np.int8)
# p_seq = np.asarray(p, dtype=np.int32)

# # ----------------------------------------------------------------------
# # 3.  save
# # ----------------------------------------------------------------------
# np.save(CACHE_DIR / "X_seq.npy", X_seq)
# np.save(CACHE_DIR / "y_seq.npy", y_seq)
# np.save(CACHE_DIR / "p_seq.npy", p_seq)

# print(f"✅  Saved {len(X_seq)} non-overlapping sequences "
#       f"from {df['pid'].nunique()} pedestrians.")
# print("    X shape:", X_seq.shape, "dtype:", X_seq.dtype)
# print("    Label balance:", np.bincount(y_seq))
# seq_builder_min.py
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd

META_COLS = {"frame_idx", "pid", "video", "label"}

def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Enforce required columns
    missing = META_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Ensure types
    df["frame_idx"] = df["frame_idx"].astype(int)
    df["pid"] = df["pid"].astype(str)
    df["video"] = df["video"].astype(str)
    df["label"] = df["label"].astype(int)

    # Sort
    df = df.sort_values(["video", "pid", "frame_idx"]).reset_index(drop=True)
    return df

def detect_feature_cols(df: pd.DataFrame) -> List[str]:
    """Everything not in META_COLS is treated as a BIU feature."""
    feats = [c for c in df.columns if c not in META_COLS]
    if not feats:
        raise ValueError("No BIU feature columns found.")
    return feats

def build_sequences(
    df: pd.DataFrame,
    W: int = 10,
    H: int = 5,
    skip: int = 2,
    target_mode: str = "any"  # "any", "all", "majority", "next"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict]:
    feat_cols = detect_feature_cols(df)
    X_list, y_list, groups = [], [], []

    def agg_target(arr: np.ndarray) -> int:
        if target_mode == "any":      return int((arr == 1).any())
        if target_mode == "all":      return int((arr == 1).all())
        if target_mode == "majority": return int((arr == 1).sum() >= (len(arr)+1)//2)
        if target_mode == "next":     return int(arr[0] == 1)
        raise ValueError(f"Unknown target_mode={target_mode}")

    for (video, pid), g in df.groupby(["video", "pid"], sort=False):
        g = g.iloc[::(skip+1)].reset_index(drop=True)  # sample every (skip+1)th frame
        if len(g) < W + H:
            continue

        F = g[feat_cols].to_numpy(dtype=np.float32)  # [T, D]
        L = g["label"].to_numpy(dtype=int)           # [T]

        for t in range(W, len(g) - H + 1):
            X_list.append(F[t-W:t].reshape(-1))     # flatten past W frames
            y_list.append(agg_target(L[t:t+H]))     # label from next H frames
            groups.append(f"{video}/{pid}")

    if not X_list:
        raise RuntimeError("No sequences built — try smaller W/H or smaller skip.")

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)
    groups = np.array(groups)
    feat_names = [f"feat[t-{W-i}].{c}" for i in range(W) for c in feat_cols]

    meta = {
        "W": W, "H": H, "skip": skip,
        "target_mode": target_mode,
        "feat_cols": feat_cols,
        "D": len(feat_cols)
    }
    return X, y, groups, feat_names, meta
