import pickle, numpy as np
from sklearn.model_selection import GroupShuffleSplit

# ------------------------------------------------------------------
# 1.  Load the tensors produced by seq_builder.py
# ------------------------------------------------------------------
X   = np.load("X_seq.npy")   # shape: [N, seq_len, feat_dim]
y   = np.load("y_seq.npy")   # shape: [N]
pID = np.load("p_seq.npy")   # shape: [N]  (per-sequence pedestrian ID)

# ------------------------------------------------------------------
# 2.  First split: hold-out 20 % for the test set  (no leakage)
# ------------------------------------------------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx0, test_idx = next(gss.split(X, y, groups=pID))

# ------------------------------------------------------------------
# 3.  Second split: carve 10 % of the *training* set for validation
#     NOTE: indices returned by gss_val are RELATIVE to train_idx0,
#           so we must map them back to the absolute indices.
# ------------------------------------------------------------------
gss_val = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=1)
sub_train_idx, val_idx_rel = next(
        gss_val.split(X[train_idx0], y[train_idx0], groups=pID[train_idx0]))

train_idx = train_idx0[sub_train_idx]   # absolute indices
val_idx   = train_idx0[val_idx_rel]     # absolute indices

# ------------------------------------------------------------------
# 4.  Save the split
# ------------------------------------------------------------------
split = {"train": train_idx, "val": val_idx, "test": test_idx}
pickle.dump(split, open("split_pid.pkl", "wb"))

# ------------------------------------------------------------------
# 5.  Quick sanity print
# ------------------------------------------------------------------
print({k: len(v) for k, v in split.items()})


    