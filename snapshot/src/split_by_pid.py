import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Load dataset 
df = pd.read_csv("../data/hybrid_biu_keypoints_with_pid.csv")

# Ensure required columns exist
assert "pid" in df.columns and "label" in df.columns, "Missing required columns."

# Create GroupSplit to avoid PID leakage
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(df, groups=df["pid"]))

# Split data
df_train = df.iloc[train_idx].reset_index(drop=True)
df_test = df.iloc[test_idx].reset_index(drop=True)

# Save splits
df_train.to_csv("train_pid_grouped.csv", index=False)
df_test.to_csv("test_pid_grouped.csv", index=False)

print("✅ Saved:")
print("Train set:", df_train.shape)
print("Test set: ", df_test.shape)
print("Unique PIDs - Train:", df_train['pid'].nunique(), "| Test:", df_test['pid'].nunique())
