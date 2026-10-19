import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split


# Load your saved sequences
X_seq = np.load("X_seq.npy", allow_pickle=True)
y_seq = np.load("y_seq.npy", allow_pickle=True)


# Flatten: (N, 10, D) → (N, 10 × D)
X_flat = X_seq.reshape((X_seq.shape[0], -1))

# Split
#  Load the pid list from the sequence source DataFrame
# This assumes you saved the matching pids when generating X_seq/y_seq
pid_list = np.load("p_seq.npy")  # Shape (N,)

# Split on unique PIDs
unique_pids = np.unique(pid_list)
train_pids, test_pids = train_test_split(unique_pids, test_size=0.2, random_state=42)

# Mask sequences
train_mask = np.isin(pid_list, train_pids)
test_mask = np.isin(pid_list, test_pids)

# Apply masks
X_train = X_seq[train_mask].reshape(train_mask.sum(), -1)
y_train = y_seq[train_mask]
X_test = X_seq[test_mask].reshape(test_mask.sum(), -1)
y_test = y_seq[test_mask]


# Train Random Forest
clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train, y_train)

# Predict & evaluate
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))
print("F1 Score:", f1_score(y_test, y_pred))


