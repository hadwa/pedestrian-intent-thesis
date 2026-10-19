# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
# import shap
# import matplotlib.pyplot as plt

# # Load the hybrid feature dataset
# df = pd.read_csv("../data/hybrid_biu_keypoints_v2.csv")

# df.dropna(inplace=True)

# # Extract features and labels
# X = df.drop(columns=["label", "frame"])
# y = df["label"]
# feature_names = X.columns.tolist()

# # Split into train/test sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # Train the Random Forest model
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train, y_train)

# # Predict and evaluate
# y_pred = clf.predict(X_test)
# y_probs = clf.predict_proba(X_test)[:, 1]

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("AUC:", roc_auc_score(y_test, y_probs))
# print("F1 Score:", f1_score(y_test, y_pred))

# # Feature importance (Random Forest)
# importances = clf.feature_importances_
# print("\n--- Random Forest Feature Importances ---")
# for name, score in zip(feature_names, importances):
#     print(f"{name}: {score:.4f}")

# # Plot bar chart
# plt.figure(figsize=(10, 5))
# plt.barh(feature_names, importances)
# plt.title("Random Forest Feature Importances")
# plt.xlabel("Importance")
# plt.tight_layout()
# plt.savefig("rf_feature_importances.png")
# plt.show()

# # SHAP explanation (CPU-compatible version)
# explainer = shap.Explainer(clf, X_train, feature_names=feature_names)
# shap_values = explainer(X_test)

# # SHAP summary plot
# shap.summary_plot(shap_values, X_test, show=False)
# plt.savefig("shap_summary_plot.png", bbox_inches='tight')
# plt.show()
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import joblib

# Load pre-split datasets
df_train = pd.read_csv("train_pid_grouped.csv")
df_test = pd.read_csv("test_pid_grouped.csv")

# Features & labels
drop_cols = [col for col in df_train.columns if col.startswith("frame") or col in ("pid", "label")]
X_train = df_train.drop(columns=drop_cols, errors="ignore")
y_train = df_train["label"]

X_test = df_test.drop(columns=drop_cols, errors="ignore")
y_test = df_test["label"]

# Train model
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
y_probs = clf.predict_proba(X_test)[:, 1]

# Metrics
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_probs)
f1 = f1_score(y_test, y_pred)

print(f"✅ Random Forest Evaluation on PID-grouped split:")
print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")
print(f"F1 Score: {f1:.4f}")

# Optional: Save model
joblib.dump(clf, "rf_model_pid_grouped.joblib")
