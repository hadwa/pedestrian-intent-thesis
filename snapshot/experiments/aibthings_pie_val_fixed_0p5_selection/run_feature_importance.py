"""Feature importances for the selected PIE BIU+BBox Random Forest.

Inputs (read-only):
- Reproduced estimator: experiments/aibthings_gpu_runtime_minimal/pie_biu_bbox_random_forest_reproduced.joblib
  (verified to match saved val/test probabilities within 1e-12; see
   experiments/aibthings_model_only_runtime/model_reproduction_check.csv)
- Feature table (column order): /home/abdelrahman/Hadwa/BIU-PID-Distillation/data/processed/features_wacv_pcpa_30_60_overlap06/biu_bbox_test.csv

Outputs (this folder):
- rf_biu_bbox_feature_importances.csv     : per-feature Gini importances
- rf_biu_bbox_group_importances.csv       : aggregated by BIU cue / BBox family
- fig_biu_bbox_importance.pdf/.png        : paper figure
"""

import os

import joblib
import numpy as np
import pandas as pd

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = "/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/aibthings_gpu_runtime_minimal/pie_biu_bbox_random_forest_reproduced.joblib"
FEATS = "/home/abdelrahman/Hadwa/BIU-PID-Distillation/data/processed/features_wacv_pcpa_30_60_overlap06/biu_bbox_test.csv"

est = joblib.load(MODEL)
df = pd.read_csv(FEATS, nrows=5)
feature_cols = [c for c in df.columns if c.startswith("biu__") or c.startswith("bbox__")]
model = est[-1] if hasattr(est, "steps") else est
imp = model.feature_importances_
assert len(imp) == len(feature_cols), (len(imp), len(feature_cols))

fi = pd.DataFrame({"feature": feature_cols, "importance": imp}).sort_values("importance", ascending=False)
fi.to_csv(os.path.join(OUT_DIR, "rf_biu_bbox_feature_importances.csv"), index=False)

BIU_CUES = [
    "torso_lean_angle", "stride_length", "head_yaw_angle", "lead_foot_elevation",
    "lead_foot_lifted", "arm_swing_asymmetry", "gaze_to_car_angle", "body_yaw_angle",
    "face_yaw_angle", "norm_shoulder_width",
]


def group_of(col):
    if col.startswith("biu__"):
        base = col[len("biu__"):]
        for cue in sorted(BIU_CUES, key=len, reverse=True):
            if base.startswith(cue):
                return "BIU: " + cue
        return "BIU: other"
    return "BBox"


fi["group"] = fi["feature"].map(group_of)
grp = fi.groupby("group", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
grp.to_csv(os.path.join(OUT_DIR, "rf_biu_bbox_group_importances.csv"), index=False)
print(grp.to_string(index=False))
print("\nBIU total:", fi.loc[fi["feature"].str.startswith("biu__"), "importance"].sum())
print("BBox total:", fi.loc[fi["feature"].str.startswith("bbox__"), "importance"].sum())
print("\nTop 15 individual features:")
print(fi.head(15).to_string(index=False))
