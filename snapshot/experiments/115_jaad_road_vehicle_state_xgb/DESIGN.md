# Exp115: road context plus vehicle state on three JAAD cohorts

Authorized 2026-09-12: user requests one model containing both inputs for clean JAAD, JAAD_beh and JAAD_all. No existing combined checkpoint was found; nine new lightweight XGBoost fits are authorized, three seeds for each independently trained cohort. Reuse cached upstream features; do not repeat perception.

## Frozen comparison

- R242: saved road-context parent (Exp081-D with Exp082 test evaluation for clean JAAD; Exp083 for API-style JAAD_beh/all). Reuse checkpoints and probabilities after authentication and probability parity checks.
- RV257: exact R242 features plus the 15 categorical vehicle-state features defined by Exp099. Same training rows, labels, XGBoost parameters and seed. Fit nine models, seeds42/43/44, 250 trees, depth4, learning rate0.05, subsample/column subsample0.8, L2=1, train negative/positive class weight, median imputation, four CPU threads. No validation selection, early stopping, calibration, hyperparameter search, train+validation refit or test-based decisions. All nine frozen candidates advance to test regardless of validation metrics.
- Clean JAAD additional context: reuse Exp099-B's 197-D vehicle-state model as V197 reference, with the same sample IDs. Its comparison with RV257 tests adding road. No matching vehicle-only models are trained for API cohorts in this experiment.

## Inputs and protocols

Shared portable182 = BBox53 + historical Combined BIU108 (including two pose-quality counters) + temporal21. Compensated road60 + nominal vehicle-state15 give257 features. BIU version is Combined throughout; no physical measured ego speed. Vehicle categories: stopped, moving_slow, moving_fast, decelerating, accelerating. Encode observation fraction, first one-hot and last one-hot (five each), without treating codes as ordered numeric speed. Use exact observed-frame XML annotations, fail closed on missing or invalid values, never drop windows. No future frames.

Preserve all parent cohort definitions: clean JAAD event-aligned behavior-pedestrian windows; API-style JAAD_beh and JAAD_all remain distinct. All use16 observed track frames and TTE30–60, overlap0.8; fixed decision threshold0.5. Carry forward Exp083's documented two nonconsecutive JAAD_all training windows on video_0098:0_98_544 with max gap134. Use their actual recorded observation frames, not a synthetic consecutive range. All validation/test windows must be raw-frame consecutive. No split changes or cohort pooling.

## Stages and reporting

1. Prepare train/validation features and audit XML coverage, source hashes, feature contracts, split video/track disjointness and parent validation-prediction parity. Freeze design/code, all development features, parent references and exact model parameters before fitting.
2. Train all9 RV257 models using only training labels. Persist and reload checkpoints, validation predictions, train imputer checks and parameter checks. Freeze model hashes before loading test features for this run.
3. Reuse matching test features and check observation XML coverage and split isolation; evaluate every frozen model exactly at0.5 and reuse matched parent results after parity checks. No tuning based on results. Previously exposed JAAD test sets make this a retrospective extension, not pristine external confirmation.
4. Report per-seed and mean/sampleSD accuracy, precision, recall, F1, AP(PR-AUC), ROC-AUC, balanced accuracy, Brier, log loss, FPR/FNR and confusion counts in CSVs. Compact main tables omit FP/FN. Paired per-seed deltas versus R242 and clean V197 are descriptive, not significance claims. Time only model fit and cached downstream prediction; do not call this end-to-end latency.
5. Add exact branch TreeSHAP on all RV257 validation/test windows, following previous attribution method: feature contributions in raw log-odds, sum signed within each branch, mean absolute net, normalize within seed, then mean/sampleSD. Preserve label strata and individual feature magnitudes. Attribution is diagnostic, not accuracy/causal benefit; pose-derived temporal/road inputs mean static BIU share is not total pose influence.
6. Independently verify saved probabilities, metrics, hashes, ID/label equality, partitioning, SHAP additivity and reported summaries. Produce a printable result/attribution supplement and append it to the previous PDF in this new folder.

Final JAAD interpretation/calibration remains open. Vehicle state is annotation-derived; no deployed detector claim. Exp082's historical validation-selected follow-up status remains explicit. Preserve all earlier results/manuscript edits; write only in this folder; no commit/push.
