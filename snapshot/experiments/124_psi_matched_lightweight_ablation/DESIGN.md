# Exp124: complete matched PSI lightweight architecture ablation

User-approved completion of the missing controls,2026-09-18. Retrospective
matched evidence on previously exposed PSI cohorts. No original outputs changed.

Four fixed arms: A=BBox53 (new); B=BBox53+canonical BIU56=109 (new);
C=BBox53+BIU56+21temporal+5ego=135 (reuse Exp116 psi_ego);
D=C+60road=195 (reuse Exp116 psi_road_ego). No7aux, no v1 or union.
The temporal21 includes12 BIU-derived,6 BBox and3 pose-confidence summaries;
confidence is outside canonical BIU56. Ego is official speed summaries; road
features are the existing TwinLiteNet+ geometry. No re-extraction or feature change.

Use exact Exp116 extended PSI rows/order and float32 features:26295 training,
14922 validation,20547 test windows. Exclude the same585 training windows from
videos0032/0055/0093 for missing official speed. All four arms share that
population, including A/B that do not themselves need speed. PSI-native task:
15 observed frames predict human-intent mean-vote binary target at next frame16;
NOT16 observations/TTE30--60. Preserve labels/ties and frozen original splits.

Clone saved Exp116 psi_ego pipeline per seed42/43/44: training-only median
imputation,XGBoost250trees,depth4,LR.05,subsample.8,colsample.8,L2=1,4CPUthreads,
training negative/positive weight7697/18598. No early stopping, augmentation,
resampling, calibration, tuning or validation-based model/threshold selection.
Fixed threshold.5; AP=average precision. Only selected input columns differ.

Six new fits; reuse six extended models/predictions rather than train duplicates.
Before fitting verify named core56/absence of auxiliaries/v1, exact feature
prefixes, matching C/D development rows and first135columns, split/video/track
disjointness, parent recipe and replay of existing validation probabilities.
Freeze code/design/source hashes. Train all six models and freeze hashes before
opening test arrays. Evaluate EVERY predeclared new arm once, regardless of
validation scores, on the existing fixed test cohort. Reuse C/D saved results
with row/probability/metric parity checks. No outcome-based candidate selection.

Report test and validation per-seed/meanSD accuracy,macro-F1,positive-F1,AP,
ROC-AUC,precision/recall,confusion counts and adjacent paired deltas. Save exact
sources,cohort checks,feature contracts,models,predictions,logs and completion.
Original Exp113/116 simple rows remain valid on their DIFFERENT training cohort;
do not overwrite or relabel them. No thesis edits, commits,pushes or follow-ups.
