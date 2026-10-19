# Exp076 Design

Exp076 compares the completed Exp072-style one-stage XGBoost with a
capacity-matched sequential predictor on the exact PIE 16-frame, TTE 30--60,
overlap-0.6 cohort.

## Frozen Arms

`A one_stage` receives the exact 187-D Exp072 representation from all 16
observed frames and uses 250 trees.

`B two_stage` first computes the frozen joint representation over frames 1--8:
53 BBox summaries, 72 BIU-v1 summaries, the unique BIU-v2 cues retained by
Exp072, and five early ego-speed summaries. A 125-tree XGBoost produces a
provisional probability. A second 125-tree XGBoost receives the exact Exp072
187-D frame-16 representation plus that provisional probability.

The second-stage training probability is produced out of fold using five-fold
`StratifiedGroupKFold` with `track_id` as the group. The fitted stage-1 model
never produces its own stage-2 training input.

## Fairness

Both arms use seeds 42, 43, and 44, the same training rows, fixed threshold
0.5, and the same XGBoost hyperparameters. Total tree count is 250 for both
arms. No threshold, representation, fold count, or tree allocation may change
after validation or test results are observed.

The saved Exp072 feature matrix is reused for the full-window input. Early
features are recomputed from the exact first eight BBoxes and SHA-pinned pose
measurements. Their equations are the same frozen BBox/BIU equations applied
to the shorter observed prefix; no future frame is used.

## Evaluation

Both frozen arms are reported on validation and once on set03. Primary metrics
are fixed-0.5 F1, PR-AUC, and ROC-AUC. FP/FN and high-confidence errors are
also reported. This is a controlled ablation; set03 has prior exposure in
separate work and is not described as pristine.
