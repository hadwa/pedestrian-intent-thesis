# PSI BIU-v2 attribution for the retained behavioral predictor

Authorized diagnostic, 2026-10-19. Explain the exact Exp116 `psi_ego` models,
reused as Exp124 arm C, using the complete recorded test population and seeds
42/43/44. Input: BBox53 + canonical BIU-v2 56 + temporal/quality 21 + speed 5
= 135 features. PSI task: 15 observed frames, human-rated intent at frame 16.
No retraining, feature changes, thresholds, calibration or model selection.

Use XGBoost exact `pred_contribs=True, approx_contribs=False` on the raw margin,
with the same training-path reference and grouping rules used for the thesis PIE
and clean-JAAD attribution. Exclude the intercept from all importance denominators.
For each seed, BIU share = sum of mean absolute feature contributions over its
56 columns / sum over all 135 columns. Keep the extra temporal and confidence
columns outside the BIU block. Average seed-specific percentages, not pooled models.

For each of the 15 BIU cues, report its feature-level absolute mass and percentage
of BIU/all-input mass, separately from its net grouped contribution. For window i,
sum signed SHAP within the cue, then multiply by (2*y_i-1) for label alignment.
Toward/away/neutral uses epsilon 1e-10, with all-window and positive/negative-class
summaries. These are explanations, not fractions of accuracy, removal effects or
causal evidence. No test-based selection follows.

Require exact column identity against the canonical 56-feature definition, unique
sample IDs, label and sample-order equality with Exp116 and Exp124 predictions,
probability reproduction, decision agreement, margin additivity and sigmoid
agreement. Verify source SHA-256 hashes remain unchanged. All diagnostic outputs
are new files here; existing experiment artifacts are read-only. Thesis figures
may be updated separately only after the checks pass.
