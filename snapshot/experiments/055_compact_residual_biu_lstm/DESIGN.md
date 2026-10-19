# Exp055 Frozen Design

## Objective

Develop a compact, fully neural and meaningfully recurrent PIE pedestrian-intention
predictor on the 16-observed-frame, TTE 30--60 benchmark. The model must preserve
BBox+BIU as its protected behavioral anchor and allow automatically obtained ego and
YOLOP context to make only a sample-dependent bounded correction.

The final target is the mean over seeds 42, 43 and 44 at fixed threshold 0.5:

- F1 greater than 0.869;
- PR-AUC greater than 0.916;
- ROC-AUC greater than 0.965.

All three conditions must hold. Exp055 is development of a new neural architecture,
not a controlled single-factor ablation against Exp053.

## Frozen Benchmark And Discipline

- Observation: exactly 16 consecutive frames.
- Time to event: 30--60 frames.
- Train: set01, set02 and set06, 2,366 windows.
- Validation: set04 and set05, 3,619 windows.
- Final test: set03, 3,760 windows.
- Seeds: 42, 43 and 44.
- Primary threshold: fixed 0.5.
- Development and candidate selection use validation only.
- The test cache may not be constructed, loaded or authenticated until one final
  candidate has been frozen in `FROZEN_FINAL_CANDIDATE.json`.
- Previous independent set03 exposure is disclosed. No Exp055 decision may be changed
  after the final candidate is frozen and set03 evaluation begins.

## Inputs

Only automatically/runtime-obtainable inputs are permitted:

- 53-D frozen Exp012 BBox window summary;
- a 16-frame BBox sequence, either absent, normalized raw 4-D `cx,cy,w,h`, or the
  frozen 8-D motion sequence described below;
- 72-D BIU-v1, exact 56-D Exp012 BIU-v2 core, or their deduplicated union;
- 16-frame ego speed and two-component ego acceleration;
- 16 x 20 YOLOP features.

The 8-D BBox temporal input per frame is:

`cx, cy, log(w), log(h), delta_cx, delta_cy, delta_log(w), delta_log(h)`.

The first-frame deltas are zero. Width and height are clamped to `1e-6` before the
logarithm. All sequence values are computed from observation frames only.

The exact 56-D BIU-v2 core is selected by feature name from the frozen Exp012 schema,
not by position. For the joint representation, BIU-v1 is followed by only those
BIU-v2-core features whose names do not already occur in BIU-v1. Feature ordering is
frozen and saved during preparation.

Forbidden inputs include GT Action, GT Look, GT static semantic context, raw RGB,
future frames, test-derived features, RF/XGBoost predictions, and per-frame pose
confidence as a separate predictor stream. The explicitly approved frozen 72-D BIU-v1
is preserved exactly, including its `biu_valid_frame_count` and
`biu_low_confidence_frame_count` window summaries. Therefore `v1` and `joint_unique`
contain these two frozen quality-count cues; they do not consume the 16 x 17 confidence
matrix directly.

## Train-Only Preprocessing

- Reuse the authenticated Exp050 train/validation cache and Exp053 BBox53 cache.
- Fit BIU mean/std statistics on the 2,366 original training rows only.
- Fit BBox sequence mean/std statistics over frames from the 2,366 original training
  windows only.
- Replace non-finite values with training medians before standardization.
- Constant dimensions use standard deviation 1.
- Apply the frozen statistics unchanged to augmented training rows and validation.
- Test transforms are produced only after the final candidate is frozen.
- Reuse Exp050's seed-specific oversampling and Gaussian-augmentation manifests.

The failed Exp054 result is retained as evidence that BIU-v2 standardization alone did
not improve the oversized Exp053 model. Exp055 standardization is part of a different
compact MLP anchor and is not interpreted as a repeat of Exp054.

## Frozen Architecture

### Summary anchor

Concatenate standardized BBox53 with one BIU representation. Apply:

`Linear(input_dim, 96) -> GELU -> Dropout(0.20) -> Linear(96, 48) -> GELU`.

### Temporal BBox encoder

When enabled, apply a one-layer bidirectional LSTM with hidden size 16 per direction,
followed by learned temporal attention from 32 to 16 to 1. Its output is 32-D.

Concatenate the 48-D summary embedding with the optional 32-D BBox temporal embedding,
then apply `Linear(anchor_input_dim, 48) -> GELU -> Dropout(0.20)`. A final linear layer
produces the scalar `anchor_logit`.

### Context encoders

Each active context stream has its own one-layer bidirectional LSTM and temporal
attention:

- ego input: 3-D per frame, hidden size 12 per direction, 24-D output;
- YOLOP input: 20-D per frame, hidden size 16 per direction, 32-D output.

Active context embeddings are concatenated and projected by
`Linear(context_input_dim, 32) -> GELU -> Dropout(0.20)`.

### Protected residual fusion

The context branch produces:

- `bounded_delta = 2 * tanh(Linear(context_embedding, 1))`;
- `gate = sigmoid(Linear([anchor_embedding, context_embedding], 16) -> GELU -> Linear(16, 1))`.

The final scalar logit is:

`final_logit = anchor_logit + gate * bounded_delta`.

The delta output layer is initialized to zero. The final gate bias is initialized to
`-2`, so context begins conservatively. Context never replaces or globally averages
the anchor.

During training only, each active context modality is independently dropped for an
entire sample with probability 0.20. Anchor inputs are never modality-dropped.

## Loss And Optimization

- Primary loss: binary cross entropy on `final_logit`.
- Auxiliary anchor loss: binary cross entropy on `anchor_logit`, weight 0.30.
- Residual penalty: mean squared applied correction, weight 0.001.
- Optimizer: AdamW, learning rate 0.001, weight decay 0.0001.
- Batch size: 64.
- Maximum epochs: 60.
- Early stopping patience: 10 completed epochs without validation-score improvement.
- Gradient norm clipping: 5.0.
- No scheduler, label smoothing, focal loss or calibration.

The checkpoint score is the harmonic mean of validation fixed-0.5 F1 and validation
PR-AUC. A strictly larger score wins; exact ties choose the earlier epoch.

## Validation-Only Candidate Search

Stage 1 trains all nine anchor candidates across all three seeds:

- BIU: `v1`, `v2_core`, `joint_unique`;
- BBox temporal mode: `summary_only`, `raw4`, `motion8`;
- context: none.

Select the Stage-1 anchor by highest mean checkpoint score across seeds. Ties within
`1e-6` are broken by higher mean PR-AUC, then higher mean fixed-0.5 F1, then fewer
parameters, then lexical candidate ID.

Stage 2 keeps the selected anchor fixed and trains four context modes across all seeds:

- none, reusing the exact Stage-1 artifacts;
- ego only;
- YOLOP only;
- ego plus YOLOP.

Select the final candidate using the same frozen aggregate rule. Write its full identity,
selection inputs and artifact hashes to `FROZEN_FINAL_CANDIDATE.json`. Only then may the
runner access set03 once for all three frozen-seed checkpoints.

## Required Reporting

Report per-seed and mean/std fixed-0.5 accuracy, precision, recall and F1, plus ROC-AUC
and PR-AUC. Also report parameter counts, predictor latency, anchor-only metrics,
context gate/delta distributions, and high-confidence FP/FN counts at probabilities
at least 0.9 or at most 0.1. Preserve validation search tables separately from final
test results.
