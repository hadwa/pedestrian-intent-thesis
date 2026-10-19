# Exp032 Design

## Research question

Can BIU-v2 improve an MSLSTM-style temporal pedestrian-intention model when BIU-v2 is
introduced as an explicit auxiliary behavioral representation and both BIU-v2 and the raw
pose stream are derived from the same recovered MSLSTM pose measurements?

## Thesis-safe hypothesis

BIU-v2 does not introduce new physical information beyond pose. It is a deterministic
transformation of the same pose measurements. The hypothesis is narrower:

> BIU-v2 may expose normalized, direction-aware, articulated behavioral relationships in a
> representation that the MSLSTM architecture can use more reliably than raw coordinates
> alone.

## Protocol basis

Exp032 follows the recovered **local published/reproduced MSLSTM lineage**, not the
standardized Exp018 benchmark:

- dataset: PIE
- observation: `30` consecutive raw frames
- target: `pred_len = 1` raw frame after the observed window
- label: `behavior.cross` at the target frame
- split lineage: train `set01,set02,set04`; val `set05,set06`; test `set03`

If any of those task-defining items cannot be reproduced exactly from local evidence, stop.

## Variants

### Variant A — published-setting MSLSTM control

- Freshly trained in the Exp032 runner.
- Same recovered task construction, same pose source, same seeds, same training recipe,
  same checkpoint-selection rule, same evaluation code as Variant B.
- No BIU branch.

### Variant B — published-setting MSLSTM + BIU-v2 auxiliary branch

- Same as Variant A, plus one late auxiliary BIU-v2 branch.
- BIU-v2 remains a window-level representation and is fused late with the existing MSLSTM
  fused representation.
- The raw pose stream remains intact.

### Variant C — BIU-v2 replacing the raw pose stream

- Default status: **NOT RUN — architecturally non-equivalent**.
- Only revisit if a later implementation audit shows a fair replacement path without
  redefining BIU-v2 as a different temporal representation.

## Shared pose source rule

Exp032 will use the pose source belonging to the recovered local MSLSTM lineage:

- raw source: `MSLSTM-PID/Extracted Poses/{set}/{set}_{video}_poses.pkl`
- detector lineage: `Offline Extraction/yolo_pose_extraction.ipynb`
- detector family: `yolov8x-pose-p6`

Both paths must derive from this same source:

- raw pose branch input
- BIU-v2 extraction input

BIU-v2 must be derived from the same stored normalized MSLSTM pose estimates **after
deterministic de-normalization back to original image pixel coordinates** and before
calling the frozen BIU-v2 extractor.

Frozen de-normalization formula:

- `x_px = x_norm * frame_width`
- `y_px = y_norm * frame_height`

Required verification:

- record the original frame dimensions used for each sequence source
- on a small sampled set, re-normalize the de-normalized pixel coordinates and verify that
  the recovered normalized values match the stored pose vectors within numerical tolerance
  before any model fitting

No substitution with `data_keypoints_all.jsonl` is allowed for Exp032 unless a new review
explicitly changes scope.

## BIU-v2 definition freeze

Use the repository's frozen final BIU-v2 definition from `src/biu_v2_extractor.py`.

Do not:

- redesign BIUs
- add BIU-v3
- add confidence/reliability
- change BIU normalization to chase performance
- approximate missing BIUs silently

If the recovered MSLSTM pose skeleton cannot support the frozen BIU-v2 semantics, stop.

## BIU branch design

### Integration point

Late fusion only:

1. existing MSLSTM streams produce their context embeddings;
2. BIU-v2 window vector is projected through a small MLP;
3. BIU embedding is fused with the existing fused representation;
4. the original classifier head consumes the augmented fused representation.

### Fixed BIU projection rule

The BIU projection dimension is frozen at **128**.

Rationale:

- it matches the local MSLSTM attention dimension;
- it is small relative to the `512`-D fused stream context size;
- it avoids turning BIU into a large second network;
- it is fixed before any result inspection.

Allowed branch shape:

- `Linear(BIU_dim -> 128)`
- activation matching the implementation style (`ReLU` unless the recovered runner uses a
  different house style)
- optional dropout only if the baseline head already uses dropout in that location

No branch-size search is allowed.

## Capacity fairness rule

Primary rule:

- document Variant A vs Variant B total/trainable parameter counts and percentage increase

Secondary control:

- only consider a generic parameter-control branch if BIU raises trainable parameters by
  more than `10%` and the control is straightforward to add without opening a new
  architecture-search branch

This control is not part of the default primary run.

## Training policy

Use the recovered local MSLSTM training lineage wherever recoverable:

- seeds: `42`, `43`, `44`
- immutable balanced training index: `MSLSTM-PID/checkpoints/train_index.pkl`
  (`109,934` unique windows; `54,967` examples per class; undersampling seed `42`)
- optimizer: Adam
- learning rate: `1e-4`
- batch size: `32`
- epochs: `5`
- checkpoint selection: best validation `F1@0.5`

Validation and test retain all reconstructed windows. The balanced index applies
only to training, matching the active local notebook path.

Other implementation details may be filled from the local notebook lineage during
implementation, but no task-defining protocol item may drift.

## Revised parity standard

Published metrics from `ICVES_samalouty.pdf` are **descriptive paper-level reference
values only**, not a hard numerical parity gate:

- `Accuracy = 0.916`
- `AUC = 0.970`
- `F1 ≈ 0.899`
- `Precision ≈ 0.929`
- `Recall ≈ 0.871`

Do not claim exact reproduction of the paper.

Fresh Exp032 `Variant A` is the true controlled baseline for this experiment. The
scientific conclusion must come from the paired fresh comparison:

`Variant A -> Variant B`

under identical:

- task construction
- cohort
- pose source
- preprocessing
- training recipe
- seeds
- checkpoint selection
- evaluation rule

### Published-setting plausibility check

The published paper remains a plausibility/sanity check only.

If fresh Variant A is **wildly inconsistent** with the paper-level reference, investigate
before interpreting Variant B. This is a diagnostic gate, not an exact numerical
reproduction requirement, because:

- the exact published headline threshold/decision rule is not pinned
- no local pose-capable paper-run artifact bundle is preserved

### Wild inconsistency triggers

Investigate Variant A before interpreting Variant B if any of the following occur relative
to the paper-level reference:

- absolute `F1` drop greater than `0.08`
- absolute `ROC-AUC` drop greater than `0.05`
- architecture/modalities/task implementation drift discovered during the run audit

If such a mismatch appears, stop interpretation, audit the run, and document the cause
rather than forcing a parity claim.

## Reporting policy

### Published-setting reporting

Preserve the local lineage convention:

- checkpoint selected by validation `F1@0.5`
- report the published-protocol decision rule separately if validation-selected threshold
  is used for a secondary table

### Thesis-comparable reporting

Always report:

- Accuracy @ `0.5`
- Precision @ `0.5`
- Recall @ `0.5`
- F1 @ `0.5`
- ROC-AUC
- PR-AUC

Secondary only:

- validation-selected threshold
- corresponding Acc/Prec/Rec/F1

No test-label threshold selection is allowed.

## Primary comparison

The primary comparison is:

`Variant A -> Variant B`

Report paired seed-level deltas for:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC

Primary interpretation weight:

1. PR-AUC
2. F1
3. ROC-AUC

## Frozen interpretation categories

### Strong transfer success

Variant B improves PR-AUC and F1 consistently across seeds, without material degradation
elsewhere.

### Ranking/complementarity success

Variant B improves PR-AUC consistently while fixed-`0.5` F1 is approximately flat.

### Selective operating-point success

F1 improves mainly through a precision/recall tradeoff while threshold-free metrics remain
stable or better.

### Neutral

Differences are trivial or inconsistent across seeds.

### Negative

Variant B consistently harms both ranking and threshold-dependent performance.

If negative, stop. No branch-size, fusion-location, confidence, or dataset expansion is
allowed afterward without a new approval cycle.

## Required pre-training audits

Before any training:

1. verify exact split membership under the recovered local protocol
2. verify no future frames enter either raw pose or BIU construction
3. verify raw pose branch and BIU branch use the same underlying pose windows
4. verify checkpoint selection uses validation only
5. verify the exact BIU-v2 feature dimension and ordered feature-name list produced from
   the recovered MSLSTM pose source
6. verify that produced BIUs match the frozen BIU-v2 definition semantically and
   structurally
7. verify de-normalization and re-normalization consistency on a small sample before BIU
   extraction for training windows
8. report pose availability by cohort and do not describe zero-filled missing poses as
   measured keypoints

## Controlled conclusion rule

The thesis conclusion for Exp032 must be based on the paired fresh `Variant A -> Variant B`
comparison only.

The published paper numbers may be cited as background reference values for the recovered
MSLSTM setting, but not as the basis for claiming success or failure of BIU-v2 transfer.

If any BIU is:

- missing
- renamed
- approximated
- non-computable from the recovered skeleton

then document it and stop for review.

## Expected artifacts

- `metrics/per_seed_results.csv`
- `metrics/summary_results.csv`
- `metrics/paired_deltas.csv`
- `metrics/fixed_0p5_results.csv`
- `metrics/published_protocol_results.csv`
- optional `metrics/val_f1_results.csv`
- saved per-sample probabilities for both variants and all seeds

If warranted by the outcome:

- `metrics/error_transitions_fixed_0p5.csv`

## Implementation stop criteria

Stop before training if any of the following occurs:

1. observation/target construction cannot be reproduced exactly
2. split lineage cannot be reproduced exactly
3. pose source for the raw stream cannot be reproduced exactly enough to share with BIU-v2
4. BIU-v2 cannot be computed faithfully from the recovered skeleton/order
5. the numeric baseline parity target cannot be established
6. A and B would require different cohorts, different pose sources, or different training
   logic beyond adding the BIU branch
