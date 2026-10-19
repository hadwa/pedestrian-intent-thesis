# Exp084 Frozen Design: PedCMT BIU Transfer

## Scientific question

Under the project's standardized PIE benchmark, does a frozen BIU representation
improve a recent published neural crossing-intention architecture, and is BIU more
useful or more compact than raw pose from the same measurements?

This is an architecture-transfer experiment. It is not an exact reproduction of
the PedCMT paper's numerical result because the released PedCMT runner uses a
random pedestrian split across PIE sets, while this experiment preserves the
project's route-separated thesis benchmark.

## Frozen benchmark

- Dataset: PIE.
- Observation: 16 consecutive frames.
- TTE: 30--60 frames.
- Train: set01, set02, set06; 2,366 windows.
- Validation: set04, set05; 3,619 windows.
- Final test: set03; 3,760 windows.
- Threshold: fixed 0.5.
- Seeds: 42, 43, 44.
- Development and checkpoint selection use train/validation only.
- Set03 is loaded only by the final evaluation command after all 15 checkpoints exist.

## Frozen source caches

Reuse, without recomputation:

- `experiments/050_mslstm_practical_behavior_replacement_pie_16f_tte30_60/cache/trainval_base/`
- `experiments/050_mslstm_practical_behavior_replacement_pie_16f_tte30_60/cache/set03_final/`

The caches provide aligned BBox, ego speed, raw 17-joint pose, 72-D BIU-v1,
63-D BIU-v2 and sequence metadata. The pose source is the SHA-pinned historical
Exp012/main-BIU cache. No pose estimator is rerun.

The PedCMT auxiliary endpoint target is reconstructed deterministically from the
PIE annotation at each row's stored `event_frame`. It is a training target only;
no future BBox enters any predictor input.

## Frozen variants

| Variant | Inputs | Extra-pathway representation |
|---|---|---|
| A | BBox + ego speed | None |
| B | A + raw pose | 16 x 34 pose sequence through a PedCMT-style temporal encoder |
| C | A + BIU-v1 | Train-standardized 72-D vector projected to one 128-D context token |
| D | A + BIU-v2 | Train-standardized 63-D vector projected to one 128-D context token |
| E | A + BIU-v1/v2 union | Train-standardized, source-namespaced 135-D concatenation projected to one 128-D context token |

The union retains all 72 v1 and 63 v2 values. Twenty shared feature names are
source-namespaced rather than dropped because audit showed that v1 and v2 values
are not numerically identical for those fields.

## Architecture

Reconstruct PedCMT from the authors' public implementation at commit
`d9be3f87d4fa0d8d935c1fbc55188cab2c079ca3`:

- embedding dimension 128;
- feed-forward dimension 256;
- 8 attention heads;
- 4 modality/cross-modal transformer layers;
- 3 bottleneck units;
- 7 bottleneck layers;
- BBox and ego self-attention streams;
- learned cross-modal global token;
- BBox endpoint auxiliary task;
- original BBox/ego bottleneck classification pathway.

For B--E, the extra representation participates only as a token in the existing
cross-modal attention meeting. It is not added as a third equally summed
bottleneck. The resulting global token is fed back into the original BBox and ego
streams exactly where PedCMT already performs cross-modal exchange.

Variant B embeds the pose sequence and applies the same style of self-attention
and feed-forward blocks before contributing its summary token. Variants C--E use
the fixed projector:

`input_dim -> Linear(input_dim,128) -> ReLU -> Dropout(0.3) -> Linear(128,128) -> LayerNorm(128)`

No projector-size, insertion-point, fusion, or BIU search is permitted.

## Training

- Adam, learning rate 5e-4, betas (0.9, 0.98), epsilon 1e-6.
- Batch size 64.
- Maximum 200 epochs.
- Early stopping patience 25.
- Deterministic train-only positive oversampling per seed.
- No validation or test balancing.
- Classification loss: binary cross entropy.
- Auxiliary loss: MSE on normalized event-frame BBox.
- Stable learned log-variance multi-task weighting, initialized equally.
- Checkpoint: minimum validation classification loss, earliest epoch on ties.
- The training recipe and data order are identical across A--E.

The maximum epoch and patience are a predeclared compute-bounded reconstruction,
not claimed to reproduce the authors' 2,000-epoch released runner.

## Evaluation and interpretation

Report per seed and mean +/- sample standard deviation:

- accuracy, precision, recall and F1 at fixed 0.5;
- ROC-AUC and PR-AUC from probabilities;
- TP, TN, FP and FN;
- high-confidence FP (`p >= 0.9`) and FN (`p <= 0.1`);
- parameter counts, training time and predictor latency.

Primary comparisons:

- A -> C/D/E: whether each BIU representation improves fresh PedCMT.
- B -> C/D/E: raw pose versus BIU from identical pose measurements.
- C <-> D: BIU-v1 versus BIU-v2.
- C/D <-> E: whether their union adds value.

The result supports BIU-specific enhancement only if a BIU variant improves A and
also improves B, or matches B within 0.005 F1 and PR-AUC with a materially smaller
extra pathway. If B improves A but BIU does not improve or match B, the conclusion
is pose-source success rather than BIU success.

## Exposure and stopping rule

Set03 has prior exposure from independent work. Exp084 design, feature definitions,
architecture, seeds and evaluation rules are frozen before Exp084 set03 access.
After one final evaluation of A--E, stop. Do not change the fusion, BIU features,
threshold or hyperparameters in response to set03.
