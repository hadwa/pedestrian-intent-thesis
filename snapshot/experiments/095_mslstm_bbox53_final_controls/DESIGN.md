# Exp095 Frozen Design

## Question

Under the standardized PIE anticipatory benchmark and the normalized MSLSTM
BBox53 architecture, how do raw pose, ground-truth Action/Look, and the combined
BIU-v1/v2 representation compare with the existing core, BIU-v1, and BIU-v2
arms?

This is a retrospective matched-control study. PIE set03 has been used by prior
experiments. Every Exp095 arm, comparison, feature definition, seed, checkpoint
rule, and threshold is frozen here before Exp095 accesses set03.

## Frozen Protocol

- Dataset: PIE.
- Observation: 16 consecutive frames.
- Time to event: 30--60 frames.
- Window overlap: 0.6.
- Train: set01/set02/set06, 2,366 windows.
- Validation: set04/set05, 3,619 windows.
- Test: set03, 3,760 windows.
- Seeds: 42, 43, and 44.
- Training: 15 epochs, Adam at `1e-4`, batch size 32.
- Checkpoint: highest validation F1 at fixed 0.5, earliest epoch on ties.
- Evaluation: fixed threshold 0.5 plus ROC-AUC and PR-AUC.
- Oversampling and Gaussian augmentation: identical to Exp050/051/053.

## Existing References

| Arm | Inputs |
|---|---|
| G | BBox53 + ego speed + ego acceleration + YOLOP |
| H | G + 72-D BIU-v1 |
| I | G + 63-D BIU-v2 |

G/H/I metrics and predictions are authenticated Exp053 artifacts. They are not
retrained or copied into new checkpoints.

## New Controls

| Arm | Inputs | Purpose |
|---|---|---|
| J | G + raw 16x34 pose | Matched raw-pose representation control |
| K | BBox53 + GT Action + GT Look + ego speed + ego acceleration + YOLOP | Matched privileged-input reference |
| L | G + 135-D BIU-Combined | v1/v2 complementarity control |

Arm K excludes static context. It differs from G only by the two ground-truth
behavior streams. Arm L retains all 72 BIU-v1 and all 63 BIU-v2 values. Every
field is source-namespaced; same-named v1/v2 fields are not dropped because the
two extractors can compute different values under their different direction,
normalization, and aggregation conventions.

The BIU-Combined pathway is:

`135 -> 128 -> ReLU -> Dropout(0.3) -> 512 -> non-affine LayerNorm`

Raw pose, Action, and Look retain the same two-layer bidirectional-LSTM and
attention pathways used by the parent architecture.

## Frozen Comparisons

- G to J/K/L: incremental raw-pose, privileged-behavior, and combined-BIU value.
- J to H/I/L: raw pose versus engineered pose representations.
- K to H/I/L: predictive utility relative to privileged Action/Look.
- H/I to L: whether BIU v1 and BIU v2 are complementary.

All comparisons are model- and protocol-specific. They do not establish that a
representation contains universally more information, that BIUs are
semantically equivalent to Action/Look, or that BIUs universally replace either
raw pose or privileged annotations.

## Reused Sources

- Exp050 train/validation, augmented, and set03 caches.
- Exp053 BBox53 train/validation, augmented, and set03 arrays.
- Exp053 G/H/I saved metrics and predictions.
- The pinned Exp012 ViTPose-S-lineage pose measurements already used by the
  parent experiments.

No pose or YOLOP extraction is rerun. New outputs are written only inside this
Exp095 folder; no earlier benchmark result is overwritten.
