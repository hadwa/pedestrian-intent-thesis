# Exp050 Frozen Design

## Question

Under the project's standardized PIE anticipatory benchmark, can automatically
pose-derived BIU-v1 or BIU-v2 recover predictive value lost when privileged PIE
Action/Look/static annotations are removed from the recovered MSLSTM system?

This repeats the Exp048 acquisition-type ablation concept under a different task.
Absolute metrics are not directly comparable to Exp048 or Exp049.

## Frozen Benchmark

- Observation: 16 consecutive frames, observation-only inputs.
- TTE: 30--60 frames, event-aligned labels, overlap 0.6.
- Train: set01/set02/set06, 2,366 windows (679 positive).
- Validation: set04/set05, 3,619 windows (751 positive).
- Final test: set03, 3,760 windows (1,005 positive).
- Splits: archive `data/splits_pie_wacv_pcpa_30_60_overlap06` CSVs.
- Seeds: 42, 43, 44.
- Test discipline: train/validation caches and all checkpoint choices are completed
  before set03 cache construction and the one frozen final evaluation.
- Existing set03 exposure is disclosed: set03 has been inspected in earlier independent
  experiments, but no Exp050 architecture, preprocessing, or success rule is selected
  using Exp050 set03 results.

## Frozen Variants

| Variant | Streams |
|---|---|
| A | BBox + GT Action + GT Look + ego speed + ego acceleration + GT static context + YOLOP |
| B | BBox + GT Action + GT Look + ego speed + ego acceleration + YOLOP |
| C | BBox + ego speed + ego acceleration + YOLOP |
| D | C + raw 17-joint Exp012 pose |
| E | C + frozen 63-D BIU-v2 |
| F | C + frozen 72-D BIU-v1 |

Action, Look, and 23-D static context are privileged PIE annotations. BBox and ego
come from PIE runtime measurements/vehicle telemetry; YOLOP and pose are cached
automatic perception. D/E/F use the same historical Exp012 pose measurements.

## Sources

- Split CSVs: `/home/abdelrahman/Hadwa/BIU-PID-Distillation/data/splits_pie_wacv_pcpa_30_60_overlap06/`
- PIE database: `/home/abdelrahman/Hadwa/MSLSTM-PID/PIE/annotations/data_cache/pie_database.pkl`
- Exp012 pose JSONL: `/home/abdelrahman/Hadwa/pedestrian-intent-biu/data/data_keypoints_all.jsonl`
- Pose SHA-256: `1f0907ebb33bb04c3a3434d9c89e7bd597e2b9e3d47fe30f488ffa9410054d87`
- YOLOP: `/home/abdelrahman/Hadwa/MSLSTM-PID/Extracted YOLOP/*.pkl`
- BIU-v1: frozen 70 geometric summaries plus valid-frame and low-confidence counts.
- BIU-v2: frozen 63-D extractor with the established BBox direction fallback.

The pose cache is byte-identical to the main historical BIU cache and documented as
ViTPose-S-lineage. The original checkpoint/configuration/execution log is unavailable,
so historical checkpoint identity is not cryptographically verifiable. This does not
confound D/E/F because all use the same pinned measurements.

## Model And Training

- Exp048 per-stream architecture: 2-layer bidirectional LSTM, hidden 256 per direction,
  temporal attention 128, 512-D context, adaptive softmax-weighted fusion.
- Head: Dropout(0.3), Linear(512,256), ReLU, Dropout(0.3), Linear(256,2).
- E projector: 63 -> 128 -> 512 with ReLU and Dropout(0.3).
- F projector: 72 -> 128 -> 512 with ReLU and Dropout(0.3).
- Adam, learning rate 1e-4, batch size 32, 15 epochs.
- Ego speed/acceleration normalization uses means and standard deviations fitted only
  from observation frames in the frozen training windows, then applied unchanged to
  validation and set03.
- Positive-class oversampling to the negative count using a seed-specific manifest.
- Gaussian noise sigma 0.02 on duplicated BBox, ego, YOLOP, and normalized pose.
- GT Action/Look/static values are not augmented.
- Best checkpoint: highest validation F1 at fixed 0.5, earliest epoch on ties.
- Final primary reporting: fixed threshold 0.5 Accuracy/Precision/Recall/F1 and
  threshold-independent ROC-AUC/PR-AUC.

## Precomputed Augmentation

Exp048 recomputed BIU inside `Dataset.__getitem__` on every epoch. Its augmentation
manifest assigns a fixed `noise_seed` to every duplicate, so those values are invariant
across epochs. Exp050 applies the same frozen augmentation and extractor once per
manifest row, authenticates the cache, and uses tensor lookup during training. A parity
test must show cached BIUs equal direct online recomputation before launch.

## Frozen Comparisons And Criteria

- A -> B: privileged static-context contribution.
- B -> C: privileged Action/Look contribution after static removal.
- C -> D/E/F: automatic behavior recovery.
- D vs E/F: raw-pose versus BIU representation from the same pose source.
- E vs F: complete BIU-v2 versus BIU-v1 representations.
- A vs E/F: privileged reference versus practical automatic representation.

Material delta and non-inferiority tolerance are both 0.005 for fixed-0.5 F1 and
PR-AUC. Favorable seed consistency means the paired delta has the same favorable sign
for all three seeds. Behavior recovery fractions are descriptive and only calculated
when B-C is positive.

No design, threshold, projector, feature, training setting, or variant may change after
detached execution starts.
