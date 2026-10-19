# Exp049: Exp048-to-JAAD Zero-Shot Evaluation

Status: complete. All 18 frozen Exp048 checkpoints were evaluated once on the
same official JAAD test cohort after the authenticated pre-inference gate.

This inference-only experiment applies Exp048 A--F to JAAD without training,
fine-tuning, calibration, threshold selection, or architecture changes. It
uses 30 consecutive observation frames, the next raw frame as target, and the
inherited fixed threshold 0.5. The approved design is in `DESIGN.md`.

## Final Cohort

- Official JAAD `default/test.txt`: 111 videos with eligible windows.
- 273 behavior-annotated pedestrian tracks.
- 44,646 windows, with 27,031 positive and 17,615 negative frame-level targets.
- One identical ordered cohort for all A--F checkpoints.
- Complete BBox, semantic, ego, YOLOP, raw-pose, BIU-v2, and BIU-v1 coverage.

The original pose cache covered only 5,331 windows. The exact pinned
ViTPose-S-lineage configuration/checkpoint was used to extend it to 44,646 of
44,646 windows after a bit-identical historical smoke check. The reconstructed
YOLOP extractor reproduced three predeclared historical PIE vectors exactly,
then produced all 52,621 required pedestrian-frame vectors from 23,670 unique
JAAD observation images. Test labels remained sealed through these stages and
were attached only after `AUDIT_COMPLETE` authenticated every input artifact.

## Final Results

Mean and sample standard deviation over seeds 42, 43, and 44:

| Variant | Fixed-0.5 F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|
| A Full privileged | 0.6834 +/- 0.0169 | 0.8762 +/- 0.0050 | 0.8166 +/- 0.0056 |
| B No GT static | **0.7579 +/- 0.0121** | 0.7995 +/- 0.0235 | 0.7601 +/- 0.0159 |
| C Practical core | 0.5791 +/- 0.0892 | 0.8806 +/- 0.0406 | 0.8294 +/- 0.0417 |
| D Core + raw pose | 0.6740 +/- 0.0921 | 0.8714 +/- 0.0382 | 0.8135 +/- 0.0400 |
| E Core + BIU-v2 | 0.6353 +/- 0.1226 | 0.8850 +/- 0.0096 | 0.8440 +/- 0.0098 |
| F Core + BIU-v1 | 0.5634 +/- 0.0301 | **0.8992 +/- 0.0105** | **0.8525 +/- 0.0171** |

## Paired Findings

- A to B: removing GT static semantics increased fixed F1 by 0.0745 on every
  seed, but reduced PR-AUC by 0.0767 on every seed.
- B to C: removing GT Action/Look reduced fixed F1 by 0.1788 on every seed,
  while PR-AUC increased by 0.0811 on every seed. GT behavior aided the
  inherited decision rule, but not zero-shot ranking.
- C to D: raw pose increased fixed F1 by 0.0949 on every seed. Its PR-AUC
  delta was mixed and averaged -0.0092.
- C to E: BIU-v2 averaged +0.0562 F1 and +0.0044 PR-AUC, but both deltas had
  mixed signs across seeds. This is not a consistent BIU-v2 transfer success.
- C to F: BIU-v1 averaged +0.0186 PR-AUC but -0.0157 F1, with mixed seed
  deltas. F ranked best overall, but its fixed operating point transferred
  poorly.
- D to E: BIU-v2 averaged +0.0136 PR-AUC and +0.0306 ROC-AUC, but -0.0387 F1;
  only ROC-AUC improved on all seeds.
- D to F: BIU-v1 averaged +0.0278 PR-AUC and +0.0390 ROC-AUC, but fixed F1 was
  lower by 0.1106 on every seed.
- E to F: BIU-v1 PR-AUC was higher by 0.0142 on every seed, while F1 was mixed
  and 0.0719 lower on average.

## Probability Finding

Several variants transfer with an extreme negative score bias. Mean
high-confidence FN counts (`p <= 0.1`) per seed were A 8,102; B 1,922; C
10,770; D 7,404; E 8,565; and F 10,873. Mean high-confidence FP counts
(`p >= 0.9`) were A 920; B 1,819; C 311; D 826; E 403; and F 162. F's mean
probability median was only 0.037 despite its best PR-AUC. No JAAD calibration
or threshold tuning was performed.

## Conclusion

Automatically derived behavior did not reliably replace privileged GT
Action/Look at the inherited fixed 0.5 operating point. Raw pose was the
strongest automatic addition for fixed F1. BIU-v1 was the strongest ranker and
BIU-v2 was intermediate, showing transferable ordering information, but the
PIE-trained score scales did not transfer reliably enough for a fixed decision
rule. This is a zero-shot result under the recovered next-frame MSLSTM task,
not the later 16-frame/TTE 30--60 benchmark and not a reproduction of the
thesis's JAAD-selected-threshold row.

Machine-readable results are in `metrics/`. Heavy authenticated caches,
predictions, and logs remain local and intentionally untracked.
