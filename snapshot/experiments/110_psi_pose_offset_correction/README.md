# Exp110: PSI pose offset correction

Status: COMPLETE. Training-only video-disjoint OOF measurement diagnostic.
No crossing classifier training, JAAD evaluation, or PSI validation/test use.

## Outcome

One frozen34-constant coordinate correction passed the predeclared aggregate
advance gate. It is a modest measurement improvement, not improved crossing F1.

| Primary metric (lower is better) | Original | Corrected | Relative reduction |
| --- | ---: | ---: | ---: |
| Keyframe joint error / BBox height, video/family macro | 0.0733306 | 0.0706641 | 3.636% |
| Keyframe BIU cue error, training-IQR-scaled video/cue macro | 1.9360185 | 1.8807305 | 2.856% |

Both improve in all3 folds. Each fold fits16 videos and evaluates8 disjoint
videos; all24 are official PSI training videos. All17 joints meet the minimum
3 fitting-video support requirement in every fold. All4,842 matched reference
records are retained; reference validity/keyframe masks determine metric support.
Four synthetic execution tests passed. Aggregate values independently recomputed
from saved per-video CSVs match the reported summaries.

Per-cue keyframe video-macro MAE changes are mixed: face-angle proxy -11.10%,
foot elevation -6.50%, foot-lift binary error -18.16%, arm asymmetry -5.10%;
body-angle proxy +2.95%, gaze proxy +4.95%, head-angle proxy +0.15%, normalized
shoulder width +9.07%, stride +1.60%, torso lean +4.72%. Minus indicates lower
measurement error. Do not call every BIU more accurate, or select a new subset
of corrections based on these held-out results.

Secondary all-annotated joint error:0.071546->0.068450; cue score:
0.935553->0.884647. Non-keyframe annotations are not treated as independent
hand-clicked ground truth. Confidence and crossing labels were not inputs.

## What this establishes

A small per-joint normalized offset learned from other training videos can improve
aggregate pose/cue agreement on this PSI export. It does not repair a wrong-person
skeleton, establish statistical significance, prove JAAD transfer, or measure
final72-D/63-D window feature fidelity. This uses the same per-frame v1 equations
as Exp017. This is not probability calibration.

The gate permits planning a controlled JAAD train/validation-only comparison.
Before that comparison, freeze a correction fitted to all24 PSI training videos
using the identical estimator, choose an existing matched JAAD parent by lineage
(not test score), retain labels/cohorts/BIU equations/classifier settings/threshold,
and run unchanged versus corrected observations. Do not selectively enable offsets
for cues that improved here. No JAAD test access is authorized by this result.

## Evidence and reproduction

- DESIGN.md: frozen correction, fold construction, evaluation and2% gate.
- official_split.json and split_provenance.json: official source metadata.
- freeze.json: input/extractor/design/runner SHA256 identities.
- metrics/folds.csv: exact video-disjoint population.
- metrics/fitted_offsets.json: per-fold corrections and support.
- metrics/joint_errors.csv, cue_errors.csv: per-video paired metrics/support.
- metrics/joint_video_macro.csv, cue_video_macro.csv and summary.csv.
- EXPERIMENT_COMPLETE.json: exact gate values; advance_to_jaad=true means gate
  eligibility, not an already executed JAAD trial.
- run.log: persistent output.

Commands run from /home/abdelrahman/Hadwa/BIU-PID-Distillation-clean:

```bash
python experiments/110_psi_pose_offset_correction/recover_split.py
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -m unittest discover -s experiments/110_psi_pose_offset_correction -p test_run.py -v
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/110_psi_pose_offset_correction/run.py | tee experiments/110_psi_pose_offset_correction/run.log
```

The runner refuses to overwrite completed results. Existing PSI caches and all
earlier experiments are unchanged. No expensive pose network inference was needed;
CPU array statistics were sufficient. No new literature-derived architecture was
introduced. Original PSI paper already exists in papers/Jing_2026_PSI_arxiv_v3.pdf.
