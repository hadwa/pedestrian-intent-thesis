# PSI label/cohort audit: protocol decision required

Completed metadata and schema feasibility audit, 2026-09-11. No predictions or
training. Exp108 remains incomplete as a classifier experiment. This audit
supersedes the earlier claim that official metadata could not be located, not any
existing benchmark result. Raw annotations remain local and ignored by git.

## Verified findings

Both official PSI2 split files were recovered from the user-provided ZIP using
validated HTTP ranges, ETag pinning, ZIP CRC verification and SHA-256 records.
They have identical train/validation memberships (108/35 videos), but the
TrainVal copy includes test video_0192 while the Test copy excludes it (50 versus
49 test videos). This is not a train/test crossover. Do not silently choose one.

The 346-track, 190-video pose cache covers:

| Split | Cached videos | Cached tracks | Pose records |
|---|---:|---:|---:|
| Train | 105 | 171 | 29516 |
| Validation | 35 | 77 | 16463 |
| Test | 49 | 96 | 27555 |

Video_0111 is outside both official split copies. Train videos 0032, 0055 and
0093 have no records in this cache. Cached-video coverage does not prove complete
pedestrian/window coverage within each video. Full frame/track alignment still
must be checked when a task is authorized.

The deterministic first training pedestrian-intent schema sample is video_0001,
track_0: 123 observed frames, 11 annotators, 19 nonempty keyframe judgments
(5 cross, 10 not_cross, 4 not_sure), and 1334 empty annotator-frame slots. Empty
keyframe slots are not negative labels. The file contains per-frame BBoxes and
annotator-specific intent arrays. The cache's static binary labels are not yet
authenticated against these judgments. Sample counts are schema evidence only,
not estimates of dataset prevalence.

The official repository propagates intent forward from annotation keyframes and
describes already-crossed separately. The PSI paper v3, section 4, uses 15 observed
frames and predicts intent at frame 16, reporting accuracy and macro F1. This is
not the event-anchored 16-frame/TTE30--60 binary task of our saved models.

## Decision gate

Do not report BIU+BBox or extended XGBoost PSI classifier metrics yet. Recovery
is feasible, but a valid experiment requires choosing explicitly between:

1. PSI-native training/evaluation with its original target/aggregation and scoring.
   Existing binary PIE/JAAD models are not drop-in native PSI classifiers.
2. A binary zero-shot task/domain-transfer diagnostic with a predeclared
   annotator aggregation, uncertainty/already-crossed handling, sampling and
   observation horizon. It must not be presented as the native PSI benchmark.

The user approved finishing this audit, not a particular new label mapping.
Stop before changing label semantics or excluding uncertain cases. Recommended
next step is to freeze the intended target explicitly, then finish the full
annotation/cache join before any predictions. For a road-aware extension, matched
road assets are also needed; do not replace missing roads/ego data with zeros.
Exp099 vehicle categories cannot be assumed equivalent to PSI driving decisions.

## Evidence and commands

- `audit.json`: source SHA-256, split discrepancies and cache coverage.
- `archive_metadata_inventory.json`: archive annotation paths, sizes and CRCs.
- `source_metadata/` (ignored): two split JSONs and a training annotation sample.
- Existing `papers/Jing_2026_PSI_arxiv_v3.pdf`, section 4; no new paper download.
- [Official dataset preparation](https://github.com/PSI-Intention2022/PSI-Dataset).
- [Official challenge](https://github.com/PSI-Intention2022/PSI-Competition).
- [PSI paper](https://arxiv.org/abs/2112.02604v3).

From `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean`:

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/108_psi_xgboost_transfer/audit_2026_09_11/recover_audit.py > experiments/108_psi_xgboost_transfer/audit_2026_09_11/run_intent.log 2>&1
pdftotext -layout papers/Jing_2026_PSI_arxiv_v3.pdf /tmp/psi_audit_paper.txt
```

Initial generic cognitive-file sampling selected driving_decision.json; retained
run.log documents that attempt. The final script explicitly selects the first
pedestrian_intent.json. No videos or test cognitive judgments were downloaded.

Pre-commit verification: Exp110's four pytest tests passed; Exp111's four unittest
tests passed using its native `python .../test_run.py` entry point; Exp112 verify.py
reproduced all six validation probabilities/metrics and frozen hashes. A combined
pytest invocation failed on duplicate test module names, and standalone pytest
Exp111 collected its imported test_path_forbidden helper as a test; the native
unittest entry point avoids both discovery issues without altering frozen code.
