# Exp048 Protocol Amendment: BIU-v1 Variant F

## Status

Approved on 2026-08-19 before Variant E, Variant F, or any Exp048 set03
evaluation. This is a transparent pre-test addition to the frozen A-E study;
it does not reinterpret or modify A-E.

The complete design and rationale are recorded in
`docs/superpowers/specs/2026-08-19-exp048-biu-v1-protocol-amendment-design.md`.

## Frozen Variant

Variant F is:

`BBox + Ego Speed + Ego Acceleration + YOLOP + exact 72-D BIU-v1`

BIU-v1 is extracted from the same SHA-pinned historical Exp012 pose artifact
used by Variants D and E. It contains the historical 70 geometric summaries
plus `biu_valid_frame_count` and `biu_low_confidence_frame_count`. Exp048
applies the unchanged window-agnostic formulas to its 30 observed frames.

The fixed projection is:

`Linear(72,128) -> ReLU -> Dropout(0.3) -> Linear(128,512)`

F has no raw-pose recurrent branch. All training, sampling, augmentation,
fusion, classifier, checkpoint-selection, seed, and threshold rules remain
unchanged.

## Source And Seal

- Pose path: `/home/abdelrahman/Hadwa/Pedestrian-keypoints-crossing-intent/classification/data/data_keypoints_all.jsonl`
- Pose SHA-256: `1f0907ebb33bb04c3a3434d9c89e7bd597e2b9e3d47fe30f488ffa9410054d87`
- Original core SHA-256: `cf19c532199a57ee8f53d23f07315d3aeadfe714ac43a7aeafebef17863da1dd`
- Original runner SHA-256: `7de82843f4576dacc4ff02b3a61b61ddf0869338479ecc60827ea4044feec185`

The original core and runner remain byte-identical. The amendment cache reads
only sets01/02/04/05/06; set03 lines are filtered before JSON decoding. The
original Exp048 `evaluate` and `finalize` commands are prohibited after this
amendment. Final evaluation must use the amendment runner and is authorized
only after all 18 A-F training records authenticate.

## Commands

```bash
python experiments/048_mslstm_practical_behavior_replacement/scripts/run_biu_v1_amendment.py status
python experiments/048_mslstm_practical_behavior_replacement/scripts/run_biu_v1_amendment.py --resume audit
python experiments/048_mslstm_practical_behavior_replacement/scripts/run_experiment.py --resume train-all
python experiments/048_mslstm_practical_behavior_replacement/scripts/run_biu_v1_amendment.py --resume train-all
python experiments/048_mslstm_practical_behavior_replacement/scripts/run_biu_v1_amendment.py evaluate
python experiments/048_mslstm_practical_behavior_replacement/scripts/run_biu_v1_amendment.py finalize
```

Do not run either evaluation command until the amendment status reports all
15 original A-E and three F records complete.
