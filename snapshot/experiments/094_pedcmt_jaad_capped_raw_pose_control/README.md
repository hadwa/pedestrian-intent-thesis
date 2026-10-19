# Exp094: JAAD Capped Raw-Pose Control

Exp094 supplies the missing matched control for Exp092. The frozen Exp089 PedCMT
BBox-plus-ego parent is retained, the Exp090 Variant B raw-pose residual is trained on
the natural JAAD development distribution, and its inference correction is multiplied
by exactly 0.10. The resulting P10 arm is compared with the parent A arm and the frozen
Exp092 BIU-union U10 arm.

The test set has already been exposed by Exp089 and Exp092. Consequently, this is a
retrospective matched control and must not be described as pristine confirmation.

## Commands

Run these commands in order from the repository root:

```bash
python experiments/094_pedcmt_jaad_capped_raw_pose_control/scripts/run_experiment.py prepare --device cpu
python experiments/094_pedcmt_jaad_capped_raw_pose_control/scripts/run_experiment.py smoke --device cpu
python experiments/094_pedcmt_jaad_capped_raw_pose_control/scripts/run_experiment.py train-all --device cuda
python experiments/094_pedcmt_jaad_capped_raw_pose_control/scripts/run_experiment.py validate
python experiments/094_pedcmt_jaad_capped_raw_pose_control/scripts/run_experiment.py evaluate --device cuda
python experiments/094_pedcmt_jaad_capped_raw_pose_control/scripts/run_experiment.py analyze
python experiments/094_pedcmt_jaad_capped_raw_pose_control/scripts/run_experiment.py report
python experiments/094_pedcmt_jaad_capped_raw_pose_control/scripts/run_experiment.py verify
```

Use `--device cpu` for training/evaluation if CUDA is unavailable. Preparation and
validation never load test arrays. Evaluation remains locked until all three training
checkpoints and validation outputs are frozen.

## Artifact map

- `checkpoints/`: P raw-pose branch checkpoints only.
- `logs/`: training histories and completion records.
- `metrics/`: lineage, validation, final metrics, paired deltas, transitions,
  efficiency, bootstrap intervals, and verification.
- `predictions/`: capped validation/test P10 outputs and aligned A/P10/U10 tables.
- `RESULTS.md`: generated only after final evaluation and analysis.

Generated artifacts are intentionally not committed unless explicitly requested.
