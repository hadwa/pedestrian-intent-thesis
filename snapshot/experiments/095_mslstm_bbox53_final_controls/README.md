# Exp095: Final MSLSTM BBox53 Controls

Exp095 completes the standardized PIE MSLSTM representation table with three
new matched arms: raw pose (J), GT Action/Look (K), and BIU-Combined (L). It
reuses the authenticated G/H/I references from Exp053.

## Commands

Run from the repository root with the project environment:

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python \
  experiments/095_mslstm_bbox53_final_controls/scripts/run_experiment.py prepare
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python \
  experiments/095_mslstm_bbox53_final_controls/scripts/run_experiment.py gpu-smoke
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python \
  experiments/095_mslstm_bbox53_final_controls/scripts/run_experiment.py train-all
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python \
  experiments/095_mslstm_bbox53_final_controls/scripts/run_experiment.py evaluate
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python \
  experiments/095_mslstm_bbox53_final_controls/scripts/run_experiment.py verify
```

The runner is resumable by arm and seed. `evaluate` remains locked until all
nine new checkpoints authenticate. `EXPERIMENT_COMPLETE` is written only after
the final metrics, predictions, report, and checksums verify.
