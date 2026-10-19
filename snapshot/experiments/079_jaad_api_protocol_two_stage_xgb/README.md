# Exp079: JAAD API-Protocol Two-Stage XGBoost Diagnostic

Status: complete.

This package reruns Exp077's frozen one-stage and 8+8 two-stage XGBoost
comparison on Exp031 Protocol A's reconstructed flawed JAAD subsets. See
`DESIGN.md` and `pre_registration.json` for the frozen contract.

Expected commands:

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python scripts/prepare_pose_manifest.py
LD_LIBRARY_PATH=/usr/lib/wsl/lib /home/abdelrahman/miniconda3/envs/cv_env/bin/python scripts/run_pose_extraction.py --device cuda:0
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python scripts/run_experiment.py prepare
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python scripts/run_experiment.py validate
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python scripts/run_experiment.py evaluate
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python scripts/run_experiment.py report
```

All paths above are relative to this experiment directory except where the
runner resolves repository artifacts explicitly.

## Final Mean Test Results

| Setting | Model | F1 | PR-AUC | ROC-AUC | Accuracy | High-conf FP | High-conf FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flawed Protocol A native | BBox53 XGB (Exp031) | 0.6603 | 0.7495 | 0.9278 | 0.8502 | 190.3 | 20.7 |
| Flawed Protocol A native | One-stage 182-D | 0.6860 | 0.7751 | 0.9304 | 0.8740 | 155.0 | 56.3 |
| Flawed Protocol A native | Two-stage 8+8 | 0.6553 | 0.7597 | 0.9270 | 0.8471 | 194.3 | 33.3 |
| PIE to flawed Protocol A zero-shot | One-stage 182-D | 0.2434 | 0.1865 | 0.4699 | 0.4611 | 1697.7 | 268.3 |
| PIE to flawed Protocol A zero-shot | Two-stage 8+8 | 0.2390 | 0.1877 | 0.4480 | 0.4105 | 1391.0 | 205.0 |

The one-stage native model is the strongest Exp079 arm. It improves over the
same-protocol BBox53 control, but the two-stage design does not improve it.

The zero-shot collapse is caused by Protocol A's additional non-behavior
population: 4,851 test windows are non-behavior tracks and all are negative.
Exact-overlap features and probabilities match Exp077 with maximum absolute
difference zero, excluding runner drift as the explanation.

## Pose Runtime

The RTX 3070 GPU extraction completed 56,463 missing pedestrian-frame records
in 1,016.38 seconds. Steady-state CUDA-synchronized ViTPose-S inference was
19.42 ms median and 40.94 ms p95 per unique source image. The median-derived
sequential 16-frame estimate is 310.71 ms; it is not a batched-video runtime.

See `FINAL_REPORT.md` for full per-seed results and protocol caveats.
