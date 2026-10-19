# Exp100: JAAD Exp099 BIU pathway separation

Historical v1 and v2-core-56 pathways with identical BBox and categorical
vehicle-action inputs. Reuse Exp099's combined pathway; train six new models.
See DESIGN.md for dimensions and interpretation limits. No pose re-extraction.

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/100_jaad_exp099_biu_pathways/scripts/run_experiment.py all
```

Stages: prepare, validate, evaluate, report, verify. Results and exact sources
are under metrics/, saved probabilities under predictions/, new fitted models
under models/, and persistent progress under logs/. Test is locked until all
validation models are frozen. Completion: EXPERIMENT_COMPLETE.json.
