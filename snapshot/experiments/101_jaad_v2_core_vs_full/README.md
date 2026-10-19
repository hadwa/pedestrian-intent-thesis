# Exp101: JAAD v2-core-56 versus v2-full-63

Append seven existing direction/geometry values to Exp100's v2 pathway, with
all other features and training settings unchanged. Three new fits; three
authenticated parent baselines reused. See DESIGN.md for the frozen protocol.

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/101_jaad_v2_core_vs_full/scripts/run_experiment.py all
```

Phases: prepare, validate, evaluate, report, verify. Outputs: metrics/,
predictions/, models/, cache/, logs/, FINAL_REPORT.md, EXPERIMENT_COMPLETE.json.
No pose re-extraction. No changes to existing experiments. Stop after this run.

## Completed result

All three new full63 fits completed, all three baseline reuse checks passed,
all 12 saved validation/test prediction files reproduced their metrics, and
all nine unit tests passed. EXPERIMENT_COMPLETE.json records verified completion.

| Test metric, mean over seeds42/43/44 | Core56 | Full63 | Full minus core |
|---|---:|---:|---:|
| F1 at0.5 | 0.725254 | 0.718818 | -0.006436 |
| PR-AUC (average precision) | 0.730421 | 0.732446 | +0.002025 |
| ROC-AUC | 0.764966 | 0.761364 | -0.003602 |

F1 and ROC-AUC decreased in all three seeds; PR-AUC increased in only one.
The seven extras did not provide a consistent improvement in this matched
configuration. Every paired track-bootstrap interval includes zero, so this
is not evidence of universal full63 inferiority. This is a retrospective
matched ablation on previously exposed JAAD test data, not untouched confirmation.

Mean FP/FN counts changed from370.33/99.67 to377.33/104.33. High-confidence
FP>=0.9 increased from220.33 to231.00, while FN<=0.1 decreased from58.00 to50.33.
These are seed means, not fractional people in an individual run.

Source and result details: [FINAL_REPORT.md](FINAL_REPORT.md),
[per-seed metrics](metrics/per_seed_results.csv),
[source/model freeze](metrics/freeze.json),
[baseline reuse audit](metrics/baseline_reuse_audit.json),
[error transitions](metrics/error_transitions_fixed_0p5.csv).
The first145 columns and all56 recomputed core values matched exactly on
train1552, validation274 and test1501. The seven added values were read from
the same cached-pose extractor call, not a different full63 cache.

## Executed commands

From /home/abdelrahman/Hadwa/BIU-PID-Distillation-clean:

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -m pytest experiments/101_jaad_v2_core_vs_full/scripts/test_exp101.py -q
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/101_jaad_v2_core_vs_full/scripts/run_experiment.py prepare
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/101_jaad_v2_core_vs_full/scripts/run_experiment.py all > experiments/101_jaad_v2_core_vs_full/logs/run.log 2>&1
```

Persistent phase timestamps: logs/progress.log. Exact Python commands:
logs/commands.log. Existing cached inputs and saved baselines were reused;
only the three full63 models were trained. No subsequent experiment launched.
