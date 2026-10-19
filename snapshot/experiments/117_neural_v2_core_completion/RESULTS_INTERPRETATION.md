# Exp117: what the neural completion establishes

Nine new models completed on CUDA and passed independent verification. All
training and validation selection finished before final test access. Existing
63-D, v1, raw-pose, base and union references were reused, not retrained.

## Direct core56 versus full63

Mean results over seeds 42/43/44, fixed threshold 0.5; PR-AUC is average precision.

| Model and cohort | BIU input | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|
| MSLSTM, standardized PIE | full63 | 0.828921 | 0.859921 | 0.938407 |
| MSLSTM, standardized PIE | core56 | 0.812193 | 0.851442 | 0.932063 |
| PedCMT, standardized PIE | full63 | 0.809673 | 0.789451 | 0.925997 |
| PedCMT, standardized PIE | core56 | 0.821133 | 0.849174 | 0.930261 |
| PedCMT, clean JAAD | full63 | 0.637304 | 0.660957 | 0.687890 |
| PedCMT, clean JAAD | core56 | 0.591135 | 0.632710 | 0.669930 |

### MSLSTM PIE

Core56 loses F1, PR-AUC and ROC-AUC in all three paired seeds. Mean deltas are
-0.016729 F1, -0.008479 AP and -0.006344 ROC-AUC. The result does not support
replacing full63 with core56 as an improvement in this matched architecture.

### PedCMT PIE

Core56 has better means but not consistently favorable paired seed deltas:
F1 improves in one of three seeds, AP in two, and ROC-AUC in one. The mean gain
partly reflects avoiding the weak full63 seed44. Core56 has substantially smaller
observed seed SD (F1 0.003583 versus 0.040305; AP 0.011925 versus 0.105651), but
three seeds do not establish general stability beyond this experiment.

Relative to the saved no-BIU baseline, core56 improves mean F1 from 0.792614 to
0.821133 but decreases mean AP from 0.858395 to 0.849174. Relative to raw pose,
it improves both means (raw-pose F1/AP: 0.774570/0.821714). These are distinct
comparisons; an improvement over raw pose is not automatically an improvement
over the no-pose baseline in every metric.

### PedCMT clean JAAD

Core56 has lower means across all three metrics. F1 is lower in every seed.
AP improves in one seed and ROC-AUC in two, so the mean ranking losses must not
be described as consistent in every seed. The operating point is unstable:
core56 seed42 predicts CROSS for all 1501 test windows (781 FP, zero FN), while
the full63 reference does so in seed44. This behavior follows the saved
validation-BCE checkpoint selection; no threshold or checkpoint was changed to
hide it. No causal explanation is established by these results alone.

## Efficiency and scope

The only changed weight matrix is 128x63 -> 128x56, removing 896 parameters:

| Architecture | Full63 total | Core56 total |
|---|---:|---:|
| MSLSTM | 6,839,562 | 6,838,666 |
| PedCMT | 1,859,376 | 1,858,480 |

This is a negligible whole-model size saving, not a defensible major inference
speed improvement. Summed measured per-seed training-stage durations were
389.70 seconds, including the runner's checkpoint/validation work; this is not
predictor latency or end-to-end perception latency.

The new results support retaining precise core56/full63 labels in the evidence.
They do not justify retrospectively relabeling all neural v2 results as 56-D,
or claiming one variant is best in every model/dataset. A single future default
is a scope/design decision; historical results must retain their actual inputs.

This completion covers the three standardized neural tables in the current
thesis scope. It does not rerun native 30-frame protocols or every historical
architecture search. No thesis text was edited and no additional experiment
was launched after these results.

## Reproducibility

- Exact execution command and source lineage: `DESIGN.md`, `README.md`.
- Complete metrics and sample SD: `metrics/per_seed_results.csv`,
  `metrics/summary_results.csv`.
- Paired changes against every reused control: `metrics/paired_deltas.csv`.
- Error transitions: `metrics/error_transitions_vs_full63.csv`.
- Source paths and SHA-256: `metrics/all_sources.json`.
- Model-shape and parent replay checks: `metrics/all_audits.json`.
- Independent saved-result and checkpoint-selection verification:
  `INDEPENDENT_VERIFICATION.json`.

All results are retrospective matched comparisons on previously exposed
cohorts, not untouched generalization confirmations.
