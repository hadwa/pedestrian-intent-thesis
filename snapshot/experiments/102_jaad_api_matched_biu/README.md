# Exp102: API-Style JAAD Matched BIU Comparison

Five frozen XGBoost input arms on the exact Exp083 JAAD_beh and JAAD_all
populations, each trained separately. All metrics use fixed0.5 and seeds42/43/44.

Status: COMPLETE. All27 new fits and60 prediction-file checks passed; nine unit
tests passed. Three historical baseline runs were reused rather than repeated.
See [FINAL_REPORT.md](FINAL_REPORT.md) for full mean/sampleSD and seed deltas.

## Result Summary

| Cohort | Input | F1 mean | PR-AUC mean | ROC-AUC mean |
|---|---|---:|---:|---:|
| JAAD_beh | BBox | 0.776370 | 0.722481 | 0.651162 |
| JAAD_beh | BBox + v1 | 0.781669 | 0.745902 | 0.684650 |
| JAAD_beh | BBox + core-v2 | 0.778232 | 0.745954 | 0.688548 |
| JAAD_beh | BBox + full-v2 | 0.779577 | 0.753045 | 0.691779 |
| JAAD_beh | BBox + both | 0.782335 | 0.772707 | 0.722024 |
| JAAD_all | BBox | 0.660318 | 0.749522 | 0.927780 |
| JAAD_all | BBox + v1 | 0.690144 | 0.772970 | 0.932490 |
| JAAD_all | BBox + core-v2 | 0.677206 | 0.768659 | 0.928386 |
| JAAD_all | BBox + full-v2 | 0.679608 | 0.772136 | 0.928459 |
| JAAD_all | BBox + both | 0.689324 | 0.774357 | 0.932026 |

BIU-containing bundles improve mean F1 and PR-AUC over BBox within both
populations. V1 has higher mean F1 than either v2 variant in both populations;
full-v2 gives better mean ranking than v1 on behavior pedestrians but not on
JAAD_all. Combining versions produces the strongest mean JAAD_beh metrics.
On JAAD_all, v1 and the union are close, with different metric winners.

All predeclared bootstrap F1 intervals include zero. For JAAD_beh union minus
v1, the seed-ensemble PR-AUC interval is[+0.006627,+0.049636] and ROC-AUC
interval[+0.015788,+0.061606]; these exclude zero but are descriptive and not
multiplicity-adjusted. They are not intervals for mean-per-seed deltas.
No universal BIU-version superiority is established. No further run launched.

## Frozen Inputs

| Arm | Input | Features |
|---|---|---:|
| A | BBox | 53 |
| B | BBox + v1 | 125 |
| C | BBox + v2-core | 109 |
| D | BBox + full-v2 | 116 |
| E | BBox + complete v1 and full-v2, namespaced | 188 |

No temporal contrasts, road/ego stream, raw-pose stream or new perception.
Core56 is a strict column subset of the same full63 measurements. BIUs use
Exp098/089 preprocessing with original per-video image dimensions; the union
is not the historical name-deduplicated Exp083 union.

## Avoided Duplicate Work

- Reuse three Exp031 JAAD_all BBox XGBoost probability sets after source,
  population, ordering, BBox and recipe checks. Recompute fixed0.5 reporting,
  not the historical validation-selected thresholds. No saved model/runtime
  lock exists, so this is reconstructed lineage reuse, not live model parity.
- Reuse Exp083 manifests and authenticated cached poses; no pose estimator runs.
- Compute features once for each JAAD_all source window and reuse them for its
  identical JAAD_beh window. Only27 new models, rather than30, are fitted.

See [REUSE_AUDIT.md](REUSE_AUDIT.md) and metrics/reuse_audit.json. Shared clean/API
development rows passed extraction parity:657 training and121 validation
windows, maximum absolute difference9.1e-13. No existing experiment is edited.

## Protocol Limits

API TTE30--60 means track-index steps to the sequence endpoint, not invariably
30--60 raw frames to a verified crossing onset. Preserve Exp083's exact
observed-crossing/unverified-onset cases and non-behavior negatives. Two all
training windows contain raw observation gaps. This is a retrospective
published-protocol-aligned comparison, not the clean anticipatory benchmark,
not proof of row-identical reproduction of another paper, and not untouched
test confirmation. Report both cohorts and every arm regardless of outcome.

## Execution

Commands from /home/abdelrahman/Hadwa/BIU-PID-Distillation-clean:

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -m pytest experiments/102_jaad_api_matched_biu/scripts/test_exp102.py -q
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/102_jaad_api_matched_biu/scripts/run_experiment.py prepare >> experiments/102_jaad_api_matched_biu/logs/prepare.log 2>&1
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/102_jaad_api_matched_biu/scripts/run_experiment.py all > experiments/102_jaad_api_matched_biu/logs/run.log 2>&1
```

Phases: prepare, validate, evaluate, report, verify. Source hashes and execution
versions: metrics/freeze.json. Progress: logs/progress.log and logs/run.log.
The initial pretraining guard rejection and correction are preserved in
logs/prepare.log; no training occurred before that check was corrected to
match Exp083's actual track-index TTE. Regression test added before the fix.

Final outputs: FINAL_REPORT.md; metrics/summary_results.csv,
per_seed_results.csv, paired_deltas.csv, ensemble_track_bootstrap.csv,
error_transitions_fixed_0p5.csv; predictions/; models/;
metrics/verification.json; EXPERIMENT_COMPLETE.json only after verification.
