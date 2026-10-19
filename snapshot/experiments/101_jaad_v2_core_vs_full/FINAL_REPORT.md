# Exp101 Final Report

Clean JAAD, 16 observations, TTE30--60, fixed0.5, seeds42/43/44.
Three core56 baselines reused from Exp100; three new full63 fits.

| Pathway | Total features | F1 mean +/- SD | PR-AUC mean +/- SD | ROC-AUC mean +/- SD |
|---|---:|---:|---:|---:|
| core56 | 145 | 0.725254 +/- 0.001950 | 0.730421 +/- 0.006032 | 0.764966 +/- 0.002559 |
| full63 | 152 | 0.718818 +/- 0.002102 | 0.732446 +/- 0.005640 | 0.761364 +/- 0.004733 |

## Paired full63 minus core56

- f1: -0.006436; favorable seeds 0/3.
- pr_auc: +0.002025; favorable seeds 1/3.
- roc_auc: -0.003602; favorable seeds 0/3.

## Seed-ensemble paired track bootstrap

| Metric | Delta | 95% lower | 95% upper |
|---|---:|---:|---:|
| f1 | -0.009432 | -0.020587 | +0.000562 |
| pr_auc | +0.001572 | -0.021453 | +0.026472 |
| roc_auc | -0.003751 | -0.015712 | +0.008485 |

## Interpretation and provenance

The comparison adds only the seven existing direction/geometry outputs listed in DESIGN.md.
All first145 columns are exactly unchanged, including BBox, temporal summaries, existing
pose-quality and annotated categorical vehicle action (not measured physical ego speed).
All56 core features were rederived from the same observation-only cached pose/BBoxes
and checked for parity; no pose estimator or baseline was retrained.

The historical default1920x1080 normalization and direction fallback remain unchanged.
This isolates seven values within this pathway, not universal BIU-version superiority.
The test cohort was previously exposed: retrospective matched ablation, not untouched
confirmation. Bootstrap intervals use seed-ensemble probabilities, not mean-per-seed
metrics, and are descriptive without multiplicity adjustment.

Detailed source hashes and parity: metrics/*_source_audit.json, baseline_reuse_audit.json,
feature_contract.json and freeze.json. Exact commands: logs/commands.log.
Per-seed scores, error transitions and high-confidence mistakes are in metrics/.
