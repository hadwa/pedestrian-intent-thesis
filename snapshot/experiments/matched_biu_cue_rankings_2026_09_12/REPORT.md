# Matched cue importance and direction rankings

This is a diagnostic explanation of existing saved XGBoost models, not a feature-removal experiment. Every version comparison uses the same windows within its setting. Cross-setting rankings describe different fitted models and are not a controlled dataset or culture comparison.

Importance (%) is the sum of absolute feature SHAP magnitudes for a cue, divided by the total static BIU magnitude. Quality counters and full-v2 metadata are included in this denominator and labeled; extra early/late pose branches are separate. Importance measures reliance, not correctness.

For each window, sum a cue's signed feature SHAP values and multiply by +1 for a positive label or -1 for a negative label. Toward/away percentages count positive/negative results. They are percentages of windows pushed toward/away from the label, NOT percentages of correctly classified windows. A cue may push toward the label while the total model prediction remains wrong.

Balanced net is the average of the positive-class and negative-class mean signed contributions, with equal weight for each class. Positive supports the label on average; negative opposes it. Units are log-odds relative to the fitted model's SHAP baseline. Positive support and opposing magnitude, window-weighted and equal-track net, class-specific percentages and TP/TN/FP/FN results are in the CSV files.

Tables are ordered by VALIDATION importance. Support ranks in review_table.csv use VALIDATION balanced net. Test explains generalization only and must not select cues for deletion. All results average seeds 42/43/44; seed variation is not uncertainty across pedestrians. Overlapping windows and JAAD cohorts are not independent samples.

Correlated inputs, feature grouping, interaction allocation and different model baselines affect SHAP values. A low-importance cue is not necessarily useless, and an adverse cue is not automatically worth removing. These rankings identify candidates for scrutiny; effects of removal require a separate controlled validation experiment. No fitting or removal was done here.

## Coverage

| Setting | Versions | Other inputs |
| --- | --- | --- |
| PIE / 096 | v1, core56, Combined108 | Temporal + numeric speed |
| Clean JAAD / 098 | v1, full63, union135 | BBox + BIU |
| Clean JAAD / 100 | v1, core56, Combined108 | Temporal + vehicle state |
| JAAD_beh / 102 | v1, core56, full63, union135 | BBox + BIU |
| JAAD_all / 102 | v1, core56, full63, union135 | BBox + BIU |
| PSI / 113 | v1, full63; no matched union | BBox + BIU; native intent task |

Historical Combined108 retains v1 anatomy plus 36 v2 motion features. Union135 keeps v1-72 and full-v2-63 separately, including both angle conventions. They are different representations.

Exp096/100 retain common BBox and ego context, but v2 and Combined also include v2-specific early/late pose features omitted from v1. Their version comparison is a pathway comparison, not a change to static BIUs alone.

PIE and clean JAAD use 16 observations / TTE 30-60. API JAAD_beh/all use their recorded API window construction, including protocol limitations documented in Exp102. PSI uses 15 observations to predict human-rated intent at frame 16. Its labels and task are different; its signed magnitudes are not a future-crossing benchmark comparison.

The latest road-context architectures Exp075, Exp082-D, Exp083, Exp114 and Exp115 do not have complete matched v1/v2/Combined triplets in the inspected artifacts. Their Combined-only importance reports cannot answer a version comparison. Exp113 cannot borrow Exp114 as its union arm, because other inputs differ. No missing model was refitted.

No universal version or cue-removal verdict is selected here. JAAD's final interpretation and calibration remain open.

## Strongest validation support and cues to scrutinize

Order below uses validation equal-class net contribution. Test columns are descriptive follow-up; no feature-selection decisions are made. Full rankings and per-class results are in review_table.csv, rankings.csv and summary.csv.

### PIE / Exp096 / temporal + speed / v1 (72)

| label | val_importance_pct | test_importance_pct | val_balanced_net | test_balanced_net | test_toward_pct | pattern |
| --- | --- | --- | --- | --- | --- | --- |
| v1 / Vertical ankle separation | 24.806 | 24.082 | 0.153 | 0.150 | 60.966 | Supportive averages |
| v1 / Ankle separation / shoulder width | 12.140 | 11.058 | 0.075 | 0.075 | 57.278 | Supportive averages |
| v1 / Nose–shoulder orientation | 13.317 | 14.150 | 0.040 | 0.039 | 43.378 | Supportive averages |
| v1 / Nose–ear orientation | 13.041 | 13.484 | -0.009 | 0.017 | 61.392 | Split sign changes; mixed test classes |
| v1 / Torso lean | 3.130 | 3.174 | 0.002 | -0.001 | 56.853 | Split sign changes; mixed test classes |

### PIE / Exp096 / temporal + speed / v2 core (56)

| label | val_importance_pct | test_importance_pct | val_balanced_net | test_balanced_net | test_toward_pct | pattern |
| --- | --- | --- | --- | --- | --- | --- |
| v2 / Shoulder-center horizontal motion | 15.836 | 16.351 | 0.189 | 0.254 | 86.250 | Supportive averages |
| v2 / Vertical ankle separation | 18.788 | 17.940 | 0.131 | 0.130 | 63.076 | Supportive averages |
| v2 / Lower-body center motion | 7.172 | 7.976 | 0.104 | 0.154 | 87.074 | Supportive averages |
| v2 / Forward ankle displacement | 2.921 | 2.784 | -0.006 | -0.001 | 42.420 | Adverse averages; mixed test classes |
| v2 / Ankle-center horizontal motion | 2.235 | 2.163 | -0.001 | 0.006 | 49.202 | Split sign changes |

### PIE / Exp096 / temporal + speed / Historical Combined (108)

| label | val_importance_pct | test_importance_pct | val_balanced_net | test_balanced_net | test_toward_pct | pattern |
| --- | --- | --- | --- | --- | --- | --- |
| v2 / Shoulder-center horizontal motion | 11.095 | 11.567 | 0.208 | 0.287 | 85.966 | Supportive averages |
| v1 / Vertical ankle separation | 17.293 | 16.344 | 0.161 | 0.147 | 62.730 | Supportive averages |
| v2 / Lower-body center motion | 6.083 | 6.538 | 0.128 | 0.183 | 86.720 | Supportive averages |
| v1 / Nose–ear orientation | 8.620 | 8.721 | -0.013 | 0.014 | 55.124 | Split sign changes; mixed test classes |
| v2 / Forward ankle displacement | 0.988 | 1.061 | 0.000 | 0.005 | 55.177 | Supportive averages |

### Clean JAAD / Exp098 / BBox + BIU / v1 (72)

| label | val_importance_pct | test_importance_pct | val_balanced_net | test_balanced_net | test_toward_pct | pattern |
| --- | --- | --- | --- | --- | --- | --- |
| v1 / Normalized shoulder width | 17.722 | 18.669 | 0.211 | 0.182 | 62.469 | Supportive averages |
| v1 / Wrist-height asymmetry | 9.353 | 9.594 | 0.084 | 0.041 | 57.340 | Supportive averages |
| v1 / Nose–hip orientation | 10.525 | 11.190 | 0.076 | -0.030 | 49.767 | Split sign changes; mixed test classes |
| v1 / Binary ankle-elevation indicator | 1.135 | 1.006 | 0.004 | 0.001 | 49.745 | Supportive averages; mixed test classes |
| v1 / Ankle separation / shoulder width | 7.438 | 7.305 | 0.036 | 0.014 | 52.099 | Supportive averages; mixed test classes |

### Clean JAAD / Exp098 / BBox + BIU / v2 full (63)

| label | val_importance_pct | test_importance_pct | val_balanced_net | test_balanced_net | test_toward_pct | pattern |
| --- | --- | --- | --- | --- | --- | --- |
| v2 / Shoulder width [scale] | 10.571 | 14.829 | 0.112 | 0.131 | 56.274 | Supportive averages |
| v2 / Nose–shoulder orientation | 5.767 | 6.116 | 0.087 | 0.004 | 52.632 | Supportive averages; mixed test classes |
| v2 / Forward ankle displacement | 4.744 | 4.384 | 0.061 | 0.016 | 56.229 | Supportive averages; mixed test classes |
| v2 / Shoulder-line orientation | 4.612 | 4.245 | -0.013 | 0.037 | 57.650 | Split sign changes |
| v2 / Ankle-center horizontal motion | 2.641 | 2.701 | -0.009 | 0.028 | 59.316 | Split sign changes |

### Clean JAAD / Exp098 / BBox + BIU / Full namespaced union (135)

| label | val_importance_pct | test_importance_pct | val_balanced_net | test_balanced_net | test_toward_pct | pattern |
| --- | --- | --- | --- | --- | --- | --- |
| v1 / Normalized shoulder width | 9.202 | 9.821 | 0.192 | 0.152 | 61.626 | Supportive averages |
| v1 / Wrist-height asymmetry | 6.679 | 6.471 | 0.093 | 0.041 | 56.740 | Supportive averages |
| v1 / Nose–hip orientation | 5.787 | 6.150 | 0.067 | -0.019 | 50.677 | Split sign changes; mixed test classes |
| v2 / Ankle-center horizontal motion | 1.831 | 1.801 | -0.007 | 0.025 | 59.050 | Split sign changes |
| v2 / Lower-body motion magnitude | 1.118 | 1.333 | -0.006 | 0.015 | 60.515 | Split sign changes |

### Clean JAAD / Exp100 / temporal + vehicle state / v1 (72)

| label | val_importance_pct | test_importance_pct | val_balanced_net | test_balanced_net | test_toward_pct | pattern |
| --- | --- | --- | --- | --- | --- | --- |
| v1 / Normalized shoulder width | 15.884 | 17.547 | 0.167 | 0.151 | 62.669 | Supportive averages |
| v1 / Wrist-height asymmetry | 8.486 | 8.789 | 0.082 | 0.035 | 58.006 | Supportive averages; mixed test classes |
| v1 / Nose–hip orientation | 9.540 | 10.250 | 0.074 | -0.016 | 51.166 | Split sign changes; mixed test classes |
| v1 / Binary ankle-elevation indicator | 0.838 | 0.740 | 0.005 | 0.000 | 49.456 | Supportive averages; mixed test classes |
| v1 / Ankle separation / shoulder width | 8.857 | 8.520 | 0.009 | 0.014 | 54.453 | Supportive averages; mixed test classes |

### Clean JAAD / Exp100 / temporal + vehicle state / v2 core (56)

| label | val_importance_pct | test_importance_pct | val_balanced_net | test_balanced_net | test_toward_pct | pattern |
| --- | --- | --- | --- | --- | --- | --- |
| v2 / Nose–shoulder orientation | 11.014 | 11.716 | 0.089 | 0.008 | 51.677 | Supportive averages; mixed test classes |
| v2 / Vertical ankle separation | 20.964 | 19.245 | 0.084 | -0.012 | 47.257 | Split sign changes; mixed test classes |
| v2 / Forward ankle displacement | 8.901 | 8.048 | 0.060 | 0.025 | 56.962 | Supportive averages |
| v2 / Shoulder-line orientation | 9.873 | 9.227 | -0.017 | 0.046 | 59.449 | Split sign changes |
| v2 / Lower-body center motion | 0.952 | 1.132 | 0.000 | 0.006 | 57.317 | Supportive averages; mixed test classes |

### Clean JAAD / Exp100 / temporal + vehicle state / Historical Combined (108)

| label | val_importance_pct | test_importance_pct | val_balanced_net | test_balanced_net | test_toward_pct | pattern |
| --- | --- | --- | --- | --- | --- | --- |
| v1 / Normalized shoulder width | 12.785 | 13.714 | 0.166 | 0.136 | 62.625 | Supportive averages |
| v1 / Wrist-height asymmetry | 8.165 | 8.111 | 0.089 | 0.036 | 57.562 | Supportive averages; mixed test classes |
| v1 / Nose–hip orientation | 6.752 | 7.603 | 0.070 | -0.016 | 51.455 | Split sign changes; mixed test classes |
| v2 / Ankle-center horizontal motion | 2.117 | 2.156 | -0.002 | 0.017 | 60.382 | Split sign changes |
| v2 / Lower-body center motion | 1.071 | 1.166 | 0.002 | 0.010 | 59.827 | Supportive averages |

### JAAD_beh / Exp102 / BBox + BIU / v1 (72)

| label | val_importance_pct | test_importance_pct | val_balanced_net | test_balanced_net | test_toward_pct | pattern |
| --- | --- | --- | --- | --- | --- | --- |
| v1 / Vertical ankle separation | 15.953 | 15.993 | 0.258 | 0.070 | 62.857 | Supportive averages; mixed test classes |
| v1 / Ankle separation / shoulder width | 12.048 | 13.151 | 0.226 | 0.104 | 60.287 | Supportive averages |
| v1 / Normalized shoulder width | 10.735 | 11.004 | 0.182 | 0.111 | 67.145 | Supportive averages; mixed test classes |
| v1 / Shoulder-line orientation | 4.520 | 4.002 | 0.006 | -0.003 | 53.518 | Split sign changes; mixed test classes |
| v1 / Binary ankle-elevation indicator | 1.117 | 0.965 | 0.013 | 0.002 | 61.882 | Supportive averages; mixed test classes |

### JAAD_beh / Exp102 / BBox + BIU / v2 core (56)

| label | val_importance_pct | test_importance_pct | val_balanced_net | test_balanced_net | test_toward_pct | pattern |
| --- | --- | --- | --- | --- | --- | --- |
| v2 / Vertical ankle separation | 16.414 | 15.543 | 0.344 | 0.071 | 61.669 | Supportive averages; mixed test classes |
| v2 / Ankle separation / shoulder width | 10.692 | 11.209 | 0.228 | 0.106 | 60.358 | Supportive averages |
| v2 / Nose–shoulder orientation | 24.177 | 25.085 | 0.199 | 0.267 | 66.029 | Supportive averages |
| v2 / Lower-body motion magnitude | 4.105 | 4.196 | -0.014 | 0.015 | 54.740 | Split sign changes; mixed test classes |
| v2 / Knee-center horizontal motion | 2.044 | 2.092 | -0.008 | 0.007 | 56.318 | Split sign changes; mixed test classes |

### JAAD_beh / Exp102 / BBox + BIU / v2 full (63)

| label | val_importance_pct | test_importance_pct | val_balanced_net | test_balanced_net | test_toward_pct | pattern |
| --- | --- | --- | --- | --- | --- | --- |
| v2 / Vertical ankle separation | 13.847 | 13.212 | 0.299 | 0.054 | 61.173 | Supportive averages; mixed test classes |
| v2 / Ankle separation / shoulder width | 8.415 | 9.003 | 0.190 | 0.085 | 60.181 | Supportive averages |
| v2 / Nose–shoulder orientation | 21.430 | 22.393 | 0.176 | 0.250 | 65.302 | Supportive averages |
| v2 / Lower-body motion magnitude | 3.406 | 3.731 | -0.023 | 0.013 | 54.864 | Split sign changes; mixed test classes |
| v2 / Knee-center horizontal motion | 2.494 | 2.420 | -0.015 | 0.011 | 56.016 | Split sign changes; mixed test classes |

### JAAD_beh / Exp102 / BBox + BIU / Full namespaced union (135)

| label | val_importance_pct | test_importance_pct | val_balanced_net | test_balanced_net | test_toward_pct | pattern |
| --- | --- | --- | --- | --- | --- | --- |
| v1 / Normalized shoulder width | 8.791 | 9.454 | 0.186 | 0.136 | 66.738 | Supportive averages; mixed test classes |
| v1 / Vertical ankle separation | 7.391 | 7.336 | 0.184 | 0.037 | 60.216 | Supportive averages; mixed test classes |
| v1 / Wrist-height asymmetry | 8.335 | 7.870 | 0.183 | 0.071 | 62.254 | Supportive averages; mixed test classes |
| v2 / Lower-body motion magnitude | 1.840 | 2.225 | -0.030 | 0.014 | 58.001 | Split sign changes; mixed test classes |
| v2 / Ankle-center horizontal motion | 2.882 | 2.938 | -0.013 | 0.026 | 56.069 | Split sign changes; mixed test classes |

### JAAD_all / Exp102 / BBox + BIU / v1 (72)

| label | val_importance_pct | test_importance_pct | val_balanced_net | test_balanced_net | test_toward_pct | pattern |
| --- | --- | --- | --- | --- | --- | --- |
| v1 / Nose–ear orientation | 14.607 | 15.116 | 0.115 | 0.147 | 73.237 | Supportive averages |
| v1 / Ankle separation / shoulder width | 14.840 | 15.362 | 0.097 | 0.083 | 59.254 | Supportive averages |
| v1 / Vertical ankle separation | 12.954 | 12.793 | 0.084 | 0.050 | 57.303 | Supportive averages; mixed test classes |
| v1 / Nose–hip orientation | 7.647 | 7.329 | -0.018 | 0.026 | 62.265 | Split sign changes |
| v1 / Torso lean | 5.425 | 5.330 | -0.001 | -0.002 | 46.742 | Adverse averages; mixed test classes |

### JAAD_all / Exp102 / BBox + BIU / v2 core (56)

| label | val_importance_pct | test_importance_pct | val_balanced_net | test_balanced_net | test_toward_pct | pattern |
| --- | --- | --- | --- | --- | --- | --- |
| v2 / Ankle separation / shoulder width | 14.456 | 14.868 | 0.093 | 0.080 | 59.680 | Supportive averages |
| v2 / Nose–shoulder orientation | 13.922 | 14.747 | 0.085 | 0.124 | 71.138 | Supportive averages |
| v2 / Forward ankle displacement | 7.330 | 8.457 | 0.062 | 0.052 | 58.982 | Supportive averages |
| v2 / Torso lean | 9.411 | 10.071 | -0.017 | 0.022 | 56.749 | Split sign changes |
| v2 / Torso motion magnitude | 2.987 | 2.826 | -0.013 | -0.005 | 52.129 | Adverse averages; mixed test classes |

### JAAD_all / Exp102 / BBox + BIU / v2 full (63)

| label | val_importance_pct | test_importance_pct | val_balanced_net | test_balanced_net | test_toward_pct | pattern |
| --- | --- | --- | --- | --- | --- | --- |
| v2 / Torso length [scale] | 7.608 | 7.403 | 0.095 | 0.064 | 64.741 | Supportive averages |
| v2 / Ankle separation / shoulder width | 12.492 | 12.886 | 0.087 | 0.080 | 59.779 | Supportive averages |
| v2 / Nose–shoulder orientation | 12.021 | 12.812 | 0.084 | 0.119 | 71.257 | Supportive averages |
| v2 / Torso lean | 8.187 | 8.718 | -0.012 | 0.020 | 57.120 | Split sign changes |
| v2 / Torso motion magnitude | 3.113 | 3.044 | -0.011 | -0.006 | 50.931 | Adverse averages; mixed test classes |

### JAAD_all / Exp102 / BBox + BIU / Full namespaced union (135)

| label | val_importance_pct | test_importance_pct | val_balanced_net | test_balanced_net | test_toward_pct | pattern |
| --- | --- | --- | --- | --- | --- | --- |
| v1 / Nose–ear orientation | 10.168 | 10.482 | 0.123 | 0.140 | 72.915 | Supportive averages |
| v1 / Vertical ankle separation | 8.836 | 8.527 | 0.090 | 0.058 | 54.536 | Supportive averages; mixed test classes |
| v1 / Binary ankle-elevation indicator | 5.668 | 5.045 | 0.084 | 0.072 | 54.397 | Supportive averages |
| v1 / Nose–hip orientation | 4.474 | 3.945 | -0.026 | 0.006 | 62.701 | Split sign changes; mixed test classes |
| v2 / Torso motion magnitude | 2.274 | 2.170 | -0.013 | -0.009 | 49.916 | Adverse averages; mixed test classes |

### PSI / Exp113 / BBox + BIU / native intent task / v1 (72)

| label | val_importance_pct | test_importance_pct | val_balanced_net | test_balanced_net | test_toward_pct | pattern |
| --- | --- | --- | --- | --- | --- | --- |
| v1 / Ankle separation / shoulder width | 25.562 | 24.736 | 0.136 | 0.104 | 63.031 | Supportive averages |
| v1 / Nose–ear orientation | 14.882 | 15.447 | 0.050 | 0.033 | 62.903 | Supportive averages; mixed test classes |
| v1 / Nose–shoulder orientation | 9.847 | 10.767 | 0.033 | 0.011 | 59.606 | Supportive averages; mixed test classes |
| v1 / Wrist-height asymmetry | 6.399 | 6.558 | -0.012 | -0.003 | 54.404 | Adverse averages; mixed test classes |
| v1 / Normalized shoulder width | 9.190 | 8.487 | 0.003 | -0.002 | 45.700 | Split sign changes; mixed test classes |

### PSI / Exp113 / BBox + BIU / native intent task / v2 full (63)

| label | val_importance_pct | test_importance_pct | val_balanced_net | test_balanced_net | test_toward_pct | pattern |
| --- | --- | --- | --- | --- | --- | --- |
| v2 / Nose–shoulder orientation | 35.520 | 37.492 | 0.282 | 0.245 | 65.236 | Supportive averages |
| v2 / Ankle separation / shoulder width | 12.631 | 12.724 | 0.093 | 0.067 | 60.323 | Supportive averages |
| v2 / Relative ankle-displacement separation | 8.249 | 7.585 | 0.045 | 0.033 | 63.680 | Supportive averages |
| v2 / Shoulder width [scale] | 5.891 | 3.979 | -0.033 | -0.005 | 42.079 | Adverse averages; mixed test classes |
| v2 / Torso motion magnitude | 1.812 | 1.997 | -0.006 | -0.003 | 44.102 | Adverse averages; mixed test classes |

## Reproducibility

Commands, run from `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/matched_biu_cue_rankings_2026_09_12/analyze.py
MPLCONFIGDIR=/tmp/thesis-input-attribution-mpl /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/matched_biu_cue_rankings_2026_09_12/report.py
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/matched_biu_cue_rankings_2026_09_12/verify.py
```

Source paths and SHA-256 hashes: sources.json. Per-feature membership: membership.csv. Model/split/seed probability parity and SHAP additivity: checks.csv and verification.json. Compact cue-level per-window arrays are saved here to avoid repeating SHAP in future audits. No benchmark artifacts or manuscript files were changed. An incomplete first attempt stopped on an incorrect prepared-feature path; its intermediate arrays are isolated in attempt_001_source_path_error and excluded.
