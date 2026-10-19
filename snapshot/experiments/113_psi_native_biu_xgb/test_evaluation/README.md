# Exp113 frozen PSI test result

All nine previously trained models evaluated with no retraining or threshold
changes. A=BBox53, B=BBox53+72-D v1, C=BBox53+63-D full v2, XGBoost seeds42/43/44.
PSI-native mean-vote binary labels;15 observed frames predict intent at frame16.
This is not the PIE/JAAD16-frame/TTE30--60 benchmark. See DESIGN.md for freeze.

## Test metrics

Mean +/- sample std, fixed predictor threshold.5:

| Variant | Accuracy | Crossing F1 | Macro-F1 | PR-AUC (AP) | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| A: BBox | .647183 +/- .008661 | .758027 +/- .007075 | .553438 +/- .007436 | .844446 +/- .004250 | .627414 +/- .008859 |
| B: BBox+v1 | .636865 +/- .003010 | .745929 +/- .002422 | .554829 +/- .003112 | .840465 +/- .003782 | .623225 +/- .004199 |
| C: BBox+v2 | .653786 +/- .003625 | .759333 +/- .002711 | .571337 +/- .003956 | .863645 +/- .000752 | .660694 +/- .000110 |

v2 versus BBox: mean AP +.019198, ROC-AUC +.033280, macro-F1 +.017899;
all three improve on every seed. Positive F1 changes +.001306 on average but
decreases .007012 for seed44; this is not consistent positive-F1 improvement.
Accuracy improves on average but not every seed. v1 does not improve mean AP,
ROC-AUC or crossing F1 over BBox; its very small macro-F1 gain is inconsistent.
v2 beats v1 on accuracy, crossing F1, macro-F1, AP and ROC-AUC on every seed.
These are descriptive seed comparisons, not independence-based significance tests.

| Variant | Mean FP | Mean FN | High-confidence FP (p>=.9) | High-confidence FN (p<=.1) |
|---|---:|---:|---:|---:|
| A | 3143.67 | 4105.67 | 911.67 | 644.67 |
| B | 2952.33 | 4509.00 | 689.67 | 699.33 |
| C | 2874.00 | 4239.67 | 739.00 | 727.00 |

Compared with A, C reduces mean FP by269.67 but adds134 FN. Total mean errors
decrease135.67. C high-confidence FP decrease172.67 but high-confidence FN
increase82.33. Thus even the stronger representation does not solve both error
types. Counts are averages across seeds, not pooled independent pedestrians.
For A->C, wrong-to-correct/correct-to-wrong counts are1694/1499,1767/1478,1541/1618
for seeds42/43/44. Detailed transition counts are saved in metrics/error_transitions.csv.

## Cohort and limitations

20547 windows,49 videos,96 tracks;15462 positive,5085 negative (75.252% positive).
The test-release split explicitly excludes video_0192, which the older TrainVal
split includes. This release-based choice was frozen before reading test labels
or scoring, not chosen for performance. Original test-release annotation member
paths and SHA hashes are recorded in recovery.json. All49 videos remain after
native processing; database.log records shorter/gapped track processing.

All20547 eligible windows have complete cached pose, so complete-pose metrics
equal full-cohort metrics. No zero/missing-pose windows or pose-based sample drops.
v2 invokes its frozen BBox direction fallback333 times (1.621%).3050 targets have
an exact mean vote of.5; native aggregation maps these ties to crossing. Uncertain
and initial unannotated judgments follow the official extension; exported static
cache labels are not used. This is human-estimated intent, not event-anchored
crossing occurrence. The perception checkpoint lineage limitation remains.

Always predicting CROSS gives accuracy.752519, crossing F1.858785,
macro-F1.429393, AP.752519, ROC-AUC.5. None of A/B/C beats this trivial rule on
accuracy or positive F1; all improve macro-F1 and ranking. Do not claim strong
overall accuracy from this experiment. The fixed class-weighting recipe and
threshold are unchanged. Overlapping windows mean these are not independent
samples; no test-driven adjustment follows this result.

Conclusion: v2 provides the more useful tested BIU representation for ranking and
macro-F1 on this PSI cohort; its tiny mean positive-F1 gain over BBox alone is not
consistent. v1's validation ranking gains did not carry over. This supports a
bounded representation finding, not universal v2 superiority or solved PSI intent.
The extended XGBoost/context trial has not started.

## Sources, checks and commands

Sources: ../models/{A,B,C}_seed{42,43,44}.joblib, ../feature_contract.json,
../official_code/, shared uncorrected data/psi_vitpose_s_all_tracks.jsonl;
freeze.json pins models, runner and design. Test annotations come from
PSI/PSI2.0_Test/annotations/cognitive_annotation_key_frame/ inside the previously
provided PSI_Dataset_1_2.zip. Local staging uses a TrainVal-named folder solely to
run the unchanged official hardcoded extension, never the real development folder.
No videos, scene perception or pose extraction was run.

All nine saved validation predictions reproduced exactly; raw feature checks used
deterministic first/middle/last validation windows. Test outputs were read back
and compared exactly. verify.py independently reproduces all test probabilities,
metrics, source hashes and15-frame input/next-frame label identities.

Working directory: /home/abdelrahman/Hadwa/BIU-PID-Distillation-clean

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/113_psi_native_biu_xgb/test_evaluation/evaluate.py recover > experiments/113_psi_native_biu_xgb/test_evaluation/recovery.log 2>&1
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/113_psi_native_biu_xgb/test_evaluation/evaluate.py evaluate > experiments/113_psi_native_biu_xgb/test_evaluation/evaluate.log 2>&1
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/113_psi_native_biu_xgb/test_evaluation/verify.py
```

Artifacts: metrics/, predictions/, cache/, recovery.json, freeze.json,
EXPERIMENT_COMPLETE.json, VERIFIED.json. New test outputs do not replace validation.
