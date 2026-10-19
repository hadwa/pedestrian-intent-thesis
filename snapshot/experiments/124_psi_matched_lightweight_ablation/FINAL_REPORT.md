# Matched PSI lightweight architecture results

All four arms share26295train/14922validation/20547test windows, labels/order, seeds42/43/44, model recipe and threshold0.5. A/B are new; C/D reuse Exp116. PSI-native15 observed frames -> intent at frame16; not the16-frame/TTE30--60 task. Retrospective matched evidence on previously exposed cohorts.

## Test means

| Input | Accuracy | Macro-F1 | Positive F1 | AP | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| BBox53 | 0.633604 | 0.545775 | 0.745502 | 0.833242 | 0.603776 |
| BBox53+BIU56 | 0.653818 | 0.569558 | 0.760001 | 0.858669 | 0.651741 |
| +temporal+ego | 0.704288 | 0.600988 | 0.803998 | 0.847409 | 0.660772 |
| +road | 0.732402 | 0.629678 | 0.824709 | 0.862200 | 0.690414 |
## Validation means

| Input | Accuracy | Macro-F1 | Positive F1 | AP | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| BBox53 | 0.716660 | 0.594540 | 0.817057 | 0.885678 | 0.710118 |
| BBox53+BIU56 | 0.720078 | 0.619176 | 0.815197 | 0.909650 | 0.748888 |
| +temporal+ego | 0.735201 | 0.626636 | 0.827963 | 0.908419 | 0.749345 |
| +road | 0.742282 | 0.624620 | 0.834776 | 0.904291 | 0.747452 |

## Controlled scope

A53/B109/C135/D195 inputs. Static BIU is exact named56-D core, with no7aux/v1/union. Temporal21 includes3pose-confidence summaries outside BIU56. All usefloat32, training-only median imputation,250trees/depth4/LR.05/subsample.8/colsample.8/L2=1, class weight7697/18598. No search, early stopping, calibration or resampling. The same585 missing-speed training windows are excluded from every arm.

The B->C contrast adds temporal and ego blocks together; it does not isolate their individual contributions. Adjacent paired seed deltas and mean/sampleSD are in metrics/. These are descriptive comparisons, not significance tests. Class imbalance makes positive-F1 alone insufficient.

Exact sources and hashes: metrics/source_hashes.json. Inputs: Exp116 cache/psi_ego and cache/psi_road_ego/{train,validation,test}; reused models/predictions: Exp116 models/psi_{ego,road_ego}_seed*.joblib and predictions/psi_{ego,road_ego}_{validation,test}_seed*.csv. Original Exp113/116 simple results remain on their original larger training population.

Command: `env PYTHONDONTWRITEBYTECODE=1 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/124_psi_matched_lightweight_ablation/run.py`. CPU fitting preserves the historical4-thread recipe; no GPU perception needed.

All six new models frozen before test; all24 saved prediction files checked; reused probabilities replayed and source hashes unchanged. No thesis changes, commits, pushes or further experiments.
