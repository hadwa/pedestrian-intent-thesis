# Why clean-JAAD MSLSTM did not benefit from BIU-v2

Post-hoc descriptive audit, 2026-09-18. No retraining, alternate test checkpoint evaluation, threshold selection, benchmark overwrite or thesis modification. This identifies mechanisms and hypotheses, not an isolated causal explanation.

## Direct evidence: checkpoint selection and population composition

The frozen runner selects the epoch with highest validation F1 at 0.5, earliest tie (`../scripts/run.py`, function `train`). Both BIU seed44 models selected epoch1. These are the specified selection rule's actual results, not a checkpoint-loading failure.

| Seed44 arm | Selected epoch | Selected validation F1 | Selected validation AP | Epoch11 validation F1 | Epoch11 validation AP |
| --- | --- | --- | --- | --- | --- |
| Core56 | 1 | 0.787037 | 0.734209 | 0.764368 | 0.855557 |
| Full63 | 1 | 0.788863 | 0.714464 | 0.753709 | 0.856875 |

Core56 epoch1 gets all170 validation positives correct but misclassifies92/104 negatives. Full63 misclassifies91/104 negatives. Later models rank better on validation but lose under the predeclared F1 selection criterion. Their test performance is unknown: no later checkpoint was evaluated here. Sources: `../logs/C_seed44.history.csv`, `../logs/D_seed44.history.csv`, completion markers and `checkpoint_and_fusion.csv`.

JAAD original train/validation/test positive fractions are 999/1552 (64.37%),170/274 (62.04%),720/1501 (47.97%). PIE original train/validation fractions are679/2366 (28.70%),751/3619 (20.75%). Always predicting CROSS gives validation F1 0.765766 on JAAD, versus0.343707 on PIE. Thus positive-class F1 rewards a high-recall/low-specificity model much more strongly in JAAD validation. This is a concrete selection-pressure difference, not evidence of wrong labels or a reason to alter the official split. Source: `cohort_quality.csv`, computed from Exp089 and Exp050 source rows.

The selected seed44 core/full models subsequently produce604/622 test FP, versus359 for baseline. The problem is not confined to that seed: core/full test F1/AP/ROC are below baseline in seeds42 and43 too (`../metrics/per_seed_results.csv`). Therefore checkpoint selection cannot be claimed as the sole cause. Validation F1 is already below baseline for every BIU seed; this was not exclusively an unexpected test-only reversal.

## Direct evidence: fusion remains close to equal, baseline is not protected

Checkpoint fusion softmax coefficients are measured in `checkpoint_and_fusion.csv`. BIU coefficients range0.24731--0.24920 for core and0.24819--0.24908 for full. BBox changes from approximately0.342--0.345 in three-stream baselines to0.252--0.259 in four-stream BIU models. These are scalar coefficients on LayerNorm-normalized contexts, NOT feature importance or causal contribution percentages. The raw-pose control also has near-equal four-stream fusion.

The model retrains every branch and classifier jointly; it does not freeze or preserve baseline predictions. The fusion is global, not sample-specific reliability gating. This makes interference between streams plausible, but near-equal weights alone do not prove that equal weighting causes failure. Sources: Exp053 `scripts/exp053_core.py`, `MslstmBbox53Model`; Exp120 checkpoints and `DESIGN.md`.

Capped PedCMT differs in both training and integration and preserves a frozen parent with bounded correction. Exp119's clean-JAAD full63 improves mean F1/AP/ROC over its parent. This contradicts a blanket claim that BIU-v2 has no useful JAAD information, but does not isolate the benefit of capping versus other architecture/preprocessing differences. Source: `../../119_pedcmt_v2_capped_completion/FINAL_REPORT.md`.

## Preprocessing and augmentation: plausible contributors, not proven causes

The inherited MSLSTM pipeline standardizes BBox53 but not BIUs. LayerNorm is applied after BIU projection, not independently to input features. The raw input mixes magnitudes: JAAD training torso-center velocity mean has standard deviation0.001303; body-yaw delta2.974; auxiliary torso-length mean32.310. Core stride-length standard deviation has median0.411 but maximum191.818. These finite but heterogeneous scales/tails are documented in `feature_distributions.csv`. Their causal impact has not been isolated in JAAD.

Do not assert standardization is a guaranteed solution: Exp054 already tested train-only BIU standardization on PIE; F1 decreased0.828921->0.817401 while AP increased0.859921->0.862008. Source: `../../054_mslstm_biu_v2_input_standardization/FINAL_REPORT.md`.

JAAD training oversampling duplicates446 negatives, leaving999 examples of each class. Gaussian noise std0.02 is applied to normalized continuous pose/BBox/YOLOP only for these duplicates; BIUs are recomputed from the perturbed measurements. Vehicle state is unchanged. Therefore do not claim training was left unbalanced or positives were oversampled. PIE has the opposite original class majority. Noise effects on derived angles/velocities may differ from effects on raw coordinates; whether this harms learning needs a separate controlled development study, not an inference from these results. Sources: `augmentation_counts.csv`, `../metrics/augmentation_seed*.csv`, Exp050 `scripts/exp050_core.py::augment_measurements` and Exp120 `augment`.

## What the integrity checks do and do not establish

- All JAAD cached train/validation/test joint coordinates are nonzero by the simple any-XY-nonzero criterion; there are no completely zero windows. This is coverage, NOT proof of correct joint localization or person association (`cohort_quality.csv`).
- All inspected numeric BIU feature arrays are finite (`feature_distributions.csv`).
- Deterministic reconstruction of32 uniformly spaced DEVELOPMENT windows matches the declared63-column names. Maximum absolute difference is0.000644684; all32 pass relative1e-4 plus absolute1e-4 tolerance. They are not byte-identical, and this limited check is not an exhaustive reconstruction audit (`reconstruction_check.json`, `reconstruction_details.csv`).
- Input streams, seed counts and selection policy follow the recorded JAAD adaptation. JAAD uses vehicle state and no acceleration; PIE uses measured ego streams. This is not a dataset-only intervention (`../DESIGN.md`).
- JAAD has1552 original training windows and274 validation windows versus2366/3619 on PIE. This increases concern about selection stability, but window counts are not independent sample counts; JAAD actually has more distinct training videos (98 versus9). Dataset size alone is not a demonstrated explanation (`cohort_quality.csv`).

## Conclusion and next decision

The strongest supported diagnosis is an unfavorable interaction between this MSLSTM integration/selection recipe and clean-JAAD, particularly validation-F1 selection of near-always-positive epoch1 models in seed44. A useful representation is not guaranteed to improve a jointly retrained model. Remaining losses in seeds42/43 prevent reducing the whole outcome to that one selection event. Neither generally poor JAAD poses, absent BIU information, overfitting as the sole cause, nor a magic threshold/scaler fix has been established.

If a follow-up is approved, first define a validation-only checkpoint-selection diagnostic shared by all arms, without changing representations/fusion/augmentation simultaneously. Epoch histories already justify investigating that mismatch. Do not evaluate several alternative selections on test and retain the winner; existing test results must remain reported as the original negative comparison. Any further audit of augmentation should be a separate measurement-level diagnostic before another model change. No follow-up is launched here.

## Reproduction

Command: `env PYTHONDONTWRITEBYTECODE=1 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/120_jaad_mslstm_v2_completion/diagnostics_failure/analyze.py`.

The script reads Exp120 histories/checkpoints/saved predictions, Exp089 trainval/test source rows/poses/BIUs, Exp050 development cache and frozen BIU extractor. It writes only this new diagnostic directory. No original results, checkpoints or configuration are changed.
