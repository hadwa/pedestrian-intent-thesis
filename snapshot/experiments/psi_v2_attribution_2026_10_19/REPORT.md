# PSI BIU-v2 attribution — 19 October 2026

Diagnostic explanation of retained models, not a new benchmark or feature-selection experiment. No model was trained and no existing experiment artifact was modified.

## Result

The 56 BIU-v2 features receive **31.371821%** of absolute feature-level TreeSHAP attribution (sample SD across seeds: 0.185105 percentage points). The other blocks receive 48.935716% BBox, 5.281139% early/late motion, 13.027457% ego speed and 1.383866% pose quality. Exact values are in `block_per_seed.csv` and `block_summary.csv`.

These are explanations of the retained 135-input PSI model: Exp116 `psi_ego`, reused as Exp124 arm C. All 20,547 test windows (96 tracks; 15,462 positive, 5,085 negative) and seeds 42, 43, 44 are included. PSI observes 15 frames and predicts the human-rated intention label at frame 16; it is not the PIE/JAAD future-crossing task.

## Method and verification

`analyze.py` computes exact native XGBoost TreeSHAP (`pred_contribs=True`, `approx_contribs=False`) with the training-path reference, on raw log-odds. It excludes the intercept from attribution shares. For each seed, BIU share is the sum of absolute contributions over all windows and 56 BIU columns divided by the corresponding sum over all 135 columns; reported percentages average those seed-wise shares.

Cue importance sums feature-wise absolute contributions. Toward-label percentage instead sums signed contributions within each cue, multiplies by (2y−1), and counts values above 1e-10. The two statistics measure different properties. Toward-label percentages are neither accuracy gains nor causal fractions of correct decisions. A cue may support the label while the complete model is wrong. Correlated cues may share attribution. No cues are selected or removed from these test-set explanations.

The same definition, exact TreeSHAP method and XGBoost 3.2.0 implementation are used for the existing PIE/clean-JAAD v2 report (`experiments/matched_biu_cue_rankings_2026_09_12/analyze.py`). This permits descriptive comparison of predictor reliance. Different populations, labels and other input blocks prevent interpreting a higher share as a better representation or a cultural explanation. Existing PIE and clean-JAAD shares are approximately 25.6% and 23.1% respectively.

Saved test probabilities and predictions were reproduced against both Exp116 and Exp124. Maximum probability discrepancy: 2.980117796980153e-08; maximum score-additivity residual: 5.618239811155945e-06. All 17 source hashes remain unchanged. `verify.py` independently passed 780 aggregation checks; see `verification.json`, `independent_verification.json`, `checks.csv` and `source_hashes.json`.

## Cue results

All percentages below are averages over the three seeds. Cue names are implementation identifiers; orientation cues are image-based proxies.

| Cue | Share within BIU (%) | Share of all inputs (%) | Toward-label windows (%) |
|---|---:|---:|---:|
| head_yaw_angle | 38.146 | 11.966 | 65.223 |
| stride_length | 20.501 | 6.432 | 59.850 |
| body_yaw_angle | 6.463 | 2.027 | 50.609 |
| leg_center_x | 6.268 | 1.966 | 45.604 |
| stride_opening | 6.243 | 1.958 | 63.124 |
| ankle_center_x | 5.929 | 1.861 | 55.932 |
| torso_lean_angle | 3.058 | 0.960 | 59.949 |
| knee_center_x | 3.025 | 0.949 | 53.766 |
| lower_body_motion_magnitude | 2.479 | 0.778 | 54.939 |
| hip_center_x | 2.113 | 0.662 | 48.422 |
| lead_foot_elevation | 1.642 | 0.515 | 63.721 |
| lead_ankle_forward | 1.434 | 0.450 | 67.617 |
| torso_motion_magnitude | 1.389 | 0.436 | 48.280 |
| shoulder_center_x | 0.834 | 0.262 | 62.616 |
| torso_center_x | 0.475 | 0.149 | 52.123 |

## Exact provenance

Paths below are relative to `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean`.

- Models: `experiments/116_xgboost_v2_core_completion/models/psi_ego_seed{42,43,44}.joblib`, with adjacent JSON metadata. Pipelines contain the retained imputer and XGBoost predictor. Full configurations are copied into `model_configurations.json`; no source model is rewritten.
- Test features, names and sample IDs: `experiments/116_xgboost_v2_core_completion/cache/psi_ego/test/{features.npy,columns.json,rows.csv}`.
- Saved probabilities: `experiments/116_xgboost_v2_core_completion/predictions/psi_ego_test_seed{42,43,44}.csv` and `experiments/124_psi_matched_lightweight_ablation/predictions/test_C_seed{42,43,44}.csv`.
- Feature definition: `experiments/012_pie_biu_v2_30_60_benchmark/metrics/pie_biu_v2_feature_set_definitions.csv`, `biu_v2_core` row. Columns 53–108 of the model input match these 56 named features exactly. `membership.csv` records all 135 column assignments.
- Input dimensions: BBox 53 + BIU-v2 56 + early/late motion 18 + pose quality 3 + ego speed 5. No seven auxiliary cues or road block in this model.
- Computed outputs reside only in this diagnostic folder. Grouped per-window contributions are retained in `cue_contributions_seed{42,43,44}.npz`; feature, cue and block summaries have per-seed and aggregate CSVs. Class-specific cue summaries are also retained.

## Manuscript integration

Section 4.6 and its two existing figures now show the same 56-feature BIU-v2 definition across PIE, clean JAAD and PSI. The former PSI-v1 figure has been removed from the active manuscript, while its source remains untouched. Chapter 3 forward references and Appendix A source mapping were updated. Section 4.8 was shortened to three paragraphs. Chapter 5 now emphasizes the research gap, contributions and verified results, with three numbered future directions. Introduction scope and conclusion refer to the main limitations instead of duplicating them.

Edited thesis sources: `03_methodology.tex`, `04_results_and_discussion.tex`, `01_introduction.tex`, `05_conclusion.tex`, `Backmatter/ExperimentMap.tex`, `Figures/build_ch4_attribution.py`, the two generated `Figures/ch4_v2_*.pdf` files, manuscript README, and thesis storyline/writing/source-map documents. Earlier unrelated edits were preserved.

The copied `check_consistency.py` and `check_conclusion_numbers.py` record fresh manuscript validation here, leaving the earlier consistency audit intact. Canonical output: `thesis/manuscript/Thesis.pdf`.

## Commands

Run from the active repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/psi_v2_attribution_2026_10_19/analyze.py > /tmp/psi_v2_attribution_2026_10_19.log 2>&1
PYTHONDONTWRITEBYTECODE=1 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/psi_v2_attribution_2026_10_19/verify.py
MPLCONFIGDIR=/tmp/thesis-attribution-matplotlib PYTHONDONTWRITEBYTECODE=1 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python thesis/manuscript/Figures/build_ch4_attribution.py
TEXMFVAR=/tmp/guc-thesis-texmf-var VARTEXFONTS=/tmp/guc-thesis-fonts latexmk -cd -pdf -interaction=nonstopmode -halt-on-error thesis/manuscript/Thesis.tex > /tmp/thesis_psi_v2_conclusion_2026_10_19.log 2>&1
PYTHONDONTWRITEBYTECODE=1 python experiments/psi_v2_attribution_2026_10_19/check_consistency.py
PYTHONDONTWRITEBYTECODE=1 python experiments/psi_v2_attribution_2026_10_19/check_conclusion_numbers.py
```

The diagnostic scripts refuse to overwrite a completed run. No commit or push was made.
