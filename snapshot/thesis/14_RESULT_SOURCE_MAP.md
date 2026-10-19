# Thesis result source index

**Editorial update — 2026-10-20:** The main restoration diagnostic (Exp105) and the numerical XGBoost tangent inside the neural discussion are omitted. Experiment artifacts remain unchanged. The cue-importance table accompanies the direction figure. The abstract is unchanged.

**Scope update — 2026-10-19 (author approved):** Omit the 135-D neural full-union
configuration (v1 + v2 + seven supporting cues) from manuscript tables, figures and
prose. Keep the v1, 56-D and 63-D neural comparisons and the distinct 135-input
XGBoost behavioral model. Earlier instructions below to include the neural union are
superseded. Experiment records and model identities remain unchanged.


**Scope update — 2026-10-18 (author approved):** The PSI pose-reference and
coordinate-offset diagnostics (Exp017/Exp110) are excluded from the manuscript.
Their methods subsection, Chapter 4 numerical results, Appendix A row and dependent
claims in the introduction/conclusion were removed. General pose-quality limitations
and reliability-aware future work remain. The experiment artifacts are preserved;
earlier notes below describing these diagnostics as included are superseded.

This index accompanies Appendix A. Numbered experiment folders and named reports below support the descriptive tables and figures in Chapter 4. Published comparison rows are sourced to the papers cited in the thesis, not to a controlled rerun.

GitHub repository: [hadwa/pedestrian-intent-biu](https://github.com/hadwa/pedestrian-intent-biu).

## Latest seminar-aligned results — 2026-09-29

Presentation: `docs/seminar/Seminar_Story_Flow.pdf`, pp. 17–31.

| Current exhibit | Exact source | Boundary |
|---|---|---|
| Capped PedCMT, 56/63/raw/parent | `experiments/119_pedcmt_v2_capped_completion/metrics/summary_results.csv` (`split=test`, cases `pie`/`jaad`) | Replaces main union result exhibits; does not relabel old models |
| Clean-JAAD MSLSTM | `experiments/122_jaad_mslstm_no_noise/test_evaluation/metrics/summary_results.csv` (A/B/C/D) | Approved AP-selected test evaluation; initial negative Exp120 retained in prose |
| Matched PSI architecture | `experiments/124_psi_matched_lightweight_ablation/metrics/summary_results.csv` (`split=test`, A/B/C/D) | 26,295 shared training windows; C/D reused from Exp116 |
| New PedCMT cost | `experiments/119_pedcmt_v2_capped_completion/metrics/parameter_counts.csv`, `metrics/efficiency.csv`, `metrics/latency_scope.json` | Parent included; validation windows; new inference-only sizes not audited |
| Not main test evidence | Exp121, Exp123, Exp125 | Validation-only selection, bounded-MSLSTM and BBox48 diagnostics |

## Numbered experiments

| Identifier | Folder |
|---|---|
| Exp012 | [012_pie_biu_v2_30_60_benchmark](../experiments/012_pie_biu_v2_30_60_benchmark/) |
| Exp053 | [053_mslstm_bbox53_replacement](../experiments/053_mslstm_bbox53_replacement/) |
| Exp092 | [092_pedcmt_capped_biu_correction](../experiments/092_pedcmt_capped_biu_correction/) |
| Exp093 | [093_pedcmt_pie_capped_biu_correction](../experiments/093_pedcmt_pie_capped_biu_correction/) |
| Exp094 | [094_pedcmt_jaad_capped_raw_pose_control](../experiments/094_pedcmt_jaad_capped_raw_pose_control/) |
| Exp095 | [095_mslstm_bbox53_final_controls](../experiments/095_mslstm_bbox53_final_controls/) |
| Exp096 | [096_pie_exp072_matched_biu_ablation](../experiments/096_pie_exp072_matched_biu_ablation/) |
| Exp098 | [098_jaad_matched_biu_xgb](../experiments/098_jaad_matched_biu_xgb/) |
| Exp100 | [100_jaad_exp099_biu_pathways](../experiments/100_jaad_exp099_biu_pathways/) |
| Exp101 | [101_jaad_v2_core_vs_full](../experiments/101_jaad_v2_core_vs_full/) |
| Exp102 | [102_jaad_api_matched_biu](../experiments/102_jaad_api_matched_biu/) |
| Exp113 | [113_psi_native_biu_xgb](../experiments/113_psi_native_biu_xgb/) |
| Exp116 | [116_xgboost_v2_core_completion](../experiments/116_xgboost_v2_core_completion/) |
| Exp117 | [117_neural_v2_core_completion](../experiments/117_neural_v2_core_completion/) |
| Exp119 | [119_pedcmt_v2_capped_completion](../experiments/119_pedcmt_v2_capped_completion/) |
| Exp122 | [122_jaad_mslstm_no_noise](../experiments/122_jaad_mslstm_no_noise/test_evaluation/) |
| Exp124 | [124_psi_matched_lightweight_ablation](../experiments/124_psi_matched_lightweight_ablation/) |
| Exp118 | [118_pie_ordered_history_xgb](../experiments/118_pie_ordered_history_xgb/) |

## Named reports

| Appendix label | Folder |
|---|---|
| PIE classifier report | [aibthings_pie_val_fixed_0p5_selection](../experiments/aibthings_pie_val_fixed_0p5_selection/) |
| matched cue report | [matched_biu_cue_rankings_2026_09_12](../experiments/matched_biu_cue_rankings_2026_09_12/) |
| PSI v2 cue report | [psi_v2_attribution_2026_10_19](../experiments/psi_v2_attribution_2026_10_19/) |
| V2 runtime report | [v2_xgboost_runtime_2026_09_13](../experiments/v2_xgboost_runtime_2026_09_13/) |
| model-size report | [retained-predictor audit](../docs/retained_predictor_complexity_2026_09_16/evidence/AUDIT.md), with Exp119 for current PedCMT counts/timing |

## Reading the sources

- Literature comparison mappings: the PIE table (`tab:ch4_literature_pie`) uses Exp096/116; the combined JAAD table (`tab:ch4_literature_jaad`) uses Exp102/116; the PSI table (`tab:ch4_literature_psi`) uses Exp124, whose extended models are reused from Exp116. These mappings concern only our rows, not the cited published rows. Labels are used here because adding a diagnostic table changes subsequent table numbers.

- Exp096 supplies the retained base/v1/v2 extended PIE controls; Exp116 supplies the actual 56-D completion configurations.
- Exp117 includes the retained MSLSTM references from Exp053/095, including the GT Action/Look annotation row.
- The v2 attribution figures and cue-importance table (`tab:ch4_v2_cue_importance`) use PIE Exp096 and clean JAAD with vehicle state Exp100 (`v2_core56`, test, ALL rows in the matched cue report), plus PSI Exp116 `psi_ego`, reused as Exp124 arm C.
- PSI v2 attribution uses `experiments/psi_v2_attribution_2026_10_19/summary.csv` and `block_summary.csv`: exact 56-feature BIU block, 135 total inputs, all 20,547 test windows, three seeds, verified probability replay and exact TreeSHAP. This replaces the v1 PSI exhibit without relabelling it. The current v2 classification tables use Exp124, with its extended arms reused from Exp116.
- Timing and model-size reports refer to the saved Exp096/116 models; runtime excludes upstream perception and mask-to-geometry extraction.

## Retained predictor complexity audit (2026-09-16)

Section `sec:efficiency_method` and Table `tab:ch4_model_size` use [the preserved audit](../docs/retained_predictor_complexity_2026_09_16/evidence/AUDIT.md), with `counts.csv`, `serialized_sizes.csv` and `latency_summary.csv` in the same folder, for retained Exp096/116 XGBoost and Exp053/095/117 MSLSTM. Exact checkpoints and hashes are in `sources.json` and `latency_sources.json`. The current capped PedCMT rows instead use Exp119's `metrics/parameter_counts.csv`, `metrics/efficiency.csv` and `metrics/latency_scope.json`: the same CPU timing procedure on validation windows, rather than the earlier audit's test windows. New PedCMT inference-only sizes remain unverified. No model was retrained. The 9.64 ms feature-construction-plus-prediction result comes from `experiments/v2_xgboost_runtime_2026_09_13/headline.csv` and has a separate scope. The efficiency narrative now contains only these updated measurements; the old AIBThings timing comparison was removed on 2026-10-18.

## Release status

The current working manuscript includes later, uncommitted revisions, including
Exp119, Exp122 and Exp124 integration and the October chapter reorganization.
The snapshot described below predates those revisions and is not the current PDF.

The reviewed manuscript source is preserved in commit `a606e7f` on branch `exp/050-pie-standardized-behavior`. The commit containing this index update also preserves the remaining curated experiment results and supporting reports, including Exp118, runtime measurements and model-size counts. Use that repository snapshot together with the manuscript commit for traceability; this is a working thesis snapshot, not a declared final submission. Model weights, cached feature/prediction payloads, downloaded reference-paper copies, build intermediates and duplicate manuscript snapshots remain local. Some older report links refer to those local snapshots; the current manuscript is under `thesis/manuscript/`.

## Editorial update — 2026-09-15

Appendix A was reduced to result/table → experiment/report mappings; detailed paths moved here. TrEP was removed from Chapter 4’s PSI table and numerical discussion at the author’s request. No references to it occur in Chapters 1–3, the abstract or Chapter 5. Its unused bibliography database entry is retained; it is not a cited thesis reference. Previous experiment reports and the pending repetition-review recommendations are preserved.

Commands from repository root:
```bash
python3 /tmp/appendix_and_psi_revision.py
TEXMFVAR=/tmp/guc-thesis-texmf-var VARTEXFONTS=/tmp/guc-thesis-fonts latexmk -cd -pdf -interaction=nonstopmode -halt-on-error thesis/manuscript/Thesis.tex > /tmp/thesis-appendix-psi.log 2>&1
```
Output: `thesis/manuscript/Thesis.pdf`.
