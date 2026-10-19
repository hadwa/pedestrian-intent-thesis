# Evaluate the saved thesis models

These commands perform inference only. No retraining is required or performed. The inputs, model files and reference predictions are local bundles under `data/`; they are Git-ignored and not yet publicly downloadable.

Run from the companion repository root using the tested environment. Each output directory must be new. No command imports experiment runners or reads the original research/archive repositories.

```sh
# PIE/JAAD XGBoost and the validation-only ordered-history trial
python tools/evaluate_tabular.py --output runs/tabular_evaluation_01

# MSLSTM: PIE G/H/I/J/K/core56 and clean-JAAD AP-selected A/B/C/D
python tools/evaluate_mslstm.py --device cuda --output runs/mslstm_evaluation_01

# Capped PedCMT: complete frozen parent plus correction branch
python tools/evaluate_pedcmt.py --device cuda --output runs/pedcmt_evaluation_01

# PSI XGBoost (previously verified)
python tools/replay_psi.py --bundle data/psi_frozen --reference data/psi_reference --output runs/psi_evaluation_01
```

The tabular command supports `--list` and repeated `--case CASE` arguments to select comparisons. Neural commands support `--case pie` or `--case jaad`. MSLSTM and PedCMT support `--device cpu`, but CPU replay has not been validated; the recorded neural checks used CUDA. The default `--device auto` uses CUDA if available, otherwise CPU, and still requires replay tolerances to pass. Do not interpret untested CPU support as a verified result.

Each command verifies input/model checksums and feature/row alignment before inference, preserves threshold 0.5, compares against saved reference predictions, and writes predictions plus per-seed and mean/sample-SD summaries. MSLSTM loads state dictionaries strictly and uses the original stream names, feature dimensions and non-affine context normalization. PedCMT evaluates both model parts and applies the fixed 0.10 logit-correction cap. Exp118 remains validation-only; no test job exists for it.

## Verified scope

| Route | Existing weight files exercised | Evaluations | Largest probability difference | Metric difference |
|---|---:|---:|---:|---:|
| Remaining tabular comparisons | 51 | 99 | 2.980194091506405e-08 | 0 |
| MSLSTM | 30 | 60 | 1.9656295779935107e-07 | 0 |
| Capped PedCMT | 6 parents + 18 branches | 48 | 1.6270531895679063e-06 | 2.220446049250313e-16 |
| PSI (earlier replay) | 12 | 24 | 2.980194091506405e-08 | Passed original 1e-10 check |

All 117 weight files identified by the coverage audit are exercised across these routes. Evaluations count arm/seed/split jobs, not newly trained models. The 207 new jobs matched all fixed-threshold decisions; metrics matched the corresponding saved references within floating-point tolerance. These checks establish saved-model replay on frozen inputs, not raw-video reconstruction or fresh training reproducibility.

The recorded checks ran from `/tmp` under isolated Python mode, with an audit hook denying access to the original research and archive repositories. Test-set replay here verifies an existing frozen release; it selects no model, threshold or hyperparameter.

## Probability precision and reference boundaries

PedCMT original parent references (Exp084/089) use torch float32 sigmoid; capped branches use the historical float64 clipped sigmoid. Preserving this distinction avoids changing AP tie behavior. Exp119's combined result report may derive its baseline from float64 logits, so the portable parent-reference metrics should not silently replace that historical report. The source report remains the thesis evidence. PIE validation branch references also reuse historically cached parent outputs; full-parent recomputation produces small probability differences without decision changes.

## Inputs and preparation provenance

- Tabular bundle: `data/tabular_evaluation_v2/manifest.json`. Exact original files/hashes and curator assembly code: `audit/tabular_evaluation_v2/source_hashes.json` and `prepare.py`.
- MSLSTM bundle: `data/mslstm_evaluation/manifest.json`. PIE arrays reuse Exp050/053 caches and Exp117's named core56 indices. JAAD arrays reuse Exp089 geometry/pose, the saved Exp122 BBox scaler, and cached Exp049/120 YOLOP outputs. No pose or scene network was rerun. Assembly provenance: `audit/mslstm_evaluation/`.
- PedCMT bundle: `data/pedcmt_evaluation/manifest.json`. Source arrays preserve original PIE BBox conversion, JAAD boxes, training-only pose/BIU statistics, and named BIU56 selection. Provenance: `audit/pedcmt_evaluation/`.

The curator exporters document how existing caches were assembled; they use source-workspace paths and are not public raw-data preparation commands. The portable evaluators only require the companion bundles. Full dataset acquisition and raw-input regeneration remain separate release tasks.

## Environment and source architecture

Verified: Python 3.10.20, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, XGBoost 3.2.0, joblib 1.5.3. Neural replay used torch 2.11.0+cu128 on CUDA. `requirements-psi.txt` records the tabular dependencies; clean installation of a complete neural environment has not been tested.

`tools/mslstm_model.py` is the inference-only architecture adapted from Exp050 `StreamAttention`, Exp053 `MslstmBbox53Model`, Exp095 control streams, and Exp120 JAAD stream definitions. `tools/pedcmt_model.py` is byte-identical to the Exp084 model module. `tools/pedcmt_branch.py` retains Exp090 inference with its filesystem injection removed and the BIU input width parameterized as in Exp119. Strict checkpoint loading and probability replay validate these adaptations; no weights were modified.

## Evidence and limitations

Exact validation commands/stdout/stderr are in each route's audit folder. The consolidated report is `audit/saved_evaluation_completion/REPORT.md`; `checkpoint_coverage.json` maps every evaluated checkpoint to its original source and thesis groups. Output-overwrite rejection was checked for all three new commands.

Historical failed attempts remain preserved: tabular export initially used `validation` instead of Exp102's `val` directory; the first PedCMT replay used float64 sigmoid for parent references. The final bundle/commands above supersede those attempts.

Prediction-only results (Exp012, earlier PIE classifiers and JAAD-all's reused BBox baseline) remain prediction-only; no original weights were fabricated or refitted. Existing evidence/summary commands and packaged predictions support them. The code repository is public; artifact downloads are not yet available, and a fresh installation has not been verified.
