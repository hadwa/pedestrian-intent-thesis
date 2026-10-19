from pathlib import Path
import json,hashlib,subprocess,collections
D=Path('/home/abdelrahman/Hadwa/pedestrian-intent-thesis');O=D/'audit/saved_evaluation_completion';O.mkdir(exist_ok=False)
reports={'tabular':'runs/tabular_saved_replay/verification.json','mslstm':'runs/mslstm_saved_replay/verification.json','pedcmt':'runs/pedcmt_saved_replay_v2/verification.json','psi':'runs/psi_saved_model_replay_v2/replay.json'}
results={k:json.loads((D/v).read_text()) for k,v in reports.items()}
# Confirm physical checkpoint coverage against the independently audited 117 sources.
used=set()
for rel,keys in [('data/tabular_evaluation_v2/manifest.json',['model']),('data/mslstm_evaluation/manifest.json',['checkpoint']),('data/pedcmt_evaluation/manifest.json',['parent','branch'])]:
 m=json.loads((D/rel).read_text())
 for j in m['jobs']:
  for key in keys:
   if j[key]:used.add(j[key])
psi=json.loads((D/'data/psi_reference/manifest.json').read_text())
used|={f'data/psi_reference/{n}' for n in psi['files'] if n.endswith('.joblib')}
index=collections.defaultdict(set)
for f in ['audit/copy_manifest.json','audit/saved_models/copy_manifest.json','audit/psi_workflow/reference_copy_manifest.json','audit/artifact_package_completion/copy_manifest.json']:
 for r in json.loads((D/f).read_text()):index[r['source']].add(r['destination'])
catalog=json.loads((D/'audit/result_artifact_coverage_final/artifact_catalog.json').read_text());coverage=[]
for r in catalog:
 if 'weights' not in r['roles']:continue
 destinations=sorted(index[r['path']]&used);assert destinations,r['path']
 coverage.append({'source':r['path'],'evaluated_package_paths':destinations,'groups':r['groups']})
assert len(coverage)==117 and len(used)==117
(O/'checkpoint_coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
# Output collision checks. They intentionally fail before overwriting any run.
checks=[];py='/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python'
for script,output in [('evaluate_tabular.py','tabular_saved_replay'),('evaluate_mslstm.py','mslstm_saved_replay'),('evaluate_pedcmt.py','pedcmt_saved_replay_v2')]:
 cmd=[py,'-B',str(D/'tools'/script),'--output',str(D/'runs'/output)]
 r=subprocess.run(cmd,cwd='/tmp',capture_output=True,text=True)
 assert r.returncode!=0 and 'FileExistsError' in r.stderr,(script,r.stdout,r.stderr)
 checks.append({'command':cmd,'cwd':'/tmp','returncode':r.returncode,'expected_failure':'FileExistsError','stderr':r.stderr})
(O/'output_collision_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
summary={'checkpoint_files_audited':117,'checkpoint_files_exercised':len(used),'new_evaluation_jobs':99+60+48,'previous_psi_jobs':24,'models_trained':0,'historical_reference_metrics_matched':True,'original_repositories_blocked_during_replay':True,'neural_validation_device':'cuda','cpu_neural_replay_verified':False,'fresh_environment_install_verified':False,'reports':reports}
(O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(D/'SAVED_MODEL_EVALUATION.md').write_text('''# Evaluate the saved thesis models

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

Prediction-only results (Exp012, earlier PIE classifiers and JAAD-all's reused BBox baseline) remain prediction-only; no original weights were fabricated or refitted. Existing evidence/summary commands and packaged predictions support them. The repository still has no public remote or artifact downloads, and a fresh installation has not been verified.
''')
(D/'README.md').write_text('''# Pedestrian intention thesis companion

Local preparation repository for the thesis code, saved weights and experiment evidence. No remote or public artifact downloads have been configured.

Start with:

- [Saved-model evaluation commands](SAVED_MODEL_EVALUATION.md): tabular, MSLSTM, capped PedCMT and PSI. All 117 audited existing weight files have been exercised by inference checks; no retraining.
- [Result-to-artifact coverage](RESULT_ARTIFACT_COVERAGE.md): thesis entries and exact source files.
- [Data preparation](DATA_PREPARATION.md): input contracts, historical processing and remaining raw-data gaps.
- [Original training recipes](EXPERIMENT_RECIPES.md): documentation for readers, not tasks to execute during packaging.
- [Saved weights](SAVED_WEIGHTS.md): packaged artifacts and prediction-only limitations.
- [Thesis-link handoff](THESIS_LINK_HANDOFF.md): update the thesis session after publication.

The reference `snapshot/` preserves historical evidence. Do not execute its experiment runners directly: some retain original filesystem assumptions. Use the portable commands under `tools/`. Local model/data bundles are under `data/` and are excluded from Git; new evaluation outputs go under `runs/` without overwriting earlier runs.

Release work remaining: complete raw-data acquisition/regeneration instructions or distributable input bundles; verify a fresh environment installation; trim historical staging material; publish reviewed code and downloadable artifacts with checksums; tag a release and update the thesis link. The existing inference checks do not claim raw-video reproduction or fresh training parity.
''')
for name in ['SAVED_WEIGHTS.md','DATA_PREPARATION.md','REPRODUCING.md']:
 p=D/name;s=p.read_text();p.write_text('> Evaluation update: portable tabular, MSLSTM and full capped-PedCMT saved-model replay now passes. See [SAVED_MODEL_EVALUATION.md](SAVED_MODEL_EVALUATION.md). Earlier references below to pending neural/tabular inference are historical; raw-data and publication gaps remain.\n\n'+s)
(O/'REPORT.md').write_text('# Saved-model evaluator completion\n\nAll 117 audited checkpoint files were exercised. No model was trained. See [the full guide](../../SAVED_MODEL_EVALUATION.md) for commands, exact tolerances, original input sources and limitations.\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n\nThe original repositories were read-only during input export and explicitly blocked during replay. Neural checks used CUDA; clean environment installation and CPU neural replay are not verified. Parent-reference PedCMT precision is documented in the guide. Historical thesis tables were not rewritten.\n')
(O/'commands.txt').write_text('''/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -B /tmp/tabular_eval/prepare.py
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -B /tmp/tabular_eval/prepare_v2.py
python /tmp/tabular_eval/validate.py
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -B /tmp/mslstm_eval/prepare.py
python /tmp/mslstm_eval/validate.py
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -B /tmp/pedcmt_eval/prepare.py
python /tmp/pedcmt_eval/validate.py
python /tmp/pedcmt_eval/validate_v2.py
python /tmp/finish_saved_evaluators.py

Exact isolated subprocess command arrays, cwd, stdout and stderr:
../tabular_evaluation_v2/validation_command.json
../mslstm_evaluation/validation_command.json
../pedcmt_evaluation/validation_command_v2.json

Every exported input path/hash is recorded in each route's source_hashes.json and the data bundle manifests. No training command was run. tools/evaluate_tabular.py, tools/evaluate_mslstm.py and tools/evaluate_pedcmt.py have no training entry point.
''')
(O/'finish.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(summary,indent=2))
