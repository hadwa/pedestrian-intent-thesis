from pathlib import Path
import json,csv,re,hashlib,datetime,collections
S=Path('/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean');D=Path('/home/abdelrahman/Hadwa/pedestrian-intent-thesis');ARCH=S.parent/'BIU-PID-Distillation'
O=D/'audit/result_artifact_coverage';O.mkdir(exist_ok=False)
def exp(n):return str(next((S/'experiments').glob(f'{n:03d}_*')).relative_to(S))
def E(n,p):return exp(n)+'/'+p
cache={}
def sha(p):
 p=Path(p)
 if str(p) not in cache:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1048576),b''):h.update(b)
  cache[str(p)]=h.hexdigest()
 return cache[str(p)]
def dump(name,x):(O/name).write_text(json.dumps(x,indent=2)+'\n')
def csvout(name,rows):
 if not rows:return
 with (O/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# Resolve source-to-package locations from recorded copy provenance, not basename guesses.
packages=collections.defaultdict(set)
for file in ['audit/copy_manifest.json','audit/saved_models/copy_manifest.json','audit/psi_workflow/reference_copy_manifest.json']:
 for r in json.loads((D/file).read_text()):packages[r['source']].add(r['destination'])
for split in ['train','validation','test']:
 for name in ['rows.csv','columns.json','features.npy']:
  packages[str(S/E(116,f'cache/psi_road_ego/{split}/{name}'))].add(f'data/psi_frozen/{split}/{name}')
artifacts={}
def artifact(path,role,group):
 p=Path(path);p=p if p.is_absolute() else S/p;key=str(p)
 if key not in artifacts:
  exists=p.is_file(); destinations=sorted(packages.get(key,[]));match=[]
  if exists:
   for dest in destinations:
    q=D/dest
    if q.is_file() and sha(q)==sha(p):match.append(dest)
  artifacts[key]={'path':key,'exists':exists,'bytes':p.stat().st_size if exists else None,'sha256':sha(p) if exists else '', 'matching_package_copies':match,'roles':[],'groups':[]}
 r=artifacts[key]
 if role not in r['roles']:r['roles'].append(role)
 if group not in r['groups']:r['groups'].append(group)
 return r
# Working scope, and immutable copies of the audited manuscript sources.
scopes=['thesis/14_RESULT_SOURCE_MAP.md','thesis/manuscript/Backmatter/ExperimentMap.tex','thesis/manuscript/04_results_and_discussion.tex']
for rel in scopes:
 p=S/rel;q=O/'scope'/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes());artifact(rel,'audited_scope','scope')
appendix=(S/scopes[1]).read_text();lines=[s for s in appendix.splitlines() if '&' in s and '\\ref{' in s]
assert len(lines)==13,len(lines)
groups=[]
def group(id,title,primary,code,contracts,note,weights=None,predictions=None):
 g={'id':id,'title':title,'primary':primary,'code':code,'contracts':contracts,'note':note,'weights':weights or [],'predictions':predictions or []}
 groups.append(g)
 for role in ('primary','code','contracts','weights','predictions'):
  for p in g[role]:artifact(p,role,id)
def weight(n,pat):return [str(p.relative_to(S)) for p in sorted((S/exp(n)).glob(pat))]
initial=E(12,'metrics/pie_biu_v2_benchmark_results_mean_std.csv')
fusion='experiments/aibthings_pie_val_fixed_0p5_selection/pie_val_fixed_0p5_all_models.csv'
m96=E(96,'metrics/test_summary_results.csv');m116=E(116,'metrics/summary_results.csv');m117=E(117,'metrics/summary_results.csv');m119=E(119,'metrics/summary_results.csv');m122=E(122,'test_evaluation/metrics/summary_results.csv');m124=E(124,'metrics/summary_results.csv')
pie_weights=[E(96,f'models/{a}_seed{s}.joblib') for a in ('base','v1','v2') for s in (42,43,44)]+[E(116,f'models/pie_road_seed{s}.joblib') for s in (42,43,44)]
ms_weights=[E(53,f'checkpoints/{a}_seed{s}.best.pt') for a in 'GHI' for s in (42,43,44)]+[E(95,f'checkpoints/{a}_seed{s}.best.pt') for a in 'JK' for s in (42,43,44)]+[E(117,f'mslstm_pie/checkpoints/I_seed{s}.best.pt') for s in (42,43,44)]
ped_weights=weight(119,'*/checkpoints/*.best.pt')+[E(n,f'checkpoints/A_seed{s}.best.pt') for n in (84,89) for s in (42,43,44)]+[E(94,f'checkpoints/P_seed{s}.best.pt') for s in (42,43,44)]
jaad_weights=[E(98,f'models/A_seed{s}.joblib') for s in (42,43,44)]+[E(100,f'models/v2_seed{s}.joblib') for s in (42,43,44)]+[E(102,f'models/{cohort}_{arm}_{s}.joblib') for cohort,arm in [('jaad_beh','A'),('jaad_beh','C'),('jaad_all','C')] for s in (42,43,44)]+[E(116,f'models/{case}_seed{s}.joblib') for case in ['jaad_clean_simple','jaad_clean_road','jaad_clean_road_vehicle','jaad_beh_road','jaad_beh_road_vehicle','jaad_all_road','jaad_all_road_vehicle'] for s in (42,43,44)]
psi_weights=[E(124,f'models/{a}_seed{s}.joblib') for a in 'AB' for s in (42,43,44)]+[E(116,f'models/{case}_seed{s}.joblib') for case in ['psi_ego','psi_road_ego'] for s in (42,43,44)]
archive_predictions=[str(ARCH/f'results/wacv_pcpa_30_60_overlap06/student_{fs}_supervised_wacv_pcpa_30_60_overlap06_{split}_predictions_{model}.csv') for fs in ['bbox','biu_bbox','kp_bbox','biu_kp_bbox'] for split in ['val','test'] for model in ['logistic_regression','random_forest','xgboost','lightgbm']]
group('01','Classifier/version controls and initial confusion figure',[initial,E(12,'metrics/pie_biu_v2_benchmark_results.csv')],['scripts/experiments/pie_biu_v2_30_60_benchmark.py'],[E(12,'README.md'),E(12,'metrics/pie_biu_v2_feature_set_definitions.csv')],'Exp012 predictions available; runner has no model serialization. No weights found in this experiment. Wider original-model availability needs verification.',predictions=[E(12,f'predictions/{a}__{m}__seed{s}.csv') for a in ['bbox_only','biu_v1_plus_bbox','biu_v2_core_plus_bbox'] for m in ['logreg','rf','xgb'] for s in (42,43,44)])
group('02','Earlier raw-pose and BIU fusion',[fusion],['experiments/aibthings_pie_val_fixed_0p5_selection/run_val_fixed_0p5_selection.py'],['experiments/aibthings_pie_val_fixed_0p5_selection/README.md'],'Individual runs, not three-seed averages. Source probabilities reside in archive. Original checkpoint identity for each row needs verification; a later reproduced RF is not automatically the original model.',predictions=archive_predictions)
group('03','PIE XGBoost versions and architecture',[m96,m116],[E(96,'scripts/run_experiment.py'),E(116,'scripts/run.py')],[E(96,'DESIGN.md'),E(96,'scripts/feature_contract.py'),E(116,'DESIGN.md')],'Use Exp096 base/v1/v2 and Exp116 pie_road only; union is not required for these displayed rows.',pie_weights)
group('04','Capped PedCMT and confusion figure',[m119],[E(119,'scripts/run.py'),E(93,'scripts/run_experiment.py')],[E(119,'DESIGN.md'),E(119,'metrics/all_sources.json')],'Need frozen parent plus correction branch, statistics, and inference cap .10. JAAD raw uses Exp094 P; do not replace it with an uncapped diagnostic.',ped_weights)
group('05','PIE MSLSTM and confusion figure',[m117],[E(117,'scripts/run.py'),E(53,'scripts/run_experiment.py'),E(95,'scripts/run_experiment.py')],[E(53,'DESIGN.md'),E(117,'DESIGN.md')],'G/H/I from Exp053; J raw and K GT Action/Look from Exp095; core56 from Exp117. GT row is an annotation reference; L neural union excluded.',ms_weights)
group('06','Clean-JAAD MSLSTM',[m122,E(120,'FINAL_REPORT.md')],[E(122,'test_evaluation/evaluate.py'),E(122,'scripts/run.py')],[E(122,'DESIGN.md'),E(122,'metrics/bbox53_scaler.json'),E(122,'metrics/feature_contract.json')],'A/B/C/D use AP-selected weights. Exp120/121 are development history, not alternate main test evidence.',[E(122,f'checkpoints/{a}_seed{s}_ap.pt') for a in 'ABCD' for s in (42,43,44)])
group('07','JAAD native XGBoost',[E(98,'metrics/summary_results.csv'),E(100,'metrics/summary_results.csv'),E(102,'metrics/summary_results.csv'),m116],[E(98,'scripts/run_experiment.py'),E(100,'scripts/run_experiment.py'),E(102,'scripts/run_experiment.py'),E(116,'scripts/run.py')],[E(102,'metrics/reuse_audit.json'),E(102,'REUSE_AUDIT.md')],'JAAD_all/A intentionally reuses Exp031 saved probabilities; original checkpoint unavailable per reuse audit. Keep clean/API-beh/API-all cohorts separate.',jaad_weights)
group('08','PSI XGBoost and confusion figure',[m124],[E(124,'run.py'),E(116,'scripts/run.py')],[E(124,'DESIGN.md'),E(124,'metrics/feature_contract.json')],'A/B Exp124; C/D Exp116. Frozen-feature evaluation verified; raw-video pipeline and public inputs not ready.',psi_weights)
group('09','Our rows in published-literature comparisons',[m96,m116,E(102,'metrics/summary_results.csv'),m124],[],['thesis/manuscript/bibliography.bib'],'Our rows reuse groups 03/07/08. Published third-party rows are citation-based, not local reruns; external papers were not verified in this audit.',pie_weights+jaad_weights+psi_weights)
group('10','BIU-v2 attribution and cue table',['experiments/matched_biu_cue_rankings_2026_09_12/summary.csv','experiments/psi_v2_attribution_2026_10_19/summary.csv','experiments/psi_v2_attribution_2026_10_19/block_summary.csv'],['experiments/matched_biu_cue_rankings_2026_09_12/analyze.py','experiments/psi_v2_attribution_2026_10_19/analyze.py','thesis/manuscript/Figures/build_ch4_attribution.py'],['experiments/matched_biu_cue_rankings_2026_09_12/verification.json','experiments/psi_v2_attribution_2026_10_19/verification.json'],'Select PIE_096 and clean_vehicle_100 core56 test ALL rows; PSI 56-feature v2 block. Older v1/union attribution files are not automatically main exhibits.')
group('11','Runtime table and figure',['experiments/v2_xgboost_runtime_2026_09_13/headline.csv','experiments/v2_xgboost_runtime_2026_09_13/stage_summary.csv'],['experiments/v2_xgboost_runtime_2026_09_13/run.py','experiments/v2_xgboost_runtime_2026_09_13/summarize.py'],['experiments/v2_xgboost_runtime_2026_09_13/environment.json','experiments/v2_xgboost_runtime_2026_09_13/sources.json'],'Historical timing evidence; no new timing performed. Feature-to-prediction excludes upstream perception; road begins with ready geometry.',pie_weights)
group('12','Predictor complexity and latency',['docs/retained_predictor_complexity_2026_09_16/evidence/counts.csv','docs/retained_predictor_complexity_2026_09_16/evidence/serialized_sizes.csv','docs/retained_predictor_complexity_2026_09_16/evidence/latency_summary.csv',E(119,'metrics/parameter_counts.csv'),E(119,'metrics/efficiency.csv')],[],['docs/retained_predictor_complexity_2026_09_16/evidence/AUDIT.md',E(119,'metrics/latency_scope.json')],'Inference-only serialized PedCMT sizes are explicitly unmeasured, not missing reported values. PedCMT latency uses validation; earlier audit uses test windows.',pie_weights+ms_weights+ped_weights)
group('13','Ordered-history trial',[E(118,'metrics/summary_results.csv'),E(118,'metrics/decision.json')],[E(118,'run.py')],[E(118,'DESIGN.md')],'Exploratory/negative trial retained in thesis; not the main proposed benchmark.',weight(118,'models/*.joblib'))
# Table-row selectors: exact source rows, not matching by rounded numbers.
selectors=[]
def select(table,display,file,where,model_status):
 with (S/file).open(newline='') as f: rows=list(csv.DictReader(f))
 matches=[(i+2,r) for i,r in enumerate(rows) if all(r.get(k)==v for k,v in where.items())]
 selectors.append({'table':table,'display':display,'source':file,'selector':where,'matches':len(matches),'csv_lines':[n for n,r in matches],'source_rows':[r for n,r in matches],'model_status':model_status})
for model in ['logreg','rf','xgb']:
 for fs in ['bbox_only','biu_v1_plus_bbox','biu_v2_core_plus_bbox']:select('ch4_initial',model+'/'+fs,initial,{'feature_set':fs,'model_name':model,'threshold_rule':'fixed_0_5'},'prediction-only; original weights not located')
for model in ['logistic_regression','random_forest','xgboost','lightgbm']:
 for fs in ['bbox','biu_bbox','kp_bbox','biu_kp_bbox']:select('ch4_pose_fusion',model+'/'+fs,fusion,{'feature_set':fs,'model':model},'archive probabilities; checkpoint identity needs verification')
for arm in ['base','v1','v2']:select('ch4_versions',arm,m96,{'scope':'test','arm':arm},'weights local')
for arm in ['base','v2']:select('ch4_architecture',arm,m96,{'scope':'test','arm':arm},'weights local')
select('ch4_architecture','road',m116,{'split':'test','case':'pie_road'},'weights local')
select('ch4_jaad_native','clean BBox',E(98,'metrics/summary_results.csv'),{'scope':'test','arm':'A'},'weights local')
select('ch4_jaad_native','clean v2 vehicle',E(100,'metrics/summary_results.csv'),{'scope':'test','arm':'v2'},'weights local')
for case in ['jaad_clean_simple','jaad_clean_road','jaad_clean_road_vehicle','jaad_beh_road','jaad_beh_road_vehicle','jaad_all_road','jaad_all_road_vehicle']:select('ch4_jaad_native',case,m116,{'split':'test','case':case},'weights local')
for cohort in ['jaad_beh','jaad_all']:
 for arm in ['A','C']:select('ch4_jaad_native',cohort+'/'+arm,E(102,'metrics/summary_results.csv'),{'scope':'test','cohort':cohort,'arm':arm},'original weights unavailable; Exp031 probabilities' if (cohort,arm)==('jaad_all','A') else 'weights local')
for arm in ['G','J','H','core56','I','K']:select('ch4_neural PIE',arm,m117,{'split':'test','case':'mslstm_pie','arm':arm},'weights local; K is annotation reference')
for arm in 'ABCD':select('ch4_neural JAAD',arm,m122,{'arm':arm},'AP weights packaged')
for case in ['pie','jaad']:
 for arm in ['baseline','raw','core56','full63']:select('ch4_pedcmt_capped',case+'/'+arm,m119,{'split':'test','case':case,'arm':arm},'parent/branch weights packaged')
for arm in 'ABCD':select('ch4_psi',arm,m124,{'split':'test','arm':arm},'weights packaged; replay verified')
dump('table_row_selectors.json',selectors)
# Scan recorded dependencies without importing scripts or reading array/model contents.
depfiles=[E(n,'metrics/all_sources.json') for n in [116,117,119]]+[E(122,'test_evaluation/metrics/source_hashes.json'),E(124,'metrics/source_hashes.json'),'experiments/v2_xgboost_runtime_2026_09_13/sources.json','docs/retained_predictor_complexity_2026_09_16/evidence/sources.json','docs/retained_predictor_complexity_2026_09_16/evidence/latency_sources.json','experiments/matched_biu_cue_rankings_2026_09_12/sources.json']
dependencies=[]
def absolute_paths(obj):
 if isinstance(obj,dict):
  for k,v in obj.items():
   if k.startswith('/'):yield k
   yield from absolute_paths(v)
 elif isinstance(obj,list):
  for x in obj:yield from absolute_paths(x)
 elif isinstance(obj,str) and obj.startswith('/'):yield obj
for file in depfiles:
 p=S/file
 if not p.exists():continue
 for path in sorted(set(absolute_paths(json.loads(p.read_text())))):
  q=Path(path);dest=sorted(packages.get(path,[]))
  dependencies.append({'manifest':file,'path':path,'exists':q.exists(),'packaged_locations':';'.join(dest),'note':'recorded dependency candidate; manifest may include excluded arms; existence only, not hash verification'})
csvout('recorded_dependencies.csv',dependencies)
# Collect chapter source comments and all currently included table/figure paths.
chapter=(S/scopes[2]).read_text();additional=[]
for path in sorted(set(re.findall(r'% SOURCE:\s*(\S+)',chapter))):
 a=artifact(path,'chapter_source_comment','chapter');additional.append({'path':path,'exists':a['exists'],'packaged':bool(a['matching_package_copies'])})
for rel in re.findall(r'\\input\{([^}]+)\}',chapter):
 path='thesis/manuscript/'+rel+'.tex';artifact(path,'current_exhibit','chapter')
 q=O/'scope'/path;q.parent.mkdir(parents=True,exist_ok=True)
 if (S/path).exists():q.write_bytes((S/path).read_bytes())
# Scope drift against earlier snapshot is recorded, never silently refreshed.
drift=[]
for rel in scopes[:2]:
 p=S/rel;q=D/'snapshot'/rel;drift.append({'path':rel,'live_sha256':sha(p),'snapshot_sha256':sha(q),'identical':sha(p)==sha(q)})
dump('scope_drift.json',drift)
dump('coverage_groups.json',groups);dump('artifact_catalog.json',list(artifacts.values()));csvout('chapter_sources.csv',additional)
summary=[]
for g in groups:
 def counts(role):
  records=[artifacts[str(Path(p) if Path(p).is_absolute() else S/p)] for p in g[role]]
  return len(records),sum(r['exists'] for r in records),sum(bool(r['matching_package_copies']) for r in records)
 pr=counts('primary');wt=counts('weights');co=counts('code');ct=counts('contracts');pred=counts('predictions')
 summary.append({'group':g['id'],'result':g['title'],'primary_present':f'{pr[1]}/{pr[0]}','primary_packaged':f'{pr[2]}/{pr[0]}','weights_present':f'{wt[1]}/{wt[0]}' if wt[0] else 'see limitation','weights_packaged':f'{wt[2]}/{wt[0]}' if wt[0] else 'see limitation','code_present':f'{co[1]}/{co[0]}','code_packaged':f'{co[2]}/{co[0]}','contracts_present':f'{ct[1]}/{ct[0]}','contracts_packaged':f'{ct[2]}/{ct[0]}','predictions_present':f'{pred[1]}/{pred[0]}' if pred[0] else 'not exhaustively inventoried','note':g['note']})
csvout('coverage_matrix.csv',summary)
gaps=[r for r in artifacts.values() if not r['exists'] or not r['matching_package_copies']]
dump('artifact_gaps.json',gaps)
stats={'appendix_groups':len(groups),'table_rows_traced':len(selectors),'nonunique_or_missing_selectors':sum(r['matches']!=1 for r in selectors),'primary_files':len({p for g in groups for p in g['primary']}),'missing_primary_files':sum(not (S/p).is_file() for p in {p for g in groups for p in g['primary']}),'audited_artifact_paths':len(artifacts),'missing_paths':sum(not r['exists'] for r in artifacts.values()),'not_packaged_paths':sum(r['exists'] and not r['matching_package_copies'] for r in artifacts.values()),'scope_changed_since_snapshot':any(not d['identical'] for d in drift),'models_trained':0}
dump('summary.json',stats)
report=['# Thesis result-to-artifact coverage audit','','Classification: diagnostic release-readiness audit. No training, inference, threshold selection, or benchmark recomputation. All outputs are in the independent companion repository.','','Scope: all 13 Appendix A mapping entries plus primary classifier/architecture/neural table rows. The current manuscript/source-map files are copied under `scope/` with hashes in `artifact_catalog.json`. Published comparison rows are citation evidence and were not externally verified.','','## Coverage matrix','','“Packaged” means a byte-identical copy was found through the recorded copy manifests. It does not mean runnable or publicly downloadable. Weight lists are explicit required models where resolved, not every checkpoint in an experiment folder.','', '| Group | Primary results present / packaged | Required weights present / packaged | Note |','|---|---|---|---|']
for r in summary:report.append(f"| {r['group']} {r['result']} | {r['primary_present']} / {r['primary_packaged']} | {r['weights_present']} / {r['weights_packaged']} | {r['note']} |")
report += ['', '## Actionable findings','',
'- Exp102 JAAD_all/A is intentionally prediction-only: `metrics/reuse_audit.json` states `checkpoints_available=false`. Preserve the Exp031-derived probabilities and reconstruction recipe; do not promise checkpoint replay or retrain to fill the gap.',
'- Exp012: the inspected runner writes predictions/metrics but has no model export. Original weights were not found in the experiment folder. Its saved predictions are an available evidence route, not proof that an original checkpoint is available.',
'- Earlier PIE pose-fusion rows depend on archived saved probabilities. The exact files are listed and checked in `artifact_catalog.json`. A later reproduced random-forest weight is not evidence that all original classifier weights exist.',
'- Existing PIE MSLSTM and many PIE/JAAD XGBoost weights are local but not yet copied into the companion package. The explicit missing-copy list is `artifact_gaps.json`; no refit is required for available weights.',
'- Neural and tabular model inputs remain partially unpackaged. `recorded_dependencies.csv` lists source-manifest references and availability; some manifests contain excluded arms. This is not a claim that every candidate input belongs in the final release.',
'- Current exhibit sources, plotting scripts and supporting confusion-matrix records are separate from experiment summaries. Missing package copies are visible in `chapter_sources.csv` and `artifact_gaps.json`. Historical table-generation scripts can describe earlier exhibits; do not run them to overwrite current thesis tables.',
'- PSI frozen-feature evaluation is the only independently replay-verified model route in this companion so far. Neural copies are integrity-checked only. All weights/data are still local; no public download or tagged remote exists.',
'- PedCMT inference-only serialized size is explicitly unmeasured in the thesis, so it is not treated as a missing claimed result.',
'', '## Exact traceability and limits','',
'`table_row_selectors.json` stores source CSV line numbers, exact row filters and full matching records. This verifies evidence coverage, not every rounded typeset number. Figure cells, attribution values, runtime numbers and model counts are covered by source files rather than independently recomputed. Files absent from the selected package are not automatically missing from the research workspace.',
'', '## Recommended completion order (no training)','',
'1. Preserve prediction-only evidence for Exp012, earlier PIE classifiers and JAAD_all/A, with explicit weight limitations.',
'2. Copy only resolved missing weights and their preprocessing contracts from the gap manifest; distinguish model parents, branches and selected seeds.',
'3. Package the exact current plotting/table evidence and complete portable saved-model evaluation adapters.',
'4. Resolve and document dataset/feature acquisition and distribution, then publish reviewed artifacts with hashes and update thesis links.',
'', '## Commands','',
'`python /tmp/audit_thesis_coverage.py` generated this audit. The exact script is preserved as `audit.py`; it uses Python stdlib file reads, CSV/JSON parsing, SHA-256, and file copies of the scope into this new audit directory only. No historical runner is imported. Inspection commands are recorded in `inspection_commands.txt`.',
'', '## Summary','', '```json',json.dumps(stats,indent=2),'```']
(O/'REPORT.md').write_text('\n'.join(report)+'\n');(O/'audit.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(stats,indent=2))
print('Missing direct paths:',[r['path'] for r in artifacts.values() if not r['exists']])
print('Selector failures:',[(r['table'],r['display'],r['matches']) for r in selectors if r['matches']!=1])
