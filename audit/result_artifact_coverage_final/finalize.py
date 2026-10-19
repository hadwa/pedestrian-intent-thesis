from pathlib import Path
import json,csv,hashlib,collections
D=Path('/home/abdelrahman/Hadwa/pedestrian-intent-thesis');O=D/'audit/result_artifact_coverage_final';S=Path('/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean')
rows=json.loads((O/'artifact_catalog.json').read_text());summary=json.loads((O/'summary.json').read_text());selectors=json.loads((O/'table_row_selectors.json').read_text())
assert summary['appendix_groups']==13 and summary['missing_primary_files']==0
assert len(selectors)==66 and all(r['matches']==1 for r in selectors)
for r in rows:
 if 'audited_scope' in r['roles']:
  p=Path(r['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],'Scope changed during audit'
weights=[r for r in rows if 'weights' in r['roles'] and not r['matching_package_copies']]
assert len(weights)==69 and all(r['exists'] for r in weights)
with (O/'weights_to_package.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['source','bytes','sha256','groups']);w.writeheader()
 for r in weights:w.writerow({'source':r['path'],'bytes':r['bytes'],'sha256':r['sha256'],'groups':','.join(r['groups'])})
by_exp=collections.Counter(Path(r['path']).relative_to(S).parts[1] for r in weights)
lines=['# Exact remaining packaging work','','This is an audit handoff, not permission to retrain. All 69 weight files below exist; no refit is needed. Their paths, checksums and mapped result groups are in [weights_to_package.csv](weights_to_package.csv).','','| Experiment | Existing weight files not yet packaged |','|---|---|']
for name,count in sorted(by_exp.items()):lines.append(f'| {name} | {count} |')
lines+=['','Other gaps are listed in [artifact_gaps.json](artifact_gaps.json): saved prediction inputs, current exhibit sources, bibliography and plotting code. Checkpoint counts do not include the explicitly unavailable original JAAD-all/A baseline, nor unlocated original Exp012/earlier PIE classifier weights. Preserve their prediction-supported status.','','The 985 recorded dependency references all exist locally, but their source manifests may cover excluded arms. Curate their necessity before copying; do not package every cached file automatically. Presence was checked, not the integrity of every upstream payload.','','The 25 distinct primary result files are present and already byte-identically packaged. The check resolves 66 primary table rows by exact keys; it does not independently verify every printed number or third-party citation.']
(O/'NEXT_ACTIONS.md').write_text('\n'.join(lines)+'\n')
(O/'inspection_commands.txt').write_text('''Read-only inspection was performed from /home/abdelrahman/Hadwa/BIU-PID-Distillation-clean.
Core audit commands (exact):
python /tmp/audit_thesis_coverage.py
python /tmp/audit_thesis_coverage_final.py
python /tmp/finalize_coverage_audit.py

The first command produced audit/result_artifact_coverage (preliminary; superseded). The second produced audit/result_artifact_coverage_final. The third checked unique selectors, unchanged scope hashes, and generated the weights-to-package list.

Primary scope reads:
pwd
cat /home/abdelrahman/Hadwa/pedestrian-intent-thesis/AGENTS.md
sed -n '1,125p' thesis/14_RESULT_SOURCE_MAP.md
cat thesis/manuscript/Backmatter/ExperimentMap.tex

Targeted source checks:
rg -n 'models|joblib|reus|arm ==|arm in|baseline' experiments/102_jaad_api_matched_biu/scripts/run_experiment.py
rg -n 'joblib|dump|save|predictions' scripts/experiments/pie_biu_v2_30_60_benchmark.py
head -4 experiments/116_xgboost_v2_core_completion/metrics/summary_results.csv
rg --files experiments/118_pie_ordered_history_xgb -g '*.py'

Every evidence path actually audited is listed in artifact_catalog.json; exact CSV filters and source line numbers are in table_row_selectors.json. Filesystem scans and SHA-256 operations are fully specified by audit.py. No experiment runner was executed.
''')
(O/'validation.json').write_text(json.dumps({'appendix_groups_checked':13,'unique_table_row_selectors':66,'primary_files_all_present':True,'scope_unchanged_during_audit':True,'weights_to_package_all_exist':True,'weights_to_package':69,'models_trained':0},indent=2)+'\n')
(O/'finalize.py').write_bytes(Path(__file__).read_bytes())
(D/'RESULT_ARTIFACT_COVERAGE.md').write_text('''# Thesis result-to-artifact coverage

The completed audit is [REPORT.md](audit/result_artifact_coverage_final/REPORT.md). Exact remaining packaging actions are in [NEXT_ACTIONS.md](audit/result_artifact_coverage_final/NEXT_ACTIONS.md).

All 13 Appendix A entries have primary evidence. All 25 distinct primary result files are present and byte-identically copied to the companion snapshot. The 66 primary classifier/architecture/neural table rows checked resolve uniquely to source CSV records. This is evidence coverage, not a new numerical benchmark audit.

117 explicit existing weight files support the resolved result groups; 48 are packaged and 69 remain local in the research repository. These counts do not imply original weights exist for prediction-only results. JAAD-all/A explicitly lacks original checkpoints; Exp012 and earlier PIE classifier original-weight availability is separately qualified.

No models were trained, results recomputed, thesis files edited, or files pushed. The thesis mapping matches the earlier snapshot. Portable evaluation and public artifact distribution remain separate release tasks.
''')
print(json.dumps({'validated':True,'remaining_weights_by_experiment':dict(sorted(by_exp.items()))},indent=2))
