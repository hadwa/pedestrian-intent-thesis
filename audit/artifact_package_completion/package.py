from pathlib import Path
import csv,json,hashlib,collections,shutil
D=Path('/home/abdelrahman/Hadwa/pedestrian-intent-thesis');S=Path('/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean');ARCH=S.parent/'BIU-PID-Distillation'
C=D/'audit/result_artifact_coverage_final';O=D/'audit/artifact_package_completion';O.mkdir(exist_ok=False)
records=[];reuse=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(n,x):(O/n).write_text(json.dumps(x,indent=2)+'\n')
def exp(n):return next((S/'experiments').glob(f'{n:03d}_*'))
def copy(p,dest,roles,expected=None):
 p=Path(p);q=D/dest;h=sha(p)
 if expected and h!=expected:raise RuntimeError(f'Source changed since audit: {p}')
 if q.exists():
  if sha(q)!=h:raise RuntimeError(f'Refusing overwrite: {q}')
  reuse.append(str(q));return
 q.parent.mkdir(parents=True,exist_ok=True)
 with p.open('rb') as i,q.open('xb') as o:shutil.copyfileobj(i,o)
 if sha(q)!=h or sha(p)!=h:raise RuntimeError(f'Copy/source changed: {p}')
 records.append({'source':str(p),'destination':str(dest),'bytes':q.stat().st_size,'sha256':h,'roles':roles})
gaps=json.loads((C/'artifact_gaps.json').read_text())
assert len([r for r in gaps if 'weights' in r['roles']])==69
for r in gaps:
 p=Path(r['path']);assert r['exists']
 if 'weights' in r['roles']: dest=Path('data/saved_weights')/p.relative_to(S)
 elif 'predictions' in r['roles']:
  scope,rel=('research',p.relative_to(S)) if p.is_relative_to(S) else ('archive',p.relative_to(ARCH))
  dest=Path('data/reference_predictions')/scope/rel
 else:dest=Path('evidence/release_additions')/p.relative_to(S)
 copy(p,dest,r['roles'],r['sha256'])
# Small schema and normalization dependencies from the provenance audit.
metadata=set()
for r in csv.DictReader((C/'recorded_dependencies.csv').open()):
 p=Path(r['path'])
 if p.name in {'metadata.json','columns.json','training_stats.npz','bbox53_scaler.json','feature_contract.json','feature_arms.json'} or p.name.endswith('_scaler.json'):metadata.add(p)
for n,rel in [(53,'metrics/bbox53_scaler.json'),(96,'metrics/feature_arms.json'),(12,'metrics/pie_biu_v2_feature_set_definitions.csv')]:metadata.add(exp(n)/rel)
# Explicit column lists for every selected Exp116 case, not all experiment arms.
weights=list(csv.DictReader((C/'weights_to_package.csv').open()))
cases={Path(r['source']).stem.rsplit('_seed',1)[0] for r in weights if Path(r['source']).parent.parent==exp(116)}
for case in cases:
 for split in ('train','validation','test'):metadata.add(exp(116)/f'cache/{case}/{split}/columns.json')
missing_metadata=[]
for p in sorted(metadata):
 if not p.is_file():missing_metadata.append(str(p));continue
 copy(p,Path('data/preprocessing')/p.relative_to(S),['preprocessing_metadata'])
dump('copy_manifest.json',records);dump('missing_metadata.json',missing_metadata)
# Merge provenance into a standalone portable manifest. No absolute source path is needed to verify copies.
manifest_files=['audit/copy_manifest.json','audit/saved_models/copy_manifest.json','audit/psi_workflow/reference_copy_manifest.json','audit/artifact_package_completion/copy_manifest.json']
index=collections.defaultdict(list)
for f in manifest_files:
 for r in json.loads((D/f).read_text()):index[r['source']].append(r)
coverage=[]
for r in json.loads((C/'artifact_catalog.json').read_text()):
 matches=[]
 for c in index[r['path']]:
  q=D/c['destination']
  if q.is_file() and sha(q)==r['sha256']:matches.append(c['destination'])
 coverage.append({'source':r['path'],'roles':r['roles'],'groups':r['groups'],'sha256':r['sha256'],'package_locations':sorted(set(matches))})
assert all(r['package_locations'] for r in coverage),'Unresolved directly audited artifact'
dump('coverage_after_packaging.json',coverage)
portable={}
for r in records:portable[r['destination']]=r['sha256']
for r in coverage:
 for q in r['package_locations']:portable[q]=r['sha256']
dump('verification_manifest.json',portable)
counts=collections.Counter(role for r in records for role in r['roles'])
summary={'copied_files':len(records),'copied_bytes':sum(r['bytes'] for r in records),'roles':dict(counts),'audited_paths_with_verified_copy':len(coverage),'audited_weight_paths_with_verified_copy':sum('weights' in r['roles'] for r in coverage),'missing_metadata':missing_metadata,'models_trained':0,'scope':'Direct audit gaps and selected metadata only; full feature caches and portable inference are not completed by copying.'}
dump('summary.json',summary)
(O/'package.py').write_bytes(Path(__file__).read_bytes())
(D/'tools/verify_artifact_package.py').write_text('''"""Check the packaged thesis artifacts without loading models or original files."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'audit/artifact_package_completion/verification_manifest.json').read_text())
for relative,expected in manifest.items():
    p=(ROOT/relative).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Path outside companion repository')
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    if h.hexdigest()!=expected:raise ValueError(f'Checksum mismatch: {relative}')
print(json.dumps({'verified_paths':len(manifest),'models_loaded':0,'models_trained':0}))
''')
report=['# Artifact package completion','','All direct gaps from the result-to-artifact audit now have verified local copies. Original files, benchmark outputs and thesis sources were read only. No models were trained or loaded.','','## Added artifacts','',f"- {counts['weights']} previously identified weights, preserving experiment and seed names.",f"- {counts['predictions']} saved-prediction files for Exp012, earlier PIE classifiers and the Exp031-derived JAAD-all baseline.",f"- {counts['preprocessing_metadata']} schema/normalization files, including selected Exp116 column orders and neural scalers/feature contracts.",'- Current exhibit sources, plotting code, bibliography and confusion-matrix evidence were copied into `evidence/release_additions/`; the original snapshot was not overwritten.','','## Locations','','Weights: `data/saved_weights/experiments/…`','Prediction-only evidence: `data/reference_predictions/{research,archive}/…`','Preprocessing metadata: `data/preprocessing/experiments/…`','Exact mappings: `copy_manifest.json` and `coverage_after_packaging.json`.','','## Verification','','Run `python tools/verify_artifact_package.py` from the companion repository. It checks destination files only and needs neither the original repository nor ML libraries. Hashes validate copied bytes, not model accuracy or completeness of preprocessing.','','## Remaining limitations','','- Full feature arrays, pose/road extraction assets and dataset acquisition paths are not all bundled. The small preprocessing files describe the inputs; they do not replace those inputs.', '- Portable saved-model inference for non-PSI models remains unfinished. Existing historical code must not be run directly.', '- Prediction-only results remain prediction-only. Packaging does not recover nonexistent original checkpoints.', '- Some current TeX exhibit files reference additional assets; this is an evidence package, not a standalone build of the thesis manuscript.', '- No public artifact download, GitHub remote, or tagged release exists. Data/weights remain outside Git history.','','## Summary','','```json',json.dumps(summary,indent=2),'```']
(O/'REPORT.md').write_text('\n'.join(report)+'\n')
for name in ['RESULT_ARTIFACT_COVERAGE.md','SAVED_WEIGHTS.md','README.md']:
 p=D/name;s=p.read_text();p.write_text('> Packaging update: all 69 audited outstanding weights and 60 prediction-only evidence files now have verified local copies. See [artifact package completion](audit/artifact_package_completion/REPORT.md). Earlier missing-copy counts below describe the prior audit snapshot.\n\n'+s)
(O/'commands.txt').write_text('python /tmp/complete_artifact_package.py\npython /home/abdelrahman/Hadwa/pedestrian-intent-thesis/tools/verify_artifact_package.py\n\nExact source paths and hashes: copy_manifest.json\nOutput locations: copy_manifest.json and coverage_after_packaging.json\nPreparation and read-only metadata selection are specified in package.py.\n')
print(json.dumps(summary,indent=2))
