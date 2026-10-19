from pathlib import Path
import hashlib,json,re,subprocess,datetime,ast
S=Path('/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean')
D=Path('/home/abdelrahman/Hadwa/pedestrian-intent-thesis')
if D.exists(): raise SystemExit('Destination already exists; refusing to overwrite')
D.mkdir()
A=D/'audit'; A.mkdir()
snapshot=D/'snapshot'; snapshot.mkdir()
manifest=[]; excluded=[]; findings=[]; selected=set(); scanned=set()
def read(p):
    a=p.stat(); b=p.read_bytes(); c=p.stat()
    if (a.st_size,a.st_mtime_ns)!=(c.st_size,c.st_mtime_ns): raise RuntimeError(f'Source changed during read: {p}')
    return b
mapping=read(S/'thesis/14_RESULT_SOURCE_MAP.md').decode()
# Capture the current numbered/named source lists, not superseded editorial history.
section=mapping.split('## Numbered experiments',1)[1].split('## Reading the sources',1)[0]
for rel in re.findall(r'\.\./(experiments/[^/)]+)',section): selected.add(rel)
selected.add('docs/retained_predictor_complexity_2026_09_16/evidence')
# Broad code inventory is read-only. Export only mapped experiment trees and statically discovered parents.
lookup={p.name[:3]:p for p in (S/'experiments').iterdir() if p.is_dir() and re.match(r'^\d{3}_',p.name)}
while selected-scanned:
    rel=sorted(selected-scanned)[0];scanned.add(rel); base=S/rel
    if not base.exists(): findings.append({'file':rel,'kind':'missing_mapped_source'});continue
    for p in base.rglob('*.py'):
        if any(x in {'before','build','__pycache__'} for x in p.relative_to(base).parts):continue
        t=read(p).decode(errors='replace')
        nums=set(re.findall(r'exp\(\s*(\d+)\s*\)',t))|set(re.findall(r'(?<!\d)(\d{3})_[A-Za-z]',t))
        for n in nums:
            if n.zfill(3) in lookup:selected.add(str(lookup[n.zfill(3)].relative_to(S)))
for rel in sorted(selected):
    base=S/rel
    if not base.exists():continue
    for p in base.rglob('*'):
        if not p.is_file() or p.is_symlink():continue
        parts=p.relative_to(base).parts
        if any(x in {'before','build','cache','checkpoints','models','predictions','logs','__pycache__','.git'} for x in parts):continue
        if p.suffix.lower() not in {'.py','.sh','.md','.json','.yaml','.yml','.csv','.txt','.lock'}:continue
        if p.name.startswith('.') or p.name.endswith('_COMPLETE.json') or p.name in {'FROZEN.json','FAILURE.json','TRAINED.json'}:continue
        if p.stat().st_size>2_000_000:excluded.append(str(p.relative_to(S)));continue
        manifest.append(p)
# Shared source and configuration are a staging superset; trim after dependency review.
for root in ['src','scripts','configs']:
    for p in (S/root).rglob('*'):
        if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.suffix in {'.py','.sh','.yaml','.yml','.json','.txt'} and p.stat().st_size<=2_000_000:manifest.append(p)
manifest += [S/'requirements.txt',S/'thesis/14_RESULT_SOURCE_MAP.md',S/'thesis/manuscript/Backmatter/ExperimentMap.tex',S/'README.md']
records=[]; syntax=[]
for p in sorted(set(manifest)):
    rel=p.relative_to(S); b=read(p); out=snapshot/rel;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(b)
    records.append({'source':str(p),'destination':str(out.relative_to(D)),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)})
    if p.suffix=='.py':
        try: ast.parse(b,filename=str(rel))
        except SyntaxError as e:syntax.append({'file':str(rel),'error':str(e)})
        for i,line in enumerate(b.decode(errors='replace').splitlines(),1):
            if '/home/' in line or 'ARCHIVE' in line: findings.append({'file':str(rel),'line':i,'kind':'local_path_or_archive_reference','text':line[:500]})
def dump(n,v):(A/n).write_text(json.dumps(v,indent=2)+'\n')
dump('copy_manifest.json',records);dump('portability_findings.json',findings);dump('syntax_findings.json',syntax);dump('selected_trees.json',sorted(selected));dump('excluded_large_files.json',excluded)
state={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':str(S),'commit':subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip(),'branch':subprocess.check_output(['git','-C',str(S),'branch','--show-current'],text=True).strip(),'files':len(records),'bytes':sum(r['bytes'] for r in records),'note':'Working-tree export, not a Git commit snapshot. Source may be concurrently edited. Per-file hashes record copied bytes.'}
dump('snapshot.json',state)
(A/'source_git_status.txt').write_text(subprocess.check_output(['git','-C',str(S),'status','--short'],text=True))
(A/'prepare_snapshot.py').write_bytes(Path(__file__).read_bytes())
(A/'commands.txt').write_text('Preparation command:\npython /tmp/prepare_thesis_companion_20261020.py\n\nSource inspection commands before preparation:\npwd\ncat AGENTS.md\nls -ld /home/abdelrahman/Hadwa/pedestrian-intent-thesis\nrg --files scripts | head -25\nls src\nsed -n \'1,110p\' experiments/116_xgboost_v2_core_completion/scripts/run.py\nsed -n \'1,45p\' experiments/122_jaad_mslstm_no_noise/scripts/run.py\n\nGit reads and fresh repository initialization are recorded in prepare_snapshot.py.\n')
(D/'README.md').write_text('''# Pedestrian intention thesis companion — preparation workspace

Status: unpublished staging snapshot. Not yet a runnable or validated reproduction release.

`snapshot/` contains byte-for-byte copies of selected working-tree evidence, statically discovered parent experiment trees, and a shared-code/configuration staging superset. Original experiment numbers are preserved. `audit/copy_manifest.json` records source paths and SHA-256 hashes. The original thesis mapping is provisional while thesis editing continues.

Do not run historical experiment scripts yet: some reference the original research/archive paths and write into their experiment directories. Complete isolation and output-path refactoring before executing them. Completion markers were intentionally not exported; historical scripts are retained as source evidence, not ready-to-run entry points.

## Release work remaining

- Resolve static and dynamic dependencies; this initial scan is not a complete dependency analysis.
- Replace machine-specific paths in a separate runnable implementation, retaining snapshot provenance.
- Trace datasets, pose/road extraction, frozen splits, feature caches, and parent checkpoints back to regenerable inputs.
- Audit exact environments per model family; the legacy requirements file is not a validated release environment.
- Add separate preprocessing, training, evaluation, and table-generation commands with fresh output folders.
- Review external code licenses, dataset distribution conditions, and files for publication suitability.
- Check setup and lightweight evaluation in an isolated fresh checkout. Full expensive retraining requires explicit authorization.
- Reconcile final thesis mappings; deliberately refresh only selected files with updated hashes.
- Trim the staging superset, then publish and tag the reviewed release.

No claim of benchmark reproduction is made by this export. No models were trained and no results were recomputed. No remote is configured.
''')
(D/'AGENTS.md').write_text('''# Publication workspace rules

This folder is independent of the research and archive repositories. Treat both as read-only. Never run copied historical scripts until their resolved inputs and outputs have been audited: absolute paths may point back to research files. All execution outputs and environments must remain inside this publication workspace. Do not retrain expensive models or push without explicit authorization. Preserve experiment IDs, splits, thresholds, model-selection policies and copied evidence. Do not present staging checks as benchmark reproduction. Audit files contain local paths and are staging provenance, not automatically public release content.
''')
(D/'.gitignore').write_text('.venv/\n__pycache__/\n*.pyc\nruns/\ndata/\n*.pt\n*.pth\n*.joblib\n*.npy\n*.npz\n')
# Verify copied bytes only; never import experiment modules.
assert all(hashlib.sha256((D/r['destination']).read_bytes()).hexdigest()==r['sha256'] for r in records)
subprocess.run(['git','init','-b','main',str(D)],check=True,stdout=subprocess.PIPE)
print(json.dumps({**state,'selected_trees':len(selected),'portability_findings':len(findings),'syntax_findings':len(syntax),'destination':str(D)},indent=2))
