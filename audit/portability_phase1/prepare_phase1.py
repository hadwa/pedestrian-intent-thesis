from pathlib import Path
import json,hashlib,shutil
D=Path('/home/abdelrahman/Hadwa/pedestrian-intent-thesis')
T=D/'tools';T.mkdir(exist_ok=True)
code=r'''"""Portable evidence checks. Standard library only; never imports training code."""
from pathlib import Path
import argparse,csv,json,hashlib,statistics,math,datetime
ROOT=Path(__file__).resolve().parents[1]
SNAPSHOT=ROOT/'snapshot'
SPECS={
 '119':('119_pedcmt_v2_capped_completion',['case','split','arm']),
 '122':('122_jaad_mslstm_no_noise/test_evaluation',['arm']),
 '124':('124_psi_matched_lightweight_ablation',['split','arm']),
}
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def contained(p,root=ROOT):
    p=p.resolve()
    if not p.is_relative_to(root.resolve()):raise ValueError(f'Path outside publication workspace: {p}')
    return p
def verify():
    rows=json.loads((ROOT/'audit/copy_manifest.json').read_text());errors=[]
    for r in rows:
        p=contained(ROOT/r['destination'])
        if not p.is_file() or digest(p)!=r['sha256']:errors.append(r['destination'])
    if errors:raise ValueError(f'Snapshot mismatch: {errors}')
    return {'files_verified':len(rows),'mode':'copied evidence integrity, not model reproduction'}
def readcsv(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def summaries(out):
    results=[]
    for exp,(folder,keys) in SPECS.items():
        base=SNAPSHOT/'experiments'/folder/'metrics'
        per=base/'per_seed_results.csv';summary=base/'summary_results.csv'
        rows=readcsv(per);expected=readcsv(summary);rebuilt=[];checks=0;max_error=0
        for target in expected:
            group=[r for r in rows if all(r[k]==target[k] for k in keys)]
            if exp=='122' and any(r['split']!='test' for r in group):raise ValueError('Mixed evaluation splits')
            if len(group)!=3 or {r['seed'] for r in group}!={'42','43','44'}:raise ValueError(f'Invalid seed group: {exp}, {target}')
            result={k:target[k] for k in keys}
            for col,value in target.items():
                if col in keys:continue
                metric,agg=col.rsplit('_',1)
                vals=[float(r[metric]) for r in group]
                if not all(math.isfinite(v) for v in vals):raise ValueError('Nonfinite metric')
                actual=statistics.mean(vals) if agg=='mean' else statistics.stdev(vals) if agg=='std' else None
                if actual is None:raise ValueError(f'Unknown aggregation: {col}')
                error=abs(actual-float(value));max_error=max(max_error,error);checks+=1
                if not math.isclose(actual,float(value),rel_tol=1e-10,abs_tol=1e-10):raise ValueError(f'Exp{exp} mismatch: {col}: {actual} != {value}')
                result[col]=actual
            rebuilt.append(result)
        with (out/f'exp{exp}_summary.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(expected[0]));w.writeheader();w.writerows(rebuilt)
        results.append({'experiment':exp,'rows':len(rebuilt),'numeric_checks':checks,'max_absolute_error':max_error,'per_seed_source':str(per.relative_to(ROOT)),'per_seed_sha256':digest(per),'summary_source':str(summary.relative_to(ROOT)),'summary_sha256':digest(summary)})
    return {'classification':'diagnostic aggregation verification; no predictions recomputed and no models trained','experiments':results}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['verify','summaries']);p.add_argument('--output',help='New directory inside runs/; refuses overwrite')
    args=p.parse_args()
    integrity=verify()
    if args.command=='verify':print(json.dumps(integrity,indent=2));return
    runs=contained(ROOT/'runs');runs.mkdir(exist_ok=True)
    out=contained(Path(args.output) if args.output else runs/datetime.datetime.now(datetime.timezone.utc).strftime('summary_check_%Y%m%dT%H%M%S%fZ'),runs)
    out.mkdir(parents=True,exist_ok=False)
    report=summaries(out);report['integrity']=integrity
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(out),**report},indent=2))
if __name__=='__main__':main()
'''
(T/'evidence.py').write_text(code)
# Audit authenticated input inventories without loading datasets or model checkpoints.
S=Path('/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean')
archive=Path('/home/abdelrahman/Hadwa/BIU-PID-Distillation')
inputs=[]
for folder,file in [('119_pedcmt_v2_capped_completion','all_sources.json'),('122_jaad_mslstm_no_noise/test_evaluation','source_hashes.json'),('124_psi_matched_lightweight_ablation','source_hashes.json')]:
    inventory=D/'snapshot/experiments'/folder/'metrics'/file
    for path,h in json.loads(inventory.read_text()).items():
        p=Path(path)
        if not p.is_absolute():continue
        scope='research' if p.is_relative_to(S) else 'archive' if p.is_relative_to(archive) else 'external'
        rel=p.relative_to(S) if scope=='research' else None
        copied=D/'snapshot'/rel if rel else None
        inputs.append({'experiment':folder,'inventory':str(inventory.relative_to(D)),'original_path':path,'recorded_hash':h,'scope':scope,'exists_locally':p.is_file(),'bytes':p.stat().st_size if p.is_file() else None,'included_in_snapshot':bool(copied and copied.is_file()),'hash_reverified':False})
A=D/'audit/portability_phase1';A.mkdir(exist_ok=False)
(A/'recorded_inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
# Preserve a known archive code dependency without modifying frozen source copies.
p=archive/'src/features/bbox_features.py';q=D/'vendor/research_archive/bbox_features.py';q.parent.mkdir(parents=True,exist_ok=True);b=p.read_bytes();q.write_bytes(b)
(A/'archive_code_provenance.json').write_text(json.dumps({'source':str(p),'destination':str(q.relative_to(D)),'sha256':hashlib.sha256(b).hexdigest(),'note':'Preserved dependency; not wired into historical runners yet. Distribution/license review pending.'},indent=2)+'\n')
summary={'recorded_input_references':len(inputs),'unique_paths':len({x['original_path'] for x in inputs}),'missing_local_references':sum(not x['exists_locally'] for x in inputs),'not_in_snapshot_references':sum(not x['included_in_snapshot'] for x in inputs),'archive_references':sum(x['scope']=='archive' for x in inputs)}
(A/'inventory_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
# Detect editorial changes without changing the scope or original manuscript.
changes=[]
for rel in ['thesis/14_RESULT_SOURCE_MAP.md','thesis/manuscript/Backmatter/ExperimentMap.tex']:
    old=(D/'snapshot'/rel).read_bytes();new=(S/rel).read_bytes();changes.append({'path':rel,'snapshot_sha256':hashlib.sha256(old).hexdigest(),'current_sha256':hashlib.sha256(new).hexdigest(),'identical':old==new})
(A/'mapping_drift.json').write_text(json.dumps(changes,indent=2)+'\n')
(A/'prepare_phase1.py').write_bytes(Path(__file__).read_bytes())
(D/'REPRODUCING.md').write_text('''# Reproduction status and commands

This package is not yet an end-to-end training release. Current commands use only the Python standard library and do not import or execute any historical experiment code.

From this repository:

```sh
python tools/evidence.py verify
python tools/evidence.py summaries
```

`verify` checks all copied evidence against the recorded SHA-256 hashes. `summaries` rebuilds Exp119, Exp122 test-evaluation, and Exp124 summary rows from their saved per-seed metric CSVs, using seeds 42/43/44, arithmetic means and sample standard deviations. It compares every rebuilt numeric cell to the preserved reference summary, and writes a new timestamped directory under `runs/`. It retains the original split labels, including diagnostic rows. These checks verify aggregation, not inference or training, and do not validate that per-seed source metrics were themselves correct.

The runner resolves its root from its own location. It refuses outputs outside `runs/` and refuses overwriting an existing run. Never execute scripts under `snapshot/` yet.

## Training dependencies confirmed from source

- Exp119 loads Exp093/090 model code, Exp084/089 parent checkpoints and normalization statistics, Exp050/089 source caches, and Exp093/090 cached parent outputs. These must be regenerated or supplied with checksums before training can be reproduced.
- Exp122 imports Exp121, which imports Exp120, with Exp053/050 preprocessing dependencies. Exp053 imports BBox feature code by an absolute archive path. A byte-exact copy is preserved in `vendor/research_archive/`, but the historical import chain remains unchanged and must not be executed.
- Exp124 imports Exp116's metrics helper, reads its PSI feature caches, clones its fitted parent pipelines, and reuses its C/D model predictions. Training A/B alone is not a reconstruction of all four arms.

`audit/portability_phase1/recorded_inputs.json` inventories recorded source paths and local availability for these three experiments. Presence is not a checksum verification; original recorded hashes are retained separately. This inventory is not a complete raw-data dependency graph.

Remaining work: resolve upstream raw dataset and perception dependencies; package portable feature builders and model implementations; preserve split/seed/selection/threshold contracts; separate training outputs from reference evidence; document exact environments and dataset acquisition; test without original filesystem access; reconcile final thesis mapping and review publication contents. Full training is not authorized by these checks.
''')
print(json.dumps(summary,indent=2))
