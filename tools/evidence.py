"""Portable evidence checks. Standard library only; never imports training code."""
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
