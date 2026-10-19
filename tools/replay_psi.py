"""Diagnostic replay of trusted historical checkpoints, never training."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import psi
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    reference=psi.inside(a.reference,psi.ROOT/'data');manifest=json.loads((reference/'manifest.json').read_text())
    for name,h in manifest['files'].items():psi.require(psi.sha(psi.inside(reference/name,reference))==h,f'Reference checksum mismatch: {name}')
    out=psi.fresh(a.output);rows=[];data={s:psi.load(a.bundle,s) for s in psi.SPLITS};psi.disjoint({s:d[0] for s,d in data.items()})
    for arm,dim in psi.DIMS.items():
        for seed in psi.SEEDS:
            m=psi.joblib.load(reference/f'{arm}_seed{seed}.joblib')
            expected=psi.model(seed)
            # Match explicit historical settings, not inherited defaults from a different library.
            for key,value in psi.config()['model'].items():psi.require(m[-1].get_params()[key]==value,f'Model setting differs: {arm}/{seed}/{key}')
            psi.require(m[-1].get_params()['random_state']==seed,'Seed mismatch')
            
            for key,value in expected[0].get_params().items():
                actual=m[0].get_params()[key]
                same=(isinstance(value,float) and np.isnan(value) and isinstance(actual,float) and np.isnan(actual)) or actual==value
                psi.require(same,f'Imputer configuration mismatch: {key}')
            for split in ('validation','test'):
                frame,x,cols=data[split];p=m.predict_proba(pd.DataFrame(x[:,:dim],columns=cols[:dim]))[:,1]
                saved=pd.read_csv(reference/f'{split}_{arm}_seed{seed}.csv')
                psi.require(saved.sequence_id.equals(frame.sequence_id) and saved.label.equals(frame.label),'Reference population mismatch')
                err=float(np.max(np.abs(p-saved.probability.to_numpy())))
                psi.require(np.allclose(p,saved.probability.to_numpy(),rtol=0,atol=1e-7),'Probability replay mismatch')
                metrics=psi.metrics(frame.label,p)
                for metric,value in metrics.items():
                    target=manifest['metrics'][f'{split}/{arm}/{seed}'][metric]
                    psi.require(np.isclose(value,target,rtol=0,atol=1e-10),f'Metric mismatch: {split}/{arm}/{seed}/{metric}')
                rows.append({'arm':arm,'seed':seed,'split':split,'max_probability_error':err,**metrics})
    psi.reports(out,rows)
    report={'classification':'diagnostic saved-model replay, not retraining or raw-video reproduction','rows_checked':len(rows),'max_probability_error':max(r['max_probability_error'] for r in rows),'models_fitted':0,'environment':psi.env(),'config_sha256':psi.sha(psi.CONFIG),'code_sha256':psi.sha(psi.__file__)}
    psi.dump(out/'replay.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
