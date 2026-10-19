from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
O=Path(__file__).resolve().parent
r=pd.read_csv(O/'raw_runs.csv');s=pd.read_csv(O/'stage_summary.csv');h=pd.read_csv(O/'headline.csv').set_index('model');e=pd.read_csv(O/'effective_cost.csv')
assert len(r)==29200 and not r.duplicated(['stage','case','seed','batch','repeat']).any()
assert np.isfinite(r.elapsed_ms).all() and r.elapsed_ms.gt(0).all()
for keys,g in r.groupby(['stage','case','seed','batch']):
 n=200 if keys[0]=='cached_pipeline_predict' and keys[-1]==100 else 1000 if keys[0]=='sequential_from_ready_perception' else 2000
 assert len(g)==n and set(g.repeat)==set(range(n)),keys
 row=s[(s.stage==keys[0])&(s.case==keys[1])&(s.seed==keys[2])&(s.batch==keys[3])].iloc[0]
 np.testing.assert_allclose([row.mean_ms,row.median_ms,row.p95_ms],[g.elapsed_ms.mean(),g.elapsed_ms.median(),g.elapsed_ms.quantile(.95)],rtol=1e-12)
 np.testing.assert_allclose(row.per_window_mean_ms,row.mean_ms/keys[3])
for case in ['core','road']:
 a=r[(r.case==case)&(r.stage=='cached_pipeline_predict')&(r.batch==1)]
 b=r[(r.case==case)&(r.stage=='cached_pipeline_predict')&(r.batch==100)]
 c=r[(r.case==case)&(r.stage=='sequential_from_ready_perception')]
 np.testing.assert_allclose(h.loc[case,['cached_single_ms','cached_batch_effective_ms','actual_single_including_bbox_ms','actual_single_p95_ms']].to_numpy(float),[a.elapsed_ms.mean(),b.elapsed_ms.mean()/100,c.elapsed_ms.mean(),c.elapsed_ms.quantile(.95)],rtol=1e-12)
 assert set(a.seed)=={42,43,44} and set(b.seed)=={42,43,44} and set(c.seed)=={42}
for row in e.itertuples():np.testing.assert_allclose(row.total_effective_ms,row.biu_temporal_ego_ms+row.road_summary_ms+row.fusion_ms+row.amortized_predict_ms,rtol=1e-12)
p=pd.read_csv(O/'prediction_parity.csv');assert len(p)==6 and p.max_probability_error.lt(1e-6).all() and p.worker_count_prediction_parity.all()
f=pd.read_csv(O/'feature_parity.csv');assert len(f)==195 and f.max_abs_error.lt(1e-6).all()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
sources=json.loads((O/'sources.json').read_text());models=[p for p in sources if '/models/' in p];assert len(models)==6
for p in models:assert sha(p)==sources[p]['sha256'],p
for p in sources:
 if Path(p).suffix=='.py':assert sha(p)==sources[p]['sha256'],p
result=dict(status='PASS',raw_measurements=len(r),measurement_groups=len(s),saved_models_unchanged=6,features_checked=195,observation_windows=100,max_probability_error=float(pd.read_csv(O/'prediction_parity.csv').max_probability_error.max()),new_fits=0,road_geometry_extraction_included=False)
(O/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
(O/'output_hashes.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.suffix in ['.py','.csv','.npz','.md']},indent=2))
