from pathlib import Path
import json
import numpy as np
import pandas as pd
O=Path(__file__).resolve().parent
r=pd.read_csv(O/'raw_runs.csv');assert len(r)==29200,len(r)
summary=[]
for keys,g in r.groupby(['stage','case','seed','batch'],sort=False):
 x=g.elapsed_ms.to_numpy();summary.append(dict(zip(['stage','case','seed','batch'],keys),runs=len(x),mean_ms=float(x.mean()),median_ms=float(np.median(x)),p95_ms=float(np.quantile(x,.95)),std_ms=float(x.std(ddof=1)),per_window_mean_ms=float(x.mean()/keys[-1])))
s=pd.DataFrame(summary);s.to_csv(O/'stage_summary.csv',index=False)
shared=s[s.case=='shared'].set_index('stage').mean_ms
components=['biu56','temporal21','ego5'];effective=[]
for case in ['core','road']:
 fusion=float(s[(s.case==case)&(s.stage=='fusion_name_packing')].mean_ms.iloc[0]);shared_mean=float(shared.loc[components].sum());road=float(shared['road_summary60_from_geometry']) if case=='road' else 0.
 for seed in [42,43,44]:
  batch=float(s[(s.case==case)&(s.seed==seed)&(s.batch==100)].per_window_mean_ms.iloc[0]);effective.append(dict(case=case,seed=seed,biu_temporal_ego_ms=shared_mean,road_summary_ms=road,fusion_ms=fusion,amortized_predict_ms=batch,total_effective_ms=shared_mean+road+fusion+batch))
e=pd.DataFrame(effective);e.to_csv(O/'effective_cost.csv',index=False)
final=[]
for case in ['core','road']:
 row=dict(model=case,dimensions=135 if case=='core' else 195)
 for label,batch in [('cached_single',1),('cached_batch_effective',100)]:
  q=s[(s.case==case)&(s.stage=='cached_pipeline_predict')&(s.batch==batch)];row[label+'_ms']=float(q.per_window_mean_ms.mean())
 row['effective_ready_bbox_ms']=float(e[e.case==case].total_effective_ms.mean())
 q=s[(s.case==case)&(s.stage=='sequential_from_ready_perception')].iloc[0];row['actual_single_including_bbox_ms']=float(q.mean_ms);row['actual_single_p95_ms']=float(q.p95_ms)
 final.append(row)
pd.DataFrame(final).to_csv(O/'headline.csv',index=False)
print(pd.DataFrame(final).to_string(index=False));print('\nComponents:');print(s[s.case=='shared'][['stage','mean_ms','p95_ms']].to_string(index=False))
