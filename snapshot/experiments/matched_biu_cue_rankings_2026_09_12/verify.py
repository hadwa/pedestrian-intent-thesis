"""Independent checks against stored per-window cue arrays and old feature SHAP."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
assert not (OUT/'independent_verification.json').exists()
d=pd.read_csv(OUT/'per_seed.csv');r=pd.read_csv(OUT/'rankings.csv');m=pd.read_csv(OUT/'membership.csv')
checks=[];oldchecks=[]
for (setting,split,version,seed),g in d.groupby(['setting','split','version','seed']):
 with np.load(OUT/f'cue_values_{setting}_{split}_{version}_{seed}.npz',allow_pickle=False) as z:
  y=z['label'];p=z['probability'];signed=z['contribution'];cues=list(z['cue']);ids=z['sequence_id'];tracks=z['track_id']
 assert len(set(ids))==len(ids) and signed.shape==(len(y),len(cues))
 for row in g.itertuples():
  j=cues.index(row.section+'|'+row.cue)
  mask={'ALL':np.ones(len(y),bool),'positive':y==1,'negative':y==0,'TP':(y==1)&(p>=.5),'TN':(y==0)&(p<.5),'FP':(y==0)&(p>=.5),'FN':(y==1)&(p<.5)}[row.subset]
  # Explicit reversal for negative labels, independently from producer formula.
  a=np.where(y[mask]==1,signed[mask,j],-signed[mask,j])
  assert len(a)==row.n
  expected=[a.mean(),100*np.count_nonzero(a>1e-10)/len(a),100*np.count_nonzero(a< -1e-10)/len(a),np.mean([a[tracks[mask]==track].mean() for track in np.unique(tracks[mask])])]
  np.testing.assert_allclose(expected,[row.net_mean,row.toward_pct,row.away_pct,row.track_equal_net],rtol=0,atol=2e-12)
 checks.append(dict(setting=setting,split=split,version=version,seed=int(seed),rows=len(y),cue_subset_checks=len(g)))
 # Existing full per-feature arrays independently verify grouping of union cues.
 if version=='union135':
  pop={'clean_098':'clean','beh_102':'jaad_beh','all_102':'jaad_all'}[setting]
  path=OUT.parent/f'jaad_biu_attribution_2026_09_06/contributions/{pop}_{split}_{seed}.npz'
  with np.load(path,allow_pickle=False) as old:
   np.testing.assert_array_equal(old['sequence_id'],ids)
   contract=m[(m.setting==setting)&(m.version==version)]
   np.testing.assert_array_equal(contract.feature,old['feature_names'])
   for token in cues:
    section,cue=token.split('|',1);indices=contract[(contract.section==section)&(contract.cue==cue)]['index'].to_numpy(int)
    np.testing.assert_allclose(old['shap'][:,indices].sum(1),signed[:,cues.index(token)],rtol=0,atol=1e-12)
   oldchecks.append(dict(setting=setting,split=split,seed=int(seed),cue_columns=len(cues)))

for row in r.itertuples():
 subset=d[(d.setting==row.setting)&(d.split==row.split)&(d.version==row.version)&(d.section==row.section)&(d.cue==row.cue)&d.subset.isin(['positive','negative'])]
 assert len(subset)==6
 balanced=subset.pivot(index='seed',columns='subset',values='net_mean').mean(axis=1)
 np.testing.assert_allclose([balanced.mean(),balanced.std(ddof=1)],[row.balanced_net_mean,row.balanced_net_std],rtol=0,atol=1e-12)
 assert (balanced>0).sum()==row.positive_seeds

# Independent agreement with older corrected standalone net contributions.
old=pd.read_csv(OUT.parent/'biu_cue_reliability_audit_2026_09_08/standalone_corrected/metrics/standalone_attributions.csv')
agreements=[]
for setting,family in [('pie_096','pie_extended'),('clean_098','jaad_clean'),('beh_102','jaad_beh'),('all_102','jaad_all')]:
 new=d[(d.setting==setting)&(d.subset=='ALL')&(d.section=='static_biu')].copy()
 new=new[new.version.isin(['v1','v2_core56','combined108'] if setting=='pie_096' else ['v1','v2_full63'])]
 new['oldversion']=new.version.map({'v1':'v1','v2_core56':'v2','v2_full63':'v2','combined108':'union'})
 prev=old[(old.family==family)&(old.outcome=='ALL')]
 joint=new.merge(prev,left_on=['split','oldversion','seed','cue'],right_on=['split','version','seed','cue'])
 assert len(joint)>0
 np.testing.assert_allclose(joint.net_mean,joint.truth_aligned_logodds,rtol=0,atol=2e-7)
 agreements.append(dict(setting=setting,cue_seed_comparisons=len(joint),max_error=float(np.max(np.abs(joint.net_mean-joint.truth_aligned_logodds)))))

sources=json.loads((OUT/'sources.json').read_text())
for path,item in sources.items():
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 assert h.hexdigest()==item['sha256']
result=dict(cue_arrays_verified=len(checks),cue_subset_rows_verified=sum(c['cue_subset_checks'] for c in checks),union_groups_checked_against_original_feature_arrays=oldchecks,agreement_with_previous_standalone_audit=agreements,balanced_ranking_rows_verified=len(r),sources_unchanged=True)
(OUT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)},indent=2))
