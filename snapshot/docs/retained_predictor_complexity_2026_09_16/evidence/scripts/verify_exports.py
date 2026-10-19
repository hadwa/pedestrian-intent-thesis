"""Read-only round-trip checks on the temporary inference exports."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,torch,joblib,xgboost,gc
O=Path(__file__).resolve().parents[1]
rows=json.loads((O/'counts.json').read_text());results=[]
for r in rows:
 if r['seed']!=42:continue
 if r['kind'].startswith('xgb'):
  p=O/'inference_weights'/f"{r['kind']}_seed42.booster.ubj"
  original=joblib.load(r['source_paths'][0]).named_steps['model'].get_booster()
  export=xgboost.Booster();export.load_model(p)
  assert original.get_dump(dump_format='json',with_stats=True)==export.get_dump(dump_format='json',with_stats=True)
  assert p.stat().st_size==r['booster_only_ubj_bytes']
  results.append({'kind':r['kind'],'check':'identical tree dumps and declared UBJSON size','passed':True})
 else:
  p=O/'inference_weights'/f"{r['kind']}_seed42.state_dict.pt"
  exported=torch.load(p,map_location='cpu',weights_only=True)
  if r['kind'].startswith('ped_complete'):
   expected={}
   for prefix,path in zip(['parent','correction'],r['source_paths']):
    state=torch.load(path,map_location='cpu',weights_only=False)['model_state']
    expected.update({prefix+'.'+k:v for k,v in state.items()})
  else:expected=torch.load(r['source_paths'][0],map_location='cpu',weights_only=False)['model_state']
  assert expected.keys()==exported.keys()
  assert all(torch.equal(v,exported[k]) for k,v in expected.items())
  assert p.stat().st_size==r['inference_state_dict_bytes']
  results.append({'kind':r['kind'],'check':'all state keys/tensors identical and declared export size matches','tensors':len(expected),'passed':True})
  del expected,exported
 gc.collect()
(O/'export_checks.json').write_text(json.dumps(results,indent=2)+'\n')
print('Passed all',len(results),'inference export round-trip checks.')
