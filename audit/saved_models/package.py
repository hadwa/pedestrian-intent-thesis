from pathlib import Path
import hashlib,json,csv,shutil,re
S=Path('/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean');D=Path('/home/abdelrahman/Hadwa/pedestrian-intent-thesis')
A=D/'audit/saved_models';A.mkdir(exist_ok=False)
W=D/'data/saved_weights';W.mkdir(parents=True,exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def exp(n):return next((S/'experiments').glob(f'{n:03d}_*'))
manifest=[]
def copy(p,role):
 rel=p.relative_to(S);q=W/rel;q.parent.mkdir(parents=True,exist_ok=True);h=sha(p);shutil.copyfile(p,q)
 assert sha(q)==h
 manifest.append({'source':str(p),'relative_source':str(rel),'destination':str(q.relative_to(D)),'bytes':p.stat().st_size,'sha256':h,'role':role,'copied':True})
for p in exp(119).glob('*/checkpoints/*.best.pt'):copy(p,'Exp119 capped correction branch; requires frozen parent')
for p in exp(122).glob('checkpoints/*_ap.pt'):copy(p,'Exp122 primary validation-AP-selected checkpoint')
for n in (84,89):
 for seed in (42,43,44):copy(exp(n)/f'checkpoints/A_seed{seed}.best.pt','Frozen PedCMT parent; PIE Exp084 or clean-JAAD Exp089')
 copy(exp(n)/'cache/trainval/training_stats.npz','Parent training-only pose/BIU normalization statistics')
for seed in (42,43,44):copy(exp(94)/f'checkpoints/P_seed{seed}.best.pt','Exp094 clean-JAAD raw correction branch; requires Exp089 parent')
for name in ('bbox53_scaler.json','feature_contract.json'):
 copy(exp(122)/'metrics'/name,'Exp122 input preprocessing metadata')
for case in ('pie','jaad'):copy(exp(119)/case/'metrics/feature_contract.json','Exp119 named BIU input schema')
# Inventory candidates separately; availability is not a claim that each model is a main thesis result.
selected=json.loads((D/'audit/selected_trees.json').read_text());inventory=[]
for rel in selected:
 base=S/rel
 if not base.exists():continue
 for p in base.rglob('*'):
  if not p.is_file() or p.is_symlink() or p.suffix not in {'.pt','.pth','.joblib','.pkl'}:continue
  if any(x in {'cache','before','build','official_code'} for x in p.relative_to(base).parts):continue
  hit=next((r for r in manifest if r['source']==str(p)),None)
  inventory.append({'experiment_tree':rel,'path':str(p.relative_to(S)),'bytes':p.stat().st_size,'classification':'diagnostic/checkpoint candidate' if any(x in p.name for x in ['smoke','_f1','debug']) else 'candidate; exact thesis-row mapping must be checked','copied_to_saved_weights':bool(hit),'sha256':hit['sha256'] if hit else ''})
(A/'copy_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with (A/'weight_inventory.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(inventory[0]));w.writeheader();w.writerows(inventory)
summary={'copied_files':len(manifest),'copied_bytes':sum(r['bytes'] for r in manifest),'candidate_files':len(inventory),'candidate_bytes':sum(r['bytes'] for r in inventory),'trained_models':0}
(A/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(A/'package.py').write_bytes(Path(__file__).read_bytes())
# Retain the author's no-training instruction across subsequent sessions.
p=D/'AGENTS.md';s=p.read_text();s+='\n## Current author instruction\n\nDo not retrain any model during packaging. Focus on preprocessing instructions, original training details, saved weights, and evaluation. Training commands may be documented for future readers but must not be executed unless the author explicitly changes this instruction. Preserve original research/thesis files.\n';p.write_text(s)
print(json.dumps(summary,indent=2))
