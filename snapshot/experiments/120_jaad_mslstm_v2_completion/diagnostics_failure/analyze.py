"""Post-hoc descriptive audit; no model fitting, threshold search or source writes."""
from pathlib import Path
import json
import importlib.util
import sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
torch.set_num_threads(1)
contract = json.loads((ROOT/'metrics/feature_contract.json').read_text())
results = []
for seed in (42,43,44):
    for arm in 'ABCD':
        ck = torch.load(ROOT/f'checkpoints/{arm}_seed{seed}.best.pt', map_location='cpu', weights_only=False)
        h = pd.read_csv(ROOT/f'logs/{arm}_seed{seed}.history.csv')
        selected = h.loc[h.epoch.eq(ck['best_epoch'])].iloc[0]
        best_ap = h.loc[h.pr_auc.idxmax()]
        weights = ck['model_state']['fusion_weights'].softmax(0).numpy()
        row = dict(arm=arm,seed=seed,selected_epoch=ck['best_epoch'],selected_train_loss=selected.train_loss,
                   last_train_loss=h.iloc[-1].train_loss,selected_val_f1=selected.f1,
                   selected_val_ap=selected.pr_auc,selected_val_roc=selected.roc_auc,
                   max_val_ap_epoch=best_ap.epoch,max_val_ap=best_ap.pr_auc,
                   f1_at_max_val_ap=best_ap.f1)
        row.update({f'weight_{k}':float(w) for k,w in zip(contract['streams'][arm],weights)})
        results.append(row)
pd.DataFrame(results).to_csv(OUT/'checkpoint_and_fusion.csv',index=False)

datasets = []
stats = []
for number, folder in [(89,'trainval_source'),(89,'test_source'),(50,'trainval_base')]:
    root = next((REPO/'experiments').glob(f'{number:03d}_*'))/'cache'/folder
    if not (root/'rows.csv').exists():
        continue
    rows = pd.read_csv(root/'rows.csv')
    pose = np.load(root/'pose.npy',mmap_mode='r')
    biu = np.load(root/'biu_v2.npy',mmap_mode='r')
    meta = json.loads((root/'metadata.json').read_text())
    names = meta['biu_v2_feature_names']
    for split, subset in rows.groupby('split'):
        idx = subset.index.to_numpy()
        xy = np.asarray(pose[idx]).reshape(-1,16,17,2)
        nonzero = np.any(xy!=0,axis=-1)
        datasets.append(dict(experiment=number,split=split,n=len(idx),positive=int(subset.label.sum()),
            prevalence=float(subset.label.mean()),videos=subset.video_id.nunique(),
            all_zero_windows=int((~nonzero.any(axis=(1,2))).sum()),mean_nonzero_joint_fraction=float(nonzero.mean()),
            always_cross_f1=float(2*subset.label.mean()/(1+subset.label.mean()))))
        for j,name in enumerate(names):
            v=np.asarray(biu[idx,j],dtype=float)
            stats.append(dict(experiment=number,split=split,feature=name,mean=v.mean(),std=v.std(),
                 median=np.median(v),p95=np.quantile(v,.95),max_abs=np.abs(v).max(),finite=bool(np.isfinite(v).all())))
pd.DataFrame(datasets).to_csv(OUT/'cohort_quality.csv',index=False)
pd.DataFrame(stats).to_csv(OUT/'feature_distributions.csv',index=False)

predstats=[]
for file in sorted((ROOT/'predictions').glob('*.csv')):
    d=pd.read_csv(file)
    for label in (0,1):
        p=d.loc[d.label.eq(label),'probability']
        predstats.append(dict(source=file.name,label=label,n=len(p),mean=p.mean(),median=p.median(),
            predicted_cross=int((p>=.5).sum()),prob_ge_09=int((p>=.9).sum()),prob_le_01=int((p<=.1).sum())))
pd.DataFrame(predstats).to_csv(OUT/'probability_distributions.csv',index=False)
print(pd.DataFrame(results).to_string(index=False))
print(pd.DataFrame(datasets).to_string(index=False))

# Reconstruct features only on a fixed, uniformly spaced development sample.
# No network inference, fitting or checkpoint reselection is performed.
corepath=next((REPO/'experiments').glob('050_*'))/'scripts/exp050_core.py'
spec=importlib.util.spec_from_file_location('failure_audit_core',corepath)
core=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=core
spec.loader.exec_module(core)
root=next((REPO/'experiments').glob('089_*'))/'cache/trainval_source'
rows=pd.read_csv(root/'rows.csv')
pose=np.load(root/'pose.npy')
xy=np.load(root/'bbox.npy')
bbox=np.stack([(xy[:,:,0]+xy[:,:,2])/2,(xy[:,:,1]+xy[:,:,3])/2,xy[:,:,2]-xy[:,:,0],xy[:,:,3]-xy[:,:,1]],axis=2)
saved=np.load(root/'biu_v2.npy')
errors=[]
replay=[]
for i in np.linspace(0,len(rows)-1,32,dtype=int):
    r=rows.iloc[i]
    computed,names,_=core.extract_biu_v2_window(pose[i],bbox[i],width=int(r.frame_width),height=int(r.frame_height))
    assert list(names)==contract['full63']
    errors.append(float(np.max(np.abs(computed-saved[i]))))
    j=int(np.argmax(np.abs(computed-saved[i])))
    replay.append(dict(index=int(i),feature=names[j],cached=float(saved[i,j]),reconstructed=float(computed[j]),
        max_abs_error=errors[-1],allclose_rtol_1e_4_atol_1e_4=bool(np.allclose(computed,saved[i],rtol=1e-4,atol=1e-4))))
(OUT/'reconstruction_check.json').write_text(json.dumps(dict(samples=32,selection='uniform indices in development cache',
    max_absolute_error=max(errors),allclose_at_1e_4=bool(max(errors)<1e-4)),indent=2)+'\n')
pd.DataFrame(replay).to_csv(OUT/'reconstruction_details.csv',index=False)
augrows=[]
for seed in (42,43,44):
    m=pd.read_csv(ROOT/f'metrics/augmentation_seed{seed}.csv')
    for (label,aug),group in m.groupby(['label','augmented']):
        augrows.append(dict(seed=seed,label=int(label),augmented=bool(aug),n=len(group)))
pd.DataFrame(augrows).to_csv(OUT/'augmentation_counts.csv',index=False)
print('reconstruction max error',max(errors))
print(pd.DataFrame(augrows).to_string(index=False))
