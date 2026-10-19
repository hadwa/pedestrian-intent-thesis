"""Print Exp115 results and extend the attribution report without overwriting it."""
from pathlib import Path
import json,hashlib,re,textwrap
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parents[1];A=OUT/'attribution';REPORT=OUT/'report';REPORT.mkdir(exist_ok=True)
PREV=ROOT/'experiments/exp075_exp082_exp083_exp099_exp114_input_attribution_2026_09_12'
OLD_FOLDERS=[ROOT/'experiments/exp075_exp114_input_attribution_2026_09_12',ROOT/'experiments/exp075_exp083_exp114_input_attribution_2026_09_12',PREV]
sources={}
def track(path):
 sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();return path
def read(path):return pd.read_csv(track(path),dtype={'experiment':str})
assert json.loads(track(OUT/'INDEPENDENT_VERIFICATION.json').read_text())['status']=='PASS'
for folder in OLD_FOLDERS+[A]:
 for p,h in json.loads(track(folder/'sources.json').read_text()).items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
for p,h in json.loads(track(PREV/'combined_sources.json').read_text()).items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
s=pd.concat([read(PREV/'combined_summary.csv'),read(A/'summary.csv')],ignore_index=True)
d=pd.concat([read(PREV/'combined_per_seed.csv'),read(A/'per_seed.csv')],ignore_index=True)
f=pd.concat([read(PREV/'combined_feature_summary.csv'),read(A/'feature_summary.csv')],ignore_index=True)
checks=[]
for folder in OLD_FOLDERS+[A]:
 for path in sorted(folder.glob('groups_*.npz')):
  exp,split,seed=re.fullmatch(r'groups_(.+)_(validation|test)_seed(\d+)',path.stem).groups();seed=int(seed)
  z=np.load(track(path),allow_pickle=False);groups=z['groups'].tolist();phi=z['group_contribution'];y=z['label']
  rows=d[(d.experiment==exp)&(d.split==split)&(d.seed==seed)&(d.label_group=='ALL')].set_index('group').loc[groups]
  magnitudes=np.abs(phi).mean(0)
  np.testing.assert_allclose(magnitudes/magnitudes.sum()*100,rows.net_share_pct,rtol=1e-12,atol=1e-12)
  np.testing.assert_allclose((((2*y-1)[:,None]*phi)>1e-10).mean(0)*100,rows.toward_pct,rtol=1e-12,atol=1e-12)
  np.testing.assert_allclose(1/(1+np.exp(-(phi.sum(1)+z['baseline']))),z['probability'],rtol=1e-5,atol=1e-6)
  if exp.startswith('115_'):
   cohort='jaad_'+exp.removeprefix('115_')
   saved=pd.read_csv(track(OUT/f'predictions/{cohort}_{split}_RV257_seed{seed}.csv'))
   np.testing.assert_array_equal(saved.sequence_id,z['sequence_id']);np.testing.assert_array_equal(saved.label,y)
   np.testing.assert_allclose(saved.probability,z['probability'],rtol=1e-12,atol=1e-15)
  checks.append(dict(experiment=exp,split=split,seed=seed,n=len(y)))
assert len(checks)==54
for row in s.itertuples(index=False):
 a=d[(d.experiment==row.experiment)&(d.split==row.split)&(d.group==row.group)&(d.label_group==row.label_group)];assert len(a)==3
 np.testing.assert_allclose([a.net_share_pct.mean(),a.net_share_pct.std(ddof=1)],[row.net_share_pct_mean,row.net_share_pct_std],rtol=1e-12,atol=1e-12)
s.to_csv(REPORT/'combined_attribution_summary.csv',index=False);d.to_csv(REPORT/'combined_attribution_per_seed.csv',index=False);f.to_csv(REPORT/'combined_feature_summary.csv',index=False)
m=read(OUT/'metrics/summary_results.csv');per=read(OUT/'metrics/per_seed_results.csv');delta=read(OUT/'metrics/paired_deltas.csv');cohorts=['jaad_clean','jaad_beh','jaad_all'];labels=['Clean JAAD','JAAD_beh','JAAD_all']
groups=['BBox','Combined BIU package','Explicit temporal','Road context','Vehicle state'];short=['BBox','Combined BIUs','Temporal','Road context','Vehicle state'];colors=['#356b98','#198078','#987a40','#bf704f','#657b48']
def attr(c,g,split='test',metric='net_share_pct_mean'):
 return s[(s.experiment=='115_'+c.removeprefix('jaad_'))&(s.split==split)&(s.label_group=='ALL')&(s.group==g)].iloc[0][metric]
def formatted(v):return f'{100*v:.2f}'
plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
page=plt.figure(figsize=(11.69,8.27));ax=page.add_axes([.05,.045,.9,.91]);ax.axis('off')
ax.text(0,.99,'Exp115: road context + vehicle state on three JAAD cohorts',fontsize=16,weight='bold',va='top')
ax.text(0,.935,'257 features | Combined BIUs | 16 observations / TTE 30–60 | Threshold 0.5 | Seeds 42, 43, 44',fontsize=10,va='top')
rows=[]
for c,label in zip(cohorts,labels):
 for arm in (['R242','V197','RV257'] if c=='jaad_clean' else ['R242','RV257']):
  r=m[(m.cohort==c)&(m.split=='test')&(m.arm==arm)].iloc[0]
  rows.append([label,{'R242':'Road only','V197':'Vehicle state only','RV257':'Road + vehicle state'}[arm]]+[formatted(r[k+'_mean']) for k in ['accuracy','f1','pr_auc','roc_auc']])
t=ax.table(cellText=rows,colLabels=['Test cohort','Context branches','Accuracy','F1','AP','ROC-AUC'],cellLoc='center',colWidths=[.16,.28,.14,.14,.14,.14],bbox=[0,.57,1,.31]);t.auto_set_font_size(False);t.set_fontsize(10)
for (i,j),cell in t.get_celld().items():
 cell.set_edgecolor('#cccccc')
 if i==0:cell.set_facecolor('#e7edf1');cell.set_text_props(weight='bold')
 elif rows[i-1][1]=='Road + vehicle state':cell.set_facecolor('#eef5ee')
ax.text(0,.54,'Test means in percent. AP = average precision. Seed SD, paired deltas and validation results are in the CSVs.',fontsize=9,va='top')
ax.text(0,.49,'Exp115 input attribution (%)',fontsize=12,weight='bold',va='top')
vals=[[g]+[f'{attr(c,g):.2f} ± {attr(c,g,metric="net_share_pct_std"):.2f}' for c in cohorts] for g in groups]
t2=ax.table(cellText=vals,colLabels=['Branch',*labels],cellLoc='center',colWidths=[.34,.22,.22,.22],bbox=[0,.205,1,.245]);t2.auto_set_font_size(False);t2.set_fontsize(10)
for (i,j),cell in t2.get_celld().items():
 cell.set_edgecolor('#cccccc')
 if i==0:cell.set_facecolor('#e7edf1');cell.set_text_props(weight='bold')
notes='Adding vehicle state to the road model gives small mean F1/AP gains; JAAD_beh ROC-AUC decreases slightly. On clean JAAD, the combined model has higher F1 but lower AP/ROC-AUC than vehicle state only. These are retrospective native-cohort comparisons, not zero-shot or final calibration conclusions.'
ax.text(0,.16,textwrap.fill(notes,145),fontsize=10,va='top',linespacing=1.4)
ax.text(0,.035,'Nine new saved models; 42 prediction groups / 630 metric cells independently verified. Exact sources and commands: FINAL_REPORT.md.',fontsize=9,va='top')
page.savefig(REPORT/'results_preview.png',dpi=140)
fig,axes=plt.subplots(2,2,figsize=(11.69,8.27))
for ax,c,label in zip(axes.flat,cohorts,labels):
 for j,g in enumerate(groups):
  v=attr(c,g);sd=attr(c,g,metric='net_share_pct_std');ax.barh(j,v,xerr=sd,height=.55,color=colors[j],capsize=3)
  ax.text(v+1.4,j,f'{v:.2f}%',va='center',fontsize=9)
 n=int(per[(per.cohort==c)&(per.split=='test')&(per.arm=='RV257')].iloc[0].n)
 ax.set_title(label+f' (n={n:,})',loc='left',fontsize=12,pad=10)
 ax.set_yticks(range(5),short);ax.set_ylim(4.6,-.6);ax.set_xlim(0,63);ax.set_xticks([0,20,40,60]);ax.set_xlabel('Attribution share (%)');ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
ax=axes.flat[3];ax.axis('off')
notes=[('Same five input branches','BBox53 + Combined BIU108 + temporal21 + compensated road60 + vehicle state15. No measured speed.'),('How to read the percentages','Exact feature TreeSHAP in log-odds, summed within branch, then mean absolute net and normalization per seed. Bars show mean ± seed SD.'),('Scope','Relative attribution is not an accuracy share or a causal effect. Temporal/road also use pose. Vehicle states are annotated categories. Cohorts remain distinct; JAAD interpretation/calibration stays open.')]
y=.96
for title,body in notes:
 ax.text(0,y,title,fontsize=11,weight='bold',va='top');y-=.10
 wrapped=textwrap.fill(body,58);ax.text(0,y,wrapped,fontsize=10,va='top',linespacing=1.4);y-=.068*len(wrapped.splitlines())+.09
fig.suptitle('Exp115: both context branches in each JAAD predictor',fontsize=16,y=.97)
fig.tight_layout(rect=(.02,.04,.99,.94),h_pad=2,w_pad=2)
fig.savefig(REPORT/'attribution_preview.png',dpi=140)
with PdfPages(REPORT/'Exp115_Results_and_Attribution.pdf') as pdf:pdf.savefig(page);pdf.savefig(fig)
plt.close(fig);plt.close(page)
lines=['# Exp115: JAAD road context plus vehicle state','', 'Completed diagnostic/retrospective extension, 2026-09-12. Nine new 257-D XGBoost models (three seeds per cohort) combine both context branches. All prior models, predictions, cached upstream features and manuscript edits are preserved.','', '[New two-page printable supplement](report/Exp115_Results_and_Attribution.pdf) | [Full extended attribution PDF](report/Input_Attribution_with_Exp115.pdf)','', '## Fixed-0.5 test performance','', '| Cohort | Context | Accuracy | F1 | AP (PR-AUC) | ROC-AUC |','|---|---|---:|---:|---:|---:|']
for row in rows:lines.append('| '+' | '.join(row)+' |')
lines+=['', 'All displayed metrics are percentages, means across seeds42/43/44; not ensembles. Tables are generated from metrics/summary_results.csv, with full mean/sampleSD there and per-seed data in metrics/per_seed_results.csv. No model or threshold is selected from these test results. FP/FN counts and rates are retained in CSVs, omitted from compact main tables.','', '## Matched test changes','', '| Cohort | Comparison | F1 change (percentage points) | AP change | ROC-AUC change | Favorable F1 / AP / ROC seeds |','|---|---|---:|---:|---:|---|']
for (c,comparison),a in delta[delta.split=='test'].groupby(['cohort','comparison']):
 lines.append('| '+c+' | '+comparison+' | '+' | '.join(f'{100*a[k].mean():+.3f}' for k in ['f1','pr_auc','roc_auc'])+' | '+' / '.join(f'{int((a[k]>0).sum())}/3' for k in ['f1','pr_auc','roc_auc'])+' |')
lines+=['', 'Against road-only R242, RV257 improves mean F1/AP in all three cohorts; JAAD_beh has a small mean ROC-AUC decrease and inconsistent seed changes. Clean JAAD and JAAD_all show F1/AP gains in all three seeds. These small descriptive differences are not statistical significance claims. Against clean V197, adding road increases fixed-threshold F1 in every seed but lowers AP and ROC-AUC in every seed. No universal context winner is established.','', '## Inputs and cohort boundaries','', '- R242: exact saved Exp081-D/Exp082 road parent for clean JAAD, or saved Exp083 model for JAAD_beh/all. RV257 adds only Exp099\'s15 categorical vehicle summaries with the same estimator recipe and cohort. Clean V197 is saved Exp099-B. No vehicle-only reruns were introduced for JAAD_beh/all.','- Portable182 = BBox53 + historical Combined BIU108 + explicit temporal21. Compensated road60 + vehicle state15 gives257. Both branch sets use Combined BIUs; this is not a BIU-version comparison. Two pose-quality counters are included in static BIU108.','- Vehicle state is stopped / moving_slow / moving_fast / decelerating / accelerating. Five observation occupancy fractions, five first-frame one-hot indicators and five last-frame indicators. Raw XML at exactly the recorded observation frames supplies labels; no numeric physical speed or future annotation. Missing annotations would stop the run; all windows had coverage, none were dropped.','- Clean JAAD and API-style JAAD_beh/all remain separate native-training cohorts. API cohorts include observed-crossing/unverified-onset examples and JAAD_all non-behavior negatives according to the parent definitions. This is not PIE-to-JAAD zero-shot transfer. The clean road reference retains its Exp082 validation-selected follow-up status.','- Carry forward Exp083\'s documented two JAAD_all training windows with a134-frame annotation gap. Actual observed frame IDs were used for vehicle summaries; no silent consecutive-frame substitution. Every validation/test window remains raw-frame consecutive.','', '| Cohort | Split | Windows | Positive | Tracks | Videos |','|---|---|---:|---:|---:|---:|']
for path in [OUT/'metrics/development_audit.json',OUT/'metrics/test_audit.json']:
 for a in json.loads(track(path).read_text()):lines.append(f'| {a["cohort"]} | {a["split"]} | {a["n"]} | {a["positive"]} | {a["tracks"]} | {a["videos"]} |')
lines+=['', '## Training and verification','', 'Exact parent XGBoost recipe:250 trees, depth4, LR0.05, subsample0.8, colsample_bytree0.8, L2=1, train negative/positive class weight, median imputation, four CPU threads, seeds42/43/44. No early stopping, validation-selected threshold, calibration, feature/model search or train+validation refit. All nine models were frozen before this run loaded test features; all advanced regardless of validation. Historical test exposure is disclosed, so this is retrospective evidence.','', 'All9 checkpoints were saved/reloaded and checked against their frozen parameters and train medians. Source/feature contracts, cohort IDs and labels, split video/track disjointness, raw-frame gaps and vehicle coverage were audited. Twenty-four parent model/split reuse checks reproduced saved probabilities. Independent verification recomputed42 prediction groups and630 metric cells, with max absolute difference1.11e-16, and verified all paired deltas and seed summaries. INDEPENDENT_VERIFICATION.json records the checks.','', 'Training timing and model bytes: metrics/training_timing.csv. Cached batch prediction timing: metrics/prediction_timing.csv. These single-pass CPU measurements exclude feature construction and upstream pose/road perception; they are not end-to-end latency claims.','', '## Exact input attribution','', '| Branch | Clean JAAD | JAAD_beh | JAAD_all |','|---|---:|---:|---:|']
for g in groups:lines.append('| '+g+' | '+' | '.join(f'{attr(c,g):.2f} ± {attr(c,g,metric="net_share_pct_std"):.2f}' for c in cohorts)+' |')
lines+=['', 'Test-window attribution percentages, mean ± sample seed SD. Exact training-path-statistics feature TreeSHAP uses raw log-odds: sum signed feature contributions within each branch per window; average the absolute net; normalize included branches to100 within each seed; average across seeds. Baseline excluded. This is aggregation of feature Shapley values, not an independent coalition/group-SHAP computation.','', 'Vehicle-state share is smaller with road present than in the clean V197 model; this is conditional attribution under changed feature competition and a refitted model. It does not demonstrate that the vehicle annotation lacks value, nor that road causally replaces it. Correlation, group width and cancellation affect allocation. Temporal and road features also use pose/BBox; static BIU importance is not total pose importance. These are not percentages of accuracy, probability, causal benefit or culture-specific behavior.','', 'All18 new attribution model/split checks passed saved probability and SHAP additivity checks. The report independently reconstructs attribution shares and direction from all54 saved group arrays across prior reports and this extension. report/combined_attribution_summary.csv covers all nine experiment/cohort contexts; validation, label/error strata, direction toward labels and alternative feature-mass shares remain available.','', 'Final JAAD interpretation and any calibration decision remain open. The original four PDF pages are preserved and the Exp115 pages are appended; their earlier experiment percentages remain valid for those earlier models.','', '## Exact commands and source records','', 'Working directory: /home/abdelrahman/Hadwa/BIU-PID-Distillation-clean','', '```bash']
rel=str(OUT.relative_to(ROOT));py='/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python'
for phase in ['prepare','train','evaluate']:lines.append(f'{py} -u {rel}/scripts/run.py {phase} > {rel}/logs/{phase}.log 2>&1')
lines += [f'{py} {rel}/scripts/verify.py > {rel}/logs/verify.log 2>&1',f'{py} -u {rel}/scripts/attribute.py > {rel}/logs/attribution.log 2>&1',f'MPLCONFIGDIR=/tmp/thesis-input-attribution-mpl {py} {rel}/scripts/report.py',f'python3 {rel}/scripts/merge_pdf.py','```','', 'FROZEN.json pins exact development source paths/hashes, training plans and environment versions; test_sources.json pins test inputs. TRAINED.json freezes all9 checkpoint hashes before test loading. EVALUATED.json pins caches, predictions and metric outputs. attribution/sources.json pins explanation inputs. report/sources.json pins reused and new report inputs. models/ contains the new checkpoints; cache/ contains257-D matrices and row manifests; predictions/ contains42 new/reused prediction groups. No old benchmark/model artifacts were modified, no perception rerun, no manuscript changes and no git commit/push.']
(OUT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
(OUT/'README.md').write_text('# Exp115: combined road context and vehicle state\n\nCompleted for clean JAAD, JAAD_beh and JAAD_all: three seeds each, historical Combined BIUs,257 inputs. See [full report](FINAL_REPORT.md), [two-page supplement](report/Exp115_Results_and_Attribution.pdf), [full attribution PDF](report/Input_Attribution_with_Exp115.pdf), [frozen design](DESIGN.md) and [independent verification](INDEPENDENT_VERIFICATION.json).\n\nRetrospective fixed0.5 native-cohort evidence. JAAD interpretation/calibration remains open.\n')
track(PREV/'Input_Attribution_PIE_JAAD_PSI_with_Clean_JAAD.pdf')
for name in ['report.py','attribute.py','verify.py','merge_pdf.py']:track(OUT/'scripts'/name)
for p,h in sources.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
(REPORT/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
(REPORT/'verification.json').write_text(json.dumps(dict(status='PASS',saved_group_arrays_verified=54,new_attribution_groups=18,prior_attribution_groups_reused=36,source_hashes_unchanged=True,checks=checks),indent=2)+'\n')
print('REPORT PASS: 54 persisted attribution arrays and unchanged prior results; two Exp115 pages generated')
