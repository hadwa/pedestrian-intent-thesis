"""Readable tables and printable diagnostic, from saved audit summaries."""
from pathlib import Path
import json, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from analyze import SETTINGS

OUT=Path(__file__).resolve().parent
assert (OUT/'verification.json').exists()
assert not (OUT/'Matched_BIU_Cue_Rankings.pdf').exists()
rank=pd.read_csv(OUT/'rankings.csv');summary=pd.read_csv(OUT/'summary.csv');per=pd.read_csv(OUT/'per_seed.csv');cohorts=pd.read_csv(OUT/'cohorts.csv')
dictionary=pd.read_csv(OUT.parent/'biu_cue_attribution_report_2026_09_12/cue_dictionary.csv')
labels=dict(zip(dictionary.cue_key,dictionary.label))
labels.update(shoulder_width='Shoulder width [scale]',torso_length='Torso length [scale]',direction_sign='Motion direction [metadata]',was_flipped='Mirrored flag [metadata]',direction_confidence='Direction confidence [metadata]')
versions={'v1':'v1 (72)','v2_core56':'v2 core (56)','v2_full63':'v2 full (63)','combined108':'Historical Combined (108)','union135':'Full namespaced union (135)'}

def markdown(frame):
 def fmt(v):
  return f'{v:.3f}' if isinstance(v,(float,np.floating)) else str(v)
 rows=[list(frame.columns)]+[[fmt(v).replace('|','/') for v in row] for row in frame.itertuples(index=False,name=None)]
 return '\n'.join(['| '+' | '.join(rows[0])+' |','| '+' | '.join(['---']*len(rows[0]))+' |']+['| '+' | '.join(row)+' |' for row in rows[1:]])

def label(c):
 prefix,c=c.split('__',1) if c.startswith(('v1__','v2__')) else ('',c)
 return (prefix+' / ' if prefix else '')+labels.get(c,c)

def classrows(setting,version,split,kind):
 return summary[(summary.setting==setting)&(summary.version==version)&(summary.split==split)&(summary.subset==kind)].set_index(['section','cue'])

tables=[]
for (setting,version,section),d in rank.groupby(['setting','version','section'],sort=False):
 val=d[d.split=='val'].set_index('cue');test=d[d.split=='test'].set_index('cue')
 for cue,v in val.iterrows():
  t=test.loc[cue];cp=classrows(setting,version,'test','positive').loc[(section,cue)];cn=classrows(setting,version,'test','negative').loc[(section,cue)]
  vpos=int(v.positive_seeds);tpos=int(t.positive_seeds)
  if abs(v.balanced_net_mean)<1e-10 and abs(t.balanced_net_mean)<1e-10: pattern='Near zero'
  elif v.balanced_net_mean>0 and t.balanced_net_mean>0:pattern='Supportive averages'
  elif v.balanced_net_mean<0 and t.balanced_net_mean<0:pattern='Adverse averages'
  else:pattern='Split sign changes'
  if cp.net_mean_mean*cn.net_mean_mean<0:pattern+='; mixed test classes'
  tables.append(dict(setting=setting,version=version,section=section,cue=cue,label=label(cue),validation_importance_rank=v.importance_rank,validation_support_rank=v.support_rank,val_importance_pct=v.importance_pct_mean,test_importance_pct=t.importance_pct_mean,val_toward_pct=v.toward_pct_mean,val_away_pct=v.away_pct_mean,test_toward_pct=t.toward_pct_mean,test_away_pct=t.away_pct_mean,val_balanced_net=v.balanced_net_mean,test_balanced_net=t.balanced_net_mean,val_positive_seeds=vpos,test_positive_seeds=tpos,test_positive_class_net=cp.net_mean_mean,test_negative_class_net=cn.net_mean_mean,val_track_equal_net=v.track_equal_net_mean,test_track_equal_net=t.track_equal_net_mean,pattern=pattern))
t=pd.DataFrame(tables);t.to_csv(OUT/'review_table.csv',index=False)

intro='''This is a diagnostic explanation of existing saved XGBoost models, not a feature-removal experiment. Every version comparison uses the same windows within its setting. Cross-setting rankings describe different fitted models and are not a controlled dataset or culture comparison.

Importance (%) is the sum of absolute feature SHAP magnitudes for a cue, divided by the total static BIU magnitude. Quality counters and full-v2 metadata are included in this denominator and labeled; extra early/late pose branches are separate. Importance measures reliance, not correctness.

For each window, sum a cue's signed feature SHAP values and multiply by +1 for a positive label or -1 for a negative label. Toward/away percentages count positive/negative results. They are percentages of windows pushed toward/away from the label, NOT percentages of correctly classified windows. A cue may push toward the label while the total model prediction remains wrong.

Balanced net is the average of the positive-class and negative-class mean signed contributions, with equal weight for each class. Positive supports the label on average; negative opposes it. Units are log-odds relative to the fitted model's SHAP baseline. Positive support and opposing magnitude, window-weighted and equal-track net, class-specific percentages and TP/TN/FP/FN results are in the CSV files.

Tables are ordered by VALIDATION importance. Support ranks in review_table.csv use VALIDATION balanced net. Test explains generalization only and must not select cues for deletion. All results average seeds 42/43/44; seed variation is not uncertainty across pedestrians. Overlapping windows and JAAD cohorts are not independent samples.

Correlated inputs, feature grouping, interaction allocation and different model baselines affect SHAP values. A low-importance cue is not necessarily useless, and an adverse cue is not automatically worth removing. These rankings identify candidates for scrutiny; effects of removal require a separate controlled validation experiment. No fitting or removal was done here.'''
scope=[['PIE / 096','v1, core56, Combined108','Temporal + numeric speed'],['Clean JAAD / 098','v1, full63, union135','BBox + BIU'],['Clean JAAD / 100','v1, core56, Combined108','Temporal + vehicle state'],['JAAD_beh / 102','v1, core56, full63, union135','BBox + BIU'],['JAAD_all / 102','v1, core56, full63, union135','BBox + BIU'],['PSI / 113','v1, full63; no matched union','BBox + BIU; native intent task']]
limitations='''Historical Combined108 retains v1 anatomy plus 36 v2 motion features. Union135 keeps v1-72 and full-v2-63 separately, including both angle conventions. They are different representations.

Exp096/100 retain common BBox and ego context, but v2 and Combined also include v2-specific early/late pose features omitted from v1. Their version comparison is a pathway comparison, not a change to static BIUs alone.

PIE and clean JAAD use 16 observations / TTE 30-60. API JAAD_beh/all use their recorded API window construction, including protocol limitations documented in Exp102. PSI uses 15 observations to predict human-rated intent at frame 16. Its labels and task are different; its signed magnitudes are not a future-crossing benchmark comparison.

The latest road-context architectures Exp075, Exp082-D, Exp083, Exp114 and Exp115 do not have complete matched v1/v2/Combined triplets in the inspected artifacts. Their Combined-only importance reports cannot answer a version comparison. Exp113 cannot borrow Exp114 as its union arm, because other inputs differ. No missing model was refitted.

No universal version or cue-removal verdict is selected here. JAAD's final interpretation and calibration remain open.'''

def page_text(pdf,title,body):
 fig=plt.figure(figsize=(8.27,11.69));fig.text(.07,.95,title,fontsize=17,weight='bold',va='top')
 y=.895
 for p in body.split('\n\n'):
  wrapped=textwrap.fill(p,width=92);fig.text(.07,y,wrapped,fontsize=10,va='top',linespacing=1.4);y-=.0205*(wrapped.count('\n')+1)+.027
 fig.text(.07,.035,'Diagnostic | saved models only | 12 September 2026',fontsize=8,color='#555555');pdf.savefig(fig);plt.close(fig)

pdfpath=OUT/'Matched_BIU_Cue_Rankings.pdf'
with PdfPages(pdfpath) as pdf:
 page_text(pdf,'Matched BIU cue rankings: how to read',intro)
 page_text(pdf,'Comparison coverage and limits','\n\n'.join(' | '.join(r) for r in scope)+'\n\n'+limitations)
 for setting in SETTINGS:
  versions_here=cohorts[cohorts.setting==setting].version.drop_duplicates()
  for version in versions_here:
   d=t[(t.setting==setting)&(t.version==version)&(t.section=='static_biu')].sort_values(['validation_importance_rank','cue'])
   fig=plt.figure(figsize=(11.69,8.27));fig.text(.04,.96,SETTINGS[setting][0]+' | '+versions[version],fontsize=14,weight='bold',va='top')
   info=cohorts[(cohorts.setting==setting)&(cohorts.version==version)].set_index('split')
   fig.text(.04,.915,f"Validation: {int(info.loc['val','windows']):,} windows / {int(info.loc['val','tracks'])} tracks. Test: {int(info.loc['test','windows']):,} windows / {int(info.loc['test','tracks'])} tracks. Three-seed means; sorted by validation importance.",fontsize=9)
   cells=[]
   for r in d.itertuples():
    cells.append([r.label,f'{r.val_importance_pct:.1f}',f'{r.test_importance_pct:.1f}',f'{r.val_toward_pct:.1f}',f'{r.test_toward_pct:.1f}',f'{r.test_away_pct:.1f}',f'{r.val_balanced_net:+.3f}',f'{r.test_balanced_net:+.3f}',f'{r.test_positive_class_net:+.3f}',f'{r.test_negative_class_net:+.3f}'])
   ax=fig.add_axes([.04,.16,.92,.72]);ax.axis('off')
   headings=['Cue (representation source)','Imp %\nVal','Imp %\nTest','Toward %\nVal','Toward %\nTest','Away %\nTest','Bal. net\nVal','Bal. net\nTest','Pos. net\nTest','Neg. net\nTest']
   table=ax.table(cellText=cells,colLabels=headings,colWidths=[.33]+[.67/9]*9,cellLoc='center',bbox=[0,0,1,1]);table.auto_set_font_size(False);table.set_fontsize(8)
   for (i,j),cell in table.get_celld().items():
    cell.set_linewidth(.35);cell.set_edgecolor('#cccccc')
    if i==0:cell.set_facecolor('#17334c');cell.set_text_props(color='white',weight='bold')
    else:
     cell.set_facecolor('#f0f4f7' if i%2 else 'white')
     if j==0:cell.set_text_props(ha='left')
     if j>=6 and float(cells[i-1][j])<0:cell.set_facecolor('#ffe4dc')
   note='Imp: within static BIU feature-attribution mass. Toward/away: windows pushed toward/away from the label; not decision accuracy.\nBal. net: equal-class signed log-odds. Pos./Neg.: separate label classes. Orange: negative mean, not a removal recommendation.\nQuality/scale metadata are labeled. v1/v2 prefixes retain distinct measurements; absence means not supplied, not zero importance.'
   fig.text(.04,.115,note,fontsize=8,va='top',linespacing=1.6)
   pdf.savefig(fig);plt.close(fig)

lines=['# Matched cue importance and direction rankings','',intro,'','## Coverage','',markdown(pd.DataFrame(scope,columns=['Setting','Versions','Other inputs'])),'',limitations,'','## Strongest validation support and cues to scrutinize','', 'Order below uses validation equal-class net contribution. Test columns are descriptive follow-up; no feature-selection decisions are made. Full rankings and per-class results are in review_table.csv, rankings.csv and summary.csv.','']
for setting in SETTINGS:
 for version in cohorts[cohorts.setting==setting].version.drop_duplicates():
  d=t[(t.setting==setting)&(t.version==version)&(t.section=='static_biu')].copy()
  d=d[~d.label.str.contains(r'\[quality\]|\[metadata\]',regex=True)]
  chosen=pd.concat([d.nlargest(3,'val_balanced_net'),d.nsmallest(2,'val_balanced_net')]).drop_duplicates('cue')
  lines.extend(['### '+SETTINGS[setting][0]+' / '+versions[version],'',markdown(chosen[['label','val_importance_pct','test_importance_pct','val_balanced_net','test_balanced_net','test_toward_pct','pattern']]),''])
lines.extend(['## Reproducibility','','Commands, run from `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean`:','', '```bash','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/matched_biu_cue_rankings_2026_09_12/analyze.py','MPLCONFIGDIR=/tmp/thesis-input-attribution-mpl /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/matched_biu_cue_rankings_2026_09_12/report.py','/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/matched_biu_cue_rankings_2026_09_12/verify.py','```','','Source paths and SHA-256 hashes: sources.json. Per-feature membership: membership.csv. Model/split/seed probability parity and SHAP additivity: checks.csv and verification.json. Compact cue-level per-window arrays are saved here to avoid repeating SHAP in future audits. No benchmark artifacts or manuscript files were changed. An incomplete first attempt stopped on an incorrect prepared-feature path; its intermediate arrays are isolated in attempt_001_source_path_error and excluded.'])
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('Created',pdfpath,flush=True)
