import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
src=Path('experiments/matched_biu_cue_rankings_2026_09_12/summary.csv')
rows=list(csv.DictReader(src.open()))
settings=['pie_096','clean_vehicle_100']
labels=['PIE\nBehavioral model','JAAD clean\nVehicle context','PSI\nPerceived intention']
sel=[r for r in rows if r['setting'] in settings and r['version']=='v2_core56' and r['split']=='test' and r['subset']=='ALL' and r['section']=='static_biu']
by={s:{r['cue'].split('__')[1]:r for r in sel if r['setting']==s} for s in settings}
psi=Path('experiments/psi_v2_attribution_2026_10_19')
verification=json.loads((psi/'verification.json').read_text())
assert verification['sources_unchanged'] and verification['biu_features']==56
by['psi_116']={r['cue']:r for r in csv.DictReader((psi/'summary.csv').open()) if r['subset']=='ALL'}
settings.append('psi_116')
cues=['torso_lean_angle','head_yaw_angle','body_yaw_angle','stride_length','lead_foot_elevation','hip_center_x','knee_center_x','ankle_center_x','leg_center_x','lead_ankle_forward','stride_opening','lower_body_motion_magnitude','torso_center_x','shoulder_center_x','torso_motion_magnitude']
names=['Torso lean','Head orientation','Shoulder-line orientation','Stride length','Ankle elevation','Hip-center motion','Knee-center motion','Ankle-center motion','Lower-body center motion','Forward ankle displacement','Relative ankle displacement','Lower-body motion magnitude','Torso-center motion','Shoulder-center motion','Torso motion magnitude']
assert all(set(by[s])==set(cues) for s in settings)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42})
out=Path('thesis/manuscript/Figures')
shares=[sum(float(r['all_input_pct_mean']) for r in by[s].values()) for s in settings]
fig,ax=plt.subplots(figsize=(7.1,2.8));bars=ax.bar(range(3),shares,color=['#255b88','#4b879c','#63966c'],width=.55)
ax.set_xticks(range(3),labels,fontsize=10);ax.set_ylim(0,37);ax.set_ylabel('BIU-v2 share (%)');ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
for b,v in zip(bars,shares):ax.text(b.get_x()+b.get_width()/2,v+1,f'{v:.1f}%',ha='center',fontsize=11)
fig.tight_layout();fig.savefig(out/'ch4_v2_input_share.pdf');plt.close(fig)
mat=np.array([[float(by[s][c]['toward_pct_mean']) for s in settings] for c in cues])
fig,ax=plt.subplots(figsize=(7.2,6));im=ax.imshow(mat,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
ax.set_xticks(range(3),['PIE','JAAD clean\n+ vehicle','PSI'],fontsize=11);ax.xaxis.tick_top();ax.tick_params(length=0,pad=8);ax.set_yticks(range(15),names,fontsize=11)
for i in range(15):
 for j in range(3):ax.text(j,i,f'{mat[i,j]:.1f}',ha='center',va='center',fontsize=10,color='white' if mat[i,j]>68 else 'black')
fig.colorbar(im,ax=ax,shrink=.6,pad=.025,label='Windows pushed toward the label (%)');fig.tight_layout();fig.savefig(out/'ch4_v2_cue_direction.pdf');plt.close(fig)
print('Saved 2 figures from existing v2-only summaries. Shares:',shares)
for s in settings:
 top=sorted(by[s].items(),key=lambda x:float(x[1]['importance_pct_mean']),reverse=True)[:3]
 print(s,[(c,round(float(r['importance_pct_mean']),2),round(float(r['toward_pct_mean']),2)) for c,r in top])

# Complement direction frequencies with within-BIU magnitude shares.
lines = [r"\begin{table}[!htbp]", r"\centering", r"\caption{Within-BIU absolute attribution by cue (\%); three-seed means.}", r"\label{tab:ch4_v2_cue_importance}", r"\begin{tabularx}{\textwidth}{@{}Xrrr@{}}", r"\toprule", r"\textbf{Cue} & \textbf{PIE} & \shortstack{\textbf{JAAD clean}\\\textbf{+ vehicle}} & \textbf{PSI}\\", r"\midrule"]
maxima = {s:max(float(by[s][c]['importance_pct_mean']) for c in cues) for s in settings}
for c,n in zip(cues,names):
 vals=[]
 for s in settings:
  v=float(by[s][c]['importance_pct_mean']); label=f'{v:.1f}'
  vals.append(r"\textbf{"+label+"}" if v==maxima[s] else label)
 lines.append(n+' & '+' & '.join(vals)+r"\\")
lines += [r"\bottomrule",r"\end{tabularx}",r"\end{table}"]
(out.parent/'Tables/ch4_v2_cue_importance.tex').write_text('\n'.join(lines)+'\n')
print('Saved within-BIU attribution table from the same retained summaries.')
