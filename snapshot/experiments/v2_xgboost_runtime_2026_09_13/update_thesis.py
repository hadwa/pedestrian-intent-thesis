from pathlib import Path
import shutil,csv,json,re
R=Path.cwd();O=R/'experiments/v2_xgboost_runtime_2026_09_13';M=R/'thesis/manuscript'
files=['thesis/manuscript/03_methodology.tex','thesis/manuscript/04_results_and_discussion.tex','thesis/manuscript/Figures/ch4_runtime.tex','thesis/manuscript/README.md','thesis/README.md','thesis/01_STATE_AND_STORY.md','thesis/02_ROADMAP.md','thesis/03_THESIS_WRITING_GUIDE.md','thesis/06_FIVE_CHAPTER_PLAN.md','thesis/07_CHAPTER_3_SOURCE_MAP.md','thesis/11_CHAPTER_4_SELECTION_PLAN.md','thesis/12_CHAPTER_4_TABLE_FIGURE_OUTLINE.md']
for n in files:
 p=R/n;q=O/'before'/n;assert not q.exists();q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
h={r['model']:r for r in csv.DictReader((O/'headline.csv').open())};c=h['core'];d=h['road']
p=M/'03_methodology.tex';s=p.read_text();a=s.index('The retained runtime study measures');b=s.index('\\section{Cross-Dataset Evaluation}',a)
s=s[:a]+r'''The runtime study reuses the saved BIU-v2 core and road-extended XGBoost models on
100 PIE observation windows. Their identities are the same as in the earlier AIBThings
v1/RF timing study. Before timing, feature reconstruction is checked against the saved
inputs and probabilities are checked against all three model seeds. No model is retrained.

Measurements use an Intel Core i7-11800H CPU with one XGBoost worker and one numerical
library thread. Changing the in-memory worker count preserves the recorded probabilities.
Ready-feature prediction includes median imputation and XGBoost: 200 warm-ups and 2,000
measurements for individual windows, or 20 warm-ups and 200 measurements for batches of
100, repeated for each model seed. Batch time divided by batch size describes throughput.
A separate sequential test uses seed 42, 100 warm-ups and 1,000 measurements to time
BBox53, BIU-v2, temporal and ego-feature construction, input packing and prediction together.
Both tests cycle through the same observation windows. Inputs are loaded into memory before
timing; decoding, disk I/O, model loading and training are excluded.

The road test starts with the saved $16\times20$ road-relative geometry and measures its
60-feature summary, packing and prediction in addition to the core workload. It excludes
mask-to-geometry construction. Detection, pose estimation and road segmentation remain
outside every reported downstream measurement. The earlier AIBThings cost is an effective
estimate combining feature-stage means with amortized batched prediction; its workload
and software differ from this actual single-window measurement.
% SOURCE: experiments/v2_xgboost_runtime_2026_09_13/DESIGN.md
% SOURCE: experiments/v2_xgboost_runtime_2026_09_13/run.py
% SOURCE: experiments/v2_xgboost_runtime_2026_09_13/environment.json
% SOURCE: experiments/aibthings_model_only_runtime/runtime_stage_summary.csv

'''+s[b:];p.write_text(s)
p=M/'04_results_and_discussion.tex';s=p.read_text();a=s.index('Figure~\\ref{fig:ch4_runtime} summarizes');b=s.index('\\paragraph{Competitive performance',a)
s=s[:a]+r'''Table~\ref{tab:ch4_runtime} reports the measured BIU-v2 pipeline cost on an Intel Core
i7-11800H CPU. From ready features, prediction averages COREP\,ms for the core model
and ROADP\,ms for the road model. Recomputing the core features from available boxes,
keypoints and speed, then predicting, takes CORET\,ms per observation window
(Figure~\ref{fig:ch4_runtime}). The corresponding 95th percentile is CORE95\,ms.

\input{Tables/ch4_runtime}
\input{Figures/ch4_runtime}

Feature construction dominates this implementation: BIU-v2 extraction averages 2.69\,ms,
and the early/late contrasts, which rerun the extractor on both halves, average 3.98\,ms.
The road-extended workload takes ROADT\,ms when road-relative geometry is already
available; deriving that geometry from masks is not included. These measurements support
a lightweight downstream decision stage, without assigning its latency to upstream
neural perception.

For continuity with AIBThings, using ready BBox features and amortized batched prediction
gives an effective COREE\,ms for the v2 core, versus the earlier study's 1.09\,ms
for v1/RF. The v2 workload also includes explicit temporal and ego branches, so this is
not an isolated version or classifier speed comparison. Single-window response and
effective batch cost remain different quantities.
% SOURCE: experiments/v2_xgboost_runtime_2026_09_13/headline.csv
% SOURCE: experiments/v2_xgboost_runtime_2026_09_13/stage_summary.csv
% SOURCE: experiments/aibthings_model_only_runtime/runtime_stage_summary.csv

'''.replace('COREP',f"{float(c['cached_single_ms']):.2f}").replace('ROADP',f"{float(d['cached_single_ms']):.2f}").replace('CORET',f"{float(c['actual_single_including_bbox_ms']):.2f}").replace('CORE95',f"{float(c['actual_single_p95_ms']):.2f}").replace('ROADT',f"{float(d['actual_single_including_bbox_ms']):.2f}").replace('COREE',f"{float(c['effective_ready_bbox_ms']):.2f}")+s[b:];p.write_text(s)
lines=[]
for case,label in [('core','BIU-v2 core (135-D)'),('road','BIU-v2 + roads (195-D)')]:
 x=h[case];lines.append(label+' & '+' & '.join(f'{float(x[k]):.2f}' for k in ['cached_single_ms','actual_single_including_bbox_ms','actual_single_p95_ms'])+r'\\')
(M/'Tables/ch4_runtime.tex').write_text(r'''\begin{table}[tbp]
\centering\small
\caption{Measured CPU milliseconds per observation window.}\label{tab:ch4_runtime}
\begin{tabularx}{\textwidth}{@{}Xrrr@{}}\hline
 & Ready-feature & \multicolumn{2}{c}{Feature construction + prediction}\\
Model & prediction & Mean & 95th percentile\\\hline
'''+ '\n'.join(lines)+r'''
\hline\end{tabularx}
\smallskip
\begin{minipage}{\textwidth}\footnotesize
Ready-feature means average three model seeds; sequential feature-to-prediction timings
use seed 42 and include BBox extraction. The road row starts from available road geometry.
\end{minipage}
\end{table}
''')
p=M/'Figures/ch4_runtime.tex';s=p.read_text();s=s.replace('1.09 ms / observation',f"{float(c['actual_single_including_bbox_ms']):.2f} ms / observation").replace('Effective downstream CPU time','Measured core pipeline on CPU').replace('{Ready BBox features}','{Boxes + ego speed}').replace(r'\textbf{BIU-v1}\\construction',r'\textbf{Feature}\\construction').replace(r'\textbf{Feature}\\fusion',r'\textbf{Input}\\packing').replace(r'\textbf{Random Forest}\\prediction',r'\textbf{XGBoost}\\prediction').replace('Intel Core i7-11800H CPU; classifier batches of 100','Intel Core i7-11800H CPU; one observation per prediction').replace(r'(5.3,-1.7) -- (fusion.north)',r'(2.5,-1.7) -- (biu.north)').replace('Measured AIBThings decision stage; upstream detection and pose estimation are excluded.','Measured BIU-v2 core pipeline from available boxes, keypoints and ego speed.');p.write_text(s)
for n in files[3:]:
 p=R/n;s=p.read_text();s=s.replace('thesis_neural_only_union_2026_09_13/build/Thesis.pdf','v2_xgboost_runtime_2026_09_13/build/Thesis.pdf')
 s=s.replace('A matching v2/XGBoost runtime measurement remains pending.','V2/XGBoost runtime is now measured; the road timing starts from available geometry.')
 s=s.replace('measure the main v2/XGBoost runtime','review the completed v2/XGBoost runtime audit')
 s=s.replace('v2/XGBoost latency remains to be measured.','v2/XGBoost latency is measured in the new runtime audit.')
 s=s.replace('a corresponding saved-model timing study.','the corresponding saved-model timing study now recorded in `experiments/v2_xgboost_runtime_2026_09_13/`.')
 note='''\n**Runtime completed — 2026-09-13:** Saved BIU-v2/XGBoost models were timed without refitting.
Ready-feature prediction is 1.06 ms (core) / 1.17 ms (road); sequential feature construction,
including BBox, plus prediction is 9.55 / 9.64 ms. The road row starts from cached geometry,
excluding mask-to-geometry extraction. CPU: i7-11800H, one worker. The earlier 1.09 ms is
v1/RF effective batch cost, not these timings. Source and commands:
`experiments/v2_xgboost_runtime_2026_09_13/README.md`. This supersedes earlier timing-pending notes.
'''
 pos=s.index('\n')+1;s=s[:pos]+note+s[pos:];p.write_text(s)
