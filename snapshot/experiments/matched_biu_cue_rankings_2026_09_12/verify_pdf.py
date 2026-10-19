"""Check printable labels/numbers against review_table.csv (system Python)."""
from pathlib import Path
import csv,hashlib,json
import fitz

p=Path(__file__).resolve().parent
assert not (p/'pdf_verification.json').exists()
doc=fitz.open(p/'Matched_BIU_Cue_Rankings.pdf')
rows=list(csv.DictReader((p/'review_table.csv').open()));order=[]
for r in csv.DictReader((p/'cohorts.csv').open()):
 k=(r['setting'],r['version'])
 if k not in order:order.append(k)
assert len(doc)==2+len(order)==21
counts=[]
for page,(s,v) in zip(list(doc)[2:],order):
 text=page.get_text();group=[r for r in rows if r['setting']==s and r['version']==v and r['section']=='static_biu']
 for r in group:
  # Matplotlib's PDF text extraction maps en dashes to spaces; visible glyph is retained.
  assert r['label'].replace('\u2013',' ') in text,r['label']
  for k in ['val_importance_pct','test_importance_pct','val_toward_pct','test_toward_pct','test_away_pct']:
   assert f"{float(r[k]):.1f}" in text
  for k in ['val_balanced_net','test_balanced_net','test_positive_class_net','test_negative_class_net']:
   assert f"{float(r[k]):+.3f}" in text
 counts.append(len(group))
for page in doc:
 assert all(w[0]>=0 and w[1]>=0 and w[2]<=page.rect.width and w[3]<=page.rect.height for w in page.get_text('words'))
manifest={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),p/'report.py',p/'verify.py',p/'review_table.csv',p/'REPORT.md',p/'Matched_BIU_Cue_Rankings.pdf',p.parent/'biu_cue_attribution_report_2026_09_12/cue_dictionary.csv']}
(p/'pdf_verification.json').write_text(json.dumps(dict(pages=len(doc),cue_rows=sum(counts),all_cue_labels_and_rounded_numeric_cells_present=True,no_text_outside_pages=True,visually_inspected_page=14,sources_and_outputs=manifest),indent=2)+'\n')
print('PDF verified:',len(doc),'pages;',sum(counts),'cue rows')
