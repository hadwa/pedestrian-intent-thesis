from pathlib import Path
import hashlib,difflib,json,re,csv
import fitz
R=Path.cwd();O=R/'experiments/v2_xgboost_runtime_2026_09_13';M=R/'thesis/manuscript'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
log=(O/'build/Thesis.log').read_text();assert 'undefined' not in log.lower();assert 'Float too large' not in log;assert 'LaTeX Error' not in log
pdf=fitz.open(O/'build/Thesis.pdf');texts=[p.get_text() for p in pdf];full='\n'.join(texts);(O/'Thesis.txt').write_text(full)
assert 'Combined' not in full
assert not re.search(r'\b5\s*[–−-]\s*15\b|\b15\s*[–−-]\s*30\b',full)
assert sum('tau=0.5' in p.read_text() for p in M.rglob('*.tex'))==1
h=list(csv.DictReader((O/'headline.csv').open()));table=(M/'Tables/ch4_runtime.tex').read_text()
for r in h:
 line=next(l for l in table.splitlines() if f"({r['dimensions']}-D)" in l)
 expected=[f"{float(r[k]):.2f}" for k in ['cached_single_ms','actual_single_including_bbox_ms','actual_single_p95_ms']]
 assert all(x in line for x in expected)
pages={}
for i,t in enumerate(texts):
 if i>12 and ('9.55 ms / observation' in t or 'Measured CPU milliseconds' in t):
  pages[str(i+1)]=t.strip().splitlines()[0];pdf[i].get_pixmap(matrix=fitz.Matrix(1.25,1.25)).save(str(O/f'preview_page_{i+1}.png'))
patch=[];changed=[]
for before in sorted((O/'before').rglob('*')):
 if not before.is_file():continue
 rel=before.relative_to(O/'before');after=R/rel
 if sha(before)==sha(after):continue
 changed.append(dict(path=str(rel),before_sha256=sha(before),after_sha256=sha(after)))
 patch.extend(difflib.unified_diff(before.read_text().splitlines(True),after.read_text().splitlines(True),fromfile='before/'+str(rel),tofile='after/'+str(rel)))
p=M/'Tables/ch4_runtime.tex';rel=p.relative_to(R);changed.append(dict(path=str(rel),status='new',after_sha256=sha(p)));patch.extend(difflib.unified_diff([],p.read_text().splitlines(True),fromfile='/dev/null',tofile='after/'+str(rel)))
(O/'session.patch').write_text(''.join(patch));(O/'changed_files.json').write_text(json.dumps(changed,indent=2))
result=dict(status='PASS',pdf_pages=len(pdf),runtime_viewer_pages=pages,verified_runtime_table_cells=6,undefined_refs_or_citations=0,oversized_floats=0,decision_rule_statements=1,combined_mentions=0)
(O/'pdf_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
