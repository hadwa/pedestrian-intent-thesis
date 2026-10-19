from pathlib import Path
import fitz,json,hashlib
OUT=Path(__file__).resolve().parents[1];REPORT=OUT/'report'
oldpath=OUT.parent/'exp075_exp082_exp083_exp099_exp114_input_attribution_2026_09_12'/'Input_Attribution_PIE_JAAD_PSI_with_Clean_JAAD.pdf'
old=fitz.open(oldpath);new=fitz.open(REPORT/'Exp115_Results_and_Attribution.pdf');assert len(old)==4 and len(new)==2
result=fitz.open();result.insert_pdf(old);result.insert_pdf(new);assert len(result)==6
for i in range(4):assert result[i].get_text()==old[i].get_text()
path=REPORT/'Input_Attribution_with_Exp115.pdf';result.save(path)
assert 'Exp115' in result[4].get_text() and 'Vehicle state' in result[5].get_text()
(OUT/'COMPLETE.json').write_text(json.dumps(dict(status='complete',models=9,cohorts=['jaad_clean','jaad_beh','jaad_all'],metric_verification='INDEPENDENT_VERIFICATION.json',attribution_verification='report/verification.json',pdf=str(path),pdf_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),pdf_pages=6,original_four_pages_preserved=True),indent=2)+'\n')
print('PDF PASS: six pages; original four pages preserved')
