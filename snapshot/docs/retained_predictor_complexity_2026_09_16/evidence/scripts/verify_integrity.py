"""Compare audit inputs and tracked repository files to pre-audit fingerprints; no repository writes."""
from pathlib import Path
import json,hashlib,subprocess,datetime
O=Path(__file__).resolve().parents[1]
R=Path('/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def git(*args):return subprocess.check_output(['git','--no-optional-locks',*args],cwd=R,text=True)
before=json.loads((O/'repository_before.json').read_text())
changes=[]
for rel,old in before['tracked_hashes'].items():
 p=R/rel
 if not p.is_file() or sha(p)!=old:changes.append(rel)
sources={}
for name in ['sources.json','latency_sources.json']:
 for name,meta in json.loads((O/name).read_text()).items():
  if name in sources:assert sources[name]==meta
  sources[name]=meta
source_changes=[]
for name,old in sources.items():
 p=Path(name)
 if not p.is_file() or p.stat().st_size!=old['bytes'] or sha(p)!=old['sha256']:source_changes.append(name)
status=git('status','--porcelain','--untracked-files=all')
result={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'repository':str(R),'head_unchanged':git('rev-parse','HEAD').strip()==before['head'],'tracked_files_checked':len(before['tracked_hashes']),'changed_tracked_files':changes,'source_files_checked':len(sources),'changed_source_files':source_changes,'git_status_unchanged':status==before['status'],'all_audit_outputs_directory':str(O),'note':'Pre-existing changes are preserved. No training, original model overwrite, repository writes, commit or push performed by this audit.'}
(O/'immutability_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
assert not changes and not source_changes and result['head_unchanged'] and result['git_status_unchanged']
