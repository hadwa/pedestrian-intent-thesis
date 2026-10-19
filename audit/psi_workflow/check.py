from pathlib import Path
import subprocess,json,sys
D=Path('/home/abdelrahman/Hadwa/pedestrian-intent-thesis');A=D/'audit/psi_workflow';python='/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python'
# The hook makes any accidental access to original research/archive repositories fail.
wrapper='''import sys,runpy,os
blocked=('/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean','/home/abdelrahman/Hadwa/BIU-PID-Distillation')
def guard(event,args):
 if event in ('open','os.listdir','os.scandir') and args and isinstance(args[0],(str,bytes)):
  path=os.path.realpath(os.fsdecode(args[0]))
  if any(path==root or path.startswith(root+'/') for root in blocked):raise RuntimeError('Original repository access denied: '+path)
sys.addaudithook(guard)
script=sys.argv[1];sys.argv=sys.argv[1:];sys.path.insert(0,os.path.dirname(script));runpy.run_path(script,run_name='__main__')
'''
checks=[]
def run(label,args,success=True):
 cmd=[python,'-B','-I','-c',wrapper,*args];r=subprocess.run(cmd,cwd='/tmp',capture_output=True,text=True)
 checks.append({'label':label,'command':cmd,'cwd':'/tmp','returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
 (A/'checks_v2.json').write_text(json.dumps(checks,indent=2)+'\n')
 if (r.returncode==0)!=success:raise RuntimeError(label+'\n'+r.stdout+r.stderr)
 print(label+': passed',flush=True)
run('development_preflight',[str(D/'tools/psi.py'),'preflight','--bundle',str(D/'data/psi_frozen')])
run('saved_model_replay',[str(D/'tools/replay_psi.py'),'--bundle',str(D/'data/psi_frozen'),'--reference',str(D/'data/psi_reference'),'--output',str(D/'runs/psi_saved_model_replay_v2')])
# These calls fail before fitting or writing any output.
run('reject_existing_training_output',[str(D/'tools/psi.py'),'train','--bundle',str(D/'data/psi_frozen'),'--output',str(D/'runs/psi_saved_model_replay_v2')],False)
run('reject_external_training_output',[str(D/'tools/psi.py'),'train','--bundle',str(D/'data/psi_frozen'),'--output','/tmp/psi_disallowed_output'],False)
run('reject_test_before_training',[str(D/'tools/psi.py'),'evaluate','--bundle',str(D/'data/psi_frozen'),'--trained',str(D/'runs/psi_saved_model_replay_v2'),'--output',str(D/'runs/psi_must_not_exist')],False)
assert not (D/'runs/psi_must_not_exist').exists()
(A/'check.py').write_bytes(Path(__file__).read_bytes())
