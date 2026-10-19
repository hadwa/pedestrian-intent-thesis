from pathlib import Path
import subprocess,json
D=Path('/home/abdelrahman/Hadwa/pedestrian-intent-thesis');A=D/'audit/mslstm_evaluation'
wrapper='''import sys,os,runpy
blocked=('/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean','/home/abdelrahman/Hadwa/BIU-PID-Distillation')
def guard(event,args):
 if event in ('open','os.listdir','os.scandir') and args and isinstance(args[0],(str,bytes)):
  path=os.path.realpath(os.fsdecode(args[0]))
  if any(path==r or path.startswith(r+'/') for r in blocked):raise RuntimeError('Original repository access denied: '+path)
sys.addaudithook(guard)
script=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(script,run_name='__main__')
'''
cmd=['/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python','-B','-I','-c',wrapper,str(D/'tools/evaluate_mslstm.py'),'--output',str(D/'runs/mslstm_saved_replay')]
r=subprocess.run(cmd,cwd='/tmp',capture_output=True,text=True)
(A/'validation_command.json').write_text(json.dumps({'command':cmd,'cwd':'/tmp','returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr},indent=2)+'\n');(A/'validate.py').write_bytes(Path(__file__).read_bytes())
print(r.stdout);print(r.stderr[-4000:]);raise SystemExit(r.returncode)
