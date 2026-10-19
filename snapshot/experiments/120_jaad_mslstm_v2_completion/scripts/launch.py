"""Launch the approved completion independently from the interactive session."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1]
parent=list((root.parent).glob('119_*'))[0]
assert (parent/'EXPERIMENT_COMPLETE.json').exists(), 'Wait for Exp119 GPU use to finish'
(root/'logs').mkdir(exist_ok=True)
env=dict(os.environ,LD_LIBRARY_PATH='/usr/lib/wsl/lib',PYTHONDONTWRITEBYTECODE='1',
         OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (root/'logs/pipeline.log').open('ab',buffering=0) as log:
    p=subprocess.Popen([sys.executable,'-u',str(root/'scripts/run.py')],cwd=root.parents[1],
        env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'logs/launch.json').write_text(json.dumps(dict(pid=p.pid,time=time.time(),python=sys.executable),indent=2)+'\n')
print(p.pid)
