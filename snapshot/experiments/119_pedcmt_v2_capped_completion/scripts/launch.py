"""Launch one detached process with persistent output, without tmux."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
logs = root / 'logs'
logs.mkdir(exist_ok=True)
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', LD_LIBRARY_PATH='/usr/lib/wsl/lib',
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
with (logs / 'pipeline.log').open('ab', buffering=0) as log:
    process = subprocess.Popen([sys.executable, '-u', str(root / 'scripts/run.py')],
        cwd=root.parents[1], env=env, stdin=subprocess.DEVNULL, stdout=log,
        stderr=subprocess.STDOUT, start_new_session=True)
(logs / 'launch.json').write_text(json.dumps({'pid':process.pid, 'time':time.time(),
    'python':sys.executable, 'working_directory':str(root.parents[1]), 'tmux':False}, indent=2)+'\n')
print(process.pid)
