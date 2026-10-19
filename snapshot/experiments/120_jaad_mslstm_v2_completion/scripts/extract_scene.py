"""Run the pinned historical YOLOP extractor in its existing compatible environment."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode=True
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
SCRIPT=REPO/'experiments/049_exp048_to_jaad_zero_shot/scripts/yolop.py'
sys.path.insert(0,str(SCRIPT.parent))
spec=importlib.util.spec_from_file_location('frozen_yolop',SCRIPT)
Y=importlib.util.module_from_spec(spec);spec.loader.exec_module(Y)
torch.set_num_threads(1);torch.set_num_interop_threads(1)
assert torch.cuda.is_available()
jobs=json.loads(Path(sys.argv[1]).read_text())
assert jobs and all(set(j)=={'video','frame','ped','bbox','image'} for j in jobs)
engine=Y.YolopExtractor('cuda')
started=time.time();count=0
for video in sorted({j['video'] for j in jobs}):
    records={}
    out=ROOT/'cache/yolop'/f'{video}.npz'
    if out.exists():
        with np.load(out,allow_pickle=False) as d:
            records={(str(p),int(f)):v for p,f,v in zip(d['ped_id'],d['frame'],d['features'])}
    subset=[j for j in jobs if j['video']==video]
    for fr in sorted({j['frame'] for j in subset}):
        current=[j for j in subset if j['frame']==fr]
        assert len({j['image'] for j in current})==1
        image=cv2.imread(current[0]['image'])
        if image is None: raise RuntimeError(current[0]['image'])
        det,da,ll=engine.infer_frame(image)
        for j in current:
            v=Y.compose_features(da,ll,det,j['bbox'])
            assert v.shape==(20,) and np.isfinite(v).all()
            records[(j['ped'],fr)]=v
        count+=1
    keys=sorted(records)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.with_suffix('.tmp').open('wb') as f:
        np.savez_compressed(f,video_id=np.array([video]*len(keys)),ped_id=np.array([k[0] for k in keys]),
            frame=np.array([k[1] for k in keys]),features=np.stack([records[k] for k in keys]))
    os.replace(out.with_suffix('.tmp'),out)
    print(f'YOLOP {video} images_done={count} seconds={time.time()-started:.1f}',flush=True)
