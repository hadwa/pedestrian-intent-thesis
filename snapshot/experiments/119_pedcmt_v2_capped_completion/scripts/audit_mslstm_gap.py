"""Read-only development-cohort YOLOP coverage check; no test arrays opened."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
SOURCE = REPO / 'experiments/089_pedcmt_standardized_jaad_biu_transfer/cache/trainval_source'
PARTS = REPO / 'experiments/049_exp048_to_jaad_zero_shot/cache/yolop/yolop_parts'
rows = pd.read_csv(SOURCE / 'rows.csv')
assert set(rows.split) == {'train','val'}
report = []
for split, group in rows.groupby('split'):
    required, present, image_files = set(), set(), set()
    for r in group.itertuples():
        frames = json.loads(r.observed_frames_json)
        assert len(frames) == 16 and frames == list(range(r.start_frame, r.end_frame+1))
        required.update((r.video_id, r.ped_id, f) for f in frames)
        image_files.update((r.video_id, f) for f in frames)
    for video in sorted({x[0] for x in required}):
        p = PARTS / f'{video}.npz'
        if not p.exists(): continue
        with np.load(p, allow_pickle=False) as data:
            present.update((video,str(ped),int(fr)) for ped,fr in zip(data['ped_id'], data['frame']))
    missing = required-present
    image_root = Path('/home/abdelrahman/Hadwa/datasets/JAAD/images')
    image_available = sum((image_root / video / f'{frame:05d}.png').exists() for video,frame in image_files)
    report.append(dict(split=split, windows=len(group), required_unique_track_frames=len(required),
        cached_track_frames=len(required & present), missing_track_frames=len(missing),
        unique_images=len(image_files), available_png_images=image_available,
        exact_source_rows=str(SOURCE/'rows.csv'), cache_source=str(PARTS)))
path = ROOT / 'metrics/mslstm_jaad_development_coverage.json'
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps({'test_accessed':False, 'coverage':report}, indent=2)+'\n')
print(path.read_text())
