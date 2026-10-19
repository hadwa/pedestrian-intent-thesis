"""Verify copied neural weights/metadata; does not import or train models."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
records=json.loads((ROOT/'audit/saved_models/copy_manifest.json').read_text())
for record in records:
    path=(ROOT/record['destination']).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Artifact outside companion repository')
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    if h.hexdigest()!=record['sha256']:raise ValueError(f'Checksum mismatch: {path}')
print(json.dumps({'verified_artifacts':len(records),'models_trained':0}))
