"""Check the packaged thesis artifacts without loading models or original files."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'audit/artifact_package_completion/verification_manifest.json').read_text())
for relative,expected in manifest.items():
    p=(ROOT/relative).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Path outside companion repository')
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    if h.hexdigest()!=expected:raise ValueError(f'Checksum mismatch: {relative}')
print(json.dumps({'verified_paths':len(manifest),'models_loaded':0,'models_trained':0}))
