"""Recover only the official split metadata, without test annotations."""
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'psi_reference_recovery_2026_09_11/recover.py'
spec = importlib.util.spec_from_file_location('recovery', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
remote = module.RemoteZip()
member = 'PSI/PSI2.0_TrainVal/splits/PSI2_split.json'
with zipfile.ZipFile(remote) as archive:
    payload = archive.read(member)
path = ROOT / 'official_split.json'
if path.exists():
    assert path.read_bytes() == payload
else:
    path.write_bytes(payload)
(ROOT / 'split_provenance.json').write_text(json.dumps({
    'url': module.URL, 'etag': remote.etag, 'member': member,
    'sha256': hashlib.sha256(payload).hexdigest()}, indent=2) + '\n')
print(payload.decode())
