"""Recover train/validation cognitive annotations and pin official code only."""
import hashlib
import importlib.util
import json
import shutil
import subprocess
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
spec = importlib.util.spec_from_file_location('remote_psi113', REPO / 'experiments/psi_reference_recovery_2026_09_11/recover.py')
remote_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote_module)


def main():
    sources = []
    for repo, name, paths in [('/tmp/psi-intent-native', 'baseline', ['README.md', 'opts.py', 'main.py', 'train.py', 'test.py', 'database/create_database.py', 'data/process_sequence.py', 'data/prepare_data.py']),
                              ('/tmp/psi-dataset-native', 'dataset', ['README.md', 'extend_intent_annotation.py'])]:
        commit = subprocess.check_output(['git', '-C', repo, 'rev-parse', 'HEAD'], text=True).strip()
        for relative in paths:
            src, dest = Path(repo) / relative, ROOT / 'official_code' / name / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                assert dest.read_bytes() == src.read_bytes()
            else:
                shutil.copyfile(src, dest)
            sources.append(dict(repository=name, commit=commit, path=str(dest), sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    splitpath = REPO / 'experiments/110_psi_pose_offset_correction/official_split.json'
    split = json.loads(splitpath.read_text())
    remote = remote_module.RemoteZip()
    records = []
    with zipfile.ZipFile(remote) as archive:
        for video in sorted(split['train'] + split['val']):
            member = f'PSI/PSI2.0_TrainVal/annotations/cognitive_annotation_key_frame/{video}/pedestrian_intent.json'
            info = archive.getinfo(member)
            dest = ROOT / 'dataset/PSI2.0_TrainVal/annotations/cognitive_annotation_key_frame' / video / 'pedestrian_intent.json'
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                payload = dest.read_bytes()
            else:
                previous = REPO / 'experiments/108_psi_xgboost_transfer/audit_2026_09_11/source_metadata' / member
                payload = previous.read_bytes() if previous.exists() else archive.read(member)
                dest.write_bytes(payload)
            assert len(payload) == info.file_size and zlib.crc32(payload) == info.CRC
            records.append(dict(member=member, path=str(dest), sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload)))
            print(f'Recovered {len(records)}/143: {video}', flush=True)
    out = dict(source_code=sources, annotations=records, split_path=str(splitpath),
               split_sha256=hashlib.sha256(splitpath.read_bytes()).hexdigest(),
               url=remote_module.URL, etag=remote.etag, transferred_bytes=remote.transferred,
               test_annotations_accessed=False)
    (ROOT / 'recovery.json').write_text(json.dumps(out, indent=2) + '\n')
    print('RECOVERY COMPLETE', flush=True)


if __name__ == '__main__':
    main()
