"""Metadata-only PSI recovery; no model loading, predictions, or video download."""
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
spec = importlib.util.spec_from_file_location('psi_remote', REPO / 'experiments/psi_reference_recovery_2026_09_11/recover.py')
remote_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote_module)


def main():
    remote = remote_module.RemoteZip()
    records, splits, sample = [], {}, {}
    with zipfile.ZipFile(remote) as archive:
        members = [i for i in archive.infolist() if i.filename.startswith('PSI/PSI2.0_') and not i.is_dir()]
        inventory = [dict(member=i.filename, bytes=i.file_size, compressed_bytes=i.compress_size,
                          crc32=i.CRC) for i in members if '/annotations/' in i.filename or '/splits/' in i.filename]
        (ROOT / 'archive_metadata_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
        split_names = sorted(i.filename for i in members if i.filename.endswith('/PSI2_split.json'))
        cognitive = sorted(i.filename for i in members if '/PSI2.0_TrainVal/annotations/cognitive_annotation_key_frame/' in i.filename and i.filename.endswith('/pedestrian_intent.json'))
        # Deterministic schema sample: first train/validation annotation, not selected by labels.
        for name in split_names + cognitive[:1]:
            payload = archive.read(name)
            output = ROOT / 'source_metadata' / name
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                assert output.read_bytes() == payload
            else:
                output.write_bytes(payload)
            records.append(dict(member=name, sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload)))
            parsed = json.loads(payload)
            if name in split_names:
                splits[name] = parsed
            else:
                sample = dict(member=name, top_level_keys=list(parsed) if isinstance(parsed, dict) else None,
                              excerpt=str(parsed)[:8000])
    cache = REPO / 'data/psi_vitpose_s_all_tracks.jsonl'
    counts, tracks = collections.Counter(), collections.defaultdict(set)
    for line in cache.open():
        row = json.loads(line)
        counts[row['video']] += 1
        tracks[row['video']].add(row['pid'])
    coverage = {}
    for name, split in splits.items():
        coverage[name] = {k: dict(official_videos=len(v), cached_videos=len(set(v) & counts.keys()),
            records=sum(counts[x] for x in v), tracks=sum(len(tracks[x]) for x in v),
            missing_videos=sorted(set(v) - counts.keys())) for k, v in split.items()}
        coverage[name]['outside_split'] = sorted(counts.keys() - set(sum(split.values(), [])))
    differences = {}
    if len(splits) == 2:
        a, b = list(splits.values())
        differences = {k: dict(first_only=sorted(set(a[k]) - set(b[k])),
                               second_only=sorted(set(b[k]) - set(a[k]))) for k in a}
    result = dict(url=remote_module.URL, etag=remote.etag, transferred_bytes=remote.transferred,
                  source_files=records, cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
                  coverage=coverage, split_differences=differences, cognitive_schema_sample=sample,
                  predictions_run=0, models_trained=0)
    (ROOT / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
