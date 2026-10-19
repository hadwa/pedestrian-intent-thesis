"""Read the canonical TeX input graph; write audit evidence beside this script.

Run from the repository root. No experiment or manuscript files are written.
Structural checks supplement, rather than replace, the documented editorial review.
"""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import re
import subprocess

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
ROOT = REPO / 'thesis/manuscript'
files = {}
labels = defaultdict(list)
references, captions, graphics, citations, missing = [], [], [], [], []


def uncomment(text):
    return re.sub(r'(?<!\\)%[^\n]*', '', text)


def visit(path):
    path = path.resolve()
    if path in files:
        return
    text = uncomment(path.read_text())
    files[path] = text
    name = str(path.relative_to(ROOT))
    for number, line in enumerate(text.splitlines(), 1):
        for match in re.finditer(r'\\label\{([^}]+)\}', line):
            labels[match[1]].append({'file': name, 'line': number})
        for match in re.finditer(r'\\(?:ref|eqref|pageref|autoref|cref|Cref)\*?\{([^}]+)\}', line):
            references.append({'file': name, 'line': number, 'target': match[1], 'context': line})
        if re.search(r'\\caption(?:\[|\{)', line):
            captions.append({'file': name, 'line': number, 'first_line': line})
        for match in re.finditer(r'\\(?:cite|parencite|textcite|autocite)(?:\[[^]]*\])*\{([^}]+)\}', line):
            citations.extend(match[1].split(','))
        for match in re.finditer(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', line):
            candidates = [ROOT / match[1], ROOT / 'Figures' / match[1]]
            candidates += [Path(str(p) + ext) for p in list(candidates) for ext in ('.pdf', '.png', '.jpg')]
            found = next((p for p in candidates if p.is_file()), None)
            graphics.append({'file': name, 'line': number, 'path': str(found.relative_to(ROOT)) if found else None})
        for match in re.finditer(r'\\(?:input|include)\{([^}]+)\}', line):
            child = ROOT / match[1]
            if not child.suffix:
                child = child.with_suffix('.tex')
            if child.is_file():
                visit(child)
            else:
                missing.append(str(child))


visit(ROOT / 'Thesis.tex')
bib = (ROOT / 'bibliography.bib').read_text()
bib_keys = re.findall(r'@\w+\s*\{\s*([^,\s]+)\s*,', bib)
aux = '\n'.join(p.read_text() for p in ROOT.rglob('*.aux'))
resolved = {m[1]: {'number': m[2], 'page': m[3]} for m in re.finditer(r'\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{([^}]*)\}', aux)}
for ref in references:
    ref.update(resolved.get(ref['target'], {'number': '', 'page': ''}))
    ref['destination'] = json.dumps(labels.get(ref['target'], []))

all_text = '\n'.join(files.values())
stale_patterns = {
    'removed_union_name': r'(?i)BIU[- ]Combined|full[- ]union|union135',
    'old_v2_name': r'(?i)BIU[- ]v2 core|v2-core|extended core',
    'removed_TREP': r'\bTrEP\b',
    'old_runtime': r'\b(?:1\.09|6\.82)\s*(?:\\,)?\s*ms',
    'removed_pose_diagnostic': r'\b29 tracks\b|4,842|coordinate-offset analysis',
    'unresolved_placeholder': r'\b(?:TODO|TBD|FIXME)\b|\?\?',
}
stale = {name: re.findall(pattern, all_text) for name, pattern in stale_patterns.items()}
duplicate_bib_keys = sorted({key for key in bib_keys if bib_keys.count(key) > 1})
checks = {
    'missing_input_files': missing,
    'missing_graphics': [g for g in graphics if g['path'] is None],
    'undefined_references': [r for r in references if r['target'] not in labels],
    'duplicate_labels': {key: value for key, value in labels.items() if len(value) > 1},
    'undefined_citations': sorted(set(citations) - set(bib_keys)),
    'duplicate_bibliography_keys': duplicate_bib_keys,
    'stale_pattern_matches': {key: value for key, value in stale.items() if value},
}
inventory = {
    'active_files': [str(p.relative_to(ROOT)) for p in files],
    'labels': dict(labels), 'references': references, 'captions': captions,
    'graphics': graphics, 'unique_citations': sorted(set(citations)), 'checks': checks,
}
(OUT / 'inventory_after.json').write_text(json.dumps(inventory, indent=2) + '\n')
with (OUT / 'cross_references.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(references[0]))
    writer.writeheader()
    writer.writerows(references)
hashes = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
for graphic in graphics:
    if graphic['path']:
        path = ROOT / graphic['path']
        hashes[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
(OUT / 'reviewed_source_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')

graphic_text = []
for graphic in graphics:
    if graphic['path'] and graphic['path'].endswith('.pdf'):
        result = subprocess.run(['pdftotext', '-layout', str(ROOT / graphic['path']), '-'], check=True, capture_output=True, text=True)
        graphic_text.append(graphic['path'] + '\n' + result.stdout)
(OUT / 'embedded_figure_text.txt').write_text('\n\n'.join(graphic_text))

# Literal phrase overlap, ignoring case and TeX commands. Not a semantic repetition test.
def tokens(text):
    text = re.sub(r'\\(?:ref|cite|label)\{[^}]+\}', '', uncomment(text))
    text = re.sub(r'\\[A-Za-z]+', '', text)
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text.lower())

conclusion = tokens((ROOT / '05_conclusion.tex').read_text())
overlaps = []
for path in [ROOT / '00_d_abstract.tex', *sorted(ROOT.glob('0[1-4]_*.tex'))]:
    other = tokens(path.read_text())
    index = defaultdict(list)
    for j in range(len(other) - 9):
        index[tuple(other[j:j + 10])].append(j)
    for i in range(len(conclusion) - 9):
        for j in index.get(tuple(conclusion[i:i + 10]), []):
            if i and j and conclusion[i - 1] == other[j - 1]:
                continue
            length = 10
            while i + length < len(conclusion) and j + length < len(other) and conclusion[i + length] == other[j + length]:
                length += 1
            overlaps.append({'other': path.name, 'words': length, 'text': ' '.join(conclusion[i:i + length])})
(OUT / 'chapter5_phrase_matches.json').write_text(json.dumps(overlaps, indent=2) + '\n')
summary = {'active_tex_files': len(files), 'labels': len(labels), 'reference_uses': len(references),
           'caption_commands': len(captions), 'graphics': len(graphics), 'unique_citations': len(set(citations)),
           'checks': checks}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
raise SystemExit(1 if any(checks.values()) else 0)
