"""验证划分集合的隔离性；分组真实性需要单独来源证据。"""
from collections import Counter
from pathlib import Path

from .data import read_json, write_json


def check_splits(manifest_path, output=None):
    manifest = read_json(manifest_path)
    names = ('train', 'val', 'test')
    classes = manifest['classes']
    issues, summary = [], {}
    sets = {field:{} for field in ('path', 'sha256', 'group')}
    root = Path(manifest['root']).resolve()
    for name in names:
        rows = manifest['splits'][name]
        if not rows:
            issues.append(f'{name}: empty split')
        for field in sets:
            sets[field][name] = {r[field] for r in rows}
        if len(sets['path'][name]) != len(rows):
            issues.append(f'{name}: repeated path')
        if len(sets['sha256'][name]) != len(rows):
            issues.append(f'{name}: exact duplicate content')
        counts = Counter(r['label'] for r in rows)
        if set(counts) != set(range(len(classes))):
            issues.append(f'{name}: missing or invalid class labels')
        for row in rows:
            resolved = (root / row['path']).resolve()
            if root not in resolved.parents:
                issues.append(f'{name}: path outside dataset root')
                break
        summary[name] = dict(images=len(rows), groups=len(sets['group'][name]),
                             class_counts={label:counts[i] for i,label in enumerate(classes)},
                             class_group_counts={label:len({r['group'] for r in rows if r['label']==i})
                                                 for i,label in enumerate(classes)})
    for i,a in enumerate(names):
        for b in names[i+1:]:
            for field in sets:
                overlap = sets[field][a] & sets[field][b]
                if overlap:
                    issues.append(f'{a}/{b}: overlapping {field}: {len(overlap)}')
    report = dict(passed=not issues, issues=issues, strategy=manifest['strategy'], splits=summary,
                  group_basis=manifest.get('group_basis','unspecified_legacy_manifest'),
                  patient_independence='requires original patient-to-image mapping and source verification',
                  scope='Manifest integrity only; does not infer patient IDs, detect near-duplicates, or rehash on-disk files.')
    if output:
        write_json(output, report)
    if issues:
        raise ValueError('; '.join(issues))
    return report
