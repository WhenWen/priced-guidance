"""Package one Results Explorer dataset for a shareable Google Drive submission."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import zipfile

CREDENTIAL = re.compile(r'\b(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{25,}|AKIA[A-Z0-9]{16})\b|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')

def sha(data):
    return hashlib.sha256(data).hexdigest()

def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def pack(site: Path, out: Path, dataset: str):
    if out.exists():
        raise ValueError('output exists; choose a fresh export directory')
    original = json.loads((site / 'lib/catalog.json').read_text())
    cohorts = [d for d in original['datasets'] if d['id'] == dataset]
    if len(cohorts) != 1:
        raise ValueError('select one existing dataset')
    cohort = cohorts[0]
    rows = [r for r in original['runs'] if r['dataset'] == dataset]
    papers = sorted({r['paper'] for r in rows})
    models = sorted({r['model'] for r in rows})
    groups = defaultdict(list)
    owners = {}
    for row in rows:
        groups[row['stage'], row['model']].append(row)
        if row['id']:
            if row['id'] in owners and owners[row['id']] != row['model']:
                raise ValueError('one trajectory is attributed to multiple models')
            owners[row['id']] = row['model']
    for group in groups.values():
        if len(group) != cohort['n'] or len({r['paper'] for r in group}) != cohort['n']:
            raise ValueError('incomplete or duplicate cohort')
    ids = set(owners)
    if any(member not in ids for row in rows for member in row['members']):
        raise ValueError('ensemble references a missing member')
    catalog = {**original, 'datasets': cohorts, 'runs': rows,
        'papers': {pid: original['papers'][pid] for pid in papers},
        'models': {model: original['models'][model] for model in models}}
    out.mkdir(parents=True)
    metadata = out / 'metadata'
    metadata.mkdir()
    promotions = json.loads((site / 'scripts/promotion_costs.json').read_text())
    promotions = {rid: value for rid, value in promotions.items() if rid in ids}
    fingerprints, index = {}, []
    linked_choices = 0
    per_model = defaultdict(list)
    for rid in sorted(ids):
        per_model[owners[rid]].append(rid)
    for model, run_ids in sorted(per_model.items()):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', model):
            raise ValueError('unsafe model identifier')
        archive = out / f'trajectories-{model}.zip'
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as z:
            for rid in run_ids:
                if not re.fullmatch(r'[A-Za-z0-9_-]+', rid):
                    raise ValueError('unsafe run ID')
                records = {}
                for kind in ('raw', 'runs'):
                    rel = f'public/data/{kind}/{rid}.json.gz'
                    data = (site / rel).read_bytes()
                    text = gzip.decompress(data).decode()
                    if CREDENTIAL.search(text):
                        raise ValueError(f'possible credential in {rel}')
                    records[kind] = json.loads(text)
                    if records[kind].get('id') != rid or records[kind].get('error'):
                        raise ValueError(f'invalid trajectory {rel}')
                    fingerprints[rel] = sha(data)
                    info = zipfile.ZipInfo(rel, (2026, 10, 3, 0, 0, 0))
                    info.external_attr = 0o644 << 16
                    z.writestr(info, data)
                raw, trace = records['raw'], records['runs']
                if raw['paper'] not in papers or raw['paper'] != trace['paper']:
                    raise ValueError(f'wrong paper for {rid}')
                if any('promotionCost' in e or 'display_cost' in e for e in raw['events']):
                    raise ValueError(f'raw events contain presentation accounting: {rid}')
                if not trace.get('target', {}).get('sections'):
                    raise ValueError(f'missing structured target: {rid}')
                if rid in promotions:
                    expected = promotions[rid]
                    if not math.isclose(trace['promotionCost']['bits'], float(expected['bits']), abs_tol=1e-8):
                        raise ValueError(f'promotion mismatch: {rid}')
                for row in (r for r in rows if r['id'] == rid and r['status'] == 'pass'):
                    if not math.isclose(trace['displayCost'], row['cost'], abs_tol=1e-5):
                        raise ValueError(f'endpoint mismatch: {rid}')
                started = next((e for e in raw['events'] if e['kind'] == 'run_started'), {})
                linked_choices += sum(e['kind'] == 'choice_cost' for e in raw['events'])
                index.append({'run_id': rid, 'model': model, 'target_id': raw['paper'],
                    'stage': raw['stage'], 'seed': started.get('seed'),
                    'generator_model': (raw.get('models') or {}).get('generator'),
                    'oracle_model': (raw.get('models') or {}).get('oracle'),
                    'judge_model': (raw.get('models') or {}).get('judge'),
                    'effort': raw.get('effort'), 'retained_history': raw.get('history'),
                    'judge_repeats': started.get('judge_repeats'),
                    'archive': archive.name, 'source_event_sha256': raw.get('sourceHash')})
    for row in rows:
        if row['members']:
            members = [r for r in rows if r['id'] in row['members'] and r['paper'] == row['paper'] and r['stage'] == row['stage']]
            if len(members) != len(row['members']):
                raise ValueError('unresolved ensemble member')
            # Website test ensemble always has three models; failed members still count.
            n = 3 if dataset == 'test' else len(row['members'])
            costs = [r['cost'] for r in members if r['status'] == 'pass' and r['cost'] is not None]
            lo = min(costs) if costs else math.inf
            expected = lo - math.log2(math.fsum(2 ** (lo - c) for c in costs) / n) if costs else None
            if (expected is None) != (row['cost'] is None) or (expected is not None and not math.isclose(expected, row['cost'], abs_tol=1e-5)):
                raise ValueError('ensemble cost mismatch')
    write_json(metadata / 'catalog.json', catalog)
    write_json(metadata / 'promotion_costs.json', promotions)
    with (metadata / 'results.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader()
        for row in rows:
            w.writerow({**row, 'members': json.dumps(row['members'])})
    with (metadata / 'run-index.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(index[0]), lineterminator='\n'); w.writeheader(); w.writerows(index)
    provenance = json.loads((site / 'scripts/source-provenance.json').read_text())
    provenance.update(site_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=site, text=True).strip(),
        site_catalog_sha256=sha((site / 'lib/catalog.json').read_bytes()), dataset=dataset,
        omitted_datasets=[d['id'] for d in original['datasets'] if d['id'] != dataset],
        record_semantics='Byte-identical published website exports; raw means sanitized experiment-content exports, not original private provider tapes.')
    write_json(metadata / 'provenance.json', provenance)
    report = {'status': 'ok', 'targets': len(papers), 'outcomes': len(rows),
        'trajectories': len(ids), 'trajectory_files': len(fingerprints), 'promotion_charges': len(promotions),
        'priced_choices': linked_choices, 'credential_pattern_matches': 0,
        'cohort_and_endpoint_checks': 'passed', 'ensemble_checks': 'passed',
        'statuses': dict(Counter(row['status'] for row in rows))}
    write_json(metadata / 'validation.json', report)
    (metadata / 'README.md').write_text(f'''# Priced Guidance: main-scaffold results

This is the `{dataset}` dataset used by the Results Explorer: {len(papers)} papers,
{len(rows)} outcome rows, and {len(ids)} independent recorded trajectories.
It includes Directional and Essence, the five individual models, and their reported
three-model ensemble. All failed, budget-blocked, refused and unpromoted outcomes
remain in the catalog. Development ablations are not included in this bundle.

Download `metadata.zip` and the model ZIPs, then extract them into the same directory:

```text
catalog.json             # paper/model metadata and every outcome, as in the website
results.csv              # flat view of the catalog outcomes
run-index.csv            # exact model IDs, effort, seed, and archive for each run
promotion_costs.json     # expected manuscript promotion charges
provenance.json          # website/manuscript revision and source hashes
validation.json          # cohort, endpoint, ensemble and credential checks
manifest.json            # SHA-256 of every extracted file except this manifest
verify.py               # offline integrity, completeness and endpoint verification
public/data/raw/*.json.gz   # sanitized exact event and Guide-message exports
public/data/runs/*.json.gz  # browser records with corrected costs and structured targets
```

Run `python verify.py` after extraction. Each model ZIP preserves the website paths
and exact compressed bytes. Ensembles refer to member run IDs and have no separate
trajectory. A missing run ID is explicit; it must not be replaced with a synthetic
successful record. Blank CSV costs / JSON null mean no successful finite result.

Displayed costs use {original['accounting']}. Raw exported event messages retain
historical charges; browser `displayCost` includes the manuscript corrections.
Do not add promotion costs a second time. Failed ensemble members contribute zero
probability and stay in the denominator. These are already-sanitized scientific
exports, not complete private provider tapes or original replay hash chains.

The paired code release is Priced Guidance. To contribute new compression results,
provide a Google Drive folder with this logical layout, a README with exact run
commands/configuration, all planned outcomes, and a checksum/validation report.
Upload only the sanitized research exports and give reviewers download access.
''')
    verifier = '''"""Verify an extracted Priced Guidance Drive data bundle (standard library only)."""
import gzip, hashlib, json, math
from pathlib import Path
r = Path(__file__).resolve().parent
m = json.loads((r / 'manifest.json').read_text())
for name, expected in m['files'].items():
    p = r / name
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
        raise SystemExit('Missing or changed file: ' + name)
c = json.loads((r / 'catalog.json').read_text())
ids = {row['id'] for row in c['runs'] if row['id']}
for kind in ['raw', 'runs']:
    actual = {p.name[:-8] for p in (r / 'public/data' / kind).glob('*.json.gz')}
    if actual != ids:
        raise SystemExit('Trajectory set differs: ' + kind)
for row in c['runs']:
    if row['id'] and row['status'] == 'pass':
        trace = json.loads(gzip.decompress((r / 'public/data/runs' / (row['id'] + '.json.gz')).read_bytes()))
        if not math.isclose(trace['displayCost'], row['cost'], abs_tol=1e-5):
            raise SystemExit('Endpoint differs: ' + row['id'])
print(json.dumps({'status': 'ok', 'files': len(m['files']), 'outcomes': len(c['runs']), 'trajectories': len(ids)}))
'''
    (metadata / 'verify.py').write_text(verifier)
    for path in metadata.iterdir():
        fingerprints[path.name] = sha(path.read_bytes())
    write_json(metadata / 'manifest.json', {'algorithm': 'sha256', 'files': fingerprints})
    with zipfile.ZipFile(out / 'metadata.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(metadata.iterdir()):
            z.write(path, path.name)
    (out / 'README.md').write_bytes((metadata / 'README.md').read_bytes())
    deliverables = sorted(p for p in out.iterdir() if p.is_file())
    (out / 'SHA256SUMS').write_text(''.join(f'{sha(p.read_bytes())}  {p.name}\n' for p in deliverables))
    print(json.dumps(report, indent=2))
    print(json.dumps({p.name: p.stat().st_size for p in out.iterdir() if p.is_file()}, indent=2))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--site-root', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--dataset', default='test', choices=['test'])
    args = ap.parse_args()
    pack(args.site_root.resolve(), args.out_dir.resolve(), args.dataset)
