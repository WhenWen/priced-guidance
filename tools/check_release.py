"""Check tracked release files and maintain the release snapshot manifest."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'release-manifest.json'
CREDENTIAL = re.compile(r'\b(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{25,}|AKIA[A-Z0-9]{16})\b|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest', action='store_true', help='After staging reviewed changes, refresh the snapshot hashes.')
    args = parser.parse_args()
    paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    paths = sorted(p for p in paths if p and p != MANIFEST)
    failures, hashes = [], {}
    for name in paths:
        path = ROOT / name
        if path.is_symlink():
            failures.append(f'symlink: {name}')
            continue
        if any(part in {'.venv', '.git', '__pycache__', 'generated', '.idea-arena'} for part in path.relative_to(ROOT).parts):
            failures.append(f'local state: {name}')
        if path.name == '.env' or (path.name.startswith('.env.') and path.name != '.env.example') or path.name in {'auth.json', '.credentials.json'}:
            failures.append(f'credential file: {name}')
        data = path.read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        if len(data) > 10 * 1024 * 1024:
            failures.append(f'large file: {name}')
        try:
            text = data.decode('utf-8')
        except UnicodeDecodeError:
            continue
        if CREDENTIAL.search(text):
            failures.append(f'possible credential: {name}')
        if path.suffix == '.md':
            for link in re.findall(r'\]\(([^\s)]+)\)', text):
                if '://' in link or link.startswith(('#', 'mailto:')):
                    continue
                target = link.split('#')[0]
                if target and not (path.parent / target).exists():
                    failures.append(f'broken link: {name} -> {link}')
    if failures:
        raise SystemExit('\n'.join(failures))
    if args.write_manifest:
        (ROOT / MANIFEST).write_text(json.dumps({'algorithm': 'sha256', 'files': hashes}, indent=2) + '\n')
    else:
        expected = json.loads((ROOT / MANIFEST).read_text())['files']
        if hashes != expected:
            changed = sorted(k for k in hashes.keys() | expected.keys() if hashes.get(k) != expected.get(k))
            raise SystemExit('Snapshot differs: ' + ', '.join(changed))
    print(json.dumps({'status': 'ok', 'files': len(hashes), 'credential_pattern_matches': 0, 'broken_local_links': 0}))

if __name__ == '__main__':
    main()
