#!/usr/bin/env python3
"""Check an allowlisted release and optionally build a ZIP (no network or Git)."""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import zipfile
from pathlib import Path

ROOT_FILES = {'README.md','README.zh-CN.md','pyproject.toml','requirements-tested.txt',
              'requirements-inference-tested.txt',
              '.gitignore','.gitattributes','CITATION.cff','LICENSE_STATUS.md','THIRD_PARTY_NOTICES.md','LICENSE'}
ROOT_DIRS = {'src','scripts','tests','configs','examples','docs','.github'}
TEXT_SUFFIXES = {'.py','.json','.csv','.md','.toml','.txt','.yml','.yaml','.cff'}
SKIP_PARTS = {'__pycache__','.pytest_cache','.mypy_cache'}
PATTERNS = [
    ('personal absolute path', re.compile('/' + r'(?:home|Users)/[A-Za-z0-9_.-]+/')),
    ('Hub token', re.compile(r'\bhf_[A-Za-z0-9]{24,}\b')),
    ('GitHub token', re.compile(r'\bgh[pousr]_[A-Za-z0-9]{30,}\b')),
    ('private key', re.compile('-----BEGIN ' + r'(?:RSA |EC |OPENSSH )?PRIVATE KEY-----')),
    ('AWS access key', re.compile(r'\bAKIA[A-Z0-9]{16}\b')),
]


def collect(root):
    files = []
    candidates = []
    for directory, dirs, names in os.walk(root, followlinks=False):
        current = Path(directory)
        dirs[:] = sorted(d for d in dirs if d not in SKIP_PARTS and not d.endswith('.egg-info')
                         and (current != root or d in ROOT_DIRS))
        for directory_name in dirs:
            if (current/directory_name).is_symlink():
                raise ValueError(f'Release directory symlink: {(current/directory_name).relative_to(root)}')
        candidates.extend(current/name for name in names)
    for path in sorted(candidates):
        rel = path.relative_to(root)
        if any(part in SKIP_PARTS for part in rel.parts):
            continue
        selected = rel.parts[0] in ROOT_DIRS or (len(rel.parts)==1 and rel.name in ROOT_FILES)
        if not selected:
            continue
        if path.is_symlink():
            raise ValueError(f'Release symlinks are forbidden: {rel}')
        if path.is_dir():
            continue
        if path.suffix not in TEXT_SUFFIXES and rel.name not in ROOT_FILES:
            raise ValueError(f'Unexpected release file type: {rel}')
        if path.stat().st_size > 2_000_000:
            raise ValueError(f'File exceeds 2 MB release threshold: {rel}')
        text = path.read_text(encoding='utf-8')
        for label, pattern in PATTERNS:
            if pattern.search(text):
                raise ValueError(f'{label} detected in {rel}; content is not printed.')
        if path.suffix == '.csv':
            rows = list(csv.DictReader(io.StringIO(text)))
            if rows and 'participant' in rows[0]:
                if str(rel) != 'examples/fmri_synthetic.csv' or any(not r['participant'].startswith('synthetic_') for r in rows):
                    raise ValueError(f'Unapproved participant-level CSV: {rel}')
        files.append(path)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--zip', type=Path, help='Create a new ZIP from reviewed allowlisted files only.')
    args = parser.parse_args()
    root = args.root.resolve()
    files = collect(root)
    provenance = json.loads((root/'docs/provenance.json').read_text())
    for record in provenance['reference_files']:
        path = root / record['file']
        if path not in files or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'Reference checksum mismatch: {record["file"]}')
    if args.zip:
        args.zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.zip, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                info = zipfile.ZipInfo('babyconceptlm-paper-analysis/' + path.relative_to(root).as_posix(), date_time=(2026,9,8,0,0,0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes())
        print(f'Created {args.zip} ({args.zip.stat().st_size:,} bytes).')
    print(f'PASS: {len(files)} release files; {sum(p.stat().st_size for p in files):,} uncompressed bytes; reference checksums verified.')
    if not (root/'LICENSE').exists():
        print('RELEASE GATE: author-approved code license is still pending; see LICENSE_STATUS.md.')
    print('This targeted scan supplements, but does not replace, manual licensing/privacy review.')


if __name__ == '__main__':
    main()
