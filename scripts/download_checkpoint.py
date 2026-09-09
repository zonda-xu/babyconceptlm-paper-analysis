#!/usr/bin/env python3
"""Explicit network opt-in; fetch a pinned public artifact without executing it."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, default=Path('configs/models.json'))
    parser.add_argument('--model', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--include-weights', action='store_true', help='Download and hash large weights as well as metadata/code.')
    args = parser.parse_args()
    models = json.loads(args.models.read_text())['models']
    if args.model not in models:
        parser.error('Choose one of: ' + ', '.join(models))
    entry = models[args.model]
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use an empty destination; do not mix checkpoint revisions.')
    from huggingface_hub import snapshot_download
    patterns = ['*.json', '*.py', '*.txt', '*.model', 'README.md', 'LICENSE*']
    if args.include_weights:
        patterns.append(entry['weight_filename'])
    location = Path(snapshot_download(repo_id=entry['hub_repository'], revision=entry['hub_commit'],
                                     local_dir=str(args.output_dir), allow_patterns=patterns))
    if args.include_weights:
        digest = hashlib.sha256()
        with (location/entry['weight_filename']).open('rb') as handle:
            for block in iter(lambda:handle.read(8*1024*1024), b''):
                digest.update(block)
        if digest.hexdigest() != entry['weight_sha256']:
            raise ValueError('Downloaded weight hash does not match the pinned reference. Do not use it.')
    print('Downloaded pinned artifact. Review its Python code before allowing custom-code inference.')


if __name__ == '__main__':
    main()
