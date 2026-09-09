import argparse
import hashlib
import json
from pathlib import Path

from .flops import analyze_counts
from .reference import read_csv, validate_reference
from .stats import analyze_csv


def save_json(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Offline reference checks and explicit-input reanalyses.')
    commands = parser.add_subparsers(dest='command', required=True)
    reference = commands.add_parser('reproduce', help='Verify curated aggregates and redraw figures; no models or network.')
    reference.add_argument('--reference-dir', type=Path, required=True)
    reference.add_argument('--config', type=Path, default=Path('configs/flops.json'))
    reference.add_argument('--output-dir', type=Path, required=True)
    reference.add_argument('--no-plots', action='store_true')
    paired = commands.add_parser('paired-stats', help='New paired-participant reanalysis of a provided correlation CSV.')
    paired.add_argument('--input', type=Path, required=True)
    paired.add_argument('--concept', required=True)
    paired.add_argument('--control', required=True)
    paired.add_argument('--seed', type=int, required=True)
    paired.add_argument('--samples', type=int, default=200000)
    paired.add_argument('--scale', type=float, default=100.0)
    paired.add_argument('--tasks', nargs='+', help='Explicit Holm family; otherwise all tasks in the CSV.')
    paired.add_argument('--output', type=Path, required=True)
    counts = commands.add_parser('flops-from-counts', help='New full-dev permutation reanalysis of per-sequence counts.')
    counts.add_argument('--input', type=Path, required=True)
    counts.add_argument('--config', type=Path, default=Path('configs/flops.json'))
    counts.add_argument('--pair', choices=['english','multilingual','chinese'], required=True)
    counts.add_argument('--batch-size', type=int, required=True)
    counts.add_argument('--permutations', type=int, default=200)
    counts.add_argument('--seed', type=int, required=True)
    counts.add_argument('--output', type=Path, required=True)
    counts.add_argument('--allow-unverified-input', action='store_true', help='Allow exploratory CSVs without the completed count-script metadata; never label these full-dev reproductions.')
    args = parser.parse_args()
    if args.command == 'reproduce':
        result = validate_reference(args.reference_dir, args.config)
        if not args.no_plots:
            from .plots import render
            result['figures'] = render(args.reference_dir, args.output_dir)
        save_json(args.output_dir/'validation.json', result)
        print(f'Passed {result["check_count"]} aggregate checks. Outputs: {args.output_dir}')
    elif args.command == 'paired-stats':
        save_json(args.output, analyze_csv(args.input, args.concept, args.control, args.seed, args.samples, args.scale, args.tasks))
        print(f'New paired reanalysis: {args.output}')
    else:
        config = json.loads(args.config.read_text())[args.pair]
        rows = read_csv(args.input)
        if not args.allow_unverified_input:
            meta_path = args.input.with_suffix('.meta.json')
            if not meta_path.is_file():
                parser.error('Missing complete count-script metadata; use the count script or explicitly allow exploratory input.')
            meta = json.loads(meta_path.read_text())
            if (meta.get('status') != 'complete' or meta.get('sequences') != len(rows)
                    or meta.get('sequence_length') != 256
                    or meta.get('counts_sha256') != hashlib.sha256(args.input.read_bytes()).hexdigest()):
                parser.error('Incomplete or mismatched count metadata.')
            if [int(row['sequence_index']) for row in rows] != list(range(len(rows))):
                parser.error('Missing, duplicate or reordered sequence indices.')
            expected = {k:v for k,v in config.items() if k not in {'include_residual_gate','token_control_layers','config_lm_concept_residual_gate'}}
            if meta.get('flops_shape') != expected or meta.get('residual_gate') != config['include_residual_gate']:
                parser.error('Checkpoint shape/gate does not match the selected FLOPs configuration.')
        result = analyze_counts([float(r['concept_count']) for r in rows], config, args.batch_size, args.permutations, args.seed)
        result['input_completion_verified'] = not args.allow_unverified_input
        result['input_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
        save_json(args.output, result)
        print(f'New full-dev reanalysis: {args.output}')
