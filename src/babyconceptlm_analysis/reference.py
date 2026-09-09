"""Recheck curated aggregate arithmetic; this does not rerun model evaluation."""
import csv
import json
from pathlib import Path

import numpy as np

from .flops import reproduce_moments


def read_csv(path):
    with open(path, encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def normalized_log_auc(exposure, scores):
    exposure, scores = np.asarray(exposure, dtype=float), np.asarray(scores, dtype=float)
    if exposure.ndim != 1 or len(exposure) < 2 or exposure.shape != scores.shape:
        raise ValueError('Expected equally sized one-dimensional series with at least two points.')
    if not np.isfinite(exposure).all() or not np.isfinite(scores).all() or np.any(exposure <= 0) or np.any(np.diff(exposure) <= 0):
        raise ValueError('Exposure must be positive, finite and strictly increasing; scores finite.')
    x = np.log10(exposure)
    return float(np.sum(np.diff(x) * (scores[:-1] + scores[1:])/2)/(x[-1]-x[0]))


def validate_reference(reference_dir, config_path):
    reference_dir = Path(reference_dir)
    configs = json.loads(Path(config_path).read_text())
    archived = json.loads((reference_dir/'full_dev_flops.json').read_text())
    checks = []

    def check(name, actual, expected, tolerance=1e-8):
        if not np.isfinite(actual) or abs(actual - expected) > tolerance:
            raise ValueError(f'{name}: {actual} != {expected} (tolerance {tolerance})')
        checks.append({'check':name, 'actual':actual, 'expected':expected, 'absolute_tolerance':tolerance})

    flops = {}
    for pair in archived['pairs']:
        name = pair['pair']
        result = reproduce_moments(pair, configs[name])
        for key, expected in pair['forward_gflops_per_sequence'].items():
            if key in result:
                check(f'{name}.{key}', result[key], expected)
        check(f'{name}.tokens_per_concept', result['allocated_tokens_per_concept'], pair['allocated_tokens_per_concept'])
        check(f'{name}.savings', result['padded_savings_percent'], pair['padded_savings_percent']['mean'])
        flops[name] = result

    rows = read_csv(reference_dir/'exposure_curves.csv')
    exposure = [float(row['exposure_words']) for row in rows]
    expected = json.loads((reference_dir/'exposure_summary.json').read_text())
    auc = {}
    for profile in expected['strict']:
        column = f'strict_{profile}_fast_macro'
        auc[profile] = normalized_log_auc(exposure, [float(row[column]) for row in rows])
        check(f'auc.{profile}', auc[profile], expected['strict'][profile]['normalized_log_exposure_auc'])
    auc['multilingual'] = normalized_log_auc(exposure, [float(row['multilingual_macro_zero_shot']) for row in rows])
    check('auc.multilingual', auc['multilingual'], expected['multilingual']['normalized_log_exposure_auc'])
    for row in rows:
        for profile in expected['strict']:
            tasks = ['blimp','supplement','ewok','entity_tracking','global_piqa','reading']
            mean = sum(float(row[f'strict_{profile}_{task}']) for task in tasks)/6
            check(f'macro.{profile}.{row["exposure_words"]}', mean, float(row[f'strict_{profile}_fast_macro']))
        check(f'macro.multilingual.{row["exposure_words"]}',
              sum(float(row[f'multilingual_{lang}_zero_shot']) for lang in ['eng','nld','zho'])/3,
              float(row['multilingual_macro_zero_shot']))

    lengths = read_csv(reference_dir/'segmentation_lengths.csv')
    for row in read_csv(reference_dir/'segmentation_summary.csv'):
        label = row['model']
        hist = [entry for entry in lengths if entry['model'] == label and entry['dataset'] == row['dataset']]
        count = sum(int(entry['count']) for entry in hist)
        check(f'{label}.complete_segments', count, int(row['complete_segments']))
        check(f'{label}.histogram_mean', sum(int(e['length'])*int(e['count']) for e in hist)/count, float(row['mean_segment_length']))
        check(f'{label}.token_ratio', int(row['tokens'])/int(row['concepts']), float(row['tokens_per_concept']))
        check(f'{label}.word_ratio', float(row['source_words'])/int(row['concepts']), float(row['source_words_per_concept']))
        for e in hist:
            check(f'{label}.fraction.{e["length"]}', int(e['count'])/count, float(e['fraction']))

    seen = set()
    for row in read_csv(reference_dir/'fmri_layer_sweep_aggregates.csv'):
        key = (row['domain'], row['model'], row['state_index'])
        if key in seen:
            raise ValueError(f'Duplicate layer-sweep key: {key}')
        seen.add(key)
        check('.'.join(key)+'.se', float(row['sd']) / np.sqrt(int(row['n'])), float(row['se']))
    return {'status':'passed', 'scope':'Aggregate arithmetic only; not independent raw-data replication.',
            'check_count':len(checks), 'flops':flops, 'normalized_log_auc':auc, 'checks':checks}
