"""Paired participant statistics, never pooling participants as training seeds."""
import csv
from collections import defaultdict

import numpy as np


def holm(pvalues):
    values = np.asarray(pvalues, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError('p-values must lie in [0, 1].')
    order = np.argsort(values, kind='stable')
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    return adjusted.tolist()


def paired_test(difference, seed, samples=200000):
    diff = np.asarray(difference, dtype=float)
    if diff.ndim != 1 or not 2 <= len(diff) <= 20 or not np.isfinite(diff).all():
        raise ValueError('Exact test requires 2–20 finite paired participants.')
    if samples < 100:
        raise ValueError('At least 100 bootstrap samples required.')
    n = len(diff)
    observed = float(diff.mean())
    extreme = 0
    for start in range(0, 2**n, 8192):
        masks = np.arange(start, min(start + 8192, 2**n), dtype=np.uint64)
        signs = 2 * ((masks[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(float) - 1
        extreme += int(np.count_nonzero(np.abs(signs @ diff / n) >= abs(observed) - 1e-12))
    rng = np.random.Generator(np.random.PCG64(seed))
    boot = np.empty(samples)
    for start in range(0, samples, 10000):
        size = min(10000, samples - start)
        boot[start:start+size] = diff[rng.integers(0, n, size=(size, n))].mean(axis=1)
    sd = float(diff.std(ddof=1))
    return {'n': n, 'delta': observed, 'dz': observed/sd if sd else None,
            'dz_note': None if sd else 'Undefined: zero paired-difference variance.',
            'p_exact_two_sided': extreme / 2**n,
            'bootstrap_percentile_95': np.percentile(boot, [2.5, 97.5], method='linear').tolist()}


def analyze_csv(path, concept, control, seed, samples=200000, scale=100.0, tasks=None):
    groups = defaultdict(dict)
    with open(path, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != {'model', 'task', 'participant', 'score'}:
            raise ValueError('CSV needs exactly model,task,participant,score columns.')
        for row in reader:
            if tasks is not None and row['task'] not in tasks:
                continue
            if row['model'] not in {concept, control}:
                continue
            key = (row['task'], row['model'])
            participant = row['participant']
            score = float(row['score'])
            if not participant or not row['task'] or not row['model'] or not np.isfinite(score) or not -1 <= score <= 1:
                raise ValueError('Invalid label or correlation; scores must be unscaled correlations in [-1, 1].')
            if participant in groups[key]:
                raise ValueError(f'Duplicate participant within {key}.')
            groups[key][participant] = score
    if concept == control or not groups or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Distinct model names, nonempty data and positive scale required.')
    present = {task for task, _ in groups}
    if tasks is not None and set(tasks) != present:
        raise ValueError('Some requested tasks are missing from the input.')
    tasks = sorted(present)
    results = {}
    for task in tasks:
        left, right = groups[(task, concept)], groups[(task, control)]
        if not left or set(left) != set(right):
            raise ValueError(f'Unequal or missing paired participant sets: {task}.')
        participants = sorted(left)
        diff = [(left[p] - right[p])*scale for p in participants]
        results[task] = paired_test(diff, seed, samples)
    adjusted = holm([results[t]['p_exact_two_sided'] for t in tasks])
    for task, value in zip(tasks, adjusted):
        results[task]['p_holm_selected_family'] = value
    return {'analysis': 'new_reanalysis_not_archived_bootstrap_intervals',
            'concept': concept, 'control': control, 'seed': seed, 'rng': 'PCG64',
            'bootstrap_samples': samples, 'scale': scale,
            'rng_policy': 'same seed restarted per task; intervals are task-specific',
            'holm_family': tasks, 'tasks': results}
