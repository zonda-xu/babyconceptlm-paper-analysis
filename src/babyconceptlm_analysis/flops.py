"""Dense matrix-operation estimates; two shared backbone passes, memory C + T.

A multiply and an add count as two FLOPs. Not a runtime or backward estimate.
"""
import math

import numpy as np


def self_block(length, width, ffn):
    return 8 * length * width**2 + 4 * length**2 * width + 6 * length * width * ffn


def concept_flops(count, config, token_length=256):
    count = np.asarray(count, dtype=np.float64)
    if np.any(~np.isfinite(count)) or np.any(count < 1) or np.any(count > token_length):
        raise ValueError("Concept counts must be finite and in [1, token_length].")
    d, c = config['hidden_size'], config['concept_size']
    f, cf = config['intermediate_size'], config['concept_intermediate_size']
    rf, v = config['readout_intermediate_size'], config['vocab_size']
    scan, syntax = config['scan_size'], config['boundary_syntax_dim']
    t, memory = token_length, count + token_length
    cross = 4*t*d**2 + 4*memory*c*d + 4*t*memory*d + 6*t*d*rf
    pooling = 2*t*count*d + 2*count*d*c
    prefix = 2*t*t*d + 2*t*d*c
    boundary = 2*t*((4*d + 2*syntax)*scan + scan**2 + scan)
    total = (config['num_token_layers'] * self_block(t, d, f)
             + 2 * config['num_concept_layers'] * self_block(count, c, cf)
             + config['num_readout_layers'] * cross + 3*pooling
             + 2*count*c*c + prefix + boundary + 2*t*d*v)
    if config.get('include_residual_gate', False):
        gh = config['lm_concept_gate_hidden_size']
        total = total + 2*t*((3*d + 1)*gh + gh)
    return total


def token_flops(config, layers, token_length=256):
    d, f, v = config['hidden_size'], config['intermediate_size'], config['vocab_size']
    return layers * self_block(token_length, d, f) + 2*token_length*d*v


def reproduce_moments(pair, config):
    """Recover archived padded point estimate from pooled first/second moments.

    The archived SD uses ddof=1 and each batch (including a partial one) has
    equal weight. Intervals cannot be reconstructed from these moments.
    """
    t = pair['sequence_length']
    n = pair['random_group_permutations'] * math.ceil(
        pair['dev_sequences'] / pair['random_group_batch_size'])
    moments = pair['batch_max_concept_count']
    variance = moments['std']**2 * (n - 1) / n
    # F(C) is quadratic; only the two backbone self-attention passes add C².
    coefficient = 8 * config['num_concept_layers'] * config['concept_size']
    padded = float(concept_flops(moments['mean'], config, t)) + coefficient * variance
    control = token_flops(config, config['token_control_layers'], t)
    return {
        'allocated_tokens_per_concept': t / pair['concept_count']['mean'],
        'token_control': control / 1e9,
        'concept_ideal_at_mean_count': float(concept_flops(pair['concept_count']['mean'], config, t)) / 1e9,
        'concept_padded_mean': padded / 1e9,
        'padded_savings_percent': 100 * (control - padded) / control,
    }


def analyze_counts(counts, config, batch_size, permutations, seed, token_length=256):
    """New seeded permutation analysis. Repeat intervals use mean(F(C_B))."""
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 1 or not counts.size or np.any(counts != np.floor(counts)):
        raise ValueError('Expected a nonempty one-dimensional vector of integer counts.')
    concept_flops(counts, config, token_length)
    if batch_size < 1 or permutations < 2:
        raise ValueError('Positive batch size and at least two permutations required.')
    rng = np.random.Generator(np.random.PCG64(seed))
    repeat_flops = []
    control = token_flops(config, config['token_control_layers'], token_length)
    for _ in range(permutations):
        shuffled = counts[rng.permutation(len(counts))]
        maxima = np.maximum.reduceat(shuffled, np.arange(0, len(counts), batch_size))
        repeat_flops.append(float(concept_flops(maxima, config, token_length).mean()))
    savings = 100 * (control - np.array(repeat_flops)) / control
    return {
        'analysis': 'new_reanalysis', 'seed': seed, 'rng': 'PCG64',
        'sequences': len(counts), 'batch_size': batch_size, 'permutations': permutations,
        'weighting': 'equal batch weight, including final partial batch',
        'interval_statistic': 'percentiles across mean(F(C_B)) permutation repeats',
        'tokens_per_concept': token_length / float(counts.mean()),
        'ideal_gflops_at_mean_count': float(concept_flops(counts.mean(), config, token_length))/1e9,
        'padded_gflops': float(np.mean(repeat_flops))/1e9,
        'padded_savings_percent': float(savings.mean()),
        'padded_savings_permutation_interval': np.percentile(savings, [2.5, 97.5]).tolist(),
    }
