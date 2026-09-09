"""Readable vector redraws from retained aggregate CSV data."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .reference import read_csv


def render(reference_dir, output_dir):
    reference_dir, output_dir = Path(reference_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size':10, 'pdf.fonttype':42, 'ps.fonttype':42})
    paths = []

    def save(fig, name):
        fig.savefig(output_dir/f'{name}.pdf', bbox_inches='tight')
        fig.savefig(output_dir/f'{name}.png', dpi=180, bbox_inches='tight')
        paths.append(f'{name}.pdf')
        plt.close(fig)

    rows = read_csv(reference_dir/'exposure_curves.csv')
    x = np.array([float(row['exposure_words'])/1e6 for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), constrained_layout=True)
    for profile in ['noDWA-242','DWA-242','noDWA-583','DWA-583']:
        is_242 = profile.endswith('242')
        axes[0].plot(x, [float(r[f'strict_{profile}_fast_macro']) for r in rows], label=profile,
                     color='#4878a8' if is_242 else '#b88f36',
                     linestyle='-' if profile.startswith('noDWA') else '--',
                     marker='o' if is_242 else 's', markevery=5, markersize=3)
    for language, color, style in [('eng','#4878a8','-'),('nld','#b88f36','--'),('zho','#c7773b',':')]:
        axes[1].plot(x, [float(r[f'multilingual_{language}_zero_shot']) for r in rows], label=language,
                     color=color, linestyle=style)
    axes[1].plot(x, [float(r['multilingual_macro_zero_shot']) for r in rows], color='black', label='macro')
    for ax, title in zip(axes, ['STRICT six-task fast macro', 'Multilingual zero-shot macro']):
        ax.set(xscale='log', xlabel='Words seen (millions)', ylabel='Score', title=title)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Recorded evaluation trajectories | 28 exposure anchors | recipe-level comparison', fontsize=11)
    save(fig, 'exposure_curves')

    rows = read_csv(reference_dir/'segmentation_lengths.csv')
    models = ['babyconceptLM-en-'+p for p in ['noDWA-242','DWA-242','noDWA-583','DWA-583']]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True, constrained_layout=True)
    for ax, model in zip(axes.flat, models):
        subset = [r for r in rows if r['model'] == model]
        ax.bar([int(r['length']) for r in subset], [100*float(r['fraction']) for r in subset], color='#4878a8')
        ax.set(title=model.replace('babyconceptLM-en-', ''), xlabel='Segment length (model tokens)', ylabel='Complete segments (%)', xticks=range(1,9), ylim=(0,85))
    fig.suptitle('English STRICT | 2,000 sampled narratives/model | uncensored segments', fontsize=11)
    save(fig, 'segmentation_lengths')

    rows = read_csv(reference_dir/'fmri_layer_sweep_aggregates.csv')
    for domain in sorted({r['domain'] for r in rows}):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True, constrained_layout=True)
        for ax, model, title in zip(axes, ['gpt_bert_unmatched','babyconceptlm_242_zh'], ['GPT-BERT (unmatched)', 'BabyConceptLM 2/4/2']):
            subset = sorted([r for r in rows if r['domain']==domain and r['model']==model], key=lambda r:int(r['state_index']))
            ax.bar([int(r['state_index']) for r in subset], [float(r['mean']) for r in subset],
                   yerr=[float(r['se']) for r in subset], capsize=2, color='#4878a8')
            ax.set(title=title, xlabel='Within-model returned-state index', ylabel='Correlation (r)', xticks=[int(r['state_index']) for r in subset])
        fig.suptitle(f'{domain}: six-participant descriptive sweep (mean ± SE)')
        save(fig, 'fmri_'+domain.lower())
    return paths
