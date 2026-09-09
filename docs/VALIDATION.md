# Release validation

Prepared on 2026-09-08. This note concerns the code package, not a new validation of every scientific claim in the manuscript.

## Checks performed

- Base unit suite: 10 tests covering reference arithmetic, AUC, quadratic FLOPs moments, permutation determinism, exact sign-flip tests, Holm adjustment, paired-input validation, and count-file integrity checks.
- Optional suite: 12 tests covering synthetic Hippocorpus preprocessing, aligned sidecars, segmentation classification/bootstrap, tokenizer fallback, final partial batches and language-separated packing.
- Offline `reproduce`: 346 aggregate checks; three FLOPs pairs, five AUC values, task macro arithmetic, segment distributions/ratios, and 138 layer-sweep SE identities.
- Synthetic paired-statistics CLI: seeded 200,000-draw bootstrap and exact tests complete; outputs labelled synthetic/new reanalysis.
- All shipped Python files compile; optional entry points expose `--help` without loading weights.
- A source ZIP was extracted outside the research workspace, installed editable without package-index access, and exercised without parent-repository imports. Its virtual environment reused installed scientific dependencies through `--system-site-packages`; this was not a network-clean reinstall of every dependency.
- PDF/PNG redraws were generated. Exposure curves, all four segmentation panels and a representative fMRI domain were visually inspected for units, labels, denominator context and readability.
- Allowlisted release scan and reference-file SHA-256 verification pass. The scanner is a targeted check, not a guarantee against every possible secret or rights issue.

Local versions: Python 3.11.8; NumPy 1.26.4; Matplotlib 3.8.4; PyTorch 2.4.0; Transformers 4.44.0; Tokenizers 0.19.1; Hugging Face Hub 0.24.5. The included GitHub workflow has been configured but was not executed on GitHub during preparation.

## Commands

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s tests/inference -v
python -m compileall -q src scripts tests
babyconceptlm-analysis reproduce --reference-dir examples/reference --output-dir outputs/reference
babyconceptlm-analysis paired-stats --input examples/fmri_synthetic.csv \
  --concept synthetic_concept --control synthetic_control --seed 20260908 \
  --samples 200000 --output outputs/synthetic_stats.json
python scripts/check_release.py
```

## Not rerun / remaining release decisions

No full-checkpoint GPU audit, model retraining, raw-BOLD encoding, dataset download, new matched ablation, latency benchmark or Hugging Face download was performed for this packaging task. Public model revisions/hashes are copied from the retained verified identity manifest rather than newly asserted from a mutable Hub listing.

The original code license still needs author approval. Fully automated checkpoint-driven repetition of the four public STRICT segmentation profiles also needs their public weight/revision pins; the six matched-control identities are already pinned. Third-party and aggregate-data publication rights require maintainer review. These boundaries are also stated in the README and release checklist.
