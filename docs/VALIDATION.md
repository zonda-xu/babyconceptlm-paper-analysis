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

At the initial 2026-09-08 packaging stage, the code license was undecided. The maintainer selected MIT on 2026-09-09 and confirmed the six camera-ready paper authors as copyright holders; the license and scope are now documented in `LICENSE` and `LICENSE_STATUS.md`. This licensing update does not change the earlier analysis results or imply a new full-experiment replication.

Fully automated checkpoint-driven repetition of the four public STRICT segmentation profiles still needs their public weight/revision pins; the six matched-control identities are already pinned. Third-party and aggregate-data publication rights require maintainer review. These boundaries are also stated in the README and release checklist.

## MIT update verification (2026-09-09)

- Rechecked all six copyright holders against the camera-ready manuscript and the author order in `CITATION.cff`.
- Verified SPDX `MIT` declarations in the package and citation metadata, the absence of unfilled license placeholders, and documentation links.
- Reran the 10 base and 12 optional unit tests; all 22 passed. Reran the offline aggregate validation without redrawing figures; all 346 checks passed.
- Verified that analysis code, model configurations, curated reference-file hashes, and the camera-ready manuscript source/PDF were unchanged.
- Built a wheel from an extracted source archive using an isolated build environment with `setuptools>=77.0.3`. Verified `License-Expression: MIT` and byte-identical bundled copies of `LICENSE`, `LICENSE_STATUS.md`, and `THIRD_PARTY_NOTICES.md`. Build dependencies were fetched into the temporary build environment; the original scientific environment was not upgraded.
- The source release scan and `git diff --check` passed. No commit or push was performed as part of this local licensing update.
