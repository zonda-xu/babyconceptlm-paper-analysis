# BabyConceptLM paper analysis companion

[中文说明](README.zh-CN.md)

Portable analysis code for *BabyConceptLM: Exploring Dynamic Concept Computation under Restricted Language Exposure*. This prepared release corresponds to the manuscript snapshot recorded in [provenance.json](docs/provenance.json). It is an analysis companion, not a full training repository or a claim of independent replication.

The default workflow runs offline on CPU. It verifies retained aggregate results and redraws figures without downloading checkpoints, corpora, or brain data. Optional scripts accept explicitly supplied inputs for new analyses. Project-owned code and accompanying documentation are distributed under the [MIT License](LICENSE); see [licensing scope](LICENSE_STATUS.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

## Quick start

Use Python 3.11 for the locally tested environment. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
babyconceptlm-analysis reproduce --reference-dir examples/reference --output-dir outputs/reference
```

Package installation requires access to package distributions; the reproduction command itself requires no network. It writes `validation.json` and eight PDF/PNG redraws under the requested output directory: exposure curves, segmentation lengths, and six domain-specific fMRI layer sweeps. Set `MPLCONFIGDIR` to a writable cache directory if necessary.

## What is included

| Analysis | Included evidence | What the package does |
| --- | --- | --- |
| English/MULTI/Chinese full-dev FLOPs | Shape configurations, frozen full-dev aggregate moments | Recomputes ideal and padding-aware point estimates, including memory length `C_B + T` |
| Exposure-indexed learning curves | 28-point evaluation CSV and archived AUC summary | Verifies macro scores and log-exposure AUC; redraws curves |
| English segmentation audit | Four model summaries and uncensored length histograms | Checks ratios/distributions; redraws histograms; supplies checkpoint-driven analyzer |
| fMRI layer sweep | Six-participant aggregate means, SDs, SEs | Redraws the retained unmatched, descriptive comparison |
| Paired fMRI statistics | Synthetic example only | Exact sign-flip tests, paired effect size, seeded percentile bootstrap, explicit Holm family |
| Artifact identity | Six matched Hub repositories, full revisions, weight hashes | Optional pinned download and weight-hash verification |

Raw stimuli, participant-level real scores, narratives, checkpoints, tokenizer assets, credentials, training logs, and paper source/PDF are not distributed here. [Reproducibility scope](docs/REPRODUCIBILITY.md) identifies what requires external inputs and what cannot be recovered from aggregates.

## New paired-statistics analysis

The included data below are **synthetic and must not be cited as an experiment**:

```bash
babyconceptlm-analysis paired-stats \
  --input examples/fmri_synthetic.csv \
  --concept synthetic_concept --control synthetic_control \
  --seed 20260908 --samples 200000 --output outputs/synthetic_stats.json
```

Provide a CSV with exactly `model,task,participant,score` columns and unscaled correlations in `[-1,1]`. Model differences are multiplied by 100 by default. Duplicate or unequal participant sets fail validation. Use `--tasks Cognition Language` to define that two-task Holm family; omitting `--tasks` uses every task in the CSV. Run the six-domain family separately. A composite score must first be computed within each participant; averaging task p-values is invalid. New bootstrap intervals are not presented as the paper's archived intervals.

## Checkpoint-driven analyses (optional)

```bash
python -m pip install -e '.[inference]'
python -m unittest discover -s tests/inference -v
python scripts/download_checkpoint.py --model chinese_concept \
  --output-dir checkpoints/chinese_concept --include-weights
```

`requirements-inference-tested.txt` records the versions used for local optional-dependency tests; use the appropriate PyTorch distribution for your hardware. Only helper/preprocessing tests, not full GPU checkpoint evaluation, were rerun during release preparation.

The download command is the explicit network-enabled step. Omitting `--include-weights` retrieves metadata/code only. Inspect that code before passing the custom-code consent flags below. Corpus inputs must match the checkpoint's tokenizer and frozen dev split; arbitrary replacement data produce a new experiment.

```bash
python scripts/count_concepts.py --checkpoint checkpoints/chinese_concept \
  --data inputs/chinese_dev_tokenized.bin --device cuda --batch-size 32 \
  --accept-remote-code --output outputs/chinese_counts.csv
babyconceptlm-analysis flops-from-counts --input outputs/chinese_counts.csv \
  --pair chinese --batch-size 32 --permutations 200 --seed 20260721 \
  --output outputs/chinese_full_dev_reanalysis.json
```

For multilingual data, replace `--data` with `--manifest inputs/multilingual.json`. English/MULTI use random-group batch size 16; Chinese uses 32. Inference batch size and random-group batch size are separate settings. The count script writes a completion/hash sidecar; an interrupted CSV is not a full-dev result.

`flops-from-counts` checks that completion sidecar, sequence indices, CSV hash and checkpoint shapes by default. `--allow-unverified-input` is available only for explicitly labelled exploratory inputs. Shape/hash consistency alone does not certify the original corpus split or weight identity; verify those separately against the pinned manifests.

Prepare Hippocorpus from a separately acquired source CSV and the English STRICT tokenizer:

```bash
python scripts/prepare_hippocorpus.py --source_csv inputs/hcV3-stories.csv \
  --tokenizer inputs/strict_tokenizer.json --output_prefix outputs/hippocorpus
python scripts/analyze_segmentation.py --model english=checkpoints/english_public \
  --dataset eng=outputs/hippocorpus_tokenized.bin \
  --word_counts eng=outputs/hippocorpus_word_counts.pt \
  --word_boundary_labels eng=outputs/hippocorpus_word_boundaries.pt \
  --seq_length 256 --max_documents 2000 --document_sampling reservoir \
  --windows_per_document 1 --seed 42 --num_examples 0 \
  --accept_remote_code --output_dir outputs/segmentation
```

`english_public` here is a separately supplied public STRICT checkpoint, **not** the English matched-control reference. See [data and identity requirements](docs/DATA.md). The historical analyzer retains underscore-style flags. Defaults export no decoded examples and do not contact W&B. Runtime outputs can still contain local paths and document-level metrics: keep them private and curate aggregate exports explicitly.

To rebuild learning curves from evaluation outputs, supply all three paths:

```bash
python scripts/collect_exposure_curves.py \
  --strict-results-dir inputs/strict_results \
  --multilingual-results-dir inputs/multilingual_results \
  --multilingual-collator inputs/babylm-eval/multilingual/scripts/collate_results.py \
  --output-dir outputs/recollected_curves
```

The collector executes the supplied local collator; use the reviewed revision in [evaluation_revisions.json](configs/evaluation_revisions.json).

## Repository layout

```text
configs/            Public identities, frozen shapes, evaluation revisions
examples/reference/ Curated aggregate fixtures; no real participant rows
examples/           Clearly labelled synthetic paired-statistics example
src/                FLOPs, statistics, validation and plotting library
scripts/            Explicit-input analyses and release checker
tests/              Offline unit tests; optional inference/preprocessing tests
docs/               Data requirements, provenance and release checklist
```

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream dependencies and [CITATION.cff](CITATION.cff) for the manuscript citation. No DOI, publication venue, or final GitHub URL has been invented.

## License

Copyright (c) 2026 Yizhe Xu, Ming Song, Danni He, Jianghao Liu, Yitong Wang, and Qing Cai.

Project-owned code and accompanying documentation are licensed under the [MIT License](LICENSE). Third-party components, datasets, and model artifacts retain their respective licenses and access conditions; this release does not relicense them. See [LICENSE_STATUS.md](LICENSE_STATUS.md) for scope and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for sources. Please cite the manuscript when using this work in research; the citation request does not add conditions to the MIT License.

## Prepare an upload

```bash
python scripts/check_release.py
python scripts/check_release.py --zip dist/babyconceptlm-paper-analysis.zip
```

The ZIP uses a file allowlist, excludes runtime outputs and weights, rejects personal paths/common token patterns, and verifies reference-data hashes. It will not overwrite an existing archive. It does not run Git, commit, push, or upload anything. Follow [the release checklist](docs/RELEASE_CHECKLIST.md), including third-party and data-permission review, before publishing.
