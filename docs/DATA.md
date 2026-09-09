# Inputs, model identities and provenance

## Curated reference data

`examples/reference/` contains whitelisted numeric aggregates from retained research artifacts. `docs/provenance.json` records source SHA-256 hashes and every reference/configuration file hash. Source locations are repository-relative identifiers, not hardcoded runtime dependencies. The original `summary.json` files with decoded narratives, local checkpoint paths and per-document records are deliberately excluded.

`examples/fmri_synthetic.csv` is generated illustrative data. Every participant and model label is prefixed `synthetic_`; it is never evidence for the paper.

## Public checkpoints

`configs/models.json` identifies six frozen matched concept/TA1 artifacts by full Hub commit and weight SHA-256. The download helper validates the retained `pytorch_model.bin` hash when weights are requested. Download metadata can be inspected without loading the model.

The English public STRICT segmentation/learning-curve profiles have different roles:

| Profile | Public repository |
| --- | --- |
| noDWA-242 | `zondaxyz/babyconceptLM-242-en-noDWA` |
| DWA-242 | `zondaxyz/babyconceptLM-242-en` |
| noDWA-583 | `zondaxyz/babyconceptLM-583-en-noDWA` |
| DWA-583 | `zondaxyz/babyconceptLM-583-en` |

The prepared downloader deliberately covers only the six matched identities with full revisions in the retained matched manifest. To repeat the four public STRICT segmentation audits, supply the original frozen checkpoints and compare their configuration/tokenizer fingerprints with your retained audit records; do not select a mutable Hub `main` revision or substitute a matched-control checkpoint. Additional public-profile weight/revision pinning remains a maintainer release task if fully automatic raw-input reproduction is desired.

## Corpora and token inputs

English/multilingual evaluation inputs follow [BabyLM's official evaluation repository](https://github.com/babylm-org/babylm-eval). Pin the full commits in `configs/evaluation_revisions.json`. Raw corpus licensing and access are not supplied by this code package.

Chinese corpus acquisition starts from the [original brightmart corpus repository](https://github.com/brightmart/nlp_chinese_corpus). **BaikeQA is the `baike2018qa` section, not the `wiki2019zh` section.** Acquiring the source collection alone does not recreate the exact 102M-word selection, tokenizer or held-out split. Use the retained frozen prepared input and verify its provenance.

Hippocorpus V3 is available from the [official Microsoft download page](https://www.microsoft.com/en-us/download/details.aspx?id=105291). Acquire it under its distribution terms, then supply `hcV3-stories.csv` and the original English STRICT BPE tokenizer. The preparation script expects `AssignmentId`, `memType`, `story`, preserves case/punctuation, normalizes whitespace and NFKC, rejects URLs/emails by default and deduplicates IDs/normalized text. Publication reference: [Sap et al., ACL 2020](https://aclanthology.org/2020.acl-main.178/). Generated index files may retain source identifiers; do not publish them without a separate review.

The frozen paper preprocessing yielded 6,854 stories, 1,799,978 whitespace words and 2,135,141 BPE tokens. These counts are checks for the original source/tokenizer, not targets to impose on a different dataset.

Token inputs are a PyTorch-saved list of one-dimensional integer tensors, or the original tensor-shard manifest. The loader uses `weights_only=True` and does not silently retry unrestricted pickle loading. Only use trusted files. Dataset packing inserts EOS between nonempty documents when the preceding document does not already end in EOS; it does not append an extra EOS after the last document. It pads the final partial sequence and does not pad to a distributed world-size multiple for analysis.

Multilingual manifests use relative paths:

```json
{
  "format": "multi1_multilingual_manifest_v1",
  "languages": [
    {"name": "eng", "id": 0, "validation_path": "eng_dev.bin"},
    {"name": "nld", "id": 1, "validation_path": "nld_dev.bin"},
    {"name": "zho", "id": 2, "validation_path": "zho_dev.bin"}
  ]
}
```

Each language is packed independently. Optional syntax/word sidecars must have exactly aligned document and token order. Preserve hashes of every shard, tokenizer and sidecar, not merely the top-level manifest.

## Brain data and evaluation artifacts

Use the [ChineseBabyLM pipeline at the pinned revision](https://github.com/chinese-babylm/chinese-babylm-pipeline-final/tree/f4437e1aaecf18ab5defb6b148857ef7d056e9d6) for upstream feature extraction/encoding and dataset access instructions. This release does not redistribute BOLD files, stimuli, participant scores or access credentials, and it does not assert that all upstream datasets have unrestricted redistribution rights.

The matched discourse tests used 12 participants; the word-level comparison used 11; the retained descriptive layer sweep used six. Keep these populations separate. New paired-statistics input uses unscaled correlations and pseudonymous participant keys. The output omits individual rows, but that does not authorize publication of the input file.

LTP provenance retained in the manuscript is `ltp==4.2.14`, model `LTP/small`, revision `dea87bf0da3c5d054bbd426a78d7fa9efd347799`. The revision was recovered retrospectively rather than recorded in the original preprocessing lock. LTP is not needed for the offline aggregate workflow. Installing it anew does not prove exact reproduction of the original Chinese preprocessing.
