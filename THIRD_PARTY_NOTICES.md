# Sources and third-party notices

The segmentation analyzer, Hippocorpus preparation script, learning-curve collector, dataset packing helpers and boundary-only inference path are adapted from the BabyConceptLM research workspace. Exact source snapshots are recorded by SHA-256 in [provenance.json](docs/provenance.json). Adaptation is not a claim that these components have a newly granted license; see [LICENSE_STATUS.md](LICENSE_STATUS.md).

No external evaluation repository, pretrained model implementation, tokenizer asset or raw dataset is vendored. Optional checkpoint downloads can contain custom Python code and inherit the artifact's own notices; downloading does not execute that code.

Evaluation interfaces refer to [BabyLM's official evaluation repository](https://github.com/babylm-org/babylm-eval) and the [ChineseBabyLM pipeline at the recorded revision](https://github.com/chinese-babylm/chinese-babylm-pipeline-final/tree/f4437e1aaecf18ab5defb6b148857ef7d056e9d6). These are external dependencies, not code licensed by this release.

NumPy, Matplotlib, PyTorch, Transformers, Tokenizers, Hugging Face Hub and optional W&B are separately installed dependencies. Preserve their package-level notices and review the versions actually distributed in an environment.

Hippocorpus must be acquired from its [official Microsoft distribution](https://www.microsoft.com/en-us/download/details.aspx?id=105291); its dataset terms are separate from any loader's software license. BaikeQA is documented in the [original Chinese corpus repository](https://github.com/brightmart/nlp_chinese_corpus). Neither dataset is included here.
