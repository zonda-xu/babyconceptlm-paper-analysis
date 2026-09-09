# Release checklist

This directory is prepared for a standalone repository. The initial packaging step did not initialize Git or publish the package. The local MIT update on 2026-09-09 adds licensing files and metadata without committing or pushing changes.

## Before publication

- [x] Select MIT, add the full `LICENSE` with the maintainer-confirmed copyright holders, and update licensing documentation and package metadata.
- [ ] Review any applicable institutional, contributor, and upstream obligations before public distribution.
- [ ] Review third-party notices and permission to publish the included aggregate research results.
- [ ] Confirm manuscript title/authors in `CITATION.cff`; add the real archival DOI and final repository URL only when available.
- [ ] If promising fully automatic raw-input reproduction of all four public STRICT profiles, add verified public revisions/weight hashes for those profiles. The current pinned downloader covers the six matched artifacts only.
- [ ] Run both documented test suites in a fresh environment and preserve its package versions externally.
- [ ] Run `babyconceptlm-analysis reproduce` and inspect output plots and `validation.json`.
- [ ] Run `python scripts/check_release.py`; review the candidate files manually as well.
- [ ] Keep `inputs/`, `outputs/`, `checkpoints/`, model caches, raw corpora, participant rows and credentials out of version control. Do not force-add ignored runtime artifacts.
- [ ] Inspect `git diff --cached` in the eventual standalone repository before committing. Never initialize publication from the large parent training workspace.
- [ ] Upload only after the author approves the destination and visibility.

## Build a shareable source archive

```bash
python scripts/check_release.py --zip dist/babyconceptlm-paper-analysis.zip
```

The builder includes reviewed source/config/docs/tests/reference fixtures only, uses deterministic archive timestamps and rejects an existing ZIP destination. It never includes `.git` history or runtime outputs. A targeted pattern scan cannot guarantee absence of all sensitive content; it complements manual review.

## Scientific release status

Working offline aggregate checks and new seeded analysis tools are provided. The release does not claim a new end-to-end evaluation, training replication, raw-fMRI fit, runtime benchmark, segmentation-only ablation, or bitwise reproduction of undocumented bootstrap intervals. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact scope and the corrected new-analysis FLOPs interval statistic.
