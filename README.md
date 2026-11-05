# ReCalSearch — Retriever-Aligned Code LLM Verification for Cost-Aware Neural Code Search

This repository contains original Work1 source files and frozen statistics associated with the ReCalSearch IST manuscript. It is a **source-and-evidence release**, not a single-command, end-to-end reproduction image. No model weights, raw CodeSearchNet/CoSQA datasets, original ranking caches for all experiments, or upstream CodeRankLLM repository are bundled.

## Start here

| Directory | Contents |
|---|---|
| `code/` | Main Qwen verifier training/evaluation, Nomic retrieval, UniXcoder control and runtime benchmark entry points (original source bytes) |
| `code_original_full/` | Selected **paper-related** original Work1 helper, negative-mining, dataset-building, cross-retriever and evaluation scripts |
| `experiments/reviewer_revision/` | Reviewer revision scripts, including the frozen Nomic↔Jina crossover statistics entry point |
| `experiments/reviewer_revision_results/` | Frozen manifest, four-source ablation, crossover and training protocols/results |
| `experiments/coderankllm_frozen_results/` | Frozen ReCalSearch-versus-CodeRankLLM comparison statistics |
| `baselines/coderankllm_integration/` | ReCalSearch wrappers around an **external** upstream listwise-reranking implementation |
| `analysis/`, `derived_ranks/` | CPU-only paired bootstrap and six-language Nomic/ReCalSearch frozen rank arrays |
| `statistics/` | Validation and bootstrap table inputs |
| `results/` | Numerical paper summary (not original raw inference traces for all experiments) |
| `metadata/` | A *current* environment snapshot, not a precise lock of all historical runs |

## Verified results represented in the repository

- **CodeRankLLM baseline:** 100,529 aligned TEST queries; ReCalSearch macro MRR 62.4464% vs CodeRankLLM 60.0006%, paired bootstrap Δ +2.4458 percentage points, 95% CI [2.2533, 2.6448] (10,000 replicates).
- **Four-source negative mining:** Random 60.6307±0.0984, BM25 59.9958±0.1928, Jina 62.4920±0.0729, Nomic 62.5010±0.0343 macro MRR across three training seeds in the same frozen Nomic pool protocol.
- **Nomic/Jina verifier-only crossover:** full-query source×pool interaction +1.9254 pp [1.6845, 2.1474]; shared-hit +2.7500 pp [2.4442, 3.0245] (10,000 resamples). These numbers are from the frozen `r1_1_crossover_statistics.json`, not from the main fused protocol.
- **Main dense retrieval baseline:** CPU bootstrap script and 12 frozen ranks reproduce ReCalSearch minus Nomic direct macro Δ +2.7705 pp [2.6043, 2.9280] (3,000 resamples).

The main verifier (LoRA r=16, alpha=32) and separate four-source controls (r=8, alpha=16) are different training configurations; comparisons within the four-source control use a common configuration. Follow the manuscript for the precise scope of each experiment.

## CPU statistical reproduction

Install NumPy in an isolated environment, then run from repository root:

```bash
python -m pip install 'numpy>=1.24,<3'
python analysis/paired_bootstrap.py --replicates 3000 --seed 20261104
```

This reproduces the **Nomic Direct vs ReCalSearch** frozen TEST ranking confidence intervals, not the crossover or CodeRankLLM CI. Other experiment statistics and source scripts are included, but full reruns require original per-query ranked candidates, datasets, model weights, dependencies and the historical directory layout. The original scripts frequently use absolute AutoDL paths; adapt paths in a separate working copy for a new machine. Do not write training outputs into this tracked repository.

## Evidence and source completeness

Read [`docs/SOURCE_COVERAGE.md`](docs/SOURCE_COVERAGE.md), [`docs/REPRO_NOTES.md`](docs/REPRO_NOTES.md) and [`docs/RELEASE_VERIFICATION.md`](docs/RELEASE_VERIFICATION.md). `SHA256SUMS` hashes all files except itself.

The full AutoDL collection also contained experimental Work2 integration, Safe-Swap and other exploratory scripts unrelated to this manuscript, plus a malformed historical CoSQA training script. These were **not** placed in this curated GitHub release; the author retains the original source-collection ZIP separately. Unmodified included historical source files can be compared with that original.

## Third-party rights and publication

The CodeRankLLM integration invokes upstream code from a **separate dependency checkout**; upstream model weights and code are not included here. Check authorship, third-party license terms, and data redistribution requirements before making this repository public. This archive deliberately contains **no LICENSE file** because licensing rights have not been established by the packaging process.

This package has not been pushed to GitHub automatically. See [`docs/GITHUB_UPLOAD.md`](docs/GITHUB_UPLOAD.md).
