# Reproduction notes

This public source-and-evidence release accompanies the ReCalSearch manuscript prepared for submission to the Journal of Systems and Software (JSS). The definitive coverage table is [Public reproduction scope](REPRODUCTION_SCOPE.md).

## Self-contained statistical replay

The supported CPU-only command is:

```bash
python -m pip install 'numpy>=1.24,<3'
python analysis/paired_bootstrap.py --replicates 3000 --seed 20261104
```

With the included `derived_ranks/`, it reproduces six-language paired TEST intervals for Nomic direct retrieval vs ReCalSearch. Source file manifests and `SHA256SUMS` provide byte-level integrity checks.

## Frozen experiment results

- Four-source ablation: `experiments/reviewer_revision_results/hn_source_ablation/final_protocol/`.
- 2×2 Nomic↔Jina verifier-only crossover: `experiments/reviewer_revision_results/hn_source_ablation/crossover/r1_1_final_statistics_v1/r1_1_crossover_statistics.json` (10,000 resamples, frozen 100,529 TEST queries; 76,274 shared-hit).
- CodeRankLLM vs ReCalSearch: `experiments/coderankllm_frozen_results/recal_vs_coderank_paired_bootstrap.json` (10,000 paired resamples).
- Multi-seed VALID: `statistics/`.

The per-query rank sources for the two 10,000-replicate studies are NOT bundled. Their summary numbers must not be described as newly independently re-executed from this ZIP; only the Nomic-vs-ReCalSearch bootstrap is replayable as distributed.

Main ReCalSearch Qwen LoRA rank 16/alpha 32; four-source source-controlled experiments use rank 8/alpha 16. VALID seed variation is distinct from TEST query bootstrap. CoSQA lambda=5 is a reported sensitivity point, not an independently validated held-out optimum. Runtime comparisons have distinct measured scopes and are not an end-to-end matched-hardware speedup.

No GPU inference, retraining, or full TEST evaluation is necessary for the release audit.
