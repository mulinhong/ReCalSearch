# Source coverage and exclusions

For the ESWA manuscript, this repository provides implementation sources and frozen statistical materials. See [Public reproduction scope](REPRODUCTION_SCOPE.md) for the distinction between self-contained statistical replay, archived summaries, and full experiment reruns.

## Included original sources

The release contains **217 preserved files before documentation** including original training/evaluation code, reviewer experiment scripts, frozen statistical results, and 12 compact TEST rank arrays. Selected paper-related files from the AutoDL `code_original_full/` directory are included without editing their bytes; excluded branches relate to separate exploratory projects and a malformed historical script.

Original source entrypoints:

- `code/train_qwen_verifier.py` and `code/eval_llm_verifier.py`
- `code_original_full/build_r1_1_four_source_qwen_data.sh`, `build_bm25_reviewer_groups.py`, `build_jina_reviewer_groups.py`, `build_qwen_verifier_data.py`, `audit_r1_1_four_sources.py`
- `code_original_full/build_r1_1_nomic_side_verifier_only.py`, `run_r1_1_jina_side_crossover.py`
- `experiments/reviewer_revision/hn_source_ablation/crossover/run_r1_1_final_statistics_v1.py`
- `baselines/coderankllm_integration/*.py` and `.sh`
- `analysis/paired_bootstrap.py`

Frozen result pointers:

- `experiments/reviewer_revision_results/hn_source_ablation/final_protocol/`
- `experiments/reviewer_revision_results/hn_source_ablation/crossover/r1_1_final_statistics_v1/r1_1_crossover_statistics.json`
- `experiments/coderankllm_frozen_results/recal_vs_coderank_paired_bootstrap.json`

## Not supplied

The package does **not** include model weights, adapters, original CodeSearchNet/CoSQA raw data, all candidate pools, all 2×2 per-query rank arrays, CodeRankLLM per-query reranking outputs, or upstream llm-reranker runtime. The frozen results provide archived experiment *summary data*, while the single self-contained bundled statistical replay uses the 12 Nomic-vs-ReCalSearch ranks.

Full training/inference scripts assume the historical AutoDL folder layout and exact checkpoint versions, which require separate environment configuration for another user's machine. Therefore do not describe the repository as an end-to-end reproducibility guarantee.

## Excluded from curated GitHub source

The author's uploaded **398-file original collector ZIP remains intact**. From its `code_original_full/` portion, 175 non-paper-related or malformed legacy source files are intentionally excluded from this reviewed release, including Work2 coupling, Safe-Swap, auxiliary predictors, and the pre-existing syntactically invalid `finetune_cosqa.py`. None was edited or rewritten. Their omission is to scope the publication to this manuscript, not to alter the research evidence.

No API secrets were detected by basic automated pattern scanning, but these checks are not a substitute for an author's manual privacy/licensing review.
