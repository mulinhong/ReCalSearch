# CodeRankLLM official integration for ReCalSearch revision

This patch keeps the official CoRNStack/llm-reranker listwise implementation and changes only the integration layer needed for the ReCalSearch reviewer revision:

- frozen Nomic top-100 -> BEIR + `rank.tsv` expected by the official pipeline;
- all six CodeSearchNet languages instead of the upstream wrapper's hard-coded Ruby-only list;
- local `/root/autodl-tmp/hf_models/CodeRankLLM` model path;
- published main-baseline settings: top-100, window=10, step=5;
- fail-closed checks so the run cannot silently fall back to a different retriever/model.

The upstream listwise prompt and reranking algorithm stay in `llm-reranker/scripts/rerank_llm.py` and `scripts/utils/rank_listwise_os_llm.py`.

## 1. Populate the missing upstream dependency

```bash
bash clone_llm_reranker_mirror.sh
```

## 2. Keep the current `lora` environment untouched

Create/use a separate environment for the official reranker. The upstream repository requires vLLM and its `tevatron` package. Install these in the separate environment before the first smoke run.

## 3. Copy this patch into the CoRNStack checkout

Recommended location:

```text
/root/autodl-tmp/cornstack-official/recal_patch/
```

## 4. Run

First verify paths in `run_coderankllm_official.sh`, especially `DATA_ROOT` and `REV_ROOT`, then:

```bash
bash run_coderankllm_official.sh
```

Expected official output for each language:

```text
outputs_recal_nomic100/code_datasets/csn_<language>/rerank_100_llm_gen_num.json
```

and evaluation JSON under:

```text
evaluations_recal_nomic100/eval_results/
```
