#!/usr/bin/env bash
set -euo pipefail

CORNSTACK_ROOT=${CORNSTACK_ROOT:-/root/autodl-tmp/cornstack-official}
PATCH_DIR=${PATCH_DIR:-${CORNSTACK_ROOT}/recal_patch}

MODEL=${MODEL:-/root/autodl-tmp/hf_models/CodeRankLLM}

DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/lora-code-embeddings-main/data/CSN}

POOL=${POOL:-/root/autodl-tmp/lora-code-embeddings-main/text2code/results/reviewer_revision/pools/nomic_test/python_test_nomic_top100.npz}

DATASET_DIR=${DATASET_DIR:-${CORNSTACK_ROOT}/datasets_recal_smoke3}
OUTPUT_DIR=${OUTPUT_DIR:-${CORNSTACK_ROOT}/outputs_recal_smoke3}
EVAL_DIR=${EVAL_DIR:-${CORNSTACK_ROOT}/evaluations_recal_smoke3}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0

rm -rf \
  "$DATASET_DIR" \
  "$OUTPUT_DIR" \
  "$EVAL_DIR"

mkdir -p \
  "$DATASET_DIR" \
  "$OUTPUT_DIR" \
  "$EVAL_DIR"

python "$PATCH_DIR/prepare_coderankllm_beir.py" \
  --language python \
  --split test \
  --data-root "$DATA_ROOT" \
  --candidate-pool "$POOL" \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --top-k 100 \
  --query-offset 0 \
  --max-queries 3 \
  --overwrite

python "$PATCH_DIR/rerank.py" \
  --dataset_dir "$DATASET_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --eval_dir "$EVAL_DIR" \
  --model "$MODEL" \
  --llm_reranker_dir "$CORNSTACK_ROOT/llm-reranker" \
  --languages python \
  --top_k 100 \
  --window_size 10 \
  --step_size 5 \
  --context_size 32768 \
  --do_batched 1
