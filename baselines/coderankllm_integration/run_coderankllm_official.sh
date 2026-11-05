#!/usr/bin/env bash
set -euo pipefail

CORNSTACK_ROOT=${CORNSTACK_ROOT:-/root/autodl-tmp/cornstack-official}
MODEL=${MODEL:-/root/autodl-tmp/hf_models/CodeRankLLM}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/lora-code-embeddings-main/data/CSN}
REV_ROOT=${REV_ROOT:-/root/autodl-tmp/lora-code-embeddings-main/text2code/results/reviewer_revision}
DATASET_DIR=${DATASET_DIR:-${CORNSTACK_ROOT}/datasets_recal_nomic100}
OUTPUT_DIR=${OUTPUT_DIR:-${CORNSTACK_ROOT}/outputs_recal_nomic100}
EVAL_DIR=${EVAL_DIR:-${CORNSTACK_ROOT}/evaluations_recal_nomic100}
LANGUAGES=(ruby javascript php python java go)

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$DATASET_DIR" "$OUTPUT_DIR" "$EVAL_DIR"

for lang in "${LANGUAGES[@]}"; do
  pool="${REV_ROOT}/pools/nomic_test/${lang}_test_nomic_top100.npz"
  [[ -f "$pool" ]] || { echo "Missing frozen candidate pool: $pool" >&2; exit 2; }
  python "$SCRIPT_DIR/prepare_coderankllm_beir.py" \
    --language "$lang" \
    --split test \
    --data-root "$DATA_ROOT" \
    --candidate-pool "$pool" \
    --dataset-dir "$DATASET_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --top-k 100 \
    --overwrite
done

python "$SCRIPT_DIR/rerank.py" \
  --dataset_dir "$DATASET_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --eval_dir "$EVAL_DIR" \
  --model "$MODEL" \
  --llm_reranker_dir "$CORNSTACK_ROOT/llm-reranker" \
  --languages "${LANGUAGES[@]}" \
  --top_k 100 \
  --window_size 10 \
  --step_size 5 \
  --context_size 32768 \
  --do_batched 1
