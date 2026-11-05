#!/usr/bin/env bash

set -uo pipefail

ROOT="/root/autodl-tmp/lora-code-embeddings-main/text2code"
BASE="$ROOT/results/reviewer_revision/r1_3_multilingual_matched_v1"
OUT="$BASE/train_multi5k_seed13_v2"
MODEL_OUT="$BASE/models/multi5k_seed13"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

cd "$ROOT" || exit 2

if [ -e "$MODEL_OUT" ]; then
    echo "FAIL_STOP: Model directory already exists."
    echo "2" > "$OUT/exit_code.txt"
    exit 2
fi

echo "TRAIN_START=$(date -Is)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SEED=13"
echo "OPTIMIZER=adamw_torch"
echo "TRAINING_EXAMPLES=45000"

python -u \
  "$BASE/token_optimizer_freeze_v1/train_qwen_verifier_r13.py" \
  --train-file "$BASE/multi5k_nomic_seed20261013.jsonl" \
  --output-dir "$MODEL_OUT" \
  --model-name-or-path \
    /root/autodl-tmp/hf_models/Qwen2.5-Coder-7B-Instruct \
  --dtype float16 \
  --trainable-dtype float32 \
  --device-map cuda \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --lr 1e-4 \
  --warmup-ratio 0.03 \
  --max-length 1024 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --target-modules q_proj k_proj v_proj o_proj \
  --gradient-checkpointing \
  --seed 13 \
  --logging-steps 20 \
  --save-steps 500 \
  --save-total-limit 2 \
  --dataloader-num-workers 2

RC=$?

if [ "$RC" -eq 0 ]; then

    python - "$MODEL_OUT" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])

required = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "verifier_config.json",
    "training_args.bin",
    "environment_manifest.json",
]

for name in required:
    assert (p / name).is_file(), f"Missing {name}"

cfg = json.loads((p / "verifier_config.json").read_text())
adapter = json.loads((p / "adapter_config.json").read_text())

assert cfg["num_training_examples"] == 45000
assert cfg["seed"] == 13
assert cfg["max_length"] == 1024
assert cfg["optim"] == "adamw_torch"
assert cfg["learning_rate"] == 1e-4
assert cfg["gradient_accumulation_steps"] == 16
assert adapter["r"] == 8
assert adapter["lora_alpha"] == 16
assert set(adapter["target_modules"]) == {
    "q_proj", "k_proj", "v_proj", "o_proj"
}

print("TRAINED_ARTIFACT_INTEGRITY = PASS")
print("TRAINING_CONFIG_PARITY = PASS")
PY

    CHECK_RC=$?

    if [ "$CHECK_RC" -ne 0 ]; then
        RC=$CHECK_RC
    fi
fi

echo "TRAIN_END=$(date -Is)"
echo "TRAIN_RETURN_CODE=$RC"

if [ "$RC" -eq 0 ]; then
    echo "R1_3_MULTI5K_SEED13_TRAINING = PASS"
else
    echo "R1_3_MULTI5K_SEED13_TRAINING = FAIL_STOP"
fi

echo "$RC" > "$OUT/exit_code.txt"

exit "$RC"
