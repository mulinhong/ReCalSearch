#!/usr/bin/env bash
set -uo pipefail

ROOT="/root/autodl-tmp/lora-code-embeddings-main/text2code"
BASE="$ROOT/results/reviewer_revision/r1_3_multilingual_matched_v1"

OUT="$BASE/train_multi5k_seed13_v5"
MODEL="$BASE/models/multi5k_seed13_v5"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

cd "$ROOT" || exit 2

if [ -e "$MODEL" ]; then
    echo "FAIL_STOP: Model directory exists."
    echo 2 > "$OUT/exit_code.txt"
    exit 2
fi

echo "TRAIN_START=$(date -Is)"
echo "TRAINER=FROZEN_ORIGINAL_WITH_RUNTIME_OVERRIDE"
echo "OPTIMIZER=adamw_torch"
echo "DATA_SEED=None"
echo "SEED=13"

# If the original CLI supports --optim, pass it
# explicitly too, so its own saved configuration
# agrees with the runtime override.
OPTIM_ARGS=()

if python - "$OUT/preflight_manifest.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as f:
    m = json.load(f)
sys.exit(0 if m["original_supports_optim_cli"] else 1)
PY
then
    OPTIM_ARGS=(--optim adamw_torch)
fi

python -u "$OUT/run_frozen_trainer.py" \
    --train-file \
      "$BASE/multi5k_nomic_seed20261013.jsonl" \
    --output-dir "$MODEL" \
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
    --dataloader-num-workers 2 \
    "${OPTIM_ARGS[@]}"

RC=$?

if [ "$RC" -eq 0 ]; then

python - "$MODEL" <<'PY'
import json
import sys
from pathlib import Path

import torch

p = Path(sys.argv[1])

required = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "verifier_config.json",
    "training_args.bin",
    "environment_manifest.json",
]

for name in required:
    if not (p / name).is_file():
        raise RuntimeError(
            "FAIL_STOP: missing trained artifact " + name
        )

cfg = json.loads(
    (p / "verifier_config.json").read_text()
)

adapter = json.loads(
    (p / "adapter_config.json").read_text()
)

args = torch.load(
    p / "training_args.bin",
    map_location="cpu",
    weights_only=False,
)

optim = str(getattr(args.optim, "value", args.optim))

assert optim == "adamw_torch"
assert args.data_seed is None
assert args.seed == 13

assert cfg["num_training_examples"] == 45000
assert cfg["max_length"] == 1024
assert cfg["gradient_accumulation_steps"] == 16
assert abs(cfg["learning_rate"] - 1e-4) < 1e-12

# The saved TrainingArguments are authoritative.
# The original trainer's optional summary field
# must not contradict the actual optimizer.
if "optim" in cfg:
    assert cfg["optim"] == "adamw_torch"

assert adapter["r"] == 8
assert adapter["lora_alpha"] == 16
assert set(adapter["target_modules"]) == {
    "q_proj", "k_proj", "v_proj", "o_proj"
}

print("TRAINED_ARTIFACT_INTEGRITY=PASS")
print("ACTUAL_OPTIMIZER_PARITY=PASS")
print("ACTUAL_DATA_SEED_PARITY=PASS")
PY

    CHECK_RC=$?

    if [ "$CHECK_RC" -ne 0 ]; then
        RC=$CHECK_RC
    fi

fi

echo "TRAIN_END=$(date -Is)"
echo "TRAIN_RETURN_CODE=$RC"

if [ "$RC" -eq 0 ]; then
    echo "R1_3_MULTI5K_SEED13_TRAINING=PASS"
else
    echo "R1_3_MULTI5K_SEED13_TRAINING=FAIL_STOP"
fi

echo "$RC" > "$OUT/exit_code.txt"
exit "$RC"
