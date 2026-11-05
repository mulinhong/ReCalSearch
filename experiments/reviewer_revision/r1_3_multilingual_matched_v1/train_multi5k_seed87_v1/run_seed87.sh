#!/usr/bin/env bash
set +e
set +o pipefail

cd /root/autodl-tmp/lora-code-embeddings-main/text2code || exit 2

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

OUT=/root/autodl-tmp/lora-code-embeddings-main/text2code/results/reviewer_revision/r1_3_multilingual_matched_v1/train_multi5k_seed87_v1
MODEL=/root/autodl-tmp/lora-code-embeddings-main/text2code/results/reviewer_revision/r1_3_multilingual_matched_v1/models/multi5k_seed87_v1

echo "TRAIN_START=$(date -Is)"
echo "SEED=87"
echo "R1_3_MULTI5K_SEED87=RUNNING"

/root/miniconda3/envs/lora/bin/python /root/autodl-tmp/lora-code-embeddings-main/text2code/results/reviewer_revision/r1_3_multilingual_matched_v1/train_multi5k_seed13_v5/run_frozen_trainer.py --train-file /root/autodl-tmp/lora-code-embeddings-main/text2code/results/reviewer_revision/r1_3_multilingual_matched_v1/multi5k_nomic_seed20261013.jsonl --output-dir /root/autodl-tmp/lora-code-embeddings-main/text2code/results/reviewer_revision/r1_3_multilingual_matched_v1/models/multi5k_seed87_v1 --model-name-or-path /root/autodl-tmp/hf_models/Qwen2.5-Coder-7B-Instruct --dtype float16 --trainable-dtype float32 --device-map cuda --epochs 1.0 --batch-size 1 --gradient-accumulation-steps 16 --lr 0.0001 --warmup-ratio 0.03 --max-length 1024 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --target-modules q_proj k_proj v_proj o_proj --gradient-checkpointing --seed 87 > "$OUT/train.log" 2>&1

RC=$?

echo "$RC" > "$OUT/train_exit_code.txt"

echo "TRAIN_END=$(date -Is)"
echo "TRAIN_RETURN_CODE=$RC"

if [ "$RC" -ne 0 ]; then
    echo "R1_3_MULTI5K_SEED87=TRAIN_FAIL_STOP"
    exit "$RC"
fi

python - "$MODEL" "$OUT" /root/autodl-tmp/lora-code-embeddings-main/text2code/results/reviewer_revision/r1_3_multilingual_matched_v1/models/multi5k_seed42_v1 <<'AUDIT'
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from safetensors import safe_open

model = Path(sys.argv[1])
out = Path(sys.argv[2])
parent = Path(sys.argv[3])

def gate(ok, message):
    if not ok:
        raise RuntimeError(message)

def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

names = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "verifier_config.json",
    "training_args.bin",
    "environment_manifest.json",
]

for name in names:
    p = model / name
    gate(p.is_file() and p.stat().st_size > 0, name)

c87 = json.loads(
    (model / "verifier_config.json").read_text()
)

c42 = json.loads(
    (parent / "verifier_config.json").read_text()
)

gate(int(c87["seed"]) == 87, "Incorrect seed")
gate(int(c87["num_training_examples"]) == 45000, "Example count")

c87_compare = dict(c87)
c87_compare["seed"] = 42

gate(
    c87_compare == c42,
    "Training configuration differs beyond seed"
)

tensors = 0
parameters = 0

with safe_open(
    str(model / "adapter_model.safetensors"),
    framework="pt",
    device="cpu",
) as f:
    for key in f.keys():
        x = f.get_tensor(key)
        gate(
            bool(np.isfinite(x.float().numpy()).all()),
            "Nonfinite tensor: " + key
        )
        tensors += 1
        parameters += x.numel()

gate(tensors == 224, "Adapter tensor count")
gate(parameters == 5046272, "Adapter parameter count")

gate(
    sha(model / "adapter_model.safetensors")
    != sha(parent / "adapter_model.safetensors"),
    "Seed87 weights unexpectedly identical to Seed42"
)

log = (out / "train.log").read_text(
    encoding="utf-8", errors="replace"
)

for marker in [
    "2813/2813",
    "Verifier adapter saved to:",
    "R13_RUNTIME_OPTIMIZER=adamw_torch",
    "R13_RUNTIME_DATA_SEED=None",
    "R13_RUNTIME_TRAINING_ARGUMENTS=PASS",
]:
    gate(marker in log, "Missing training marker: " + marker)

report = {
    "seed": 87,
    "training_completed": True,
    "model_configuration_parity": True,
    "adapter_tensors": tensors,
    "adapter_parameters": parameters,
    "adapter_all_finite": True,
    "model_files_sha256": {
        name: sha(model / name)
        for name in names
    },
    "training_args_pickle_deserialization": "NOT_CHECKED",
    "validation_mrr": None,
    "artifact_gate": "PASS",
}

path = out / "posttrain_artifact_audit.json"

gate(not path.exists(), "Audit report already exists")

path.write_text(
    json.dumps(report, indent=2) + "\n",
    encoding="utf-8"
)

print("SEED87_CONFIG_PARITY=PASS")
print("ADAPTER_TENSORS=", tensors)
print("ADAPTER_PARAMETERS=", parameters)
print("ADAPTER_ALL_FINITE=PASS")
print("SEED87_POSTTRAIN_ARTIFACT_GATE=PASS")
print("AUDIT_SHA256=", sha(path))
AUDIT

AUDIT_RC=$?

echo "$AUDIT_RC" > "$OUT/posttrain_exit_code.txt"

if [ "$AUDIT_RC" -eq 0 ]; then
    echo "R1_3_MULTI5K_SEED87=PASS"
else
    echo "R1_3_MULTI5K_SEED87=POSTTRAIN_FAIL_STOP"
fi

exit "$AUDIT_RC"
