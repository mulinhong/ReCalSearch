#!/usr/bin/env bash
set +e
set +o pipefail

cd /root/autodl-tmp/lora-code-embeddings-main/text2code || exit 2
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

OUT=/root/autodl-tmp/lora-code-embeddings-main/text2code/results/reviewer_revision/r1_3_multilingual_matched_v1/train_multi5k_seed42_v1
MODEL=/root/autodl-tmp/lora-code-embeddings-main/text2code/results/reviewer_revision/r1_3_multilingual_matched_v1/models/multi5k_seed42_v1

echo "TRAIN_START=$(date -Is)"
echo "R1_3_MULTI5K_SEED42=RUNNING"

/root/miniconda3/envs/lora/bin/python /root/autodl-tmp/lora-code-embeddings-main/text2code/results/reviewer_revision/r1_3_multilingual_matched_v1/train_multi5k_seed13_v5/run_frozen_trainer.py --train-file /root/autodl-tmp/lora-code-embeddings-main/text2code/results/reviewer_revision/r1_3_multilingual_matched_v1/multi5k_nomic_seed20261013.jsonl --output-dir /root/autodl-tmp/lora-code-embeddings-main/text2code/results/reviewer_revision/r1_3_multilingual_matched_v1/models/multi5k_seed42_v1 --model-name-or-path /root/autodl-tmp/hf_models/Qwen2.5-Coder-7B-Instruct --dtype float16 --trainable-dtype float32 --device-map cuda --epochs 1.0 --batch-size 1 --gradient-accumulation-steps 16 --lr 0.0001 --warmup-ratio 0.03 --max-length 1024 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --target-modules q_proj k_proj v_proj o_proj --gradient-checkpointing --seed 42 > "$OUT/train.log" 2>&1
RC=$?

echo "$RC" > "$OUT/train_exit_code.txt"
echo "TRAIN_END=$(date -Is)"
echo "TRAIN_RETURN_CODE=$RC"

if [ "$RC" -ne 0 ]; then
    echo "R1_3_MULTI5K_SEED42=FAIL_STOP"
    exit "$RC"
fi

python - "$MODEL" "$OUT" <<'AUDIT'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

model = Path(sys.argv[1])
out = Path(sys.argv[2])

required = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "verifier_config.json",
    "training_args.bin",
    "environment_manifest.json",
]

for name in required:
    assert (model / name).is_file(), f"Missing: {name}"

config = json.loads((model / "verifier_config.json").read_text())

assert int(config["seed"]) == 42
assert int(config["num_training_examples"]) == 45000

p = model / "adapter_model.safetensors"
count = 0
parameters = 0

with safe_open(str(p), framework="pt", device="cpu") as f:
    for key in f.keys():
        x = f.get_tensor(key)
        assert bool(np.isfinite(x.numpy()).all()), key
        count += 1
        parameters += x.numel()

assert count == 224, count
assert parameters == 5046272, parameters

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

report = {
    "seed": 42,
    "training_finished": True,
    "adapter_tensors": count,
    "adapter_parameters": parameters,
    "adapter_all_finite": True,
    "files": {name: sha(model / name) for name in required},
    "training_args_pickle_not_loaded": True,
    "validation_mrr": None,
    "posttrain_artifact_gate": "PASS",
}

report_path = out / "posttrain_artifact_audit.json"
report_path.write_text(json.dumps(report, indent=2) + "\n")

print("ADAPTER_TENSORS=", count)
print("ADAPTER_PARAMETERS=", parameters)
print("R1_3_MULTI5K_SEED42_POSTTRAIN_ARTIFACT_GATE=PASS")
print("POSTTRAIN_AUDIT_SHA256=", sha(report_path))
AUDIT

AUDIT_RC=$?
echo "$AUDIT_RC" > "$OUT/posttrain_exit_code.txt"

if [ "$AUDIT_RC" -eq 0 ]; then
    echo "R1_3_MULTI5K_SEED42=PASS"
else
    echo "R1_3_MULTI5K_SEED42=POSTTRAIN_AUDIT_FAIL_STOP"
fi

exit "$AUDIT_RC"
