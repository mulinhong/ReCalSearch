#!/usr/bin/env bash
set +e
set +o pipefail

cd /root/autodl-tmp/lora-code-embeddings-main/text2code || exit 2

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

OUT=/root/autodl-tmp/lora-code-embeddings-main/text2code/results/reviewer_revision/r1_3_multilingual_matched_v1/train_mono5k_ruby_seed13_v1
MODEL=/root/autodl-tmp/lora-code-embeddings-main/text2code/results/reviewer_revision/r1_3_multilingual_matched_v1/models/mono5k_ruby_seed13_v1

echo "TRAIN_START=$(date -Is)"
echo "EXPERIMENT=R1_3_MONO5K_RUBY_SEED13"

/root/miniconda3/envs/lora/bin/python /root/autodl-tmp/lora-code-embeddings-main/text2code/results/reviewer_revision/r1_3_multilingual_matched_v1/train_multi5k_seed13_v5/run_frozen_trainer.py --train-file /root/autodl-tmp/lora-code-embeddings-main/text2code/results/reviewer_revision/hn_source_ablation/qwen_data/ruby_nomic_hn7_5k.jsonl --output-dir /root/autodl-tmp/lora-code-embeddings-main/text2code/results/reviewer_revision/r1_3_multilingual_matched_v1/models/mono5k_ruby_seed13_v1 --model-name-or-path /root/autodl-tmp/hf_models/Qwen2.5-Coder-7B-Instruct --dtype float16 --trainable-dtype float32 --device-map cuda --epochs 1.0 --batch-size 1 --gradient-accumulation-steps 16 --lr 0.0001 --warmup-ratio 0.03 --max-length 1024 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --target-modules q_proj k_proj v_proj o_proj --gradient-checkpointing --seed 13 > "$OUT/train.log" 2>&1
RC=$?

echo "$RC" > "$OUT/train_exit_code.txt"
echo "TRAIN_END=$(date -Is)"
echo "TRAIN_RETURN_CODE=$RC"

if [ "$RC" -ne 0 ]; then
    echo "MONO5K_RUBY_SEED13=TRAIN_FAIL_STOP"
    exit "$RC"
fi

python -B - "$MODEL" "$OUT" /root/autodl-tmp/lora-code-embeddings-main/text2code/results/reviewer_revision/r1_3_multilingual_matched_v1/models/multi5k_seed42_v1 /root/autodl-tmp/lora-code-embeddings-main/text2code/results/reviewer_revision/hn_source_ablation/qwen_data/ruby_nomic_hn7_5k.jsonl b53d8017e1141882c6ea1f8a3a80ce95e16f1ce3d75a1ac2d85f696144571c58 <<'AUDIT'
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from safetensors import safe_open

model = Path(sys.argv[1])
out = Path(sys.argv[2])
parent = Path(sys.argv[3])
data = Path(sys.argv[4])
expected_data_sha = sys.argv[5]

def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()

def need(ok, msg):
    if not ok:
        raise RuntimeError("POSTTRAIN_FAIL_STOP: " + msg)

need(sha(data) == expected_data_sha, "Training data SHA drift")

files = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "verifier_config.json",
    "training_args.bin",
    "environment_manifest.json",
]

for name in files:
    p = model / name
    need(p.is_file() and p.stat().st_size > 0, name)

cfg = json.loads((model / "verifier_config.json").read_text())
old = json.loads((parent / "verifier_config.json").read_text())

need(int(cfg["seed"]) == 13, "Seed")
need(int(cfg["num_training_examples"]) == 45000, "Rows")
need(Path(cfg["train_file"]).resolve() == data.resolve(), "Data path")

normalized = dict(cfg)
normalized["seed"] = old["seed"]
normalized["train_file"] = old["train_file"]

need(normalized == old, "Training parameter parity")

count = 0
params = 0

with safe_open(
    str(model / "adapter_model.safetensors"),
    framework="pt",
    device="cpu",
) as f:
    for key in f.keys():
        tensor = f.get_tensor(key).float().numpy()
        need(bool(np.isfinite(tensor).all()), "Nonfinite tensor")
        count += 1
        params += tensor.size

need(count == 224, "Adapter tensor count")
need(params == 5046272, "Adapter parameter count")

log = (out / "train.log").read_text(
    encoding="utf-8", errors="replace"
)

for marker in (
    "2813/2813",
    "Verifier adapter saved to:",
    "R13_RUNTIME_OPTIMIZER=adamw_torch",
    "R13_RUNTIME_DATA_SEED=None",
    "R13_RUNTIME_TRAINING_ARGUMENTS=PASS",
):
    need(marker in log, "Training marker: " + marker)

report = {
    "experiment": "R1_3_MONO5K_RUBY_SEED13",
    "status": "PASS",
    "seed": 13,
    "language": "ruby",
    "training_examples": 45000,
    "config_parity": True,
    "adapter_tensors": count,
    "adapter_parameters": params,
    "all_weights_finite": True,
    "data_sha256": sha(data),
    "files_sha256": {
        name: sha(model / name) for name in files
    },
    "validation_mrr": None,
}

report_path = out / "posttrain_artifact_audit.json"
need(not report_path.exists(), "Audit path exists")

report_path.write_text(
    json.dumps(report, indent=2) + "\n"
)

print("POSTTRAIN_CONFIG_PARITY=PASS")
print("ADAPTER_TENSORS=", count)
print("ADAPTER_PARAMETERS=", params)
print("ALL_WEIGHTS_FINITE=PASS")
print("MONO5K_RUBY_SEED13_ARTIFACT_GATE=PASS")
print("AUDIT_SHA256=", sha(report_path))
AUDIT

AUDIT_RC=$?

echo "$AUDIT_RC" > "$OUT/posttrain_exit_code.txt"

if [ "$AUDIT_RC" -eq 0 ]; then
    echo "MONO5K_RUBY_SEED13=PASS"
else
    echo "MONO5K_RUBY_SEED13=POSTTRAIN_FAIL_STOP"
fi

exit "$AUDIT_RC"
