import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(
    "/root/autodl-tmp/lora-code-embeddings-main/text2code"
)

BASE = ROOT / (
    "results/reviewer_revision/"
    "r1_3_multilingual_matched_v1"
)

OUT = BASE / "train_multi5k_seed13_v5"

def gate(ok, message):
    if not ok:
        raise RuntimeError("FAIL_STOP: " + message)

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()

def check(path, expected):
    gate(path.is_file(), f"Missing file: {path}")
    gate(
        sha256(path) == expected,
        f"SHA mismatch: {path}",
    )
    print("SHA256_PASS =", path.name)

print("=" * 90)
print("WORK1 R1-3 MULTI5K SEED13 V5 PREFLIGHT")
print("=" * 90)

original = ROOT / "train_qwen_verifier.py"

check(
    original,
    "ec0a1307a2b1d5e2a283fa3527582b487761e388fbda26fa6a1b5b7582b487761e388fbda26fa6a1b5b758d3e2c07"
    if False else
    "ec0a1307a2b1d5e2a283fa3527582b487761e388fbda26fa6a1b5b758d3e2c07",
)

check(
    BASE / "multi5k_nomic_seed20261013.jsonl",
    "bcbd58d005cf227a52c9afed0912983340ce1b30888bd32c42643a214eb4bf3b",
)

budget_file = (
    BASE / "token_optimizer_freeze_v1/"
    "token_budget_and_trainer_freeze.json"
)

check(
    budget_file,
    "accc8292d8c869ade78cf0d3e5db927bd9f1377d8f6794cbb93a3d9eb05dffac",
)

check(
    BASE / "final_pretrain_gate_v2/"
    "final_pretrain_summary.json",
    "a1897028d037c9d84bfa74285691b0327501aae12b7bc384aa069e9dd1e5742c",
)

budget = json.loads(budget_file.read_text())

gate(
    budget["multi5k_total"]["examples"] == 45000,
    "Example count mismatch",
)

gate(
    budget["multi5k_total"]["effective_tokens"]
    == 8126238,
    "Token budget mismatch",
)

print()
print("[1] HISTORICAL TRAINING CONFIGURATION")

for seed in (13, 42, 87):

    path = (
        ROOT / "trained_verifiers/"
        "reviewer_r1_1_hn_source_ablation/"
        f"nomic_seed{seed}/training_args.bin"
    )

    gate(
        path.is_file(),
        f"Missing historical training_args seed={seed}",
    )

    args = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    optim = str(
        getattr(args.optim, "value", args.optim)
    )

    gate(
        optim == "adamw_torch",
        f"Historical optimizer mismatch seed={seed}",
    )

    gate(
        args.data_seed is None,
        f"Historical data_seed mismatch seed={seed}",
    )

    gate(
        args.gradient_accumulation_steps == 16
        and args.per_device_train_batch_size == 1,
        f"Historical batch mismatch seed={seed}",
    )

    print(
        f"SEED={seed} OPTIM={optim} "
        "DATA_SEED=None PASS"
    )

print()
print("[2] SOURCE AND WRAPPER COMPILATION")

wrapper = OUT / "run_frozen_trainer.py"

for path in (original, wrapper):

    source = path.read_text(encoding="utf-8")

    ast.parse(source)
    compile(source, str(path), "exec")

    print("PYTHON_COMPILE_PASS =", path.name)

tree = ast.parse(original.read_text(encoding="utf-8"))

training_calls = [
    x for x in ast.walk(tree)
    if isinstance(x, ast.Call)
    and isinstance(x.func, ast.Name)
    and x.func.id == "TrainingArguments"
]

gate(
    len(training_calls) == 1,
    "Original TrainingArguments call count mismatch",
)

print("ORIGINAL_TRAINER_AST=PASS")

print()
print("[3] CUDA")

gate(
    os.getenv("CUDA_VISIBLE_DEVICES") == "0",
    "CUDA visibility incorrect",
)

gate(
    torch.cuda.is_available(),
    "CUDA unavailable",
)

free_bytes, total_bytes = torch.cuda.mem_get_info(0)

print("GPU =", torch.cuda.get_device_name(0))
print(
    "GPU_FREE_GIB =",
    round(free_bytes / 1024**3, 3)
)

gate(
    free_bytes >= 16 * 1024**3,
    "Insufficient GPU memory",
)

model_path = Path(
    "/root/autodl-tmp/hf_models/"
    "Qwen2.5-Coder-7B-Instruct"
)

gate(model_path.is_dir(), "Base model missing")

print()
print("[4] WRAPPER CLI TEST")

result = subprocess.run(
    [
        sys.executable,
        str(wrapper),
        "--help",
    ],
    cwd=str(ROOT),
    capture_output=True,
    text=True,
    timeout=60,
)

gate(
    result.returncode == 0,
    "Wrapper --help failed:\n"
    + result.stderr[-2500:],
)

help_text = result.stdout

for required in (
    "--train-file",
    "--output-dir",
    "--seed",
    "--gradient-accumulation-steps",
    "--gradient-checkpointing",
):
    gate(
        required in help_text,
        f"Missing trainer CLI option: {required}",
    )

supports_optim_cli = "--optim" in help_text

print(
    "ORIGINAL_SUPPORTS_OPTIM_CLI =",
    supports_optim_cli,
)

(OUT / "cli_help.log").write_text(
    help_text,
    encoding="utf-8",
)

manifest = {
    "protocol": "work1_r1_3_multi5k_seed13_v5",
    "method": "runtime_training_arguments_override",
    "original_trainer_sha256": sha256(original),
    "wrapper_sha256": sha256(wrapper),
    "dataset_sha256": sha256(
        BASE / "multi5k_nomic_seed20261013.jsonl"
    ),
    "optimizer": "adamw_torch",
    "data_seed": None,
    "seed": 13,
    "epochs": 1,
    "training_examples": 45000,
    "effective_tokens": 8126238,
    "original_supports_optim_cli": supports_optim_cli,
    "training_started": False,
}

manifest_path = OUT / "preflight_manifest.json"

gate(not manifest_path.exists(), "Manifest exists")

manifest_path.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)

print()
print("=" * 90)
print("V5_PREFLIGHT=PASS")
print("FROZEN_TRAINER_UNMODIFIED=TRUE")
print("WRAPPER_CLI=PASS")
print("CUDA=PASS")
print("TRAINING_STARTED=FALSE")
print("MANIFEST_SHA256=", sha256(manifest_path))
