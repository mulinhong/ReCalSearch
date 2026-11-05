from pathlib import Path
import os
import ast
import hashlib
import json
import py_compile
import torch

ROOT = Path(
    "/root/autodl-tmp/lora-code-embeddings-main/text2code"
)

BASE = ROOT / (
    "results/reviewer_revision/"
    "r1_3_multilingual_matched_v1"
)

OUT = Path(os.environ["R13_V3_OUT"])

ORIGINAL = ROOT / "train_qwen_verifier.py"
NEW = OUT / "train_qwen_verifier_r13_v3.py"

EXPECTED = {
    "original":
        "ec0a1307a2b1d5e2a283fa3527582b487761e388fbda26fa6a1b5b758d3e2c07",
    "dataset":
        "bcbd58d005cf227a52c9afed0912983340ce1b30888bd32c42643a214eb4bf3b",
    "token_budget":
        "accc8292d8c869ade78cf0d3e5db927bd9f1377d8f6794cbb93a3d9eb05dffac",
    "gt_audit":
        "a1897028d037c9d84bfa74285691b0327501aae12b7bc384aa069e9dd1e5742c",
}

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()

def gate(condition, message):
    if not condition:
        raise RuntimeError("FAIL_STOP: " + message)

def check(path, expected, name):
    gate(path.is_file(), f"{name}: missing {path}")
    gate(
        digest(path) == expected,
        f"{name}: SHA256 mismatch"
    )
    print(f"{name}=PASS")

print("=" * 100)
print("WORK1 R1-3 TRAINER REPAIR V3")
print("=" * 100)

check(
    ORIGINAL, EXPECTED["original"],
    "ORIGINAL_TRAINER_SHA256"
)

check(
    BASE / "multi5k_nomic_seed20261013.jsonl",
    EXPECTED["dataset"],
    "MULTI5K_DATA_SHA256"
)

budget_file = (
    BASE / "token_optimizer_freeze_v1/"
    "token_budget_and_trainer_freeze.json"
)

check(
    budget_file, EXPECTED["token_budget"],
    "TOKEN_BUDGET_SHA256"
)

check(
    BASE / "final_pretrain_gate_v2/"
    "final_pretrain_summary.json",
    EXPECTED["gt_audit"],
    "CANONICAL_GT_SHA256"
)

budget = json.loads(
    budget_file.read_text(encoding="utf-8")
)

gate(
    budget["multi5k_total"]["examples"] == 45000,
    "Training example count mismatch"
)

gate(
    budget["multi5k_total"]["effective_tokens"]
    == 8126238,
    "Training token count mismatch"
)

print()
print("[1] HISTORICAL TRAINING_ARGS.BIN")

historical_data_seeds = []

for seed in (13, 42, 87):
    path = (
        ROOT / "trained_verifiers/"
        "reviewer_r1_1_hn_source_ablation/"
        f"nomic_seed{seed}/training_args.bin"
    )

    gate(path.is_file(), f"Missing training args seed={seed}")

    # Historical local checkpoint is a trusted artifact.
    args = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    optimizer = getattr(
        args.optim, "value", str(args.optim)
    )

    historical_data_seeds.append(args.data_seed)

    gate(
        str(optimizer) == "adamw_torch",
        f"Historical optimizer mismatch seed={seed}"
    )

    gate(
        int(args.seed) == seed,
        f"Historical seed mismatch seed={seed}"
    )

    gate(
        args.gradient_accumulation_steps == 16,
        f"Historical gradient accumulation mismatch seed={seed}"
    )

    gate(
        args.per_device_train_batch_size == 1,
        f"Historical batch mismatch seed={seed}"
    )

    gate(
        abs(float(args.learning_rate) - 1e-4)
        < 1e-12,
        f"Historical LR mismatch seed={seed}"
    )

    print(
        f"SEED={seed} "
        f"OPTIM={optimizer} "
        f"DATA_SEED={args.data_seed} "
        "PASS"
    )

gate(
    historical_data_seeds == [None, None, None],
    "Historical data_seed is not consistently None; "
    "manual provenance review required"
)

print("HISTORICAL_OPTIMIZER_AND_DATA_SEED=PASS")

print()
print("[2] GENERATE TRAINER FROM ORIGINAL SOURCE")

source = ORIGINAL.read_text(encoding="utf-8")

original_ast = ast.parse(source)

def training_calls(tree):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "TrainingArguments"
    ]

calls = training_calls(original_ast)

gate(
    len(calls) == 1,
    "Expected exactly one TrainingArguments call"
)

old_optimizer_keys = [
    k for k in calls[0].keywords
    if k.arg == "optim"
]

print(
    "ORIGINAL_TRAINING_ARGS_OPTIM_COUNT =",
    len(old_optimizer_keys)
)

# Handle both possibilities safely:
# original has no explicit optimizer, or already
# contains exactly one correct explicit optimizer.

gate(
    len(old_optimizer_keys) <= 1,
    "Original trainer has duplicate optim keywords"
)

patched = source

if len(old_optimizer_keys) == 0:
    old = (
        "        deepspeed=args.deepspeed,\n"
        "        seed=args.seed,\n"
    )

    new = (
        "        deepspeed=args.deepspeed,\n"
        '        optim="adamw_torch",\n'
        "        seed=args.seed,\n"
    )

    gate(
        patched.count(old) == 1,
        "Cannot uniquely insert optimizer"
    )

    patched = patched.replace(old, new, 1)

else:
    optimizer_node = old_optimizer_keys[0].value

    gate(
        isinstance(optimizer_node, ast.Constant)
        and optimizer_node.value == "adamw_torch",
        "Existing optimizer differs from historical"
    )

# Align data_seed with the three historical checkpoints.
old = "        data_seed=args.seed,\n"
new = "        data_seed=None,\n"

gate(
    patched.count(old) == 1,
    "Cannot uniquely align historical data_seed"
)

patched = patched.replace(old, new, 1)

# Record the explicit optimizer in verifier_config.
old = (
    '                "warmup_ratio": args.warmup_ratio,\n'
)

new = (
    '                "warmup_ratio": args.warmup_ratio,\n'
    '                "optim": "adamw_torch",\n'
)

gate(
    patched.count(old) == 1,
    "Cannot uniquely locate verifier config writer"
)

patched = patched.replace(old, new, 1)

print()
print("[3] AST AND REAL PYTHON COMPILATION")

tree = ast.parse(patched)
calls = training_calls(tree)

gate(
    len(calls) == 1,
    "Repaired trainer TrainingArguments call count incorrect"
)

optim_keys = [
    k for k in calls[0].keywords
    if k.arg == "optim"
]

gate(
    len(optim_keys) == 1,
    "Repaired trainer must have exactly one optim keyword"
)

gate(
    isinstance(optim_keys[0].value, ast.Constant)
    and optim_keys[0].value.value == "adamw_torch",
    "Repaired optimizer is not adamw_torch"
)

data_keys = [
    k for k in calls[0].keywords
    if k.arg == "data_seed"
]

gate(
    len(data_keys) == 1
    and isinstance(data_keys[0].value, ast.Constant)
    and data_keys[0].value.value is None,
    "Repaired data_seed does not match historical"
)

# ast.parse alone does not guarantee that a
# file is executable: compile catches repeated
# keyword arguments such as optim twice.
compile(
    patched,
    str(NEW),
    "exec",
    dont_inherit=True
)

gate(not NEW.exists(), "New trainer already exists")

NEW.write_text(
    patched,
    encoding="utf-8"
)

py_compile.compile(
    str(NEW),
    cfile=str(OUT / "trainer_syntax.pyc"),
    doraise=True
)

gate(
    digest(ORIGINAL) == EXPECTED["original"],
    "Original trainer changed"
)

print("TRAINING_ARGUMENTS_OPTIM_COUNT=1")
print("PYTHON_COMPILE=PASS")
print("PY_COMPILE=PASS")
print("OPTIMIZER=adamw_torch")
print("DATA_SEED=None")
print("ORIGINAL_TRAINER_UNCHANGED=PASS")
print("REPAIRED_TRAINER_SHA256=", digest(NEW))

print()
print("[4] CUDA AND MODEL PREFLIGHT")

gate(
    os.environ.get("CUDA_VISIBLE_DEVICES") == "0",
    "CUDA visibility not restored"
)

gate(
    torch.cuda.is_available(),
    "CUDA unavailable"
)

free_bytes, total_bytes = torch.cuda.mem_get_info(0)

print(
    "GPU=",
    torch.cuda.get_device_name(0)
)

print(
    "FREE_GPU_MEMORY_GIB=",
    round(free_bytes / 1024**3, 3)
)

gate(
    free_bytes >= 16 * 1024**3,
    "Not enough free GPU memory"
)

base_model = Path(
    "/root/autodl-tmp/hf_models/"
    "Qwen2.5-Coder-7B-Instruct"
)

gate(
    base_model.is_dir(),
    "Qwen base model missing"
)

manifest = {
    "format": "r1_3_trainer_repair_v3",
    "seed": 13,
    "training_examples": 45000,
    "effective_tokens": 8126238,
    "historical_optimizer": "adamw_torch",
    "new_optimizer": "adamw_torch",
    "historical_data_seed": None,
    "new_data_seed": None,
    "old_script_sha256": digest(ORIGINAL),
    "new_script_sha256": digest(NEW),
    "source_changes": [
        "Explicitly fix optimizer to adamw_torch",
        "Set data_seed=None to match historical checkpoints",
        "Record optimizer in verifier_config.json",
    ],
    "compiled": True,
    "training_started": False,
}

manifest_path = OUT / "trainer_repair_manifest.json"

gate(
    not manifest_path.exists(),
    "Manifest already exists"
)

manifest_path.write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8"
)

print()
print("=" * 100)
print("R1_3_TRAINER_V3_PREFLIGHT=PASS")
print("TRAINING_AUTHORIZED=TRUE")
print("TRAINING_STARTED=FALSE")
print("MANIFEST_SHA256=", digest(manifest_path))
