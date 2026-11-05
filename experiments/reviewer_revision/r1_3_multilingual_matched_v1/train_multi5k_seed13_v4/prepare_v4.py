from pathlib import Path
import ast
import hashlib
import difflib
import json
import os
import py_compile
import re

import torch

ROOT = Path(
    "/root/autodl-tmp/lora-code-embeddings-main/text2code"
)
BASE = ROOT / (
    "results/reviewer_revision/"
    "r1_3_multilingual_matched_v1"
)
OUT = Path(os.environ["R13_V4_OUT"])

ORIG = ROOT / "train_qwen_verifier.py"
NEW = OUT / "train_qwen_verifier_r13_v4.py"

EXPECTED = {
    "original": "ec0a1307a2b1d5e2a283fa3527582b487761e388fbda26fa6a1b5b758d3e2c07",
    "data": "bcbd58d005cf227a52c9afed0912983340ce1b30888bd32c42643a214eb4bf3b",
    "budget": "accc8292d8c869ade78cf0d3e5db927bd9f1377d8f6794cbb93a3d9eb05dffac",
    "gt": "a1897028d037c9d84bfa74285691b0327501aae12b7bc384aa069e9dd1e5742c",
}

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
    gate(path.is_file(), f"Missing: {path}")
    gate(sha256(path) == expected, f"SHA mismatch: {path}")
    print("SHA256_PASS =", path.name)

print("=" * 95)
print("WORK1 R1-3 TRAINER V4 SOURCE-LEVEL REPAIR")
print("=" * 95)

check(ORIG, EXPECTED["original"])
check(
    BASE / "multi5k_nomic_seed20261013.jsonl",
    EXPECTED["data"],
)

budget_path = BASE / (
    "token_optimizer_freeze_v1/"
    "token_budget_and_trainer_freeze.json"
)
check(budget_path, EXPECTED["budget"])
check(
    BASE / "final_pretrain_gate_v2/"
    "final_pretrain_summary.json",
    EXPECTED["gt"],
)

budget = json.loads(budget_path.read_text())

gate(
    budget["multi5k_total"]["examples"] == 45000,
    "Dataset size changed",
)
gate(
    budget["multi5k_total"]["effective_tokens"] == 8126238,
    "Token budget changed",
)

print()
print("[1] HISTORICAL OPTIMIZER PROVENANCE")

for seed in [13, 42, 87]:
    path = (
        ROOT / "trained_verifiers/"
        "reviewer_r1_1_hn_source_ablation/"
        f"nomic_seed{seed}/training_args.bin"
    )
    gate(path.is_file(), f"Historical training_args missing: {seed}")

    hist = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    optim = str(getattr(hist.optim, "value", hist.optim))

    gate(optim == "adamw_torch", f"Optimizer mismatch: {seed}")
    gate(hist.data_seed is None, f"Data seed mismatch: {seed}")
    gate(int(hist.seed) == seed, f"Seed mismatch: {seed}")

    print(
        f"SEED={seed} OPTIM={optim} "
        f"DATA_SEED={hist.data_seed} PASS"
    )

print()
print("[2] EXAMINE ORIGINAL SOURCE AST")

raw = ORIG.read_bytes()
source = raw.decode("utf-8")
tree = ast.parse(source)

byte_lines = raw.splitlines(keepends=True)
offsets = [0]

for line in byte_lines:
    offsets.append(offsets[-1] + len(line))

def node_span(node):
    start = offsets[node.lineno - 1] + node.col_offset
    end = offsets[node.end_lineno - 1] + node.end_col_offset
    return start, end

def calls_of(tree, name):
    return [
        x for x in ast.walk(tree)
        if isinstance(x, ast.Call)
        and isinstance(x.func, ast.Name)
        and x.func.id == name
    ]

calls = calls_of(tree, "TrainingArguments")
gate(len(calls) == 1, "TrainingArguments call count != 1")

call = calls[0]

def keywords(call, name):
    return [k for k in call.keywords if k.arg == name]

optimizer_keywords = keywords(call, "optim")
seed_keywords = keywords(call, "data_seed")

gate(
    len(optimizer_keywords) == 1,
    "Expected exactly one existing optimizer keyword",
)
gate(
    len(seed_keywords) == 1,
    "Expected exactly one existing data_seed keyword",
)

old_optim_expression = ast.get_source_segment(
    source, optimizer_keywords[0].value
)
old_seed_expression = ast.get_source_segment(
    source, seed_keywords[0].value
)

print("ORIGINAL_OPTIM_EXPRESSION =", old_optim_expression)
print("ORIGINAL_DATA_SEED_EXPRESSION =", old_seed_expression)

# Replace the VALUES of existing keywords.
# Crucially, do not insert any additional optim keyword.
patches = [
    (
        *node_span(optimizer_keywords[0].value),
        b'"adamw_torch"',
        "TrainingArguments.optim",
    ),
    (
        *node_span(seed_keywords[0].value),
        b"None",
        "TrainingArguments.data_seed",
    ),
]

# Locate the verifier_config dictionary by its field names.
config_dicts = []

for node in ast.walk(tree):
    if not isinstance(node, ast.Dict):
        continue

    keys = {
        k.value
        for k in node.keys
        if isinstance(k, ast.Constant)
        and isinstance(k.value, str)
    }

    if {
        "loss_type",
        "num_training_examples",
        "warmup_ratio",
    }.issubset(keys):
        config_dicts.append(node)

gate(
    len(config_dicts) == 1,
    "Unable to uniquely identify verifier_config dictionary",
)

cfg = config_dicts[0]
cfg_keys = {
    k.value: i
    for i, k in enumerate(cfg.keys)
    if isinstance(k, ast.Constant)
    and isinstance(k.value, str)
}

if "optim" in cfg_keys:
    value_node = cfg.values[cfg_keys["optim"]]
    patches.append(
        (
            *node_span(value_node),
            b'"adamw_torch"',
            "verifier_config.optim",
        )
    )
else:
    key_node = cfg.keys[cfg_keys["warmup_ratio"]]
    line_number = key_node.lineno
    line = byte_lines[line_number - 1].decode("utf-8")

    indent = re.match(r"^\s*", line).group(0)

    insert_position = offsets[line_number]
    insertion = (
        indent + '"optim": "adamw_torch",\n'
    ).encode("utf-8")

    patches.append(
        (
            insert_position,
            insert_position,
            insertion,
            "verifier_config.optim_insert",
        )
    )

print()
print("[3] APPLY EXACTLY SCOPED SOURCE PATCHES")

# Apply from bottom to top so byte offsets stay valid.
patched = raw

for start, end, replacement, label in sorted(
    patches,
    key=lambda x: x[0],
    reverse=True,
):
    gate(0 <= start <= end <= len(patched), "Invalid patch span")
    patched = patched[:start] + replacement + patched[end:]
    print("SOURCE_PATCH =", label)

patched_text = patched.decode("utf-8")
new_tree = ast.parse(patched_text)

new_calls = calls_of(new_tree, "TrainingArguments")
gate(len(new_calls) == 1, "Patched TrainingArguments count != 1")

new_optim = keywords(new_calls[0], "optim")
new_data_seed = keywords(new_calls[0], "data_seed")

gate(len(new_optim) == 1, "Duplicate or missing optim")
gate(len(new_data_seed) == 1, "Duplicate or missing data_seed")

gate(
    isinstance(new_optim[0].value, ast.Constant)
    and new_optim[0].value.value == "adamw_torch",
    "Optimizer not fixed correctly",
)
gate(
    isinstance(new_data_seed[0].value, ast.Constant)
    and new_data_seed[0].value.value is None,
    "Data seed not fixed correctly",
)

# Check verifier_config contains exactly one optim key.
new_config_dicts = []

for node in ast.walk(new_tree):
    if not isinstance(node, ast.Dict):
        continue
    keys = [
        k.value for k in node.keys
        if isinstance(k, ast.Constant)
        and isinstance(k.value, str)
    ]
    if "loss_type" in keys and "num_training_examples" in keys:
        new_config_dicts.append((node, keys))

gate(len(new_config_dicts) == 1, "Patched config ambiguous")
gate(
    new_config_dicts[0][1].count("optim") == 1,
    "Duplicate or missing config optimizer",
)

# compile(), unlike ast.parse(), rejects duplicate
# keyword arguments in a Python function call.
compile(patched_text, str(NEW), "exec")

gate(not NEW.exists(), "Trainer output already exists")
NEW.write_bytes(patched)

py_compile.compile(str(NEW), doraise=True)

diff = "".join(
    difflib.unified_diff(
        source.splitlines(keepends=True),
        patched_text.splitlines(keepends=True),
        fromfile="frozen_train_qwen_verifier.py",
        tofile="train_qwen_verifier_r13_v4.py",
    )
)

(OUT / "trainer_patch.diff").write_text(
    diff, encoding="utf-8"
)

gate(
    sha256(ORIG) == EXPECTED["original"],
    "Frozen original trainer modified",
)

print("TRAINING_ARGUMENTS_OPTIM_COUNT = 1")
print("TRAINING_ARGUMENTS_DATA_SEED = None")
print("CONFIG_OPTIM_COUNT = 1")
print("COMPILE_GATE = PASS")
print("PY_COMPILE_GATE = PASS")
print("FROZEN_TRAINER_UNCHANGED = PASS")
print("V4_TRAINER_SHA256 =", sha256(NEW))

print()
print("[4] CUDA PREFLIGHT")

gate(
    os.environ.get("CUDA_VISIBLE_DEVICES") == "0",
    "CUDA device visibility incorrect",
)
gate(torch.cuda.is_available(), "CUDA unavailable")

free_bytes, total_bytes = torch.cuda.mem_get_info(0)

print("GPU =", torch.cuda.get_device_name(0))
print("GPU_FREE_GIB =", round(free_bytes / 1024**3, 3))

gate(
    free_bytes >= 16 * 1024**3,
    "Insufficient available GPU memory",
)

model = Path(
    "/root/autodl-tmp/hf_models/Qwen2.5-Coder-7B-Instruct"
)
gate(model.is_dir(), "Base model missing")

manifest = {
    "protocol": "work1_r1_3_multi5k_seed13_v4",
    "seed": 13,
    "optimizer": "adamw_torch",
    "data_seed": None,
    "examples": 45000,
    "effective_tokens": 8126238,
    "original_trainer_sha256": sha256(ORIG),
    "patched_trainer_sha256": sha256(NEW),
    "original_optimizer_expression": old_optim_expression,
    "original_data_seed_expression": old_seed_expression,
    "patches": [p[3] for p in patches],
    "training_started": False,
}

manifest_path = OUT / "pretrain_manifest.json"
gate(not manifest_path.exists(), "Manifest exists")

manifest_path.write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8",
)

print()
print("=" * 95)
print("R1_3_V4_PREFLIGHT = PASS")
print("TRAINER_COMPILE = PASS")
print("EXPLICIT_OPTIMIZER = adamw_torch")
print("HISTORICAL_DATA_SEED_PARITY = PASS")
print("TRAINING_STARTED = FALSE")
print("MANIFEST_SHA256 =", sha256(manifest_path))
