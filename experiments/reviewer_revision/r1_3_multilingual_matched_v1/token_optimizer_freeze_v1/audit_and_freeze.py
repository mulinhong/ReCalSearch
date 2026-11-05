from pathlib import Path
from collections import Counter
import hashlib
import json
import ast
import math

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(
    "/root/autodl-tmp/lora-code-embeddings-main/text2code"
)

R13 = ROOT / (
    "results/reviewer_revision/r1_3_multilingual_matched_v1"
)

OUT = R13 / "token_optimizer_freeze_v1"

SOURCE = ROOT / (
    "results/reviewer_revision/"
    "hn_source_ablation/qwen_data"
)

MODEL = Path(
    "/root/autodl-tmp/hf_models/"
    "Qwen2.5-Coder-7B-Instruct"
)

LANGS = [
    "ruby", "javascript", "php",
    "python", "java", "go",
]

EXPECTED = {
    "trainer": (
        "ec0a1307a2b1d5e2a283fa3527582b487761e388fbda26fa6a1b5b758d3e2c07"
    ),
    "multi5k": (
        "bcbd58d005cf227a52c9afed0912983340ce1b30888bd32c42643a214eb4bf3b"
    ),
    "selection": (
        "42dbbf6e28ce73cf116a5231497a8fbb32728986772ff5308bda92a54e76a046"
    ),
    "multi30k": (
        "2240eede7a3d50bf9846562ca0eefb4b0236c8619d40c7110212e947fea16503"
    ),
    "gt_audit": (
        "a1897028d037c9d84bfa74285691b0327501aae12b7bc384aa069e9dd1e5742c"
    ),
    "pool_conversion": (
        "558902f9d89b0ec9ee3ba61e1227ad70285f87f3f5217b5418d26ad40da0ee49"
    ),
}

def require(ok, message):
    if not ok:
        raise RuntimeError("FAIL_STOP: " + message)

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(block)
    return h.hexdigest()

def check(path, expected, name):
    require(path.is_file(), f"Missing {name}: {path}")
    actual = sha256(path)
    require(actual == expected, f"{name}: SHA256 mismatch")
    print(f"{name} SHA256=PASS")
    return path

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

print("=" * 100)
print("WORK1 R1-3 TOKEN BUDGET AND OPTIMIZER FREEZE")
print("=" * 100)
print("GPU_TRAINING = DISABLED")
print("MAX_LENGTH = 1024")
print("MICRO_BATCH_SIZE = 1")
print("GRADIENT_ACCUMULATION = 16")
print()

trainer = check(
    ROOT / "train_qwen_verifier.py",
    EXPECTED["trainer"],
    "FROZEN_TRAINER"
)

multi = check(
    R13 / "multi5k_nomic_seed20261013.jsonl",
    EXPECTED["multi5k"],
    "MULTI5K_DATA"
)

selection_path = check(
    R13 / "multi5k_selection_manifest.json",
    EXPECTED["selection"],
    "MULTI5K_SELECTION"
)

check(
    SOURCE / "combined_nomic_hn7_30k.jsonl",
    EXPECTED["multi30k"],
    "FROZEN_MULTI30K"
)

gt_report = check(
    R13 / "final_pretrain_gate_v2/final_pretrain_summary.json",
    EXPECTED["gt_audit"],
    "CANONICAL_GT_AUDIT"
)

conversion_report = check(
    R13 / "validation_pool_adapter_v2/conversion_manifest.json",
    EXPECTED["pool_conversion"],
    "VALIDATION_CONVERSION"
)

gt = read_json(gt_report)
conversion = read_json(conversion_report)
selection = read_json(selection_path)

require(
    all(
        gt["canonical_gt"][lang]["gt_identity_pass"]
        for lang in LANGS
    ),
    "Canonical GT audit gate not passed",
)

require(
    conversion.get("conversion_complete") is True,
    "Validation conversion incomplete",
)

require(
    selection["total_groups"] == 5000
    and selection["total_examples"] == 45000,
    "Multi5k selection manifest mismatch",
)

print()
print("=" * 100)
print("PHASE 1: CREATE HISTORICAL-OPTIMIZER-COMPATIBLE TRAINER")
print("=" * 100)

source = trainer.read_text(encoding="utf-8")

# Insert an explicit optimizer into TrainingArguments.
# No model, loss, dataset or optimization-loop changes.
old = (
    "        deepspeed=args.deepspeed,\n"
    "        seed=args.seed,\n"
)

new = (
    "        deepspeed=args.deepspeed,\n"
    '        optim="adamw_torch",\n'
    "        seed=args.seed,\n"
)

require(
    source.count(old) == 1,
    "Could not uniquely locate TrainingArguments optimizer slot",
)

patched = source.replace(old, new, 1)

# Record the actual optimizer in the new model's config.
old_cfg = (
    '                "warmup_ratio": args.warmup_ratio,\n'
)

new_cfg = (
    '                "warmup_ratio": args.warmup_ratio,\n'
    '                "optim": "adamw_torch",\n'
)

require(
    patched.count(old_cfg) == 1,
    "Could not uniquely locate verifier configuration writer",
)

patched = patched.replace(old_cfg, new_cfg, 1)

ast.parse(patched)

new_trainer = OUT / "train_qwen_verifier_r13.py"

require(
    not new_trainer.exists(),
    "New trainer already exists",
)

new_trainer.write_text(
    patched,
    encoding="utf-8",
)

require(
    'optim="adamw_torch"' in patched,
    "Optimizer override missing",
)

require(
    sha256(trainer) == EXPECTED["trainer"],
    "Frozen original trainer changed",
)

print("HISTORICAL_TRAINER_UNCHANGED = PASS")
print("OPTIMIZER_EXPLICIT_ADAMW_TORCH = PASS")
print("NEW_TRAINER_SYNTAX = PASS")
print("NEW_TRAINER_SHA256 =", sha256(new_trainer))

print()
print("=" * 100)
print("PHASE 2: EXACT TOKENIZER-BASED TRAINING BUDGET")
print("=" * 100)

require(
    MODEL.is_dir(),
    "Frozen Qwen tokenizer directory missing",
)

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL),
    use_fast=False,
    trust_remote_code=False,
    local_files_only=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"
tokenizer.truncation_side = "left"

print("TOKENIZER_CLASS =", type(tokenizer).__name__)
print("TOKENIZER_IS_FAST =", tokenizer.is_fast)

def empty_metric():
    return {
        "examples": 0,
        "raw_tokens": 0,
        "effective_tokens": 0,
        "padded_tokens": 0,
        "answer_tokens": 0,
        "truncated_examples": 0,
        "removed_prompt_tokens": 0,
        "effective_lengths": [],
    }

def add_metric(m, prompt_len, answer_len):
    max_prompt = max(1, 1024 - answer_len)

    prompt_used = min(prompt_len, max_prompt)

    effective = prompt_used + answer_len
    raw = prompt_len + answer_len

    require(
        effective <= 1024,
        "Effective input exceeds max_length",
    )

    padded = 8 * math.ceil(effective / 8)

    m["examples"] += 1
    m["raw_tokens"] += raw
    m["effective_tokens"] += effective
    m["padded_tokens"] += padded
    m["answer_tokens"] += answer_len
    m["truncated_examples"] += int(
        prompt_len > max_prompt
    )
    m["removed_prompt_tokens"] += (
        prompt_len - prompt_used
    )
    m["effective_lengths"].append(effective)

def serialize_metric(m):
    values = np.asarray(
        m["effective_lengths"],
        dtype=np.int32
    )

    result = {
        k: int(v)
        for k, v in m.items()
        if k != "effective_lengths"
    }

    result["mean_effective_length"] = (
        float(values.mean()) if values.size else 0.0
    )

    result["p50_effective_length"] = (
        float(np.percentile(values, 50))
        if values.size else 0.0
    )

    result["p95_effective_length"] = (
        float(np.percentile(values, 95))
        if values.size else 0.0
    )

    result["max_effective_length"] = (
        int(values.max()) if values.size else 0
    )

    return result

selected = {
    lang: set(
        selection["languages"][lang]["selected_query_ids"]
    )
    for lang in LANGS
}

answer_cache = {}

def answer_length(row):
    target = row.get("target")

    if target:
        answer = str(target)
    else:
        answer = (
            " Yes" if int(row["label"]) == 1
            else " No"
        )

    if answer not in answer_cache:
        ids = tokenizer(
            answer,
            add_special_tokens=False
        ).input_ids

        require(
            len(ids) > 0,
            "Empty tokenized answer",
        )

        answer_cache[answer] = len(ids)

    return answer_cache[answer]

mono_results = {}
selected_results = {}
selected_row_hashes = Counter()

for lang in LANGS:

    path = SOURCE / f"{lang}_nomic_hn7_5k.jsonl"

    check(
        path,
        selection["languages"][lang]["input_sha256"],
        f"MONO5K_{lang}"
    )

    full_metric = empty_metric()
    selected_metric = empty_metric()

    rows = []
    raw_lines = []

    def process_batch():
        if not rows:
            return

        # Same use_fast=False tokenizer and
        # add_special_tokens=False behavior as
        # the historical VerifierDataset.
        prompts = [str(x["prompt"]) for x in rows]

        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            return_attention_mask=False,
        ).input_ids

        require(
            len(encoded) == len(rows),
            "Tokenizer batch length mismatch",
        )

        for row, line, ids in zip(
            rows, raw_lines, encoded
        ):
            qid = str(row["query_id"])

            a_len = answer_length(row)
            p_len = len(ids)

            add_metric(full_metric, p_len, a_len)

            if qid in selected[lang]:
                add_metric(
                    selected_metric, p_len, a_len
                )

                selected_row_hashes[
                    hashlib.sha256(line).hexdigest()
                ] += 1

        rows.clear()
        raw_lines.clear()

    with path.open("rb") as f:

        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            require(
                row["language"] == lang,
                f"{lang}: source language mismatch",
            )

            rows.append(row)
            raw_lines.append(line)

            if len(rows) >= 64:
                process_batch()

    process_batch()

    require(
        full_metric["examples"] == 45000,
        f"{lang}: mono example count mismatch",
    )

    expected_selected = (
        selection["languages"][lang]["selected_groups"]
        * 9
    )

    require(
        selected_metric["examples"] == expected_selected,
        f"{lang}: selected example count mismatch",
    )

    mono_results[lang] = serialize_metric(full_metric)
    selected_results[lang] = serialize_metric(
        selected_metric
    )

    print(
        f"{lang:<12} "
        f"mono_tokens={full_metric['effective_tokens']} "
        f"multi_selected_tokens={selected_metric['effective_tokens']} "
        f"truncated={full_metric['truncated_examples']} "
        "PASS",
        flush=True,
    )

# Verify the Multi5k output is exactly the same
# selected source examples, merely shuffled.
output_row_hashes = Counter()
saved_rows = 0

with multi.open("rb") as f:

    for line in f:
        if not line.strip():
            continue

        saved_rows += 1

        output_row_hashes[
            hashlib.sha256(line).hexdigest()
        ] += 1

require(
    saved_rows == 45000,
    "Multi5k saved row count mismatch",
)

require(
    output_row_hashes == selected_row_hashes,
    "Multi5k raw training rows differ from selected source rows",
)

print()
print("MULTI5K_SOURCE_EXAMPLE_PARITY = PASS")

def sum_metric(metrics, key):
    return sum(
        int(m[key]) for m in metrics.values()
    )

keys = [
    "examples",
    "raw_tokens",
    "effective_tokens",
    "padded_tokens",
    "answer_tokens",
    "truncated_examples",
    "removed_prompt_tokens",
]

multi5k_total = {
    key: sum_metric(selected_results, key)
    for key in keys
}

multi30k_total = {
    key: sum_metric(mono_results, key)
    for key in keys
}

require(
    multi5k_total["examples"] == 45000,
    "Multi5k token accounting row mismatch",
)

require(
    multi30k_total["examples"] == 270000,
    "Multi30k token accounting row mismatch",
)

comparison = {}

for lang in LANGS:

    mono = mono_results[lang]

    comparison[lang] = {
        "multi5k_vs_mono5k_effective_token_ratio":
            multi5k_total["effective_tokens"]
            / mono["effective_tokens"],

        "multi30k_vs_mono_repeated6_effective_token_ratio":
            multi30k_total["effective_tokens"]
            / (6 * mono["effective_tokens"]),

        "mono5k_nominal_optimizer_steps": 2813,
        "mono_repeated6_nominal_optimizer_steps": 16875,
    }

print()
print("=" * 100)
print("PHASE 3: BUDGET COMPARISON")
print("=" * 100)

print("MULTI5K =", multi5k_total)
print("MULTI30K =", multi30k_total)

print()
print("COMMON NOMINAL TRAINING BUDGET")

print("MULTI5K_EXAMPLES = 45000")
print("MONO5K_EXAMPLES = 45000")
print("MULTI5K_STEPS = 2813")
print("MONO5K_STEPS = 2813")

print("MULTI30K_EXAMPLES = 270000")
print("MONO_REPEATED6_EXAMPLES = 270000")
print("MULTI30K_STEPS = 16875")
print("MONO_REPEATED6_STEPS = 16875")

print()
print("EFFECTIVE TOKEN RATIOS BY LANGUAGE")

for lang in LANGS:
    c = comparison[lang]
    print(
        lang,
        "LOW_BUDGET_RATIO=",
        round(
            c["multi5k_vs_mono5k_effective_token_ratio"],
            6
        ),
        "HIGH_BUDGET_RATIO=",
        round(
            c[
                "multi30k_vs_mono_repeated6_effective_token_ratio"
            ],
            6
        ),
    )

report = {
    "format": "r1_3_token_optimizer_freeze_v1",
    "tokenizer": {
        "path": str(MODEL),
        "class": type(tokenizer).__name__,
        "is_fast": bool(tokenizer.is_fast),
        "add_special_tokens": False,
        "max_length": 1024,
        "truncation_side": "left",
        "padding_side": "right",
        "padding_multiple": 8,
    },
    "source_hashes": EXPECTED,
    "new_trainer": str(new_trainer),
    "new_trainer_sha256": sha256(new_trainer),
    "explicit_optimizer": "adamw_torch",
    "optimizer_source_patch_only": True,
    "mono5k_by_language": mono_results,
    "multi5k_selected_by_language": selected_results,
    "multi5k_total": multi5k_total,
    "multi30k_total": multi30k_total,
    "comparison": comparison,
    "fixed_seeds": [13, 42, 87],
    "groups_per_example": {
        "negatives": 7,
        "positive_repeats": 2,
    },
    "important_limitations": [
        "Equal example/update counts do not imply equal token counts.",
        "Token counts are exact under the audited tokenizer, "
        "truncation and padding recipe, not direct measured GPU FLOPs.",
        "Repeated monolingual data contains no additional unique groups.",
    ],
    "gates": {
        "frozen_input_sha256": "PASS",
        "canonical_gt": "PASS",
        "validation_pool_conversion": "PASS",
        "multi5k_raw_example_parity": "PASS",
        "exact_token_accounting": "PASS",
        "optimizer_explicit": "PASS",
        "source_trainer_unchanged": "PASS",
    },
    "gpu_training_started": False,
    "training_launch_authorized": False,
}

report_path = OUT / "token_budget_and_trainer_freeze.json"

require(
    not report_path.exists(),
    "Budget report already exists",
)

report_path.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    ) + "\n",
    encoding="utf-8",
)

print()
print("=" * 100)
print("FINAL FREEZE GATES")
print("=" * 100)

print("MULTI5K_SOURCE_EXAMPLE_PARITY = PASS")
print("EXACT_TOKEN_ACCOUNTING = PASS")
print("HISTORICAL_OPTIMIZER_OVERRIDE = PASS")
print("TRAINER_SOURCE_UNCHANGED = PASS")
print("TOKEN_BUDGET_FREEZE = PASS")
print("GPU_TRAINING_STARTED = FALSE")
print("NO_FROZEN_ARTIFACT_MODIFICATION = TRUE")
print("REPORT_JSON =", report_path)
print("REPORT_SHA256 =", sha256(report_path))
