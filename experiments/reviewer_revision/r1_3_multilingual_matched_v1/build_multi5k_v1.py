from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json
import random
import os
import sys

ROOT = Path(
    "/root/autodl-tmp/lora-code-embeddings-main/text2code"
)

SOURCE = ROOT / (
    "results/reviewer_revision/"
    "hn_source_ablation/qwen_data"
)

OUT = ROOT / (
    "results/reviewer_revision/"
    "r1_3_multilingual_matched_v1"
)

LANGS = [
    "ruby",
    "javascript",
    "php",
    "python",
    "java",
    "go",
]

ALLOCATION = {
    lang: 5000 // len(LANGS)
    + (1 if i < 5000 % len(LANGS) else 0)
    for i, lang in enumerate(LANGS)
}

SEED = 20261013

EXPECTED_30K_SHA = (
    "2240eede7a3d50bf9846562ca0eefb4b0236c8619d40c7110212e947fea16503"
)

def gate(condition, message):
    if not condition:
        raise RuntimeError("FAIL_STOP: " + message)

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()

print("=" * 105)
print("WORK1 R1-3 MULTILINGUAL 5K CONTROL DATA")
print("=" * 105)
print("DATA SPLIT = TRAIN ONLY")
print("SAMPLING SEED =", SEED)
print("TOTAL GROUPS = 5000")
print("EXAMPLES PER GROUP = 9")
print("EXPECTED TRAIN EXAMPLES = 45000")
print()

combined = (
    SOURCE / "combined_nomic_hn7_30k.jsonl"
)

gate(
    combined.is_file(),
    "Frozen Multi-30k input missing",
)

gate(
    sha256(combined) == EXPECTED_30K_SHA,
    "Frozen Multi-30k SHA mismatch",
)

print("FROZEN_MULTI30K_SHA = PASS")

# Verify that the original six language files
# concatenate byte-for-byte into frozen Multi-30k.
concat_hash = hashlib.sha256()

input_hashes = {}

for lang in LANGS:
    path = (
        SOURCE / f"{lang}_nomic_hn7_5k.jsonl"
    )

    gate(
        path.is_file(),
        f"Missing sealed monolingual input: {lang}",
    )

    input_hashes[lang] = sha256(path)

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            concat_hash.update(block)

gate(
    concat_hash.hexdigest() == EXPECTED_30K_SHA,
    "Six language inputs do not match frozen Multi-30k",
)

print("SIX_LANGUAGE_CONCAT_SHA = PASS")
print()

selected_rows = []
selection_manifest = {}

for i, lang in enumerate(LANGS):
    path = (
        SOURCE / f"{lang}_nomic_hn7_5k.jsonl"
    )

    groups = defaultdict(list)
    labels = defaultdict(Counter)

    row_count = 0

    with path.open("r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, 1):
            if not raw_line.strip():
                continue

            row = json.loads(raw_line)

            qid = str(row["query_id"])
            label = int(row["label"])

            gate(
                qid.startswith(lang + ":"),
                f"{lang}:{lineno} invalid query identity",
            )

            gate(
                row.get("language") == lang,
                f"{lang}:{lineno} language mismatch",
            )

            gate(
                label in (0, 1),
                f"{lang}:{lineno} invalid label",
            )

            if label == 1:
                gate(
                    row.get("source") == "positive"
                    and row.get("answer") == "Yes",
                    f"{lang}:{lineno} invalid positive",
                )
            else:
                gate(
                    row.get("source") == "negative"
                    and row.get("answer") == "No"
                    and "code_index" in row,
                    f"{lang}:{lineno} invalid negative",
                )

            groups[qid].append(
                raw_line if raw_line.endswith("\n")
                else raw_line + "\n"
            )
            labels[qid][label] += 1
            row_count += 1

    gate(
        row_count == 45000,
        f"{lang} expected 45000 rows, got {row_count}",
    )

    gate(
        len(groups) == 5000,
        f"{lang} expected 5000 unique groups",
    )

    for qid, counts in labels.items():
        gate(
            counts[1] == 2
            and counts[0] == 7
            and len(groups[qid]) == 9,
            f"{lang}/{qid} group composition mismatch",
        )

    # Freeze an independently seeded TRAIN-only
    # selection for this language.
    rng = random.Random(SEED + i * 1009)

    # Numeric query-ID ordering avoids dependence
    # on input JSONL shuffle/order.
    all_qids = sorted(
        groups,
        key=lambda q: int(q.split(":", 1)[1])
    )

    wanted = ALLOCATION[lang]

    chosen_qids = sorted(
        rng.sample(all_qids, wanted),
        key=lambda q: int(q.split(":", 1)[1])
    )

    for qid in chosen_qids:
        selected_rows.extend(groups[qid])

    selection_manifest[lang] = {
        "input": str(path),
        "input_sha256": input_hashes[lang],
        "available_groups": 5000,
        "selected_groups": wanted,
        "selected_query_ids": chosen_qids,
    }

    print(
        f"{lang:<12} "
        f"input=5000 groups "
        f"selected={wanted} "
        f"rows={wanted * 9} "
        "GATE=PASS",
        flush=True,
    )

gate(
    len(selected_rows) == 45000,
    "Multi-5k total example mismatch",
)

# Deterministically shuffle examples rather than
# leaving entire languages as contiguous blocks.
shuffle_rng = random.Random(SEED + 99991)
shuffle_rng.shuffle(selected_rows)

multi_path = (
    OUT / "multi5k_nomic_seed20261013.jsonl"
)

gate(
    not multi_path.exists(),
    "Multi-5k output already exists",
)

with multi_path.open("w", encoding="utf-8") as f:
    f.writelines(selected_rows)

# Independent read-back of saved file.
count = Counter()
language_counts = Counter()
query_sets = defaultdict(set)
query_labels = defaultdict(Counter)

with multi_path.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        item = json.loads(line)
        label = int(item["label"])
        lang = item["language"]
        qid = str(item["query_id"])

        count["total"] += 1
        count["positive" if label == 1 else "negative"] += 1
        language_counts[lang] += 1

        query_sets[lang].add(qid)
        query_labels[(lang, qid)][label] += 1

gate(
    count["total"] == 45000
    and count["positive"] == 10000
    and count["negative"] == 35000,
    "Saved training example count mismatch",
)

for lang in LANGS:
    gate(
        len(query_sets[lang]) == ALLOCATION[lang],
        f"{lang} selected group count mismatch",
    )

    gate(
        language_counts[lang] == ALLOCATION[lang] * 9,
        f"{lang} example count mismatch",
    )

for key, c in query_labels.items():
    gate(
        c[1] == 2 and c[0] == 7,
        f"Saved group composition mismatch: {key}",
    )

manifest = {
    "format": "r1_3_multi5k_selection_manifest_v1",
    "split": "train",
    "seed": SEED,
    "total_groups": 5000,
    "positive_examples": 10000,
    "negative_examples": 35000,
    "total_examples": 45000,
    "allocation": ALLOCATION,
    "languages": selection_manifest,
    "output": str(multi_path),
    "output_sha256": sha256(multi_path),
    "frozen_multi30k_sha256": EXPECTED_30K_SHA,
    "no_test_based_selection": True,
}

manifest_path = (
    OUT / "multi5k_selection_manifest.json"
)

manifest_path.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)

protocol = {
    "format": "r1_3_data_design_v1",
    "reviewer_item": "R1-3",
    "sampling_frozen_before_r1_3_training": True,
    "sampling_seed": SEED,
    "train_split_only": True,
    "train_sources": {
        "mono5k": {
            lang: str(
                SOURCE / f"{lang}_nomic_hn7_5k.jsonl"
            )
            for lang in LANGS
        },
        "multi5k": str(multi_path),
        "multi30k": str(combined),
    },
    "comparisons": [
        "multi5k vs mono5k: equal example counts",
        "multi30k vs mono5k repeated exposure: "
        "approximately equal example exposure/update budget",
    ],
    "fixed_training_seeds": [13, 42, 87],
    "model_family": "Qwen2.5-Coder-7B-Instruct",
    "target_lora_r": 8,
    "target_lora_alpha": 16,
    "target_lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ],
    "training_cli_and_optimizer": (
        "PENDING: independently verify against "
        "the frozen R1-1 historical training commands"
    ),
    "exact_token_budget": "PENDING_AUDIT",
    "validation_protocol": "PENDING_RESOURCE_VERIFICATION",
    "test_split_used_to_select_design": False,
    "gpu_training_allowed": False,
}

protocol_path = OUT / "r1_3_data_design_v1.json"

protocol_path.write_text(
    json.dumps(
        protocol,
        ensure_ascii=False,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)

print()
print("=" * 105)
print("R1-3 MULTI-5K DATA CLOSURE")
print("=" * 105)

print("INPUT_CONCAT_GATE = PASS")
print("SELECTED_GROUPS = 5000")
print("SAVED_EXAMPLES =", count["total"])
print("SAVED_POSITIVE =", count["positive"])
print("SAVED_NEGATIVE =", count["negative"])
print("MULTI5K_DATA_GATE = PASS")
print("TRAINING_PROTOCOL_GATE = PENDING")
print("GPU_TRAINING_STARTED = FALSE")
print()

for path in [
    multi_path,
    manifest_path,
    protocol_path,
]:
    print("FILE =", path)
    print("SHA256 =", sha256(path))

# Inventory actual validation resources without
# assuming they are in any one directory.
print()
print("=" * 105)
print("VALIDATION RESOURCE INVENTORY")
print("=" * 105)

SEARCH_ROOTS = [
    ROOT / "results/reviewer_revision/pools",
    ROOT / "results/jss_rescue_oldcsn",
    ROOT / "results/reviewer_revision/recal_valid_fresh_20260911",
]

valid_candidates = []

for base in SEARCH_ROOTS:
    if not base.is_dir():
        continue

    for p in base.rglob("*.npz"):
        name = p.name.lower()
        if (
            "valid" in name
            and (
                "nomic" in name
                or "top100" in name
            )
        ):
            valid_candidates.append(str(p))

for path in sorted(set(valid_candidates))[:60]:
    print("VALID_POOL_CANDIDATE =", path)

print(
    "VALID_POOL_CANDIDATE_COUNT =",
    len(set(valid_candidates))
)

CACHE = (
    ROOT / "data/rerank/coderank_cache_oldcsn"
)

for lang in LANGS:
    matches = (
        sorted(CACHE.glob(f"{lang}_valid_*.query.npy"))
        if CACHE.is_dir()
        else []
    )

    print(
        "VALID_QUERY_CACHE",
        lang,
        "COUNT=",
        len(matches),
    )
    for p in matches[:3]:
        print("  CACHE =", p)

print()
print("NO_GPU_INFERENCE = TRUE")
print("NO_MODEL_TRAINING = TRUE")
print("NO_FROZEN_ARTIFACT_MODIFICATION = TRUE")
