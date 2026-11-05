from pathlib import Path
import ast
import hashlib
import json
import os
import py_compile

import numpy as np
import torch

ROOT = Path.cwd()
BASE = ROOT / "results/reviewer_revision/r1_3_multilingual_matched_v1"
OUT = Path(os.environ["R13_VALID_OUT"])

LANGS = {
    "ruby": 2209,
    "javascript": 8253,
    "php": 26015,
    "python": 23107,
    "java": 15328,
    "go": 14242,
}

HASHES = {
    "ruby": "fc60c7012e7c2019f034e8d46171aa3220cf20c9e7c5afc3728aa56efa5c85ba",
    "javascript": "03b781900ae5a28820068392fcbd7ee0bc37f11b841d077ef18286b644f3446d",
    "php": "95b78c4e8ac688c22632b18011b496dcdfa8f1622f656e3868badb73e6945a8c",
    "python": "2ff29339c22c1c57bcab9e59d6e6eadf385c80baae335c78dd18f6a3909e1c43",
    "java": "b4329683443cdb75ff2b52ea23d09edb5bc0165ba290dd5e9e3b9ca7daae0f14",
    "go": "0bc7dab8f60d1867601dc63ac5817719d1fe5cb33610590393baef0daa7d0a1d",
}

def gate(ok, msg):
    if not ok:
        raise RuntimeError("FAIL_STOP: " + msg)

def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def check(p, expected):
    gate(p.is_file(), f"Missing {p}")
    gate(sha(p) == expected, f"SHA mismatch: {p}")
    print("SHA256_PASS =", p.name)

print("=" * 90)
print("R1-3 MULTI5K SEED13 FULL VALID PREFLIGHT")
print("=" * 90)

original = ROOT / "eval_llm_verifier.py"
smoke = BASE / "validation_smoke_seed13_v3"
smoke_eval = smoke / "eval_llm_verifier_external_pool.py"
smoke_manifest = smoke / "preflight_manifest.json"

check(
    original,
    "a3156c285728d54180ce0da1c65255a92adfe15f583dc987e47ad8a97b8e665a",
)

gate(smoke_manifest.is_file(), "Smoke manifest missing")
manifest = json.loads(smoke_manifest.read_text())

gate(
    manifest["original_evaluator_sha256"] == sha(original),
    "Smoke original evaluator mismatch",
)
gate(
    manifest["scoring_logic_changed"] is False,
    "Smoke scoring change detected",
)
gate(
    manifest["ranking_logic_changed"] is False,
    "Smoke ranking change detected",
)
check(smoke_eval, manifest["new_evaluator_sha256"])

gate(
    (smoke / "exit_code.txt").is_file()
    and (smoke / "exit_code.txt").read_text().strip() == "0",
    "Smoke evaluation did not pass",
)

check(
    BASE / "posttrain_closure_seed13_v1/posttrain_closure.json",
    "077d2193f144378f86bd2eac8a8aebb674ab4f42e0fe18f8dbacee3f8fca5fc1",
)

model = BASE / "models/multi5k_seed13_v5"

check(
    model / "adapter_model.safetensors",
    "c02e0fa78a351bff68fd63e6e28128e112a83a42850cbe47ef606168e045cc59",
)

check(
    model / "adapter_config.json",
    "f8bebe2bb57b113462f49a0ba7935887327c11f080aa3086acf31fc746ac07c8",
)

frozen = ROOT / (
    "results/reviewer_revision/"
    "recal_valid_fresh_20260911/"
    "recal_main_frozen_config.json"
)

check(
    frozen,
    "70789e10518f32a97f5990b234833fb0443fbcbe426e8a8c3515ab93e6b11b0f",
)

cfg = json.loads(frozen.read_text())

gate(cfg["selected_on_split"] == "valid", "Frozen split mismatch")
gate(cfg["config"]["normalization"] == "zscore", "Normalization mismatch")
gate(float(cfg["config"]["fusion_weight"]) == 1, "Alpha mismatch")
gate(int(cfg["config"]["verify_top_k"]) == 15, "Depth mismatch")

gt_report = BASE / "final_pretrain_gate_v2/final_pretrain_summary.json"

check(
    gt_report,
    "a1897028d037c9d84bfa74285691b0327501aae12b7bc384aa069e9dd1e5742c",
)

gt_audit = json.loads(gt_report.read_text())

print()
print("========== SIX LANGUAGE VALIDATION POOLS ==========")

source_hashes = {}

for lang, count in LANGS.items():

    converted = BASE / (
        "validation_pool_adapter_v2/candidate_pools/"
        f"{lang}_valid_nomic_top100.npz"
    )

    source = ROOT / (
        "results/reviewer_revision/"
        "recal_valid_fresh_20260911/"
        "validation_grid/candidate_pools/"
        f"{lang}_valid_nomic_top100.npz"
    )

    check(converted, HASHES[lang])
    gate(source.is_file(), f"Missing original valid pool: {lang}")

    with np.load(source, allow_pickle=False) as a, \
         np.load(converted, allow_pickle=False) as b:

        gt = np.asarray(a["gt_indices"], dtype=np.int64)

        gate(gt.shape == (count,), f"{lang} GT shape mismatch")
        gate(b["indices"].shape == (count, 100), f"{lang} indices shape mismatch")
        gate(b["scores"].shape == (count, 100), f"{lang} scores shape mismatch")
        gate(b["base_ranks"].shape == (count,), f"{lang} ranks shape mismatch")

        for old_key, new_key in [
            ("candidate_indices", "indices"),
            ("candidate_scores", "scores"),
            ("base_ranks", "base_ranks"),
        ]:
            gate(
                np.array_equal(a[old_key], b[new_key]),
                f"{lang} original/converted parity mismatch",
            )

        gate(
            gt_audit["canonical_gt"][lang]["gt_identity_pass"] is True,
            f"{lang} canonical GT audit missing",
        )

    gate(
        (ROOT.parent / "data/CSN" / lang / "valid.jsonl").is_file(),
        f"{lang} valid split missing",
    )

    gate(
        (ROOT.parent / "data/CSN" / lang / "codebase.jsonl").is_file(),
        f"{lang} codebase missing",
    )

    source_hashes[lang] = sha(source)
    print(lang, "VALID_POOL_PARITY=PASS", "N=", count)

print()
print("========== VALID-ONLY EVALUATOR ==========")

source = smoke_eval.read_text(encoding="utf-8")

old = (
    "    _, val_dataset, test_dataset, code_dataset = create_dataset(args.data_root, language)\n"
    '    eval_dataset = val_dataset if args.split == "valid" else test_dataset\n'
    "    gt_indices = text2code_to_array(eval_dataset.text2code, len(eval_dataset))\n"
)

new = (
    '    if args.split != "valid":\n'
    '        raise RuntimeError("R1-3 validation-only evaluator refuses TEST")\n'
    '    canonical_path = Path(\n'
    '        "results/reviewer_revision/recal_valid_fresh_20260911/"\n'
    '        "validation_grid/candidate_pools/"\n'
    '        f"{language}_valid_nomic_top100.npz"\n'
    '    )\n'
    '    with np.load(canonical_path, allow_pickle=False) as canonical_pool:\n'
    '        gt_indices = np.asarray(canonical_pool["gt_indices"], dtype=np.int64).copy()\n'
    '    eval_dataset = range(len(gt_indices))\n'
    '    print("CANONICAL_VALID_GT_FROM_AUDITED_POOL=PASS", flush=True)\n'
)

gate(
    source.count(old) == 1,
    "Unable to uniquely locate dataset loader",
)

patched = source.replace(old, new, 1)

old_ast = ast.parse(source)
new_ast = ast.parse(patched)

def functions(tree):
    return {
        n.name: ast.dump(n, include_attributes=False)
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

old_funcs = functions(old_ast)
new_funcs = functions(new_ast)

gate(set(old_funcs) == set(new_funcs), "Function inventory changed")

for name in old_funcs:
    if name != "eval_language":
        gate(
            old_funcs[name] == new_funcs[name],
            f"Unexpected function change: {name}",
        )

eval_fn = next(
    n for n in new_ast.body
    if isinstance(n, ast.FunctionDef)
    and n.name == "eval_language"
)

gate(
    not any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "create_dataset"
        for n in ast.walk(eval_fn)
    ),
    "Validation evaluator still invokes create_dataset",
)

gate(
    'args.split != "valid"' in patched,
    "VALID-only restriction missing",
)

new_file = OUT / "eval_valid_only.py"

compile(patched, str(new_file), "exec")

gate(not new_file.exists(), "New evaluator already exists")
new_file.write_text(patched, encoding="utf-8")

py_compile.compile(str(new_file), doraise=True)

gate(
    sha(smoke_eval) == manifest["new_evaluator_sha256"],
    "Smoke evaluator was modified",
)

print("SCORING_LOGIC_UNCHANGED=PASS")
print("RANKING_LOGIC_UNCHANGED=PASS")
print("CREATE_DATASET_TEST_READ_REMOVED=PASS")
print("VALID_ONLY_GUARD=PASS")
print("EVALUATOR_COMPILE=PASS")
print("NEW_EVALUATOR_SHA256=", sha(new_file))

gate(torch.cuda.is_available(), "CUDA unavailable")

free, total = torch.cuda.mem_get_info(0)

print("GPU=", torch.cuda.get_device_name(0))
print("FREE_VRAM_GIB=", round(free / 1024**3, 3))

gate(
    free >= 16 * 1024**3,
    "Insufficient free GPU memory",
)

protocol = {
    "experiment": "r1_3_multi5k_seed13_full_valid",
    "split": "valid",
    "languages": list(LANGS),
    "query_counts": LANGS,
    "total_queries": sum(LANGS.values()),
    "candidate_top_k": 100,
    "verify_top_k": 15,
    "normalization": "zscore",
    "fusion_alpha": 1.0,
    "verifier_mode": "yesno",
    "max_length": 1024,
    "max_query_chars": 512,
    "max_code_chars": 3000,
    "batch_size": 1,
    "model_adapter_sha256": sha(model / "adapter_model.safetensors"),
    "evaluator_sha256": sha(new_file),
    "source_evaluator_sha256": sha(original),
    "canonical_valid_pool_hashes": source_hashes,
    "converted_valid_pool_hashes": HASHES,
    "validation_only": True,
    "test_metrics_permitted": False,
    "training_or_test_started": False,
}

p = OUT / "evaluation_protocol.json"

gate(not p.exists(), "Protocol file already exists")
p.write_text(json.dumps(protocol, indent=2) + "\n")

print()
print("R1_3_FULL_VALID_PREFLIGHT=PASS")
print("PROTOCOL_SHA256=", sha(p))
