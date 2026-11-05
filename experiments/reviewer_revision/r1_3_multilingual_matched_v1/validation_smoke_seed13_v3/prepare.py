from pathlib import Path
import ast
import hashlib
import difflib
import json
import py_compile

import numpy as np
import torch

ROOT = Path.cwd()

BASE = ROOT / (
    "results/reviewer_revision/"
    "r1_3_multilingual_matched_v1"
)

OUT = BASE / "validation_smoke_seed13_v3"

ORIGINAL = ROOT / "eval_llm_verifier.py"
NEW = OUT / "eval_llm_verifier_external_pool.py"

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def gate(ok, message):
    if not ok:
        raise RuntimeError("FAIL_STOP: " + message)

def check(path, expected):
    gate(path.is_file(), f"Missing {path}")
    gate(sha(path) == expected, f"SHA mismatch: {path}")
    print("SHA256_PASS =", path.name)

print("=" * 90)
print("R1-3 EXTERNAL CANDIDATE POOL EVALUATOR V3")
print("=" * 90)

check(
    ORIGINAL,
    "a3156c285728d54180ce0da1c65255a92adfe15f583dc987e47ad8a97b8e665a",
)

check(
    BASE / "posttrain_closure_seed13_v1/posttrain_closure.json",
    "077d2193f144378f86bd2eac8a8aebb674ab4f42e0fe18f8dbacee3f8fca5fc1",
)

FROZEN = ROOT / (
    "results/reviewer_revision/"
    "recal_valid_fresh_20260911/"
    "recal_main_frozen_config.json"
)

check(
    FROZEN,
    "70789e10518f32a97f5990b234833fb0443fbcbe426e8a8c3515ab93e6b11b0f",
)

cfg = json.loads(FROZEN.read_text())

gate(cfg["selected_on_split"] == "valid", "Config not validation-frozen")
gate(float(cfg["config"]["fusion_weight"]) == 1, "Alpha mismatch")
gate(cfg["config"]["normalization"] == "zscore", "Normalization mismatch")
gate(int(cfg["config"]["verify_top_k"]) == 15, "Verify depth mismatch")

MODEL = BASE / "models/multi5k_seed13_v5"

check(
    MODEL / "adapter_model.safetensors",
    "c02e0fa78a351bff68fd63e6e28128e112a83a42850cbe47ef606168e045cc59",
)

check(
    MODEL / "adapter_config.json",
    "f8bebe2bb57b113462f49a0ba7935887327c11f080aa3086acf31fc746ac07c8",
)

pool_file = BASE / (
    "validation_pool_adapter_v2/"
    "candidate_pools/ruby_valid_nomic_top100.npz"
)

check(
    pool_file,
    "fc60c7012e7c2019f034e8d46171aa3220cf20c9e7c5afc3728aa56efa5c85ba",
)

with np.load(pool_file, allow_pickle=False) as pool:
    gate(pool["indices"].shape == (2209, 100), "Pool indices shape")
    gate(pool["scores"].shape == (2209, 100), "Pool scores shape")
    gate(pool["base_ranks"].shape == (2209,), "Pool ranks shape")

print()
print("[1] ISOLATED SOURCE REPAIR")

source = ORIGINAL.read_text(encoding="utf-8")

start = '    cache = coderank_cache_paths(\n'
end = '    candidate_pool_key = "dense"\n'

gate(source.count(start) == 1, "Cache block start ambiguous")
gate(source.count(end) == 1, "Cache block end ambiguous")

a = source.index(start)
b = source.index(end)

gate(a < b, "Unexpected evaluator source order")

block = source[a:b]

gate(
    'query_embeds = load_cache_array(' in block
    and 'code_embeds = load_cache_array(' in block,
    "Expected embedding cache loading block missing",
)

# Cache loading is needed only for on-the-fly retrieval.
# All frozen external-pool values are loaded later
# by the original, unmodified candidate-pool branch.

indented = "".join(
    ("    " + line if line.strip() else line)
    for line in block.splitlines(keepends=True)
)

replacement = (
    "    if not args.candidate_pool_pattern:\n"
    + indented
    + "    else:\n"
    + '        print("EXTERNAL_POOL_CACHE_BYPASS=PASS", flush=True)\n\n'
)

patched = source[:a] + replacement + source[b:]

compile(patched, str(NEW), "exec")

def function_ast(tree, name):
    matches = [
        n for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == name
    ]
    gate(len(matches) == 1, f"Function missing/duplicate: {name}")
    return ast.dump(matches[0], include_attributes=False)

old_tree = ast.parse(source)
new_tree = ast.parse(patched)

# All scoring, fusion, ranking, model-loading
# and protocol-selection functions must remain identical.
for name in [
    "load_verifier",
    "score_yesno_prompts",
    "rerank_with_verifier",
    "normalize_scores",
    "compute_metrics_from_ranks",
    "main",
]:
    gate(
        function_ast(old_tree, name)
        == function_ast(new_tree, name),
        f"Unexpected logic change in {name}",
    )

# Confirm the cache block is now the guarded branch.
eval_fn = next(
    n for n in new_tree.body
    if isinstance(n, ast.FunctionDef)
    and n.name == "eval_language"
)

guards = [
    n for n in ast.walk(eval_fn)
    if isinstance(n, ast.If)
    and isinstance(n.test, ast.UnaryOp)
    and isinstance(n.test.op, ast.Not)
    and isinstance(n.test.operand, ast.Attribute)
    and n.test.operand.attr == "candidate_pool_pattern"
]

gate(len(guards) == 1, "External pool bypass guard missing")

gate(not NEW.exists(), "Patched evaluator already exists")

NEW.write_text(patched, encoding="utf-8")

py_compile.compile(str(NEW), doraise=True)

diff = "".join(
    difflib.unified_diff(
        source.splitlines(keepends=True),
        patched.splitlines(keepends=True),
        fromfile="frozen_eval_llm_verifier.py",
        tofile="external_pool_evaluator_v3.py",
    )
)

(OUT / "evaluator_patch.diff").write_text(diff, encoding="utf-8")

gate(
    sha(ORIGINAL)
    == "a3156c285728d54180ce0da1c65255a92adfe15f583dc987e47ad8a97b8e665a",
    "Original evaluator changed",
)

print("EXTERNAL_POOL_CACHE_BYPASS_PATCH=PASS")
print("SCORING_AND_RANKING_AST_PARITY=PASS")
print("PYTHON_COMPILE=PASS")
print("FROZEN_EVALUATOR_UNCHANGED=PASS")
print("NEW_EVALUATOR_SHA256 =", sha(NEW))

print()
print("[2] CUDA / DATA")

gate(torch.cuda.is_available(), "CUDA unavailable")

DATA = ROOT.parent / "data/CSN"
gate((DATA / "ruby/valid.jsonl").is_file(), "Ruby valid missing")
gate((DATA / "ruby/codebase.jsonl").is_file(), "Ruby codebase missing")

print("GPU =", torch.cuda.get_device_name(0))
print("EXTERNAL_CANDIDATE_POOL = VERIFIED")
print("NOMIC_EMBEDDING_CACHE = NOT_REQUIRED")
print("FROZEN_PARAMETERS = UNCHANGED")
print("SPLIT = valid")
print("SMOKE_QUERY_COUNT = 8")

manifest = {
    "protocol": "r1_3_validation_smoke_seed13_v3",
    "original_evaluator_sha256": sha(ORIGINAL),
    "new_evaluator_sha256": sha(NEW),
    "candidate_pool_sha256": sha(pool_file),
    "model_adapter_sha256": sha(MODEL / "adapter_model.safetensors"),
    "frozen_config_sha256": sha(FROZEN),
    "change": "skip unused Nomic embedding cache when external pool is supplied",
    "scoring_logic_changed": False,
    "ranking_logic_changed": False,
    "split": "valid",
    "max_queries": 8,
    "verify_top_k": 15,
    "fusion_alpha": 1.0,
    "normalization": "zscore",
    "gpu_inference_started": False,
}

m = OUT / "preflight_manifest.json"
gate(not m.exists(), "Manifest already exists")

m.write_text(json.dumps(manifest, indent=2) + "\n")

print()
print("R1_3_VALIDATION_V3_PREFLIGHT=PASS")
print("MANIFEST_SHA256 =", sha(m))
