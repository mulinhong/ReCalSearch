from pathlib import Path
import hashlib
import json
import subprocess
import sys
import numpy as np
import torch

ROOT = Path.cwd()
BASE = ROOT / "results/reviewer_revision/r1_3_multilingual_matched_v1"
OUT = BASE / "validation_smoke_seed13_v2"

def require(ok, msg):
    if not ok:
        raise RuntimeError("FAIL_STOP: " + msg)

def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()

def check(p, expected):
    require(p.is_file(), f"Missing: {p}")
    require(sha(p) == expected, f"SHA mismatch: {p}")
    print("SHA256_PASS =", p.name)

check(
    ROOT / "eval_llm_verifier.py",
    "a3156c285728d54180ce0da1c65255a92adfe15f583dc987e47ad8a97b8e665a",
)

check(
    BASE / "posttrain_closure_seed13_v1/posttrain_closure.json",
    "077d2193f144378f86bd2eac8a8aebb674ab4f42e0fe18f8dbacee3f8fca5fc1",
)

config_path = (
    ROOT / "results/reviewer_revision/"
    "recal_valid_fresh_20260911/recal_main_frozen_config.json"
)

check(
    config_path,
    "70789e10518f32a97f5990b234833fb0443fbcbe426e8a8aebb674ab4f42e0fe18f8dbacee3f8fca5fc1"
    if False else
    "70789e10518f32a97f5990b234833fb0443fbcbe426e8a8c3515ab93e6b11b0f",
)

cfg = json.loads(config_path.read_text())
setting = cfg["config"]

require(cfg["selected_on_split"] == "valid", "Not validation-frozen")
require(float(setting["fusion_weight"]) == 1.0, "Fusion mismatch")
require(setting["normalization"] == "zscore", "Normalization mismatch")
require(int(setting["verify_top_k"]) == 15, "Depth mismatch")

model = BASE / "models/multi5k_seed13_v5"

check(
    model / "adapter_model.safetensors",
    "c02e0fa78a351bff68fd63e6e28128e112a83a42850cbe47ef606168e045cc59",
)

check(
    model / "adapter_config.json",
    "f8bebe2bb57b113462f49a0ba7935887327c11f080aa3086acf31fc746ac07c8",
)

expected_pool_hashes = {
    "ruby": "fc60c7012e7c2019f034e8d46171aa3220cf20c9e7c5afc3728aa56efa5c85ba",
    "javascript": "03b781900ae5a28820068392fcbd7ee0bc37f11b841d077ef18286b644f3446d",
    "php": "95b78c4e8ac688c22632b18011b496dcdfa8f1622f656e3868badb73e6945a8c",
    "python": "2ff29339c22c1c57bcab9e59d6e6eadf385c80baae335c78dd18f6a3909e1c43",
    "java": "b4329683443cdb75ff2b52ea23d09edb5bc0165ba290dd5e9e3b9ca7daae0f14",
    "go": "0bc7dab8f60d1867601dc63ac5817719d1fe5cb33610590393baef0daa7d0a1d",
}

for lang, expected in expected_pool_hashes.items():
    pool = (
        BASE / "validation_pool_adapter_v2/"
        "candidate_pools" / f"{lang}_valid_nomic_top100.npz"
    )
    check(pool, expected)

with np.load(
    BASE / "validation_pool_adapter_v2/"
    "candidate_pools/ruby_valid_nomic_top100.npz",
    allow_pickle=False,
) as pool:
    require(pool["indices"].shape == (2209, 100), "Pool shape mismatch")
    require(pool["scores"].shape == (2209, 100), "Score shape mismatch")
    require(pool["base_ranks"].shape == (2209,), "Rank shape mismatch")

require(torch.cuda.is_available(), "CUDA unavailable")

from eval_llm_verifier import coderank_cache_paths

cache = coderank_cache_paths(
    cache_dir="data/rerank/coderank_cache",
    language="ruby",
    split="valid",
    model_name="/root/autodl-tmp/hf_models/nomic-embed-code",
    max_seq_length=512,
    backend="nomic_transformers",
    query_prefix="Represent this query for searching relevant code: ",
    code_prefix="",
)

for key in ("query", "code"):
    p = Path(cache[key])
    print("NOMIC_CACHE =", key, p)
    require(p.is_file(), f"Missing Nomic cache: {p}")

require(
    (ROOT.parent / "data/CSN/ruby/valid.jsonl").is_file(),
    "Canonical Ruby validation data missing"
)

print("SPLIT=valid")
print("SMOKE_QUERIES=8")
print("FUSION_ALPHA=1")
print("NORMALIZATION=zscore")
print("VERIFY_TOP_K=15")
print("NO_TEST_ACCESS=TRUE")
print("VALIDATION_SMOKE_PREFLIGHT=PASS")
