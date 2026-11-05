from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(
    "/root/autodl-tmp/lora-code-embeddings-main/text2code"
)

SOURCE = ROOT / (
    "results/reviewer_revision/"
    "recal_valid_fresh_20260911/"
    "validation_grid/candidate_pools"
)

OUT = ROOT / (
    "results/reviewer_revision/"
    "r1_3_multilingual_matched_v1/"
    "validation_pool_adapter_v1"
)

POOL_OUT = OUT / "candidate_pools"

CACHE = ROOT / "data/rerank/coderank_cache_oldcsn"
DATA = ROOT.parent / "data/CSN"

EXPECTED = {
    "ruby": 2209,
    "javascript": 8253,
    "php": 26015,
    "python": 23107,
    "java": 15328,
    "go": 14242,
}

FIELDS = (
    "candidate_indices",
    "candidate_scores",
    "verifier_scores",
    "gt_indices",
    "base_ranks",
)

def gate(condition, message):
    if not condition:
        raise RuntimeError("FAIL_STOP: " + message)

def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1048576),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()

def load_source(lang):
    path = SOURCE / f"{lang}_valid_nomic_top100.npz"

    gate(path.is_file(), f"Missing source {path}")

    with np.load(path, allow_pickle=False) as z:
        gate(
            set(FIELDS).issubset(set(z.files)),
            f"{lang}: source fields inconsistent",
        )

        return {
            name: z[name].copy()
            for name in FIELDS
        }

def check_source(lang, arrays):
    n = EXPECTED[lang]

    idx = arrays["candidate_indices"]
    score = arrays["candidate_scores"]
    verify = arrays["verifier_scores"]
    gt = arrays["gt_indices"]
    ranks = arrays["base_ranks"]

    gate(
        idx.shape == (n, 100),
        f"{lang}: candidate shape mismatch",
    )

    gate(
        score.shape == (n, 100),
        f"{lang}: score shape mismatch",
    )

    gate(
        verify.shape == (n, 15),
        f"{lang}: historical verifier shape mismatch",
    )

    gate(
        gt.shape == (n,) and ranks.shape == (n,),
        f"{lang}: ground truth/rank shape mismatch",
    )

    gate(
        np.issubdtype(idx.dtype, np.integer)
        and np.issubdtype(gt.dtype, np.integer)
        and np.issubdtype(ranks.dtype, np.integer),
        f"{lang}: integer dtype mismatch",
    )

    gate(
        np.issubdtype(score.dtype, np.floating)
        and np.issubdtype(verify.dtype, np.floating),
        f"{lang}: float dtype mismatch",
    )

    gate(
        np.isfinite(score).all()
        and np.isfinite(verify).all(),
        f"{lang}: nonfinite scores",
    )

    gate(
        (idx >= 0).all()
        and (gt >= 0).all()
        and (ranks >= 0).all(),
        f"{lang}: invalid negative index/rank",
    )

    # Every candidate row must contain 100
    # distinct codebase indices.
    sorted_indices = np.sort(idx, axis=1)

    gate(
        np.all(np.diff(sorted_indices, axis=1) != 0),
        f"{lang}: duplicate top100 candidates",
    )

    # Candidate scores should be ordered by
    # the original Nomic retrieval ranking.
    score_differences = np.diff(
        score.astype(np.float64),
        axis=1,
    )

    gate(
        np.all(score_differences <= 1e-5),
        f"{lang}: retrieval scores not descending",
    )

    hits = idx == gt[:, None]
    hit_count = hits.sum(axis=1)

    gate(
        (hit_count <= 1).all(),
        f"{lang}: duplicate GT occurrences",
    )

    hit_mask = hit_count == 1

    # The original GT's position in the retrieved
    # top100 must equal its zero-based base rank.
    position = np.argmax(hits, axis=1)

    gate(
        np.array_equal(
            ranks[hit_mask],
            position[hit_mask],
        ),
        f"{lang}: GT/base-rank semantic mismatch",
    )

    miss_mask = ~hit_mask

    # A query whose GT is outside top100 must
    # not claim a rank inside top100.
    gate(
        np.all(ranks[miss_mask] >= 100),
        f"{lang}: outside-top100 base rank mismatch",
    )

    # Check row alignment with the validation
    # query embedding cache.
    query_caches = sorted(
        CACHE.glob(f"{lang}_valid_*.query.npy")
    )

    gate(
        len(query_caches) == 1,
        f"{lang}: ambiguous query cache",
    )

    cache = np.load(
        query_caches[0],
        mmap_mode="r",
        allow_pickle=False,
    )

    gate(
        cache.ndim == 2
        and cache.shape[0] == n,
        f"{lang}: query cache count mismatch",
    )

    valid_file = DATA / lang / "valid.jsonl"

    gate(
        valid_file.is_file(),
        f"{lang}: missing original valid split",
    )

    with valid_file.open("r", encoding="utf-8") as f:
        valid_count = sum(
            1 for line in f if line.strip()
        )

    gate(
        valid_count == n,
        f"{lang}: original valid query count mismatch",
    )

    return {
        "queries": n,
        "top100_hit_count": int(hit_mask.sum()),
        "top100_miss_count": int(miss_mask.sum()),
        "top100_hit_rate": float(hit_mask.mean()),
        "base_mrr": float(
            np.mean(
                1.0 / (
                    ranks.astype(np.float64) + 1.0
                )
            )
        ),
        "rank_semantic_gate": "PASS",
        "candidate_uniqueness_gate": "PASS",
        "retrieval_score_order_gate": "PASS",
        "valid_split_count_gate": "PASS",
        "query_cache_count_gate": "PASS",
        "original_query_cache": str(query_caches[0]),
    }

print("=" * 105)
print("WORK1 R1-3 VALIDATION POOL ADAPTER V1")
print("=" * 105)
print("NO TRAINING / NO GPU / NO DENSE RETRIEVAL")
print("=" * 105)

generator = ROOT / "select_recal_valid_grid.py"
evaluator = ROOT / "eval_llm_verifier.py"

gate(generator.is_file(), "Historical generator missing")
gate(evaluator.is_file(), "Evaluator missing")

generator_text = generator.read_text(
    encoding="utf-8",
    errors="replace",
)

evaluator_text = evaluator.read_text(
    encoding="utf-8",
    errors="replace",
)

for field in FIELDS:
    gate(
        field in generator_text,
        f"Historical generator lacks field: {field}",
    )

for field in ("indices", "scores", "base_ranks"):
    gate(
        f'pool["{field}"]' in evaluator_text,
        f"Evaluator contract lacks field: {field}",
    )

print("GENERATOR_FIELD_PROVENANCE = PASS")
print("EVALUATOR_FIELD_CONTRACT = PASS")

source_hashes = {}
validated = {}

# Phase 1: validate all six original pools
# before writing any converted files.
print()
print("[PHASE 1] ORIGINAL VALIDATION")

for lang in EXPECTED:
    src = SOURCE / f"{lang}_valid_nomic_top100.npz"

    source_hashes[lang] = sha256(src)

    arrays = load_source(lang)
    validated[lang] = check_source(lang, arrays)

    print(
        f"{lang:<12} "
        f"N={validated[lang]['queries']} "
        f"GT_HIT={validated[lang]['top100_hit_count']} "
        "RANK=PASS SCORE=PASS CACHE=PASS",
        flush=True,
    )

print()
print("ALL_SIX_ORIGINAL_POOL_GATES = PASS")

# Phase 2: write new compatible copies.
# The historical verifier_scores array is intentionally
# excluded: every new R1-3 model must infer its own.
POOL_OUT.mkdir(parents=True, exist_ok=False)

artifacts = {}

print()
print("[PHASE 2] SAFE COMPATIBILITY EXPORT")

for lang in EXPECTED:

    arrays = load_source(lang)

    dst = (
        POOL_OUT
        / f"{lang}_valid_nomic_top100.npz"
    )

    gate(not dst.exists(), f"Destination exists: {dst}")

    np.savez_compressed(
        dst,
        indices=arrays["candidate_indices"],
        scores=arrays["candidate_scores"],
        base_ranks=arrays["base_ranks"],
    )

    with np.load(dst, allow_pickle=False) as z:

        gate(
            set(z.files)
            == {"indices", "scores", "base_ranks"},
            f"{lang}: converted field mismatch",
        )

        gate(
            np.array_equal(
                z["indices"],
                arrays["candidate_indices"],
            ),
            f"{lang}: index conversion mismatch",
        )

        gate(
            np.array_equal(
                z["scores"],
                arrays["candidate_scores"],
            ),
            f"{lang}: score conversion mismatch",
        )

        gate(
            np.array_equal(
                z["base_ranks"],
                arrays["base_ranks"],
            ),
            f"{lang}: rank conversion mismatch",
        )

    src = SOURCE / f"{lang}_valid_nomic_top100.npz"

    gate(
        sha256(src) == source_hashes[lang],
        f"{lang}: historical source changed",
    )

    artifacts[lang] = {
        "source": str(src),
        "source_sha256": source_hashes[lang],
        "converted": str(dst),
        "converted_sha256": sha256(dst),
        "validation": validated[lang],
    }

    print(
        f"{lang:<12} "
        "EXACT_ARRAY_PARITY=PASS "
        f"SHA256={artifacts[lang]['converted_sha256']}",
        flush=True,
    )

manifest = {
    "format": "r1_3_validation_pool_adapter_v1",
    "split": "valid",
    "candidate_top_k": 100,
    "verify_top_k": 15,
    "source_generator": str(generator),
    "source_generator_sha256": sha256(generator),
    "evaluator_sha256": sha256(evaluator),
    "source_fields": list(FIELDS),
    "converted_fields": [
        "indices", "scores", "base_ranks"
    ],
    "mapping": {
        "candidate_indices": "indices",
        "candidate_scores": "scores",
        "base_ranks": "base_ranks",
    },
    "historical_verifier_scores_reused": False,
    "test_data_accessed": False,
    "artifacts": artifacts,
    "gates": {
        "all_source_pools_valid": "PASS",
        "source_field_provenance": "PASS",
        "candidate_and_rank_semantics": "PASS",
        "array_conversion_exact": "PASS",
        "original_source_sha256_unchanged": "PASS",
        "canonical_gt_identity_independent_replay":
            "NOT_CHECKED",
    },
}

manifest_path = OUT / "conversion_manifest.json"

gate(
    not manifest_path.exists(),
    "Manifest already exists",
)

manifest_path.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)

print()
print("=" * 105)
print("FINAL RESULTS")
print("=" * 105)
print("SIX_LANGUAGE_VALIDATION_ADAPTER = PASS")
print("ORIGINAL_VALIDATION_POOL_UNCHANGED = TRUE")
print("HISTORICAL_VERIFIER_SCORES_REUSED = FALSE")
print("NO_TEST_ACCESS = TRUE")
print("NO_GPU = TRUE")
print("NO_TRAINING = TRUE")
print("CONVERSION_MANIFEST =", manifest_path)
print("CONVERSION_MANIFEST_SHA256 =", sha256(manifest_path))
print("=" * 105)
