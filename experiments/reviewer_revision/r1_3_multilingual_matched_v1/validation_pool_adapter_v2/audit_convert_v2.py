from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(
    "/root/autodl-tmp/lora-code-embeddings-main/text2code"
)

BASE = ROOT / (
    "results/reviewer_revision/"
    "r1_3_multilingual_matched_v1"
)

SOURCE = ROOT / (
    "results/reviewer_revision/"
    "recal_valid_fresh_20260911/"
    "validation_grid/candidate_pools"
)

OUT = BASE / "validation_pool_adapter_v2"
DEST = OUT / "candidate_pools"

DATA = ROOT.parent / "data/CSN"
CACHE = ROOT / "data/rerank/coderank_cache_oldcsn"

SCHEMA_MANIFEST = (
    BASE / "training_protocol_audit_v2"
    / "validation_schema_audit.json"
)

EXPECTED_N = {
    "ruby": 2209,
    "javascript": 8253,
    "php": 26015,
    "python": 23107,
    "java": 15328,
    "go": 14242,
}

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(chunk)
    return h.hexdigest()

def require(ok, message):
    if not ok:
        raise RuntimeError("FAIL_STOP: " + str(message))

def count_jsonl(path):
    require(path.is_file(), f"Missing file: {path}")
    with path.open("rb") as f:
        return sum(1 for line in f if line.strip())

def load_pool(path):
    require(path.is_file(), f"Missing pool: {path}")
    with np.load(path, allow_pickle=False) as z:
        fields = [
            "candidate_indices",
            "candidate_scores",
            "gt_indices",
            "base_ranks",
        ]
        require(
            all(k in z.files for k in fields),
            f"Missing required fields: {path}",
        )
        return {k: z[k].copy() for k in fields}

print("=" * 105)
print("WORK1 R1-3 VALIDATION POOL ADAPTER V2")
print("STRICT-GREATER RANK / TIE-AWARE AUDIT")
print("=" * 105)

print("NO GPU")
print("NO MODEL TRAINING")
print("NO DENSE RETRIEVAL")
print("NO TEST DATA")
print("NO HISTORICAL ARTIFACT MODIFICATION")

require(
    SCHEMA_MANIFEST.is_file(),
    "Previous schema audit missing",
)

require(
    sha256(SCHEMA_MANIFEST)
    == "e9fd1481d92b417a86c5772eebc999f8812796baa6f26c91912add83369d3861",
    "Previous schema audit SHA mismatch",
)

schema = json.loads(
    SCHEMA_MANIFEST.read_text(encoding="utf-8")
)

summary = {
    "version": "r1_3_validation_adapter_v2",
    "rank_definition":
        "Count all codebase scores strictly greater "
        "than the GT score, not GT top-k position.",
    "source_pools": {},
    "diagnostics": {},
    "converted_pools": {},
    "conversion_allowed": False,
    "historical_verifier_scores_reused": False,
    "independent_canonical_gt_replay":
        "NOT_YET_CHECKED",
}

all_valid = True
source_hashes = {}

print()
print("=" * 105)
print("PHASE 1: ALL-SIX VALIDATION AND TIE CLASSIFICATION")
print("=" * 105)

for lang, n in EXPECTED_N.items():

    src = SOURCE / f"{lang}_valid_nomic_top100.npz"
    require(src.is_file(), f"Missing pool {lang}")

    source_hashes[lang] = sha256(src)

    previous_sha = schema["languages"][lang]["sha256"]

    require(
        source_hashes[lang] == previous_sha,
        f"{lang}: source SHA differs from prior audit",
    )

    arrays = load_pool(src)

    idx = arrays["candidate_indices"]
    scores = arrays["candidate_scores"]
    gt = arrays["gt_indices"]
    ranks = arrays["base_ranks"]

    require(
        idx.shape == (n, 100),
        f"{lang}: index shape mismatch",
    )
    require(
        scores.shape == (n, 100),
        f"{lang}: score shape mismatch",
    )
    require(
        gt.shape == (n,)
        and ranks.shape == (n,),
        f"{lang}: rank/GT shape mismatch",
    )

    require(
        np.issubdtype(idx.dtype, np.integer)
        and np.issubdtype(gt.dtype, np.integer)
        and np.issubdtype(ranks.dtype, np.integer),
        f"{lang}: integer dtype mismatch",
    )

    require(
        np.issubdtype(scores.dtype, np.floating)
        and np.isfinite(scores).all(),
        f"{lang}: score type/finite mismatch",
    )

    codebase = DATA / lang / "codebase.jsonl"
    validfile = DATA / lang / "valid.jsonl"

    ncodes = count_jsonl(codebase)
    nvalid = count_jsonl(validfile)

    require(
        nvalid == n,
        f"{lang}: original validation split count mismatch",
    )

    require(
        np.all(idx >= 0)
        and np.all(idx < ncodes)
        and np.all(gt >= 0)
        and np.all(gt < ncodes),
        f"{lang}: candidate or GT index out of range",
    )

    require(
        np.all(ranks >= 0)
        and np.all(ranks < ncodes),
        f"{lang}: base rank outside codebase",
    )

    # Verify no duplicated codebase IDs inside top100.
    sorted_idx = np.sort(idx, axis=1)

    require(
        np.all(np.diff(sorted_idx, axis=1) != 0),
        f"{lang}: duplicate top100 indices",
    )

    # Verify Nomic retrieval scores are descending.
    require(
        np.all(np.diff(scores, axis=1) <= 1e-5),
        f"{lang}: candidate scores not descending",
    )

    caches = sorted(
        CACHE.glob(f"{lang}_valid_*.query.npy")
    )

    require(
        len(caches) == 1,
        f"{lang}: ambiguous/missing query cache",
    )

    ca = np.load(
        caches[0],
        mmap_mode="r",
        allow_pickle=False,
    )

    require(
        ca.ndim == 2 and ca.shape[0] == n,
        f"{lang}: query cache count mismatch",
    )

    # Historical Nomic rank is:
    # sum(all_code_scores > ground_truth_score).
    #
    # GT position in torch.topk need not equal that
    # value when tied scores occupy different slots.

    matches = idx == gt[:, None]
    hits_per_query = matches.sum(axis=1)

    require(
        np.all(hits_per_query <= 1),
        f"{lang}: duplicate GT in candidate list",
    )

    hit = hits_per_query == 1
    miss = ~hit

    hit_ids = np.flatnonzero(hit)

    positions = np.argmax(
        matches[hit],
        axis=1,
    )

    original_ranks = ranks[hit]

    gt_scores_in_topk = scores[
        hit_ids,
        positions,
    ]

    # Because GT appears in top100, every score
    # strictly greater than GT must also be there.
    # Therefore this count must equal the original
    # full-codebase strict-greater rank.
    strictly_higher_in_topk = (
        scores[hit] >
        gt_scores_in_topk[:, None]
    ).sum(axis=1)

    position_equal = (
        positions == original_ranks
    )

    strict_rank_equal = (
        strictly_higher_in_topk == original_ranks
    )

    position_mismatch = ~position_equal

    position_mismatch_n = int(
        position_mismatch.sum()
    )

    strict_failure_n = int(
        (~strict_rank_equal).sum()
    )

    # Differences explained by tied scores:
    # GT appears later among equal-scoring items,
    # but its strict-greater rank is unchanged.
    tie_explained = (
        position_mismatch & strict_rank_equal
    )

    tie_explained_n = int(tie_explained.sum())

    unexplained = np.flatnonzero(
        ~strict_rank_equal
    )

    samples = []

    for k in unexplained[:12]:
        row_id = int(hit_ids[k])
        pos = int(positions[k])

        samples.append({
            "query_row": row_id,
            "gt_index": int(gt[row_id]),
            "gt_top100_position": pos,
            "saved_base_rank": int(ranks[row_id]),
            "strictly_higher_in_top100":
                int(strictly_higher_in_topk[k]),
            "gt_score": float(scores[row_id, pos]),
        })

    # It is possible for GT to be absent from
    # torch.topk even if its strict-greater rank
    # is <100: a tie at the top-k boundary can
    # exclude it. Report, do not silently rewrite.
    missing_gt_with_rank_lt100 = int(
        (ranks[miss] < 100).sum()
    )

    lang_valid = strict_failure_n == 0

    if not lang_valid:
        all_valid = False

    diagnostic = {
        "n_queries": n,
        "n_codebase": ncodes,
        "gt_hit_top100": int(hit.sum()),
        "gt_miss_top100": int(miss.sum()),
        "gt_position_equals_base_rank":
            int(position_equal.sum()),
        "gt_position_differs_base_rank":
            position_mismatch_n,
        "position_mismatches_explained_by_ties":
            tie_explained_n,
        "strict_greater_rank_failures":
            strict_failure_n,
        "gt_missing_with_rank_under100":
            missing_gt_with_rank_lt100,
        "unexplained_examples": samples,
        "strict_rank_gate":
            "PASS" if lang_valid else "FAIL",
    }

    summary["source_pools"][lang] = {
        "path": str(src),
        "sha256": source_hashes[lang],
    }

    summary["diagnostics"][lang] = diagnostic

    print()
    print("LANGUAGE =", lang)

    print(
        "QUERIES =", n,
        "CODEBASE =", ncodes,
    )

    print(
        "GT_HIT =", int(hit.sum()),
        "GT_MISS =", int(miss.sum()),
    )

    print(
        "GT_POSITION_MISMATCH =",
        position_mismatch_n,
    )

    print(
        "TIE_EXPLAINED_MISMATCH =",
        tie_explained_n,
    )

    print(
        "STRICT_RANK_UNEXPLAINED =",
        strict_failure_n,
    )

    print(
        "GT_MISSING_WITH_RANK_LT100 =",
        missing_gt_with_rank_lt100,
    )

    print(
        "LANGUAGE_STRICT_RANK_GATE =",
        diagnostic["strict_rank_gate"],
        flush=True,
    )

# Always preserve a diagnostic report, including
# any detected failure, before deciding whether
# conversion is authorized.
diagnostic_path = OUT / "tie_rank_diagnostics.json"

require(
    not diagnostic_path.exists(),
    "Diagnostic report already exists",
)

diagnostic_path.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ) + "\n",
    encoding="utf-8",
)

print()
print("=" * 105)
print("PHASE 1 FINAL")
print("=" * 105)

print(
    "ALL_SIX_STRICT_RANK_GATES =",
    "PASS" if all_valid else "FAIL",
)

print(
    "DIAGNOSTICS_SHA256 =",
    sha256(diagnostic_path),
)

if not all_valid:
    print("CONVERSION = BLOCKED")
    print("NO_COMPATIBLE_POOLS_WRITTEN = TRUE")
    raise SystemExit(2)

# Only after all six languages pass:
# create evaluator-compatible copies.
print()
print("=" * 105)
print("PHASE 2: SAFE SCHEMA CONVERSION")
print("=" * 105)

require(
    not DEST.exists(),
    "Converted pool directory already exists",
)

DEST.mkdir(parents=True)

for lang in EXPECTED_N:

    src = SOURCE / f"{lang}_valid_nomic_top100.npz"
    a = load_pool(src)

    dst = DEST / f"{lang}_valid_nomic_top100.npz"

    require(not dst.exists(), f"Destination exists: {dst}")

    np.savez_compressed(
        dst,
        indices=a["candidate_indices"],
        scores=a["candidate_scores"],
        base_ranks=a["base_ranks"],
    )

    with np.load(dst, allow_pickle=False) as z:

        require(
            set(z.files)
            == {"indices", "scores", "base_ranks"},
            f"{lang}: converted schema mismatch",
        )

        require(
            np.array_equal(
                z["indices"],
                a["candidate_indices"],
            ),
            f"{lang}: indices changed",
        )

        require(
            np.array_equal(
                z["scores"],
                a["candidate_scores"],
            ),
            f"{lang}: scores changed",
        )

        require(
            np.array_equal(
                z["base_ranks"],
                a["base_ranks"],
            ),
            f"{lang}: base ranks changed",
        )

    require(
        sha256(src) == source_hashes[lang],
        f"{lang}: original source changed",
    )

    summary["converted_pools"][lang] = {
        "path": str(dst),
        "sha256": sha256(dst),
        "array_parity": "PASS",
    }

    print(
        f"{lang:<12} CONVERSION=PASS "
        f"ARRAY_PARITY=PASS "
        f"SHA256={sha256(dst)}",
        flush=True,
    )

summary["conversion_allowed"] = True
summary["conversion_complete"] = True

summary["gates"] = {
    "six_language_strict_rank_semantics": "PASS",
    "source_sha256_preserved": "PASS",
    "converted_array_parity": "PASS",
    "historical_verifier_scores_excluded": "PASS",
    "independent_dataset_gt_identity":
        "NOT_YET_CHECKED",
}

manifest_path = OUT / "conversion_manifest.json"

require(
    not manifest_path.exists(),
    "Conversion manifest already exists",
)

manifest_path.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ) + "\n",
    encoding="utf-8",
)

print()
print("=" * 105)
print("FINAL GATES")
print("=" * 105)

print("SIX_LANGUAGE_VALIDATION_ADAPTER = PASS")
print("ALL_SIX_STRICT_RANK_GATES = PASS")
print("SOURCE_ARRAY_PARITY = PASS")
print("ORIGINAL_VALIDATION_POOL_UNCHANGED = TRUE")
print("HISTORICAL_VERIFIER_SCORES_REUSED = FALSE")
print("INDEPENDENT_GT_IDENTITY_CHECK = PENDING")
print("R1_3_GPU_TRAINING_STARTED = FALSE")
print("NO_TEST_ACCESS = TRUE")

print("CONVERSION_MANIFEST =", manifest_path)
print("CONVERSION_MANIFEST_SHA256 =", sha256(manifest_path))
