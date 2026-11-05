#!/usr/bin/env python3

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import eval_llm_verifier as E
from revision_utils import save_frozen_config


LANGUAGES = [
    "ruby",
    "javascript",
    "php",
    "python",
    "java",
    "go",
]

EXPECTED_N = {
    "ruby": 2209,
    "javascript": 8253,
    "php": 26015,
    "python": 23107,
    "java": 15328,
    "go": 14242,
}

VERIFY_KS = [5, 10, 15]
NORMALIZATIONS = ["none", "minmax", "zscore"]
ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0]

NORM_ORDER = {
    "none": 0,
    "minmax": 1,
    "zscore": 2,
}

ROOT = Path(
    "results/reviewer_revision/"
    "recal_valid_fresh_20260911"
)

SCORE_DIR = ROOT / "scores"
GRID_DIR = ROOT / "validation_grid"
POOL_DIR = GRID_DIR / "candidate_pools"
RANK_DIR = GRID_DIR / "selected_ranks"

GRID_DIR.mkdir(parents=True, exist_ok=True)
POOL_DIR.mkdir(parents=True, exist_ok=True)
RANK_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = (
    "/root/autodl-tmp/lora-code-embeddings-main/data/CSN"
)

NOMIC_MODEL = (
    "/root/autodl-tmp/hf_models/nomic-embed-code"
)

NOMIC_CACHE = (
    "data/rerank/coderank_cache_oldcsn"
)

VERIFIER_MODEL = (
    "/root/autodl-tmp/lora-code-embeddings-main/"
    "text2code/trained_verifiers/"
    "combined_qwen25coder7b_nomic_hn7_30k_attn_lora"
)

VERIFIER_BASE_MODEL = (
    "/root/autodl-tmp/hf_models/"
    "Qwen2.5-Coder-7B-Instruct"
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            x = f.read(1024 * 1024)
            if not x:
                break
            h.update(x)
    return h.hexdigest()


def exact_mrr_percent(ranks):
    x = np.asarray(ranks, dtype=np.float64)
    return float(
        np.mean(1.0 / (x + 1.0)) * 100.0
    )


# ------------------------------------------------------------
# PREDECLARE THE SELECTION PROTOCOL BEFORE LOOKING AT RESULTS
# ------------------------------------------------------------

protocol = {
    "selection_split": "valid",
    "selection_metric":
        "equal_language_macro_mrr",
    "selection_metric_precision":
        "unrounded_per_query_ranks",
    "verify_top_k": VERIFY_KS,
    "normalization": NORMALIZATIONS,
    "fusion_weight": ALPHAS,
    "n_configurations": 54,
    "test_metrics_used_for_selection": False,
    "tie_breaker": [
        "maximize exact macro MRR",
        "exact tie: smaller verify_top_k",
        "then none < minmax < zscore",
        "then smaller fusion_weight",
    ],
}

with open(
    GRID_DIR / "selection_protocol.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        protocol,
        f,
        indent=2,
        ensure_ascii=False,
    )
    f.write("\n")


# ------------------------------------------------------------
# 1. VERIFY THE SIX FRESH SCORE CACHES
# ------------------------------------------------------------

print("=" * 78)
print("1. FRESH VALID SCORE CACHE AUDIT")
print("=" * 78)

score_paths = {}

for lang in LANGUAGES:

    p = E.score_cache_path(
        cache_dir=str(SCORE_DIR),
        language=lang,
        split="valid",
        verifier_model=VERIFIER_MODEL,
        verifier_base_model=VERIFIER_BASE_MODEL,
        candidate_top_k=100,
        verify_top_k=15,
        max_length=1024,
        max_query_chars=512,
        max_code_chars=3000,
        query_offset=0,
        max_queries=None,
        verifier_mode="yesno",
        max_new_tokens=64,
        candidate_pool_key="dense",
    )

    if not p.exists():
        raise FileNotFoundError(
            f"Missing fresh cache: {p}"
        )

    arr = np.load(
        p,
        mmap_mode="r",
    )

    expected = (
        EXPECTED_N[lang],
        15,
    )

    if arr.shape != expected:
        raise RuntimeError(
            f"{lang}: "
            f"{arr.shape} != {expected}"
        )

    if not np.all(np.isfinite(arr)):
        raise RuntimeError(
            f"{lang}: non-finite scores"
        )

    score_paths[lang] = p

    print(
        f"{lang:<11} "
        f"shape={str(arr.shape):<16} "
        f"sha256={sha256_file(p)}"
    )

print("FRESH_SCORE_CACHE_AUDIT=PASSED")


# ------------------------------------------------------------
# 2. REPLAY VALID EVALUATOR, REUSING THE CACHES.
#    CAPTURE EXACT NOMIC TOP100 ARRAYS.
# ------------------------------------------------------------

captured = {}

original_rerank = (
    E.rerank_with_verifier
)

call_counter = [0]


def capture_rerank(
    candidate_indices,
    candidate_scores,
    verifier_scores,
    gt_indices,
    base_ranks,
    verify_top_k,
    fusion_alpha,
    normalization,
):
    idx = call_counter[0]

    if idx >= len(LANGUAGES):
        raise RuntimeError(
            "Unexpected rerank call"
        )

    lang = LANGUAGES[idx]

    captured[lang] = {
        "candidate_indices":
            np.asarray(
                candidate_indices
            ).copy(),

        "candidate_scores":
            np.asarray(
                candidate_scores
            ).copy(),

        "verifier_scores":
            np.asarray(
                verifier_scores
            ).copy(),

        "gt_indices":
            np.asarray(
                gt_indices
            ).copy(),

        "base_ranks":
            np.asarray(
                base_ranks
            ).copy(),
    }

    call_counter[0] += 1

    return original_rerank(
        candidate_indices=
            candidate_indices,
        candidate_scores=
            candidate_scores,
        verifier_scores=
            verifier_scores,
        gt_indices=
            gt_indices,
        base_ranks=
            base_ranks,
        verify_top_k=
            verify_top_k,
        fusion_alpha=
            fusion_alpha,
        normalization=
            normalization,
    )


E.rerank_with_verifier = (
    capture_rerank
)

old_argv = sys.argv[:]

sys.argv = [
    "eval_llm_verifier.py",

    "--data-root",
    DATA_ROOT,

    "--languages",
    *LANGUAGES,

    "--split",
    "valid",

    "--experiment-name",
    "reviewer_revision_valid_grid_capture",

    "--experiment-family",
    "reviewer_revision",

    "--nomic-model-path",
    NOMIC_MODEL,

    "--nomic-cache-dir",
    NOMIC_CACHE,

    "--nomic-max-length",
    "512",

    "--candidate-top-k",
    "100",

    "--verify-top-k",
    "15",

    "--verifier-model",
    VERIFIER_MODEL,

    "--verifier-base-model",
    VERIFIER_BASE_MODEL,

    "--verifier-dtype",
    "float16",

    "--verifier-device-map",
    "cuda",

    "--verifier-mode",
    "yesno",

    "--verifier-batch-size",
    "4",

    "--max-length",
    "1024",

    "--max-new-tokens",
    "64",

    "--max-query-chars",
    "512",

    "--max-code-chars",
    "3000",

    "--score-cache-dir",
    str(SCORE_DIR),

    "--reuse-score-cache",

    "--fusion-alphas",
    "0",

    "--fusion-normalization",
    "zscore",

    "--json-output",
    str(
        GRID_DIR /
        "capture_replay_eval.json"
    ),
]

try:
    E.main()
finally:
    sys.argv = old_argv
    E.rerank_with_verifier = (
        original_rerank
    )


if list(captured.keys()) != LANGUAGES:
    raise RuntimeError(
        "Did not capture exactly six "
        f"languages: {list(captured.keys())}"
    )


# ------------------------------------------------------------
# 3. CAPTURE AUDIT
# ------------------------------------------------------------

print()
print("=" * 78)
print("2. CANDIDATE POOL AUDIT")
print("=" * 78)

for lang in LANGUAGES:
    d = captured[lang]
    n = EXPECTED_N[lang]

    if (
        d["candidate_indices"].shape
        != (n, 100)
    ):
        raise RuntimeError(
            f"{lang}: bad candidate shape "
            f"{d['candidate_indices'].shape}"
        )

    if (
        d["candidate_scores"].shape
        != (n, 100)
    ):
        raise RuntimeError(
            f"{lang}: bad score shape "
            f"{d['candidate_scores'].shape}"
        )

    if (
        d["verifier_scores"].shape
        != (n, 15)
    ):
        raise RuntimeError(
            f"{lang}: bad verifier shape "
            f"{d['verifier_scores'].shape}"
        )

    if len(d["gt_indices"]) != n:
        raise RuntimeError(
            f"{lang}: bad GT length"
        )

    if len(d["base_ranks"]) != n:
        raise RuntimeError(
            f"{lang}: bad base rank length"
        )

    p = (
        POOL_DIR /
        f"{lang}_valid_nomic_top100.npz"
    )

    np.savez_compressed(
        p,
        candidate_indices=
            d["candidate_indices"],
        candidate_scores=
            d["candidate_scores"],
        verifier_scores=
            d["verifier_scores"],
        gt_indices=
            d["gt_indices"],
        base_ranks=
            d["base_ranks"],
    )

    print(
        f"{lang:<11} "
        f"candidates="
        f"{d['candidate_indices'].shape} "
        f"verifier="
        f"{d['verifier_scores'].shape}"
    )

print(
    "CANDIDATE_POOL_CAPTURE_AUDIT=PASSED"
)


# ------------------------------------------------------------
# 4. RUN ALL 54 VALIDATION CONFIGS
# ------------------------------------------------------------

print()
print("=" * 78)
print("3. VALIDATION-ONLY 54-CONFIG GRID")
print("=" * 78)

records = {}
selected_rank_cache = {}

for k in VERIFY_KS:
    for norm in NORMALIZATIONS:
        for alpha in ALPHAS:

            per_lang = {}
            total_rr = 0.0
            total_n = 0

            key = (
                k,
                norm,
                alpha,
            )

            selected_rank_cache[key] = {}

            for lang in LANGUAGES:
                d = captured[lang]

                ranks = original_rerank(
                    candidate_indices=
                        d["candidate_indices"],
                    candidate_scores=
                        d["candidate_scores"],
                    verifier_scores=
                        d["verifier_scores"],
                    gt_indices=
                        d["gt_indices"],
                    base_ranks=
                        d["base_ranks"],
                    verify_top_k=k,
                    fusion_alpha=alpha,
                    normalization=norm,
                )

                ranks = np.asarray(
                    ranks,
                    dtype=np.int32,
                )

                mrr = (
                    exact_mrr_percent(
                        ranks
                    )
                )

                per_lang[lang] = mrr

                total_rr += float(
                    np.sum(
                        1.0 /
                        (
                            ranks.astype(
                                np.float64
                            )
                            + 1.0
                        )
                    )
                )

                total_n += len(ranks)

                selected_rank_cache[
                    key
                ][lang] = ranks

            macro = float(
                np.mean(
                    [
                        per_lang[x]
                        for x
                        in LANGUAGES
                    ]
                )
            )

            micro = float(
                total_rr /
                total_n *
                100.0
            )

            records[key] = {
                "verify_top_k": k,
                "normalization": norm,
                "fusion_weight": alpha,
                "macro_mrr_exact": macro,
                "micro_mrr_exact": micro,
                "n_queries": total_n,
                "per_language_mrr":
                    per_lang,
            }

            print(
                f"k={k:2d} "
                f"norm={norm:<6} "
                f"alpha={alpha:<4g} "
                f"macro={macro:.8f} "
                f"micro={micro:.8f}"
            )


if len(records) != 54:
    raise RuntimeError(
        f"Expected 54 configs; "
        f"got {len(records)}"
    )


# ------------------------------------------------------------
# 5. VALIDATION-ONLY SELECTION
# ------------------------------------------------------------

def sort_key(r):
    return (
        -r["macro_mrr_exact"],
        r["verify_top_k"],
        NORM_ORDER[
            r["normalization"]
        ],
        r["fusion_weight"],
    )


ordered = sorted(
    records.values(),
    key=sort_key,
)

best = ordered[0]

best_score = (
    best["macro_mrr_exact"]
)

exact_ties = [
    r for r in ordered
    if r["macro_mrr_exact"]
    == best_score
]


# ------------------------------------------------------------
# 6. SAVE GRID JSON + CSV
# ------------------------------------------------------------

grid_json = (
    GRID_DIR /
    "validation_grid_54.json"
)

with open(
    grid_json,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            "protocol": protocol,
            "selected": best,
            "exact_best_tie_count":
                len(exact_ties),
            "records": ordered,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )
    f.write("\n")


grid_csv = (
    GRID_DIR /
    "validation_grid_54.csv"
)

fields = [
    "rank",
    "verify_top_k",
    "normalization",
    "fusion_weight",
    "macro_mrr_exact",
    "micro_mrr_exact",
    "n_queries",
]

for lang in LANGUAGES:
    fields.append(
        f"{lang}_mrr"
    )

with open(
    grid_csv,
    "w",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    writer.writeheader()

    for idx, r in enumerate(
        ordered,
        start=1,
    ):
        row = {
            "rank": idx,
            "verify_top_k":
                r["verify_top_k"],
            "normalization":
                r["normalization"],
            "fusion_weight":
                r["fusion_weight"],
            "macro_mrr_exact":
                r["macro_mrr_exact"],
            "micro_mrr_exact":
                r["micro_mrr_exact"],
            "n_queries":
                r["n_queries"],
        }

        for lang in LANGUAGES:
            row[
                f"{lang}_mrr"
            ] = (
                r[
                    "per_language_mrr"
                ][lang]
            )

        writer.writerow(row)


# ------------------------------------------------------------
# 7. SAVE SELECTED VALID RANKS
# ------------------------------------------------------------

best_key = (
    best["verify_top_k"],
    best["normalization"],
    best["fusion_weight"],
)

for lang in LANGUAGES:
    np.save(
        RANK_DIR /
        (
            f"{lang}_valid_"
            "selected.ranks.npy"
        ),
        selected_rank_cache[
            best_key
        ][lang],
    )


# ------------------------------------------------------------
# 8. FREEZE CONFIG
# ------------------------------------------------------------

frozen_path = (
    ROOT /
    "recal_main_frozen_config.json"
)

save_frozen_config(
    frozen_path,

    method=
        "recal_llm_verifier",

    selected_on_split=
        "valid",

    selected_metric=
        "equal_language_macro_mrr",

    config={
        "fusion_weight":
            float(
                best[
                    "fusion_weight"
                ]
            ),

        "normalization":
            best[
                "normalization"
            ],

        "verify_top_k":
            int(
                best[
                    "verify_top_k"
                ]
            ),

        "candidate_top_k":
            100,

        "verifier_mode":
            "yesno",
    },

    validation_result={
        "macro_mrr_exact":
            best[
                "macro_mrr_exact"
            ],

        "micro_mrr_exact":
            best[
                "micro_mrr_exact"
            ],

        "n_queries":
            best["n_queries"],

        "per_language_mrr":
            best[
                "per_language_mrr"
            ],

        "grid_size":
            54,

        "exact_best_tie_count":
            len(exact_ties),
    },
)


# ------------------------------------------------------------
# 9. SHA256 MANIFEST
# ------------------------------------------------------------

manifest = (
    GRID_DIR /
    "sha256_manifest.txt"
)

files_to_hash = [
    GRID_DIR /
        "selection_protocol.json",
    grid_json,
    grid_csv,
    frozen_path,
]

files_to_hash += list(
    score_paths.values()
)

files_to_hash += sorted(
    POOL_DIR.glob("*.npz")
)

files_to_hash += sorted(
    RANK_DIR.glob("*.npy")
)

with open(
    manifest,
    "w",
    encoding="utf-8",
) as f:
    for p in files_to_hash:
        f.write(
            f"{sha256_file(p)}  {p}\n"
        )


# ------------------------------------------------------------
# 10. REPORT
# ------------------------------------------------------------

print()
print("=" * 78)
print("TOP 10 VALIDATION CONFIGS")
print("=" * 78)

for i, r in enumerate(
    ordered[:10],
    start=1,
):
    print(
        f"{i:2d}. "
        f"k={r['verify_top_k']:2d} "
        f"norm={r['normalization']:<6} "
        f"alpha={r['fusion_weight']:<4g} "
        f"macro="
        f"{r['macro_mrr_exact']:.8f} "
        f"micro="
        f"{r['micro_mrr_exact']:.8f}"
    )

print()
print("=" * 78)
print("VALIDATION-SELECTED FROZEN CONFIG")
print("=" * 78)

print(
    "verify_top_k  =",
    best["verify_top_k"],
)

print(
    "normalization =",
    best["normalization"],
)

print(
    "fusion_weight =",
    best["fusion_weight"],
)

print(
    "macro MRR     =",
    f"{best['macro_mrr_exact']:.10f}",
)

print(
    "micro MRR     =",
    f"{best['micro_mrr_exact']:.10f}",
)

print(
    "exact ties    =",
    len(exact_ties),
)

print()
print(
    "Per-language validation MRR:"
)

for lang in LANGUAGES:
    print(
        f"  {lang:<11} "
        f"{best['per_language_mrr'][lang]:.10f}"
    )

print()
print(
    "FROZEN_CONFIG =",
    frozen_path,
)

print(
    "GRID_JSON     =",
    grid_json,
)

print(
    "GRID_CSV      =",
    grid_csv,
)

print(
    "SHA_MANIFEST  =",
    manifest,
)

print()
print(
    "VALIDATION_GRID_SELECTION=PASSED"
)
print(
    "DO_NOT_INSPECT_TEST_FOR_SELECTION"
)
