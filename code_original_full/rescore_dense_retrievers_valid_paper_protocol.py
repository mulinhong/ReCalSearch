#!/usr/bin/env python3

import csv
import json
from pathlib import Path

import numpy as np
import torch

from data_utils import create_dataset
from eval_llm_verifier import text2code_to_array
from eval_nomic_embed import compute_retrieval_metrics_from_cache


LANGS = [
    "ruby",
    "javascript",
    "php",
    "python",
    "java",
    "go",
]

DATA_ROOT = (
    "/root/autodl-tmp/"
    "lora-code-embeddings-main/data/CSN"
)

NOMIC_ROOT = Path(
    "data/rerank/coderank_cache_oldcsn"
)

UNIX_ROOT = Path(
    "results/work2_candidate_recovery_oldcsn/"
    "external_candidates/unixcoder_base_cache"
)

ROOT = Path(
    "results/reviewer_revision/"
    "hn_source_ablation/"
    "retriever_strength_valid"
)

JINA_ROOT = ROOT / "jina_cache"

OUT = ROOT / "paper_protocol"
RANKS = OUT / "ranks"

OUT.mkdir(parents=True, exist_ok=True)
RANKS.mkdir(parents=True, exist_ok=True)


# Sealed fresh-validation Nomic results from the
# actual ReCalSearch evaluation pipeline.
EXPECTED_NOMIC = {
    "ruby": {
        "r1": 64.4636,
        "r5": 80.0815,
        "r10": 84.1557,
        "mrr": 71.7640,
    },
    "javascript": {
        "r1": 46.2377,
        "r5": 64.0010,
        "r10": 68.7750,
        "mrr": 54.3066,
    },
    "php": {
        "r1": 33.8728,
        "r5": 52.3121,
        "r10": 58.3625,
        "mrr": 42.4904,
    },
    "python": {
        "r1": 57.4198,
        "r5": 76.7516,
        "r10": 81.8150,
        "mrr": 66.2151,
    },
    "java": {
        "r1": 47.2077,
        "r5": 66.7732,
        "r10": 71.3531,
        "mrr": 56.1273,
    },
    "go": {
        "r1": 71.8579,
        "r5": 88.2882,
        "r10": 91.7989,
        "mrr": 79.2909,
    },
}


def nomic_paths(lang):
    tag = (
        "root_autodl-tmp_hf_models_"
        "nomic-embed-code_msl512_14805c3aa7"
    )
    return {
        "query": NOMIC_ROOT / f"{lang}_valid_{tag}.query.npy",
        "code": NOMIC_ROOT / f"{lang}_{tag}.code.npy",
    }


def unix_paths(lang):
    return {
        "query": UNIX_ROOT / (
            f"{lang}_valid_unixcoder_base_mean_msl512.query.npy"
        ),
        "code": UNIX_ROOT / (
            f"{lang}_unixcoder_base_mean_msl512.code.npy"
        ),
    }


def jina_paths(lang):
    return {
        "query": JINA_ROOT / (
            f"{lang}_valid_jina_code_0p5b_"
            "nl2code_msl512.query.npy"
        ),
        "code": JINA_ROOT / (
            f"{lang}_jina_code_0p5b_"
            "nl2code_msl512.code.npy"
        ),
    }


def exact_metrics(ranks):
    ranks = np.asarray(ranks, dtype=np.int64)

    return {
        "n_queries": int(len(ranks)),
        "r1": 100.0 * float(np.mean(ranks < 1)),
        "r5": 100.0 * float(np.mean(ranks < 5)),
        "r10": 100.0 * float(np.mean(ranks < 10)),
        "r100": 100.0 * float(np.mean(ranks < 100)),
        "mrr": 100.0 * float(
            np.mean(
                1.0 / (
                    ranks.astype(np.float64) + 1.0
                )
            )
        ),
    }


device = torch.device("cuda")

results = {
    "protocol":
        "reviewer_r1_1_retriever_strength_"
        "valid_paper_gt_v1",
    "split": "valid",
    "ground_truth":
        "data_utils.create_dataset(...)."
        "val_dataset.text2code",
    "test_accessed": False,
    "retrievers": {},
}

pooled = {
    "nomic": [],
    "unixcoder": [],
    "jina": [],
}

for lang in LANGS:

    print()
    print("=" * 78)
    print("LANGUAGE =", lang)
    print("=" * 78)

    _, val_dataset, _, code_dataset = create_dataset(
        DATA_ROOT,
        lang,
    )

    gt = text2code_to_array(
        val_dataset.text2code,
        len(val_dataset),
    )

    gt = np.asarray(
        gt,
        dtype=np.int64,
    )

    paths = {
        "nomic": nomic_paths(lang),
        "unixcoder": unix_paths(lang),
        "jina": jina_paths(lang),
    }

    for name, p in paths.items():

        q = np.load(
            p["query"],
            mmap_mode="r",
        )

        c = np.load(
            p["code"],
            mmap_mode="r",
        )

        if q.shape[0] != len(gt):
            raise RuntimeError(
                f"{lang}/{name}: query rows "
                f"{q.shape[0]} != {len(gt)}"
            )

        if np.any(gt < 0) or np.any(gt >= c.shape[0]):
            raise RuntimeError(
                f"{lang}/{name}: bad GT indices"
            )

        rank_file = (
            RANKS /
            f"{lang}_{name}_valid_paper_gt.ranks.npy"
        )

        print(
            f"{lang:<11} {name:<11} "
            f"q={q.shape} c={c.shape}"
        )

        compute_retrieval_metrics_from_cache(
            query_embeds=q,
            code_embeds=c,
            gt_indices=torch.from_numpy(gt),
            device=device,
            query_chunk_size=64,
            code_chunk_size=4096,
            rank_output=str(rank_file),
        )

        ranks = np.load(rank_file)

        m = exact_metrics(ranks)

        results["retrievers"].setdefault(
            name,
            {},
        )[lang] = m

        pooled[name].append(ranks)

        print(json.dumps(m, indent=2))


# --------------------------------------------------
# Mandatory parity gate:
# Nomic MUST reproduce sealed ReCalSearch validation
# numbers before Jina/UniXcoder can be interpreted.
# --------------------------------------------------

parity = True

for lang in LANGS:

    got = results["retrievers"]["nomic"][lang]
    exp = EXPECTED_NOMIC[lang]

    fields = ["r1", "r5", "r10", "mrr"]

    ok = all(
        round(got[k], 4) == round(exp[k], 4)
        for k in fields
    )

    parity &= ok

    print(
        f"PARITY {lang:<11} "
        f"got_mrr={got['mrr']:.4f} "
        f"expected={exp['mrr']:.4f} "
        f"PASS={ok}"
    )

print()
print(
    "NOMIC_PAPER_PROTOCOL_PARITY_PASS=",
    parity,
)

if not parity:
    raise SystemExit(
        "STOP: Nomic parity failed. "
        "Do not interpret alternative retrievers."
    )


# --------------------------------------------------
# Only after parity passes: macro / micro summaries.
# --------------------------------------------------

for name in [
    "nomic",
    "unixcoder",
    "jina",
]:

    lang_rows = (
        results["retrievers"][name]
    )

    results["retrievers"][name]["macro"] = {
        metric: float(
            np.mean(
                [
                    lang_rows[l][metric]
                    for l in LANGS
                ]
            )
        )
        for metric in [
            "r1",
            "r5",
            "r10",
            "r100",
            "mrr",
        ]
    }

    pr = np.concatenate(
        pooled[name]
    )

    results["retrievers"][name]["micro"] = (
        exact_metrics(pr)
    )


nomic_mrr = (
    results["retrievers"]
    ["nomic"]["macro"]["mrr"]
)

jina_mrr = (
    results["retrievers"]
    ["jina"]["macro"]["mrr"]
)

unix_mrr = (
    results["retrievers"]
    ["unixcoder"]["macro"]["mrr"]
)

jina_gap = jina_mrr - nomic_mrr
unix_gap = unix_mrr - nomic_mrr


# This rule was frozen before seeing the
# corrected benchmark. It is a control-selection
# heuristic, not a reviewer-mandated threshold.
if abs(jina_gap) <= 3.0:
    jina_class = "performance_comparable"
elif abs(jina_gap) <= 5.0:
    jina_class = (
        "independent_dense_"
        "but_not_matched_strength"
    )
else:
    jina_class = "not_matched_strength"


results["selection"] = {
    "primary_metric":
        "equal_language_macro_mrr",
    "nomic_macro_mrr":
        nomic_mrr,
    "jina_macro_mrr":
        jina_mrr,
    "unixcoder_macro_mrr":
        unix_mrr,
    "jina_minus_nomic_pp":
        jina_gap,
    "unixcoder_minus_nomic_pp":
        unix_gap,
    "jina_classification":
        jina_class,
}


with (
    OUT /
    "retriever_strength_valid_paper_gt.json"
).open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False,
    )
    f.write("\n")


with (
    OUT /
    "retriever_strength_valid_paper_gt.csv"
).open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    w = csv.writer(f)

    w.writerow([
        "retriever",
        "language",
        "n_queries",
        "R@1",
        "R@5",
        "R@10",
        "R@100",
        "MRR",
    ])

    for name in [
        "nomic",
        "unixcoder",
        "jina",
    ]:
        for lang in LANGS:
            r = results["retrievers"][name][lang]
            w.writerow([
                name,
                lang,
                r["n_queries"],
                r["r1"],
                r["r5"],
                r["r10"],
                r["r100"],
                r["mrr"],
            ])


print()
print("=" * 78)
print("PAPER-PROTOCOL VALIDATION RETRIEVER SUMMARY")
print("=" * 78)

for name in [
    "nomic",
    "unixcoder",
    "jina",
]:
    m = results["retrievers"][name]["macro"]

    print(
        f"{name:<12} "
        f"R@1={m['r1']:.4f} "
        f"R@5={m['r5']:.4f} "
        f"R@10={m['r10']:.4f} "
        f"R@100={m['r100']:.4f} "
        f"MRR={m['mrr']:.4f}"
    )

print()
print(
    f"Jina - Nomic macro MRR = "
    f"{jina_gap:+.4f} pp"
)

print(
    f"UniXcoder - Nomic macro MRR = "
    f"{unix_gap:+.4f} pp"
)

print(
    "JINA_CLASSIFICATION =",
    jina_class,
)

print(
    "PAPER_PROTOCOL_RETRIEVER_BENCHMARK=PASSED"
)
