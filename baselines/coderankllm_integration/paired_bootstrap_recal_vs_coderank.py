#!/usr/bin/env python3

import csv
import json
import hashlib
from pathlib import Path
import numpy as np

B = 10000
SEED = 20260911
BOOT_BATCH = 32

ROOT = Path("/root/autodl-tmp/cornstack-official")
RECAL_ROOT = Path(
    "/root/autodl-tmp/lora-code-embeddings-main/text2code/"
    "results/verifier_ranks_oldcsn/alpha_sweep_zscore"
)

OUT = ROOT / "results_coderankllm_stats"
OUT.mkdir(parents=True, exist_ok=True)

LANGS = ["ruby", "javascript", "php", "python", "java", "go"]

EXPECTED_N = {
    "ruby": 2279,
    "javascript": 6483,
    "php": 28391,
    "python": 22176,
    "java": 26909,
    "go": 14291,
}

EXPECTED_OUTSIDE100 = {
    "ruby": 274,
    "javascript": 1101,
    "php": 7390,
    "python": 1415,
    "java": 4839,
    "go": 2034,
}

EXPECTED_RECAL = {
    "ruby": 0.6876589371912725,
    "javascript": 0.5913381606363096,
    "php": 0.43472663340900214,
    "python": 0.7133578453117376,
    "java": 0.5927504754488071,
    "go": 0.7269513265113109,
}

EXPECTED_CR = {
    "ruby": 0.6627421778390444,
    "javascript": 0.5658997602084317,
    "php": 0.4113530234328496,
    "python": 0.6843374795960268,
    "java": 0.5717616295707189,
    "go": 0.7039399889306166,
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            x = f.read(1024 * 1024)
            if not x:
                break
            h.update(x)
    return h.hexdigest()


def mrr(ranks):
    ranks = np.asarray(ranks, dtype=np.float64)
    return float(np.mean(1.0 / (ranks + 1.0)))


def boot_mean(delta, seed):
    delta = np.asarray(delta, dtype=np.float64)
    n = len(delta)
    rng = np.random.default_rng(seed)

    result = np.empty(B, dtype=np.float64)

    done = 0
    while done < B:
        b = min(BOOT_BATCH, B - done)

        idx = rng.integers(
            0,
            n,
            size=(b, n),
            dtype=np.int32,
        )

        result[done:done+b] = delta[idx].mean(axis=1)
        done += b

    return result


def ci95(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def recal_path(lang):
    return RECAL_ROOT / (
        f"{lang}_test_llm_verifier_"
        f"cand100_verify15_alpha=1.ranks.npy"
    )


def coderank_path(lang):
    p = (
        ROOT /
        "runs_coderankllm_full" /
        lang /
        f"{lang}_coderankllm_fullranks.npy"
    )

    if not p.exists():
        hits = list(
            (ROOT / "runs_coderankllm_full" / lang).rglob(
                f"{lang}_coderankllm_fullranks.npy"
            )
        )
        if len(hits) != 1:
            raise RuntimeError(
                f"{lang}: CodeRank full rank file count={len(hits)}"
            )
        p = hits[0]

    return p


print("=" * 80)
print("RECALSEARCH vs CODERANKLLM — QUERY-PAIRED BOOTSTRAP")
print("=" * 80)

data = {}
manifest = {}

for li, lang in enumerate(LANGS):

    rp = recal_path(lang)
    cp = coderank_path(lang)

    assert rp.exists(), rp
    assert cp.exists(), cp

    r = np.asarray(np.load(rp), dtype=np.int64)
    c = np.asarray(np.load(cp), dtype=np.int64)

    n = EXPECTED_N[lang]

    assert r.shape == (n,), (lang, r.shape)
    assert c.shape == (n,), (lang, c.shape)

    rm = mrr(r)
    cm = mrr(c)

    assert abs(rm - EXPECTED_RECAL[lang]) < 1e-12
    assert abs(cm - EXPECTED_CR[lang]) < 1e-12

    # ----------------------------------------------------------
    # Alignment gate:
    #
    # GT outside Nomic top100 cannot be changed by either
    # reranker; both must preserve exactly the original dense
    # full-codebase rank for those queries.
    # ----------------------------------------------------------
    outside = c >= 100

    assert int(outside.sum()) == EXPECTED_OUTSIDE100[lang], (
        lang,
        int(outside.sum()),
        EXPECTED_OUTSIDE100[lang],
    )

    if not np.array_equal(r[outside], c[outside]):
        mismatch = np.sum(r[outside] != c[outside])
        raise RuntimeError(
            f"{lang}: QUERY ALIGNMENT FAILED; "
            f"outside-top100 rank mismatches={mismatch}"
        )

    rr_r = 1.0 / (r.astype(np.float64) + 1.0)
    rr_c = 1.0 / (c.astype(np.float64) + 1.0)

    delta = rr_r - rr_c

    wins = int(np.sum(delta > 0))
    ties = int(np.sum(delta == 0))
    losses = int(np.sum(delta < 0))

    boot = boot_mean(delta, SEED + li)
    lo, hi = ci95(boot)

    data[lang] = {
        "ranks_recal": r,
        "ranks_coderank": c,
        "rr_recal": rr_r,
        "rr_coderank": rr_c,
        "delta": delta,
        "boot": boot,
        "recal_mrr": rm,
        "coderank_mrr": cm,
        "delta_mrr": float(delta.mean()),
        "ci_low": lo,
        "ci_high": hi,
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }

    manifest[lang] = {
        "recal_path": str(rp),
        "recal_sha256": sha256(rp),
        "coderank_path": str(cp),
        "coderank_sha256": sha256(cp),
    }

    print(
        f"{lang:10s} "
        f"N={n:6d} "
        f"outside100={int(outside.sum()):5d} "
        f"alignment=PASS "
        f"ReCal={rm*100:8.4f} "
        f"CR={cm*100:8.4f} "
        f"Delta={delta.mean()*100:+8.4f} "
        f"CI=[{lo*100:+.4f},{hi*100:+.4f}] "
        f"W/T/L={wins}/{ties}/{losses}"
    )

print()
print("ALL QUERY ALIGNMENT GATES PASSED")


# ============================================================
# Macro — equal language weighting
# ============================================================

macro_recal = float(np.mean([
    data[x]["recal_mrr"] for x in LANGS
]))

macro_cr = float(np.mean([
    data[x]["coderank_mrr"] for x in LANGS
]))

macro_delta = macro_recal - macro_cr

# Bootstrap independently inside each language, then equal mean.
macro_boot = np.vstack([
    data[x]["boot"] for x in LANGS
]).mean(axis=0)

macro_lo, macro_hi = ci95(macro_boot)


# ============================================================
# Pooled / micro — query weighting
# ============================================================

all_rr_r = np.concatenate([
    data[x]["rr_recal"] for x in LANGS
])

all_rr_c = np.concatenate([
    data[x]["rr_coderank"] for x in LANGS
])

all_delta = all_rr_r - all_rr_c

assert len(all_delta) == 100529

micro_recal = float(all_rr_r.mean())
micro_cr = float(all_rr_c.mean())
micro_delta = float(all_delta.mean())

micro_boot = boot_mean(
    all_delta,
    SEED + 99999,
)

micro_lo, micro_hi = ci95(micro_boot)

global_w = int(np.sum(all_delta > 0))
global_t = int(np.sum(all_delta == 0))
global_l = int(np.sum(all_delta < 0))


# ============================================================
# Save per-language table
# ============================================================

rows = []

for lang in LANGS:
    x = data[lang]

    rows.append({
        "language": lang,
        "queries": EXPECTED_N[lang],
        "recal_mrr": x["recal_mrr"],
        "coderankllm_mrr": x["coderank_mrr"],
        "delta_mrr": x["delta_mrr"],
        "delta_mrr_points": x["delta_mrr"] * 100,
        "ci95_low": x["ci_low"],
        "ci95_high": x["ci_high"],
        "ci95_low_points": x["ci_low"] * 100,
        "ci95_high_points": x["ci_high"] * 100,
        "wins": x["wins"],
        "ties": x["ties"],
        "losses": x["losses"],
        "outside_top100": EXPECTED_OUTSIDE100[lang],
    })

csv_path = OUT / "recal_vs_coderank_paired_bootstrap_by_language.csv"

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# Per-query artifacts
# ============================================================

pq = OUT / "per_query"
pq.mkdir(exist_ok=True)

for lang in LANGS:
    x = data[lang]

    np.savez_compressed(
        pq / f"{lang}_recal_vs_coderank.npz",
        query_index=np.arange(
            EXPECTED_N[lang],
            dtype=np.int32,
        ),
        recal_rank=x["ranks_recal"].astype(np.int32),
        coderank_rank=x["ranks_coderank"].astype(np.int32),
        recal_rr=x["rr_recal"].astype(np.float32),
        coderank_rr=x["rr_coderank"].astype(np.float32),
        delta_rr=x["delta"].astype(np.float32),
    )


# ============================================================
# JSON report
# ============================================================

report = {
    "protocol": "recal_vs_coderank_paired_bootstrap_v1",

    "important_protocol_note": (
        "ReCalSearch ranks correspond to the historical alpha=1 "
        "test-sweep configuration. These statistics quantify the "
        "historical result and CodeRankLLM comparison, not the final "
        "validation-frozen revision protocol."
    ),

    "bootstrap_replicates": B,
    "bootstrap_seed": SEED,
    "total_queries": 100529,

    "per_language": {
        row["language"]: row
        for row in rows
    },

    "macro_equal_language_weight": {
        "recal_mrr": macro_recal,
        "coderankllm_mrr": macro_cr,
        "delta_mrr": macro_delta,
        "delta_mrr_points": macro_delta * 100,
        "ci95": [macro_lo, macro_hi],
        "ci95_points": [
            macro_lo * 100,
            macro_hi * 100,
        ],
    },

    "pooled_micro": {
        "recal_mrr": micro_recal,
        "coderankllm_mrr": micro_cr,
        "delta_mrr": micro_delta,
        "delta_mrr_points": micro_delta * 100,
        "ci95": [micro_lo, micro_hi],
        "ci95_points": [
            micro_lo * 100,
            micro_hi * 100,
        ],
        "wins": global_w,
        "ties": global_t,
        "losses": global_l,
    },

    "alignment": {
        "status": "PASS",
        "anchor": (
            "exact equality of preserved full-codebase dense ranks "
            "for all ground truths outside Nomic top-100"
        ),
    },

    "input_manifest": manifest,
}

json_path = OUT / "recal_vs_coderank_paired_bootstrap.json"

json_path.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    )
)


# ============================================================
# SHA256 output manifest
# ============================================================

artifacts = [
    json_path,
    csv_path,
    *sorted(pq.glob("*.npz")),
]

sha_manifest = {
    str(x): sha256(x)
    for x in artifacts
}

(OUT / "sha256_manifest.json").write_text(
    json.dumps(
        sha_manifest,
        indent=2,
    )
)


# ============================================================
# Report
# ============================================================

print()
print("=" * 80)
print("MACRO / EQUAL LANGUAGE WEIGHT")
print("=" * 80)

print("ReCalSearch MRR =", repr(macro_recal))
print("CodeRankLLM MRR =", repr(macro_cr))
print("Delta MRR       =", repr(macro_delta))
print("Delta points    =", macro_delta * 100)
print(
    "95% CI points  =",
    [macro_lo * 100, macro_hi * 100]
)

print()
print("=" * 80)
print("POOLED / MICRO / 100,529 QUERIES")
print("=" * 80)

print("ReCalSearch MRR =", repr(micro_recal))
print("CodeRankLLM MRR =", repr(micro_cr))
print("Delta MRR       =", repr(micro_delta))
print("Delta points    =", micro_delta * 100)
print(
    "95% CI points  =",
    [micro_lo * 100, micro_hi * 100]
)
print(
    "Wins/Ties/Losses =",
    global_w,
    global_t,
    global_l,
)

print()
print("Saved:")
print(json_path)
print(csv_path)
print(pq)
print(OUT / "sha256_manifest.json")

print()
print("PAIRED BOOTSTRAP VALIDATION PASSED")
