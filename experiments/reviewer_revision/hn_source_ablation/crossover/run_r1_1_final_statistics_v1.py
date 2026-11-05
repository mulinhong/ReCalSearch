from pathlib import Path
import numpy as np
import json
import hashlib
import re
import sys

ROOT = Path(
    "/root/autodl-tmp/lora-code-embeddings-main/text2code"
)

BASE = ROOT / (
    "results/reviewer_revision/hn_source_ablation/crossover"
)

NOM = BASE / "nomic_pool_verifier_only"
JIN = BASE / "jina_pool_verifier_only"
POOLS = ROOT / "results/reviewer_revision/pools"
OUT = BASE / "r1_1_final_statistics_v1"

LANGS = {
    "ruby": 2279,
    "javascript": 6483,
    "php": 28391,
    "python": 22176,
    "java": 26909,
    "go": 14291,
}

SEEDS = [13, 42, 87]
B = 10000
BOOT_SEED = 20261013

EXPECTED_NOMIC = {
    "nomic_seed13": 0.590326338728,
    "nomic_seed42": 0.588193835757,
    "nomic_seed87": 0.587644015845,
    "jina_seed13": 0.586480748517,
    "jina_seed42": 0.580853375849,
    "jina_seed87": 0.582576217078,
}

EXPECTED_JINA = {
    "nomic_seed13": 0.541695750455,
    "nomic_seed42": 0.538044293691,
    "nomic_seed87": 0.536867215984,
    "jina_seed13": 0.554694668283,
    "jina_seed42": 0.551373832130,
    "jina_seed87": 0.552047618986,
}

JINA_HASHES = {
    "nomic_seed13":
        "d1480dfad07c35a759cf9a4f0755899a796ab65d94acebd122dacbb89a301b65",
    "nomic_seed42":
        "e6996dff8f37ba7c1f60ed25d1a4debb67f381b8911f10c6a2062d09cc724f91",
    "nomic_seed87":
        "4f9086c63ea9d8fd662899bedca9602e3e01bd213f6446f4a4212836e700cd7b",
    "jina_seed13":
        "6cb582ea021df7fe9efa99a8602fbf50a0eace01da370fc2c389e9ef3a3172c6",
    "jina_seed42":
        "297f50c7e013de56e121d3b0419ae68dcc690d6ba02096adb3049ab170979922",
    "jina_seed87":
        "a32854ad92ec0192f599ad5d764b289247d80c3b1b7984dcc08fdd35e1243677",
}

NOMIC_MANIFEST_HASH = (
    "9ddf79935765e1e2fe7dfe70ceafd40e1d06f8f644613bcbcaf657dcd6f3cb19"
)

def check(condition, message):
    if not condition:
        raise RuntimeError("GATE_FAIL: " + message)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()

print("=" * 100)
print("WORK1 R1-1 FINAL CROSSOVER STATISTICS")
print("=" * 100)

check(not OUT.exists(), "Output already exists")

nom_manifest = (
    NOM / "nomic_side_verifier_only_manifest.json"
)

check(nom_manifest.is_file(), "Nomic manifest missing")
check(
    sha256(nom_manifest) == NOMIC_MANIFEST_HASH,
    "Nomic manifest SHA mismatch",
)

for run, expected in JINA_HASHES.items():
    p = JIN / run / "manifest.json"

    check(p.is_file(), f"Missing manifest {run}")
    check(
        sha256(p) == expected,
        f"Jina manifest SHA mismatch {run}",
    )

    m = json.loads(p.read_text(encoding="utf-8"))

    check(
        m.get("condition") ==
        ("V_N@J" if run.startswith("nomic") else "V_J@J"),
        f"Condition mismatch: {run}",
    )
    check(
        m.get("protocol", {}).get("dense_fusion") is False,
        f"Dense fusion mismatch: {run}",
    )
    check(
        m.get("protocol", {}).get("verify_top_k") == 15,
        f"verify_top_k mismatch: {run}",
    )

# Read Nomic-side rank locations without modifying artifacts.
nom_files = list(NOM.rglob("*.npy"))

def token_match(s, token):
    return bool(
        re.search(
            r"(^|[/_.-])" +
            re.escape(token) +
            r"($|[/_.-])",
            s,
        )
    )

def nomic_rank(run, lang):
    matches = []

    for p in nom_files:
        name = str(p)

        if not token_match(name, run):
            continue
        if not token_match(p.name, lang):
            continue
        if "rank" not in name.lower():
            continue

        matches.append(p)

    check(
        len(matches) == 1,
        f"Cannot uniquely resolve Nomic rank "
        f"for {run}/{lang}. Matches: {matches}",
    )

    return matches[0]

# Order of the four experimental cells:
# 0: Nomic-HN @ Nomic
# 1: Jina-HN  @ Nomic
# 2: Nomic-HN @ Jina
# 3: Jina-HN  @ Jina

DATA = {}
MASKS = {}
FILE_RECORDS = {}

RUN_METRICS = {}

for pool in ("nomic", "jina"):
    for source in ("nomic", "jina"):
        for seed in SEEDS:
            RUN_METRICS[
                (pool, f"{source}_seed{seed}")
            ] = []

print()
print("========== DATA AND RANK INTEGRITY ==========")

for lang, n in LANGS.items():

    nompool = (
        POOLS / "nomic_test"
        / f"{lang}_test_nomic_top100.npz"
    )
    jinpool = (
        POOLS / "jina_test"
        / f"{lang}_test_jina_top100.npz"
    )

    check(
        nompool.is_file() and jinpool.is_file(),
        f"Missing candidate pool: {lang}",
    )

    with np.load(nompool, allow_pickle=False) as z:
        nom_indices = np.asarray(z["indices"])
        nom_gt = np.asarray(z["gt_indices"])

    with np.load(jinpool, allow_pickle=False) as z:
        jin_indices = np.asarray(z["indices"])
        jin_gt = np.asarray(z["gt_indices"])

    check(
        nom_indices.shape == (n, 100),
        f"Nomic pool shape {lang}",
    )
    check(
        jin_indices.shape == (n, 100),
        f"Jina pool shape {lang}",
    )
    check(
        np.array_equal(nom_gt, jin_gt),
        f"GT/query order mismatch {lang}",
    )

    nom_hit = np.any(
        nom_indices == nom_gt[:, None],
        axis=1,
    )
    jin_hit = np.any(
        jin_indices == jin_gt[:, None],
        axis=1,
    )

    shared = nom_hit & jin_hit

    check(shared.any(), f"No shared-hit queries: {lang}")

    MASKS[lang] = shared

    a = np.empty((3, 4, n), dtype=np.float64)

    for si, seed in enumerate(SEEDS):

        for vi, source in enumerate(("nomic", "jina")):

            run = f"{source}_seed{seed}"

            nr = nomic_rank(run, lang)

            jr = (
                JIN / run / "ranks"
                / (
                    f"{lang}_test_{run}_on_jina_pool_"
                    "cand100_verify15_verifier_only.ranks.npy"
                )
            )

            for pi, p in enumerate((nr, jr)):

                check(p.is_file(), f"Missing rank: {p}")

                ranks = np.load(
                    p,
                    allow_pickle=False,
                )

                check(
                    ranks.shape == (n,),
                    f"Rank shape mismatch: {p}",
                )
                check(
                    np.issubdtype(ranks.dtype, np.integer),
                    f"Rank dtype mismatch: {p}",
                )
                check(
                    np.all(ranks >= 0),
                    f"Negative rank: {p}",
                )

                cell = pi * 2 + vi

                a[si, cell] = (
                    1.0 / (
                        ranks.astype(np.float64) + 1.0
                    )
                )

                pool_name = (
                    "nomic" if pi == 0 else "jina"
                )

                RUN_METRICS[(pool_name, run)].append(
                    float(a[si, cell].mean())
                )

                FILE_RECORDS[
                    f"{pool_name}/{run}/{lang}"
                ] = {
                    "path": str(p),
                    "sha256": sha256(p),
                }

    DATA[lang] = a

    print(
        f"{lang:<12} "
        f"n={n:<7} "
        f"shared_hit={int(shared.sum()):<7} "
        f"rate={shared.mean():.6f}"
    )

# Check the reconstructed macro numbers against prior sealed results.
for pool, expected_dict in [
    ("nomic", EXPECTED_NOMIC),
    ("jina", EXPECTED_JINA),
]:
    for run, expected in expected_dict.items():

        observed = float(
            np.mean(RUN_METRICS[(pool, run)])
        )

        check(
            abs(observed - expected) < 1e-9,
            f"Sealed macro mismatch: {pool}/{run} "
            f"{observed} != {expected}",
        )

        print(
            "SEALED_PARITY=PASS",
            pool,
            run,
            f"{observed:.12f}",
        )

HIT_DATA = {
    lang: DATA[lang][:, :, MASKS[lang]]
    for lang in LANGS
}

def aggregate(dataset, micro=False):

    rows = np.array([
        dataset[lang].mean(axis=(0, 2))
        for lang in LANGS
    ])

    if not micro:
        return rows.mean(axis=0)

    weights = np.array([
        dataset[lang].shape[2]
        for lang in LANGS
    ])

    return np.average(
        rows,
        axis=0,
        weights=weights,
    )

def contrasts(cells):
    nomic_match = cells[0] - cells[1]
    jina_match = cells[3] - cells[2]

    interaction = nomic_match + jina_match

    return np.array([
        nomic_match,
        jina_match,
        interaction,
    ])

OBS_CELLS = [
    aggregate(DATA),
    aggregate(DATA, micro=True),
    aggregate(HIT_DATA),
    aggregate(HIT_DATA, micro=True),
]

OBS = np.array([
    contrasts(x)
    for x in OBS_CELLS
])

print()
print("========== OBSERVED EFFECTS (MRR PP) ==========")

NAMES = [
    "full_macro",
    "full_micro",
    "shared_macro",
    "shared_micro",
]

for name, c, d in zip(NAMES, OBS_CELLS, OBS):
    print(
        name,
        "cells=",
        np.round(c * 100, 6).tolist(),
        "effects=",
        np.round(d * 100, 6).tolist(),
    )

print()
print("========== 10000 PAIRED BOOTSTRAP ==========")

rng = np.random.default_rng(BOOT_SEED)

# Dimensions:
# bootstrap × aggregation × comparison
DRAWS = np.empty((B, 4, 3), dtype=np.float64)
SEED_DRAWS = np.empty((B, 3), dtype=np.int8)

languages = list(LANGS)

full_weights = np.array([
    LANGS[lang] for lang in languages
])

hit_weights = np.array([
    int(MASKS[lang].sum())
    for lang in languages
])

for b in range(B):

    # Same seed draw for all four cells.
    seed_ids = rng.integers(
        0, 3, size=3
    )
    SEED_DRAWS[b] = seed_ids

    full_rows = []
    hit_rows = []

    for lang in languages:

        full = DATA[lang][seed_ids].mean(axis=0)
        hit = HIT_DATA[lang][seed_ids].mean(axis=0)

        qi = rng.integers(
            0,
            full.shape[1],
            size=full.shape[1],
        )

        hi = rng.integers(
            0,
            hit.shape[1],
            size=hit.shape[1],
        )

        # Paired query indices across the four methods.
        full_rows.append(
            full[:, qi].mean(axis=1)
        )
        hit_rows.append(
            hit[:, hi].mean(axis=1)
        )

    fr = np.stack(full_rows)
    hr = np.stack(hit_rows)

    bootstrap_cells = [
        fr.mean(axis=0),
        np.average(
            fr, axis=0, weights=full_weights
        ),
        hr.mean(axis=0),
        np.average(
            hr, axis=0, weights=hit_weights
        ),
    ]

    for k, cells in enumerate(bootstrap_cells):
        DRAWS[b, k] = contrasts(cells)

    if (b + 1) % 2000 == 0:
        print(
            "BOOTSTRAP_PROGRESS",
            b + 1,
            "/",
            B,
            flush=True,
        )

LABELS = [
    "nomic_pool_matching_advantage",
    "jina_pool_matching_advantage",
    "difference_in_differences",
]

STATS = {}

print()
print("========== CONFIDENCE INTERVALS ==========")

for k, name in enumerate(NAMES):

    STATS[name] = {}

    for j, label in enumerate(LABELS):

        draws = DRAWS[:, k, j]

        lo, hi = np.percentile(
            draws,
            [2.5, 97.5],
        )

        p_tail = min(
            1.0,
            2.0 * min(
                ((draws <= 0).sum() + 1) / (B + 1),
                ((draws >= 0).sum() + 1) / (B + 1),
            ),
        )

        STATS[name][label] = {
            "effect_pp": float(OBS[k, j] * 100),
            "ci95_low_pp": float(lo * 100),
            "ci95_high_pp": float(hi * 100),
            "bootstrap_sign_tail": float(p_tail),
        }

        print(
            name,
            label,
            f"effect={OBS[k,j]*100:.6f} pp",
            f"CI=[{lo*100:.6f},{hi*100:.6f}]",
        )

full = STATS["full_macro"]
shared = STATS["shared_macro"]

GATES = {
    "nomic_direction":
        full[LABELS[0]]["effect_pp"] > 0,
    "jina_direction":
        full[LABELS[1]]["effect_pp"] > 0,
    "shared_nomic_direction":
        shared[LABELS[0]]["effect_pp"] > 0,
    "shared_jina_direction":
        shared[LABELS[1]]["effect_pp"] > 0,
    "interaction_ci_positive":
        full[LABELS[2]]["ci95_low_pp"] > 0,
}

print()
print("========== PREDECLARED CLAIM GATES ==========")

for k, v in GATES.items():
    print(k, "=", v)

GATE = (
    "PASS"
    if all(GATES.values())
    else "NOT_ESTABLISHED"
)

print("STATISTICAL_CROSSOVER_GATE =", GATE)

# Write only new analysis artifacts.
check(not OUT.exists(), "Output collision")
OUT.mkdir(parents=True)

summary = {
    "reviewer_item": "R1-1",
    "replicates": B,
    "bootstrap_seed": BOOT_SEED,
    "bootstrap_design":
        "Paired resampling of queries within each of six "
        "fixed language strata, with three training seeds "
        "resampled jointly across all four conditions.",
    "full_query_count": sum(LANGS.values()),
    "shared_hit_count": sum(
        int(x.sum()) for x in MASKS.values()
    ),
    "shared_hit_by_language": {
        lang: int(MASKS[lang].sum())
        for lang in LANGS
    },
    "cell_order": [
        "V_N@N",
        "V_J@N",
        "V_N@J",
        "V_J@J",
    ],
    "observed_cells": {
        name: cells.tolist()
        for name, cells in zip(NAMES, OBS_CELLS)
    },
    "statistics": STATS,
    "gates": GATES,
    "statistical_gate": GATE,
    "scope":
        "Results conditional on six languages and three "
        "observed training seeds; negative-hardness "
        "confounding is a separate audit.",
    "rank_artifacts": FILE_RECORDS,
}

summary_path = OUT / "r1_1_crossover_statistics.json"

summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ) + "\n",
    encoding="utf-8",
)

draw_path = OUT / "r1_1_bootstrap_draws.npz"

np.savez_compressed(
    draw_path,
    draws=DRAWS,
    seed_draws=SEED_DRAWS,
)

print()
print("========== OUTPUT FILES ==========")

for p in (summary_path, draw_path):
    print(p)
    print("SHA256 =", sha256(p))

print()
print("ANALYSIS_COMPLETE = TRUE")
print("NO_GPU_INFERENCE = TRUE")
print("NO_TRAINING = TRUE")
print("NO_FROZEN_ARTIFACT_MODIFICATION = TRUE")
