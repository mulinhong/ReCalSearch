from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from eval_llm_verifier import rerank_with_verifier


ROOT = Path(
    "/root/autodl-tmp/lora-code-embeddings-main/text2code"
).resolve()

POOL_ROOT = (
    ROOT
    / "results/reviewer_revision/pools/nomic_test"
)

EVAL_ROOT = (
    ROOT
    / "results/reviewer_revision/hn_source_ablation/"
      "evaluation/nomic_pool"
)

OUT_ROOT = (
    ROOT
    / "results/reviewer_revision/hn_source_ablation/"
      "crossover/nomic_pool_verifier_only"
)

LANGS = [
    "ruby",
    "javascript",
    "php",
    "python",
    "java",
    "go",
]

SOURCES = [
    "nomic",
    "jina",
]

SEEDS = [
    13,
    42,
    87,
]

CANDIDATE_TOP_K = 100
VERIFY_TOP_K = 15


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(
        path.name + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


def find_exactly_one(
    directory: Path,
    pattern: str,
) -> Path:

    hits = sorted(
        directory.glob(pattern)
    )

    if len(hits) != 1:
        raise RuntimeError(
            f"Expected exactly one match:\n"
            f"  dir={directory}\n"
            f"  pattern={pattern}\n"
            f"  found={hits}"
        )

    return hits[0]


def metrics(ranks: np.ndarray):
    ranks = np.asarray(
        ranks,
        dtype=np.int64,
    )

    return {
        "n":
            int(len(ranks)),
        "r1":
            float(
                np.mean(
                    ranks < 1
                )
            ),
        "r5":
            float(
                np.mean(
                    ranks < 5
                )
            ),
        "r10":
            float(
                np.mean(
                    ranks < 10
                )
            ),
        "mrr":
            float(
                np.mean(
                    1.0
                    / (
                        ranks.astype(
                            np.float64
                        )
                        + 1.0
                    )
                )
            ),
    }


OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

records = []

print("=" * 110)
print("R1-1 NOMIС-SIDE VERIFIER-ONLY OFFLINE REPLAY")
print("=" * 110)
print(
    "Protocol: candidate=Nomic top100; "
    "verify top15; ranking=verifier logit only; "
    "positions 16-100 preserve retriever order."
)
print(
    "Safety gate: reconstruct historical "
    "alpha=1,zscore ranks exactly before "
    "accepting alpha=0 output."
)
print("=" * 110)


for source in SOURCES:
    for seed in SEEDS:

        run_name = (
            f"{source}_seed{seed}"
        )

        run_root = (
            EVAL_ROOT
            / run_name
        )

        score_dir = (
            run_root
            / "scores"
        )

        historical_rank_dir = (
            run_root
            / "ranks"
        )

        out_dir = (
            OUT_ROOT
            / run_name
        )

        rank_out_dir = (
            out_dir
            / "ranks"
        )

        rank_out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for lang in LANGS:

            pool_path = (
                POOL_ROOT
                / (
                    f"{lang}_test_"
                    "nomic_top100.npz"
                )
            )

            if not pool_path.is_file():
                raise FileNotFoundError(
                    pool_path
                )

            score_path = find_exactly_one(
                score_dir,
                (
                    f"{lang}_test_*"
                    "cand100_verify15_"
                    "*.scores.npy"
                ),
            )

            historical_rank_path = (
                historical_rank_dir
                / (
                    f"{lang}_test_"
                    "llm_verifier_"
                    "nomic_test_sealed_"
                    "cand100_verify15_"
                    "alpha=1.ranks.npy"
                )
            )

            if not historical_rank_path.is_file():
                raise FileNotFoundError(
                    historical_rank_path
                )

            with np.load(
                pool_path,
                allow_pickle=False,
            ) as z:

                required = {
                    "indices",
                    "scores",
                    "base_ranks",
                    "gt_indices",
                }

                missing = (
                    required
                    - set(z.files)
                )

                if missing:
                    raise RuntimeError(
                        f"{lang}: missing "
                        f"pool arrays {missing}"
                    )

                candidate_indices = (
                    np.asarray(
                        z["indices"],
                        dtype=np.int64,
                    )
                )

                candidate_scores = (
                    np.asarray(
                        z["scores"],
                        dtype=np.float32,
                    )
                )

                base_ranks = (
                    np.asarray(
                        z["base_ranks"],
                        dtype=np.int64,
                    )
                )

                gt_indices = (
                    np.asarray(
                        z["gt_indices"],
                        dtype=np.int64,
                    )
                )

            verifier_scores = np.load(
                score_path,
                allow_pickle=False,
            ).astype(
                np.float32,
                copy=False,
            )

            historical_alpha1 = (
                np.load(
                    historical_rank_path,
                    allow_pickle=False,
                )
                .astype(
                    np.int64,
                    copy=False,
                )
            )

            n = len(gt_indices)

            if candidate_indices.shape != (
                n,
                CANDIDATE_TOP_K,
            ):
                raise RuntimeError(
                    f"{run_name}/{lang}: "
                    "bad candidate shape "
                    f"{candidate_indices.shape}"
                )

            if candidate_scores.shape != (
                n,
                CANDIDATE_TOP_K,
            ):
                raise RuntimeError(
                    f"{run_name}/{lang}: "
                    "bad score shape "
                    f"{candidate_scores.shape}"
                )

            if verifier_scores.shape != (
                n,
                VERIFY_TOP_K,
            ):
                raise RuntimeError(
                    f"{run_name}/{lang}: "
                    "bad verifier-score shape "
                    f"{verifier_scores.shape}"
                )

            if base_ranks.shape != (
                n,
            ):
                raise RuntimeError(
                    f"{run_name}/{lang}: "
                    "bad base-rank shape"
                )

            if gt_indices.shape != (
                n,
            ):
                raise RuntimeError(
                    f"{run_name}/{lang}: "
                    "bad GT shape"
                )

            if historical_alpha1.shape != (
                n,
            ):
                raise RuntimeError(
                    f"{run_name}/{lang}: "
                    "bad historical alpha1 shape"
                )

            # --------------------------------------------------
            # Gate A:
            # reconstruct the already-sealed source-ablation
            # result using the saved raw verifier scores.
            # --------------------------------------------------

            replay_alpha1 = (
                rerank_with_verifier(
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
                        VERIFY_TOP_K,
                    fusion_alpha=
                        1.0,
                    normalization=
                        "zscore",
                )
            )

            replay_equal = (
                np.array_equal(
                    replay_alpha1,
                    historical_alpha1,
                )
            )

            mismatch_count = int(
                np.count_nonzero(
                    replay_alpha1
                    != historical_alpha1
                )
            )

            if not replay_equal:
                raise RuntimeError(
                    f"{run_name}/{lang}: "
                    "ALPHA=1 REPLAY FAILED "
                    f"({mismatch_count} mismatches). "
                    "Verifier-only output NOT accepted."
                )

            # --------------------------------------------------
            # Gate B:
            # frozen crossover Nomic-side condition.
            #
            # alpha=0 in the historical evaluator means:
            # final_scores = verifier_scores
            #
            # No dense fusion.
            # --------------------------------------------------

            verifier_only = (
                rerank_with_verifier(
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
                        VERIFY_TOP_K,
                    fusion_alpha=
                        0.0,
                    normalization=
                        "zscore",
                )
            )

            output_path = (
                rank_out_dir
                / (
                    f"{lang}_test_"
                    f"{source}_hn_verifier_"
                    "on_nomic_pool_"
                    "cand100_verify15_"
                    "verifier_only.ranks.npy"
                )
            )

            if output_path.exists():
                existing = np.load(
                    output_path,
                    allow_pickle=False,
                ).astype(
                    np.int64,
                    copy=False,
                )

                if not np.array_equal(
                    existing,
                    verifier_only,
                ):
                    raise RuntimeError(
                        "Existing verifier-only "
                        "rank artifact differs: "
                        f"{output_path}"
                    )

                output_status = (
                    "reused_exact_existing"
                )

            else:
                tmp = output_path.with_name(
                    output_path.name
                    + ".tmp"
                )

                with tmp.open("wb") as f:
                    np.save(
                        f,
                        verifier_only.astype(
                            np.int32
                        ),
                    )

                os.replace(
                    tmp,
                    output_path,
                )

                output_status = (
                    "written_new"
                )

            rec = {
                "condition":
                    (
                        "V_N@N"
                        if source == "nomic"
                        else "V_J@N"
                    ),
                "hn_source":
                    source,
                "seed":
                    seed,
                "language":
                    lang,
                "candidate_pool":
                    "nomic_test_top100",
                "candidate_top_k":
                    CANDIDATE_TOP_K,
                "verify_top_k":
                    VERIFY_TOP_K,
                "ranking_score":
                    "verifier_logit_difference_only",
                "dense_fusion":
                    False,
                "positions_after_15":
                    "preserve_nomic_retriever_order",
                "alpha1_replay_gate": {
                    "normalization":
                        "zscore",
                    "alpha":
                        1.0,
                    "exact_equal":
                        replay_equal,
                    "mismatch_count":
                        mismatch_count,
                },
                "metrics":
                    metrics(
                        verifier_only
                    ),
                "files": {
                    "pool":
                        str(
                            pool_path.resolve()
                        ),
                    "score_cache":
                        str(
                            score_path.resolve()
                        ),
                    "sealed_alpha1_rank":
                        str(
                            historical_rank_path.resolve()
                        ),
                    "verifier_only_rank":
                        str(
                            output_path.resolve()
                        ),
                },
                "sha256": {
                    "pool":
                        sha256(pool_path),
                    "score_cache":
                        sha256(score_path),
                    "sealed_alpha1_rank":
                        sha256(
                            historical_rank_path
                        ),
                    "verifier_only_rank":
                        sha256(
                            output_path
                        ),
                },
                "output_status":
                    output_status,
            }

            records.append(
                rec
            )

            print(
                f"{run_name:<14} "
                f"{lang:<12} "
                "alpha1_replay=PASS "
                f"verifier_only_MRR="
                f"{rec['metrics']['mrr']:.12f}"
            )


manifest = {
    "artifact_type":
        "r1_1_nomic_side_verifier_only_replay_v1",
    "reviewer_item":
        "R1-1",
    "scientific_role":
        (
            "Nomic-pool half of the frozen "
            "Nomic/Jina crossover."
        ),
    "protocol": {
        "candidate_pool":
            "Nomic TEST top100",
        "verifier_sources":
            [
                "Nomic-HN",
                "Jina-HN",
            ],
        "seeds":
            SEEDS,
        "languages":
            LANGS,
        "candidate_top_k":
            100,
        "verify_top_k":
            15,
        "ranking_score":
            "verifier_logit_difference_only",
        "dense_fusion":
            False,
        "no_new_verifier_inference":
            True,
        "alpha1_replay_required":
            True,
    },
    "records":
        records,
}

manifest_path = (
    OUT_ROOT
    / "nomic_side_verifier_only_manifest.json"
)

atomic_json(
    manifest_path,
    manifest,
)

if len(records) != 36:
    raise RuntimeError(
        f"Expected 36 records, "
        f"got {len(records)}"
    )

if not all(
    x["alpha1_replay_gate"][
        "exact_equal"
    ]
    for x in records
):
    raise RuntimeError(
        "At least one alpha=1 replay "
        "gate failed"
    )

print()
print("=" * 110)
print("FINAL CLOSURE")
print("=" * 110)
print("records                         =", len(records))
print("expected_records                = 36")
print("all_alpha1_replay_exact         = True")
print("new_verifier_inference          = False")
print("dense_fusion                    = False")
print("verify_top_k                    = 15")
print("manifest                        =", manifest_path)
print("manifest_sha256                 =", sha256(manifest_path))
print("=" * 110)
