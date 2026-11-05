from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import benchmark_dense_retrievers_valid as hist
from data_utils import create_dataset
from eval_nomic_embed import get_query_text as nomic_get_query_text


ROOT = Path(
    "/root/autodl-tmp/lora-code-embeddings-main/text2code"
).resolve()

DATA_ROOT = Path(
    "/root/autodl-tmp/lora-code-embeddings-main/data/CSN"
).resolve()

JINA_CODE_ROOT = (
    ROOT
    / "results/reviewer_revision/hn_source_ablation/"
      "retriever_strength_valid/jina_cache"
)

NOMIC_POOL_ROOT = (
    ROOT
    / "results/reviewer_revision/pools/nomic_test"
)

OUT_ROOT = (
    ROOT
    / "results/reviewer_revision/pools/jina_test"
)

QUERY_CACHE_ROOT = (
    ROOT
    / "results/reviewer_revision/hn_source_ablation/"
      "crossover/jina_test_query_cache"
)

LANGS = [
    "ruby",
    "javascript",
    "php",
    "python",
    "java",
    "go",
]

TOP_K = 100
DIM = 896

# Match the historical Jina generation/retrieval setup.
QUERY_BATCH = 64
QUERY_CHUNK = 64
CODE_CHUNK = 4096


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def text2code_to_array(
    text2code,
    n_queries: int,
) -> np.ndarray:

    if isinstance(text2code, torch.Tensor):
        return (
            text2code
            .long()
            .cpu()
            .numpy()
        )

    if isinstance(text2code, dict):
        return np.asarray(
            [
                text2code[i]
                for i in range(n_queries)
            ],
            dtype=np.int64,
        )

    return np.asarray(
        list(text2code),
        dtype=np.int64,
    )


def resolve_protocol_gt(
    lang: str,
    test_rows,
) -> np.ndarray:
    """
    Exact ReCalSearch protocol GT, then gated against
    the already sealed Nomic TEST pool.
    """

    _, _, test_dataset, _ = create_dataset(
        str(DATA_ROOT),
        lang,
    )

    if len(test_dataset) != len(test_rows):
        raise RuntimeError(
            f"{lang}: test dataset/jsonl length mismatch: "
            f"{len(test_dataset)} vs {len(test_rows)}"
        )

    gt = text2code_to_array(
        test_dataset.text2code,
        len(test_dataset),
    ).astype(
        np.int64,
        copy=False,
    )

    nomic_path = (
        NOMIC_POOL_ROOT
        / f"{lang}_test_nomic_top100.npz"
    )

    if not nomic_path.is_file():
        raise FileNotFoundError(
            f"Missing sealed Nomic TEST pool: "
            f"{nomic_path}"
        )

    with np.load(
        nomic_path,
        allow_pickle=False,
    ) as z:
        sealed_gt = np.asarray(
            z["gt_indices"],
            dtype=np.int64,
        )

    if not np.array_equal(gt, sealed_gt):
        raise RuntimeError(
            f"{lang}: GT/order mismatch against "
            f"sealed Nomic TEST pool"
        )

    return gt


def check_query_contract(
    lang: str,
    rows,
) -> list[str]:
    """
    Jina and Nomic must see exactly the same TEST
    query strings. This was audited globally; keep
    it as an execution-time gate.
    """

    out = []

    for i, row in enumerate(rows):
        jq = hist.get_query_text(row)
        nq = nomic_get_query_text(
            row,
            use_raw=True,
        )

        if jq != nq:
            raise RuntimeError(
                f"{lang}: Jina/Nomic query-text "
                f"mismatch at TEST row {i}"
            )

        out.append(jq)

    return out


def expected_query_recipe():
    return {
        "model":
            str(hist.JINA_MODEL),
        "max_length":
            int(hist.MAX_LENGTH),
        "dimension":
            DIM,
        "query_prefix":
            hist.JINA_QUERY_PREFIX,
        "code_prefix":
            hist.JINA_CODE_PREFIX,
        "pooling":
            "last_token",
        "normalization":
            "l2_after_pooling_in_fp32",
        "padding_side":
            "left",
        "forward_dtype_policy":
            "bfloat16_if_supported_else_float16",
        "cache_dtype":
            "float16",
        "query_batch":
            QUERY_BATCH,
    }


def encode_test_queries(
    lang: str,
    queries,
    cache_path: Path,
):
    """
    Fresh TEST queries only.
    Frozen historical JinaEncoder recipe.
    """

    meta_path = cache_path.with_suffix(
        ".json"
    )

    expected = expected_query_recipe()

    if cache_path.is_file():
        if not meta_path.is_file():
            raise RuntimeError(
                "Existing query cache has no "
                f"provenance metadata: {cache_path}"
            )

        meta = json.loads(
            meta_path.read_text(
                encoding="utf-8"
            )
        )

        if meta.get("recipe") != expected:
            raise RuntimeError(
                f"{lang}: existing query-cache "
                "recipe mismatch"
            )

        arr = np.load(
            cache_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        if arr.shape != (
            len(queries),
            DIM,
        ):
            raise RuntimeError(
                f"{lang}: existing query cache "
                f"bad shape {arr.shape}"
            )

        if arr.dtype != np.float16:
            raise RuntimeError(
                f"{lang}: existing query cache "
                f"bad dtype {arr.dtype}"
            )

        print(
            "REUSING_FROZEN_JINA_TEST_QUERY_CACHE",
            cache_path,
        )

        return str(
            meta.get(
                "actual_model_dtype",
                "unknown",
            )
        )

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = cache_path.with_name(
        cache_path.name + ".tmp"
    )

    if tmp.exists():
        tmp.unlink()

    encoder = hist.JinaEncoder()

    actual_dtype = str(
        next(
            encoder.model.parameters()
        ).dtype
    )

    print(
        "JINA_TEST_ENCODER_DTYPE=",
        actual_dtype,
    )

    out = np.lib.format.open_memmap(
        tmp,
        mode="w+",
        dtype=np.float16,
        shape=(
            len(queries),
            DIM,
        ),
    )

    offset = 0

    for start in tqdm(
        range(
            0,
            len(queries),
            QUERY_BATCH,
        ),
        desc=f"{lang} frozen Jina TEST query",
    ):
        batch_text = queries[
            start:
            start + QUERY_BATCH
        ]

        emb = encoder.encode_batch(
            batch_text,
            hist.JINA_QUERY_PREFIX,
        )

        if emb.shape != (
            len(batch_text),
            DIM,
        ):
            raise RuntimeError(
                f"{lang}: bad embedding "
                f"shape {emb.shape}"
            )

        out[
            offset:
            offset + len(emb)
        ] = emb

        offset += len(emb)

    if offset != len(queries):
        raise RuntimeError(
            f"{lang}: encoded {offset}, "
            f"expected {len(queries)}"
        )

    out.flush()
    del out

    os.replace(
        tmp,
        cache_path,
    )

    meta = {
        "format":
            "r1_1_jina_test_query_cache_v1",
        "language":
            lang,
        "split":
            "test",
        "queries":
            len(queries),
        "recipe":
            expected,
        "actual_model_dtype":
            actual_dtype,
        "cache_file":
            str(cache_path.resolve()),
        "cache_sha256":
            sha256(cache_path),
    }

    atomic_json(
        meta_path,
        meta,
    )

    del encoder
    gc.collect()
    torch.cuda.empty_cache()

    print(
        "JINA_TEST_QUERY_CACHE_SAVED",
        cache_path,
    )

    return actual_dtype


@torch.no_grad()
def retrieve_exact_protocol(
    query_cache_path: Path,
    code_cache_path: Path,
    gt_indices: np.ndarray,
):
    """
    Exact rank semantics copied from the frozen
    ReCalSearch/Nomic exporter:

      top100 = torch.topk(full scores)
      base_rank = #(score > GT score)

    Therefore rank is:
      - full-codebase
      - zero-based
      - tie-aware
    """

    q_np = np.load(
        query_cache_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    c_np = np.load(
        code_cache_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    if (
        q_np.ndim != 2
        or q_np.shape[1] != DIM
    ):
        raise RuntimeError(
            f"Bad query cache {q_np.shape}"
        )

    if (
        c_np.ndim != 2
        or c_np.shape[1] != DIM
    ):
        raise RuntimeError(
            f"Bad code cache {c_np.shape}"
        )

    if len(q_np) != len(gt_indices):
        raise RuntimeError(
            "Query/GT length mismatch"
        )

    n_queries = q_np.shape[0]
    n_codes = c_np.shape[0]

    if np.any(
        gt_indices < 0
    ) or np.any(
        gt_indices >= n_codes
    ):
        raise RuntimeError(
            "GT index outside Jina codebase"
        )

    final_scores = np.empty(
        (n_queries, TOP_K),
        dtype=np.float32,
    )

    final_indices = np.empty(
        (n_queries, TOP_K),
        dtype=np.int64,
    )

    full_ranks = np.empty(
        n_queries,
        dtype=np.int64,
    )

    device = torch.device("cuda")

    for qs in tqdm(
        range(
            0,
            n_queries,
            QUERY_CHUNK,
        ),
        desc="Frozen Jina TEST full-code retrieval",
    ):
        qe = min(
            qs + QUERY_CHUNK,
            n_queries,
        )

        q = torch.from_numpy(
            np.asarray(
                q_np[qs:qe],
                dtype=np.float32,
            ).copy()
        ).to(device)

        q = F.normalize(
            q,
            dim=1,
        )

        q_len = qe - qs

        # Deliberately materialize the complete score
        # row, matching the frozen Nomic rank contract.
        chunk_scores = torch.empty(
            (
                q_len,
                n_codes,
            ),
            dtype=torch.float32,
            device=device,
        )

        for cs in range(
            0,
            n_codes,
            CODE_CHUNK,
        ):
            ce = min(
                cs + CODE_CHUNK,
                n_codes,
            )

            c = torch.from_numpy(
                np.asarray(
                    c_np[cs:ce],
                    dtype=np.float32,
                ).copy()
            ).to(device)

            c = F.normalize(
                c,
                dim=1,
            )

            chunk_scores[
                :,
                cs:ce
            ] = q @ c.T

            del c

        top_scores, top_indices = (
            torch.topk(
                chunk_scores,
                k=min(
                    TOP_K,
                    n_codes,
                ),
                dim=1,
            )
        )

        gt_chunk = torch.from_numpy(
            gt_indices[qs:qe]
        ).to(device)

        gt_scores = chunk_scores.gather(
            1,
            gt_chunk.unsqueeze(1),
        ).squeeze(1)

        # Exact frozen semantics:
        # zero-based + strict greater-than + tie-aware.
        ranks = (
            chunk_scores
            > gt_scores.unsqueeze(1)
        ).sum(dim=1)

        final_scores[qs:qe] = (
            top_scores
            .float()
            .cpu()
            .numpy()
        )

        final_indices[qs:qe] = (
            top_indices
            .long()
            .cpu()
            .numpy()
        )

        full_ranks[qs:qe] = (
            ranks
            .long()
            .cpu()
            .numpy()
        )

        del (
            q,
            chunk_scores,
            top_scores,
            top_indices,
            gt_chunk,
            gt_scores,
            ranks,
        )

        torch.cuda.empty_cache()

    return (
        final_scores,
        final_indices,
        full_ranks,
        int(n_codes),
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--language",
        required=True,
        choices=LANGS,
    )

    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required"
        )

    lang = args.language

    test_path = (
        DATA_ROOT
        / lang
        / "test.jsonl"
    )

    codebase_path = (
        DATA_ROOT
        / lang
        / "codebase.jsonl"
    )

    if not test_path.is_file():
        raise FileNotFoundError(
            test_path
        )

    if not codebase_path.is_file():
        raise FileNotFoundError(
            codebase_path
        )

    test_rows = hist.read_jsonl(
        test_path
    )

    queries = check_query_contract(
        lang,
        test_rows,
    )

    gt_indices = resolve_protocol_gt(
        lang,
        test_rows,
    )

    code_cache = (
        JINA_CODE_ROOT
        / (
            f"{lang}_"
            "jina_code_0p5b_"
            "nl2code_msl512.code.npy"
        )
    )

    if not code_cache.is_file():
        raise FileNotFoundError(
            "Frozen historical Jina code "
            f"cache missing: {code_cache}"
        )

    code_arr = np.load(
        code_cache,
        mmap_mode="r",
        allow_pickle=False,
    )

    if (
        code_arr.ndim != 2
        or code_arr.shape[1] != DIM
    ):
        raise RuntimeError(
            f"Bad frozen code cache "
            f"shape {code_arr.shape}"
        )

    del code_arr

    QUERY_CACHE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    query_cache = (
        QUERY_CACHE_ROOT
        / (
            f"{lang}_test_"
            "jina_code_0p5b_"
            "nl2code_msl512.query.npy"
        )
    )

    output_file = (
        OUT_ROOT
        / (
            f"{lang}_test_"
            "jina_top100.npz"
        )
    )

    if output_file.exists():
        raise RuntimeError(
            "REFUSING TO OVERWRITE EXISTING "
            f"JINA TEST POOL: {output_file}"
        )

    print("=" * 80)
    print("R1-1 FROZEN JINA TEST POOL")
    print("=" * 80)
    print("language       =", lang)
    print("test queries   =", len(queries))
    print("model          =", hist.JINA_MODEL)
    print("query prefix   =", repr(hist.JINA_QUERY_PREFIX))
    print("code prefix    =", repr(hist.JINA_CODE_PREFIX))
    print("max length     =", hist.MAX_LENGTH)
    print("dim            =", DIM)
    print("top_k          =", TOP_K)
    print("query chunk    =", QUERY_CHUNK)
    print("code chunk     =", CODE_CHUNK)
    print("code cache     =", code_cache)
    print("query cache    =", query_cache)
    print("output         =", output_file)
    print(
        "rank semantics = "
        "#full-codebase scores strictly greater "
        "than GT score"
    )
    print("=" * 80)

    actual_dtype = encode_test_queries(
        lang,
        queries,
        query_cache,
    )

    (
        scores,
        indices,
        base_ranks,
        n_codes,
    ) = retrieve_exact_protocol(
        query_cache,
        code_cache,
        gt_indices,
    )

    if indices.shape != (
        len(queries),
        TOP_K,
    ):
        raise RuntimeError(
            f"Bad top100 shape "
            f"{indices.shape}"
        )

    if scores.shape != indices.shape:
        raise RuntimeError(
            "scores/indices shape mismatch"
        )

    if base_ranks.shape != (
        len(queries),
    ):
        raise RuntimeError(
            "Bad base_ranks shape"
        )

    if not np.isfinite(scores).all():
        raise RuntimeError(
            "Non-finite top100 scores"
        )

    if np.any(indices < 0) or np.any(
        indices >= n_codes
    ):
        raise RuntimeError(
            "Candidate index outside codebase"
        )

    actual_hit = np.any(
        indices
        == gt_indices[:, None],
        axis=1,
    )

    tie_rank_recall = (
        base_ranks < TOP_K
    )

    mrr = float(
        np.mean(
            1.0
            / (
                base_ranks.astype(
                    np.float64
                )
                + 1.0
            )
        )
    )

    tmp_output = output_file.with_name(
        output_file.name + ".tmp"
    )

    with tmp_output.open("wb") as f:
        np.savez_compressed(
            f,
            indices=np.asarray(
                indices,
                dtype=np.int64,
            ),
            scores=np.asarray(
                scores,
                dtype=np.float32,
            ),
            base_ranks=np.asarray(
                base_ranks,
                dtype=np.int64,
            ),
            gt_indices=np.asarray(
                gt_indices,
                dtype=np.int64,
            ),
        )

    os.replace(
        tmp_output,
        output_file,
    )

    builder_path = Path(__file__).resolve()

    summary = {
        "format":
            "r1_1_jina_test_top100_frozen_v1",
        "reviewer_item":
            "R1-1",
        "language":
            lang,
        "split":
            "test",
        "queries":
            int(len(queries)),
        "codes":
            int(n_codes),
        "top_k":
            TOP_K,
        "recipe": {
            **expected_query_recipe(),
            "actual_model_dtype":
                actual_dtype,
            "retrieval_query_chunk":
                QUERY_CHUNK,
            "retrieval_code_chunk":
                CODE_CHUNK,
            "scoring":
                "L2-normalized dot product",
            "full_codebase":
                True,
            "rank_semantics":
                "count(full_scores > gt_score)",
            "rank_zero_based":
                True,
            "rank_tie_policy":
                "strict-greater; equal-score candidates "
                "do not increase GT rank",
            "test_time_candidate_filtering":
                False,
        },
        "alignment_gates": {
            "gt_exact_equal_to_sealed_nomic":
                True,
            "jina_nomic_query_text_exact_equal":
                True,
        },
        "metrics": {
            "mrr":
                mrr,
            "r1":
                float(
                    np.mean(
                        base_ranks < 1
                    )
                ),
            "r5":
                float(
                    np.mean(
                        base_ranks < 5
                    )
                ),
            "r10":
                float(
                    np.mean(
                        base_ranks < 10
                    )
                ),
            "rank_based_recall_at_100":
                float(
                    np.mean(
                        tie_rank_recall
                    )
                ),
            "actual_gt_membership_in_top100":
                float(
                    np.mean(
                        actual_hit
                    )
                ),
            "actual_gt_hits":
                int(
                    actual_hit.sum()
                ),
        },
        "artifacts": {
            "test_jsonl":
                str(test_path.resolve()),
            "codebase_jsonl":
                str(codebase_path.resolve()),
            "frozen_jina_code_cache":
                str(code_cache.resolve()),
            "test_query_cache":
                str(query_cache.resolve()),
            "sealed_nomic_alignment_pool":
                str(
                    (
                        NOMIC_POOL_ROOT
                        / f"{lang}_test_nomic_top100.npz"
                    ).resolve()
                ),
            "output_pool":
                str(output_file.resolve()),
        },
        "sha256": {
            "builder":
                sha256(builder_path),
            "historical_jina_source":
                sha256(
                    ROOT
                    / "benchmark_dense_retrievers_valid.py"
                ),
            "frozen_recipe_source":
                sha256(
                    ROOT
                    / "build_jina_reviewer_groups.py"
                ),
            "nomic_rank_contract_source":
                sha256(
                    ROOT
                    / "export_nomic_candidate_pool.py"
                ),
            "query_cache":
                sha256(query_cache),
            "output_pool":
                sha256(output_file),
        },
    }

    summary_path = (
        output_file.with_suffix(
            ".json"
        )
    )

    atomic_json(
        summary_path,
        summary,
    )

    print()
    print("=" * 80)
    print("JINA TEST POOL COMPLETE")
    print("=" * 80)
    print(
        "language                       =",
        lang,
    )
    print(
        "queries                        =",
        len(queries),
    )
    print(
        "codes                          =",
        n_codes,
    )
    print(
        "MRR                            =",
        f"{mrr:.12f}",
    )
    print(
        "R@1                            =",
        f"{np.mean(base_ranks < 1):.12f}",
    )
    print(
        "R@5                            =",
        f"{np.mean(base_ranks < 5):.12f}",
    )
    print(
        "R@10                           =",
        f"{np.mean(base_ranks < 10):.12f}",
    )
    print(
        "rank_based_recall@100          =",
        f"{np.mean(tie_rank_recall):.12f}",
    )
    print(
        "actual_gt_membership@100       =",
        f"{np.mean(actual_hit):.12f}",
    )
    print(
        "actual_gt_hits                 =",
        int(actual_hit.sum()),
    )
    print(
        "output                         =",
        output_file,
    )
    print(
        "output_sha256                  =",
        sha256(output_file),
    )
    print(
        "summary                        =",
        summary_path,
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
