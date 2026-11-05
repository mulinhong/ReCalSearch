#!/usr/bin/env python3

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer

from build_unixcoder_external_candidates import (
    encode_texts,
    validate_cache,
)

from eval_llm_verifier import retrieve_topk_from_cache

from eval_nomic_embed import (
    coderank_cache_paths,
    get_code_text,
    get_query_text,
)


def read_jsonl(path):
    rows = []
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )
    return rows


def load_selected_rows(
    path,
    indices,
):
    wanted = set(int(x) for x in indices)
    found = {}

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as f:
        for i, line in enumerate(f):
            if i in wanted:
                found[i] = json.loads(line)

                if len(found) == len(wanted):
                    break

    missing = wanted - set(found)

    if missing:
        raise RuntimeError(
            f"Missing train rows: "
            f"{sorted(missing)[:10]}"
        )

    return [
        found[int(i)]
        for i in indices
    ]


def row_from_qid(qid, language):
    prefix = f"{language}:"

    if not str(qid).startswith(prefix):
        raise ValueError(
            f"Unexpected query_id={qid}"
        )

    return int(
        str(qid)[len(prefix):]
    )


def canonicalize(text):
    return re.sub(
        r"\s+",
        " ",
        str(text).strip(),
    )


def canon_hash(text):
    return hashlib.sha1(
        canonicalize(text).encode(
            "utf-8",
            errors="ignore",
        )
    ).digest()


def describe(x):
    a = np.asarray(
        x,
        dtype=np.float64,
    )

    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "p10": float(
            np.quantile(a, 0.10)
        ),
        "p25": float(
            np.quantile(a, 0.25)
        ),
        "p50": float(
            np.quantile(a, 0.50)
        ),
        "p75": float(
            np.quantile(a, 0.75)
        ),
        "p90": float(
            np.quantile(a, 0.90)
        ),
        "p95": float(
            np.quantile(a, 0.95)
        ),
        "max": float(a.max()),
    }


def sha256_file(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--language",
        required=True,
    )

    ap.add_argument(
        "--data-root",
        required=True,
    )

    ap.add_argument(
        "--reference-nomic-groups",
        required=True,
    )

    ap.add_argument(
        "--model-path",
        default=(
            "/root/autodl-tmp/"
            "hf_models/unixcoder-base"
        ),
    )

    ap.add_argument(
        "--code-cache",
        required=True,
    )

    ap.add_argument(
        "--query-cache",
        required=True,
    )

    ap.add_argument(
        "--output-groups",
        required=True,
    )

    ap.add_argument(
        "--output-matched-groups",
        required=True,
    )

    ap.add_argument(
        "--report-file",
        required=True,
    )

    ap.add_argument(
        "--max-length",
        type=int,
        default=512,
    )

    ap.add_argument(
        "--query-batch-size",
        type=int,
        default=64,
    )

    ap.add_argument(
        "--candidate-top-k",
        type=int,
        default=100,
    )

    ap.add_argument(
        "--raw-retrieval-k",
        type=int,
        default=512,
    )

    ap.add_argument(
        "--query-chunk-size",
        type=int,
        default=64,
    )

    ap.add_argument(
        "--code-chunk-size",
        type=int,
        default=4096,
    )

    ap.add_argument(
        "--nomic-cache-dir",
        default=(
            "data/rerank/"
            "coderank_cache_oldcsn"
        ),
    )

    ap.add_argument(
        "--nomic-model-path",
        default=(
            "/root/autodl-tmp/"
            "hf_models/nomic-embed-code"
        ),
    )

    args = ap.parse_args()

    lang = args.language
    data_root = Path(args.data_root)

    refs = read_jsonl(
        args.reference_nomic_groups
    )

    if len(refs) != 5000:
        raise RuntimeError(
            f"Expected 5000 reference "
            f"groups, got {len(refs)}"
        )

    row_ids = np.asarray(
        [
            row_from_qid(
                r["query_id"],
                lang,
            )
            for r in refs
        ],
        dtype=np.int64,
    )

    if len(
        np.unique(row_ids)
    ) != len(row_ids):
        raise RuntimeError(
            "Duplicate train query IDs"
        )

    train_rows = load_selected_rows(
        data_root
        / lang
        / "train.jsonl",
        row_ids,
    )

    queries = [
        get_query_text(x)
        for x in train_rows
    ]

    for i, (ref, q) in enumerate(
        zip(refs, queries)
    ):
        if str(ref["query"]) != str(q):
            raise RuntimeError(
                f"Query alignment "
                f"failed at {i}"
            )

    positive_indices = np.asarray(
        [
            int(
                r["positive"][
                    "code_index"
                ]
            )
            for r in refs
        ],
        dtype=np.int64,
    )

    print(
        "TRAIN_QUERY_ALIGNMENT=PASSED "
        f"language={lang} "
        f"queries={len(refs)}"
    )

    # --------------------------------------------------
    # Full codebase audit + canonical hashes
    # --------------------------------------------------

    codebase_path = (
        data_root
        / lang
        / "codebase.jsonl"
    )

    code_hashes = []

    print(
        f"Reading {lang} codebase "
        "canonical hashes..."
    )

    with codebase_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in tqdm(
            f,
            desc="Codebase audit",
        ):
            if not line.strip():
                continue

            row = json.loads(line)

            code_hashes.append(
                canon_hash(
                    get_code_text(row)
                )
            )

    n_codes = len(code_hashes)

    if np.any(
        positive_indices < 0
    ) or np.any(
        positive_indices >= n_codes
    ):
        raise RuntimeError(
            "Positive index outside "
            "full codebase"
        )

    # --------------------------------------------------
    # Reuse existing full-codebase UniXcoder cache
    # --------------------------------------------------

    code_cache = Path(
        args.code_cache
    )

    code_embeddings = validate_cache(
        code_cache,
        n_codes,
        768,
    )

    print(
        "UNIXCODER_CODE_CACHE_AUDIT=PASSED "
        f"shape={code_embeddings.shape} "
        f"dtype={code_embeddings.dtype}"
    )

    # --------------------------------------------------
    # Encode ONLY the exact 5000 train queries
    # with original Work2 encoding protocol.
    # --------------------------------------------------

    query_cache = Path(
        args.query_cache
    )

    if query_cache.exists():

        query_embeddings = (
            validate_cache(
                query_cache,
                len(queries),
                768,
            )
        )

        print(
            "Reusing reviewer "
            f"query cache: {query_cache}"
        )

    else:

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA required to encode "
                "UniXcoder train queries"
            )

        device = torch.device("cuda")

        tokenizer = (
            RobertaTokenizer
            .from_pretrained(
                args.model_path
            )
        )

        model = (
            RobertaModel
            .from_pretrained(
                args.model_path
            )
            .to(device)
        )

        model.eval()

        if int(
            model.config.hidden_size
        ) != 768:
            raise RuntimeError(
                "Unexpected UniXcoder "
                "hidden size"
            )

        query_embeddings = (
            encode_texts(
                model=model,
                tokenizer=tokenizer,
                texts=queries,
                device=device,
                batch_size=(
                    args.query_batch_size
                ),
                max_length=(
                    args.max_length
                ),
                description=(
                    f"{lang} reviewer "
                    "train5000 query"
                ),
            )
        )

        query_cache.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            query_cache,
            query_embeddings,
        )

        print(
            "Saved reviewer query "
            f"cache: {query_cache}"
        )

        del model
        del tokenizer

        torch.cuda.empty_cache()

        query_embeddings = (
            validate_cache(
                query_cache,
                len(queries),
                768,
            )
        )

    print(
        "UNIXCODER_TRAIN_QUERY_CACHE=PASSED "
        f"shape={query_embeddings.shape} "
        f"dtype={query_embeddings.dtype}"
    )

    # --------------------------------------------------
    # Retrieve a larger raw pool, then remove:
    #   1) exact positive index
    #   2) canonical positive duplicates
    #
    # Final first 100 = UniXcoder full-codebase top100
    # negative candidate set.
    # --------------------------------------------------

    if (
        args.raw_retrieval_k
        < args.candidate_top_k + 1
    ):
        raise ValueError(
            "raw-retrieval-k must exceed "
            "candidate-top-k"
        )

    device = torch.device("cuda")

    raw_scores, raw_indices, _ = (
        retrieve_topk_from_cache(
            query_embeds=query_embeddings,
            code_embeds=code_embeddings,
            gt_indices=positive_indices,
            device=device,
            top_k=args.raw_retrieval_k,
            query_chunk_size=(
                args.query_chunk_size
            ),
            code_chunk_size=(
                args.code_chunk_size
            ),
        )
    )

    n_query = len(refs)
    k = args.candidate_top_k

    candidate_indices = np.empty(
        (n_query, k),
        dtype=np.int64,
    )

    candidate_scores = np.empty(
        (n_query, k),
        dtype=np.float32,
    )

    candidate_ranks = np.empty(
        (n_query, k),
        dtype=np.int32,
    )

    positive_removed = 0
    canonical_removed = 0

    for i in range(n_query):

        pos_idx = int(
            positive_indices[i]
        )

        pos_hash = code_hashes[
            pos_idx
        ]

        ids = []
        scores = []
        ranks = []

        for rank, (
            idx,
            score,
        ) in enumerate(
            zip(
                raw_indices[i],
                raw_scores[i],
            ),
            start=1,
        ):

            idx = int(idx)

            if idx == pos_idx:
                positive_removed += 1
                continue

            if (
                code_hashes[idx]
                == pos_hash
            ):
                canonical_removed += 1
                continue

            ids.append(idx)
            scores.append(
                float(score)
            )
            ranks.append(rank)

            if len(ids) == k:
                break

        if len(ids) != k:
            raise RuntimeError(
                f"Query {i} has only "
                f"{len(ids)} valid "
                "UniXcoder negatives"
            )

        candidate_indices[i] = ids
        candidate_scores[i] = scores
        candidate_ranks[i] = ranks

    print(
        "UNIXCODER_FULLCODE_TOP100=PASSED"
    )

    print(
        "positive_removed =",
        positive_removed,
    )

    print(
        "canonical_duplicates_removed =",
        canonical_removed,
    )

    del raw_scores
    del raw_indices

    # --------------------------------------------------
    # Score all UniXcoder top100 candidates with the
    # deployed Nomic retriever space.
    # --------------------------------------------------

    nomic_cache = coderank_cache_paths(
        cache_dir=(
            args.nomic_cache_dir
        ),
        language=lang,
        split="train",
        model_name=(
            args.nomic_model_path
        ),
        max_seq_length=512,
        backend="nomic_transformers",
        query_prefix=(
            "Represent this query for "
            "searching relevant code: "
        ),
        code_prefix="",
    )

    nomic_q = np.load(
        nomic_cache["query"],
        mmap_mode="r",
    )

    nomic_c = np.load(
        nomic_cache["code"],
        mmap_mode="r",
    )

    if (
        nomic_c.shape[0]
        != n_codes
    ):
        raise RuntimeError(
            "Nomic/codebase shape "
            "mismatch"
        )

    if (
        int(row_ids.max())
        >= nomic_q.shape[0]
    ):
        raise RuntimeError(
            "Nomic train query "
            "cache too short"
        )

    deployed_scores = np.empty(
        candidate_scores.shape,
        dtype=np.float32,
    )

    nomic_bs = 32

    for start in tqdm(
        range(
            0,
            n_query,
            nomic_bs,
        ),
        desc=(
            "Nomic scoring "
            "UniXcoder top100"
        ),
    ):

        end = min(
            start + nomic_bs,
            n_query,
        )

        q_np = np.asarray(
            nomic_q[
                row_ids[start:end]
            ],
            dtype=np.float32,
        )

        ids = candidate_indices[
            start:end
        ]

        c_np = np.asarray(
            nomic_c[
                ids.reshape(-1)
            ],
            dtype=np.float32,
        ).reshape(
            end - start,
            k,
            -1,
        )

        q_t = torch.from_numpy(
            q_np
        ).to(device)

        c_t = torch.from_numpy(
            c_np
        ).to(device)

        q_t = F.normalize(
            q_t,
            dim=1,
        )

        c_t = F.normalize(
            c_t,
            dim=2,
        )

        sims = torch.einsum(
            "bd,bkd->bk",
            q_t,
            c_t,
        )

        deployed_scores[
            start:end
        ] = (
            sims.cpu()
            .numpy()
            .astype(np.float32)
        )

        del q_t
        del c_t
        del sims

    print(
        "UNIXCODER_NOMIC_SCORING=PASSED"
    )

    # --------------------------------------------------
    # Write ordinary UniXcoder top100 groups.
    # First 7 will later be used as UniXcoder-HN.
    # --------------------------------------------------

    groups_path = Path(
        args.output_groups
    )

    groups_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_rows = []

    with groups_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for i, ref in enumerate(refs):

            row = {
                "format":
                    "grouped_indices_v2_"
                    "unixcoder_base",
                "mining":
                    "unixcoder_base",
                "mining_retriever":
                    "microsoft/"
                    "unixcoder-base",
                "query_id":
                    ref["query_id"],
                "query":
                    ref["query"],
                "language":
                    lang,
                "positive": {
                    "code_index":
                        int(
                            positive_indices[i]
                        ),
                    "code": None,
                    "retrieval_rank":
                        None,
                    "retrieval_score":
                        None,
                },
                "negative_code_indices":
                    candidate_indices[
                        i
                    ].astype(
                        int
                    ).tolist(),
                "negative_retrieval_ranks":
                    candidate_ranks[
                        i
                    ].astype(
                        int
                    ).tolist(),
                "negative_retrieval_scores":
                    candidate_scores[
                        i
                    ].astype(
                        float
                    ).tolist(),
                "negative_deployed_nomic_scores":
                    deployed_scores[
                        i
                    ].astype(
                        float
                    ).tolist(),
            }

            source_rows.append(row)

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # --------------------------------------------------
    # Hungarian optimal difficulty matching:
    # 7 Nomic targets <-> 100 UniXcoder candidates
    # --------------------------------------------------

    matched_path = Path(
        args.output_matched_groups
    )

    matched_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    target_all = []
    matched_all = []
    error_all = []
    per_query_mae = []
    matched_source_ranks = []

    below_target_max = 0
    below_target_min = 0

    q_mae_gt005 = 0
    q_mae_gt010 = 0

    unix7_ref_overlap = 0
    unix100_ref_overlap = 0
    matched_ref_overlap = 0

    with matched_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for i, ref in enumerate(refs):

            target = np.asarray(
                ref[
                    "negative_retrieval_scores"
                ][:7],
                dtype=np.float64,
            )

            cand = np.asarray(
                deployed_scores[i],
                dtype=np.float64,
            )

            if target.shape != (7,):
                raise RuntimeError(
                    "Reference Nomic group "
                    "does not have top-7 "
                    "scores"
                )

            cost = np.abs(
                target[:, None]
                - cand[None, :]
            )

            rows, cols = (
                linear_sum_assignment(
                    cost
                )
            )

            order = np.argsort(rows)
            rows = rows[order]
            cols = cols[order]

            selected_scores = cand[
                cols
            ]

            err = np.abs(
                target - selected_scores
            )

            qmae = float(
                err.mean()
            )

            target_all.extend(
                target.tolist()
            )

            matched_all.extend(
                selected_scores.tolist()
            )

            error_all.extend(
                err.tolist()
            )

            per_query_mae.append(
                qmae
            )

            matched_source_ranks.extend(
                candidate_ranks[
                    i,
                    cols,
                ].tolist()
            )

            if cand.max() < target.max():
                below_target_max += 1

            if cand.max() < target.min():
                below_target_min += 1

            if qmae > 0.05:
                q_mae_gt005 += 1

            if qmae > 0.10:
                q_mae_gt010 += 1

            ref7 = set(
                int(x)
                for x in ref[
                    "negative_code_indices"
                ][:7]
            )

            unix7 = set(
                int(x)
                for x in candidate_indices[
                    i,
                    :7,
                ]
            )

            unix100 = set(
                int(x)
                for x in candidate_indices[
                    i
                ]
            )

            selected_ids = [
                int(x)
                for x in candidate_indices[
                    i,
                    cols,
                ]
            ]

            unix7_ref_overlap += len(
                unix7 & ref7
            )

            unix100_ref_overlap += len(
                unix100 & ref7
            )

            matched_ref_overlap += sum(
                x in ref7
                for x in selected_ids
            )

            row = {
                "format":
                    "grouped_indices_v3_"
                    "unixcoder_nomic_"
                    "difficulty_matched",
                "mining":
                    "unixcoder_base_"
                    "difficulty_matched",
                "mining_retriever":
                    "microsoft/"
                    "unixcoder-base",
                "difficulty_reference":
                    "deployed_nomic_"
                    "similarity",
                "matching":
                    "hungarian_min_"
                    "absolute_error",
                "query_id":
                    ref["query_id"],
                "query":
                    ref["query"],
                "language":
                    lang,
                "positive": {
                    "code_index":
                        int(
                            positive_indices[i]
                        ),
                    "code": None,
                    "retrieval_rank":
                        None,
                    "retrieval_score":
                        None,
                },
                "negative_code_indices":
                    selected_ids,
                "negative_retrieval_ranks":
                    candidate_ranks[
                        i,
                        cols,
                    ].astype(
                        int
                    ).tolist(),
                "negative_retrieval_scores":
                    candidate_scores[
                        i,
                        cols,
                    ].astype(
                        float
                    ).tolist(),
                "negative_deployed_nomic_scores":
                    deployed_scores[
                        i,
                        cols,
                    ].astype(
                        float
                    ).tolist(),
                "matched_reference_nomic_scores":
                    target.astype(
                        float
                    ).tolist(),
            }

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    target_all = np.asarray(
        target_all,
        dtype=np.float64,
    )

    matched_all = np.asarray(
        matched_all,
        dtype=np.float64,
    )

    error_all = np.asarray(
        error_all,
        dtype=np.float64,
    )

    per_query_mae = np.asarray(
        per_query_mae,
        dtype=np.float64,
    )

    matched_source_ranks = np.asarray(
        matched_source_ranks,
        dtype=np.float64,
    )

    mean_shift = float(
        matched_all.mean()
        - target_all.mean()
    )

    mae = float(
        error_all.mean()
    )

    accepted = (
        mae <= 0.02
        and abs(mean_shift) <= 0.02
    )

    report = {
        "protocol":
            "reviewer_r1_1_"
            "unixcoder_dense_control_v1",
        "language":
            lang,
        "queries":
            n_query,
        "codebase_size":
            n_codes,
        "retriever":
            "microsoft/unixcoder-base",
        "adapter":
            None,
        "pooling":
            "attention_mask_mean",
        "l2_normalized":
            True,
        "max_length":
            args.max_length,
        "candidate_top_k":
            k,
        "raw_retrieval_k":
            args.raw_retrieval_k,
        "query_cache":
            str(
                query_cache.resolve()
            ),
        "code_cache":
            str(
                code_cache.resolve()
            ),
        "target_nomic_similarity":
            describe(target_all),
        "matched_nomic_similarity":
            describe(matched_all),
        "absolute_similarity_error":
            describe(error_all),
        "per_query_mae":
            describe(per_query_mae),
        "mean_similarity_shift":
            mean_shift,
        "queries_candidate_max_"
        "below_target_max":
            below_target_max,
        "queries_candidate_max_"
        "below_target_min":
            below_target_min,
        "queries_mae_gt_0_05":
            q_mae_gt005,
        "queries_mae_gt_0_10":
            q_mae_gt010,
        "matched_source_ranks":
            describe(
                matched_source_ranks
            ),
        "unixcoder_top7_overlap_"
        "with_nomic_top7_pairs":
            unix7_ref_overlap,
        "unixcoder_top100_overlap_"
        "with_nomic_top7_pairs":
            unix100_ref_overlap,
        "matched7_overlap_"
        "with_nomic_top7_pairs":
            matched_ref_overlap,
        "acceptance_thresholds": {
            "mae_max": 0.02,
            "absolute_mean_shift_max":
                0.02,
        },
        "difficulty_match_accepted":
            accepted,
    }

    report_path = Path(
        args.report_file
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")

    print(
        json.dumps(
            report,
            indent=2,
        )
    )

    print(
        "SOURCE_GROUP_SHA256 =",
        sha256_file(
            groups_path
        ),
    )

    print(
        "MATCHED_GROUP_SHA256 =",
        sha256_file(
            matched_path
        ),
    )

    print(
        "REPORT_SHA256 =",
        sha256_file(
            report_path
        ),
    )

    print(
        "UNIXCODER_REVIEWER_PILOT=PASSED"
    )

    print(
        "DIFFICULTY_MATCH_ACCEPTED =",
        accepted,
    )


if __name__ == "__main__":
    main()
