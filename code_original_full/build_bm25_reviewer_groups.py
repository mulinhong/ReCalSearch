#!/usr/bin/env python3

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from eval_nomic_embed import (
    coderank_cache_paths,
    get_code_text,
    get_query_text,
)

TOKEN_RE = re.compile(r"(?u)\b\w+\b")


def tokenize(text):
    return TOKEN_RE.findall(str(text).lower())


def canonicalize_code(code):
    return re.sub(r"\s+", " ", str(code).strip())


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_selected_train_rows(path, row_ids):
    wanted = set(int(x) for x in row_ids)
    found = {}

    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in wanted:
                found[i] = json.loads(line)
                if len(found) == len(wanted):
                    break

    missing = sorted(wanted - set(found))
    if missing:
        raise RuntimeError(
            f"Missing train rows: {missing[:10]}"
        )

    return [found[int(i)] for i in row_ids]


def qid_to_row(qid, language):
    prefix = f"{language}:"
    if not str(qid).startswith(prefix):
        raise ValueError(
            f"Unexpected query_id={qid}"
        )
    return int(str(qid)[len(prefix):])


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def describe(x):
    a = np.asarray(x, dtype=np.float64)

    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "p10": float(np.quantile(a, 0.10)),
        "p25": float(np.quantile(a, 0.25)),
        "p50": float(np.quantile(a, 0.50)),
        "p75": float(np.quantile(a, 0.75)),
        "p90": float(np.quantile(a, 0.90)),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--language", required=True)
    ap.add_argument(
        "--data-root",
        default="/root/autodl-tmp/"
        "lora-code-embeddings-main/data/CSN",
    )
    ap.add_argument(
        "--reference-nomic-groups",
        required=True,
    )
    ap.add_argument("--output-file", required=True)
    ap.add_argument("--report-file", required=True)

    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument(
        "--score-batch-size",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--nomic-model-path",
        default="/root/autodl-tmp/"
        "hf_models/nomic-embed-code",
    )
    ap.add_argument(
        "--nomic-cache-dir",
        default="data/rerank/coderank_cache_oldcsn",
    )
    ap.add_argument(
        "--nomic-max-length",
        type=int,
        default=512,
    )
    ap.add_argument(
        "--nomic-query-prefix",
        default=(
            "Represent this query for searching "
            "relevant code: "
        ),
    )
    ap.add_argument(
        "--nomic-code-prefix",
        default="",
    )

    args = ap.parse_args()

    lang = args.language
    data_root = Path(args.data_root)

    refs = read_jsonl(
        args.reference_nomic_groups
    )

    if len(refs) != 5000:
        raise RuntimeError(
            f"Expected 5000 reference groups, "
            f"got {len(refs)}"
        )

    row_ids = np.asarray(
        [
            qid_to_row(r["query_id"], lang)
            for r in refs
        ],
        dtype=np.int64,
    )

    if len(np.unique(row_ids)) != len(row_ids):
        raise RuntimeError(
            "Duplicate query row IDs"
        )

    train_rows = load_selected_train_rows(
        data_root / lang / "train.jsonl",
        row_ids,
    )

    queries = [
        get_query_text(x)
        for x in train_rows
    ]

    for i, (ref, query) in enumerate(
        zip(refs, queries)
    ):
        if str(ref["query"]) != str(query):
            raise RuntimeError(
                f"Query mismatch at {i}"
            )

    positive_indices = np.asarray(
        [
            int(r["positive"]["code_index"])
            for r in refs
        ],
        dtype=np.int64,
    )

    print(
        f"REFERENCE_ALIGNMENT=PASSED "
        f"language={lang} n={len(refs)}"
    )

    print(
        f"Loading full codebase for {lang}..."
    )

    code_texts = []
    doc_lengths = []

    codebase_path = (
        data_root / lang / "codebase.jsonl"
    )

    with codebase_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in tqdm(
            f,
            desc="Reading codebase",
        ):
            if not line.strip():
                continue

            row = json.loads(line)
            code = get_code_text(row)

            code_texts.append(code)
            doc_lengths.append(
                len(tokenize(code))
            )

    n_codes = len(code_texts)

    if np.any(positive_indices < 0):
        raise RuntimeError(
            "Negative positive index"
        )

    if np.any(positive_indices >= n_codes):
        raise RuntimeError(
            "Positive code index outside codebase"
        )

    doc_lengths = np.asarray(
        doc_lengths,
        dtype=np.float32,
    )

    avgdl = float(doc_lengths.mean())

    query_terms = sorted(
        {
            tok
            for q in queries
            for tok in tokenize(q)
        }
    )

    if not query_terms:
        raise RuntimeError(
            "Empty query vocabulary"
        )

    vocab = {
        term: i
        for i, term in enumerate(query_terms)
    }

    print(
        f"codebase={n_codes} "
        f"query_vocab={len(vocab)} "
        f"avgdl={avgdl:.4f}"
    )

    vectorizer = CountVectorizer(
        vocabulary=vocab,
        tokenizer=tokenize,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        dtype=np.int32,
    )

    print(
        "Building sparse code term matrix..."
    )

    X = vectorizer.transform(
        code_texts
    ).tocsr()

    Q = vectorizer.transform(
        queries
    ).tocsr().astype(np.float32)

    # Standard BM25 query-side treatment:
    # each query term contributes once.
    Q.data.fill(1.0)

    df = np.asarray(
        (X > 0).sum(axis=0)
    ).ravel().astype(np.float32)

    N = float(n_codes)

    idf = np.log(
        1.0
        + (
            N - df + 0.5
        ) / (
            df + 0.5
        )
    ).astype(np.float32)

    print(
        "Converting document term counts "
        "to BM25 weights..."
    )

    coo = X.tocoo(copy=False)

    tf = coo.data.astype(
        np.float32,
        copy=False,
    )

    norm = (
        args.k1
        * (
            1.0
            - args.b
            + args.b
            * (
                doc_lengths[coo.row]
                / avgdl
            )
        )
    )

    weights = (
        idf[coo.col]
        * (
            tf
            * (
                args.k1 + 1.0
            )
        )
        / (
            tf + norm
        )
    ).astype(np.float32)

    D = sp.csr_matrix(
        (
            weights,
            (
                coo.row,
                coo.col,
            ),
        ),
        shape=X.shape,
        dtype=np.float32,
    )

    del X
    del coo
    del tf
    del weights
    del norm

    DT = D.transpose().tocsc()

    top_k = int(args.top_k)

    candidate_indices = np.empty(
        (
            len(queries),
            top_k,
        ),
        dtype=np.int64,
    )

    candidate_scores = np.empty(
        (
            len(queries),
            top_k,
        ),
        dtype=np.float32,
    )

    zero_tail_queries = 0
    probe_k = min(
        n_codes,
        max(
            top_k + 2048,
            top_k * 20,
        ),
    )

    print(
        f"BM25 retrieval: "
        f"queries={len(queries)} "
        f"codebase={n_codes} "
        f"top_k={top_k}"
    )

    for start in tqdm(
        range(
            0,
            len(queries),
            args.score_batch_size,
        ),
        desc="BM25 retrieval",
    ):
        end = min(
            start
            + args.score_batch_size,
            len(queries),
        )

        scores = (
            Q[start:end] @ DT
        ).toarray().astype(
            np.float32,
            copy=False,
        )

        for local in range(end - start):
            qi = start + local
            row = scores[local]

            pos_idx = int(
                positive_indices[qi]
            )

            row[pos_idx] = -np.inf

            candidate = np.argpartition(
                -row,
                probe_k - 1,
            )[:probe_k]

            candidate = candidate[
                np.lexsort(
                    (
                        candidate,
                        -row[candidate],
                    )
                )
            ]

            pos_canon = canonicalize_code(
                code_texts[pos_idx]
            )

            chosen = []
            chosen_scores = []

            for idx in candidate:
                idx = int(idx)

                if idx == pos_idx:
                    continue

                if canonicalize_code(
                    code_texts[idx]
                ) == pos_canon:
                    continue

                chosen.append(idx)
                chosen_scores.append(
                    float(row[idx])
                )

                if len(chosen) >= top_k:
                    break

            if len(chosen) < top_k:
                raise RuntimeError(
                    f"{lang} query {qi}: "
                    f"only {len(chosen)} "
                    f"BM25 candidates "
                    f"after filtering"
                )

            candidate_indices[qi] = (
                np.asarray(
                    chosen,
                    dtype=np.int64,
                )
            )

            candidate_scores[qi] = (
                np.asarray(
                    chosen_scores,
                    dtype=np.float32,
                )
            )

            if chosen_scores[-1] <= 0:
                zero_tail_queries += 1

    print(
        "BM25_FULL_CODEBASE_RETRIEVAL=PASSED"
    )

    print(
        "Loading deployed Nomic "
        "train/code caches..."
    )

    cache = coderank_cache_paths(
        cache_dir=args.nomic_cache_dir,
        language=lang,
        split="train",
        model_name=args.nomic_model_path,
        max_seq_length=args.nomic_max_length,
        backend="nomic_transformers",
        query_prefix=(
            args.nomic_query_prefix
        ),
        code_prefix=(
            args.nomic_code_prefix
        ),
    )

    query_emb = np.load(
        cache["query"],
        mmap_mode="r",
    )

    code_emb = np.load(
        cache["code"],
        mmap_mode="r",
    )

    if code_emb.shape[0] != n_codes:
        raise RuntimeError(
            f"Nomic code cache rows "
            f"{code_emb.shape[0]} "
            f"!= codebase {n_codes}"
        )

    if int(row_ids.max()) >= (
        query_emb.shape[0]
    ):
        raise RuntimeError(
            "Query row exceeds "
            "Nomic train cache"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    deployed_scores = np.empty(
        candidate_scores.shape,
        dtype=np.float32,
    )

    score_bs = 32

    for start in tqdm(
        range(
            0,
            len(refs),
            score_bs,
        ),
        desc="Nomic scoring BM25 candidates",
    ):
        end = min(
            start + score_bs,
            len(refs),
        )

        q_np = np.asarray(
            query_emb[
                row_ids[start:end]
            ],
            dtype=np.float32,
        )

        ids = candidate_indices[
            start:end
        ]

        c_np = np.asarray(
            code_emb[
                ids.reshape(-1)
            ],
            dtype=np.float32,
        ).reshape(
            end - start,
            top_k,
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

        s = torch.einsum(
            "bd,bkd->bk",
            q_t,
            c_t,
        )

        deployed_scores[
            start:end
        ] = (
            s.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        del q_t
        del c_t
        del s

        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_path = Path(
        args.output_file
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for i, ref in enumerate(refs):

            row = {
                "format":
                    "grouped_indices_v2_bm25_fullcode",
                "mining": "bm25",
                "mining_retriever":
                    "bm25_full_codebase",
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
                    "retrieval_rank": None,
                    "retrieval_score": None,
                },
                "negative_code_indices":
                    candidate_indices[
                        i
                    ].astype(
                        int
                    ).tolist(),
                "negative_retrieval_ranks":
                    list(
                        range(
                            1,
                            top_k + 1,
                        )
                    ),
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

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    report = {
        "protocol":
            "reviewer_r1_1_bm25_fullcode_v1",
        "language":
            lang,
        "queries":
            len(refs),
        "codebase_size":
            n_codes,
        "top_k":
            top_k,
        "bm25_k1":
            args.k1,
        "bm25_b":
            args.b,
        "query_vocabulary_size":
            len(vocab),
        "average_document_length":
            avgdl,
        "zero_score_at_rank_k_queries":
            int(zero_tail_queries),
        "deployed_nomic_similarity":
            describe(
                deployed_scores.reshape(-1)
            ),
        "bm25_score":
            describe(
                candidate_scores.reshape(-1)
            ),
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

    print(
        json.dumps(
            report,
            indent=2,
        )
    )

    print(
        "OUTPUT_SHA256 =",
        sha256_file(
            output_path
        ),
    )

    print(
        "BM25_REVIEWER_GROUP_BUILD=PASSED"
    )


if __name__ == "__main__":
    main()
