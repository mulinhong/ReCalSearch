#!/usr/bin/env python3

import argparse
import gc
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import benchmark_dense_retrievers_valid as hist


ROOT = Path(
    "/root/autodl-tmp/lora-code-embeddings-main/text2code"
)

DATA_ROOT = Path(
    "/root/autodl-tmp/lora-code-embeddings-main/data/CSN"
)

MANIFEST_PATH = (
    ROOT
    / "results/reviewer_revision"
    / "hn_source_ablation"
    / "final_protocol"
    / "train_query_manifest.json"
)

NOMIC_GROUP_ROOT = (
    ROOT
    / "results/jss_rescue_oldcsn/groups"
)

JINA_CODE_ROOT = (
    ROOT
    / "results/reviewer_revision"
    / "hn_source_ablation"
    / "retriever_strength_valid"
    / "jina_cache"
)

OUT_ROOT = (
    ROOT
    / "results/reviewer_revision"
    / "hn_source_ablation"
    / "final_groups"
)

JINA_TRAIN_CACHE_ROOT = (
    OUT_ROOT / "jina_train_cache"
)

JINA_TOP100_ROOT = (
    OUT_ROOT / "jina_top100"
)

LANGS = [
    "ruby",
    "javascript",
    "php",
    "python",
    "java",
    "go",
]

QUERY_BATCH = 64
QUERY_CHUNK = 64
CODE_CHUNK = 4096
TOP_K = 100
NUM_NEGATIVES = 7
DIM = 896


def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def canonicalize_code(code):
    return re.sub(
        r"\s+",
        " ",
        str(code).strip(),
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


def read_selected_rows(path, indices):
    wanted = set(
        int(x)
        for x in indices
    )

    found = {}

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as f:

        for idx, line in enumerate(f):

            if idx not in wanted:
                continue

            if not line.strip():
                raise RuntimeError(
                    f"Empty selected row {idx}"
                )

            found[idx] = json.loads(line)

            if len(found) == len(wanted):
                break

    missing = [
        idx
        for idx in indices
        if idx not in found
    ]

    if missing:
        raise RuntimeError(
            f"Missing selected rows: "
            f"{missing[:20]}"
        )

    return [
        found[int(idx)]
        for idx in indices
    ]


def load_manifest():
    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def encode_train_queries(
    encoder,
    lang,
    rows,
    cache_path,
):
    cache_path = Path(cache_path)

    queries = [
        hist.get_query_text(row)
        for row in rows
    ]

    if cache_path.exists():
        arr = np.load(
            cache_path,
            mmap_mode="r",
        )

        if arr.shape != (
            len(rows),
            DIM,
        ):
            raise RuntimeError(
                f"Bad train query cache "
                f"{cache_path}: {arr.shape}"
            )

        print(
            "REUSING_JINA_TRAIN_QUERY_CACHE",
            cache_path,
        )

        return cache_path

    out = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.float16,
        shape=(
            len(rows),
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
        desc=f"{lang} Jina frozen train query",
    ):

        batch_text = queries[
            start:start + QUERY_BATCH
        ]

        # CRITICAL:
        # exact historical encoder + prefix.
        emb = encoder.encode_batch(
            batch_text,
            hist.JINA_QUERY_PREFIX,
        )

        out[
            offset:
            offset + len(emb)
        ] = emb

        offset += len(emb)

    if offset != len(rows):
        raise RuntimeError(
            f"{lang}: encoded "
            f"{offset}, expected {len(rows)}"
        )

    out.flush()
    del out

    print(
        "JINA_TRAIN_QUERY_CACHE_SAVED",
        cache_path,
    )

    return cache_path


@torch.no_grad()
def retrieve_top100(
    query_cache_path,
    code_cache_path,
    output_npz,
):
    q_np = np.load(
        query_cache_path,
        mmap_mode="r",
    )

    c_np = np.load(
        code_cache_path,
        mmap_mode="r",
    )

    if q_np.shape[1] != DIM:
        raise RuntimeError(
            f"Bad query dim {q_np.shape}"
        )

    if c_np.shape[1] != DIM:
        raise RuntimeError(
            f"Bad code dim {c_np.shape}"
        )

    n_queries = q_np.shape[0]
    n_codes = c_np.shape[0]

    final_scores = np.empty(
        (n_queries, TOP_K),
        dtype=np.float32,
    )

    final_indices = np.empty(
        (n_queries, TOP_K),
        dtype=np.int64,
    )

    device = torch.device("cuda")

    for qs in tqdm(
        range(
            0,
            n_queries,
            QUERY_CHUNK,
        ),
        desc="Jina full-code top100",
    ):
        qe = min(
            qs + QUERY_CHUNK,
            n_queries,
        )

        q = torch.from_numpy(
            np.asarray(
                q_np[qs:qe],
                dtype=np.float32,
            )
        ).to(device)

        # Same scoring normalization
        # as historical evaluator.
        q = F.normalize(
            q,
            dim=1,
        )

        best_scores = None
        best_indices = None

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
                )
            ).to(device)

            c = F.normalize(
                c,
                dim=1,
            )

            scores = q @ c.T

            local_k = min(
                TOP_K,
                ce - cs,
            )

            local_scores, local_idx = (
                torch.topk(
                    scores,
                    k=local_k,
                    dim=1,
                    largest=True,
                    sorted=True,
                )
            )

            local_idx = (
                local_idx + cs
            )

            if best_scores is None:
                best_scores = local_scores
                best_indices = local_idx
            else:
                merged_scores = torch.cat(
                    [
                        best_scores,
                        local_scores,
                    ],
                    dim=1,
                )

                merged_indices = torch.cat(
                    [
                        best_indices,
                        local_idx,
                    ],
                    dim=1,
                )

                keep_scores, keep_pos = (
                    torch.topk(
                        merged_scores,
                        k=TOP_K,
                        dim=1,
                        largest=True,
                        sorted=True,
                    )
                )

                best_indices = torch.gather(
                    merged_indices,
                    1,
                    keep_pos,
                )

                best_scores = keep_scores

            del (
                c,
                scores,
                local_scores,
                local_idx,
            )

        final_scores[qs:qe] = (
            best_scores
            .float()
            .cpu()
            .numpy()
        )

        final_indices[qs:qe] = (
            best_indices
            .long()
            .cpu()
            .numpy()
        )

        del (
            q,
            best_scores,
            best_indices,
        )

        torch.cuda.empty_cache()

    np.savez(
        output_npz,
        scores=final_scores,
        indices=final_indices,
    )

    print(
        "JINA_TOP100_SAVED",
        output_npz,
    )

    return (
        final_scores,
        final_indices,
    )


def collect_candidate_code(
    codebase_path,
    candidate_indices,
):
    wanted = set(
        int(x)
        for x in
        np.unique(candidate_indices)
    )

    result = {}

    with Path(codebase_path).open(
        "r",
        encoding="utf-8",
    ) as f:

        for idx, line in enumerate(
            tqdm(
                f,
                desc="Collect candidate code text",
            )
        ):
            if idx not in wanted:
                continue

            row = json.loads(line)

            result[idx] = (
                canonicalize_code(
                    hist.get_code_text(row)
                )
            )

            if len(result) == len(wanted):
                break

    if len(result) != len(wanted):
        missing = sorted(
            wanted - set(result)
        )

        raise RuntimeError(
            f"Missing candidate code text: "
            f"{missing[:20]}"
        )

    return result


def nomic_cache_paths(lang):
    tag = (
        "root_autodl-tmp_hf_models_"
        "nomic-embed-code_"
        "msl512_14805c3aa7"
    )

    return {
        "query":
            ROOT
            / "data/rerank/"
            "coderank_cache_oldcsn"
            / f"{lang}_train_{tag}.query.npy",

        "code":
            ROOT
            / "data/rerank/"
            "coderank_cache_oldcsn"
            / f"{lang}_{tag}.code.npy",
    }


def compute_nomic_scores(
    nomic_query_cache,
    nomic_code_cache,
    train_indices,
    selected_negative_indices,
):
    q_np = np.load(
        nomic_query_cache,
        mmap_mode="r",
    )

    c_np = np.load(
        nomic_code_cache,
        mmap_mode="r",
    )

    output = []

    for row_idx, negs in zip(
        train_indices,
        selected_negative_indices,
    ):
        q = np.asarray(
            q_np[int(row_idx)],
            dtype=np.float32,
        )

        q = q / max(
            np.linalg.norm(q),
            1e-12,
        )

        c = np.asarray(
            c_np[
                np.asarray(
                    negs,
                    dtype=np.int64,
                )
            ],
            dtype=np.float32,
        )

        c /= np.maximum(
            np.linalg.norm(
                c,
                axis=1,
                keepdims=True,
            ),
            1e-12,
        )

        scores = c @ q

        output.append(
            [
                float(x)
                for x in scores
            ]
        )

    return output


def build_language(lang):

    manifest = load_manifest()

    indices = [
        int(x)
        for x in
        manifest["languages"][lang]["indices"]
    ]

    if len(indices) != 5000:
        raise RuntimeError(
            f"{lang}: expected 5000 indices"
        )

    train_path = (
        DATA_ROOT
        / lang
        / "train.jsonl"
    )

    codebase_path = (
        DATA_ROOT
        / lang
        / "codebase.jsonl"
    )

    nomic_group_path = (
        NOMIC_GROUP_ROOT
        / (
            f"{lang}_train_"
            "nomic_hn31_5000.jsonl"
        )
    )

    rows = read_selected_rows(
        train_path,
        indices,
    )

    nomic_groups = read_jsonl(
        nomic_group_path
    )

    if len(nomic_groups) != 5000:
        raise RuntimeError(
            "Bad Nomic group count"
        )

    expected_qids = [
        f"{lang}:{idx}"
        for idx in indices
    ]

    actual_qids = [
        str(x["query_id"])
        for x in nomic_groups
    ]

    if actual_qids != expected_qids:
        raise RuntimeError(
            "Nomic/manifest qid mismatch"
        )

    # Exact query identity gate.
    for i, (
        row,
        group,
    ) in enumerate(
        zip(rows, nomic_groups)
    ):
        q = hist.get_query_text(row)

        if q != group["query"]:
            raise RuntimeError(
                f"{lang} query mismatch "
                f"at local row {i}"
            )

    JINA_TRAIN_CACHE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    JINA_TOP100_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    query_cache = (
        JINA_TRAIN_CACHE_ROOT
        / (
            f"{lang}_train_manifest5000_"
            "jina_code_0p5b_"
            "nl2code_msl512.query.npy"
        )
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
            code_cache
        )

    encoder = hist.JinaEncoder()

    encode_train_queries(
        encoder,
        lang,
        rows,
        query_cache,
    )

    del encoder
    gc.collect()
    torch.cuda.empty_cache()

    top100_npz = (
        JINA_TOP100_ROOT
        / (
            f"{lang}_train_"
            "jina_top100_5000.npz"
        )
    )

    if top100_npz.exists():
        payload = np.load(
            top100_npz
        )

        top_scores = payload["scores"]
        top_indices = payload["indices"]

        if (
            top_scores.shape
            != (5000, TOP_K)
            or top_indices.shape
            != (5000, TOP_K)
        ):
            raise RuntimeError(
                "Bad existing top100"
            )

        print(
            "REUSING_JINA_TOP100",
            top100_npz,
        )
    else:
        (
            top_scores,
            top_indices,
        ) = retrieve_top100(
            query_cache,
            code_cache,
            top100_npz,
        )

    candidate_code = (
        collect_candidate_code(
            codebase_path,
            top_indices,
        )
    )

    selected_negs = []
    selected_ranks = []
    selected_scores = []
    positive_ranks = []
    positive_scores = []

    for local_i in range(5000):

        row = rows[local_i]
        group = nomic_groups[local_i]

        pos_idx = int(
            group["positive"]["code_index"]
        )

        pos_canon = canonicalize_code(
            hist.get_code_text(row)
        )

        used = {pos_idx}

        negs = []
        ranks = []
        scores = []

        pos_rank = None
        pos_score = None

        for rank, (
            score,
            code_idx,
        ) in enumerate(
            zip(
                top_scores[local_i],
                top_indices[local_i],
            ),
            start=1,
        ):
            code_idx = int(code_idx)
            score = float(score)

            is_positive = (
                code_idx == pos_idx
                or candidate_code[
                    code_idx
                ] == pos_canon
            )

            if is_positive:
                if pos_rank is None:
                    pos_rank = rank
                    pos_score = score

                used.add(code_idx)
                continue

            if code_idx in used:
                continue

            negs.append(code_idx)
            ranks.append(rank)
            scores.append(score)

            used.add(code_idx)

            if len(negs) == NUM_NEGATIVES:
                break

        if len(negs) != NUM_NEGATIVES:
            raise RuntimeError(
                f"{lang}:{indices[local_i]} "
                f"only {len(negs)} legal "
                f"negatives in Jina top100"
            )

        selected_negs.append(negs)
        selected_ranks.append(ranks)
        selected_scores.append(scores)
        positive_ranks.append(pos_rank)
        positive_scores.append(pos_score)

    nomic_paths = nomic_cache_paths(
        lang
    )

    for p in nomic_paths.values():
        if not Path(p).is_file():
            raise FileNotFoundError(p)

    deployed_nomic_scores = (
        compute_nomic_scores(
            nomic_paths["query"],
            nomic_paths["code"],
            indices,
            selected_negs,
        )
    )

    output_path = (
        OUT_ROOT
        / (
            f"{lang}_train_"
            "jina_top100_5000.jsonl"
        )
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for i in range(5000):

            group = nomic_groups[i]

            payload = {
                "format":
                    "grouped_indices_v1",

                "mining":
                    "jina",

                "mining_retriever":
                    "jina-code-embeddings-0.5b",

                "query_id":
                    group["query_id"],

                "query":
                    group["query"],

                "language":
                    lang,

                "positive": {
                    "code_index":
                        int(
                            group[
                                "positive"
                            ][
                                "code_index"
                            ]
                        ),

                    "code": None,

                    "retrieval_rank":
                        positive_ranks[i],

                    "retrieval_score":
                        positive_scores[i],
                },

                "negative_code_indices":
                    selected_negs[i],

                "negative_retrieval_ranks":
                    selected_ranks[i],

                "negative_retrieval_scores":
                    selected_scores[i],

                "negative_deployed_nomic_scores":
                    deployed_nomic_scores[i],

                "source_top_k":
                    TOP_K,
            }

            f.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Immediate source-specific audit.
    rows_out = read_jsonl(
        output_path
    )

    pass_count = (
        len(rows_out) == 5000
    )

    qids_ok = (
        [
            x["query_id"]
            for x in rows_out
        ]
        == expected_qids
    )

    query_ok = all(
        a["query"] == b["query"]
        for a, b in zip(
            rows_out,
            nomic_groups,
        )
    )

    positive_ok = all(
        int(
            a["positive"]["code_index"]
        )
        ==
        int(
            b["positive"]["code_index"]
        )
        for a, b in zip(
            rows_out,
            nomic_groups,
        )
    )

    seven_ok = all(
        len(
            x["negative_code_indices"]
        ) == 7
        and len(
            set(
                x["negative_code_indices"]
            )
        ) == 7
        and len(
            x[
                "negative_deployed_nomic_scores"
            ]
        ) == 7
        for x in rows_out
    )

    all_pass = all([
        pass_count,
        qids_ok,
        query_ok,
        positive_ok,
        seven_ok,
    ])

    provenance = {
        "format":
            "jina_reviewer_hn_v1",

        "language":
            lang,

        "manifest":
            str(MANIFEST_PATH),

        "manifest_sha256":
            sha256(MANIFEST_PATH),

        "historical_generator":
            str(
                Path(
                    hist.__file__
                ).resolve()
            ),

        "historical_generator_sha256":
            sha256(hist.__file__),

        "query_batch":
            QUERY_BATCH,

        "query_chunk":
            QUERY_CHUNK,

        "code_chunk":
            CODE_CHUNK,

        "source_top_k":
            TOP_K,

        "negatives":
            NUM_NEGATIVES,

        "query_cache":
            str(query_cache),

        "query_cache_sha256":
            sha256(query_cache),

        "jina_code_cache":
            str(code_cache),

        "jina_code_cache_sha256":
            sha256(code_cache),

        "nomic_query_cache":
            str(
                nomic_paths["query"]
            ),

        "nomic_code_cache":
            str(
                nomic_paths["code"]
            ),

        "top100_file":
            str(top100_npz),

        "top100_sha256":
            sha256(top100_npz),

        "output_group":
            str(output_path),

        "output_sha256":
            sha256(output_path),

        "audit": {
            "rows_5000":
                pass_count,

            "qid_order":
                qids_ok,

            "query_equal":
                query_ok,

            "positive_equal":
                positive_ok,

            "exactly_7_unique_negatives":
                seven_ok,

            "pass":
                all_pass,
        },
    }

    prov_path = (
        OUT_ROOT
        / (
            f"{lang}_train_"
            "jina_top100_5000."
            "provenance.json"
        )
    )

    with prov_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            provenance,
            f,
            indent=2,
            ensure_ascii=False,
        )

        f.write("\n")

    print()
    print(
        f"{lang} "
        f"JINA_GROUP_ROWS={len(rows_out)} "
        f"QID={qids_ok} "
        f"QUERY={query_ok} "
        f"POSITIVE={positive_ok} "
        f"NEG7={seven_ok} "
        f"PASS={all_pass}"
    )

    print(
        "OUTPUT=",
        output_path,
    )

    print(
        "OUTPUT_SHA256=",
        sha256(output_path),
    )

    if not all_pass:
        raise SystemExit(2)

    print(
        f"JINA_{lang.upper()}_GROUP_PASS=True"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--language",
        required=True,
        choices=LANGS,
    )

    args = parser.parse_args()

    build_language(
        args.language
    )


if __name__ == "__main__":
    main()
