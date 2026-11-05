#!/usr/bin/env python3

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


LANGUAGES = [
    "ruby",
    "javascript",
    "php",
    "python",
    "java",
    "go",
]

DATA_ROOT = Path(
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

OUT_ROOT = Path(
    "results/reviewer_revision/"
    "hn_source_ablation/"
    "retriever_strength_valid"
)

JINA_CACHE = OUT_ROOT / "jina_cache"
RANK_ROOT = OUT_ROOT / "ranks"

JINA_MODEL = Path(
    "/root/autodl-tmp/hf_models/"
    "jina-code-embeddings-0.5b"
)

JINA_QUERY_PREFIX = (
    "Find the most relevant code snippet "
    "given the following query:\n"
)

JINA_CODE_PREFIX = (
    "Candidate code snippet:\n"
)

MAX_LENGTH = 512

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

JINA_CACHE.mkdir(
    parents=True,
    exist_ok=True,
)

RANK_ROOT.mkdir(
    parents=True,
    exist_ok=True,
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


def get_query_text(row):
    for key in [
        "doc",
        "docstring",
        "query",
        "nl",
        "text",
    ]:
        if key in row and row[key]:
            value = row[key]

            if isinstance(value, list):
                return " ".join(
                    map(str, value)
                )

            return str(value)

    if (
        "docstring_tokens" in row
        and row["docstring_tokens"]
    ):
        value = row["docstring_tokens"]

        if isinstance(value, list):
            return " ".join(
                map(str, value)
            )

        return str(value)

    raise KeyError(
        "Cannot resolve query text"
    )


def get_code_text(row):
    if "code" in row and row["code"]:
        return str(row["code"])

    if (
        "code_tokens" in row
        and row["code_tokens"]
    ):
        value = row["code_tokens"]

        if isinstance(value, list):
            return " ".join(
                map(str, value)
            )

        return str(value)

    raise KeyError(
        "Cannot resolve code text"
    )


def nomic_paths(lang):
    tag = (
        "root_autodl-tmp_hf_models_"
        "nomic-embed-code_"
        "msl512_14805c3aa7"
    )

    return {
        "query":
            NOMIC_ROOT /
            f"{lang}_valid_{tag}.query.npy",

        "code":
            NOMIC_ROOT /
            f"{lang}_{tag}.code.npy",
    }


def unixcoder_paths(lang):
    return {
        "query":
            UNIX_ROOT /
            (
                f"{lang}_valid_"
                "unixcoder_base_mean_"
                "msl512.query.npy"
            ),

        "code":
            UNIX_ROOT /
            (
                f"{lang}_"
                "unixcoder_base_mean_"
                "msl512.code.npy"
            ),
    }


def jina_paths(lang):
    return {
        "query":
            JINA_CACHE /
            (
                f"{lang}_valid_"
                "jina_code_0p5b_"
                "nl2code_msl512.query.npy"
            ),

        "code":
            JINA_CACHE /
            (
                f"{lang}_"
                "jina_code_0p5b_"
                "nl2code_msl512.code.npy"
            ),
    }


def load_shape(path):
    if not Path(path).is_file():
        raise FileNotFoundError(path)

    return np.load(
        path,
        mmap_mode="r",
    ).shape


class JinaEncoder:

    def __init__(self):

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA required"
            )

        self.device = torch.device(
            "cuda"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                JINA_MODEL,
                local_files_only=True,
            )
        )

        self.tokenizer.padding_side = (
            "left"
        )

        if (
            self.tokenizer.pad_token_id
            is None
        ):
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

        self.model = (
            AutoModel.from_pretrained(
                JINA_MODEL,
                dtype=dtype,
                local_files_only=True,
            )
            .to(self.device)
            .eval()
        )

        hidden = int(
            self.model.config.hidden_size
        )

        if hidden != 896:
            raise RuntimeError(
                f"Expected hidden=896, "
                f"got {hidden}"
            )

        print(
            "JINA_MODEL_READY "
            f"dtype={dtype} "
            f"hidden={hidden}"
        )


    @torch.no_grad()
    def encode_batch(
        self,
        texts,
        prefix,
    ):

        texts = [
            prefix + str(x)
            for x in texts
        ]

        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(
            **batch,
            use_cache=False,
        )

        hidden = (
            outputs.last_hidden_state
        )

        mask = batch[
            "attention_mask"
        ]

        # Official Jina last-token pooling.
        left_padding = bool(
            mask[:, -1].sum()
            == mask.shape[0]
        )

        if left_padding:
            pooled = hidden[:, -1]
        else:
            seq_lengths = (
                mask.sum(dim=1) - 1
            )

            pooled = hidden[
                torch.arange(
                    hidden.shape[0],
                    device=hidden.device,
                ),
                seq_lengths,
            ]

        pooled = F.normalize(
            pooled.float(),
            dim=1,
        )

        result = (
            pooled.cpu()
            .numpy()
            .astype(
                np.float16,
                copy=False,
            )
        )

        del (
            batch,
            outputs,
            hidden,
            mask,
            pooled,
        )

        return result


def encode_jina_queries(
    encoder,
    lang,
    valid_rows,
    path,
    batch_size=64,
):
    path = Path(path)

    if path.exists():

        arr = np.load(
            path,
            mmap_mode="r",
        )

        if arr.shape != (
            len(valid_rows),
            896,
        ):
            raise RuntimeError(
                f"Bad query cache: "
                f"{path} {arr.shape}"
            )

        print(
            f"Reusing Jina query "
            f"cache: {path}"
        )

        return


    queries = [
        get_query_text(x)
        for x in valid_rows
    ]

    out = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float16,
        shape=(
            len(queries),
            896,
        ),
    )

    offset = 0

    for start in tqdm(
        range(
            0,
            len(queries),
            batch_size,
        ),
        desc=f"{lang} Jina valid query",
    ):

        batch_text = queries[
            start:
            start + batch_size
        ]

        emb = encoder.encode_batch(
            batch_text,
            JINA_QUERY_PREFIX,
        )

        out[
            offset:
            offset + len(emb)
        ] = emb

        offset += len(emb)

    out.flush()

    del out

    print(
        "JINA_QUERY_CACHE_SAVED",
        path,
    )


def encode_jina_codebase(
    encoder,
    lang,
    code_path,
    cache_path,
    expected_rows,
    batch_size=32,
):

    cache_path = Path(
        cache_path
    )

    if cache_path.exists():

        arr = np.load(
            cache_path,
            mmap_mode="r",
        )

        if arr.shape != (
            expected_rows,
            896,
        ):
            raise RuntimeError(
                f"Bad Jina code cache: "
                f"{cache_path} "
                f"{arr.shape}"
            )

        print(
            f"Reusing Jina code "
            f"cache: {cache_path}"
        )

        return


    out = (
        np.lib.format.open_memmap(
            cache_path,
            mode="w+",
            dtype=np.float16,
            shape=(
                expected_rows,
                896,
            ),
        )
    )

    texts = []
    offset = 0

    def flush():
        nonlocal texts, offset

        if not texts:
            return

        emb = encoder.encode_batch(
            texts,
            JINA_CODE_PREFIX,
        )

        out[
            offset:
            offset + len(emb)
        ] = emb

        offset += len(emb)

        texts = []


    with Path(code_path).open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in tqdm(
            f,
            desc=f"{lang} Jina code",
            total=expected_rows,
        ):

            if not line.strip():
                continue

            row = json.loads(line)

            texts.append(
                get_code_text(row)
            )

            if len(texts) >= batch_size:
                flush()

    flush()

    if offset != expected_rows:
        raise RuntimeError(
            f"{lang}: encoded "
            f"{offset} codes, "
            f"expected {expected_rows}"
        )

    out.flush()
    del out

    print(
        "JINA_CODE_CACHE_SAVED",
        cache_path,
    )


@torch.no_grad()
def evaluate_cache(
    query_path,
    code_path,
    gt_indices,
    rank_output,
    query_chunk=64,
    code_chunk=4096,
):

    q_np = np.load(
        query_path,
        mmap_mode="r",
    )

    c_np = np.load(
        code_path,
        mmap_mode="r",
    )

    if q_np.shape[0] != len(
        gt_indices
    ):
        raise RuntimeError(
            "Query cache row mismatch: "
            f"{query_path}"
        )

    n_queries = q_np.shape[0]
    n_codes = c_np.shape[0]

    if np.any(
        gt_indices < 0
    ) or np.any(
        gt_indices >= n_codes
    ):
        raise RuntimeError(
            "GT index outside codebase"
        )

    device = torch.device(
        "cuda"
    )

    ranks = np.empty(
        n_queries,
        dtype=np.int64,
    )

    for qs in tqdm(
        range(
            0,
            n_queries,
            query_chunk,
        ),
        desc=(
            f"Metric "
            f"{Path(query_path).name[:30]}"
        ),
    ):

        qe = min(
            qs + query_chunk,
            n_queries,
        )

        q = torch.from_numpy(
            np.asarray(
                q_np[qs:qe],
                dtype=np.float32,
            )
        ).to(device)

        q = F.normalize(
            q,
            dim=1,
        )

        gt = gt_indices[qs:qe]

        gt_emb = torch.from_numpy(
            np.asarray(
                c_np[gt],
                dtype=np.float32,
            )
        ).to(device)

        gt_emb = F.normalize(
            gt_emb,
            dim=1,
        )

        gt_scores = (
            q * gt_emb
        ).sum(dim=1)

        rank = torch.zeros(
            qe - qs,
            dtype=torch.long,
            device=device,
        )

        for cs in range(
            0,
            n_codes,
            code_chunk,
        ):

            ce = min(
                cs + code_chunk,
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

            rank += (
                scores
                > gt_scores[:, None]
            ).sum(dim=1)

            del c, scores

        ranks[qs:qe] = (
            rank.cpu().numpy()
        )

        del (
            q,
            gt_emb,
            gt_scores,
            rank,
        )

        torch.cuda.empty_cache()


    np.save(
        rank_output,
        ranks,
    )

    rr = (
        1.0
        / (
            ranks.astype(
                np.float64
            )
            + 1.0
        )
    )

    metrics = {
        "n_queries":
            int(n_queries),

        "n_codes":
            int(n_codes),

        "r1":
            100.0 * float(
                np.mean(ranks < 1)
            ),

        "r5":
            100.0 * float(
                np.mean(ranks < 5)
            ),

        "r10":
            100.0 * float(
                np.mean(ranks < 10)
            ),

        "r100":
            100.0 * float(
                np.mean(ranks < 100)
            ),

        "mrr":
            100.0 * float(
                rr.mean()
            ),

        "_sum_rr":
            float(rr.sum()),
    }

    return metrics


def build_url_map(
    codebase_path,
):

    mapping = {}

    with Path(codebase_path).open(
        "r",
        encoding="utf-8",
    ) as f:

        for idx, line in enumerate(f):

            if not line.strip():
                continue

            row = json.loads(line)

            if "url" not in row:
                raise RuntimeError(
                    "Codebase row missing URL"
                )

            # Same semantics as existing
            # eval_nomic_embed.py dictionary.
            mapping[row["url"]] = idx

    return mapping


def main():

    # Load model once; reused for all six
    # languages while creating Jina caches.
    encoder = JinaEncoder()

    results = {
        "protocol":
            "reviewer_r1_1_"
            "retriever_strength_valid_v1",

        "split":
            "valid",

        "test_accessed":
            False,

        "max_length":
            512,

        "ranking_rule":
            "count_scores_strictly_"
            "greater_than_ground_truth",

        "retrievers": {},
    }

    all_ranks = {
        "nomic": [],
        "unixcoder": [],
        "jina": [],
    }

    for lang in LANGUAGES:

        print()
        print("=" * 78)
        print(
            f"LANGUAGE = {lang}"
        )
        print("=" * 78)

        valid_path = (
            DATA_ROOT /
            lang /
            "valid.jsonl"
        )

        codebase_path = (
            DATA_ROOT /
            lang /
            "codebase.jsonl"
        )

        valid_rows = read_jsonl(
            valid_path
        )

        npth = nomic_paths(lang)
        upth = unixcoder_paths(lang)
        jpth = jina_paths(lang)

        # Existing Nomic full-codebase
        # cache defines expected code count.
        nomic_code = np.load(
            npth["code"],
            mmap_mode="r",
        )

        n_codes = int(
            nomic_code.shape[0]
        )

        del nomic_code

        # Validate existing UniXcoder
        # cache against same codebase size.
        if load_shape(
            upth["code"]
        )[0] != n_codes:
            raise RuntimeError(
                f"{lang}: UniXcoder "
                "code size mismatch"
            )

        # Build Jina caches only if absent.
        encode_jina_queries(
            encoder,
            lang,
            valid_rows,
            jpth["query"],
        )

        encode_jina_codebase(
            encoder,
            lang,
            codebase_path,
            jpth["code"],
            n_codes,
        )

        # Ground truth follows exact existing
        # CodeSearchNet URL mapping.
        url_to_code = build_url_map(
            codebase_path
        )

        gt = np.asarray(
            [
                url_to_code[
                    row["url"]
                ]
                for row in valid_rows
            ],
            dtype=np.int64,
        )

        source_paths = {
            "nomic": npth,
            "unixcoder": upth,
            "jina": jpth,
        }

        for name, paths in (
            source_paths.items()
        ):

            print()
            print(
                f"--- {lang} / "
                f"{name} ---"
            )

            rank_file = (
                RANK_ROOT /
                f"{lang}_{name}_valid.ranks.npy"
            )

            metrics = evaluate_cache(
                paths["query"],
                paths["code"],
                gt,
                rank_file,
            )

            results[
                "retrievers"
            ].setdefault(
                name,
                {}
            )[lang] = metrics

            ranks = np.load(
                rank_file
            )

            all_ranks[
                name
            ].append(ranks)

            print(
                json.dumps(
                    metrics,
                    indent=2,
                )
            )


    # Model no longer required.
    del encoder
    torch.cuda.empty_cache()

    # Equal-language macro and
    # query-weighted micro.
    for name in [
        "nomic",
        "unixcoder",
        "jina",
    ]:

        rows = (
            results["retrievers"][
                name
            ]
        )

        macro = {}

        for metric in [
            "r1",
            "r5",
            "r10",
            "r100",
            "mrr",
        ]:
            macro[metric] = float(
                np.mean(
                    [
                        rows[l][metric]
                        for l in LANGUAGES
                    ]
                )
            )

        pooled_ranks = (
            np.concatenate(
                all_ranks[name]
            )
        )

        micro = {
            "n_queries":
                int(
                    len(pooled_ranks)
                ),

            "r1":
                100.0 * float(
                    np.mean(
                        pooled_ranks < 1
                    )
                ),

            "r5":
                100.0 * float(
                    np.mean(
                        pooled_ranks < 5
                    )
                ),

            "r10":
                100.0 * float(
                    np.mean(
                        pooled_ranks < 10
                    )
                ),

            "r100":
                100.0 * float(
                    np.mean(
                        pooled_ranks < 100
                    )
                ),

            "mrr":
                100.0 * float(
                    np.mean(
                        1.0 /
                        (
                            pooled_ranks
                            .astype(
                                np.float64
                            )
                            + 1.0
                        )
                    )
                ),
        }

        results[
            "retrievers"
        ][name]["macro"] = macro

        results[
            "retrievers"
        ][name]["micro"] = micro


    n_mrr = (
        results["retrievers"]
        ["nomic"]["macro"]["mrr"]
    )

    j_mrr = (
        results["retrievers"]
        ["jina"]["macro"]["mrr"]
    )

    u_mrr = (
        results["retrievers"]
        ["unixcoder"]["macro"]["mrr"]
    )

    jina_gap = (
        j_mrr - n_mrr
    )

    unix_gap = (
        u_mrr - n_mrr
    )

    if abs(jina_gap) <= 3.0:
        jina_class = (
            "performance_comparable"
        )

    elif abs(jina_gap) <= 5.0:
        jina_class = (
            "independent_dense_"
            "but_not_matched_strength"
        )

    else:
        jina_class = (
            "not_matched_strength"
        )

    results["selection"] = {
        "primary_metric":
            "equal_language_macro_mrr",

        "nomic_macro_mrr":
            n_mrr,

        "jina_macro_mrr":
            j_mrr,

        "unixcoder_macro_mrr":
            u_mrr,

        "jina_minus_nomic_pp":
            jina_gap,

        "unixcoder_minus_nomic_pp":
            unix_gap,

        "jina_classification":
            jina_class,

        "frozen_rule":
            {
                "abs_gap_le_3":
                    "performance_comparable",

                "abs_gap_3_to_5":
                    "independent_dense_"
                    "not_matched_strength",

                "abs_gap_gt_5":
                    "not_matched_strength",
            },
    }


    json_path = (
        OUT_ROOT /
        "retriever_strength_valid.json"
    )

    with json_path.open(
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


    csv_path = (
        OUT_ROOT /
        "retriever_strength_valid.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "retriever",
                "language",
                "n_queries",
                "R@1",
                "R@5",
                "R@10",
                "R@100",
                "MRR",
            ]
        )

        for name in [
            "nomic",
            "unixcoder",
            "jina",
        ]:

            for lang in LANGUAGES:

                r = (
                    results[
                        "retrievers"
                    ][name][lang]
                )

                writer.writerow(
                    [
                        name,
                        lang,
                        r["n_queries"],
                        r["r1"],
                        r["r5"],
                        r["r10"],
                        r["r100"],
                        r["mrr"],
                    ]
                )


    print()
    print("=" * 78)
    print(
        "VALIDATION RETRIEVER "
        "STRENGTH SUMMARY"
    )
    print("=" * 78)

    for name in [
        "nomic",
        "unixcoder",
        "jina",
    ]:

        macro = (
            results["retrievers"]
            [name]["macro"]
        )

        print(
            f"{name:<12} "
            f"R1={macro['r1']:.4f} "
            f"R5={macro['r5']:.4f} "
            f"R10={macro['r10']:.4f} "
            f"R100={macro['r100']:.4f} "
            f"MRR={macro['mrr']:.4f}"
        )


    print()
    print(
        "Jina - Nomic macro MRR = "
        f"{jina_gap:+.4f} pp"
    )

    print(
        "UniXcoder - Nomic "
        "macro MRR = "
        f"{unix_gap:+.4f} pp"
    )

    print(
        "JINA_CLASSIFICATION =",
        jina_class,
    )

    print()
    print(
        "VALID_RETRIEVER_"
        "BENCHMARK=PASSED"
    )


if __name__ == "__main__":
    main()
