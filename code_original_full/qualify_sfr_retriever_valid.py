#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel

from data_utils import csn_search_eval_text, csn_search_eval_code
def text2code_to_array(text2code, n_queries: int) -> np.ndarray:
    if isinstance(text2code, torch.Tensor):
        return text2code.long().cpu().numpy()
    if isinstance(text2code, dict):
        return np.asarray(
            [text2code[i] for i in range(n_queries)],
            dtype=np.int64,
        )
    return np.asarray(list(text2code), dtype=np.int64)



LANGS = [
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

MODEL_PATH = (
    "/root/autodl-tmp/hf_models/"
    "SFR-Embedding-Code-2B_R"
)

ROOT = Path(
    "results/reviewer_revision/"
    "hn_source_ablation/"
    "retriever_strength_valid/"
    "sfr_code_2b"
)

CACHE_ROOT = ROOT / "cache"
RANK_ROOT = ROOT / "ranks"

CACHE_ROOT.mkdir(parents=True, exist_ok=True)
RANK_ROOT.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 512

# Frozen Nomic paper-protocol validation values.
NOMIC = {
    "ruby": 71.76400266178253,
    "javascript": 54.306595934433055,
    "php": 42.49038024243681,
    "python": 66.21507884807897,
    "java": 56.12726292007676,
    "go": 79.2908826388427,
}


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
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
                return " ".join(map(str, value))
            return str(value)

    if row.get("docstring_tokens"):
        value = row["docstring_tokens"]
        if isinstance(value, list):
            return " ".join(map(str, value))
        return str(value)

    raise KeyError("Cannot resolve query text")


def get_code_text(row):
    if row.get("code"):
        return str(row["code"])

    if row.get("code_tokens"):
        value = row["code_tokens"]
        if isinstance(value, list):
            return " ".join(map(str, value))
        return str(value)

    raise KeyError("Cannot resolve code text")


def cache_paths(lang):
    return {
        "query":
            CACHE_ROOT /
            f"{lang}_valid_sfr_code_2b_msl512.query.npy",

        "code":
            CACHE_ROOT /
            f"{lang}_sfr_code_2b_msl512.code.npy",
    }


class SFREncoder:

    def __init__(self):
        self.device = torch.device("cuda")

        print("Loading SFR model...")

        self.model = AutoModel.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()

        print(
            "SFR_MODEL_READY",
            "device=", next(self.model.parameters()).device,
            "dtype=", next(self.model.parameters()).dtype,
        )


    @torch.no_grad()
    def encode_queries(self, texts):
        emb = self.model.encode_queries(
            texts,
            instruction=(
                "Given Code or Text, "
                "retrieval relevant content"
            ),
            max_length=MAX_LENGTH,
        )

        emb = F.normalize(
            emb.float(),
            p=2,
            dim=1,
        )

        return (
            emb.cpu()
            .numpy()
            .astype(np.float16)
        )


    @torch.no_grad()
    def encode_corpus(self, texts):
        emb = self.model.encode_corpus(
            texts,
            max_length=MAX_LENGTH,
        )

        emb = F.normalize(
            emb.float(),
            p=2,
            dim=1,
        )

        return (
            emb.cpu()
            .numpy()
            .astype(np.float16)
        )


def encode_queries(
    encoder,
    rows,
    output_path,
    lang,
    batch_size=32,
):
    output_path = Path(output_path)

    if output_path.exists():
        arr = np.load(
            output_path,
            mmap_mode="r",
        )

        if arr.shape == (
            len(rows),
            2304,
        ):
            print(
                "REUSE_QUERY_CACHE",
                output_path,
                arr.shape,
            )
            return

        raise RuntimeError(
            f"Bad existing query cache "
            f"{output_path}: {arr.shape}"
        )

    texts = [
        get_query_text(row)
        for row in rows
    ]

    out = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(texts), 2304),
    )

    offset = 0

    for start in tqdm(
        range(0, len(texts), batch_size),
        desc=f"{lang} SFR valid query",
    ):
        batch = texts[
            start:start + batch_size
        ]

        emb = encoder.encode_queries(
            batch
        )

        out[
            offset:offset + len(emb)
        ] = emb

        offset += len(emb)

    if offset != len(texts):
        raise RuntimeError(
            f"{lang}: query rows "
            f"{offset} != {len(texts)}"
        )

    out.flush()
    del out

    print(
        "SAVED_QUERY_CACHE",
        output_path,
    )


def encode_codebase(
    encoder,
    code_path,
    output_path,
    n_codes,
    lang,
    batch_size=16,
):
    output_path = Path(output_path)

    if output_path.exists():
        arr = np.load(
            output_path,
            mmap_mode="r",
        )

        if arr.shape == (
            n_codes,
            2304,
        ):
            print(
                "REUSE_CODE_CACHE",
                output_path,
                arr.shape,
            )
            return

        raise RuntimeError(
            f"Bad existing code cache "
            f"{output_path}: {arr.shape}"
        )

    out = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float16,
        shape=(n_codes, 2304),
    )

    buffer = []
    offset = 0

    def flush():
        nonlocal buffer, offset

        if not buffer:
            return

        emb = encoder.encode_corpus(
            buffer
        )

        out[
            offset:offset + len(emb)
        ] = emb

        offset += len(emb)
        buffer = []

    with Path(code_path).open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in tqdm(
            f,
            total=n_codes,
            desc=f"{lang} SFR code",
        ):
            if not line.strip():
                continue

            row = json.loads(line)

            buffer.append(
                get_code_text(row)
            )

            if len(buffer) >= batch_size:
                flush()

    flush()

    if offset != n_codes:
        raise RuntimeError(
            f"{lang}: code rows "
            f"{offset} != {n_codes}"
        )

    out.flush()
    del out

    print(
        "SAVED_CODE_CACHE",
        output_path,
    )


@torch.no_grad()
def compute_ranks(
    query_path,
    code_path,
    gt,
    output_path,
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

    n_queries = q_np.shape[0]
    n_codes = c_np.shape[0]

    if n_queries != len(gt):
        raise RuntimeError(
            "GT/query length mismatch"
        )

    ranks = np.empty(
        n_queries,
        dtype=np.int32,
    )

    device = torch.device("cuda")

    for qs in tqdm(
        range(
            0,
            n_queries,
            query_chunk,
        ),
        desc="SFR full-codebase ranks",
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

        gt_chunk = gt[qs:qe]

        gold = torch.from_numpy(
            np.asarray(
                c_np[gt_chunk],
                dtype=np.float32,
            )
        ).to(device)

        gold = F.normalize(
            gold,
            dim=1,
        )

        gold_score = (
            q * gold
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
                scores >
                gold_score[:, None]
            ).sum(dim=1)

            del c, scores

        ranks[qs:qe] = (
            rank.cpu().numpy()
        )

        del (
            q,
            gold,
            gold_score,
            rank,
        )

        torch.cuda.empty_cache()

    np.save(
        output_path,
        ranks,
    )

    return ranks


def metrics(ranks):
    ranks = np.asarray(
        ranks,
        dtype=np.int64,
    )

    return {
        "n_queries":
            int(len(ranks)),

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
                np.mean(
                    1.0 /
                    (
                        ranks.astype(
                            np.float64
                        ) + 1.0
                    )
                )
            ),
    }


def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required"
        )

    encoder = SFREncoder()

    result = {
        "protocol":
            "reviewer_r1_1_"
            "sfr_validation_qualification_v1",

        "split":
            "valid",

        "test_accessed":
            False,

        "data_loading":
            "csn_search_eval_text(valid)+"
            "csn_search_eval_code(codebase.jsonl)",

        "ground_truth":
            "csn_search_eval_text(valid).text2code;"
            "equivalent to create_dataset validation branch",

        "max_length":
            MAX_LENGTH,

        "results": {},
    }

    pooled = []

    for lang in LANGS:

        print()
        print("=" * 78)
        print("LANGUAGE =", lang)
        print("=" * 78)

        valid_path = (
            DATA_ROOT /
            lang /
            "valid.jsonl"
        )

        code_path = (
            DATA_ROOT /
            lang /
            "codebase.jsonl"
        )

        valid_rows = read_jsonl(
            valid_path
        )

        # Validation-only paper protocol.
        # This is exactly the validation/codebase branch used by
        # create_dataset(), without instantiating the test split.
        val_dataset = csn_search_eval_text(
            str(DATA_ROOT),
            lang,
            "valid",
        )

        code_dataset = csn_search_eval_code(
            str(DATA_ROOT),
            lang,
            "codebase.jsonl",
        )

        gt = np.asarray(
            text2code_to_array(
                val_dataset.text2code,
                len(val_dataset),
            ),
            dtype=np.int64,
        )

        n_codes = len(code_dataset)

        if len(valid_rows) != len(gt):
            raise RuntimeError(
                f"{lang}: valid/GT mismatch"
            )

        p = cache_paths(lang)

        encode_queries(
            encoder,
            valid_rows,
            p["query"],
            lang,
        )

        encode_codebase(
            encoder,
            code_path,
            p["code"],
            n_codes,
            lang,
        )

        rank_path = (
            RANK_ROOT /
            f"{lang}_sfr_valid.ranks.npy"
        )

        ranks = compute_ranks(
            p["query"],
            p["code"],
            gt,
            rank_path,
        )

        m = metrics(ranks)

        result["results"][lang] = m
        pooled.append(ranks)

        print(
            json.dumps(
                m,
                indent=2,
            )
        )

    macro_mrr = float(
        np.mean([
            result["results"][l]["mrr"]
            for l in LANGS
        ])
    )

    macro_r1 = float(
        np.mean([
            result["results"][l]["r1"]
            for l in LANGS
        ])
    )

    macro_r5 = float(
        np.mean([
            result["results"][l]["r5"]
            for l in LANGS
        ])
    )

    macro_r10 = float(
        np.mean([
            result["results"][l]["r10"]
            for l in LANGS
        ])
    )

    macro_r100 = float(
        np.mean([
            result["results"][l]["r100"]
            for l in LANGS
        ])
    )

    nomic_macro = float(
        np.mean(
            list(NOMIC.values())
        )
    )

    gap = (
        macro_mrr -
        nomic_macro
    )

    if abs(gap) <= 3.0:
        classification = (
            "performance_comparable"
        )
    elif abs(gap) <= 5.0:
        classification = (
            "independent_dense_"
            "but_not_matched_strength"
        )
    else:
        classification = (
            "not_matched_strength"
        )

    result["macro"] = {
        "r1": macro_r1,
        "r5": macro_r5,
        "r10": macro_r10,
        "r100": macro_r100,
        "mrr": macro_mrr,
    }

    pooled = np.concatenate(
        pooled
    )

    result["micro"] = metrics(
        pooled
    )

    result["selection"] = {
        "nomic_macro_mrr":
            nomic_macro,

        "sfr_macro_mrr":
            macro_mrr,

        "sfr_minus_nomic_pp":
            gap,

        "classification":
            classification,
    }

    output = (
        ROOT /
        "sfr_validation_qualification.json"
    )

    with output.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")

    print()
    print("=" * 78)
    print("SFR VALIDATION QUALIFICATION")
    print("=" * 78)

    print(
        f"Nomic macro MRR = "
        f"{nomic_macro:.4f}"
    )

    print(
        f"SFR macro MRR   = "
        f"{macro_mrr:.4f}"
    )

    print(
        f"SFR - Nomic     = "
        f"{gap:+.4f} pp"
    )

    print(
        "CLASSIFICATION =",
        classification,
    )

    print(
        "SFR_VALIDATION_"
        "QUALIFICATION_PASS=True"
    )


if __name__ == "__main__":
    main()
