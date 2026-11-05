#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from data_utils import csn_search_eval_text


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
    "Qwen3-Embedding-4B"
)

ROOT = Path(
    "results/reviewer_revision/"
    "hn_source_ablation/"
    "retriever_strength_valid/"
    "qwen3_embedding_4b"
)

CACHE_ROOT = ROOT / "cache"
RANK_ROOT = ROOT / "ranks"

CACHE_ROOT.mkdir(parents=True, exist_ok=True)
RANK_ROOT.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 512
DIM = 2560

QUERY_BATCH_SIZE = 16
CODE_BATCH_SIZE = 16

TASK = (
    "Given a natural language query, retrieve the relevant "
    "code snippet that implements the described functionality"
)

# Frozen paper-protocol validation Nomic MRR (%).
NOMIC = {
    "ruby": 71.7640026618,
    "javascript": 54.3065959344,
    "php": 42.4903802424,
    "python": 66.2150788481,
    "java": 56.1272629201,
    "go": 79.2908826388,
}

NOMIC_MACRO = float(np.mean(list(NOMIC.values())))
MATCH_THRESHOLD_PP = 3.0
LOWER_BOUND = NOMIC_MACRO - MATCH_THRESHOLD_PP


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


# Must match Nomic retrieval textual inputs.
def get_query_text(row):
    value = row.get("docstring_tokens")
    if value:
        if isinstance(value, list):
            return " ".join(map(str, value))
        return str(value)
    raise KeyError(
        f"No docstring_tokens in keys={list(row.keys())}"
    )


def get_code_text(row):
    for key in ["code_tokens", "function_tokens"]:
        value = row.get(key)
        if value:
            if isinstance(value, list):
                return " ".join(map(str, value))
            return str(value)

    value = row.get("code")
    if value:
        return str(value)

    raise KeyError(
        f"No code text in keys={list(row.keys())}"
    )


def instruct(query):
    return f"Instruct: {TASK}\nQuery:{query}"


def last_token_pool(
    last_hidden_states: Tensor,
    attention_mask: Tensor,
):
    left_padding = (
        attention_mask[:, -1].sum()
        == attention_mask.shape[0]
    )

    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = (
        attention_mask.sum(dim=1) - 1
    )

    batch_size = last_hidden_states.shape[0]

    return last_hidden_states[
        torch.arange(
            batch_size,
            device=last_hidden_states.device,
        ),
        sequence_lengths,
    ]


def text2code_to_array(text2code, n_queries):
    if isinstance(text2code, torch.Tensor):
        return text2code.long().cpu().numpy()

    if isinstance(text2code, dict):
        return np.asarray(
            [text2code[i] for i in range(n_queries)],
            dtype=np.int64,
        )

    return np.asarray(
        list(text2code),
        dtype=np.int64,
    )


class Qwen3Encoder:

    def __init__(self):
        print("Loading Qwen3-Embedding-4B...")

        self.device = torch.device("cuda")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            padding_side="left",
        )

        self.model = AutoModel.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        ).to(self.device).eval()

        self.model.config.use_cache = False

        print(
            "QWEN3_MODEL_READY",
            "dtype=",
            next(self.model.parameters()).dtype,
            "device=",
            next(self.model.parameters()).device,
        )


    @torch.inference_mode()
    def encode(self, texts, is_query):
        if is_query:
            texts = [
                instruct(text)
                for text in texts
            ]

        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        batch = {
            k: v.to(self.device)
            for k, v in batch.items()
        }

        output = self.model(
            **batch,
            use_cache=False,
        )

        emb = last_token_pool(
            output.last_hidden_state,
            batch["attention_mask"],
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


def cache_paths(lang):
    return {
        "query":
            CACHE_ROOT /
            f"{lang}_valid_qwen3embed4b_msl512.query.npy",

        "code":
            CACHE_ROOT /
            f"{lang}_qwen3embed4b_msl512.code.npy",
    }


def encode_query_cache(
    encoder,
    rows,
    path,
    lang,
):
    if path.exists():
        arr = np.load(path, mmap_mode="r")

        if arr.shape == (len(rows), DIM):
            print(
                "REUSE_QUERY_CACHE",
                path,
                arr.shape,
            )
            return

        raise RuntimeError(
            f"Bad query cache {path}: {arr.shape}"
        )

    texts = [
        get_query_text(row)
        for row in rows
    ]

    out = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float16,
        shape=(len(texts), DIM),
    )

    offset = 0

    for start in tqdm(
        range(
            0,
            len(texts),
            QUERY_BATCH_SIZE,
        ),
        desc=f"{lang} Qwen3 valid query",
    ):
        batch = texts[
            start:start + QUERY_BATCH_SIZE
        ]

        emb = encoder.encode(
            batch,
            is_query=True,
        )

        out[
            offset:offset + len(emb)
        ] = emb

        offset += len(emb)

    if offset != len(texts):
        raise RuntimeError(
            f"{lang}: incomplete query cache"
        )

    out.flush()
    del out

    print(
        "SAVED_QUERY_CACHE",
        path,
    )


def encode_code_cache(
    encoder,
    rows,
    path,
    lang,
):
    if path.exists():
        arr = np.load(path, mmap_mode="r")

        if arr.shape == (len(rows), DIM):
            print(
                "REUSE_CODE_CACHE",
                path,
                arr.shape,
            )
            return

        raise RuntimeError(
            f"Bad code cache {path}: {arr.shape}"
        )

    out = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float16,
        shape=(len(rows), DIM),
    )

    offset = 0

    for start in tqdm(
        range(
            0,
            len(rows),
            CODE_BATCH_SIZE,
        ),
        desc=f"{lang} Qwen3 code",
    ):
        batch_rows = rows[
            start:start + CODE_BATCH_SIZE
        ]

        texts = [
            get_code_text(row)
            for row in batch_rows
        ]

        emb = encoder.encode(
            texts,
            is_query=False,
        )

        out[
            offset:offset + len(emb)
        ] = emb

        offset += len(emb)

    if offset != len(rows):
        raise RuntimeError(
            f"{lang}: incomplete code cache"
        )

    out.flush()
    del out

    print(
        "SAVED_CODE_CACHE",
        path,
    )


@torch.inference_mode()
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

    if len(q_np) != len(gt):
        raise RuntimeError(
            "query/GT length mismatch"
        )

    ranks = np.empty(
        len(gt),
        dtype=np.int32,
    )

    device = torch.device("cuda")

    for qs in tqdm(
        range(0, len(gt), query_chunk),
        desc="Qwen3 full-codebase ranks",
    ):
        qe = min(
            qs + query_chunk,
            len(gt),
        )

        q = torch.from_numpy(
            np.asarray(
                q_np[qs:qe],
                dtype=np.float32,
            )
        ).to(device)

        q = F.normalize(q, dim=1)

        gt_chunk = gt[qs:qe]

        gold = torch.from_numpy(
            np.asarray(
                c_np[gt_chunk],
                dtype=np.float32,
            )
        ).to(device)

        gold = F.normalize(gold, dim=1)

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
            len(c_np),
            code_chunk,
        ):
            ce = min(
                cs + code_chunk,
                len(c_np),
            )

            c = torch.from_numpy(
                np.asarray(
                    c_np[cs:ce],
                    dtype=np.float32,
                )
            ).to(device)

            c = F.normalize(c, dim=1)

            score = q @ c.T

            rank += (
                score >
                gold_score[:, None]
            ).sum(dim=1)

            del c, score

        ranks[qs:qe] = (
            rank.cpu().numpy()
        )

    np.save(output_path, ranks)

    return ranks


def metrics(ranks):
    ranks = np.asarray(
        ranks,
        dtype=np.int64,
    )

    return {
        "n_queries": int(len(ranks)),

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
        raise RuntimeError("CUDA required")

    encoder = Qwen3Encoder()

    result = {
        "protocol":
            "reviewer_r1_1_"
            "qwen3_embedding_4b_"
            "validation_qualification_v1",

        "split": "valid",

        "test_accessed": False,

        "model":
            "Qwen/Qwen3-Embedding-4B",

        "max_length":
            MAX_LENGTH,

        "pooling":
            "last_token",

        "normalization":
            "l2",

        "query_instruction":
            TASK,

        "input_protocol":
            "same raw token-joined CSN retrieval "
            "text as Nomic; model-specific "
            "query instruction only",

        "ground_truth":
            "csn_search_eval_text(valid)."
            "text2code",

        "selection_rule": {
            "primary":
                "equal_language_macro_mrr",

            "matched_strength_pp":
                MATCH_THRESHOLD_PP,

            "nomic_macro_mrr":
                NOMIC_MACRO,
        },

        "results": {},
    }

    completed_mrr = []

    for lang_idx, lang in enumerate(LANGS):

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

        code_rows = read_jsonl(
            code_path
        )

        # Validation-only paper GT.
        # This constructor opens only valid.jsonl
        # and codebase.jsonl.
        val_dataset = csn_search_eval_text(
            str(DATA_ROOT),
            lang,
            "valid",
        )

        gt = text2code_to_array(
            val_dataset.text2code,
            len(val_dataset),
        )

        if len(valid_rows) != len(gt):
            raise RuntimeError(
                f"{lang}: valid/GT mismatch"
            )

        if len(code_rows) <= int(gt.max()):
            raise RuntimeError(
                f"{lang}: GT outside codebase"
            )

        p = cache_paths(lang)

        encode_query_cache(
            encoder,
            valid_rows,
            p["query"],
            lang,
        )

        encode_code_cache(
            encoder,
            code_rows,
            p["code"],
            lang,
        )

        rank_path = (
            RANK_ROOT /
            f"{lang}_qwen3embed4b_valid.ranks.npy"
        )

        ranks = compute_ranks(
            p["query"],
            p["code"],
            gt,
            rank_path,
        )

        m = metrics(ranks)

        result["results"][lang] = m

        completed_mrr.append(
            m["mrr"]
        )

        print(
            json.dumps(
                m,
                indent=2,
            )
        )

        print(
            f"NOMIC_{lang}_MRR="
            f"{NOMIC[lang]:.10f}"
        )

        print(
            f"QWEN3_MINUS_NOMIC_{lang}_PP="
            f"{m['mrr'] - NOMIC[lang]:+.10f}"
        )

        # Precommitted mathematical futility check.
        remaining = (
            len(LANGS)
            - len(completed_mrr)
        )

        max_possible_macro = (
            sum(completed_mrr)
            + 100.0 * remaining
        ) / len(LANGS)

        print(
            "MAX_POSSIBLE_FINAL_MACRO=",
            f"{max_possible_macro:.10f}",
        )

        if (
            max_possible_macro
            < LOWER_BOUND
        ):
            result["stopped_for_futility"] = True
            result["max_possible_final_macro"] = (
                max_possible_macro
            )

            output = (
                ROOT /
                "qwen3_validation_"
                "qualification.json"
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
            print(
                "QWEN3_QUALIFICATION_"
                "FUTILITY_STOP=True"
            )

            print(
                "Reason: even MRR=100 for every "
                "remaining language cannot reach "
                f"{LOWER_BOUND:.4f}"
            )

            return

    macro = float(
        np.mean([
            result["results"][lang]["mrr"]
            for lang in LANGS
        ])
    )

    gap = macro - NOMIC_MACRO

    if abs(gap) <= MATCH_THRESHOLD_PP:
        classification = (
            "performance_comparable"
        )
    else:
        classification = (
            "not_matched_strength"
        )

    result["macro_mrr"] = macro

    result["qwen3_minus_nomic_pp"] = gap

    result["classification"] = (
        classification
    )

    result["stopped_for_futility"] = False

    output = (
        ROOT /
        "qwen3_validation_"
        "qualification.json"
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
    print("QWEN3 VALIDATION QUALIFICATION")
    print("=" * 78)

    print(
        f"Nomic macro MRR = "
        f"{NOMIC_MACRO:.10f}"
    )

    print(
        f"Qwen3 macro MRR = "
        f"{macro:.10f}"
    )

    print(
        f"Qwen3 - Nomic   = "
        f"{gap:+.10f} pp"
    )

    print(
        "CLASSIFICATION =",
        classification,
    )

    print(
        "QWEN3_VALIDATION_"
        "QUALIFICATION_PASS=True"
    )


if __name__ == "__main__":
    main()
