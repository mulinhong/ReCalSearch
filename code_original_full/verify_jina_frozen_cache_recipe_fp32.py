import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


LANGS = [
    "ruby",
    "javascript",
    "php",
    "python",
    "java",
    "go",
]

ROOT = Path(
    "/root/autodl-tmp/lora-code-embeddings-main/text2code"
)

DATA_ROOT = Path(
    "/root/autodl-tmp/lora-code-embeddings-main/data/CSN"
)

MODEL_PATH = (
    "/root/autodl-tmp/hf_models/"
    "jina-code-embeddings-0.5b"
)

CACHE_ROOT = (
    ROOT
    / "results/reviewer_revision"
    / "hn_source_ablation"
    / "retriever_strength_valid"
    / "jina_cache"
)

MAX_LENGTH = 512
DIM = 896

QUERY_PREFIX = (
    "Find the most relevant code snippet "
    "given the following query:\n"
)

CODE_PREFIX = "Candidate code snippet:\n"

# Small deterministic parity cohort.
N_CHECK = 16
BATCH_SIZE = 16


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


# IMPORTANT:
# Same tokenized-field retrieval text used by the Nomic
# paper protocol: use_raw=False.
def get_query_text(row):
    value = row.get("docstring_tokens")

    if value:
        if isinstance(value, list):
            return " ".join(map(str, value))
        return str(value)

    raise KeyError(
        f"No docstring_tokens: {list(row.keys())}"
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
        f"No code text: {list(row.keys())}"
    )


def last_token_pool(last_hidden, attention_mask):
    left_padding = (
        attention_mask[:, -1].sum()
        == attention_mask.shape[0]
    )

    if left_padding:
        return last_hidden[:, -1]

    lengths = attention_mask.sum(dim=1) - 1

    return last_hidden[
        torch.arange(
            last_hidden.shape[0],
            device=last_hidden.device,
        ),
        lengths,
    ]


@torch.inference_mode()
def encode(model, tokenizer, texts, prefix):
    result = []

    for start in range(
        0,
        len(texts),
        BATCH_SIZE,
    ):
        part = texts[
            start:start + BATCH_SIZE
        ]

        part = [
            prefix + x
            for x in part
        ]

        batch = tokenizer(
            part,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        batch = {
            k: v.cuda()
            for k, v in batch.items()
        }

        out = model(
            **batch,
            use_cache=False,
        )

        emb = last_token_pool(
            out.last_hidden_state,
            batch["attention_mask"],
        )

        emb = F.normalize(
            emb.float(),
            p=2,
            dim=1,
        )

        result.append(
            emb.cpu().numpy()
        )

    return np.concatenate(
        result,
        axis=0,
    )


def compare(name, fresh, cached):
    cached = np.asarray(
        cached,
        dtype=np.float32,
    )

    fresh = np.asarray(
        fresh,
        dtype=np.float32,
    )

    fresh = fresh / np.linalg.norm(
        fresh,
        axis=1,
        keepdims=True,
    )

    cached = cached / np.linalg.norm(
        cached,
        axis=1,
        keepdims=True,
    )

    cos = np.sum(
        fresh * cached,
        axis=1,
    )

    abs_diff = np.abs(
        fresh - cached
    )

    result = {
        "mean_cosine":
            float(cos.mean()),

        "min_cosine":
            float(cos.min()),

        "max_abs_diff":
            float(abs_diff.max()),

        "pass":
            bool(
                cos.mean() >= 0.99995
                and cos.min() >= 0.99990
            ),
    }

    print(
        f"{name:<24} "
        f"mean_cos={result['mean_cosine']:.8f} "
        f"min_cos={result['min_cosine']:.8f} "
        f"max_abs={result['max_abs_diff']:.8f} "
        f"PASS={result['pass']}"
    )

    return result


def main():

    assert torch.cuda.is_available()

    print("Loading Jina model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    # Freeze padding behavior explicitly.
    tokenizer.padding_side = "left"

    model = AutoModel.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        torch_dtype=torch.float32,
    ).cuda().eval()

    model.config.use_cache = False

    print(
        "MODEL_DTYPE=",
        next(model.parameters()).dtype,
    )

    print(
        "TOKENIZER_PADDING_SIDE=",
        tokenizer.padding_side,
    )

    report = {
        "protocol":
            "jina_frozen_cache_recipe_parity_fp32_v1",

        "model":
            MODEL_PATH,

        "max_length":
            MAX_LENGTH,

        "dimension":
            DIM,

        "query_prefix":
            QUERY_PREFIX,

        "code_prefix":
            CODE_PREFIX,

        "pooling":
            "last_token",

        "normalization":
            "l2",

        "padding_side":
            tokenizer.padding_side,

        "languages": {},
    }

    all_pass = True

    for lang in LANGS:

        valid_rows = read_jsonl(
            DATA_ROOT / lang / "valid.jsonl"
        )

        code_rows = read_jsonl(
            DATA_ROOT / lang / "codebase.jsonl"
        )

        query_texts = [
            get_query_text(x)
            for x in valid_rows[:N_CHECK]
        ]

        code_texts = [
            get_code_text(x)
            for x in code_rows[:N_CHECK]
        ]

        query_cache_path = (
            CACHE_ROOT /
            (
                f"{lang}_valid_"
                "jina_code_0p5b_"
                "nl2code_msl512.query.npy"
            )
        )

        code_cache_path = (
            CACHE_ROOT /
            (
                f"{lang}_"
                "jina_code_0p5b_"
                "nl2code_msl512.code.npy"
            )
        )

        if not query_cache_path.is_file():
            raise FileNotFoundError(
                query_cache_path
            )

        if not code_cache_path.is_file():
            raise FileNotFoundError(
                code_cache_path
            )

        q_cache = np.load(
            query_cache_path,
            mmap_mode="r",
        )

        c_cache = np.load(
            code_cache_path,
            mmap_mode="r",
        )

        assert q_cache.shape[1] == DIM
        assert c_cache.shape[1] == DIM

        fresh_q = encode(
            model,
            tokenizer,
            query_texts,
            QUERY_PREFIX,
        )

        fresh_c = encode(
            model,
            tokenizer,
            code_texts,
            CODE_PREFIX,
        )

        q_result = compare(
            f"{lang} QUERY",
            fresh_q,
            q_cache[:N_CHECK],
        )

        c_result = compare(
            f"{lang} CODE",
            fresh_c,
            c_cache[:N_CHECK],
        )

        lang_pass = (
            q_result["pass"]
            and c_result["pass"]
        )

        report["languages"][lang] = {
            "query": q_result,
            "code": c_result,
            "pass": lang_pass,
        }

        all_pass = (
            all_pass
            and lang_pass
        )

    report["all_pass"] = all_pass

    output = (
        ROOT
        / "results/reviewer_revision"
        / "hn_source_ablation"
        / "final_protocol"
        / "jina_cache_recipe_parity_fp32.json"
    )

    with output.open(
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

    print()
    print("REPORT=", output)
    print(
        "JINA_FROZEN_CACHE_RECIPE_PASS=",
        all_pass,
    )

    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
