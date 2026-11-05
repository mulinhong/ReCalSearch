import json
from pathlib import Path

import numpy as np
import torch

import benchmark_dense_retrievers_valid as hist


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

CACHE_ROOT = (
    ROOT
    / "results/reviewer_revision"
    / "hn_source_ablation"
    / "retriever_strength_valid"
    / "jina_cache"
)

OUT = (
    ROOT
    / "results/reviewer_revision"
    / "hn_source_ablation"
    / "final_protocol"
    / "jina_exact_generator_parity.json"
)

# These are the ORIGINAL historical cache batch sizes.
QUERY_BATCH = 64
CODE_BATCH = 32

MEAN_COS_THRESHOLD = 0.99995
MIN_COS_THRESHOLD = 0.99990


def read_first_jsonl(path, n):
    rows = []

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            if not line.strip():
                continue

            rows.append(
                json.loads(line)
            )

            if len(rows) >= n:
                break

    return rows


def compare(name, fresh, cached):

    # Historical generator itself returns float16.
    assert fresh.dtype == np.float16
    assert cached.dtype == np.float16

    exact = np.array_equal(
        fresh,
        cached,
    )

    fresh32 = np.asarray(
        fresh,
        dtype=np.float32,
    )

    cached32 = np.asarray(
        cached,
        dtype=np.float32,
    )

    fresh32 /= np.linalg.norm(
        fresh32,
        axis=1,
        keepdims=True,
    )

    cached32 /= np.linalg.norm(
        cached32,
        axis=1,
        keepdims=True,
    )

    cos = np.sum(
        fresh32 * cached32,
        axis=1,
    )

    diff = np.abs(
        fresh.astype(np.float32)
        - cached.astype(np.float32)
    )

    mean_cos = float(
        cos.mean()
    )

    min_cos = float(
        cos.min()
    )

    max_abs = float(
        diff.max()
    )

    n_exact_rows = int(
        np.all(
            fresh == cached,
            axis=1,
        ).sum()
    )

    passed = bool(
        mean_cos >= MEAN_COS_THRESHOLD
        and min_cos >= MIN_COS_THRESHOLD
    )

    print(
        f"{name:<24} "
        f"exact_array={exact} "
        f"exact_rows={n_exact_rows}/{len(fresh)} "
        f"mean_cos={mean_cos:.8f} "
        f"min_cos={min_cos:.8f} "
        f"max_abs={max_abs:.8f} "
        f"PASS={passed}"
    )

    return {
        "exact_array": bool(exact),
        "exact_rows": n_exact_rows,
        "rows": int(len(fresh)),
        "mean_cosine": mean_cos,
        "min_cosine": min_cos,
        "max_abs_diff": max_abs,
        "pass": passed,
    }


def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required"
        )

    print(
        "PYTORCH_VERSION=",
        torch.__version__,
    )

    import transformers

    print(
        "TRANSFORMERS_VERSION=",
        transformers.__version__,
    )

    print(
        "HIST_SCRIPT=",
        hist.__file__,
    )

    print(
        "QUERY_BATCH=",
        QUERY_BATCH,
    )

    print(
        "CODE_BATCH=",
        CODE_BATCH,
    )

    # CRITICAL:
    # Instantiate the EXACT historical encoder class.
    encoder = hist.JinaEncoder()

    report = {
        "format":
            "jina_exact_historical_generator_parity_v1",

        "generator":
            str(hist.__file__),

        "query_batch":
            QUERY_BATCH,

        "code_batch":
            CODE_BATCH,

        "mean_cos_threshold":
            MEAN_COS_THRESHOLD,

        "min_cos_threshold":
            MIN_COS_THRESHOLD,

        "languages": {},
    }

    all_pass = True

    for lang in LANGS:

        # Exact first historical query batch.
        valid_rows = read_first_jsonl(
            DATA_ROOT
            / lang
            / "valid.jsonl",
            QUERY_BATCH,
        )

        # Exact first historical code batch.
        code_rows = read_first_jsonl(
            DATA_ROOT
            / lang
            / "codebase.jsonl",
            CODE_BATCH,
        )

        # CRITICAL:
        # Use the historical generator's own text
        # extraction functions, not reconstructed ones.
        queries = [
            hist.get_query_text(x)
            for x in valid_rows
        ]

        codes = [
            hist.get_code_text(x)
            for x in code_rows
        ]

        # CRITICAL:
        # Same encode_batch implementation, same prefixes.
        fresh_q = encoder.encode_batch(
            queries,
            hist.JINA_QUERY_PREFIX,
        )

        fresh_c = encoder.encode_batch(
            codes,
            hist.JINA_CODE_PREFIX,
        )

        q_path = (
            CACHE_ROOT
            / (
                f"{lang}_valid_"
                "jina_code_0p5b_"
                "nl2code_msl512.query.npy"
            )
        )

        c_path = (
            CACHE_ROOT
            / (
                f"{lang}_"
                "jina_code_0p5b_"
                "nl2code_msl512.code.npy"
            )
        )

        q_cache = np.load(
            q_path,
            mmap_mode="r",
        )[:QUERY_BATCH]

        c_cache = np.load(
            c_path,
            mmap_mode="r",
        )[:CODE_BATCH]

        q_res = compare(
            f"{lang} QUERY",
            fresh_q,
            q_cache,
        )

        c_res = compare(
            f"{lang} CODE",
            fresh_c,
            c_cache,
        )

        lang_pass = (
            q_res["pass"]
            and c_res["pass"]
        )

        report[
            "languages"
        ][lang] = {
            "query": q_res,
            "code": c_res,
            "pass": lang_pass,
        }

        all_pass &= lang_pass

    report["all_pass"] = bool(
        all_pass
    )

    with OUT.open(
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
    print(
        "REPORT=",
        OUT,
    )

    print(
        "JINA_EXACT_GENERATOR_PARITY_PASS=",
        all_pass,
    )

    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
