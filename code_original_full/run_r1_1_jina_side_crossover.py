from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from eval_llm_verifier import (
    load_verifier,
    single_token_id,
    build_yesno_prompt,
    score_yesno_prompts,
    rerank_with_verifier,
    read_jsonl,
    get_query_text,
    get_code_text,
)

ROOT = Path(
    "/root/autodl-tmp/lora-code-embeddings-main/text2code"
).resolve()

DATA_ROOT = Path(
    "/root/autodl-tmp/lora-code-embeddings-main/data/CSN"
).resolve()

POOL_ROOT = (
    ROOT
    / "results/reviewer_revision/pools/jina_test"
)

OUT_ROOT = (
    ROOT
    / "results/reviewer_revision/hn_source_ablation/"
      "crossover/jina_pool_verifier_only"
)

BASE_MODEL = Path(
    "/root/autodl-tmp/hf_models/"
    "Qwen2.5-Coder-7B-Instruct"
)

LANGS = [
    "ruby",
    "javascript",
    "php",
    "python",
    "java",
    "go",
]

CANDIDATE_TOP_K = 100
VERIFY_TOP_K = 15

VERIFIER_DTYPE = "float16"
DEVICE_MAP = "cuda"

VERIFIER_BATCH_SIZE = 4
MAX_LENGTH = 1024
MAX_QUERY_CHARS = 512
MAX_CODE_CHARS = 3000
MAX_NEW_TOKENS = 64


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(b)
    return h.hexdigest()


def atomic_npy(path: Path, arr: np.ndarray):
    tmp = path.with_name(path.name + ".tmp")

    if tmp.exists():
        tmp.unlink()

    with tmp.open("wb") as f:
        np.save(f, arr)

    os.replace(tmp, path)


def atomic_json(path: Path, obj):
    tmp = path.with_name(path.name + ".tmp")

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    os.replace(tmp, path)


def metric_dict(ranks):
    r = np.asarray(
        ranks,
        dtype=np.int64,
    )

    return {
        "n_queries":
            int(len(r)),
        "mrr":
            float(
                np.mean(
                    1.0 /
                    (
                        r.astype(
                            np.float64
                        ) + 1.0
                    )
                )
            ),
        "r1":
            float(np.mean(r < 1)),
        "r5":
            float(np.mean(r < 5)),
        "r10":
            float(np.mean(r < 10)),
    }


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--source",
        required=True,
        choices=[
            "nomic",
            "jina",
        ],
    )

    ap.add_argument(
        "--seed",
        required=True,
        type=int,
        choices=[
            13,
            42,
            87,
        ],
    )

    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable"
        )

    run_name = (
        f"{args.source}_seed{args.seed}"
    )

    verifier_path = (
        ROOT
        / "trained_verifiers/"
          "reviewer_r1_1_hn_source_ablation"
        / run_name
    )

    if not verifier_path.is_dir():
        raise FileNotFoundError(
            verifier_path
        )

    adapter_cfg = (
        verifier_path
        / "adapter_config.json"
    )

    if not adapter_cfg.is_file():
        raise FileNotFoundError(
            adapter_cfg
        )

    cfg = json.loads(
        adapter_cfg.read_text(
            encoding="utf-8"
        )
    )

    expected_base = str(
        BASE_MODEL
    )

    actual_base = str(
        cfg.get(
            "base_model_name_or_path",
            "",
        )
    )

    if actual_base != expected_base:
        raise RuntimeError(
            "Base-model mismatch:\n"
            f"expected={expected_base}\n"
            f"actual={actual_base}"
        )

    out_dir = (
        OUT_ROOT
        / run_name
    )

    score_dir = (
        out_dir
        / "scores"
    )

    rank_dir = (
        out_dir
        / "ranks"
    )

    score_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rank_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
    )

    print("=" * 100)
    print("R1-1 JINA-POOL CROSSOVER")
    print("=" * 100)

    print(
        "condition         =",
        (
            "V_N@J"
            if args.source == "nomic"
            else "V_J@J"
        ),
    )

    print(
        "run               =",
        run_name,
    )

    print(
        "verifier          =",
        verifier_path,
    )

    print(
        "base model        =",
        BASE_MODEL,
    )

    print(
        "candidate pool    =",
        "Jina TEST top100",
    )

    print(
        "candidate_top_k   =",
        CANDIDATE_TOP_K,
    )

    print(
        "verify_top_k      =",
        VERIFY_TOP_K,
    )

    print(
        "ranking           =",
        "verifier logit difference only",
    )

    print(
        "dense fusion      =",
        False,
    )

    print(
        "dtype             =",
        VERIFIER_DTYPE,
    )

    print(
        "device_map        =",
        DEVICE_MAP,
    )

    print(
        "batch_size        =",
        VERIFIER_BATCH_SIZE,
    )

    print(
        "max_length        =",
        MAX_LENGTH,
    )

    print(
        "max_query_chars   =",
        MAX_QUERY_CHARS,
    )

    print(
        "max_code_chars    =",
        MAX_CODE_CHARS,
    )

    print("=" * 100)

    model, tokenizer = (
        load_verifier(
            str(verifier_path),
            base_model_path=
                str(BASE_MODEL),
            dtype_name=
                VERIFIER_DTYPE,
            device=
                device,
            trust_remote_code=
                False,
            device_map=
                DEVICE_MAP,
        )
    )

    yes_id = single_token_id(
        tokenizer,
        [
            " Yes",
            "Yes",
            " yes",
            "yes",
        ],
        "yes",
    )

    no_id = single_token_id(
        tokenizer,
        [
            " No",
            "No",
            " no",
            "no",
        ],
        "no",
    )

    records = []

    for lang in LANGS:

        print()
        print("=" * 100)
        print(
            "LANGUAGE:",
            lang,
        )
        print("=" * 100)

        pool_path = (
            POOL_ROOT
            / (
                f"{lang}_test_"
                "jina_top100.npz"
            )
        )

        if not pool_path.is_file():
            raise FileNotFoundError(
                pool_path
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

            if not required.issubset(
                set(z.files)
            ):
                raise RuntimeError(
                    f"{lang}: bad pool keys "
                    f"{z.files}"
                )

            indices = np.asarray(
                z["indices"],
                dtype=np.int64,
            )

            retrieval_scores = (
                np.asarray(
                    z["scores"],
                    dtype=np.float32,
                )
            )

            base_ranks = np.asarray(
                z["base_ranks"],
                dtype=np.int64,
            )

            gt = np.asarray(
                z["gt_indices"],
                dtype=np.int64,
            )

        n = len(gt)

        if indices.shape != (
            n,
            CANDIDATE_TOP_K,
        ):
            raise RuntimeError(
                f"{lang}: bad "
                f"indices shape "
                f"{indices.shape}"
            )

        if retrieval_scores.shape != (
            n,
            CANDIDATE_TOP_K,
        ):
            raise RuntimeError(
                f"{lang}: bad score shape"
            )

        if base_ranks.shape != (
            n,
        ):
            raise RuntimeError(
                f"{lang}: bad rank shape"
            )

        test_rows = read_jsonl(
            DATA_ROOT
            / lang
            / "test.jsonl"
        )

        code_rows = read_jsonl(
            DATA_ROOT
            / lang
            / "codebase.jsonl"
        )

        if len(test_rows) != n:
            raise RuntimeError(
                f"{lang}: query row "
                "count mismatch"
            )

        queries = [
            get_query_text(row)
            for row in test_rows
        ]

        codes = [
            get_code_text(row)
            for row in code_rows
        ]

        if np.any(
            indices < 0
        ) or np.any(
            indices >= len(codes)
        ):
            raise RuntimeError(
                f"{lang}: candidate index "
                "outside codebase"
            )

        score_path = (
            score_dir
            / (
                f"{lang}_test_"
                f"{run_name}_on_"
                "jina_pool_"
                "cand100_verify15."
                "scores.npy"
            )
        )

        rank_path = (
            rank_dir
            / (
                f"{lang}_test_"
                f"{run_name}_on_"
                "jina_pool_"
                "cand100_verify15_"
                "verifier_only."
                "ranks.npy"
            )
        )

        if (
            score_path.exists()
            and rank_path.exists()
        ):
            print(
                "REUSING COMPLETE "
                "EXISTING ARTIFACTS"
            )

            verifier_scores = np.load(
                score_path,
                allow_pickle=False,
            )

            ranks = np.load(
                rank_path,
                allow_pickle=False,
            ).astype(
                np.int64,
                copy=False,
            )

            if verifier_scores.shape != (
                n,
                VERIFY_TOP_K,
            ):
                raise RuntimeError(
                    "Existing score "
                    "artifact bad shape"
                )

            if ranks.shape != (
                n,
            ):
                raise RuntimeError(
                    "Existing rank "
                    "artifact bad shape"
                )

        else:

            if (
                score_path.exists()
                != rank_path.exists()
            ):
                raise RuntimeError(
                    "Partial existing "
                    "artifact detected; "
                    "inspect before rerun:\n"
                    f"{score_path}\n"
                    f"{rank_path}"
                )

            prompts = []

            for i, query in enumerate(
                queries
            ):
                for code_idx in (
                    indices[
                        i,
                        :VERIFY_TOP_K
                    ]
                ):
                    prompts.append(
                        build_yesno_prompt(
                            query=query,
                            code=codes[
                                int(code_idx)
                            ],
                            max_query_chars=
                                MAX_QUERY_CHARS,
                            max_code_chars=
                                MAX_CODE_CHARS,
                        )
                    )

            print(
                "prompts =",
                len(prompts),
            )

            flat_scores = (
                score_yesno_prompts(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    yes_id=yes_id,
                    no_id=no_id,
                    device=device,
                    batch_size=
                        VERIFIER_BATCH_SIZE,
                    max_length=
                        MAX_LENGTH,
                )
            )

            verifier_scores = (
                flat_scores.reshape(
                    n,
                    VERIFY_TOP_K,
                ).astype(
                    np.float32,
                    copy=False,
                )
            )

            ranks = (
                rerank_with_verifier(
                    candidate_indices=
                        indices,
                    candidate_scores=
                        retrieval_scores,
                    verifier_scores=
                        verifier_scores,
                    gt_indices=
                        gt,
                    base_ranks=
                        base_ranks,
                    verify_top_k=
                        VERIFY_TOP_K,
                    fusion_alpha=
                        0.0,
                    normalization=
                        "zscore",
                )
            ).astype(
                np.int64,
                copy=False,
            )

            # Save only after the full language
            # inference completed successfully.
            atomic_npy(
                score_path,
                verifier_scores.astype(
                    np.float32
                ),
            )

            atomic_npy(
                rank_path,
                ranks.astype(
                    np.int32
                ),
            )

            del prompts
            del flat_scores

        m = metric_dict(
            ranks
        )

        print(
            f"{lang}: "
            f"MRR={m['mrr']:.12f} "
            f"R1={m['r1']:.12f} "
            f"R5={m['r5']:.12f} "
            f"R10={m['r10']:.12f}"
        )

        print(
            "score_sha256 =",
            sha256(score_path),
        )

        print(
            "rank_sha256  =",
            sha256(rank_path),
        )

        records.append({
            "language":
                lang,
            "condition":
                (
                    "V_N@J"
                    if args.source
                    == "nomic"
                    else "V_J@J"
                ),
            "hn_source":
                args.source,
            "seed":
                args.seed,
            "candidate_pool":
                "jina_test_top100",
            "candidate_top_k":
                CANDIDATE_TOP_K,
            "verify_top_k":
                VERIFY_TOP_K,
            "ranking":
                "verifier_logit_difference_only",
            "dense_fusion":
                False,
            "positions_after_15":
                "preserve_jina_retriever_order",
            "metrics":
                m,
            "files": {
                "pool":
                    str(
                        pool_path.resolve()
                    ),
                "scores":
                    str(
                        score_path.resolve()
                    ),
                "ranks":
                    str(
                        rank_path.resolve()
                    ),
            },
            "sha256": {
                "pool":
                    sha256(
                        pool_path
                    ),
                "scores":
                    sha256(
                        score_path
                    ),
                "ranks":
                    sha256(
                        rank_path
                    ),
            },
        })

        del (
            test_rows,
            code_rows,
            queries,
            codes,
            indices,
            retrieval_scores,
            base_ranks,
            gt,
            verifier_scores,
            ranks,
        )

        gc.collect()
        torch.cuda.empty_cache()

    manifest = {
        "artifact_type":
            "r1_1_jina_side_crossover_v1",
        "reviewer_item":
            "R1-1",
        "condition":
            (
                "V_N@J"
                if args.source
                == "nomic"
                else "V_J@J"
            ),
        "hn_source":
            args.source,
        "seed":
            args.seed,
        "protocol": {
            "candidate_pool":
                "Jina TEST top100",
            "candidate_top_k":
                100,
            "verify_top_k":
                15,
            "verifier_mode":
                "yesno",
            "ranking":
                "verifier_logit_difference_only",
            "dense_fusion":
                False,
            "verifier_dtype":
                VERIFIER_DTYPE,
            "device_map":
                DEVICE_MAP,
            "batch_size":
                VERIFIER_BATCH_SIZE,
            "max_length":
                MAX_LENGTH,
            "max_query_chars":
                MAX_QUERY_CHARS,
            "max_code_chars":
                MAX_CODE_CHARS,
            "max_new_tokens":
                MAX_NEW_TOKENS,
        },
        "verifier": {
            "adapter":
                str(
                    verifier_path.resolve()
                ),
            "base_model":
                str(
                    BASE_MODEL.resolve()
                ),
            "adapter_sha256":
                sha256(
                    adapter_cfg
                ),
        },
        "records":
            records,
    }

    manifest_path = (
        out_dir
        / "manifest.json"
    )

    atomic_json(
        manifest_path,
        manifest,
    )

    macro = float(
        np.mean([
            x["metrics"]["mrr"]
            for x in records
        ])
    )

    total_n = sum(
        x["metrics"]["n_queries"]
        for x in records
    )

    micro = float(
        sum(
            x["metrics"]["mrr"]
            * x["metrics"]["n_queries"]
            for x in records
        )
        / total_n
    )

    print()
    print("=" * 100)
    print("RUN COMPLETE")
    print("=" * 100)

    print(
        "condition       =",
        manifest["condition"],
    )

    print(
        "run             =",
        run_name,
    )

    print(
        "languages       =",
        len(records),
    )

    print(
        "macro_mrr       =",
        f"{macro:.12f}",
    )

    print(
        "weighted_mrr    =",
        f"{micro:.12f}",
    )

    print(
        "manifest        =",
        manifest_path,
    )

    print(
        "manifest_sha256 =",
        sha256(
            manifest_path
        ),
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
