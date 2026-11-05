import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from config import DATA_ROOT, RESULTS_DIR
from data_utils import create_dataset
from eval_llm_verifier import (
    average_metric_results,
    compute_metrics_from_ranks,
    normalize_scores,
    read_jsonl,
    retrieve_topk_from_cache,
    text2code_to_array,
)
from eval_nomic_embed import coderank_cache_paths, get_code_text, get_query_text, load_cache_array
from eval_reranker import load_reranker, score_reranker_pairs


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def score_cache_path(
    cache_dir: str,
    language: str,
    split: str,
    reranker_dir: str,
    reranker_model_name: str,
    candidate_top_k: int,
    rerank_top_k: int,
    max_length: int,
    max_query_length: int,
    query_offset: int,
    max_queries: Optional[int],
) -> Path:
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    key = json.dumps(
        {
            "language": language,
            "split": split,
            "reranker_dir": str(Path(reranker_dir).resolve()),
            "reranker_model_name": reranker_model_name,
            "candidate_top_k": candidate_top_k,
            "rerank_top_k": rerank_top_k,
            "max_length": max_length,
            "max_query_length": max_query_length,
            "query_offset": query_offset,
            "max_queries": max_queries,
            "scorer": "cross_encoder_v1",
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return root / (
        f"{language}_{split}_{sanitize_name(reranker_dir)}_"
        f"cand{candidate_top_k}_rerank{rerank_top_k}_{digest}.scores.npy"
    )


def rerank_with_cross_encoder(
    candidate_indices: np.ndarray,
    candidate_scores: np.ndarray,
    reranker_scores: np.ndarray,
    gt_indices: np.ndarray,
    base_ranks: np.ndarray,
    rerank_top_k: int,
    fusion_alpha: float,
    normalization: str,
) -> np.ndarray:
    ranks = base_ranks.copy()
    n_queries = candidate_indices.shape[0]

    for i in range(n_queries):
        candidates = candidate_indices[i]
        retrieval = candidate_scores[i]
        reranked_candidates = candidates[:rerank_top_k]
        reranked_retrieval = retrieval[:rerank_top_k]
        reranked_scores = reranker_scores[i, :rerank_top_k]

        if fusion_alpha > 0.0:
            final_scores = normalize_scores(reranked_scores, normalization) + fusion_alpha * normalize_scores(
                reranked_retrieval,
                normalization,
            )
        else:
            final_scores = reranked_scores

        order = np.argsort(-final_scores)
        final_indices = list(reranked_candidates[order])
        if rerank_top_k < len(candidates):
            seen = set(int(x) for x in final_indices)
            final_indices.extend(int(x) for x in candidates[rerank_top_k:] if int(x) not in seen)

        matches = np.where(np.asarray(final_indices) == int(gt_indices[i]))[0]
        if len(matches) > 0:
            ranks[i] = int(matches[0])

    return ranks


def eval_language(args, language: str, device: torch.device, reranker_model, reranker_tokenizer):
    print(f"\n--- Cross-encoder reranker on Nomic candidates: {language} / {args.split} ---")
    _, val_dataset, test_dataset, _ = create_dataset(args.data_root, language)
    eval_dataset = val_dataset if args.split == "valid" else test_dataset
    gt_indices = text2code_to_array(eval_dataset.text2code, len(eval_dataset))

    eval_rows = read_jsonl(Path(args.data_root) / language / f"{args.split}.jsonl")
    code_rows = read_jsonl(Path(args.data_root) / language / "codebase.jsonl")
    if args.max_queries is not None:
        eval_rows = eval_rows[args.query_offset:args.query_offset + args.max_queries]
        gt_indices = gt_indices[args.query_offset:args.query_offset + args.max_queries]
    elif args.query_offset:
        eval_rows = eval_rows[args.query_offset:]
        gt_indices = gt_indices[args.query_offset:]

    queries = [get_query_text(row) for row in eval_rows]
    codes = [get_code_text(row) for row in code_rows]

    cache = coderank_cache_paths(
        cache_dir=args.nomic_cache_dir,
        language=language,
        split=args.split,
        model_name=args.nomic_model_path,
        max_seq_length=args.nomic_max_length,
        backend="nomic_transformers",
        query_prefix=args.nomic_query_prefix,
        code_prefix=args.nomic_code_prefix,
    )
    query_embeds = load_cache_array(cache["query"], len(eval_dataset), "Nomic query")
    if args.max_queries is not None or args.query_offset:
        end = None if args.max_queries is None else args.query_offset + args.max_queries
        query_embeds = query_embeds[args.query_offset:end]
    code_embeds = load_cache_array(cache["code"], len(codes), "Nomic code")

    top_scores, top_indices, base_ranks = retrieve_topk_from_cache(
        query_embeds=query_embeds,
        code_embeds=code_embeds,
        gt_indices=gt_indices,
        device=device,
        top_k=args.candidate_top_k,
        query_chunk_size=args.query_chunk_size,
        code_chunk_size=args.code_chunk_size,
    )
    print(f"  nomic_direct_subset: {compute_metrics_from_ranks(base_ranks)}")

    rerank_top_k = min(args.rerank_top_k, args.candidate_top_k)
    score_cache = None
    if args.score_cache_dir:
        score_cache = score_cache_path(
            cache_dir=args.score_cache_dir,
            language=language,
            split=args.split,
            reranker_dir=args.reranker_dir,
            reranker_model_name=args.reranker_model_name,
            candidate_top_k=args.candidate_top_k,
            rerank_top_k=rerank_top_k,
            max_length=args.reranker_max_length,
            max_query_length=args.reranker_max_query_length,
            query_offset=args.query_offset,
            max_queries=args.max_queries,
        )

    reranker_scores = None
    if score_cache is not None and args.reuse_score_cache and score_cache.exists():
        reranker_scores = np.load(score_cache)
        if reranker_scores.shape != (len(queries), rerank_top_k):
            print(f"  ignoring score cache with shape={reranker_scores.shape}, expected={(len(queries), rerank_top_k)}")
            reranker_scores = None
        else:
            print(f"  loaded cross-encoder score cache: {score_cache}")

    if reranker_scores is None:
        all_rows: List[np.ndarray] = []
        reranker_model.to(device)
        reranker_model.eval()
        for i, query in enumerate(tqdm(queries, desc="Cross-encoder scoring")):
            candidate_codes = [codes[int(idx)] for idx in top_indices[i, :rerank_top_k]]
            scores = score_reranker_pairs(
                reranker_model=reranker_model,
                reranker_tokenizer=reranker_tokenizer,
                device=device,
                query=query,
                candidate_codes=candidate_codes,
                batch_size=args.reranker_batch_size,
                max_length=args.reranker_max_length,
                max_query_length=args.reranker_max_query_length,
            )
            all_rows.append(scores.numpy().astype(np.float32))
        reranker_scores = np.stack(all_rows, axis=0).astype(np.float32)
        if score_cache is not None:
            np.save(score_cache, reranker_scores)
            print(f"  saved cross-encoder score cache: {score_cache}")

    results: Dict[str, Dict[str, Any]] = {}
    for alpha in args.fusion_alphas:
        ranks = rerank_with_cross_encoder(
            candidate_indices=top_indices,
            candidate_scores=top_scores,
            reranker_scores=reranker_scores,
            gt_indices=gt_indices,
            base_ranks=base_ranks,
            rerank_top_k=rerank_top_k,
            fusion_alpha=float(alpha),
            normalization=args.fusion_normalization,
        )
        key = f"alpha={alpha:g}"
        result = compute_metrics_from_ranks(ranks)
        result["fusion_alpha"] = float(alpha)
        result["candidate_top_k"] = args.candidate_top_k
        result["rerank_top_k"] = rerank_top_k
        results[key] = result
        print(f"  {language} / {key}: {result}")

        if args.rank_output_dir:
            rank_dir = Path(args.rank_output_dir)
            rank_dir.mkdir(parents=True, exist_ok=True)
            rank_path = rank_dir / (
                f"{language}_{args.split}_cross_encoder_"
                f"cand{args.candidate_top_k}_rerank{rerank_top_k}_{key}.ranks.npy"
            )
            np.save(rank_path, ranks.astype(np.int32))
            print(f"  saved ranks to: {rank_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a supervised cross-encoder reranker on Nomic candidates.")
    parser.add_argument("--reranker-dir", type=str, required=True)
    parser.add_argument("--reranker-model-name", type=str, default="microsoft/unixcoder-base")
    parser.add_argument("--languages", type=str, nargs="+", default=["ruby"])
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--experiment-name", type=str, default="cross_encoder_on_nomic")
    parser.add_argument("--nomic-model-path", type=str, default="/root/autodl-tmp/hf_models/nomic-embed-code")
    parser.add_argument("--nomic-cache-dir", type=str, default="data/rerank/coderank_cache")
    parser.add_argument("--nomic-max-length", type=int, default=512)
    parser.add_argument("--nomic-query-prefix", type=str, default="Represent this query for searching relevant code: ")
    parser.add_argument("--nomic-code-prefix", type=str, default="")
    parser.add_argument("--candidate-top-k", type=int, default=100)
    parser.add_argument("--rerank-top-k", type=int, default=100)
    parser.add_argument("--reranker-batch-size", type=int, default=32)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--reranker-max-query-length", type=int, default=96)
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument("--code-chunk-size", type=int, default=4096)
    parser.add_argument("--score-cache-dir", type=str, default="results/cross_encoder_scores")
    parser.add_argument("--reuse-score-cache", action="store_true")
    parser.add_argument("--fusion-alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument(
        "--fusion-normalization",
        type=str,
        default="zscore",
        choices=["none", "zscore", "minmax", "rank", "softmax"],
    )
    parser.add_argument("--rank-output-dir", type=str, default="")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--query-offset", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for cross-encoder reranker evaluation.")
    device = torch.device("cuda")

    reranker_model, reranker_tokenizer = load_reranker(
        reranker_dir=args.reranker_dir,
        model_name=args.reranker_model_name,
    )

    language_results: Dict[str, Any] = {}
    for language in args.languages:
        language_results[language] = eval_language(args, language, device, reranker_model, reranker_tokenizer)

    print("\n" + "=" * 60)
    all_results: Dict[str, Any] = {}
    for alpha in args.fusion_alphas:
        key = f"alpha={alpha:g}"
        per_lang = [language_results[lang][key] for lang in args.languages]
        avg = average_metric_results(per_lang)
        all_results[key] = {lang: language_results[lang][key] for lang in args.languages}
        all_results[key]["average"] = avg
        print(f"{key} Average: {avg}")
    print("=" * 60)

    output_path = RESULTS_DIR / f"{args.experiment_name}_eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"JSON results saved to {output_path}")


if __name__ == "__main__":
    main()
