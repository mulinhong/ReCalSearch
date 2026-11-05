import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import DATA_ROOT, RESULTS_DIR
from eval_nomic_embed import coderank_cache_paths, load_cache_array, read_jsonl


def metrics_from_ranks(ranks: np.ndarray) -> Dict[str, float]:
    ranks = np.asarray(ranks, dtype=np.int64)
    return {
        "r1": round(100.0 * float(np.mean(ranks < 1)), 2),
        "r5": round(100.0 * float(np.mean(ranks < 5)), 2),
        "r10": round(100.0 * float(np.mean(ranks < 10)), 2),
        "mrr": round(100.0 * float(np.mean(1.0 / (ranks + 1))), 2),
    }


def upper_bounds_from_ranks(ranks: np.ndarray, ks: List[int]) -> Dict[str, float]:
    ranks = np.asarray(ranks, dtype=np.int64)
    out = {}
    for k in ks:
        out[f"recall@{k}"] = round(100.0 * float(np.mean(ranks < k)), 2)
        oracle_rr = np.where(ranks < k, 1.0 / (ranks + 1), 0.0)
        out[f"oracle_mrr@{k}"] = round(100.0 * float(np.mean(oracle_rr)), 2)
    return out


@torch.no_grad()
def compute_ranks_from_cache(
    query_embeds,
    code_embeds,
    gt_indices: np.ndarray,
    device: torch.device,
    query_chunk_size: int,
    code_chunk_size: int,
) -> np.ndarray:
    n_queries = query_embeds.shape[0]
    n_codes = code_embeds.shape[0]
    ranks = []

    for q_start in tqdm(range(0, n_queries, query_chunk_size), desc="Nomic rank scan"):
        q_end = min(q_start + query_chunk_size, n_queries)
        q_chunk = torch.from_numpy(np.asarray(query_embeds[q_start:q_end]).copy()).to(device).float()
        q_chunk = F.normalize(q_chunk, dim=1)
        q_len = q_end - q_start
        gt_chunk = torch.from_numpy(gt_indices[q_start:q_end]).to(device)

        gt_scores = torch.empty(q_len, dtype=torch.float32, device=device)
        greater_counts = torch.zeros(q_len, dtype=torch.long, device=device)

        for c_start in range(0, n_codes, code_chunk_size):
            c_end = min(c_start + code_chunk_size, n_codes)
            c_chunk = torch.from_numpy(np.asarray(code_embeds[c_start:c_end]).copy()).to(device).float()
            c_chunk = F.normalize(c_chunk, dim=1)
            scores = q_chunk @ c_chunk.t()

            mask = (gt_chunk >= c_start) & (gt_chunk < c_end)
            if mask.any():
                local_gt = gt_chunk[mask] - c_start
                gt_scores[mask] = scores[mask, local_gt]

            del c_chunk, scores
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for c_start in range(0, n_codes, code_chunk_size):
            c_end = min(c_start + code_chunk_size, n_codes)
            c_chunk = torch.from_numpy(np.asarray(code_embeds[c_start:c_end]).copy()).to(device).float()
            c_chunk = F.normalize(c_chunk, dim=1)
            scores = q_chunk @ c_chunk.t()
            greater_counts += (scores > gt_scores.unsqueeze(1)).sum(dim=1)

            del c_chunk, scores
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ranks.append(greater_counts.cpu().numpy())
        del q_chunk, gt_chunk, gt_scores, greater_counts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.concatenate(ranks, axis=0).astype(np.int64)


def eval_language(args, language: str, device: torch.device) -> Dict[str, Any]:
    print(f"\n--- Nomic recall analysis {language} / {args.split} ---")
    data_root = Path(args.data_root)
    eval_examples = read_jsonl(data_root / language / f"{args.split}.jsonl")
    codebase_examples = read_jsonl(data_root / language / "codebase.jsonl")
    url_to_code = {example["url"]: idx for idx, example in enumerate(codebase_examples)}
    gt_indices = np.asarray([url_to_code[example["url"]] for example in eval_examples], dtype=np.int64)

    cache = coderank_cache_paths(
        cache_dir=args.cache_dir,
        language=language,
        split=args.split,
        model_name=args.model_path,
        max_seq_length=args.max_length,
        backend="nomic_transformers",
        query_prefix=args.query_prefix,
        code_prefix=args.code_prefix,
    )
    query_embeds = load_cache_array(cache["query"], len(eval_examples), "Nomic query")
    code_embeds = load_cache_array(cache["code"], len(codebase_examples), "Nomic code")

    ranks = compute_ranks_from_cache(
        query_embeds=query_embeds,
        code_embeds=code_embeds,
        gt_indices=gt_indices,
        device=device,
        query_chunk_size=args.query_chunk_size,
        code_chunk_size=args.code_chunk_size,
    )
    result = {
        "direct": metrics_from_ranks(ranks),
        "upper_bounds": upper_bounds_from_ranks(ranks, args.ks),
        "num_queries": int(len(ranks)),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.rank_output_dir:
        rank_dir = Path(args.rank_output_dir)
        rank_dir.mkdir(parents=True, exist_ok=True)
        rank_path = rank_dir / f"{language}_{args.split}_nomic_direct.ranks.npy"
        np.save(rank_path, ranks.astype(np.int32))
        print(f"Saved ranks to: {rank_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Analyze Nomic direct rank and recall@K from cached embeddings.")
    parser.add_argument("--languages", type=str, nargs="+", required=True)
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--model-path", type=str, default="/root/autodl-tmp/hf_models/nomic-embed-code")
    parser.add_argument("--cache-dir", type=str, default="data/rerank/coderank_cache")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--query-prefix", type=str, default="Represent this query for searching relevant code: ")
    parser.add_argument("--code-prefix", type=str, default="")
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 20, 50, 100, 200, 500, 1000])
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument("--code-chunk-size", type=int, default=4096)
    parser.add_argument("--rank-output-dir", type=str, default="")
    parser.add_argument("--experiment-name", type=str, default="nomic_recall_analysis")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Nomic recall analysis.")
    device = torch.device("cuda")

    all_results = {}
    for language in args.languages:
        all_results[language] = eval_language(args, language, device)

    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    json_output = results_dir / f"{args.experiment_name}.json"
    with json_output.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Saved analysis to: {json_output}")


if __name__ == "__main__":
    main()
