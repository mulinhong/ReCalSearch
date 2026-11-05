import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import DATA_ROOT
from evaluate import calculate_ndcg


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def cache_paths(
    cache_dir: str,
    language: str,
    split: str,
    model_name: str,
    max_seq_length: int,
    query_prefix: str,
    code_prefix: str,
) -> Dict[str, Path]:
    root = Path(cache_dir)
    model_key = sanitize_name(model_name)
    cache_key = json.dumps(
        {
            "model_name": model_name,
            "backend": "nomic_transformers",
            "query_prefix": query_prefix,
            "code_prefix": code_prefix,
            "max_seq_length": max_seq_length,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:10]
    prefix = root / f"{language}_{model_key}_msl{max_seq_length}_{digest}"
    return {
        "query": root / f"{language}_{split}_{model_key}_msl{max_seq_length}_{digest}.query.npy",
        "code": prefix.with_suffix(".code.npy"),
    }


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_ranks(path: str) -> np.ndarray:
    ranks = np.load(path)
    if ranks.ndim != 1:
        raise ValueError(f"Expected 1D ranks in {path}, got shape={ranks.shape}")
    return ranks.astype(np.int64)


def metrics_from_ranks(ranks: np.ndarray) -> Dict[str, float]:
    ndcgs = []
    for rank in ranks:
        rel = [0] * 10
        if rank < 10:
            rel[int(rank)] = 1
        ndcgs.append(calculate_ndcg(rel, k=10))
    return {
        "r1": round(100.0 * float(np.mean(ranks < 1)), 2),
        "r5": round(100.0 * float(np.mean(ranks < 5)), 2),
        "r10": round(100.0 * float(np.mean(ranks < 10)), 2),
        "mrr": round(100.0 * float(np.mean(1.0 / (ranks + 1))), 2),
        "ndcg@10": round(100.0 * float(np.mean(ndcgs)), 2),
    }


@torch.no_grad()
def compute_top2_margins(
    query_embeds,
    code_embeds,
    device: torch.device,
    query_chunk_size: int,
    code_chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_queries = query_embeds.shape[0]
    margins = []
    top1_scores = []

    for q_start in tqdm(range(0, n_queries, query_chunk_size), desc="Nomic top2 margins"):
        q_end = min(q_start + query_chunk_size, n_queries)
        q_chunk = torch.from_numpy(np.asarray(query_embeds[q_start:q_end])).to(device).float()
        q_chunk = F.normalize(q_chunk, dim=1)
        q_len = q_end - q_start

        best_scores = torch.full((q_len, 2), -1e9, dtype=torch.float32, device=device)
        for c_start in range(0, code_embeds.shape[0], code_chunk_size):
            c_end = min(c_start + code_chunk_size, code_embeds.shape[0])
            c_chunk = torch.from_numpy(np.asarray(code_embeds[c_start:c_end])).to(device).float()
            c_chunk = F.normalize(c_chunk, dim=1)
            scores = q_chunk @ c_chunk.t()
            chunk_top = torch.topk(scores, k=min(2, scores.size(1)), dim=1).values
            merged = torch.cat([best_scores, chunk_top], dim=1)
            best_scores = torch.topk(merged, k=2, dim=1).values
            del c_chunk, scores, chunk_top, merged
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        margin = (best_scores[:, 0] - best_scores[:, 1]).cpu().numpy()
        margins.append(margin)
        top1_scores.append(best_scores[:, 0].cpu().numpy())
        del q_chunk, best_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.concatenate(margins), np.concatenate(top1_scores)


def route_by_threshold(
    margins: np.ndarray,
    direct_ranks: np.ndarray,
    fusion_ranks: np.ndarray,
    threshold: float,
) -> np.ndarray:
    # Low margin means Nomic is uncertain, so route to fusion.
    use_fusion = margins <= threshold
    return np.where(use_fusion, fusion_ranks, direct_ranks)


def main():
    parser = argparse.ArgumentParser(description="Analyze margin-based adaptive routing between Nomic direct and fusion.")
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--split", type=str, default="valid", choices=["valid", "test"])
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="/root/autodl-tmp/hf_models/nomic-embed-code")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--query-prefix", type=str, default="Represent this query for searching relevant code: ")
    parser.add_argument("--code-prefix", type=str, default="")
    parser.add_argument("--direct-ranks", type=str, required=True)
    parser.add_argument("--fusion-ranks", type=str, required=True)
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument("--code-chunk-size", type=int, default=4096)
    parser.add_argument("--num-thresholds", type=int, default=101)
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for margin analysis.")
    device = torch.device("cuda")

    eval_rows = read_jsonl(Path(args.data_root) / args.language / f"{args.split}.jsonl")
    caches = cache_paths(
        cache_dir=args.cache_dir,
        language=args.language,
        split=args.split,
        model_name=args.model_path,
        max_seq_length=args.max_length,
        query_prefix=args.query_prefix,
        code_prefix=args.code_prefix,
    )
    query_embeds = np.load(caches["query"], mmap_mode="r")
    code_embeds = np.load(caches["code"], mmap_mode="r")
    if query_embeds.shape[0] != len(eval_rows):
        raise ValueError(f"Query cache has {query_embeds.shape[0]} rows, expected {len(eval_rows)}")

    direct_ranks = load_ranks(args.direct_ranks)
    fusion_ranks = load_ranks(args.fusion_ranks)
    if len(direct_ranks) != len(fusion_ranks) or len(direct_ranks) != len(eval_rows):
        raise ValueError("Rank files and eval split have different lengths.")

    margins, top1_scores = compute_top2_margins(
        query_embeds=query_embeds,
        code_embeds=code_embeds,
        device=device,
        query_chunk_size=args.query_chunk_size,
        code_chunk_size=args.code_chunk_size,
    )

    thresholds = np.quantile(margins, np.linspace(0.0, 1.0, args.num_thresholds))
    threshold_results = []
    best = None
    for threshold in thresholds:
        routed = route_by_threshold(margins, direct_ranks, fusion_ranks, float(threshold))
        metrics = metrics_from_ranks(routed)
        use_fusion = margins <= threshold
        item = {
            "threshold": float(threshold),
            "fusion_rate": round(100.0 * float(use_fusion.mean()), 2),
            **metrics,
        }
        threshold_results.append(item)
        if best is None or item["mrr"] > best["mrr"]:
            best = item

    oracle_ranks = np.minimum(direct_ranks, fusion_ranks)
    results = {
        "direct": metrics_from_ranks(direct_ranks),
        "fusion": metrics_from_ranks(fusion_ranks),
        "oracle": metrics_from_ranks(oracle_ranks),
        "best_margin_router": best,
        "margin_summary": {
            "min": float(np.min(margins)),
            "p25": float(np.quantile(margins, 0.25)),
            "median": float(np.median(margins)),
            "p75": float(np.quantile(margins, 0.75)),
            "max": float(np.max(margins)),
            "top1_mean": float(np.mean(top1_scores)),
        },
        "threshold_results": threshold_results,
    }

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved routing heuristic analysis to: {output_path}")


if __name__ == "__main__":
    main()
