#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data_utils import create_dataset
from eval_nomic_embed import coderank_cache_paths, load_cache_array, read_jsonl


def text2code_to_array(text2code, n_queries: int) -> np.ndarray:
    """Exact conversion used by the ReCalSearch main evaluator."""
    if isinstance(text2code, torch.Tensor):
        return text2code.long().cpu().numpy()
    if isinstance(text2code, dict):
        return np.asarray([text2code[i] for i in range(n_queries)], dtype=np.int64)
    return np.asarray(list(text2code), dtype=np.int64)


@torch.no_grad()
def retrieve_topk_from_cache(
    query_embeds,
    code_embeds,
    gt_indices: np.ndarray,
    device: torch.device,
    top_k: int,
    query_chunk_size: int,
    code_chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Byte-for-byte logic equivalent to eval_llm_verifier.retrieve_topk_from_cache.

    Kept locally on purpose so this exporter does not import eval_llm_verifier and therefore
    does not depend on reviewer-revision helper modules such as revision_utils.py.
    """
    n_queries = query_embeds.shape[0]
    n_codes = code_embeds.shape[0]
    top_k = min(top_k, n_codes)

    all_top_scores = []
    all_top_indices = []
    all_ranks = []

    for q_start in tqdm(range(0, n_queries, query_chunk_size), desc="Nomic top-k"):
        q_end = min(q_start + query_chunk_size, n_queries)
        q_chunk = torch.from_numpy(np.asarray(query_embeds[q_start:q_end]).copy()).to(device).float()
        q_chunk = F.normalize(q_chunk, dim=1)
        q_len = q_end - q_start
        chunk_scores = torch.empty((q_len, n_codes), dtype=torch.float32, device=device)

        for c_start in range(0, n_codes, code_chunk_size):
            c_end = min(c_start + code_chunk_size, n_codes)
            c_chunk = torch.from_numpy(np.asarray(code_embeds[c_start:c_end]).copy()).to(device).float()
            c_chunk = F.normalize(c_chunk, dim=1)
            chunk_scores[:, c_start:c_end] = q_chunk @ c_chunk.t()
            del c_chunk

        top_scores, top_indices = torch.topk(chunk_scores, k=top_k, dim=1)
        gt_chunk = torch.from_numpy(gt_indices[q_start:q_end]).to(device)
        gt_scores = chunk_scores.gather(1, gt_chunk.unsqueeze(1)).squeeze(1)
        ranks = (chunk_scores > gt_scores.unsqueeze(1)).sum(dim=1)

        all_top_scores.append(top_scores.cpu().numpy())
        all_top_indices.append(top_indices.cpu().numpy())
        all_ranks.append(ranks.cpu().numpy())

        del q_chunk, chunk_scores, top_scores, top_indices, gt_chunk, gt_scores, ranks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return (
        np.concatenate(all_top_scores, axis=0),
        np.concatenate(all_top_indices, axis=0),
        np.concatenate(all_ranks, axis=0).astype(np.int64),
    )


def resolve_protocol_gt_indices(data_root: Path, language: str, split: str) -> np.ndarray:
    """Use the exact ground-truth mapping used by ReCalSearch's main verifier evaluator."""
    _, val_dataset, test_dataset, _ = create_dataset(str(data_root), language)
    eval_dataset = val_dataset if split == "valid" else test_dataset
    gt_indices = text2code_to_array(eval_dataset.text2code, len(eval_dataset)).astype(np.int64, copy=False)

    eval_rows = read_jsonl(data_root / language / f"{split}.jsonl")
    if len(eval_rows) != len(eval_dataset):
        raise RuntimeError(
            f"Protocol mismatch for {language}/{split}: JSONL has {len(eval_rows)} rows, "
            f"create_dataset returned {len(eval_dataset)} rows."
        )
    return gt_indices


def compute_metrics_from_ranks(ranks: np.ndarray, top_k: int) -> Dict[str, float]:
    ranks = np.asarray(ranks, dtype=np.int64)
    return {
        "r1": float(np.mean(ranks < 1)),
        "r5": float(np.mean(ranks < 5)),
        "r10": float(np.mean(ranks < 10)),
        "mrr": float(np.mean(1.0 / (ranks.astype(np.float64) + 1.0))),
        "candidate_recall_at_k": float(np.mean(ranks < top_k)),
    }


def validate_reference_ranks(new_ranks: np.ndarray, reference_path: str) -> Dict[str, Any]:
    reference = np.asarray(np.load(reference_path), dtype=np.int64)
    new_ranks = np.asarray(new_ranks, dtype=np.int64)
    if reference.shape != new_ranks.shape:
        raise RuntimeError(
            f"Reference rank shape mismatch: reference={reference.shape}, new={new_ranks.shape}, "
            f"file={reference_path}"
        )
    diff = new_ranks - reference
    return {
        "reference_rank_file": str(Path(reference_path).resolve()),
        "reference_exact_equal": bool(np.array_equal(reference, new_ranks)),
        "reference_different_queries": int(np.count_nonzero(diff)),
        "reference_max_abs_diff": int(np.max(np.abs(diff))) if diff.size else 0,
        "reference_mrr": float(np.mean(1.0 / (reference.astype(np.float64) + 1.0))),
        "new_mrr": float(np.mean(1.0 / (new_ranks.astype(np.float64) + 1.0))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Export a frozen Nomic top-k pool using the exact ReCalSearch main-evaluator protocol. "
            "This standalone exporter intentionally does not import eval_llm_verifier.py."
        )
    )
    ap.add_argument("--language", required=True)
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--nomic-model-path", default="/root/autodl-tmp/hf_models/nomic-embed-code")
    ap.add_argument("--nomic-cache-dir", default="data/rerank/coderank_cache_oldcsn")
    ap.add_argument("--nomic-max-length", type=int, default=512)
    ap.add_argument("--nomic-query-prefix", default="Represent this query for searching relevant code: ")
    ap.add_argument("--nomic-code-prefix", default="")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--query-chunk-size", type=int, default=64)
    ap.add_argument("--code-chunk-size", type=int, default=4096)
    ap.add_argument("--output-file", required=True)
    ap.add_argument(
        "--reference-ranks",
        default="",
        help="Historical per-query Nomic rank .npy; mismatch aborts the export.",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to export Nomic candidate pools.")

    data_root = Path(args.data_root)
    gt_indices = resolve_protocol_gt_indices(data_root, args.language, args.split)

    code_rows = read_jsonl(data_root / args.language / "codebase.jsonl")
    n_codes = len(code_rows)

    cache = coderank_cache_paths(
        cache_dir=args.nomic_cache_dir,
        language=args.language,
        split=args.split,
        model_name=args.nomic_model_path,
        max_seq_length=args.nomic_max_length,
        backend="nomic_transformers",
        query_prefix=args.nomic_query_prefix,
        code_prefix=args.nomic_code_prefix,
    )
    query_embeds = load_cache_array(cache["query"], len(gt_indices), "Nomic query")
    code_embeds = load_cache_array(cache["code"], n_codes, "Nomic code")

    top_scores, top_indices, base_ranks = retrieve_topk_from_cache(
        query_embeds=query_embeds,
        code_embeds=code_embeds,
        gt_indices=gt_indices,
        device=torch.device("cuda"),
        top_k=args.top_k,
        query_chunk_size=args.query_chunk_size,
        code_chunk_size=args.code_chunk_size,
    )

    metrics = compute_metrics_from_ranks(base_ranks, args.top_k)
    summary: Dict[str, Any] = {
        "protocol": "recal_main_text2code_v1_standalone",
        "language": args.language,
        "split": args.split,
        "queries": int(len(gt_indices)),
        "codes": int(n_codes),
        "top_k": int(args.top_k),
        "r1": metrics["r1"],
        "r5": metrics["r5"],
        "r10": metrics["r10"],
        "base_mrr": metrics["mrr"],
        "candidate_recall_at_k": metrics["candidate_recall_at_k"],
        "gt_source": "create_dataset(...).text2code",
        "retrieval_impl": "local exact copy of eval_llm_verifier.retrieve_topk_from_cache",
        "query_cache": str(Path(cache["query"]).resolve()),
        "code_cache": str(Path(cache["code"]).resolve()),
    }

    if args.reference_ranks:
        summary.update(validate_reference_ranks(base_ranks, args.reference_ranks))
        if not summary["reference_exact_equal"]:
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            raise SystemExit(
                "ERROR: newly computed base_ranks do not exactly match the historical Nomic ranks. "
                "Pool was NOT written."
            )

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_file,
        indices=np.asarray(top_indices, dtype=np.int64),
        scores=np.asarray(top_scores, dtype=np.float32),
        base_ranks=np.asarray(base_ranks, dtype=np.int64),
        gt_indices=np.asarray(gt_indices, dtype=np.int64),
    )
    summary["output_file"] = str(output_file.resolve())

    summary_path = output_file.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
