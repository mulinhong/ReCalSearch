import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from config import DATA_ROOT
from data_utils import create_dataset
from eval_llm_verifier import (
    build_yesno_prompt,
    read_jsonl,
    retrieve_topk_from_cache,
    single_token_id,
    text2code_to_array,
    load_verifier,
)
from eval_nomic_embed import coderank_cache_paths, get_code_text, get_query_text, load_cache_array


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def select_queries(top_scores: np.ndarray, call_rate: float) -> np.ndarray:
    n_queries = top_scores.shape[0]
    if call_rate >= 100.0:
        return np.arange(n_queries, dtype=np.int64)
    if call_rate <= 0.0:
        return np.asarray([], dtype=np.int64)

    margins = top_scores[:, 0] - top_scores[:, 1]
    keep = max(1, int(round(n_queries * call_rate / 100.0)))
    return np.argsort(margins)[:keep].astype(np.int64)


@torch.no_grad()
def score_one_query(
    model,
    tokenizer,
    prompts: List[str],
    yes_id: int,
    no_id: int,
    device: torch.device,
    batch_size: int,
    max_length: int,
):
    scores = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        enc = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        outputs = model(**enc)
        last_pos = enc["attention_mask"].sum(dim=1) - 1
        batch_idx = torch.arange(len(batch_prompts), device=device)
        next_logits = outputs.logits[batch_idx, last_pos, :].float()
        scores.append((next_logits[:, yes_id] - next_logits[:, no_id]).detach().cpu())
        del enc, outputs, next_logits
    return torch.cat(scores).numpy()


def benchmark_language(args, language: str, device: torch.device, model, tokenizer, yes_id: int, no_id: int):
    print(f"\n--- Runtime benchmark {language} / {args.split} ---")
    _, val_dataset, test_dataset, _ = create_dataset(args.data_root, language)
    eval_dataset = val_dataset if args.split == "valid" else test_dataset
    gt_indices = text2code_to_array(eval_dataset.text2code, len(eval_dataset))

    eval_rows = read_jsonl(Path(args.data_root) / language / f"{args.split}.jsonl")
    code_rows = read_jsonl(Path(args.data_root) / language / "codebase.jsonl")
    if args.max_queries_per_language is not None:
        rng = np.random.default_rng(args.seed)
        n = len(eval_rows)
        take = min(args.max_queries_per_language, n)
        if args.sample_queries:
            chosen = np.sort(rng.choice(n, size=take, replace=False))
            eval_rows = [eval_rows[int(i)] for i in chosen]
            gt_indices = gt_indices[chosen]
            query_slice_note = f"random_sample={take}"
        else:
            eval_rows = eval_rows[:take]
            gt_indices = gt_indices[:take]
            query_slice_note = f"first={take}"
    else:
        query_slice_note = "all"

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
    if args.max_queries_per_language is not None:
        if args.sample_queries:
            query_embeds = query_embeds[chosen]
        else:
            query_embeds = query_embeds[:len(queries)]
    code_embeds = load_cache_array(cache["code"], len(codes), "Nomic code")

    top_scores, top_indices, _ = retrieve_topk_from_cache(
        query_embeds=query_embeds,
        code_embeds=code_embeds,
        gt_indices=gt_indices,
        device=device,
        top_k=args.candidate_top_k,
        query_chunk_size=args.query_chunk_size,
        code_chunk_size=args.code_chunk_size,
    )

    selected = select_queries(top_scores, args.call_rate)
    latencies = []
    n_pairs = 0
    start_wall = time.perf_counter()
    for query_idx in tqdm(selected.tolist(), desc=f"{language} verifier calls"):
        prompts = [
            build_yesno_prompt(
                query=queries[query_idx],
                code=codes[int(code_idx)],
                max_query_chars=args.max_query_chars,
                max_code_chars=args.max_code_chars,
            )
            for code_idx in top_indices[query_idx, :args.verify_top_k]
        ]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = score_one_query(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            yes_id=yes_id,
            no_id=no_id,
            device=device,
            batch_size=args.verifier_batch_size,
            max_length=args.max_length,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        n_pairs += len(prompts)
    wall = time.perf_counter() - start_wall

    total_queries = len(queries)
    called_queries = len(selected)
    amortized = wall / total_queries if total_queries else 0.0
    called_avg = float(np.mean(latencies)) if latencies else 0.0
    return {
        "language": language,
        "query_slice": query_slice_note,
        "total_queries": total_queries,
        "called_queries": called_queries,
        "call_rate": round(100.0 * called_queries / total_queries, 4) if total_queries else 0.0,
        "verify_top_k": args.verify_top_k,
        "candidate_top_k": args.candidate_top_k,
        "pairs_scored": n_pairs,
        "wall_seconds": round(wall, 4),
        "amortized_latency_s_per_query": round(amortized, 6),
        "called_latency_mean_s": round(called_avg, 6),
        "called_latency_p50_s": round(percentile(latencies, 50), 6),
        "called_latency_p90_s": round(percentile(latencies, 90), 6),
        "called_latency_p95_s": round(percentile(latencies, 95), 6),
        "called_latency_p99_s": round(percentile(latencies, 99), 6),
        "qps_all_queries": round(total_queries / wall, 6) if wall > 0 else 0.0,
        "qps_called_queries": round(called_queries / wall, 6) if wall > 0 else 0.0,
        "pairs_per_second": round(n_pairs / wall, 6) if wall > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Measure actual verifier wall-clock latency and throughput.")
    parser.add_argument("--languages", type=str, nargs="+", default=["ruby"])
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--nomic-model-path", type=str, default="/root/autodl-tmp/hf_models/nomic-embed-code")
    parser.add_argument("--nomic-cache-dir", type=str, default="data/rerank/coderank_cache")
    parser.add_argument("--nomic-max-length", type=int, default=512)
    parser.add_argument("--nomic-query-prefix", type=str, default="Represent this query for searching relevant code: ")
    parser.add_argument("--nomic-code-prefix", type=str, default="")
    parser.add_argument("--candidate-top-k", type=int, default=100)
    parser.add_argument("--verify-top-k", type=int, default=15)
    parser.add_argument("--call-rate", type=float, default=100.0)
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument("--code-chunk-size", type=int, default=4096)
    parser.add_argument("--verifier-model", type=str, required=True)
    parser.add_argument("--verifier-base-model", type=str, default="")
    parser.add_argument("--verifier-dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--verifier-device-map", type=str, default="cuda", choices=["cuda", "auto", "none"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--verifier-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-query-chars", type=int, default=512)
    parser.add_argument("--max-code-chars", type=int, default=3000)
    parser.add_argument("--max-queries-per-language", type=int, default=None)
    parser.add_argument("--sample-queries", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=str, default="results/oldcsn_verifier_runtime_benchmark.csv")
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for verifier runtime benchmarking.")
    device = torch.device("cuda")

    model, tokenizer = load_verifier(
        args.verifier_model,
        base_model_path=args.verifier_base_model,
        dtype_name=args.verifier_dtype,
        device=device,
        trust_remote_code=args.trust_remote_code,
        device_map=args.verifier_device_map,
    )
    yes_id = single_token_id(tokenizer, [" Yes", "Yes", " yes", "yes"], "yes")
    no_id = single_token_id(tokenizer, [" No", "No", " no", "no"], "no")

    rows = [
        benchmark_language(args, language, device, model, tokenizer, yes_id, no_id)
        for language in args.languages
    ]
    total_queries = sum(int(row["total_queries"]) for row in rows)
    total_called = sum(int(row["called_queries"]) for row in rows)
    total_pairs = sum(int(row["pairs_scored"]) for row in rows)
    total_wall = sum(float(row["wall_seconds"]) for row in rows)
    rows.append(
        {
            "language": "average_weighted",
            "query_slice": "weighted",
            "total_queries": total_queries,
            "called_queries": total_called,
            "call_rate": round(100.0 * total_called / total_queries, 4) if total_queries else 0.0,
            "verify_top_k": args.verify_top_k,
            "candidate_top_k": args.candidate_top_k,
            "pairs_scored": total_pairs,
            "wall_seconds": round(total_wall, 4),
            "amortized_latency_s_per_query": round(total_wall / total_queries, 6) if total_queries else 0.0,
            "called_latency_mean_s": "",
            "called_latency_p50_s": "",
            "called_latency_p90_s": "",
            "called_latency_p95_s": "",
            "called_latency_p99_s": "",
            "qps_all_queries": round(total_queries / total_wall, 6) if total_wall > 0 else 0.0,
            "qps_called_queries": round(total_called / total_wall, 6) if total_wall > 0 else 0.0,
            "pairs_per_second": round(total_pairs / total_wall, 6) if total_wall > 0 else 0.0,
        }
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved runtime CSV to: {output_csv}")
    for row in rows:
        print(row)

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"Saved runtime JSON to: {output_json}")


if __name__ == "__main__":
    main()
