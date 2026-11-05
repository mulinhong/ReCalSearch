import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from tqdm import tqdm

from evaluate import load_model, resolve_eval_options
from config import DATA_ROOT


def read_jsonl(path: Path, max_samples: Optional[int] = None, shuffle: bool = False, seed: int = 42) -> List[Dict[str, Any]]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
            if max_samples is not None and not shuffle and len(data) >= max_samples:
                break

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(data)
        if max_samples is not None:
            data = data[:max_samples]

    return data


def get_query_text(example: Dict[str, Any]) -> str:
    # 常见字段
    for key in ["doc", "docstring", "query", "nl", "text"]:
        if key in example and example[key]:
            value = example[key]
            if isinstance(value, list):
                return " ".join(map(str, value))
            return str(value)

    # CodeSearchNet 常见字段
    if "docstring_tokens" in example and example["docstring_tokens"]:
        value = example["docstring_tokens"]
        if isinstance(value, list):
            return " ".join(map(str, value))
        return str(value)

    raise KeyError(f"Could not find query text field in example keys: {list(example.keys())}")


def get_code_text(example: Dict[str, Any]) -> str:
    # 原始代码字段
    if "code" in example and example["code"]:
        return str(example["code"])

    # CodeSearchNet 常见字段
    if "code_tokens" in example and example["code_tokens"]:
        value = example["code_tokens"]
        if isinstance(value, list):
            return " ".join(map(str, value))
        return str(value)

    raise KeyError(f"Could not find code field in example keys: {list(example.keys())}")


def canonicalize_code(code: str) -> str:
    code = re.sub(r"\s+", " ", code.strip())
    return code


@torch.no_grad()
def encode_texts(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
    pooling_mode: str,
    text_prefix: str,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    outputs = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding", leave=False):
        batch_texts = texts[start:start + batch_size]
        if text_prefix:
            batch_texts = [text_prefix + t for t in batch_texts]

        batch = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        ids = batch["input_ids"]
        mask = batch["attention_mask"]
        hidden = model(ids, attention_mask=mask)[0]

        if pooling_mode == "last_token":
            seq_lengths = mask.sum(dim=1) - 1
            batch_idx = torch.arange(hidden.size(0), device=hidden.device)
            pooled = hidden[batch_idx, seq_lengths.long()]
        elif pooling_mode == "attention":
            pooled = model.pooling_head(hidden, mask)
        else:
            in_mask = mask.unsqueeze(-1).expand(hidden.size()).float()
            pooled = torch.sum(hidden * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-6)

        pooled = torch.nn.functional.normalize(pooled, dim=1)
        outputs.append(pooled.cpu())

        del batch, ids, mask, hidden, pooled
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(outputs, dim=0)


@torch.no_grad()
def retrieve_topk(
    query_embeds: torch.Tensor,
    code_embeds: torch.Tensor,
    device: torch.device,
    top_k: int = 50,
    query_chunk_size: int = 128,
    code_chunk_size: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        top_scores: [n_queries, top_k]
        top_indices: [n_queries, top_k]
    """
    query_embeds = query_embeds.to(device)
    n_queries = query_embeds.size(0)
    n_codes = code_embeds.size(0)

    all_top_scores = []
    all_top_indices = []

    for q_start in tqdm(range(0, n_queries, query_chunk_size), desc="Retrieving top-k"):
        q_end = min(q_start + query_chunk_size, n_queries)
        q_chunk = query_embeds[q_start:q_end]

        cur_top_scores = torch.full((q_chunk.size(0), top_k), -1e9, device=device)
        cur_top_indices = torch.zeros((q_chunk.size(0), top_k), dtype=torch.long, device=device)

        for c_start in range(0, n_codes, code_chunk_size):
            c_end = min(c_start + code_chunk_size, n_codes)
            c_chunk = code_embeds[c_start:c_end].to(device)

            scores = q_chunk @ c_chunk.t()
            k = min(top_k, scores.size(1))
            chunk_scores, chunk_idx = torch.topk(scores, k=k, dim=1)
            chunk_idx = chunk_idx + c_start

            merged_scores = torch.cat([cur_top_scores, chunk_scores], dim=1)
            merged_indices = torch.cat([cur_top_indices, chunk_idx], dim=1)

            merged_top_scores, merged_pos = torch.topk(merged_scores, k=top_k, dim=1)
            merged_top_indices = merged_indices.gather(1, merged_pos)

            cur_top_scores = merged_top_scores
            cur_top_indices = merged_top_indices

            del c_chunk, scores, chunk_scores, chunk_idx, merged_scores, merged_indices, merged_top_scores, merged_pos, merged_top_indices
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_top_scores.append(cur_top_scores.cpu())
        all_top_indices.append(cur_top_indices.cpu())

        del q_chunk, cur_top_scores, cur_top_indices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(all_top_scores, dim=0), torch.cat(all_top_indices, dim=0)


def build_rerank_examples(
    train_examples: List[Dict[str, Any]],
    codebase_examples: List[Dict[str, Any]],
    top_scores: torch.Tensor,
    top_indices: torch.Tensor,
    num_negatives: int,
    language: str,
) -> List[Dict[str, Any]]:
    rerank_rows = []

    code_texts = [get_code_text(x) for x in codebase_examples]
    canon_code_texts = [canonicalize_code(x) for x in code_texts]
    canon_to_indices: Dict[str, List[int]] = {}
    for code_idx, canon in enumerate(canon_code_texts):
        canon_to_indices.setdefault(canon, []).append(code_idx)

    for i, ex in enumerate(tqdm(train_examples, desc="Building rerank rows")):
        query = get_query_text(ex)
        pos_code = get_code_text(ex)
        pos_canon = canonicalize_code(pos_code)
        query_id = f"{language}:{i}"
        pos_code_indices = canon_to_indices.get(pos_canon, [])
        pos_code_index = pos_code_indices[0] if pos_code_indices else None

        retrieved_scores = top_scores[i].tolist()
        retrieved_indices = top_indices[i].tolist()

        positive_rank = None
        positive_score = None
        for rank, (score, idx) in enumerate(zip(retrieved_scores, retrieved_indices), start=1):
            if idx == pos_code_index or canon_code_texts[idx] == pos_canon:
                positive_rank = rank
                positive_score = float(score)
                break

        # 正样本
        rerank_rows.append(
            {
                "query_id": query_id,
                "candidate_id": 0,
                "query": query,
                "code": pos_code,
                "label": 1,
                "language": language,
                "source": "positive",
                "code_index": pos_code_index,
                "retrieval_rank": positive_rank,
                "retrieval_score": positive_score,
            }
        )

        used = set(pos_code_indices)
        neg_count = 0

        for rank, (score, idx) in enumerate(zip(retrieved_scores, retrieved_indices), start=1):
            cand_code = code_texts[idx]
            cand_canon = canon_code_texts[idx]

            if cand_canon == pos_canon:
                continue
            if idx in used:
                continue

            rerank_rows.append(
                {
                    "query_id": query_id,
                    "candidate_id": neg_count + 1,
                    "query": query,
                    "code": cand_code,
                    "label": 0,
                    "language": language,
                    "source": "retrieved_negative",
                    "code_index": int(idx),
                    "retrieval_rank": rank,
                    "retrieval_score": float(score),
                }
            )
            used.add(idx)
            neg_count += 1

            if neg_count >= num_negatives:
                break

    return rerank_rows


def build_rerank_groups(
    train_examples: List[Dict[str, Any]],
    codebase_codes: List[str],
    top_scores: torch.Tensor,
    top_indices: torch.Tensor,
    num_negatives: int,
    language: str,
) -> Iterable[Dict[str, Any]]:
    code_texts = codebase_codes
    canon_code_texts = [canonicalize_code(x) for x in code_texts]
    canon_to_indices: Dict[str, List[int]] = {}
    for code_idx, canon in enumerate(canon_code_texts):
        canon_to_indices.setdefault(canon, []).append(code_idx)

    for i, ex in enumerate(tqdm(train_examples, desc="Building rerank groups")):
        query = get_query_text(ex)
        pos_code = get_code_text(ex)
        pos_canon = canonicalize_code(pos_code)
        query_id = f"{language}:{i}"
        pos_code_indices = canon_to_indices.get(pos_canon, [])
        pos_code_index = pos_code_indices[0] if pos_code_indices else None

        retrieved_scores = top_scores[i].tolist()
        retrieved_indices = top_indices[i].tolist()

        positive_rank = None
        positive_score = None
        for rank, (score, idx) in enumerate(zip(retrieved_scores, retrieved_indices), start=1):
            if idx == pos_code_index or canon_code_texts[idx] == pos_canon:
                positive_rank = rank
                positive_score = float(score)
                break

        used = set(pos_code_indices)
        negative_indices = []
        negative_ranks = []
        negative_scores = []

        for rank, (score, idx) in enumerate(zip(retrieved_scores, retrieved_indices), start=1):
            cand_canon = canon_code_texts[idx]
            if cand_canon == pos_canon or idx in used:
                continue

            negative_indices.append(int(idx))
            negative_ranks.append(int(rank))
            negative_scores.append(float(score))
            used.add(idx)

            if len(negative_indices) >= num_negatives:
                break

        yield {
            "format": "grouped_indices_v1",
            "query_id": query_id,
            "query": query,
            "language": language,
            "positive": {
                "code_index": pos_code_index,
                "code": pos_code if pos_code_index is None else None,
                "retrieval_rank": positive_rank,
                "retrieval_score": positive_score,
            },
            "negative_code_indices": negative_indices,
            "negative_retrieval_ranks": negative_ranks,
            "negative_retrieval_scores": negative_scores,
        }


def save_jsonl(rows: Iterable[Dict[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Build reranker training data from a bi-encoder retriever.")
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="microsoft/unixcoder-base")
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--output-file", type=str, required=True)

    parser.add_argument("--pooling", type=str, default="auto", choices=["auto", "mean", "last_token", "attention"])
    parser.add_argument("--query-prefix", type=str, default=None)
    parser.add_argument("--code-prefix", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--pooling-hidden-dim", type=int, default=768)

    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--query-chunk-size", type=int, default=128)
    parser.add_argument("--code-chunk-size", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--num-negatives", type=int, default=31)

    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    train_path = data_root / args.language / "train.jsonl"
    codebase_path = data_root / args.language / "codebase.jsonl"

    print(f"Loading train examples from: {train_path}")
    train_examples = read_jsonl(train_path, max_samples=args.max_samples, shuffle=args.shuffle, seed=args.seed)
    print(f"Loaded {len(train_examples)} train examples")

    print(f"Loading codebase examples from: {codebase_path}")
    codebase_examples = read_jsonl(codebase_path)
    print(f"Loaded {len(codebase_examples)} codebase examples")

    pooling_mode, query_prefix, code_prefix, max_length, pooling_hidden_dim = resolve_eval_options(
        args.adapter_path,
        args.pooling,
        args.query_prefix,
        args.code_prefix,
        args.max_length,
        args.pooling_hidden_dim,
    )

    model, tokenizer = load_model(args.model_name, args.adapter_path, pooling_mode, pooling_hidden_dim)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for efficient rerank data generation.")

    device = torch.device("cuda")
    model.to(device)
    model.eval()

    train_queries = [get_query_text(x) for x in train_examples]
    codebase_codes = [get_code_text(x) for x in codebase_examples]
    del codebase_examples

    print("Encoding train queries...")
    train_query_embeds = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=train_queries,
        device=device,
        pooling_mode=pooling_mode,
        text_prefix=query_prefix or "",
        batch_size=args.encode_batch_size,
        max_length=max_length,
    )

    print("Encoding codebase codes...")
    codebase_embeds = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=codebase_codes,
        device=device,
        pooling_mode=pooling_mode,
        text_prefix=code_prefix or "",
        batch_size=args.encode_batch_size,
        max_length=max_length,
    )

    print("Retrieving top-k candidates...")
    top_scores, top_indices = retrieve_topk(
        query_embeds=train_query_embeds,
        code_embeds=codebase_embeds,
        device=device,
        top_k=args.top_k,
        query_chunk_size=args.query_chunk_size,
        code_chunk_size=args.code_chunk_size,
    )
    del train_query_embeds, codebase_embeds, train_queries
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Building reranker dataset...")
    rerank_rows = build_rerank_groups(
        train_examples=train_examples,
        codebase_codes=codebase_codes,
        top_scores=top_scores,
        top_indices=top_indices,
        num_negatives=args.num_negatives,
        language=args.language,
    )

    output_path = Path(args.output_file)
    saved_count = save_jsonl(rerank_rows, output_path)
    print(f"Saved {saved_count} query groups to {output_path}")


if __name__ == "__main__":
    main()
