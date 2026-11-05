import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import DATA_ROOT
from eval_nomic_embed import coderank_cache_paths, get_code_text, get_query_text, load_cache_array, read_jsonl


def canonicalize_code(code: str) -> str:
    return re.sub(r"\s+", " ", code.strip())


def load_query_indices(path: str, language: str) -> Optional[List[int]]:
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return [int(x) for x in payload]
    if "languages" in payload:
        entry = payload["languages"].get(language)
        if entry is None:
            raise KeyError(f"Language {language!r} not found in query manifest {path}")
        values = entry.get("indices", entry) if isinstance(entry, dict) else entry
        return [int(x) for x in values]
    values = payload.get(language)
    if values is None:
        raise KeyError(f"Language {language!r} not found in query manifest {path}")
    return [int(x) for x in values]


def resolve_positive_code_index(example: Dict[str, Any], url_to_code: Dict[str, int], canon_to_indices: Dict[str, List[int]]) -> Optional[int]:
    url = example.get("url")
    if url is not None and url in url_to_code:
        return int(url_to_code[url])

    try:
        code = get_code_text(example)
    except Exception:
        return None
    candidates = canon_to_indices.get(canonicalize_code(code), [])
    return int(candidates[0]) if candidates else None


def make_group(
    language: str,
    row_idx: int,
    example: Dict[str, Any],
    pos_code_index: Optional[int],
    top_scores: np.ndarray,
    top_indices: np.ndarray,
    canon_code_texts: List[str],
    num_negatives: int,
) -> Dict[str, Any]:
    query = get_query_text(example)
    pos_code = get_code_text(example)
    pos_canon = canonicalize_code(pos_code)
    used = set()
    if pos_code_index is not None:
        used.add(int(pos_code_index))

    positive_rank = None
    positive_score = None
    negative_indices = []
    negative_ranks = []
    negative_scores = []

    for rank, (score, idx) in enumerate(zip(top_scores.tolist(), top_indices.tolist()), start=1):
        idx = int(idx)
        score = float(score)
        is_positive = False
        if pos_code_index is not None and idx == int(pos_code_index):
            is_positive = True
        elif canon_code_texts[idx] == pos_canon:
            is_positive = True

        if is_positive:
            if positive_rank is None:
                positive_rank = int(rank)
                positive_score = score
            used.add(idx)
            continue

        if idx in used:
            continue

        negative_indices.append(idx)
        negative_ranks.append(int(rank))
        negative_scores.append(score)
        used.add(idx)

        if len(negative_indices) >= num_negatives:
            break

    return {
        "format": "grouped_indices_v1",
        "mining": "nomic",
        "mining_retriever": "nomic",
        "query_id": f"{language}:{row_idx}",
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


@torch.no_grad()
def iter_topk_chunks(
    query_embeds,
    code_embeds,
    device: torch.device,
    top_k: int,
    query_chunk_size: int,
    code_chunk_size: int,
) -> Iterable[Tuple[int, np.ndarray, np.ndarray]]:
    n_queries = query_embeds.shape[0]
    n_codes = code_embeds.shape[0]
    top_k = min(top_k, n_codes)

    for q_start in tqdm(range(0, n_queries, query_chunk_size), desc="Nomic train retrieval"):
        q_end = min(q_start + query_chunk_size, n_queries)
        q_chunk = torch.from_numpy(np.asarray(query_embeds[q_start:q_end]).copy()).to(device).float()
        q_chunk = F.normalize(q_chunk, dim=1)
        q_len = q_end - q_start

        cur_scores = torch.full((q_len, top_k), -1e9, dtype=torch.float32, device=device)
        cur_indices = torch.zeros((q_len, top_k), dtype=torch.long, device=device)

        for c_start in range(0, n_codes, code_chunk_size):
            c_end = min(c_start + code_chunk_size, n_codes)
            c_chunk = torch.from_numpy(np.asarray(code_embeds[c_start:c_end]).copy()).to(device).float()
            c_chunk = F.normalize(c_chunk, dim=1)
            scores = q_chunk @ c_chunk.t()
            k = min(top_k, scores.size(1))
            chunk_scores, chunk_indices = torch.topk(scores, k=k, dim=1)
            chunk_indices = chunk_indices + c_start

            merged_scores = torch.cat([cur_scores, chunk_scores], dim=1)
            merged_indices = torch.cat([cur_indices, chunk_indices], dim=1)
            cur_scores, merged_pos = torch.topk(merged_scores, k=top_k, dim=1)
            cur_indices = merged_indices.gather(1, merged_pos)

            del c_chunk, scores, chunk_scores, chunk_indices, merged_scores, merged_indices, merged_pos
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        yield q_start, cur_scores.cpu().numpy(), cur_indices.cpu().numpy()

        del q_chunk, cur_scores, cur_indices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Build reranker training data from Nomic cached embeddings.")
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"])
    parser.add_argument("--nomic-model-path", type=str, default="/root/autodl-tmp/hf_models/nomic-embed-code")
    parser.add_argument("--nomic-cache-dir", type=str, default="data/rerank/coderank_cache")
    parser.add_argument("--nomic-max-length", type=int, default=512)
    parser.add_argument("--nomic-query-prefix", type=str, default="Represent this query for searching relevant code: ")
    parser.add_argument("--nomic-code-prefix", type=str, default="")
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--num-negatives", type=int, default=31)
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument("--code-chunk-size", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--query-indices-file", type=str, default="", help="Immutable query manifest used to keep every negative-source control on identical training queries.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Nomic rerank data mining.")
    device = torch.device("cuda")

    data_root = Path(args.data_root)
    examples = read_jsonl(data_root / args.language / f"{args.split}.jsonl")
    codebase_examples = read_jsonl(data_root / args.language / "codebase.jsonl")
    print(f"Read {len(examples)} data from {data_root / args.language / f'{args.split}.jsonl'}")
    print(f"Read {len(codebase_examples)} data from {data_root / args.language / 'codebase.jsonl'}")

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
    query_embeds = load_cache_array(cache["query"], len(examples), "Nomic query")
    code_embeds = load_cache_array(cache["code"], len(codebase_examples), "Nomic code")

    original_examples = examples
    selected_indices = load_query_indices(args.query_indices_file, args.language)
    if selected_indices is not None:
        if args.query_offset or args.max_samples is not None:
            raise ValueError("Use either --query-indices-file or --query-offset/--max-samples, not both.")
        if any(i < 0 or i >= len(original_examples) for i in selected_indices):
            raise IndexError("Query manifest contains out-of-range indices.")
        examples = [original_examples[i] for i in selected_indices]
        query_embeds = np.asarray(query_embeds[selected_indices])
        print(f"Using immutable query manifest {args.query_indices_file}: rows={len(examples)}")
    else:
        end = None if args.max_samples is None else args.query_offset + args.max_samples
        if args.query_offset or args.max_samples is not None:
            examples = original_examples[args.query_offset:end]
            query_embeds = query_embeds[args.query_offset:end]
            print(f"Using query slice offset={args.query_offset}, rows={len(examples)}")

    code_texts = [get_code_text(row) for row in codebase_examples]
    canon_code_texts = [canonicalize_code(code) for code in code_texts]
    canon_to_indices: Dict[str, List[int]] = {}
    for idx, canon in enumerate(canon_code_texts):
        canon_to_indices.setdefault(canon, []).append(idx)
    url_to_code = {row["url"]: idx for idx, row in enumerate(codebase_examples) if "url" in row}

    pos_indices = [
        resolve_positive_code_index(row, url_to_code=url_to_code, canon_to_indices=canon_to_indices)
        for row in examples
    ]
    missing_pos = sum(1 for idx in pos_indices if idx is None)
    if missing_pos:
        print(f"[WARN] {missing_pos} examples have no resolved codebase index; positive code will be stored inline.")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as f:
        for q_start, top_scores_chunk, top_indices_chunk in iter_topk_chunks(
            query_embeds=query_embeds,
            code_embeds=code_embeds,
            device=device,
            top_k=max(args.top_k, args.num_negatives + 1),
            query_chunk_size=args.query_chunk_size,
            code_chunk_size=args.code_chunk_size,
        ):
            for local_idx in range(top_indices_chunk.shape[0]):
                row_idx = q_start + local_idx
                group = make_group(
                    language=args.language,
                    row_idx=(selected_indices[row_idx] if selected_indices is not None else args.query_offset + row_idx),
                    example=examples[row_idx],
                    pos_code_index=pos_indices[row_idx],
                    top_scores=top_scores_chunk[local_idx],
                    top_indices=top_indices_chunk[local_idx],
                    canon_code_texts=canon_code_texts,
                    num_negatives=args.num_negatives,
                )
                if len(group["negative_code_indices"]) < min(args.num_negatives, 1):
                    skipped += 1
                    continue
                f.write(json.dumps(group, ensure_ascii=False) + "\n")
                saved += 1

    print(f"Saved {saved} Nomic rerank groups to: {output_path}")
    if skipped:
        print(f"Skipped {skipped} groups without negatives.")


if __name__ == "__main__":
    main()
