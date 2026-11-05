import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from config import DATA_ROOT, RESULTS_DIR
from evaluate import calculate_ndcg


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def coderank_cache_paths(
    cache_dir: str,
    language: str,
    split: str,
    model_name: str,
    max_seq_length: int,
    backend: str,
    query_prefix: str,
    code_prefix: str,
) -> Dict[str, Path]:
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    model_key = sanitize_name(model_name)
    cache_key = json.dumps(
        {
            "model_name": model_name,
            "backend": backend,
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
        "code": prefix.with_suffix(".code.npy"),
        "query": root / f"{language}_{split}_{model_key}_msl{max_seq_length}_{digest}.query.npy",
    }


def save_embedding_cache(
    query_embeds: torch.Tensor,
    code_embeds: torch.Tensor,
    language: str,
    split: str,
    model_name: str,
    max_seq_length: int,
    query_prefix: str,
    code_prefix: str,
    cache_dir: str,
    embedding_dtype: str,
):
    if not cache_dir:
        return

    dtype = np.float16 if embedding_dtype == "float16" else np.float32
    cache = coderank_cache_paths(
        cache_dir=cache_dir,
        language=language,
        split=split,
        model_name=model_name,
        max_seq_length=max_seq_length,
        backend="nomic_transformers",
        query_prefix=query_prefix,
        code_prefix=code_prefix,
    )
    print(f"Saving Nomic query cache to: {cache['query']}")
    np.save(cache["query"], query_embeds.numpy().astype(dtype, copy=False))
    print(f"Saving Nomic code cache to: {cache['code']}")
    np.save(cache["code"], code_embeds.numpy().astype(dtype, copy=False))


def export_dtype(dtype_name: str):
    if dtype_name == "float16":
        return np.float16
    if dtype_name == "float32":
        return np.float32
    raise ValueError(f"Unsupported export dtype: {dtype_name}")


def cache_matches(path: Path, expected_rows: int, label: str) -> bool:
    if not path.exists():
        return False
    try:
        arr = np.load(path, mmap_mode="r")
    except Exception as exc:
        print(f"Ignoring unreadable {label} cache {path}: {exc}")
        return False
    if arr.shape[0] != expected_rows:
        print(f"Ignoring {label} cache with {arr.shape[0]} rows; expected {expected_rows}: {path}")
        return False
    print(f"Reusing {label} cache: {path}")
    return True


def load_cache_array(path: Path, expected_rows: int, label: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} cache: {path}")
    arr = np.load(path, mmap_mode="r")
    if arr.shape[0] != expected_rows:
        raise ValueError(f"{label} cache has {arr.shape[0]} rows, expected {expected_rows}: {path}")
    print(f"Loaded {label} cache: {path} shape={arr.shape} dtype={arr.dtype}")
    return arr


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_query_text(example: Dict[str, Any], use_raw: bool = True) -> str:
    keys = ["docstring", "doc", "query", "nl", "text"] if use_raw else []
    for key in keys:
        if key in example and example[key]:
            value = example[key]
            if isinstance(value, list):
                return " ".join(map(str, value))
            return str(value)

    if "docstring_tokens" in example and example["docstring_tokens"]:
        value = example["docstring_tokens"]
        if isinstance(value, list):
            return " ".join(map(str, value))
        return str(value)

    raise KeyError(f"Could not find query text field in example keys: {list(example.keys())}")


def get_code_text(example: Dict[str, Any], use_raw: bool = True) -> str:
    if use_raw and "code" in example and example["code"]:
        return str(example["code"])

    for key in ["code_tokens", "function_tokens"]:
        if key in example and example[key]:
            value = example[key]
            if isinstance(value, list):
                return " ".join(map(str, value))
            return str(value)

    if "code" in example and example["code"]:
        return str(example["code"])

    raise KeyError(f"Could not find code field in example keys: {list(example.keys())}")


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index]


def text2code_to_tensor(text2code, n_queries: int) -> torch.Tensor:
    if isinstance(text2code, torch.Tensor):
        return text2code.long().cpu()
    if isinstance(text2code, dict):
        return torch.tensor([text2code[i] for i in range(n_queries)], dtype=torch.long)
    return torch.tensor(list(text2code), dtype=torch.long)


def load_nomic_model(model_path: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def encode_texts(
    model,
    tokenizer,
    data_loader,
    device: torch.device,
    max_length: int,
    prefix: str,
    desc: str,
) -> torch.Tensor:
    embeds = []
    for texts in tqdm(data_loader, total=len(data_loader), desc=desc):
        if prefix:
            texts = [prefix + text for text in texts]
        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        hidden = model(**batch).last_hidden_state
        last_pos = batch["attention_mask"].sum(dim=1) - 1
        batch_idx = torch.arange(hidden.size(0), device=device)
        pooled = hidden[batch_idx, last_pos]
        pooled = F.normalize(pooled.float(), dim=1)
        embeds.append(pooled.cpu())

        del batch, hidden, pooled
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(embeds, dim=0)


@torch.no_grad()
def encode_texts_to_npy(
    model,
    tokenizer,
    data_loader,
    device: torch.device,
    max_length: int,
    prefix: str,
    output_path: Path,
    embedding_dtype: str,
    desc: str,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    dtype = export_dtype(embedding_dtype)
    total_rows = len(data_loader.dataset)
    mmap = None
    offset = 0

    for texts in tqdm(data_loader, total=len(data_loader), desc=desc):
        if prefix:
            texts = [prefix + text for text in texts]
        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        hidden = model(**batch).last_hidden_state
        last_pos = batch["attention_mask"].sum(dim=1) - 1
        batch_idx = torch.arange(hidden.size(0), device=device)
        pooled = hidden[batch_idx, last_pos]
        pooled = F.normalize(pooled.float(), dim=1)
        pooled_np = pooled.cpu().numpy().astype(dtype, copy=False)

        if mmap is None:
            mmap = np.lib.format.open_memmap(
                tmp_path,
                mode="w+",
                dtype=dtype,
                shape=(total_rows, pooled_np.shape[1]),
            )

        end = offset + pooled_np.shape[0]
        mmap[offset:end] = pooled_np
        offset = end

        del batch, hidden, pooled, pooled_np
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if mmap is None:
        raise ValueError(f"No texts were encoded for {desc}")
    if offset != total_rows:
        raise RuntimeError(f"Encoded {offset} rows for {desc}, expected {total_rows}")

    mmap.flush()
    del mmap
    tmp_path.replace(output_path)
    print(f"Saved {desc} cache to: {output_path}")


@torch.no_grad()
def compute_retrieval_metrics(
    query_embeds: torch.Tensor,
    code_embeds: torch.Tensor,
    gt_indices: torch.Tensor,
    device: torch.device,
    query_chunk_size: int,
    code_chunk_size: int,
) -> Dict[str, float]:
    query_embeds = F.normalize(query_embeds, dim=1)
    code_embeds = F.normalize(code_embeds, dim=1)

    n_queries = query_embeds.size(0)
    n_codes = code_embeds.size(0)
    ranks = []
    ndcgs = []

    for q_start in tqdm(range(0, n_queries, query_chunk_size), desc="Retrieval metrics"):
        q_end = min(q_start + query_chunk_size, n_queries)
        q_chunk = query_embeds[q_start:q_end].to(device)
        q_len = q_end - q_start
        chunk_scores = torch.empty((q_len, n_codes), dtype=torch.float32, device=device)

        for c_start in range(0, n_codes, code_chunk_size):
            c_end = min(c_start + code_chunk_size, n_codes)
            c_chunk = code_embeds[c_start:c_end].to(device)
            chunk_scores[:, c_start:c_end] = q_chunk @ c_chunk.t()
            del c_chunk

        gt_chunk = gt_indices[q_start:q_end].to(device)
        gt_scores = chunk_scores.gather(1, gt_chunk.unsqueeze(1)).squeeze(1)
        rank_chunk = (chunk_scores > gt_scores.unsqueeze(1)).sum(dim=1).cpu().numpy()
        ranks.extend(rank_chunk.tolist())

        for rank in rank_chunk:
            rel = [0] * 10
            if rank < 10:
                rel[int(rank)] = 1
            ndcgs.append(calculate_ndcg(rel, k=10))

        del q_chunk, chunk_scores, gt_chunk, gt_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ranks_np = np.array(ranks)
    return {
        "r1": round(100.0 * float(np.mean(ranks_np < 1)), 2),
        "r5": round(100.0 * float(np.mean(ranks_np < 5)), 2),
        "r10": round(100.0 * float(np.mean(ranks_np < 10)), 2),
        "mrr": round(100.0 * float(np.mean(1.0 / (ranks_np + 1))), 2),
        "ndcg@10": round(100.0 * float(np.mean(ndcgs)), 2),
    }


@torch.no_grad()
def compute_retrieval_metrics_from_cache(
    query_embeds,
    code_embeds,
    gt_indices: torch.Tensor,
    device: torch.device,
    query_chunk_size: int,
    code_chunk_size: int,
    rank_output: str = "",
) -> Dict[str, float]:
    n_queries = query_embeds.shape[0]
    n_codes = code_embeds.shape[0]
    ranks = []
    ndcgs = []

    for q_start in tqdm(range(0, n_queries, query_chunk_size), desc="Retrieval metrics"):
        q_end = min(q_start + query_chunk_size, n_queries)
        q_chunk = torch.from_numpy(np.asarray(query_embeds[q_start:q_end])).to(device).float()
        q_chunk = F.normalize(q_chunk, dim=1)
        q_len = q_end - q_start
        chunk_scores = torch.empty((q_len, n_codes), dtype=torch.float32, device=device)

        for c_start in range(0, n_codes, code_chunk_size):
            c_end = min(c_start + code_chunk_size, n_codes)
            c_chunk = torch.from_numpy(np.asarray(code_embeds[c_start:c_end])).to(device).float()
            c_chunk = F.normalize(c_chunk, dim=1)
            chunk_scores[:, c_start:c_end] = q_chunk @ c_chunk.t()
            del c_chunk

        gt_chunk = gt_indices[q_start:q_end].to(device)
        gt_scores = chunk_scores.gather(1, gt_chunk.unsqueeze(1)).squeeze(1)
        rank_chunk = (chunk_scores > gt_scores.unsqueeze(1)).sum(dim=1).cpu().numpy()
        ranks.extend(rank_chunk.tolist())

        for rank in rank_chunk:
            rel = [0] * 10
            if rank < 10:
                rel[int(rank)] = 1
            ndcgs.append(calculate_ndcg(rel, k=10))

        del q_chunk, chunk_scores, gt_chunk, gt_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ranks_np = np.array(ranks)
    if rank_output:
        rank_path = Path(rank_output)
        rank_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(rank_path, ranks_np.astype(np.int32))
        print(f"Saved ranks to: {rank_path}")
    return {
        "r1": round(100.0 * float(np.mean(ranks_np < 1)), 2),
        "r5": round(100.0 * float(np.mean(ranks_np < 5)), 2),
        "r10": round(100.0 * float(np.mean(ranks_np < 10)), 2),
        "mrr": round(100.0 * float(np.mean(1.0 / (ranks_np + 1))), 2),
        "ndcg@10": round(100.0 * float(np.mean(ndcgs)), 2),
    }


def eval_language(
    model,
    tokenizer,
    language: str,
    split: str,
    data_root: str,
    device: torch.device,
    max_length: int,
    query_prefix: str,
    code_prefix: str,
    text_batch_size: int,
    code_batch_size: int,
    num_workers: int,
    query_chunk_size: int,
    code_chunk_size: int,
    use_raw_jsonl: bool,
    export_coderank_cache_dir: str,
    export_embedding_dtype: str,
    model_path_for_cache: str,
    export_only: bool,
    reuse_export_cache: bool,
    score_from_cache: bool,
    rank_output_dir: str,
) -> Dict[str, float]:
    print(f"\n--- Nomic embedding evaluating {language} / {split} ---")
    data_root_path = Path(data_root)
    eval_examples = read_jsonl(data_root_path / language / f"{split}.jsonl")
    codebase_examples = read_jsonl(data_root_path / language / "codebase.jsonl")
    print(f"Read {len(eval_examples)} data from {data_root_path / language / f'{split}.jsonl'}")
    print(f"Read {len(codebase_examples)} data from {data_root_path / language / 'codebase.jsonl'}")

    queries = [get_query_text(example, use_raw=use_raw_jsonl) for example in eval_examples]
    codes = [get_code_text(example, use_raw=use_raw_jsonl) for example in codebase_examples]

    eval_loader = torch.utils.data.DataLoader(
        TextDataset(queries),
        batch_size=text_batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )
    code_loader = torch.utils.data.DataLoader(
        TextDataset(codes),
        batch_size=code_batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    if score_from_cache:
        if not export_coderank_cache_dir:
            raise ValueError("--score-from-cache requires --export-coderank-cache-dir")

        cache = coderank_cache_paths(
            cache_dir=export_coderank_cache_dir,
            language=language,
            split=split,
            model_name=model_path_for_cache,
            max_seq_length=max_length,
            backend="nomic_transformers",
            query_prefix=query_prefix,
            code_prefix=code_prefix,
        )
        query_embeds = load_cache_array(cache["query"], len(queries), "Nomic query")
        code_embeds = load_cache_array(cache["code"], len(codes), "Nomic code")

        url_to_code = {example["url"]: idx for idx, example in enumerate(codebase_examples)}
        gt_indices = torch.tensor([url_to_code[example["url"]] for example in eval_examples], dtype=torch.long)
        result = compute_retrieval_metrics_from_cache(
            query_embeds=query_embeds,
            code_embeds=code_embeds,
            gt_indices=gt_indices,
            device=device,
            query_chunk_size=query_chunk_size,
            code_chunk_size=code_chunk_size,
            rank_output=(
                str(Path(rank_output_dir) / f"{language}_{split}_nomic_direct.ranks.npy")
                if rank_output_dir
                else ""
            ),
        )
        print(f"  {language}: {result}")
        return result

    if export_only:
        if not export_coderank_cache_dir:
            raise ValueError("--export-only requires --export-coderank-cache-dir")

        cache = coderank_cache_paths(
            cache_dir=export_coderank_cache_dir,
            language=language,
            split=split,
            model_name=model_path_for_cache,
            max_seq_length=max_length,
            backend="nomic_transformers",
            query_prefix=query_prefix,
            code_prefix=code_prefix,
        )
        if not (reuse_export_cache and cache_matches(cache["query"], len(queries), "Nomic query")):
            encode_texts_to_npy(
                model=model,
                tokenizer=tokenizer,
                data_loader=eval_loader,
                device=device,
                max_length=max_length,
                prefix=query_prefix,
                output_path=cache["query"],
                embedding_dtype=export_embedding_dtype,
                desc="Text feats",
            )
        if not (reuse_export_cache and cache_matches(cache["code"], len(codes), "Nomic code")):
            encode_texts_to_npy(
                model=model,
                tokenizer=tokenizer,
                data_loader=code_loader,
                device=device,
                max_length=max_length,
                prefix=code_prefix,
                output_path=cache["code"],
                embedding_dtype=export_embedding_dtype,
                desc="Code feats",
            )
        print(f"  {language}: export-only mode, skipped retrieval metrics.")
        return {"r1": 0.0, "r5": 0.0, "r10": 0.0, "mrr": 0.0, "ndcg@10": 0.0}

    url_to_code = {example["url"]: idx for idx, example in enumerate(codebase_examples)}
    gt_indices = torch.tensor([url_to_code[example["url"]] for example in eval_examples], dtype=torch.long)

    query_embeds = encode_texts(
        model,
        tokenizer,
        eval_loader,
        device,
        max_length=max_length,
        prefix=query_prefix,
        desc="Text feats",
    )
    code_embeds = encode_texts(
        model,
        tokenizer,
        code_loader,
        device,
        max_length=max_length,
        prefix=code_prefix,
        desc="Code feats",
    )
    result = compute_retrieval_metrics(
        query_embeds=query_embeds,
        code_embeds=code_embeds,
        gt_indices=gt_indices,
        device=device,
        query_chunk_size=query_chunk_size,
        code_chunk_size=code_chunk_size,
    )
    save_embedding_cache(
        query_embeds=query_embeds,
        code_embeds=code_embeds,
        language=language,
        split=split,
        model_name=model_path_for_cache,
        max_seq_length=max_length,
        query_prefix=query_prefix,
        code_prefix=code_prefix,
        cache_dir=export_coderank_cache_dir,
        embedding_dtype=export_embedding_dtype,
    )
    print(f"  {language}: {result}")
    return result


def average_metric_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    return {
        "r1": round(float(np.mean([r["r1"] for r in results])), 2),
        "r5": round(float(np.mean([r["r5"] for r in results])), 2),
        "r10": round(float(np.mean([r["r10"] for r in results])), 2),
        "mrr": round(float(np.mean([r["mrr"] for r in results])), 2),
        "ndcg@10": round(float(np.mean([r["ndcg@10"] for r in results])), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate nomic-embed-code as a first-stage text2code retriever.")
    parser.add_argument("--model-path", type=str, default="/root/autodl-tmp/hf_models/nomic-embed-code")
    parser.add_argument("--languages", type=str, nargs="+", default=["javascript"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--experiment-name", type=str, default="nomic_embed_eval")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--query-prefix", type=str, default="Represent this query for searching relevant code: ")
    parser.add_argument("--code-prefix", type=str, default="")
    parser.add_argument("--text-batch-size", type=int, default=4)
    parser.add_argument("--code-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument("--code-chunk-size", type=int, default=4096)
    parser.add_argument("--use-tokenized-fields", action="store_true")
    parser.add_argument(
        "--export-coderank-cache-dir",
        type=str,
        default="",
        help="Save embeddings in the cache format consumed by eval_reranker.py --candidate-source dense_coderank.",
    )
    parser.add_argument(
        "--export-embedding-dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Dtype used when exporting embeddings for reranker candidate cache.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Stream embeddings directly to cache and skip retrieval metrics. Use this for large codebases such as PHP.",
    )
    parser.add_argument(
        "--reuse-export-cache",
        action="store_true",
        help="When --export-only is set, skip query/code encoding if a matching cache already exists.",
    )
    parser.add_argument(
        "--score-from-cache",
        action="store_true",
        help="Load exported Nomic embeddings from cache and compute retrieval metrics without loading the Nomic model.",
    )
    parser.add_argument(
        "--rank-output-dir",
        type=str,
        default="",
        help="Optional directory for saving per-query 0-based rank arrays.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for nomic embedding evaluation.")
    device = torch.device("cuda")

    model = None
    tokenizer = None
    if not args.score_from_cache:
        print(f"Loading nomic embedding model from: {args.model_path}")
        model, tokenizer = load_nomic_model(args.model_path, device)
        print(f"Model device: {next(model.parameters()).device}")

    all_results = {}
    for lang in args.languages:
        all_results[lang] = eval_language(
            model=model,
            tokenizer=tokenizer,
            language=lang,
            split=args.split,
            data_root=args.data_root,
            device=device,
            max_length=args.max_length,
            query_prefix=args.query_prefix,
            code_prefix=args.code_prefix,
            text_batch_size=args.text_batch_size,
            code_batch_size=args.code_batch_size,
            num_workers=args.num_workers,
            query_chunk_size=args.query_chunk_size,
            code_chunk_size=args.code_chunk_size,
            use_raw_jsonl=not args.use_tokenized_fields,
            export_coderank_cache_dir=args.export_coderank_cache_dir,
            export_embedding_dtype=args.export_embedding_dtype,
            model_path_for_cache=args.model_path,
            export_only=args.export_only,
            reuse_export_cache=args.reuse_export_cache,
            score_from_cache=args.score_from_cache,
            rank_output_dir=args.rank_output_dir,
        )

    all_results["average"] = average_metric_results(list(all_results.values()))
    print("\n" + "=" * 60)
    print(f"Average: {all_results['average']}")
    print("=" * 60)

    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"{args.experiment_name}_eval_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"JSON results saved to {output_path}")


if __name__ == "__main__":
    main()
