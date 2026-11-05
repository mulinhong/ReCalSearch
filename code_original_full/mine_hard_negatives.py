"""Hard negative mining for text2code retrieval.

Memory-optimized version that uses:
  - np.memmap for embeddings (disk-backed, not RAM)
  - Two-level chunked dense similarity (query chunks x code chunks)
  - Phased processing to avoid holding all data structures at once
  - Streaming output to avoid building full enriched records in memory
"""

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer

from train_utils import LastTokenPooling, build_pooling
from config import DATA_ROOT, LANGUAGES, resolve_model_path

EmbeddingStore = Union[torch.Tensor, np.ndarray, np.memmap]


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def tokens_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(t) for t in value)
    raise TypeError(f"Unsupported: {type(value)}")


def extract_text_code(item: Dict[str, Any]) -> Tuple[str, str]:
    for text_key, code_key in [
        ("docstring_tokens", "code_tokens"),
        ("doc", "code_tokens"),
        ("text", "code"),
        ("docstring", "code"),
    ]:
        if text_key in item and code_key in item:
            return tokens_to_text(item[text_key]), tokens_to_text(item[code_key])
    raise KeyError(f"Cannot extract text/code from keys: {list(item.keys())}")


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).expand(hidden.size()).float()
    return torch.sum(hidden * m, dim=1) / torch.clamp(m.sum(dim=1), min=1e-6)


def _name_suffix(output_suffix: str) -> str:
    output_suffix = output_suffix.strip()
    return f"_{output_suffix}" if output_suffix else ""


def build_negatives_output_path(data_dir: Path, mode: str, output_suffix: str = "", use_fn_filter: bool = False) -> Path:
    filename = f"train_{mode}"
    if use_fn_filter:
        filename += "_filtered"
    filename += _name_suffix(output_suffix)
    return data_dir / f"{filename}.jsonl"


def resolve_existing_component_path(data_dir: Path, mode: str, output_suffix: str = "") -> Optional[Path]:
    """Return an existing component file for the same suffix only.

    Important: when output_suffix is provided, we do NOT fall back to unsuffixed files.
    This prevents hybrid from silently reusing negatives mined under another config.
    """
    candidates = [
        build_negatives_output_path(data_dir, mode, output_suffix, use_fn_filter=False),
        build_negatives_output_path(data_dir, mode, output_suffix, use_fn_filter=True),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def write_negative_metadata(output_path: Path, meta: Dict[str, Any]) -> None:
    meta_path = output_path.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def build_negative_metadata(
    mode: str,
    model_name: str,
    adapter_path: Optional[str],
    pooling_mode: str,
    pooling_hidden_dim: int,
    query_prefix: str,
    code_prefix: str,
    max_length: int,
    output_suffix: str,
    use_fn_filter: bool,
    token_overlap_threshold: float,
    min_edit_distance: int,
) -> Dict[str, Any]:
    return {
        "mode": mode,
        "model_name": model_name,
        "adapter_path": adapter_path,
        "pooling_mode": pooling_mode,
        "pooling_hidden_dim": pooling_hidden_dim,
        "query_prefix": query_prefix,
        "code_prefix": code_prefix,
        "max_length": max_length,
        "output_suffix": output_suffix,
        "use_fn_filter": use_fn_filter,
        "token_overlap_threshold": token_overlap_threshold,
        "min_edit_distance": min_edit_distance,
    }


def read_adapter_mining_config(adapter_path: Optional[str]) -> Dict[str, Any]:
    if adapter_path is None:
        return {}

    adapter_dir = Path(adapter_path)
    config_path = adapter_dir / "pooling_config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    meta_path = adapter_dir / "experiment_meta.txt"
    if not meta_path.exists():
        return {}

    config: Dict[str, Any] = {}
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.strip().split(":", 1)
            key = key.strip()
            value = value.strip()
            if key == "pooling":
                config["pooling_mode"] = value
            elif key == "query_prefix":
                config["query_prefix"] = value.strip("'\"")
            elif key == "code_prefix":
                config["code_prefix"] = value.strip("'\"")
    return config


def resolve_mining_options(
    adapter_path: Optional[str],
    pooling_mode: str,
    query_prefix: Optional[str],
    code_prefix: Optional[str],
    pooling_hidden_dim: int,
) -> Tuple[str, str, str, int]:
    adapter_config = read_adapter_mining_config(adapter_path)
    if pooling_mode == "auto":
        pooling_mode = adapter_config.get("pooling_mode", "mean")
    if query_prefix is None:
        query_prefix = adapter_config.get("query_prefix", "")
    if code_prefix is None:
        code_prefix = adapter_config.get("code_prefix", "")
    pooling_hidden_dim = int(adapter_config.get("pooling_hidden_dim", pooling_hidden_dim))

    if pooling_mode == "query_aware":
        raise ValueError(
            "query_aware pooling makes code embeddings query-dependent, so it is not compatible "
            "with this offline dense miner. Mine with mean/last_token/attention or add a separate "
            "query-aware reranking miner."
        )
    return pooling_mode, query_prefix, code_prefix, pooling_hidden_dim


def load_pooling_module(
    adapter_path: Optional[str],
    pooling_mode: str,
    pooling_hidden_dim: int,
    device: torch.device,
) -> torch.nn.Module:
    pooling = build_pooling(pooling_mode, pooling_hidden_dim)
    state_path = Path(adapter_path) / "pooling_head.pt" if adapter_path is not None else None

    if pooling_mode == "attention":
        if state_path is None or not state_path.exists():
            raise FileNotFoundError(
                f"attention pooling requires a trained pooling_head.pt, but it was not found at {state_path}"
            )
        pooling.load_state_dict(torch.load(state_path, map_location="cpu"))

    pooling.to(device)
    pooling.eval()
    return pooling


@torch.no_grad()
def encode_texts_to_memmap(
    model: RobertaModel,
    tokenizer: RobertaTokenizer,
    texts_iter,
    n_texts: int,
    device: torch.device,
    output_path: Path,
    batch_size: int,
    max_length: int,
    desc: str,
    pooling_mode: str = "mean",
    pooling_module: Optional[torch.nn.Module] = None,
) -> np.memmap:
    """Encode texts to a disk-backed memmap file, processing from an iterator."""
    hidden_size = int(model.config.hidden_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings = np.memmap(output_path, dtype=np.float32, mode="w+", shape=(n_texts, hidden_size))

    batch_buf: List[str] = []
    write_pos = 0
    total_batches = (n_texts + batch_size - 1) // batch_size
    pbar = tqdm(total=total_batches, desc=desc)

    for text in texts_iter:
        batch_buf.append(text)
        if len(batch_buf) >= batch_size:
            batch = tokenizer(batch_buf, truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(device)
            outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])[0]
            pooled_raw = (
                pooling_module(outputs, batch["attention_mask"])
                if pooling_module is not None
                else LastTokenPooling()(outputs, batch["attention_mask"])
                if pooling_mode == "last_token"
                else mean_pool(outputs, batch["attention_mask"])
            )
            pooled = F.normalize(pooled_raw, p=2, dim=1)
            embeddings[write_pos : write_pos + len(batch_buf)] = pooled.cpu().numpy().astype(np.float32, copy=False)
            write_pos += len(batch_buf)
            batch_buf = []
            del batch, outputs, pooled_raw, pooled
            if device.type == "cuda":
                torch.cuda.empty_cache()
            pbar.update(1)

    if batch_buf:
        batch = tokenizer(batch_buf, truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(device)
        outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])[0]
        pooled_raw = (
            pooling_module(outputs, batch["attention_mask"])
            if pooling_module is not None
            else LastTokenPooling()(outputs, batch["attention_mask"])
            if pooling_mode == "last_token"
            else mean_pool(outputs, batch["attention_mask"])
        )
        pooled = F.normalize(pooled_raw, p=2, dim=1)
        embeddings[write_pos : write_pos + len(batch_buf)] = pooled.cpu().numpy().astype(np.float32, copy=False)
        del batch, outputs, pooled_raw, pooled
        if device.type == "cuda":
            torch.cuda.empty_cache()
        pbar.update(1)

    pbar.close()
    embeddings.flush()
    return embeddings


def mine_bm25_negatives(
    train_path: Path,
    num_negatives: int,
    max_corpus_size: int = 20000,
) -> List[List[int]]:
    """Mine lexical hard negatives. Currently implemented with TF-IDF."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts: List[str] = []
    codes: List[str] = []
    for record in iter_jsonl(train_path):
        t, c = extract_text_code(record)
        texts.append(t)
        codes.append(c)

    n = len(codes)
    max_features = 50000 if n <= 200000 else 20000
    c_chunk = min(max_corpus_size, n)

    print(f"  Building TF-IDF vectors for {n} codes (max_features={max_features})...")
    vectorizer = TfidfVectorizer(max_features=max_features, sublinear_tf=True, dtype=np.float32)
    code_tfidf = vectorizer.fit_transform(codes)
    del codes
    gc.collect()

    query_tfidf = vectorizer.transform(texts)
    del texts, vectorizer
    gc.collect()

    n_queries = query_tfidf.shape[0]
    n_codes = code_tfidf.shape[0]
    q_chunk = 512
    all_top_indices = [[] for _ in range(n_queries)]
    all_top_scores = [[] for _ in range(n_queries)]

    print(f"  Computing similarities (q_chunk={q_chunk}, c_chunk={c_chunk})...")
    for q_start in tqdm(range(0, n_queries, q_chunk), desc="  Lexical mining"):
        q_end = min(q_start + q_chunk, n_queries)
        q_block = query_tfidf[q_start:q_end]
        for c_start in range(0, n_codes, c_chunk):
            c_end = min(c_start + c_chunk, n_codes)
            sim = (q_block @ code_tfidf[c_start:c_end].T).toarray()
            for local_i in range(sim.shape[0]):
                global_i = q_start + local_i
                scores = sim[local_i]
                if c_start <= global_i < c_end:
                    scores[global_i - c_start] = -1.0
                k = min(max(num_negatives * 2, 1), max(len(scores) - 1, 1))
                if k <= 0:
                    continue
                local_top = np.argpartition(-scores, k - 1)[:k]
                for idx in local_top:
                    all_top_indices[global_i].append(idx + c_start)
                    all_top_scores[global_i].append(float(scores[idx]))
            del sim
        gc.collect()

    del code_tfidf, query_tfidf
    gc.collect()

    result: List[List[int]] = []
    for i in range(n_queries):
        order = np.argsort([-s for s in all_top_scores[i]])
        seen = set()
        final = []
        for o in order:
            idx = all_top_indices[i][o]
            if idx == i:
                continue
            if idx not in seen:
                seen.add(idx)
                final.append(idx)
            if len(final) >= num_negatives:
                break
        result.append(final)

    del all_top_indices, all_top_scores
    gc.collect()
    return result


def _embedding_slice_to_tensor(embeddings: EmbeddingStore, start: int, end: int, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(embeddings[start:end], dtype=np.float32)).to(device)


def mine_dense_negatives(
    query_embeddings: EmbeddingStore,
    code_embeddings: EmbeddingStore,
    num_negatives: int,
    query_chunk_size: int = 128,
    code_chunk_size: int = 4096,
    device: Optional[torch.device] = None,
) -> List[List[int]]:
    """Two-level chunked dense mining. Never materializes full sim matrix."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_queries = int(query_embeddings.shape[0])
    n_codes = int(code_embeddings.shape[0])
    k = min(num_negatives, max(n_codes - 1, 1))
    all_negatives: List[List[int]] = []

    for q_start in tqdm(range(0, n_queries, query_chunk_size), desc="Dense mining"):
        q_end = min(q_start + query_chunk_size, n_queries)
        q_len = q_end - q_start
        chunk_q = _embedding_slice_to_tensor(query_embeddings, q_start, q_end, device)
        best_scores = torch.full((q_len, k), -float("inf"), device=device)
        best_indices = torch.full((q_len, k), -1, dtype=torch.long, device=device)

        for c_start in range(0, n_codes, code_chunk_size):
            c_end = min(c_start + code_chunk_size, n_codes)
            chunk_c = _embedding_slice_to_tensor(code_embeddings, c_start, c_end, device)
            sim_chunk = chunk_q @ chunk_c.T

            for local_i in range(q_len):
                pos = q_start + local_i
                if c_start <= pos < c_end:
                    sim_chunk[local_i, pos - c_start] = -float("inf")

            candidate_indices = torch.arange(c_start, c_end, device=device, dtype=torch.long).unsqueeze(0).expand(q_len, -1)
            merged_scores = torch.cat([best_scores, sim_chunk], dim=1)
            merged_indices = torch.cat([best_indices, candidate_indices], dim=1)
            best_scores, best_pos = torch.topk(merged_scores, k=k, dim=1)
            best_indices = merged_indices.gather(1, best_pos)
            del chunk_c, sim_chunk, candidate_indices, merged_scores, merged_indices, best_pos
            if device.type == "cuda":
                torch.cuda.empty_cache()

        for row in best_indices.cpu().tolist():
            all_negatives.append([int(idx) for idx in row if idx >= 0])
        del chunk_q, best_scores, best_indices
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return all_negatives


def _token_overlap(code_a: str, code_b: str) -> float:
    tokens_a = set(code_a.lower().split())
    tokens_b = set(code_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / min(len(tokens_a), len(tokens_b))


def _edit_distance_approx(a: str, b: str, max_len: int = 500) -> int:
    a = a[:max_len]
    b = b[:max_len]
    if len(a) > len(b):
        a, b = b, a
    distances = list(range(len(a) + 1))
    for j, char_b in enumerate(b):
        new_distances = [j + 1]
        for i, char_a in enumerate(a):
            if char_a == char_b:
                new_distances.append(distances[i])
            else:
                new_distances.append(1 + min(distances[i], distances[i + 1], new_distances[-1]))
        distances = new_distances
    return distances[-1]


def filter_false_negatives(
    positive_code: str,
    candidate_codes: List[str],
    candidate_indices: List[int],
    token_overlap_threshold: float = 0.8,
    min_edit_distance: int = 10,
) -> Tuple[List[str], List[int]]:
    filtered_codes = []
    filtered_indices = []
    for code, idx in zip(candidate_codes, candidate_indices):
        if _token_overlap(positive_code, code) >= token_overlap_threshold:
            continue
        if _edit_distance_approx(positive_code, code) < min_edit_distance:
            continue
        filtered_codes.append(code)
        filtered_indices.append(idx)
    return filtered_codes, filtered_indices


def load_hard_negative_indices(path: Path, limit: int) -> List[List[int]]:
    return [[int(x) for x in r.get("hard_negative_indices", [])[:limit]] for r in iter_jsonl(path)]


def merge_negative_indices(first: List[List[int]], second: List[List[int]], limit: int) -> List[List[int]]:
    if len(first) != len(second):
        raise ValueError(f"Cannot merge negative lists with different lengths: {len(first)} vs {len(second)}")
    merged = []
    for left, right in zip(first, second):
        seen = set()
        cur = []
        for idx in left + right:
            if idx not in seen:
                seen.add(idx)
                cur.append(int(idx))
            if len(cur) >= limit:
                break
        merged.append(cur)
    return merged


def _load_codes_only(train_path: Path) -> List[str]:
    codes = []
    for record in iter_jsonl(train_path):
        _, code = extract_text_code(record)
        codes.append(code)
    return codes


def stream_enriched_records(
    train_path: Path,
    output_path: Path,
    codes: List[str],
    neg_indices: List[List[int]],
    num_hard_negatives: int,
    use_fn_filter: bool,
    token_overlap_threshold: float,
    min_edit_distance: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with train_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        row_idx = 0
        for line in src:
            line = line.strip()
            if not line:
                continue
            if row_idx >= len(neg_indices):
                raise ValueError(f"Missing negative indices for row {row_idx} in {train_path}")
            record = json.loads(line)
            positive_code = extract_text_code(record)[1]
            candidate_indices = [int(x) for x in neg_indices[row_idx]]
            candidate_codes = [codes[j] for j in candidate_indices]
            if use_fn_filter:
                candidate_codes, candidate_indices = filter_false_negatives(
                    positive_code,
                    candidate_codes,
                    candidate_indices,
                    token_overlap_threshold,
                    min_edit_distance,
                )
            enriched = dict(record)
            enriched["hard_negative_codes"] = candidate_codes[:num_hard_negatives]
            enriched["hard_negative_indices"] = [int(x) for x in candidate_indices[:num_hard_negatives]]
            dst.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            row_idx += 1


def mine_and_save(
    data_root: str,
    language: str,
    model_name: str,
    adapter_path: Optional[str] = None,
    mode: str = "dense",
    num_hard_negatives: int = 5,
    use_fn_filter: bool = False,
    token_overlap_threshold: float = 0.8,
    min_edit_distance: int = 10,
    batch_size: int = 64,
    max_length: int = 512,
    output_suffix: str = "",
    max_corpus_size: int = 100000,
    dense_query_chunk_size: int = 128,
    dense_code_chunk_size: int = 4096,
    pooling_mode: str = "auto",
    query_prefix: Optional[str] = None,
    code_prefix: Optional[str] = None,
    pooling_hidden_dim: int = 768,
) -> Path:
    pooling_mode, query_prefix, code_prefix, pooling_hidden_dim = resolve_mining_options(
        adapter_path, pooling_mode, query_prefix, code_prefix, pooling_hidden_dim
    )

    data_dir = Path(data_root) / language
    train_path = data_dir / "train.jsonl"
    n_records = count_lines(train_path)
    print(f"Loaded {n_records} training records for {language}")

    dense_component_path = resolve_existing_component_path(data_dir, "dense", output_suffix)
    bm25_component_path = resolve_existing_component_path(data_dir, "bm25", output_suffix)

    query_emb = None
    code_emb = None
    cache_paths: List[Path] = []

    # ---- Phase 1: encode to memmap only when really needed ----
    need_dense_from_scratch = mode == "dense" or (mode == "hybrid" and dense_component_path is None)
    if need_dense_from_scratch:
        model_path = resolve_model_path(model_name)
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaModel.from_pretrained(model_path)
        if adapter_path:
            print(f"  Loading mining adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        pooling_module = load_pooling_module(adapter_path, pooling_mode, pooling_hidden_dim, device)
        cache_dir = data_dir / ".mining_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        query_cache = cache_dir / f"{language}_query_emb.f32"
        code_cache = cache_dir / f"{language}_code_emb.f32"
        cache_paths.extend([query_cache, code_cache])

        print("  Phase 1a: encoding queries to disk...")
        def _query_iter():
            for r in iter_jsonl(train_path):
                yield query_prefix + extract_text_code(r)[0]

        query_emb = encode_texts_to_memmap(
            model,
            tokenizer,
            _query_iter(),
            n_records,
            device,
            query_cache,
            batch_size,
            max_length,
            "Encoding queries",
            pooling_mode,
            pooling_module,
        )

        print("  Phase 1b: encoding codes to disk...")
        def _code_iter():
            for r in iter_jsonl(train_path):
                yield code_prefix + extract_text_code(r)[1]

        code_emb = encode_texts_to_memmap(
            model,
            tokenizer,
            _code_iter(),
            n_records,
            device,
            code_cache,
            batch_size,
            max_length,
            "Encoding codes",
            pooling_mode,
            pooling_module,
        )

        del model, tokenizer, pooling_module
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- Phase 2: mine negatives ----
    if mode == "bm25":
        neg_indices = mine_bm25_negatives(train_path, num_hard_negatives, max_corpus_size)

    elif mode == "dense":
        if query_emb is None or code_emb is None:
            raise RuntimeError("Dense embeddings were not prepared.")
        neg_indices = mine_dense_negatives(
            query_emb,
            code_emb,
            num_hard_negatives,
            dense_query_chunk_size,
            dense_code_chunk_size,
        )
        del query_emb, code_emb
        gc.collect()

    elif mode == "hybrid":
        if bm25_component_path is not None and dense_component_path is not None:
            print(
                "  Hybrid: merging existing matching component files "
                f"{bm25_component_path.name} + {dense_component_path.name}"
            )
            neg_indices = merge_negative_indices(
                load_hard_negative_indices(bm25_component_path, num_hard_negatives),
                load_hard_negative_indices(dense_component_path, num_hard_negatives),
                num_hard_negatives,
            )
        else:
            print("  Hybrid: no complete matching component set found; mining missing parts from scratch")
            if dense_component_path is not None:
                print(f"  Using existing matching dense negatives from {dense_component_path.name}")
                dense_negs = load_hard_negative_indices(dense_component_path, num_hard_negatives)
            else:
                if query_emb is None or code_emb is None:
                    raise RuntimeError("Dense embeddings were not prepared for hybrid mining.")
                n_bm25 = num_hard_negatives // 2
                n_dense = num_hard_negatives - n_bm25
                print("  Phase 2a: dense mining...")
                dense_negs = mine_dense_negatives(
                    query_emb,
                    code_emb,
                    n_dense,
                    dense_query_chunk_size,
                    dense_code_chunk_size,
                )
                del query_emb, code_emb
                gc.collect()

            if bm25_component_path is not None:
                print(f"  Using existing matching lexical negatives from {bm25_component_path.name}")
                bm25_negs = load_hard_negative_indices(bm25_component_path, num_hard_negatives)
            else:
                n_bm25 = num_hard_negatives // 2
                print("  Phase 2b: BM25 mining...")
                bm25_negs = mine_bm25_negatives(train_path, n_bm25, max_corpus_size)

            neg_indices = merge_negative_indices(bm25_negs, dense_negs, num_hard_negatives)
            del bm25_negs, dense_negs
            gc.collect()

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ---- Phase 3: enrich and write output ----
    output_path = build_negatives_output_path(data_dir, mode, output_suffix, use_fn_filter=use_fn_filter)

    print("  Phase 3: writing enriched records...")
    codes = _load_codes_only(train_path)
    stream_enriched_records(
        train_path,
        output_path,
        codes,
        neg_indices,
        num_hard_negatives,
        use_fn_filter,
        token_overlap_threshold,
        min_edit_distance,
    )
    del codes, neg_indices
    gc.collect()

    meta = build_negative_metadata(
        mode=mode,
        model_name=model_name,
        adapter_path=adapter_path,
        pooling_mode=pooling_mode,
        pooling_hidden_dim=pooling_hidden_dim,
        query_prefix=query_prefix,
        code_prefix=code_prefix,
        max_length=max_length,
        output_suffix=output_suffix,
        use_fn_filter=use_fn_filter,
        token_overlap_threshold=token_overlap_threshold,
        min_edit_distance=min_edit_distance,
    )
    write_negative_metadata(output_path, meta)

    print(f"Saved {n_records} records to {output_path}")
    for cache_path in cache_paths:
        if cache_path.exists():
            cache_path.unlink()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine hard negatives for text2code")
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--language", type=str, default=None, help="Single language or 'all'")
    parser.add_argument("--model-name", type=str, default="microsoft/unixcoder-base")
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--mode", type=str, default="dense", choices=["bm25", "dense", "hybrid"])
    parser.add_argument("--num-hard-negatives", type=int, default=5)
    parser.add_argument("--use-fn-filter", action="store_true")
    parser.add_argument("--token-overlap-threshold", type=float, default=0.8)
    parser.add_argument("--min-edit-distance", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--pooling", type=str, default="auto", choices=["auto", "mean", "last_token", "attention", "query_aware"])
    parser.add_argument("--pooling-hidden-dim", type=int, default=768)
    parser.add_argument("--query-prefix", type=str, default=None)
    parser.add_argument("--code-prefix", type=str, default=None)
    parser.add_argument("--output-suffix", type=str, default="")
    parser.add_argument("--max-corpus-size", type=int, default=20000, help="BM25 shard size")
    parser.add_argument("--dense-query-chunk-size", type=int, default=64, help="Dense mining query chunk size")
    parser.add_argument("--dense-code-chunk-size", type=int, default=2048, help="Dense mining code chunk size")
    args = parser.parse_args()

    languages = LANGUAGES if args.language is None or args.language == "all" else [args.language]
    for lang in languages:
        print(f"\n===== Mining {args.mode} hard negatives for {lang} =====")
        mine_and_save(
            data_root=args.data_root,
            language=lang,
            model_name=args.model_name,
            adapter_path=args.adapter_path,
            mode=args.mode,
            num_hard_negatives=args.num_hard_negatives,
            use_fn_filter=args.use_fn_filter,
            token_overlap_threshold=args.token_overlap_threshold,
            min_edit_distance=args.min_edit_distance,
            batch_size=args.batch_size,
            max_length=args.max_length,
            output_suffix=args.output_suffix,
            max_corpus_size=args.max_corpus_size,
            dense_query_chunk_size=args.dense_query_chunk_size,
            dense_code_chunk_size=args.dense_code_chunk_size,
            pooling_mode=args.pooling,
            query_prefix=args.query_prefix,
            code_prefix=args.code_prefix,
            pooling_hidden_dim=args.pooling_hidden_dim,
        )


if __name__ == "__main__":
    main()