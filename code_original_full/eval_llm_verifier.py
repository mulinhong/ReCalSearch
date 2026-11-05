import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DATA_ROOT, RESULTS_DIR
from data_utils import create_dataset
from eval_nomic_embed import coderank_cache_paths, get_code_text, get_query_text, load_cache_array
from revision_utils import enforce_frozen_test_protocol, load_json, save_frozen_config, write_environment_manifest


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def text2code_to_array(text2code, n_queries: int) -> np.ndarray:
    if isinstance(text2code, torch.Tensor):
        return text2code.long().cpu().numpy()
    if isinstance(text2code, dict):
        return np.asarray([text2code[i] for i in range(n_queries)], dtype=np.int64)
    return np.asarray(list(text2code), dtype=np.int64)


def calculate_dcg(relevance_scores, k):
    relevance_scores = np.asarray(relevance_scores[:k])
    return np.sum(relevance_scores / np.log2(np.arange(2, relevance_scores.size + 2)))


def calculate_ndcg(relevance_scores, k):
    dcg = calculate_dcg(relevance_scores, k)
    ideal = sorted(relevance_scores, reverse=True)
    idcg = calculate_dcg(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics_from_ranks(ranks: np.ndarray) -> Dict[str, float]:
    ranks = np.asarray(ranks)
    ndcgs = []
    for rank in ranks:
        rel = [0] * 10
        if rank < 10:
            rel[int(rank)] = 1
        ndcgs.append(calculate_ndcg(rel, k=10))

    reciprocal = 1.0 / (ranks + 1.0)
    r1_hits = (ranks < 1).astype(np.float64)
    r5_hits = (ranks < 5).astype(np.float64)
    r10_hits = (ranks < 10).astype(np.float64)
    return {
        "r1": round(100.0 * float(r1_hits.mean()), 4),
        "r5": round(100.0 * float(r5_hits.mean()), 4),
        "r10": round(100.0 * float(r10_hits.mean()), 4),
        "mrr": round(100.0 * float(reciprocal.mean()), 4),
        "ndcg@10": round(100.0 * float(np.mean(ndcgs)), 4),
        "n_queries": int(ranks.size),
        "_sum_r1": float(r1_hits.sum()),
        "_sum_r5": float(r5_hits.sum()),
        "_sum_r10": float(r10_hits.sum()),
        "_sum_rr": float(reciprocal.sum()),
        "_sum_ndcg10": float(np.sum(ndcgs)),
    }


def normalize_scores(scores: np.ndarray, mode: str) -> np.ndarray:
    scores = scores.astype(np.float32, copy=False)
    if mode == "none":
        return scores
    if mode == "zscore":
        std = float(scores.std())
        if std < 1e-6:
            return scores - float(scores.mean())
        return (scores - float(scores.mean())) / std
    if mode == "minmax":
        score_min = float(scores.min())
        score_max = float(scores.max())
        denom = max(score_max - score_min, 1e-6)
        return (scores - score_min) / denom
    if mode == "rank":
        if scores.size <= 1:
            return np.zeros_like(scores, dtype=np.float32)
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=np.float32)
        ranks[order] = np.arange(scores.size, dtype=np.float32)
        return ranks / float(scores.size - 1)
    if mode == "softmax":
        shifted = scores - float(scores.max())
        exp_scores = np.exp(shifted).astype(np.float32)
        denom = max(float(exp_scores.sum()), 1e-6)
        return exp_scores / denom
    raise ValueError(f"Unknown normalization mode: {mode}")


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


def load_verifier(
    model_path: str,
    base_model_path: str,
    dtype_name: str,
    device: torch.device,
    trust_remote_code: bool,
    device_map: str,
):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[dtype_name]
    tokenizer_path = model_path
    if base_model_path and not (Path(model_path) / "tokenizer_config.json").exists():
        tokenizer_path = base_model_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=trust_remote_code,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"

    load_kwargs = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "use_safetensors": True,
    }
    if device_map == "cuda":
        load_kwargs["device_map"] = {"": device.index or 0}
    elif device_map == "auto":
        load_kwargs["device_map"] = "auto"

    load_model_path = base_model_path if base_model_path else model_path
    model = AutoModelForCausalLM.from_pretrained(load_model_path, **load_kwargs)
    if base_model_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("peft is required when --verifier-base-model is used.") from exc
        model = PeftModel.from_pretrained(model, model_path)
    if device_map == "none":
        model = model.to(device)
    model.eval()
    return model, tokenizer


def single_token_id(tokenizer, candidates: List[str], label: str) -> int:
    for candidate in candidates:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            print(f"  {label} token: {candidate!r} -> {ids[0]}")
            return int(ids[0])
    fallback = tokenizer.encode(candidates[0], add_special_tokens=False)[-1]
    print(f"  [WARN] No single-token {label} candidate; using last token of {candidates[0]!r}: {fallback}")
    return int(fallback)


def build_yesno_prompt(
    query: str,
    code: str,
    max_query_chars: int,
    max_code_chars: int,
) -> str:
    query = query[:max_query_chars]
    code = code[:max_code_chars]
    return (
        "You are a strict code search relevance verifier.\n"
        "Decide whether the code correctly implements or answers the natural language query.\n"
        "Answer with only Yes or No.\n\n"
        f"Query:\n{query}\n\n"
        f"Code:\n{code}\n\n"
        "Answer:"
    )


def build_listwise_prompt(
    query: str,
    candidate_codes: List[str],
    max_query_chars: int,
    max_code_chars: int,
) -> str:
    query = query[:max_query_chars]
    blocks = []
    for idx, code in enumerate(candidate_codes, start=1):
        blocks.append(f"[{idx}]\n{code[:max_code_chars]}")
    candidates = "\n\n".join(blocks)
    return (
        "You are a strict code search reranker.\n"
        "Rank the candidate code snippets by how well they implement the natural language query.\n"
        "Return only candidate numbers in best-to-worst order, separated by spaces.\n\n"
        f"Query:\n{query}\n\n"
        f"Candidates:\n{candidates}\n\n"
        "Ranking:"
    )


def build_pairwise_prompt(
    query: str,
    code_a: str,
    code_b: str,
    max_query_chars: int,
    max_code_chars: int,
) -> str:
    query = query[:max_query_chars]
    per_code_chars = max(128, max_code_chars // 2)
    code_a = code_a[:per_code_chars]
    code_b = code_b[:per_code_chars]
    return (
        "You are a strict code search reranker.\n"
        "Decide which candidate code snippet better implements the natural language query.\n"
        "Answer with only A or B.\n\n"
        f"Query:\n{query}\n\n"
        f"Candidate A:\n{code_a}\n\n"
        f"Candidate B:\n{code_b}\n\n"
        "Better candidate:"
    )


def parse_listwise_order(text: str, num_candidates: int) -> List[int]:
    order = []
    seen = set()
    for match in re.finditer(r"\d+", text):
        idx = int(match.group(0)) - 1
        if 0 <= idx < num_candidates and idx not in seen:
            order.append(idx)
            seen.add(idx)
    for idx in range(num_candidates):
        if idx not in seen:
            order.append(idx)
    return order


@torch.no_grad()
def score_listwise_prompts(
    model,
    tokenizer,
    prompts: List[str],
    num_candidates: int,
    device: torch.device,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
) -> np.ndarray:
    all_scores = []
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in tqdm(range(0, len(prompts), batch_size), desc="Listwise verifier"):
            batch_prompts = prompts[start:start + batch_size]
            enc = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            outputs = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            input_len = enc["input_ids"].shape[1]
            decoded = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
            for text in decoded:
                order = parse_listwise_order(text, num_candidates)
                scores = np.zeros(num_candidates, dtype=np.float32)
                for pos, cand_idx in enumerate(order):
                    scores[cand_idx] = float(num_candidates - pos)
                all_scores.append(scores)

            del enc, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        tokenizer.padding_side = old_padding_side

    return np.stack(all_scores, axis=0).astype(np.float32)


@torch.no_grad()
def score_pairwise_prompts(
    model,
    tokenizer,
    prompts: List[str],
    a_id: int,
    b_id: int,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    scores = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="Pairwise verifier"):
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
        a_scores = next_logits[:, a_id]
        b_scores = next_logits[:, b_id]
        scores.append((a_scores - b_scores).detach().cpu().numpy().astype(np.float32))

        del enc, outputs, next_logits, a_scores, b_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.concatenate(scores, axis=0)


@torch.no_grad()
def score_yesno_prompts(
    model,
    tokenizer,
    prompts: List[str],
    yes_id: int,
    no_id: int,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    scores = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="Verifier scoring"):
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
        yes_scores = next_logits[:, yes_id]
        no_scores = next_logits[:, no_id]
        scores.append((yes_scores - no_scores).detach().cpu().numpy().astype(np.float32))

        del enc, outputs, next_logits, yes_scores, no_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.concatenate(scores, axis=0)


def score_cache_path(
    cache_dir: str,
    language: str,
    split: str,
    verifier_model: str,
    verifier_base_model: str,
    candidate_top_k: int,
    verify_top_k: int,
    max_length: int,
    max_query_chars: int,
    max_code_chars: int,
    query_offset: int,
    max_queries: Optional[int],
    verifier_mode: str,
    max_new_tokens: int,
    candidate_pool_key: str = "dense",
) -> Path:
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    prompt_version = {
        "yesno": "yesno_v1",
        "listwise": "listwise_v1",
        "pairwise": "pairwise_v1",
    }[verifier_mode]
    key = json.dumps(
        {
            "language": language,
            "split": split,
            "verifier_model": verifier_model,
            "verifier_base_model": verifier_base_model,
            "candidate_top_k": candidate_top_k,
            "verify_top_k": verify_top_k,
            "max_length": max_length,
            "max_query_chars": max_query_chars,
            "max_code_chars": max_code_chars,
            "query_offset": query_offset,
            "max_queries": max_queries,
            "verifier_mode": verifier_mode,
            "max_new_tokens": max_new_tokens,
            "prompt": prompt_version,
            "candidate_pool_key": candidate_pool_key,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return root / (
        f"{language}_{split}_{sanitize_name(verifier_model)}_"
        f"cand{candidate_top_k}_verify{verify_top_k}_{digest}.scores.npy"
    )


def rerank_with_verifier(
    candidate_indices: np.ndarray,
    candidate_scores: np.ndarray,
    verifier_scores: np.ndarray,
    gt_indices: np.ndarray,
    base_ranks: np.ndarray,
    verify_top_k: int,
    fusion_alpha: float,
    normalization: str,
) -> np.ndarray:
    ranks = base_ranks.copy()
    n_queries = candidate_indices.shape[0]

    for i in range(n_queries):
        candidates = candidate_indices[i]
        retrieval = candidate_scores[i]
        verified_candidates = candidates[:verify_top_k]
        verified_retrieval = retrieval[:verify_top_k]
        verified_scores = verifier_scores[i, :verify_top_k]

        if fusion_alpha > 0.0:
            final_scores = normalize_scores(verified_scores, normalization) + fusion_alpha * normalize_scores(
                verified_retrieval,
                normalization,
            )
        else:
            final_scores = verified_scores

        verified_order = np.argsort(-final_scores)
        reranked = list(verified_candidates[verified_order])
        if verify_top_k < len(candidates):
            seen = set(int(x) for x in reranked)
            reranked.extend(int(x) for x in candidates[verify_top_k:] if int(x) not in seen)

        matches = np.where(np.asarray(reranked) == int(gt_indices[i]))[0]
        if len(matches) > 0:
            ranks[i] = int(matches[0])

    return ranks


def eval_language(args, language: str, device: torch.device):
    print(f"\n--- LLM verifier reranking {language} / {args.split} ---")
    _, val_dataset, test_dataset, code_dataset = create_dataset(args.data_root, language)
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

    candidate_pool_key = "dense"
    if args.candidate_pool_pattern:
        candidate_pool_path = Path(
            args.candidate_pool_pattern.format(language=language, split=args.split)
        ).expanduser()
        pool = np.load(candidate_pool_path, allow_pickle=False)
        required = [name for name in ("indices", "scores", "base_ranks") if name not in pool]
        if required:
            raise ValueError(f"Candidate pool {candidate_pool_path} is missing arrays: {required}")
        top_indices = np.asarray(pool["indices"], dtype=np.int64)
        top_scores = np.asarray(pool["scores"], dtype=np.float32)
        base_ranks = np.asarray(pool["base_ranks"], dtype=np.int64)
        if top_indices.ndim != 2 or top_scores.shape != top_indices.shape:
            raise ValueError(
                f"Invalid candidate pool shapes: indices={top_indices.shape}, scores={top_scores.shape}"
            )
        if base_ranks.ndim != 1 or len(base_ranks) != len(top_indices):
            raise ValueError(f"Invalid base_ranks shape in {candidate_pool_path}: {base_ranks.shape}")
        expected_rows = len(eval_dataset)
        if len(top_indices) != expected_rows:
            raise ValueError(
                f"Candidate pool has {len(top_indices)} rows, expected full split size {expected_rows}."
            )
        if args.max_queries is not None or args.query_offset:
            end = None if args.max_queries is None else args.query_offset + args.max_queries
            top_indices = top_indices[args.query_offset:end]
            top_scores = top_scores[args.query_offset:end]
            base_ranks = base_ranks[args.query_offset:end]
        if top_indices.shape[1] != args.candidate_top_k:
            raise ValueError(
                f"Candidate pool width is {top_indices.shape[1]}, but --candidate-top-k={args.candidate_top_k}."
            )
        if np.any(top_indices < 0) or np.any(top_indices >= len(codes)):
            raise ValueError(f"Candidate pool contains out-of-range code indices: {candidate_pool_path}")
        candidate_pool_key = args.candidate_pool_tag or candidate_pool_path.stem
        print(f"  loaded candidate action pool: {candidate_pool_path} shape={top_indices.shape}")
    else:
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

    verify_top_k = min(args.verify_top_k, args.candidate_top_k)
    score_cache = None
    if args.score_cache_dir:
        score_cache = score_cache_path(
            cache_dir=args.score_cache_dir,
            language=language,
            split=args.split,
            verifier_model=args.verifier_model,
            verifier_base_model=args.verifier_base_model,
            candidate_top_k=args.candidate_top_k,
            verify_top_k=verify_top_k,
            max_length=args.max_length,
            max_query_chars=args.max_query_chars,
            max_code_chars=args.max_code_chars,
            query_offset=args.query_offset,
            max_queries=args.max_queries,
            verifier_mode=args.verifier_mode,
            max_new_tokens=args.max_new_tokens,
            candidate_pool_key=candidate_pool_key,
        )

    verifier_scores = None
    if score_cache is not None and args.reuse_score_cache and score_cache.exists():
        verifier_scores = np.load(score_cache)
        if verifier_scores.shape != (len(queries), verify_top_k):
            print(f"  ignoring score cache with shape={verifier_scores.shape}, expected={(len(queries), verify_top_k)}")
            verifier_scores = None
        else:
            print(f"  loaded verifier score cache: {score_cache}")

    if verifier_scores is None:
        verifier_model, verifier_tokenizer = load_verifier(
            args.verifier_model,
            base_model_path=args.verifier_base_model,
            dtype_name=args.verifier_dtype,
            device=device,
            trust_remote_code=args.trust_remote_code,
            device_map=args.verifier_device_map,
        )
        if args.verifier_mode == "yesno":
            yes_id = single_token_id(verifier_tokenizer, [" Yes", "Yes", " yes", "yes"], "yes")
            no_id = single_token_id(verifier_tokenizer, [" No", "No", " no", "no"], "no")

            prompts = []
            for i, query in enumerate(queries):
                for code_idx in top_indices[i, :verify_top_k]:
                    prompts.append(
                        build_yesno_prompt(
                            query=query,
                            code=codes[int(code_idx)],
                            max_query_chars=args.max_query_chars,
                            max_code_chars=args.max_code_chars,
                        )
                    )
            flat_scores = score_yesno_prompts(
                model=verifier_model,
                tokenizer=verifier_tokenizer,
                prompts=prompts,
                yes_id=yes_id,
                no_id=no_id,
                device=device,
                batch_size=args.verifier_batch_size,
                max_length=args.max_length,
            )
            verifier_scores = flat_scores.reshape(len(queries), verify_top_k)
        elif args.verifier_mode == "pairwise":
            a_id = single_token_id(verifier_tokenizer, [" A", "A", " a", "a"], "A")
            b_id = single_token_id(verifier_tokenizer, [" B", "B", " b", "b"], "B")

            prompts = []
            pair_meta = []
            for i, query in enumerate(queries):
                topk_codes = [codes[int(code_idx)] for code_idx in top_indices[i, :verify_top_k]]
                for left in range(verify_top_k):
                    for right in range(left + 1, verify_top_k):
                        prompts.append(
                            build_pairwise_prompt(
                                query=query,
                                code_a=topk_codes[left],
                                code_b=topk_codes[right],
                                max_query_chars=args.max_query_chars,
                                max_code_chars=args.max_code_chars,
                            )
                        )
                        pair_meta.append((i, left, right))

            flat_scores = score_pairwise_prompts(
                model=verifier_model,
                tokenizer=verifier_tokenizer,
                prompts=prompts,
                a_id=a_id,
                b_id=b_id,
                device=device,
                batch_size=args.verifier_batch_size,
                max_length=args.max_length,
            )
            verifier_scores = np.zeros((len(queries), verify_top_k), dtype=np.float32)
            for (query_idx, left, right), score in zip(pair_meta, flat_scores):
                verifier_scores[query_idx, left] += score
                verifier_scores[query_idx, right] -= score
        elif args.verifier_mode == "listwise":
            prompts = []
            for i, query in enumerate(queries):
                candidate_codes = [codes[int(code_idx)] for code_idx in top_indices[i, :verify_top_k]]
                prompts.append(
                    build_listwise_prompt(
                        query=query,
                        candidate_codes=candidate_codes,
                        max_query_chars=args.max_query_chars,
                        max_code_chars=args.max_code_chars,
                    )
                )
            verifier_scores = score_listwise_prompts(
                model=verifier_model,
                tokenizer=verifier_tokenizer,
                prompts=prompts,
                num_candidates=verify_top_k,
                device=device,
                batch_size=args.verifier_batch_size,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            raise ValueError(f"Unknown verifier mode: {args.verifier_mode}")

        if score_cache is not None:
            np.save(score_cache, verifier_scores.astype(np.float32))
            print(f"  saved verifier score cache: {score_cache}")

        del verifier_model, verifier_tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = {}
    for alpha in args.fusion_alphas:
        ranks = rerank_with_verifier(
            candidate_indices=top_indices,
            candidate_scores=top_scores,
            verifier_scores=verifier_scores,
            gt_indices=gt_indices,
            base_ranks=base_ranks,
            verify_top_k=verify_top_k,
            fusion_alpha=float(alpha),
            normalization=args.fusion_normalization,
        )
        key = f"alpha={alpha:g}"
        result = compute_metrics_from_ranks(ranks)
        result["fusion_alpha"] = float(alpha)
        result["candidate_top_k"] = args.candidate_top_k
        result["verify_top_k"] = verify_top_k
        result["candidate_pool"] = candidate_pool_key
        results[key] = result
        print(f"  {language} / {key}: {result}")

        if args.rank_output_dir:
            rank_dir = Path(args.rank_output_dir)
            rank_dir.mkdir(parents=True, exist_ok=True)
            rank_path = rank_dir / (
                f"{language}_{args.split}_llm_verifier_"
                f"{sanitize_name(candidate_pool_key)}_cand{args.candidate_top_k}_"
                f"verify{verify_top_k}_{key}.ranks.npy"
            )
            np.save(rank_path, ranks.astype(np.int32))
            print(f"  saved ranks to: {rank_path}")

    return results


def average_metric_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    return {
        "r1": round(float(np.mean([r["r1"] for r in results])), 2),
        "r5": round(float(np.mean([r["r5"] for r in results])), 2),
        "r10": round(float(np.mean([r["r10"] for r in results])), 2),
        "mrr": round(float(np.mean([r["mrr"] for r in results])), 2),
        "ndcg@10": round(float(np.mean([r["ndcg@10"] for r in results])), 2),
    }


def pooled_metric_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    n = sum(int(r.get("n_queries", 0)) for r in results)
    if n <= 0:
        return {}
    return {
        "r1": round(100.0 * sum(float(r.get("_sum_r1", 0.0)) for r in results) / n, 4),
        "r5": round(100.0 * sum(float(r.get("_sum_r5", 0.0)) for r in results) / n, 4),
        "r10": round(100.0 * sum(float(r.get("_sum_r10", 0.0)) for r in results) / n, 4),
        "mrr": round(100.0 * sum(float(r.get("_sum_rr", 0.0)) for r in results) / n, 4),
        "ndcg@10": round(100.0 * sum(float(r.get("_sum_ndcg10", 0.0)) for r in results) / n, 4),
        "n_queries": n,
        "aggregation": "pooled_micro",
    }


def main():
    parser = argparse.ArgumentParser(description="Rerank Nomic candidates with a local LLM verifier.")
    parser.add_argument("--languages", type=str, nargs="+", default=["ruby"])
    parser.add_argument("--split", type=str, default="valid", choices=["valid", "test"])
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--experiment-name", type=str, default="llm_verifier_eval")
    parser.add_argument(
        "--experiment-family",
        type=str,
        default="",
        help="Optional namespace recorded in the result JSON (for example, work2_candidate_recovery).",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default="",
        help="Optional explicit result JSON path. The default remains RESULTS_DIR/<experiment>_eval_results.json.",
    )

    parser.add_argument("--nomic-model-path", type=str, default="/root/autodl-tmp/hf_models/nomic-embed-code")
    parser.add_argument("--nomic-cache-dir", type=str, default="data/rerank/coderank_cache")
    parser.add_argument("--nomic-max-length", type=int, default=512)
    parser.add_argument("--nomic-query-prefix", type=str, default="Represent this query for searching relevant code: ")
    parser.add_argument("--nomic-code-prefix", type=str, default="")
    parser.add_argument("--candidate-top-k", type=int, default=100)
    parser.add_argument(
        "--candidate-pool-pattern",
        type=str,
        default="",
        help=(
            "Optional .npz pattern with {language}/{split}; arrays: indices, scores, base_ranks. "
            "This evaluates a candidate-recovery action while retaining Nomic direct as fallback."
        ),
    )
    parser.add_argument(
        "--candidate-pool-tag",
        type=str,
        default="",
        help="Short action name used in score-cache keys and rank filenames.",
    )
    parser.add_argument("--verify-top-k", type=int, default=20)
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument("--code-chunk-size", type=int, default=4096)

    parser.add_argument("--verifier-model", type=str, required=True)
    parser.add_argument(
        "--verifier-base-model",
        type=str,
        default="",
        help="Base causal LM path when --verifier-model points to a LoRA adapter directory.",
    )
    parser.add_argument("--verifier-dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--verifier-device-map",
        type=str,
        default="cuda",
        choices=["cuda", "auto", "none"],
        help="Use cuda/auto device_map to avoid a large CPU RAM peak while loading verifier models.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--verifier-mode", type=str, default="yesno", choices=["yesno", "pairwise", "listwise"])
    parser.add_argument("--verifier-batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-query-chars", type=int, default=512)
    parser.add_argument("--max-code-chars", type=int, default=3000)
    parser.add_argument("--score-cache-dir", type=str, default="results/verifier_scores")
    parser.add_argument("--reuse-score-cache", action="store_true")

    parser.add_argument("--fusion-alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument(
        "--fusion-normalization",
        type=str,
        default="zscore",
        choices=["none", "zscore", "minmax", "rank", "softmax"],
    )
    parser.add_argument("--frozen-config", type=str, default="", help="Validation-selected immutable config required for test evaluation.")
    parser.add_argument("--save-frozen-config", type=str, default="", help="On valid, save the best alpha/normalization/depth configuration here.")
    parser.add_argument("--selection-metric", type=str, default="mrr", choices=["mrr", "r1", "ndcg@10"])
    parser.add_argument("--unsafe-allow-test-without-frozen-config", action="store_true", help="Historical reproduction only; never use for paper tables.")
    parser.add_argument("--rank-output-dir", type=str, default="")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--query-offset", type=int, default=0)
    args = parser.parse_args()

    if args.frozen_config:
        frozen = load_json(args.frozen_config)
        if frozen.get("selected_on_split") != "valid":
            raise ValueError("Frozen config was not selected on validation.")
        cfg = frozen.get("config", {})
        args.fusion_alphas = [float(cfg["fusion_weight"])]
        args.fusion_normalization = str(cfg.get("normalization", args.fusion_normalization))
        args.verify_top_k = int(cfg.get("verify_top_k", args.verify_top_k))
    if args.split == "test" and not args.unsafe_allow_test_without_frozen_config:
        enforce_frozen_test_protocol(args.split, args.fusion_alphas, args.frozen_config, "fusion_weight")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LLM verifier evaluation.")
    device = torch.device("cuda")

    all_results = {f"alpha={alpha:g}": {} for alpha in args.fusion_alphas}
    for language in args.languages:
        results = eval_language(args, language, device)
        for key, value in results.items():
            all_results[key][language] = value

    print("\n" + "=" * 60)
    for key, lang_results in all_results.items():
        per_language = list(lang_results.values())
        lang_results["macro_average"] = average_metric_results(per_language)
        lang_results["pooled_micro"] = pooled_metric_results(per_language)
        # Backward-compatible alias; paper tables should label this explicitly as macro.
        lang_results["average"] = lang_results["macro_average"]
        print(f"{key} Macro: {lang_results['macro_average']}")
        print(f"{key} Pooled/Micro: {lang_results['pooled_micro']}")
    print("=" * 60)

    if args.split == "valid" and args.save_frozen_config:
        candidates = []
        for alpha in args.fusion_alphas:
            key = f"alpha={alpha:g}"
            score = float(all_results[key]["macro_average"][args.selection_metric])
            candidates.append((score, float(alpha), key))
        best_score, best_alpha, best_key = max(candidates, key=lambda x: x[0])
        save_frozen_config(
            args.save_frozen_config,
            method="recal_llm_verifier",
            selected_on_split="valid",
            selected_metric=args.selection_metric,
            config={
                "fusion_weight": best_alpha,
                "normalization": args.fusion_normalization,
                "verify_top_k": args.verify_top_k,
                "candidate_top_k": args.candidate_top_k,
                "verifier_mode": args.verifier_mode,
            },
            validation_result=all_results[best_key]["macro_average"],
        )
        print(f"Saved validation-frozen config to {args.save_frozen_config}: alpha={best_alpha:g}")

    if args.experiment_family:
        all_results["_metadata"] = {
            "experiment_family": args.experiment_family,
            "experiment_name": args.experiment_name,
            "split": args.split,
            "languages": args.languages,
            "candidate_pool_tag": args.candidate_pool_tag,
        }

    if args.json_output:
        json_output = Path(args.json_output).expanduser()
    else:
        json_output = Path(RESULTS_DIR) / f"{args.experiment_name}_eval_results.json"
    json_output.parent.mkdir(parents=True, exist_ok=True)
    with json_output.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    write_environment_manifest(
        json_output.with_suffix(json_output.suffix + ".environment.json"),
        extra={
            "role": "llm_verifier_evaluation",
            "split": args.split,
            "languages": args.languages,
            "verifier_model": args.verifier_model,
            "verifier_base_model": args.verifier_base_model,
            "frozen_config": args.frozen_config,
            "fusion_alphas": args.fusion_alphas,
            "normalization": args.fusion_normalization,
            "verify_top_k": args.verify_top_k,
        },
    )
    print(f"JSON results saved to {json_output}")


if __name__ == "__main__":
    main()
