import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import DATA_ROOT, RESULTS_DIR
from data_utils import create_dataset, create_loader
from build_teacher_rerank_data import NomicTransformersTeacher, resolve_teacher_backend
from evaluate import load_model as load_retriever_model
from evaluate import resolve_eval_options

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None


def resolve_model_path(model_name: str) -> str:
    local_root = Path("/root/autodl-tmp/hf_models")
    model_dir_map = {
        "microsoft/codebert-base": local_root / "codebert-base",
        "microsoft/graphcodebert-base": local_root / "graphcodebert-base",
        "microsoft/unixcoder-base": local_root / "unixcoder-base",
    }
    local_path = model_dir_map.get(model_name)
    if local_path is not None and local_path.exists():
        return str(local_path)
    return model_name


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def get_query_text(example: Dict[str, Any]) -> str:
    for key in ["doc", "docstring", "query", "nl", "text"]:
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


def get_code_text(example: Dict[str, Any]) -> str:
    if "code" in example and example["code"]:
        return str(example["code"])

    if "code_tokens" in example and example["code_tokens"]:
        value = example["code_tokens"]
        if isinstance(value, list):
            return " ".join(map(str, value))
        return str(value)

    raise KeyError(f"Could not find code field in example keys: {list(example.keys())}")


def text2code_to_tensor(text2code, n_queries: int) -> torch.Tensor:
    if isinstance(text2code, torch.Tensor):
        return text2code.long().cpu()

    if isinstance(text2code, dict):
        return torch.tensor([text2code[i] for i in range(n_queries)], dtype=torch.long)

    return torch.tensor(list(text2code), dtype=torch.long)


def tokenize_pairs(tokenizer, queries, codes, max_length=512, max_query_length=96):
    special_tokens = tokenizer.num_special_tokens_to_add(pair=True)
    query_budget = max(1, min(max_query_length, max_length - special_tokens - 1))
    code_budget = max(1, max_length - query_budget - special_tokens)

    query_enc = tokenizer(
        queries,
        add_special_tokens=False,
        truncation=True,
        max_length=query_budget,
        padding=False,
        return_attention_mask=False,
        verbose=False,
    )
    code_enc = tokenizer(
        codes,
        add_special_tokens=False,
        truncation=True,
        max_length=code_budget,
        padding=False,
        return_attention_mask=False,
        verbose=False,
    )

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    if cls_id is None:
        cls_id = tokenizer.bos_token_id
    if sep_id is None:
        sep_id = tokenizer.eos_token_id
    if cls_id is None or sep_id is None:
        raise ValueError(
            f"Tokenizer {tokenizer.__class__.__name__} is missing cls/sep token ids."
        )

    input_ids = [
        [int(cls_id)] + list(query_ids) + [int(sep_id), int(sep_id)] + list(code_ids) + [int(sep_id)]
        for query_ids, code_ids in zip(query_enc["input_ids"], code_enc["input_ids"])
    ]
    max_batch_length = max(len(ids) for ids in input_ids)
    pad_id = tokenizer.pad_token_id
    padded_ids = []
    attention_mask = []
    for ids in input_ids:
        pad_len = max_batch_length - len(ids)
        padded_ids.append(ids + [pad_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)

    return {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


@torch.no_grad()
def get_feats(
    model,
    tokenizer,
    data_loader,
    max_length,
    device,
    pooling_mode="mean",
    desc="Get feats",
    text_prefix="",
):
    embeds = []

    for text in tqdm(data_loader, total=len(data_loader), desc=desc):
        if text_prefix:
            text = [text_prefix + t for t in text]

        text_input = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        ids = text_input["input_ids"]
        mask = text_input["attention_mask"]
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
        embeds.append(pooled.cpu())

        del text_input, ids, mask, hidden, pooled
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(embeds, dim=0)


class RerankerModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        score_dim: int = 1,
    ):
        super().__init__()

        from peft import LoraConfig, get_peft_model
        from transformers import RobertaModel

        model_path = resolve_model_path(model_name)
        self.encoder = RobertaModel.from_pretrained(model_path)
        self.score_dim = score_dim
        target_modules = target_modules or ["query", "value"]

        for param in self.encoder.parameters():
            param.requires_grad = False
            if param.ndim == 1:
                param.data = param.data.to(torch.float32)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            task_type="FEATURE_EXTRACTION",
        )
        self.encoder = get_peft_model(self.encoder, lora_config)

        if hasattr(self.encoder, "config"):
            self.encoder.config.use_cache = False

        self.classifier = nn.Linear(self.encoder.config.hidden_size, score_dim)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        raw_logits = self.classifier(cls)
        logits = raw_logits.squeeze(-1) if raw_logits.shape[-1] == 1 else raw_logits

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}


def has_reranker_weights(path: Path) -> bool:
    return (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()


def checkpoint_step(path: Path) -> int:
    if not path.name.startswith("checkpoint-"):
        return -1
    try:
        return int(path.name.rsplit("-", 1)[-1])
    except ValueError:
        return -1


def resolve_reranker_weight_dir(reranker_dir: Path) -> Path:
    if has_reranker_weights(reranker_dir):
        return reranker_dir

    checkpoint_dirs = [p for p in reranker_dir.glob("checkpoint-*") if p.is_dir()]
    weight_dirs = [p for p in checkpoint_dirs if has_reranker_weights(p)]
    if weight_dirs:
        latest = max(weight_dirs, key=checkpoint_step)
        print(f"Loading reranker weights from latest checkpoint: {latest}")
        return latest

    searched = ", ".join(p.name for p in sorted(checkpoint_dirs, key=checkpoint_step)[-5:])
    if not searched:
        searched = "no checkpoint-* directories found"
    raise FileNotFoundError(
        f"Could not find reranker weights in {reranker_dir}. "
        f"Searched root and recent checkpoints: {searched}. "
        "Expected model.safetensors or pytorch_model.bin. "
        "If this is a DeepSpeed-only checkpoint, finish a Trainer save or consolidate it first."
    )


def load_reranker(reranker_dir: str, model_name: str):
    from transformers import RobertaTokenizer

    reranker_dir = Path(reranker_dir)
    weight_dir = resolve_reranker_weight_dir(reranker_dir)
    config_path = reranker_dir / "reranker_config.json"
    if not config_path.exists():
        config_path = weight_dir / "reranker_config.json"
    reranker_config = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            reranker_config = json.load(f)
    config_model_name = reranker_config.get("model_name", model_name)
    model_path = resolve_model_path(config_model_name)

    if (reranker_dir / "tokenizer_config.json").exists():
        tokenizer = RobertaTokenizer.from_pretrained(reranker_dir)
    elif (weight_dir / "tokenizer_config.json").exists():
        tokenizer = RobertaTokenizer.from_pretrained(weight_dir)
    else:
        tokenizer = RobertaTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    safetensor_path = weight_dir / "model.safetensors"
    bin_path = weight_dir / "pytorch_model.bin"

    if safetensor_path.exists() and safe_load_file is not None:
        state_dict = safe_load_file(str(safetensor_path))
    elif bin_path.exists():
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(
            f"Could not find reranker weights in {weight_dir}. "
            "Expected model.safetensors or pytorch_model.bin."
        )

    classifier_weight = state_dict.get("classifier.weight")
    score_dim = reranker_config.get("score_dim")
    if score_dim is None and classifier_weight is not None:
        score_dim = int(classifier_weight.shape[0])
    if score_dim is None:
        score_dim = 2

    model = RerankerModel(
        model_name=config_model_name,
        lora_r=int(reranker_config.get("lora_r", 16)),
        lora_alpha=int(reranker_config.get("lora_alpha", 32)),
        lora_dropout=float(reranker_config.get("lora_dropout", 0.1)),
        target_modules=reranker_config.get("target_modules", ["query", "value"]),
        score_dim=int(score_dim),
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading reranker: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading reranker: {unexpected}")

    model.eval()
    return model, tokenizer


@torch.no_grad()
def retrieve_topk_and_ranks(
    query_embeds: torch.Tensor,
    code_embeds: torch.Tensor,
    gt_indices: torch.Tensor,
    device: torch.device,
    top_k: int = 50,
    query_chunk_size: int = 64,
    code_chunk_size: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_embeds = torch.nn.functional.normalize(query_embeds, dim=1)
    code_embeds = torch.nn.functional.normalize(code_embeds, dim=1)

    n_queries = query_embeds.size(0)
    n_codes = code_embeds.size(0)

    all_top_scores = []
    all_top_indices = []
    all_ranks = []

    for q_start in tqdm(range(0, n_queries, query_chunk_size), desc="First-stage retrieval"):
        q_end = min(q_start + query_chunk_size, n_queries)
        q_chunk = query_embeds[q_start:q_end].to(device)
        q_len = q_end - q_start

        chunk_scores = torch.empty((q_len, n_codes), dtype=torch.float32, device=device)

        for c_start in range(0, n_codes, code_chunk_size):
            c_end = min(c_start + code_chunk_size, n_codes)
            c_chunk = code_embeds[c_start:c_end].to(device)
            chunk_scores[:, c_start:c_end] = q_chunk @ c_chunk.t()

            del c_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        top_scores, top_indices = torch.topk(chunk_scores, k=top_k, dim=1)

        gt_chunk = gt_indices[q_start:q_end].to(device)
        gt_scores = chunk_scores.gather(1, gt_chunk.unsqueeze(1)).squeeze(1)
        ranks = (chunk_scores > gt_scores.unsqueeze(1)).sum(dim=1)

        all_top_scores.append(top_scores.cpu())
        all_top_indices.append(top_indices.cpu())
        all_ranks.append(ranks.cpu())

        del q_chunk, chunk_scores, top_scores, top_indices, gt_chunk, gt_scores, ranks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return (
        torch.cat(all_top_scores, dim=0),
        torch.cat(all_top_indices, dim=0),
        torch.cat(all_ranks, dim=0),
    )


@torch.no_grad()
def score_reranker_pairs(
    reranker_model,
    reranker_tokenizer,
    device: torch.device,
    query: str,
    candidate_codes: List[str],
    batch_size: int = 16,
    max_length: int = 512,
    max_query_length: int = 96,
) -> torch.Tensor:
    scores = []

    for start in range(0, len(candidate_codes), batch_size):
        batch_codes = candidate_codes[start:start + batch_size]

        queries = ["query: " + query for _ in batch_codes]
        codes = ["code: " + code for code in batch_codes]

        enc = tokenize_pairs(
            reranker_tokenizer,
            queries,
            codes,
            max_length=max_length,
            max_query_length=max_query_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        outputs = reranker_model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )

        logits = outputs["logits"]
        if logits.ndim == 1:
            pos_scores = logits
        elif logits.shape[-1] == 1:
            pos_scores = logits.squeeze(-1)
        else:
            pos_scores = logits[:, 1]

        scores.append(pos_scores.cpu())

        del enc, outputs, logits, pos_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(scores, dim=0)


def calculate_dcg(relevance_scores, k):
    relevance_scores = np.array(relevance_scores[:k])
    return np.sum(relevance_scores / np.log2(np.arange(2, relevance_scores.size + 2)))


def calculate_ndcg(relevance_scores, k):
    dcg = calculate_dcg(relevance_scores, k)
    ideal = sorted(relevance_scores, reverse=True)
    idcg = calculate_dcg(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics_from_ranks(ranks: np.ndarray):
    r1 = 100.0 * np.mean(ranks < 1)
    r5 = 100.0 * np.mean(ranks < 5)
    r10 = 100.0 * np.mean(ranks < 10)
    mrr = 100.0 * np.mean(1.0 / (ranks + 1))

    ndcgs = []
    for r in ranks:
        rel = [0] * 10
        if r < 10:
            rel[int(r)] = 1
        ndcgs.append(calculate_ndcg(rel, k=10))

    ndcg10 = 100.0 * np.mean(ndcgs)

    return {
        "r1": round(r1, 2),
        "r5": round(r5, 2),
        "r10": round(r10, 2),
        "mrr": round(mrr, 2),
        "ndcg@10": round(ndcg10, 2),
    }


def compute_upper_bounds_from_ranks(ranks: np.ndarray, ks: List[int]) -> Dict[str, float]:
    upper_bounds: Dict[str, float] = {}
    for k in ks:
        if k <= 0:
            continue
        recall = 100.0 * np.mean(ranks < k)
        oracle_ranks = np.where(ranks < k, 0, ranks)
        oracle_mrr = 100.0 * np.mean(1.0 / (oracle_ranks + 1))
        upper_bounds[f"recall@{k}"] = round(recall, 2)
        upper_bounds[f"oracle_mrr@{k}"] = round(oracle_mrr, 2)
    return upper_bounds


def retrieve_tfidf_topk(
    queries: Sequence[str],
    codes: Sequence[str],
    top_k: int,
    query_chunk_size: int = 512,
    code_chunk_size: int = 20000,
    max_features: int = 50000,
) -> torch.Tensor:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    code_tfidf = vectorizer.fit_transform(codes)
    query_tfidf = vectorizer.transform(queries)

    n_queries = query_tfidf.shape[0]
    n_codes = code_tfidf.shape[0]
    top_k = min(top_k, n_codes)

    all_top_indices = []
    for q_start in tqdm(range(0, n_queries, query_chunk_size), desc="TF-IDF retrieval"):
        q_end = min(q_start + query_chunk_size, n_queries)
        q_block = query_tfidf[q_start:q_end]
        q_len = q_end - q_start

        cur_scores = np.full((q_len, top_k), -np.inf, dtype=np.float32)
        cur_indices = np.zeros((q_len, top_k), dtype=np.int64)

        for c_start in range(0, n_codes, code_chunk_size):
            c_end = min(c_start + code_chunk_size, n_codes)
            sim = (q_block @ code_tfidf[c_start:c_end].T).toarray().astype(np.float32, copy=False)
            k = min(top_k, sim.shape[1])
            if k <= 0:
                continue

            local_top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            local_scores = np.take_along_axis(sim, local_top, axis=1)
            local_indices = local_top + c_start

            merged_scores = np.concatenate([cur_scores, local_scores], axis=1)
            merged_indices = np.concatenate([cur_indices, local_indices], axis=1)
            merged_top = np.argpartition(-merged_scores, top_k - 1, axis=1)[:, :top_k]
            cur_scores = np.take_along_axis(merged_scores, merged_top, axis=1)
            cur_indices = np.take_along_axis(merged_indices, merged_top, axis=1)

            del sim, local_top, local_scores, local_indices, merged_scores, merged_indices, merged_top

        order = np.argsort(-cur_scores, axis=1)
        cur_indices = np.take_along_axis(cur_indices, order, axis=1)
        all_top_indices.append(torch.from_numpy(cur_indices.copy()))

    return torch.cat(all_top_indices, dim=0)


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def load_sentence_embedder(
    model_name: str,
    device: torch.device,
    max_seq_length: int,
    trust_remote_code: bool,
    backend: str = "auto",
    dtype_name: str = "auto",
):
    backend = resolve_teacher_backend(model_name, backend)
    print(f"  external embedding backend: {backend}")

    if backend == "nomic_transformers":
        return NomicTransformersTeacher(
            model_name=model_name,
            device=str(device),
            max_seq_length=max_seq_length,
            trust_remote_code=trust_remote_code,
            dtype_name=dtype_name,
        )

    if backend != "sentence_transformer":
        raise ValueError(f"Unknown external embedding backend: {backend}")

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise ImportError(
            "sentence-transformers is required for dense_coderank candidates. "
            "Install it with: pip install -U sentence-transformers"
        ) from exc

    kwargs = {"device": str(device)}
    if trust_remote_code:
        kwargs["trust_remote_code"] = True

    try:
        model = SentenceTransformer(model_name, **kwargs)
    except TypeError:
        kwargs.pop("trust_remote_code", None)
        model = SentenceTransformer(model_name, **kwargs)

    if max_seq_length > 0:
        model.max_seq_length = max_seq_length
    return model


def coderank_cache_paths(
    cache_dir: str,
    language: str,
    split: str,
    model_name: str,
    max_seq_length: int,
    backend: str,
    query_prefix: str,
    code_prefix: str,
) -> Optional[Dict[str, Path]]:
    if not cache_dir:
        return None
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


def load_cached_embeddings(path: Path, expected_rows: int, label: str) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    embeds = np.load(path, mmap_mode="r")
    if embeds.shape[0] != expected_rows:
        print(f"  ignoring {label} cache with {embeds.shape[0]} rows; expected {expected_rows}")
        return None
    print(f"  loaded {label} cache: {path}")
    return embeds


def encode_sentence_embeddings(
    model,
    texts: Sequence[str],
    batch_size: int,
    prefix: str,
    desc: str,
) -> np.ndarray:
    if prefix:
        texts = [prefix + text for text in texts]
    print(f"  {desc}: encoding {len(texts)} texts")
    embeds = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeds, dtype=np.float32)


def get_coderank_embeddings(
    queries: Sequence[str],
    codes: Sequence[str],
    language: str,
    split: str,
    model_name: str,
    backend: str,
    dtype_name: str,
    query_prefix: str,
    code_prefix: str,
    max_seq_length: int,
    batch_size: int,
    cache_dir: str,
    trust_remote_code: bool,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    resolved_backend = resolve_teacher_backend(model_name, backend)
    cache = coderank_cache_paths(
        cache_dir,
        language,
        split,
        model_name,
        max_seq_length,
        resolved_backend,
        query_prefix,
        code_prefix,
    )
    query_embeds = None
    code_embeds = None
    if cache is not None:
        query_embeds = load_cached_embeddings(cache["query"], len(queries), "CodeRank query")
        code_embeds = load_cached_embeddings(cache["code"], len(codes), "CodeRank code")

    if query_embeds is not None and code_embeds is not None:
        return query_embeds, code_embeds

    model = load_sentence_embedder(
        model_name=model_name,
        device=device,
        max_seq_length=max_seq_length,
        trust_remote_code=trust_remote_code,
        backend=resolved_backend,
        dtype_name=dtype_name,
    )

    if query_embeds is None:
        query_embeds = encode_sentence_embeddings(
            model,
            queries,
            batch_size=batch_size,
            prefix=query_prefix,
            desc="CodeRank queries",
        )
        if cache is not None:
            np.save(cache["query"], query_embeds)

    if code_embeds is None:
        code_embeds = encode_sentence_embeddings(
            model,
            codes,
            batch_size=batch_size,
            prefix=code_prefix,
            desc="CodeRank codes",
        )
        if cache is not None:
            np.save(cache["code"], code_embeds)

    return query_embeds, code_embeds


@torch.no_grad()
def retrieve_embedding_topk(
    query_embeds: np.ndarray,
    code_embeds: np.ndarray,
    device: torch.device,
    top_k: int,
    query_chunk_size: int,
    code_chunk_size: int,
    desc: str = "Embedding retrieval",
) -> Tuple[torch.Tensor, torch.Tensor]:
    n_queries = query_embeds.shape[0]
    n_codes = code_embeds.shape[0]
    top_k = min(top_k, n_codes)

    all_top_scores = []
    all_top_indices = []
    for q_start in tqdm(range(0, n_queries, query_chunk_size), desc=desc):
        q_end = min(q_start + query_chunk_size, n_queries)
        q_chunk = torch.from_numpy(query_embeds[q_start:q_end]).to(device)
        q_len = q_end - q_start

        cur_scores = torch.full((q_len, top_k), -1e9, dtype=torch.float32, device=device)
        cur_indices = torch.zeros((q_len, top_k), dtype=torch.long, device=device)

        for c_start in range(0, n_codes, code_chunk_size):
            c_end = min(c_start + code_chunk_size, n_codes)
            c_chunk = torch.from_numpy(code_embeds[c_start:c_end]).to(device)
            scores = (q_chunk @ c_chunk.t()).float()
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

        all_top_scores.append(cur_scores.cpu())
        all_top_indices.append(cur_indices.cpu())

        del q_chunk, cur_scores, cur_indices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(all_top_scores, dim=0), torch.cat(all_top_indices, dim=0)


def merge_dense_tfidf_candidates(
    dense_indices: torch.Tensor,
    tfidf_indices: torch.Tensor,
    top_k: int,
    dense_keep: int,
    tfidf_keep: int,
) -> torch.Tensor:
    merged_rows = []
    dense_keep = max(0, dense_keep)
    tfidf_keep = max(0, tfidf_keep)

    for dense_row, tfidf_row in zip(dense_indices.tolist(), tfidf_indices.tolist()):
        merged = []
        seen = set()

        for idx in dense_row[:dense_keep]:
            if idx not in seen:
                merged.append(idx)
                seen.add(idx)
            if len(merged) >= top_k:
                break

        for idx in tfidf_row[:tfidf_keep]:
            if idx not in seen:
                merged.append(idx)
                seen.add(idx)
            if len(merged) >= top_k:
                break

        for idx in dense_row:
            if idx not in seen:
                merged.append(idx)
                seen.add(idx)
            if len(merged) >= top_k:
                break

        if len(merged) < top_k:
            raise ValueError(f"Only built {len(merged)} candidates, expected {top_k}.")
        merged_rows.append(merged[:top_k])

    return torch.tensor(merged_rows, dtype=torch.long)


def merge_dense_external_candidates(
    dense_indices: torch.Tensor,
    external_indices: torch.Tensor,
    top_k: int,
    dense_keep: int,
    external_keep: int,
) -> torch.Tensor:
    return merge_dense_tfidf_candidates(
        dense_indices=dense_indices,
        tfidf_indices=external_indices,
        top_k=top_k,
        dense_keep=dense_keep,
        tfidf_keep=external_keep,
    )


def compute_candidate_pool_recall(candidate_indices: torch.Tensor, gt_indices: torch.Tensor) -> Dict[str, float]:
    matches = (candidate_indices == gt_indices.unsqueeze(1)).any(dim=1).float()
    return {f"candidate_recall@{candidate_indices.size(1)}": round(100.0 * float(matches.mean().item()), 2)}


def normalize_scores(scores: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return scores
    if mode == "zscore":
        std = scores.std(unbiased=False)
        if float(std.item()) < 1e-6:
            return scores - scores.mean()
        return (scores - scores.mean()) / torch.clamp(std, min=1e-6)
    if mode == "minmax":
        score_min = scores.min()
        score_max = scores.max()
        denom = torch.clamp(score_max - score_min, min=1e-6)
        return (scores - score_min) / denom
    raise ValueError(f"Unknown score normalization mode: {mode}")


ROUTE_SCORE_SUFFIXES = [
    "top1",
    "top2",
    "top5",
    "top10",
    "margin_1_2",
    "margin_1_5",
    "margin_1_10",
    "top5_mean",
    "top10_mean",
    "top5_std",
    "top10_std",
    "top5_entropy",
    "top10_entropy",
]


def route_score_feature_names(prefix: str) -> List[str]:
    return [f"{prefix}_{suffix}" for suffix in ROUTE_SCORE_SUFFIXES]


def score_entropy(scores: torch.Tensor) -> float:
    if scores.numel() == 0:
        return 0.0
    probs = torch.softmax(scores.float(), dim=0)
    return float(-(probs * torch.log(torch.clamp(probs, min=1e-12))).sum().item())


def route_score_features(scores: torch.Tensor) -> List[float]:
    sorted_scores = torch.sort(scores.float(), descending=True).values

    def kth(k: int) -> float:
        pos = min(k - 1, sorted_scores.numel() - 1)
        return float(sorted_scores[pos].item())

    top5 = sorted_scores[: min(5, sorted_scores.numel())]
    top10 = sorted_scores[: min(10, sorted_scores.numel())]
    top1 = kth(1)
    return [
        top1,
        kth(2),
        kth(5),
        kth(10),
        top1 - kth(2),
        top1 - kth(5),
        top1 - kth(10),
        float(top5.mean().item()),
        float(top10.mean().item()),
        float(top5.std(unbiased=False).item()),
        float(top10.std(unbiased=False).item()),
        score_entropy(top5),
        score_entropy(top10),
    ]


def reciprocal_rank_in_row(row: torch.Tensor, target: int, missing_rank: int) -> float:
    match = (row == int(target)).nonzero(as_tuple=False)
    if match.numel() == 0:
        return 1.0 / float(missing_rank + 1)
    return 1.0 / float(int(match[0, 0].item()) + 1)


def overlap_ratio(left: torch.Tensor, right: torch.Tensor, k: int) -> float:
    k = min(k, left.numel(), right.numel())
    if k <= 0:
        return 0.0
    return float(len(set(left[:k].tolist()).intersection(right[:k].tolist())) / k)


def save_route_features(
    output_dir: str,
    language: str,
    split: str,
    candidate_source: str,
    top_k: int,
    alpha: float,
    features: List[List[float]],
    feature_names: List[str],
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    rank_key = sanitize_filename(alpha_result_key(alpha))
    output_path = output_root / f"{language}_{split}_{candidate_source}_top{top_k}_{rank_key}.route_features.npz"
    feature_array = np.asarray(features, dtype=np.float32)
    if feature_array.ndim != 2 or feature_array.shape[1] != len(feature_names):
        raise ValueError(
            f"Route feature shape mismatch: shape={feature_array.shape}, names={len(feature_names)}"
        )
    np.savez_compressed(
        output_path,
        features=feature_array,
        feature_names=np.asarray(feature_names, dtype=object),
        language=np.asarray(language),
        split=np.asarray(split),
        candidate_source=np.asarray(candidate_source),
        top_k=np.asarray(top_k, dtype=np.int32),
        alpha=np.asarray(alpha, dtype=np.float32),
    )
    print(f"  saved route features to: {output_path}")


def format_alpha(alpha: float) -> str:
    return f"{alpha:g}"


def alpha_result_key(alpha: float) -> str:
    return f"alpha={format_alpha(alpha)}"


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value).strip("_")


def average_metric_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    avg_mrr = np.mean([r["mrr"] for r in results])
    avg_r1 = np.mean([r["r1"] for r in results])
    avg_r5 = np.mean([r["r5"] for r in results])
    avg_r10 = np.mean([r["r10"] for r in results])
    avg_ndcg = np.mean([r["ndcg@10"] for r in results])

    return {
        "r1": round(avg_r1, 2),
        "r5": round(avg_r5, 2),
        "r10": round(avg_r10, 2),
        "mrr": round(avg_mrr, 2),
        "ndcg@10": round(avg_ndcg, 2),
    }


@torch.no_grad()
def rerank_and_eval_language(
    retriever_model,
    retriever_tokenizer,
    reranker_model,
    reranker_tokenizer,
    language: str,
    split: str,
    data_root: str,
    retriever_pooling_mode: str,
    query_prefix: str,
    code_prefix: str,
    retriever_max_length: int,
    text_batch_size: int,
    code_batch_size: int,
    num_workers: int,
    device: torch.device,
    top_k: int,
    reranker_batch_size: int,
    reranker_max_length: int,
    reranker_max_query_length: int,
    query_chunk_size: int,
    code_chunk_size: int,
    fusion_alphas: List[float],
    fusion_normalization: str,
    candidate_source: str,
    dense_candidate_k: int,
    tfidf_candidate_k: int,
    tfidf_query_chunk_size: int,
    tfidf_code_chunk_size: int,
    tfidf_max_features: int,
    coderank_candidate_k: int,
    coderank_model: str,
    coderank_backend: str,
    coderank_dtype: str,
    coderank_query_prefix: str,
    coderank_code_prefix: str,
    coderank_max_seq_length: int,
    coderank_batch_size: int,
    coderank_query_chunk_size: int,
    coderank_code_chunk_size: int,
    coderank_cache_dir: str,
    coderank_trust_remote_code: bool,
    coderank_score_mode: str,
    coderank_score_alpha: float,
    rank_output_dir: str,
    route_feature_output_dir: str,
):
    print(f"\n--- Reranker evaluating {language} / {split} ---")

    _, val_dataset, test_dataset, code_dataset = create_dataset(str(data_root), language)
    eval_dataset = val_dataset if split == "valid" else test_dataset

    eval_loader, code_loader = create_loader(
        [eval_dataset, code_dataset],
        [None, None],
        batch_size=[text_batch_size, code_batch_size],
        num_workers=[num_workers, num_workers],
        is_trains=[False, False],
        collate_fns=[None, None],
    )

    retriever_model.to(device)
    retriever_model.eval()

    text_embeds = get_feats(
        retriever_model,
        retriever_tokenizer,
        eval_loader,
        retriever_max_length,
        device,
        retriever_pooling_mode,
        "Text feats",
        query_prefix,
    )

    code_embeds = get_feats(
        retriever_model,
        retriever_tokenizer,
        code_loader,
        retriever_max_length,
        device,
        retriever_pooling_mode,
        "Code feats",
        code_prefix,
    )

    gt_indices = text2code_to_tensor(eval_loader.dataset.text2code, len(eval_loader.dataset))

    top_scores, top_indices, orig_ranks = retrieve_topk_and_ranks(
        query_embeds=text_embeds,
        code_embeds=code_embeds,
        gt_indices=gt_indices,
        device=device,
        top_k=max(top_k, dense_candidate_k),
        query_chunk_size=query_chunk_size,
        code_chunk_size=code_chunk_size,
    )

    orig_ranks_np = orig_ranks.numpy()
    first_stage_result = compute_metrics_from_ranks(orig_ranks_np)
    upper_bound_ks = sorted({1, 5, 10, 20, 50, 100, 200, top_k})
    upper_bounds = compute_upper_bounds_from_ranks(orig_ranks_np, upper_bound_ks)
    print(f"  first-stage: {first_stage_result}")
    print(f"  upper bounds: {upper_bounds}")

    test_examples = read_jsonl(Path(data_root) / language / f"{split}.jsonl")
    code_examples = read_jsonl(Path(data_root) / language / "codebase.jsonl")

    test_queries = [get_query_text(x) for x in test_examples]
    code_texts = [get_code_text(x) for x in code_examples]

    candidate_indices = top_indices[:, :top_k]
    candidate_pool_result = compute_candidate_pool_recall(candidate_indices, gt_indices)
    coderank_query_embeds = None
    coderank_code_embeds = None
    coderank_top_scores_for_features = None
    coderank_top_indices_for_features = None
    if candidate_source == "dense_tfidf":
        if dense_candidate_k <= 0 or tfidf_candidate_k <= 0:
            raise ValueError("--dense-candidate-k and --tfidf-candidate-k must be positive for dense_tfidf.")
        print(
            f"  building dense+TF-IDF candidate pool: "
            f"dense_keep={dense_candidate_k}, tfidf_keep={tfidf_candidate_k}, final_top_k={top_k}"
        )
        tfidf_top_indices = retrieve_tfidf_topk(
            queries=test_queries,
            codes=code_texts,
            top_k=tfidf_candidate_k,
            query_chunk_size=tfidf_query_chunk_size,
            code_chunk_size=tfidf_code_chunk_size,
            max_features=tfidf_max_features,
        )
        candidate_indices = merge_dense_tfidf_candidates(
            dense_indices=top_indices,
            tfidf_indices=tfidf_top_indices,
            top_k=top_k,
            dense_keep=dense_candidate_k,
            tfidf_keep=tfidf_candidate_k,
        )
        candidate_pool_result = compute_candidate_pool_recall(candidate_indices, gt_indices)
        print(f"  candidate pool: {candidate_pool_result}")
    elif candidate_source == "dense_coderank":
        if dense_candidate_k <= 0 or coderank_candidate_k <= 0:
            raise ValueError("--dense-candidate-k and --coderank-candidate-k must be positive for dense_coderank.")
        print(
            f"  building dense+CodeRank candidate pool: "
            f"dense_keep={dense_candidate_k}, coderank_keep={coderank_candidate_k}, final_top_k={top_k}"
        )
        coderank_query_embeds, coderank_code_embeds = get_coderank_embeddings(
            queries=test_queries,
            codes=code_texts,
            language=language,
            split=split,
            model_name=coderank_model,
            backend=coderank_backend,
            dtype_name=coderank_dtype,
            query_prefix=coderank_query_prefix,
            code_prefix=coderank_code_prefix,
            max_seq_length=coderank_max_seq_length,
            batch_size=coderank_batch_size,
            cache_dir=coderank_cache_dir,
            trust_remote_code=coderank_trust_remote_code,
            device=device,
        )
        coderank_top_scores, coderank_top_indices = retrieve_embedding_topk(
            query_embeds=coderank_query_embeds,
            code_embeds=coderank_code_embeds,
            device=device,
            top_k=coderank_candidate_k,
            query_chunk_size=coderank_query_chunk_size,
            code_chunk_size=coderank_code_chunk_size,
            desc="CodeRank retrieval",
        )
        coderank_pool_result = compute_candidate_pool_recall(coderank_top_indices, gt_indices)
        if route_feature_output_dir:
            coderank_top_scores_for_features = coderank_top_scores
            coderank_top_indices_for_features = coderank_top_indices
        candidate_indices = merge_dense_external_candidates(
            dense_indices=top_indices,
            external_indices=coderank_top_indices,
            top_k=top_k,
            dense_keep=dense_candidate_k,
            external_keep=coderank_candidate_k,
        )
        candidate_pool_result = compute_candidate_pool_recall(candidate_indices, gt_indices)
        candidate_pool_result.update({f"coderank_{k}": v for k, v in coderank_pool_result.items()})
        print(f"  candidate pool: {candidate_pool_result}")
        if not route_feature_output_dir:
            del coderank_top_scores, coderank_top_indices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif candidate_source != "dense":
        raise ValueError(f"Unknown candidate source: {candidate_source}")

    reranker_model.to(device)
    reranker_model.eval()

    fusion_alphas = list(dict.fromkeys(float(alpha) for alpha in fusion_alphas))
    if not fusion_alphas:
        raise ValueError("At least one fusion alpha is required.")
    needs_normalized_scores = any(alpha > 0.0 for alpha in fusion_alphas)
    final_ranks_by_alpha = {alpha: orig_ranks.clone() for alpha in fusion_alphas}
    route_features_by_alpha = {alpha: [] for alpha in fusion_alphas} if route_feature_output_dir else {}
    base_route_feature_names = [
        "query_chars",
        "query_words",
        "log_query_chars",
        "log_query_words",
        *route_score_feature_names("dense_global"),
        *route_score_feature_names("dense_candidate"),
        *route_score_feature_names("retriever_candidate"),
        *route_score_feature_names("reranker"),
    ]
    route_feature_names = list(base_route_feature_names)
    if candidate_source == "dense_coderank":
        route_feature_names.extend(
            [
                *route_score_feature_names("coderank_global"),
                *route_score_feature_names("coderank_candidate"),
                "dense_coderank_top1_same",
                "dense_coderank_overlap_at_5",
                "dense_coderank_overlap_at_10",
                "dense_coderank_overlap_at_50",
                "dense_top1_rank_in_coderank_recip",
                "coderank_top1_rank_in_dense_recip",
                "reranker_top1_eq_dense_top1",
                "reranker_top1_eq_coderank_top1",
                "retriever_top1_eq_dense_top1",
                "retriever_top1_eq_coderank_top1",
                "reranker_retriever_top1_same",
            ]
        )
    route_feature_names.extend(
        [
            *route_score_feature_names("fusion"),
            "fusion_top1_eq_dense_top1",
            "fusion_top1_eq_reranker_top1",
            "fusion_top1_eq_retriever_top1",
            "fusion_top1_eq_coderank_top1",
        ]
    )

    for i in tqdm(range(len(test_queries)), desc="Reranking top-k"):
        query = test_queries[i]
        query_candidate_indices = candidate_indices[i].tolist()
        candidate_indices_tensor = torch.tensor(query_candidate_indices)
        candidate_codes = [code_texts[idx] for idx in query_candidate_indices]

        rerank_scores = score_reranker_pairs(
            reranker_model=reranker_model,
            reranker_tokenizer=reranker_tokenizer,
            device=device,
            query=query,
            candidate_codes=candidate_codes,
            batch_size=reranker_batch_size,
            max_length=reranker_max_length,
            max_query_length=reranker_max_query_length,
        )

        if candidate_source == "dense":
            first_stage_scores = top_scores[i, :top_k]
            dense_scores = first_stage_scores
            coderank_scores = None
        else:
            dense_scores = torch.sum(
                code_embeds[candidate_indices_tensor] * text_embeds[i].unsqueeze(0),
                dim=1,
            )
            first_stage_scores = dense_scores
            coderank_scores = None

            if candidate_source == "dense_coderank" and (coderank_score_mode != "dense" or route_feature_output_dir):
                if coderank_query_embeds is None or coderank_code_embeds is None:
                    raise RuntimeError("CodeRank embeddings are required for CodeRank score fusion.")
                candidate_np = candidate_indices_tensor.numpy()
                coderank_scores_np = coderank_code_embeds[candidate_np] @ coderank_query_embeds[i]
                coderank_scores = torch.from_numpy(np.asarray(coderank_scores_np, dtype=np.float32))

                if coderank_score_mode == "dense":
                    first_stage_scores = dense_scores
                elif coderank_score_mode == "coderank":
                    first_stage_scores = coderank_scores
                elif coderank_score_mode == "dense_coderank":
                    first_stage_scores = dense_scores + coderank_score_alpha * coderank_scores
                else:
                    raise ValueError(f"Unknown CodeRank score mode: {coderank_score_mode}")

        if needs_normalized_scores:
            retriever_scores = normalize_scores(first_stage_scores, fusion_normalization)
            reranker_scores = normalize_scores(rerank_scores, fusion_normalization)
        else:
            retriever_scores = None
            reranker_scores = None

        gt = int(gt_indices[i].item())
        for alpha in fusion_alphas:
            if alpha > 0.0:
                final_scores = reranker_scores + alpha * retriever_scores
            else:
                final_scores = rerank_scores

            reranked_pos = torch.argsort(final_scores, descending=True)
            reranked_indices = candidate_indices_tensor[reranked_pos]
            match = (reranked_indices == gt).nonzero(as_tuple=False)

            if match.numel() > 0:
                final_ranks_by_alpha[alpha][i] = match[0, 0]

            if route_feature_output_dir:
                query_chars = float(len(query))
                query_words = float(len(query.split()))
                dense_top_row = top_scores[i, : min(top_scores.size(1), top_k)]
                dense_top1 = int(top_indices[i, 0].item())
                reranker_top1 = int(candidate_indices_tensor[int(torch.argmax(rerank_scores).item())].item())
                retriever_top1 = int(candidate_indices_tensor[int(torch.argmax(first_stage_scores).item())].item())
                fusion_top1 = int(reranked_indices[0].item())
                base_features = [
                    query_chars,
                    query_words,
                    float(np.log1p(query_chars)),
                    float(np.log1p(query_words)),
                    *route_score_features(dense_top_row),
                    *route_score_features(dense_scores),
                    *route_score_features(first_stage_scores),
                    *route_score_features(rerank_scores),
                ]

                coderank_top1 = -1
                if candidate_source == "dense_coderank":
                    if coderank_top_scores_for_features is None or coderank_top_indices_for_features is None:
                        raise RuntimeError("CodeRank top-k features are not available.")
                    if coderank_scores is None:
                        raise RuntimeError("CodeRank candidate scores are required for route feature export.")
                    coderank_top_row = coderank_top_scores_for_features[i]
                    coderank_indices_row = coderank_top_indices_for_features[i]
                    coderank_top1 = int(coderank_indices_row[0].item())
                    dense_indices_row = top_indices[i, : min(top_indices.size(1), top_k)]
                    base_features.extend(
                        [
                            *route_score_features(coderank_top_row),
                            *route_score_features(coderank_scores),
                            float(dense_top1 == coderank_top1),
                            overlap_ratio(dense_indices_row, coderank_indices_row, 5),
                            overlap_ratio(dense_indices_row, coderank_indices_row, 10),
                            overlap_ratio(dense_indices_row, coderank_indices_row, 50),
                            reciprocal_rank_in_row(coderank_indices_row, dense_top1, coderank_indices_row.numel()),
                            reciprocal_rank_in_row(dense_indices_row, coderank_top1, dense_indices_row.numel()),
                            float(reranker_top1 == dense_top1),
                            float(reranker_top1 == coderank_top1),
                            float(retriever_top1 == dense_top1),
                            float(retriever_top1 == coderank_top1),
                            float(reranker_top1 == retriever_top1),
                        ]
                    )

                alpha_features = [
                    *route_score_features(final_scores),
                    float(fusion_top1 == dense_top1),
                    float(fusion_top1 == reranker_top1),
                    float(fusion_top1 == retriever_top1),
                    float(coderank_top1 >= 0 and fusion_top1 == coderank_top1),
                ]
                route_features_by_alpha[alpha].append(base_features + alpha_features)

        del rerank_scores, first_stage_scores, candidate_indices_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if route_feature_output_dir:
        for alpha, feature_rows in route_features_by_alpha.items():
            save_route_features(
                output_dir=route_feature_output_dir,
                language=language,
                split=split,
                candidate_source=candidate_source,
                top_k=top_k,
                alpha=alpha,
                features=feature_rows,
                feature_names=route_feature_names,
            )

    results_by_alpha = {}
    for alpha in fusion_alphas:
        result = compute_metrics_from_ranks(final_ranks_by_alpha[alpha].numpy())
        if len(fusion_alphas) > 1:
            result["fusion_alpha"] = alpha
        result["first_stage"] = first_stage_result
        result["upper_bounds"] = upper_bounds
        result["candidate_source"] = candidate_source
        result["candidate_pool"] = candidate_pool_result
        if candidate_source == "dense_coderank":
            result["coderank_score_mode"] = coderank_score_mode
            result["coderank_score_alpha"] = coderank_score_alpha
        results_by_alpha[alpha_result_key(alpha)] = result
        if rank_output_dir:
            rank_dir = Path(rank_output_dir)
            rank_dir.mkdir(parents=True, exist_ok=True)
            rank_key = sanitize_filename(alpha_result_key(alpha))
            rank_path = rank_dir / f"{language}_{split}_{candidate_source}_top{top_k}_{rank_key}.ranks.npy"
            np.save(rank_path, final_ranks_by_alpha[alpha].numpy().astype(np.int32))
            print(f"  saved ranks to: {rank_path}")

    if len(fusion_alphas) == 1:
        result = next(iter(results_by_alpha.values()))
        print(f"  {language}: {result}")
        return result

    for alpha in fusion_alphas:
        key = alpha_result_key(alpha)
        print(f"  {language} / {key}: {results_by_alpha[key]}")
    return results_by_alpha


def main():
    parser = argparse.ArgumentParser(description="Evaluate a top-k reranker on text2code retrieval.")

    parser.add_argument("--retriever-model-name", type=str, default="microsoft/unixcoder-base")
    parser.add_argument("--retriever-adapter-path", type=str, required=True)
    parser.add_argument("--reranker-dir", type=str, required=True)
    parser.add_argument("--reranker-model-name", type=str, default="microsoft/unixcoder-base")

    parser.add_argument("--languages", type=str, nargs="+", default=["python"])
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--experiment-name", type=str, default="")

    parser.add_argument("--retriever-pooling", type=str, default="auto", choices=["auto", "mean", "last_token", "attention"])
    parser.add_argument("--query-prefix", type=str, default=None)
    parser.add_argument("--code-prefix", type=str, default=None)
    parser.add_argument("--retriever-max-length", type=int, default=None)
    parser.add_argument("--pooling-hidden-dim", type=int, default=768)

    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--code-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--reranker-max-query-length", type=int, default=96)
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument("--code-chunk-size", type=int, default=4096)
    parser.add_argument("--fusion-alpha", type=float, default=0.0)
    parser.add_argument(
        "--fusion-alphas",
        type=float,
        nargs="+",
        default=None,
        help="Evaluate multiple fusion alpha values in one reranking pass. Overrides --fusion-alpha.",
    )
    parser.add_argument("--fusion-normalization", type=str, default="zscore", choices=["none", "zscore", "minmax"])
    parser.add_argument("--candidate-source", type=str, default="dense", choices=["dense", "dense_tfidf", "dense_coderank"])
    parser.add_argument("--dense-candidate-k", type=int, default=None)
    parser.add_argument("--tfidf-candidate-k", type=int, default=30)
    parser.add_argument("--tfidf-query-chunk-size", type=int, default=512)
    parser.add_argument("--tfidf-code-chunk-size", type=int, default=20000)
    parser.add_argument("--tfidf-max-features", type=int, default=50000)
    parser.add_argument("--coderank-candidate-k", type=int, default=50)
    parser.add_argument("--coderank-model", type=str, default="nomic-ai/CodeRankEmbed")
    parser.add_argument(
        "--coderank-backend",
        type=str,
        default="auto",
        choices=["auto", "sentence_transformer", "nomic_transformers"],
        help="Embedding backend for dense_coderank candidates. Use nomic_transformers for nomic-embed-code.",
    )
    parser.add_argument(
        "--coderank-dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for nomic_transformers candidate encoder.",
    )
    parser.add_argument("--coderank-query-prefix", type=str, default="Represent this query for searching relevant code: ")
    parser.add_argument("--coderank-code-prefix", type=str, default="")
    parser.add_argument("--coderank-max-seq-length", type=int, default=512)
    parser.add_argument("--coderank-batch-size", type=int, default=32)
    parser.add_argument("--coderank-query-chunk-size", type=int, default=64)
    parser.add_argument("--coderank-code-chunk-size", type=int, default=4096)
    parser.add_argument("--coderank-cache-dir", type=str, default="")
    parser.add_argument("--coderank-trust-remote-code", action="store_true")
    parser.add_argument(
        "--coderank-score-mode",
        type=str,
        default="dense",
        choices=["dense", "coderank", "dense_coderank"],
        help="For dense_coderank candidates, choose the retrieval score fused with reranker logits.",
    )
    parser.add_argument(
        "--coderank-score-alpha",
        type=float,
        default=1.0,
        help="Weight for CodeRank score when --coderank-score-mode dense_coderank.",
    )
    parser.add_argument(
        "--rank-output-dir",
        type=str,
        default="",
        help="Optional directory for saving per-query 0-based final rank arrays.",
    )
    parser.add_argument(
        "--route-feature-output-dir",
        type=str,
        default="",
        help="Optional directory for saving per-query route feature .npz files for adaptive routing.",
    )

    args = parser.parse_args()

    retriever_pooling_mode, query_prefix, code_prefix, retriever_max_length, pooling_hidden_dim = resolve_eval_options(
        args.retriever_adapter_path,
        args.retriever_pooling,
        args.query_prefix,
        args.code_prefix,
        args.retriever_max_length,
        args.pooling_hidden_dim,
    )

    retriever_model, retriever_tokenizer = load_retriever_model(
        args.retriever_model_name,
        args.retriever_adapter_path,
        retriever_pooling_mode,
        pooling_hidden_dim,
    )

    reranker_model, reranker_tokenizer = load_reranker(
        reranker_dir=args.reranker_dir,
        model_name=args.reranker_model_name,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for reranker evaluation.")

    device = torch.device("cuda")

    fusion_alphas = args.fusion_alphas if args.fusion_alphas is not None else [args.fusion_alpha]
    fusion_alphas = list(dict.fromkeys(float(alpha) for alpha in fusion_alphas))
    multi_alpha = len(fusion_alphas) > 1
    if multi_alpha:
        print(f"Evaluating fusion alphas in one pass: {[format_alpha(a) for a in fusion_alphas]}")

    dense_candidate_k = args.dense_candidate_k
    if dense_candidate_k is None:
        if args.candidate_source == "dense":
            dense_candidate_k = args.top_k
        elif args.candidate_source == "dense_tfidf":
            dense_candidate_k = max(0, args.top_k - args.tfidf_candidate_k)
        elif args.candidate_source == "dense_coderank":
            dense_candidate_k = max(0, args.top_k - args.coderank_candidate_k)
        else:
            raise ValueError(f"Unknown candidate source: {args.candidate_source}")

    all_results = {alpha_result_key(alpha): {} for alpha in fusion_alphas} if multi_alpha else {}

    for lang in args.languages:
        result = rerank_and_eval_language(
            retriever_model=retriever_model,
            retriever_tokenizer=retriever_tokenizer,
            reranker_model=reranker_model,
            reranker_tokenizer=reranker_tokenizer,
            language=lang,
            split=args.split,
            data_root=args.data_root,
            retriever_pooling_mode=retriever_pooling_mode,
            query_prefix=query_prefix or "",
            code_prefix=code_prefix or "",
            retriever_max_length=retriever_max_length,
            text_batch_size=args.text_batch_size,
            code_batch_size=args.code_batch_size,
            num_workers=args.num_workers,
            device=device,
            top_k=args.top_k,
            reranker_batch_size=args.reranker_batch_size,
            reranker_max_length=args.reranker_max_length,
            reranker_max_query_length=args.reranker_max_query_length,
            query_chunk_size=args.query_chunk_size,
            code_chunk_size=args.code_chunk_size,
            fusion_alphas=fusion_alphas,
            fusion_normalization=args.fusion_normalization,
            candidate_source=args.candidate_source,
            dense_candidate_k=dense_candidate_k,
            tfidf_candidate_k=args.tfidf_candidate_k,
            tfidf_query_chunk_size=args.tfidf_query_chunk_size,
            tfidf_code_chunk_size=args.tfidf_code_chunk_size,
            tfidf_max_features=args.tfidf_max_features,
            coderank_candidate_k=args.coderank_candidate_k,
            coderank_model=args.coderank_model,
            coderank_backend=args.coderank_backend,
            coderank_dtype=args.coderank_dtype,
            coderank_query_prefix=args.coderank_query_prefix,
            coderank_code_prefix=args.coderank_code_prefix,
            coderank_max_seq_length=args.coderank_max_seq_length,
            coderank_batch_size=args.coderank_batch_size,
            coderank_query_chunk_size=args.coderank_query_chunk_size,
            coderank_code_chunk_size=args.coderank_code_chunk_size,
            coderank_cache_dir=args.coderank_cache_dir,
            coderank_trust_remote_code=args.coderank_trust_remote_code,
            coderank_score_mode=args.coderank_score_mode,
            coderank_score_alpha=args.coderank_score_alpha,
            rank_output_dir=args.rank_output_dir,
            route_feature_output_dir=args.route_feature_output_dir,
        )

        if multi_alpha:
            for alpha in fusion_alphas:
                key = alpha_result_key(alpha)
                all_results[key][lang] = result[key]
        else:
            all_results[lang] = result

    print("\n" + "=" * 60)
    if multi_alpha:
        for alpha in fusion_alphas:
            key = alpha_result_key(alpha)
            all_results[key]["average"] = average_metric_results(list(all_results[key].values()))
            print(f"{key} Average: {all_results[key]['average']}")
    else:
        all_results["average"] = average_metric_results(list(all_results.values()))
        print(f"Average: {all_results['average']}")
    print("=" * 60)

    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    default_name = f"rerank_{args.split}_eval"
    json_output = results_dir / f"{args.experiment_name or default_name}_eval_results.json"
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"JSON results saved to {json_output}")


if __name__ == "__main__":
    main()
