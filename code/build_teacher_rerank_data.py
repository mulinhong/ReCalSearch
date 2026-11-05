import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from config import DATA_ROOT


DEFAULT_QUERY_PREFIX = "Represent this query for searching relevant code: "
NOMIC_MODEL_MARKERS = ("nomic-embed-code",)


def resolve_teacher_backend(model_name: str, backend: str) -> str:
    if backend != "auto":
        return backend

    normalized = model_name.replace("\\", "/").lower()
    if any(marker in normalized for marker in NOMIC_MODEL_MARKERS):
        return "nomic_transformers"
    return "sentence_transformer"


def resolve_torch_dtype(dtype_name: str):
    import torch

    if dtype_name == "auto":
        return torch.float16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported teacher dtype: {dtype_name}")


def resolve_numpy_dtype(dtype_name: str):
    if dtype_name == "float32":
        return np.float32
    if dtype_name == "float16":
        return np.float16
    raise ValueError(f"Unsupported embedding dtype: {dtype_name}")


def get_code_text(example: Dict[str, Any]) -> str:
    if "code" in example and example["code"]:
        return str(example["code"])
    if "code_tokens" in example and example["code_tokens"]:
        value = example["code_tokens"]
        if isinstance(value, list):
            return " ".join(map(str, value))
        return str(value)
    if "function_tokens" in example and example["function_tokens"]:
        value = example["function_tokens"]
        if isinstance(value, list):
            return " ".join(map(str, value))
        return str(value)
    raise KeyError(f"Could not find code field in example keys: {list(example.keys())}")


def read_jsonl(path: Path, max_rows: Optional[int] = None, shuffle: bool = False, seed: int = 42) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows is not None and not shuffle and len(rows) >= max_rows:
                break

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)
        if max_rows is not None:
            rows = rows[:max_rows]
    return rows


def read_codebase_codes(path: Path) -> List[str]:
    codes = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                codes.append(get_code_text(json.loads(line)))
    return codes


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def cache_paths(
    cache_dir: Path,
    input_file: Path,
    language: str,
    teacher_model: str,
    max_seq_length: int,
    teacher_backend: str,
    teacher_query_prefix: str,
    teacher_code_prefix: str,
    embedding_dtype: str,
) -> Dict[str, Path]:
    cache_key = json.dumps(
        {
            "teacher_model": teacher_model,
            "teacher_backend": teacher_backend,
            "teacher_query_prefix": teacher_query_prefix,
            "teacher_code_prefix": teacher_code_prefix,
            "max_seq_length": max_seq_length,
            "embedding_dtype": embedding_dtype,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:10]
    prefix = cache_dir / f"{language}_{input_file.stem}_{sanitize_name(teacher_model)}_msl{max_seq_length}_{digest}"
    return {
        "query": prefix.with_suffix(".query.npy"),
        "code": prefix.with_suffix(".code.npy"),
        "code_indices": prefix.with_suffix(".code_indices.json"),
        "inline_positive": prefix.with_suffix(".inline_positive.npy"),
        "inline_rows": prefix.with_suffix(".inline_rows.json"),
    }


class NomicTransformersTeacher:
    """Minimal sentence-embedding wrapper for nomic-embed-code.

    The local nomic-embed-code checkpoint does not load reliably through
    SentenceTransformer in some environments, so this backend mirrors
    eval_nomic_embed.py: slow tokenizer, AutoModel, last-token pooling.
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        max_seq_length: int,
        trust_remote_code: bool,
        dtype_name: str,
    ):
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for nomic_transformers teacher, but CUDA is not available.")

        self.torch = torch
        self.F = F
        self.device = torch.device(device)
        self.max_seq_length = max_seq_length
        dtype = resolve_torch_dtype(dtype_name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

    def encode(
        self,
        texts: List[str],
        batch_size: int,
        show_progress_bar: bool = True,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ):
        torch = self.torch
        iterator = range(0, len(texts), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, total=(len(texts) + batch_size - 1) // batch_size, desc="Batches")

        embeds = []
        with torch.no_grad():
            for start in iterator:
                batch_texts = texts[start:start + batch_size]
                batch = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt",
                ).to(self.device)

                hidden = self.model(**batch).last_hidden_state
                last_pos = batch["attention_mask"].sum(dim=1) - 1
                batch_idx = torch.arange(hidden.size(0), device=self.device)
                pooled = hidden[batch_idx, last_pos]
                pooled = pooled.float()
                if normalize_embeddings:
                    pooled = self.F.normalize(pooled, dim=1)
                embeds.append(pooled.cpu())

                del batch, hidden, pooled
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        embeds = torch.cat(embeds, dim=0)
        return embeds.numpy() if convert_to_numpy else embeds


def load_teacher_model(
    model_name: str,
    device: str,
    max_seq_length: int,
    trust_remote_code: bool,
    backend: str,
    dtype_name: str,
):
    backend = resolve_teacher_backend(model_name, backend)
    print(f"Teacher backend: {backend}")

    if backend == "nomic_transformers":
        return NomicTransformersTeacher(
            model_name=model_name,
            device=device,
            max_seq_length=max_seq_length,
            trust_remote_code=trust_remote_code,
            dtype_name=dtype_name,
        )

    if backend != "sentence_transformer":
        raise ValueError(f"Unknown teacher backend: {backend}")

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise ImportError(
            "sentence-transformers is required for teacher rerank data. "
            "Install it on the training machine with: pip install -U sentence-transformers"
        ) from exc

    kwargs = {"device": device}
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


def encode_texts(
    model,
    texts: List[str],
    batch_size: int,
    prefix: str = "",
    desc: str = "Encoding",
    output_dtype: str = "float32",
) -> np.ndarray:
    if prefix:
        texts = [prefix + text for text in texts]
    embeds = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeds = np.asarray(embeds, dtype=resolve_numpy_dtype(output_dtype))
    if embeds.ndim != 2:
        raise ValueError(f"Expected 2D embeddings for {desc}, got shape={embeds.shape}")
    return embeds


def collect_candidate_code_indices(rows: List[Dict[str, Any]]) -> Tuple[List[int], Dict[int, str]]:
    code_indices = set()
    inline_positive_texts: Dict[int, str] = {}

    for row_idx, row in enumerate(rows):
        positive = row.get("positive", {})
        pos_idx = positive.get("code_index")
        if pos_idx is None:
            code = positive.get("code")
            if code:
                inline_positive_texts[row_idx] = str(code)
        else:
            code_indices.add(int(pos_idx))

        for code_idx in row.get("negative_code_indices", []):
            code_indices.add(int(code_idx))

    return sorted(code_indices), inline_positive_texts


def normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - mean) / std).astype(np.float32)


def mixed_sort_scores(items: List[Dict[str, Any]], teacher_sort_alpha: float) -> np.ndarray:
    teacher = np.array([item["teacher_score"] for item in items], dtype=np.float32)
    retrieval = []
    for item in items:
        score = item.get("retrieval_score")
        if score is None:
            score = -float(item.get("retrieval_rank", 10**9))
        retrieval.append(float(score))
    retrieval = np.array(retrieval, dtype=np.float32)

    alpha = float(np.clip(teacher_sort_alpha, 0.0, 1.0))
    return alpha * normalize(teacher) + (1.0 - alpha) * normalize(retrieval)


def filter_and_sort_negatives(
    negatives: List[Dict[str, Any]],
    positive_teacher_score: float,
    false_negative_margin: float,
    disable_filter: bool,
    min_negatives: int,
    max_negatives: int,
    teacher_sort_alpha: float,
) -> Tuple[List[Dict[str, Any]], int, bool]:
    if not negatives:
        return [], 0, False

    if disable_filter:
        kept = list(negatives)
        filtered_count = 0
        rescued = False
    else:
        threshold = positive_teacher_score - false_negative_margin
        safe = [item for item in negatives if item["teacher_score"] < threshold]
        suspicious = [item for item in negatives if item["teacher_score"] >= threshold]
        target_min = min(min_negatives, max_negatives if max_negatives > 0 else min_negatives)
        rescued = False
        if len(safe) < target_min and suspicious:
            needed = target_min - len(safe)
            rescue = sorted(suspicious, key=lambda item: item["teacher_score"])[:needed]
            kept = safe + rescue
            rescued = bool(rescue)
        else:
            kept = safe
        filtered_count = len(negatives) - len(kept)

    if kept:
        sort_scores = mixed_sort_scores(kept, teacher_sort_alpha)
        for item, sort_score in zip(kept, sort_scores.tolist()):
            item["teacher_sort_score"] = float(sort_score)
        kept.sort(key=lambda item: (-item["teacher_sort_score"], int(item.get("retrieval_rank", 10**9))))

    if max_negatives > 0:
        kept = kept[:max_negatives]
    return kept, filtered_count, rescued


def score_and_write_rows(
    rows: List[Dict[str, Any]],
    output_file: Path,
    query_embeds: np.ndarray,
    code_embeds: np.ndarray,
    code_pos: Dict[int, int],
    inline_positive_embeds: Optional[np.ndarray],
    inline_row_pos: Dict[int, int],
    args,
) -> Dict[str, Any]:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_groups = 0
    total_input_negatives = 0
    total_output_negatives = 0
    total_filtered_negatives = 0
    groups_with_filter = 0
    groups_with_rescue = 0
    groups_without_positive_teacher = 0

    with output_file.open("w", encoding="utf-8") as f:
        for row_idx, row in enumerate(tqdm(rows, desc="Teacher scoring groups")):
            query_embed = query_embeds[row_idx]
            positive = dict(row.get("positive", {}))
            pos_idx = positive.get("code_index")

            if pos_idx is None:
                inline_pos = inline_row_pos.get(row_idx)
                if inline_pos is None or inline_positive_embeds is None:
                    groups_without_positive_teacher += 1
                    continue
                pos_embed = inline_positive_embeds[inline_pos]
            else:
                pos_embed = code_embeds[code_pos[int(pos_idx)]]

            positive_teacher_score = float(np.dot(query_embed, pos_embed))
            negative_indices = [int(x) for x in row.get("negative_code_indices", [])]
            negative_ranks = [int(x) for x in row.get("negative_retrieval_ranks", [])]
            negative_scores = [
                float(x) if x is not None else None
                for x in row.get("negative_retrieval_scores", [])
            ]

            negatives = []
            for i, code_idx in enumerate(negative_indices):
                if code_idx not in code_pos:
                    continue
                rank = negative_ranks[i] if i < len(negative_ranks) else i + 1
                retrieval_score = negative_scores[i] if i < len(negative_scores) else None
                teacher_score = float(np.dot(query_embed, code_embeds[code_pos[code_idx]]))
                negatives.append(
                    {
                        "code_index": code_idx,
                        "retrieval_rank": rank,
                        "retrieval_score": retrieval_score,
                        "teacher_score": teacher_score,
                    }
                )

            kept, filtered_count, rescued = filter_and_sort_negatives(
                negatives=negatives,
                positive_teacher_score=positive_teacher_score,
                false_negative_margin=args.false_negative_margin,
                disable_filter=args.disable_false_negative_filter,
                min_negatives=args.min_negatives,
                max_negatives=args.max_negatives,
                teacher_sort_alpha=args.teacher_sort_alpha,
            )
            if not kept:
                continue

            positive["teacher_score"] = positive_teacher_score
            output_row = dict(row)
            output_row["format"] = "grouped_indices_v1"
            output_row["positive"] = positive
            output_row["positive_teacher_score"] = positive_teacher_score
            output_row["negative_code_indices"] = [item["code_index"] for item in kept]
            output_row["negative_retrieval_ranks"] = [item["retrieval_rank"] for item in kept]
            output_row["negative_retrieval_scores"] = [item["retrieval_score"] for item in kept]
            output_row["negative_teacher_scores"] = [item["teacher_score"] for item in kept]
            output_row["negative_teacher_sort_scores"] = [item["teacher_sort_score"] for item in kept]
            output_row["teacher_model"] = args.teacher_model
            output_row["teacher_backend"] = resolve_teacher_backend(args.teacher_model, args.teacher_backend)
            output_row["teacher_query_prefix"] = args.teacher_query_prefix
            output_row["teacher_filter"] = {
                "false_negative_margin": args.false_negative_margin,
                "disable_false_negative_filter": args.disable_false_negative_filter,
                "teacher_sort_alpha": args.teacher_sort_alpha,
                "embedding_dtype": args.embedding_dtype,
                "input_negatives": len(negatives),
                "output_negatives": len(kept),
                "filtered_negatives": filtered_count,
                "rescued_to_min_negatives": rescued,
            }

            f.write(json.dumps(output_row, ensure_ascii=False) + "\n")

            total_groups += 1
            total_input_negatives += len(negatives)
            total_output_negatives += len(kept)
            total_filtered_negatives += filtered_count
            groups_with_filter += int(filtered_count > 0)
            groups_with_rescue += int(rescued)

    return {
        "saved_groups": total_groups,
        "avg_input_negatives": round(total_input_negatives / max(total_groups, 1), 2),
        "avg_output_negatives": round(total_output_negatives / max(total_groups, 1), 2),
        "filtered_negatives": total_filtered_negatives,
        "groups_with_filter": groups_with_filter,
        "groups_with_rescue": groups_with_rescue,
        "groups_without_positive_teacher": groups_without_positive_teacher,
    }


def load_or_encode_embeddings(
    rows: List[Dict[str, Any]],
    codebase_codes: List[str],
    code_indices: List[int],
    inline_positive_texts: Dict[int, str],
    args,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], Optional[np.ndarray], Dict[int, int]]:
    cache = None
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        resolved_backend = resolve_teacher_backend(args.teacher_model, args.teacher_backend)
        cache = cache_paths(
            cache_dir,
            Path(args.input_file),
            args.language,
            args.teacher_model,
            args.teacher_max_seq_length,
            resolved_backend,
            args.teacher_query_prefix,
            args.teacher_code_prefix,
            args.embedding_dtype,
        )

    if cache and not args.overwrite_cache and all(path.exists() for path in cache.values() if path.name.endswith((".npy", ".json"))):
        print(f"Loading teacher embedding cache from: {cache['query'].parent}")
        query_embeds = np.load(cache["query"])
        code_embeds = np.load(cache["code"])
        with cache["code_indices"].open("r", encoding="utf-8") as f:
            cached_code_indices = [int(x) for x in json.load(f)]
        missing_code_indices = sorted(set(code_indices) - set(cached_code_indices))
        if query_embeds.shape[0] == len(rows) and not missing_code_indices:
            code_pos = {idx: pos for pos, idx in enumerate(cached_code_indices)}

            inline_positive_embeds = None
            inline_row_pos = {}
            if cache["inline_positive"].exists() and cache["inline_rows"].exists():
                inline_positive_embeds = np.load(cache["inline_positive"])
                with cache["inline_rows"].open("r", encoding="utf-8") as f:
                    inline_row_pos = {int(k): int(v) for k, v in json.load(f).items()}
            return query_embeds, code_embeds, code_pos, inline_positive_embeds, inline_row_pos

        reason = []
        if query_embeds.shape[0] != len(rows):
            reason.append(f"query_count cache={query_embeds.shape[0]} current={len(rows)}")
        if missing_code_indices:
            preview = ", ".join(map(str, missing_code_indices[:5]))
            reason.append(f"missing_code_indices={len(missing_code_indices)} preview=[{preview}]")
        print(f"Teacher cache is incompatible ({'; '.join(reason)}); rebuilding cache.")

    device = "cuda" if args.device == "auto" else args.device
    teacher = load_teacher_model(
        model_name=args.teacher_model,
        device=device,
        max_seq_length=args.teacher_max_seq_length,
        trust_remote_code=args.trust_remote_code,
        backend=args.teacher_backend,
        dtype_name=args.teacher_dtype,
    )

    queries = [str(row["query"]) for row in rows]
    print(f"Encoding {len(queries)} queries with teacher: {args.teacher_model}")
    query_embeds = encode_texts(
        teacher,
        queries,
        batch_size=args.teacher_batch_size,
        prefix=args.teacher_query_prefix,
        desc="teacher queries",
        output_dtype=args.embedding_dtype,
    )

    code_texts = [codebase_codes[idx] for idx in code_indices]
    print(f"Encoding {len(code_texts)} unique candidate codes with teacher: {args.teacher_model}")
    code_embeds = encode_texts(
        teacher,
        code_texts,
        batch_size=args.teacher_batch_size,
        prefix=args.teacher_code_prefix,
        desc="teacher codes",
        output_dtype=args.embedding_dtype,
    )
    code_pos = {idx: pos for pos, idx in enumerate(code_indices)}

    inline_positive_embeds = None
    inline_row_pos: Dict[int, int] = {}
    if inline_positive_texts:
        inline_rows = sorted(inline_positive_texts)
        inline_texts = [inline_positive_texts[row_idx] for row_idx in inline_rows]
        print(f"Encoding {len(inline_texts)} inline positive codes with teacher")
        inline_positive_embeds = encode_texts(
            teacher,
            inline_texts,
            batch_size=args.teacher_batch_size,
            prefix=args.teacher_code_prefix,
            desc="inline positives",
            output_dtype=args.embedding_dtype,
        )
        inline_row_pos = {row_idx: pos for pos, row_idx in enumerate(inline_rows)}

    if cache:
        np.save(cache["query"], query_embeds)
        np.save(cache["code"], code_embeds)
        with cache["code_indices"].open("w", encoding="utf-8") as f:
            json.dump(code_indices, f)
        if inline_positive_embeds is not None:
            np.save(cache["inline_positive"], inline_positive_embeds)
            with cache["inline_rows"].open("w", encoding="utf-8") as f:
                json.dump(inline_row_pos, f)
        else:
            np.save(cache["inline_positive"], np.zeros((0, code_embeds.shape[1]), dtype=resolve_numpy_dtype(args.embedding_dtype)))
            with cache["inline_rows"].open("w", encoding="utf-8") as f:
                json.dump({}, f)

    return query_embeds, code_embeds, code_pos, inline_positive_embeds, inline_row_pos


def main():
    parser = argparse.ArgumentParser(description="Add teacher scores and false-negative filtering to reranker data.")
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--codebase-file", type=str, default=None)

    parser.add_argument("--teacher-model", type=str, default="nomic-ai/CodeRankEmbed")
    parser.add_argument(
        "--teacher-backend",
        type=str,
        default="auto",
        choices=["auto", "sentence_transformer", "nomic_transformers"],
        help="Use nomic_transformers for nomic-embed-code; auto selects it by model path/name.",
    )
    parser.add_argument(
        "--teacher-dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for nomic_transformers teacher.",
    )
    parser.add_argument("--teacher-query-prefix", type=str, default=DEFAULT_QUERY_PREFIX)
    parser.add_argument("--teacher-code-prefix", type=str, default="")
    parser.add_argument("--teacher-max-seq-length", type=int, default=512)
    parser.add_argument("--teacher-batch-size", type=int, default=32)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    parser.add_argument("--false-negative-margin", type=float, default=0.02)
    parser.add_argument("--disable-false-negative-filter", action="store_true")
    parser.add_argument("--teacher-sort-alpha", type=float, default=0.85)
    parser.add_argument("--max-negatives", type=int, default=63)
    parser.add_argument("--min-negatives", type=int, default=16)

    parser.add_argument("--cache-dir", type=str, default="")
    parser.add_argument(
        "--embedding-dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Dtype used to keep/save teacher embeddings. Use float16 for large nomic-embed-code runs.",
    )
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    codebase_file = Path(args.codebase_file) if args.codebase_file else Path(args.data_root) / args.language / "codebase.jsonl"

    print(f"Loading rerank groups from: {input_file}")
    rows = read_jsonl(input_file, max_rows=args.max_groups, shuffle=args.shuffle, seed=args.seed)
    print(f"Loaded {len(rows)} groups")

    print(f"Loading codebase from: {codebase_file}")
    codebase_codes = read_codebase_codes(codebase_file)
    print(f"Loaded {len(codebase_codes)} codebase codes")

    code_indices, inline_positive_texts = collect_candidate_code_indices(rows)
    print(f"Teacher will score {len(code_indices)} unique indexed codes and {len(inline_positive_texts)} inline positives")

    query_embeds, code_embeds, code_pos, inline_positive_embeds, inline_row_pos = load_or_encode_embeddings(
        rows=rows,
        codebase_codes=codebase_codes,
        code_indices=code_indices,
        inline_positive_texts=inline_positive_texts,
        args=args,
    )

    stats = score_and_write_rows(
        rows=rows,
        output_file=output_file,
        query_embeds=query_embeds,
        code_embeds=code_embeds,
        code_pos=code_pos,
        inline_positive_embeds=inline_positive_embeds,
        inline_row_pos=inline_row_pos,
        args=args,
    )
    print(f"Saved teacher rerank data to: {output_file}")
    print(f"Teacher data stats: {stats}")


if __name__ == "__main__":
    main()
