from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer

from eval_llm_verifier import retrieve_topk_from_cache
from eval_nomic_embed import get_code_text, get_query_text


EXPERIMENT_FAMILY = "work2_candidate_recovery"
EXPECTED_DENSE_ACTION = "dense500_selector_source"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def scalar_string(archive: np.lib.npyio.NpzFile, key: str) -> str:
    if key not in archive:
        raise ValueError(f"Archive is missing {key!r} metadata.")
    return str(np.asarray(archive[key]).item())


def validate_cache(path: Path, rows: int, hidden_size: int) -> np.ndarray:
    values = np.load(path, mmap_mode="r")
    if values.shape != (rows, hidden_size):
        raise ValueError(
            f"Embedding cache shape mismatch for {path}: "
            f"expected {(rows, hidden_size)}, got {values.shape}."
        )
    if values.dtype != np.float16:
        raise ValueError(f"Embedding cache must use float16: {path}, {values.dtype}")
    for start in range(0, rows, 4096):
        if not np.isfinite(values[start : start + 4096]).all():
            raise ValueError(f"Embedding cache contains NaN or infinity: {path}")
    return values


@torch.no_grad()
def encode_texts(
    model: RobertaModel,
    tokenizer: RobertaTokenizer,
    texts: Sequence[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
    description: str,
) -> np.ndarray:
    parts: List[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc=description):
        batch = list(texts[start : start + batch_size])
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        hidden = model(
            encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )[0]
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = torch.sum(hidden * mask, dim=1) / torch.clamp(
            mask.sum(dim=1), min=1e-6
        )
        pooled = F.normalize(pooled.float(), dim=1)
        parts.append(pooled.cpu().numpy().astype(np.float16, copy=False))
        del encoded, hidden, mask, pooled
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = np.concatenate(parts, axis=0)
    if result.shape[0] != len(texts) or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid embedding output for {description}: {result.shape}")
    return result


def load_or_encode(
    path: Path,
    model: RobertaModel,
    tokenizer: RobertaTokenizer,
    texts: Sequence[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
    hidden_size: int,
    description: str,
    reuse_cache: bool,
) -> np.ndarray:
    if path.exists():
        if not reuse_cache:
            raise FileExistsError(
                f"Refusing to overwrite embedding cache without --reuse-cache: {path}"
            )
        print(f"Reusing {description} cache: {path}")
        return validate_cache(path, len(texts), hidden_size)

    values = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        description=description,
    )
    if values.shape != (len(texts), hidden_size):
        raise RuntimeError(
            f"Unexpected {description} shape: {values.shape}; "
            f"expected {(len(texts), hidden_size)}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values)
    print(f"Saved {description} cache: {path}")
    return np.load(path, mmap_mode="r")


def reorder_novel_candidates(
    external_indices: np.ndarray,
    external_scores: np.ndarray,
    dense_baseline: np.ndarray,
    novel_budget: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    reordered_indices = np.empty_like(external_indices)
    reordered_scores = np.empty_like(external_scores)
    novel_counts = np.empty(len(external_indices), dtype=np.int32)

    for row_index, (candidate_row, score_row, dense_row) in enumerate(
        zip(external_indices, external_scores, dense_baseline)
    ):
        dense_set = {int(value) for value in dense_row}
        novel_positions = [
            position
            for position, value in enumerate(candidate_row)
            if int(value) not in dense_set
        ]
        if len(novel_positions) < novel_budget:
            raise RuntimeError(
                f"Query {row_index} has only {len(novel_positions)} external candidates "
                f"outside Dense top-{dense_baseline.shape[1]}; need {novel_budget}. "
                "Increase --external-retrieval-k."
            )
        selected = novel_positions[:novel_budget]
        selected_set = set(selected)
        remaining = [
            position
            for position in range(len(candidate_row))
            if position not in selected_set
        ]
        order = np.asarray(selected + remaining, dtype=np.int64)
        reordered_indices[row_index] = candidate_row[order]
        reordered_scores[row_index] = score_row[order]
        novel_counts[row_index] = len(novel_positions)

    return reordered_indices, reordered_scores, novel_counts


def build_language(
    args: argparse.Namespace,
    language: str,
    model: RobertaModel,
    tokenizer: RobertaTokenizer,
    device: torch.device,
    hidden_size: int,
) -> Dict[str, Any]:
    print(f"\n--- UniXcoder base candidates for {language} / valid ---")
    data_root = Path(args.data_root)
    query_path = data_root / language / "valid.jsonl"
    code_path = data_root / language / "codebase.jsonl"
    dense_path = Path(args.dense_pool_pattern.format(language=language)).expanduser()
    for path in (query_path, code_path, dense_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    query_rows = read_jsonl(query_path)
    code_rows = read_jsonl(code_path)
    queries = [get_query_text(row) for row in query_rows]
    codes = [get_code_text(row) for row in code_rows]

    with np.load(dense_path, allow_pickle=False) as archive:
        dense_indices = np.asarray(archive["indices"], dtype=np.int64)
        ground_truth = np.asarray(archive["ground_truth_indices"], dtype=np.int64)
        base_ranks = np.asarray(archive["base_ranks"], dtype=np.int64)
        family = scalar_string(archive, "experiment_family")
        source = scalar_string(archive, "source")
        action_name = scalar_string(archive, "action_name")
    if family != EXPERIMENT_FAMILY or source != "dense":
        raise ValueError(f"Unexpected Dense source metadata: {dense_path}")
    if action_name != EXPECTED_DENSE_ACTION:
        raise ValueError(f"Unexpected Dense source action {action_name!r}: {dense_path}")
    if dense_indices.shape[0] != len(queries) or dense_indices.shape[1] < args.baseline_pool_k:
        raise ValueError(f"Dense source shape mismatch: {dense_indices.shape}")
    if ground_truth.shape != (len(queries),) or base_ranks.shape != (len(queries),):
        raise ValueError("Ground-truth/base-rank query alignment mismatch.")
    if np.any(ground_truth < 0) or np.any(ground_truth >= len(codes)):
        raise ValueError("Ground-truth indices are outside the codebase.")

    cache_dir = Path(args.cache_dir)
    query_cache = cache_dir / f"{language}_valid_unixcoder_base_mean_msl{args.max_length}.query.npy"
    code_cache = cache_dir / f"{language}_unixcoder_base_mean_msl{args.max_length}.code.npy"
    query_embeddings = load_or_encode(
        query_cache,
        model,
        tokenizer,
        queries,
        device,
        args.query_batch_size,
        args.max_length,
        hidden_size,
        f"{language} valid query",
        args.reuse_cache,
    )
    code_embeddings = load_or_encode(
        code_cache,
        model,
        tokenizer,
        codes,
        device,
        args.code_batch_size,
        args.max_length,
        hidden_size,
        f"{language} code",
        args.reuse_cache,
    )

    external_scores, external_indices, external_ranks = retrieve_topk_from_cache(
        query_embeds=query_embeddings,
        code_embeds=code_embeddings,
        gt_indices=ground_truth,
        device=device,
        top_k=args.external_retrieval_k,
        query_chunk_size=args.query_chunk_size,
        code_chunk_size=args.code_chunk_size,
    )
    dense_baseline = dense_indices[:, : args.baseline_pool_k]
    external_indices, external_scores, novel_counts = reorder_novel_candidates(
        external_indices,
        external_scores,
        dense_baseline,
        args.novel_budget,
    )

    selected_novel = external_indices[:, : args.novel_budget]
    dense_hits = base_ranks < args.baseline_pool_k
    external_hits = external_ranks < args.external_retrieval_k
    selected_hits = np.any(selected_novel == ground_truth[:, None], axis=1)
    union_hits = dense_hits | selected_hits

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{language}_valid_unixcoder_base_novel_candidates.npz"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite external candidates: {output_path}")
    np.savez_compressed(
        output_path,
        indices=external_indices.astype(np.int32),
        scores=external_scores.astype(np.float32),
        external_ranks=external_ranks.astype(np.int32),
        ground_truth_indices=ground_truth.astype(np.int32),
        base_ranks=base_ranks.astype(np.int32),
        novel_candidate_counts=novel_counts.astype(np.int32),
        experiment_family=np.asarray(EXPERIMENT_FAMILY),
        source_split=np.asarray("valid"),
        language=np.asarray(language),
        generator=np.asarray("unixcoder_base_mean_pooling"),
        baseline_pool_k=np.asarray(args.baseline_pool_k, dtype=np.int32),
        novel_budget=np.asarray(args.novel_budget, dtype=np.int32),
        external_retrieval_k=np.asarray(args.external_retrieval_k, dtype=np.int32),
        dense_source_pool=np.asarray(str(dense_path.resolve())),
        leakage_guard=np.asarray(
            "frozen base model; valid queries only; no labels used for candidate selection"
        ),
    )

    result = {
        "language": language,
        "queries": len(queries),
        "codes": len(codes),
        "external_retrieval_k": args.external_retrieval_k,
        "novel_budget": args.novel_budget,
        "minimum_novel_candidates_available": int(novel_counts.min()),
        "dense_top100_hits": int(dense_hits.sum()),
        "unixcoder_topk_hits": int(external_hits.sum()),
        "novel7_hits": int(selected_hits.sum()),
        "union_hits": int(union_hits.sum()),
        "recovered_from_dense_top100": int(np.sum(~dense_hits & selected_hits)),
        "dense_top100_recall_pct": 100.0 * float(dense_hits.mean()),
        "union_recall_pct": 100.0 * float(union_hits.mean()),
        "union_delta_pp": 100.0 * float((union_hits.astype(np.int8) - dense_hits.astype(np.int8)).mean()),
        "output_file": str(output_path.resolve()),
        "query_cache": str(query_cache.resolve()),
        "code_cache": str(code_cache.resolve()),
        "test_accessed": False,
    }
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export valid-only candidates from a frozen UniXcoder base model for "
            "Work 2 complementary-retrieval screening."
        )
    )
    parser.add_argument("--languages", nargs="+", default=["ruby"])
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dense-pool-pattern", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--baseline-pool-k", type=int, default=100)
    parser.add_argument("--novel-budget", type=int, default=7)
    parser.add_argument("--external-retrieval-k", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--code-batch-size", type=int, default=32)
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument("--code-chunk-size", type=int, default=4096)
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()
    if args.split != "valid":
        raise ValueError(
            "Test is locked until the independent candidate generator is frozen on valid."
        )
    if args.novel_budget != 7:
        raise ValueError("The preregistered complementary-candidate budget is exactly 7.")
    if args.external_retrieval_k < args.novel_budget:
        raise ValueError("external-retrieval-k must be at least novel-budget.")
    if min(
        args.baseline_pool_k,
        args.max_length,
        args.query_batch_size,
        args.code_batch_size,
        args.query_chunk_size,
        args.code_chunk_size,
    ) <= 0:
        raise ValueError("Retrieval sizes must be positive.")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for UniXcoder candidate export.")
    model_path = Path(args.model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "valid_unixcoder_base_candidate_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite summary: {summary_path}")

    tokenizer = RobertaTokenizer.from_pretrained(str(model_path))
    model = RobertaModel.from_pretrained(str(model_path))
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    hidden_size = int(model.config.hidden_size)

    results = [
        build_language(args, language, model, tokenizer, device, hidden_size)
        for language in args.languages
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved summary: {summary_path.resolve()}")
    print("UNIXCODER_BASE_VALID_CANDIDATE_EXPORT_PASSED")


if __name__ == "__main__":
    main()
