from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from build_repairability_benchmark import model_identity, retrieve_embedding_topk
from build_residual_tail_groups import read_selected_jsonl
from eval_nomic_embed import get_code_text, get_query_text
from work2_repairability_core import (
    BASELINE_K,
    EXPERIMENT_FAMILY,
    LANGUAGES,
    METHOD_FAMILY,
    OPERATOR_BUDGET,
    RETRIEVAL_K,
    ROLE_NAMES,
    STATE_NAMES,
    atomic_save_npy,
    atomic_save_npz,
    atomic_write_json,
    checked_file,
    file_sha256,
    first_n_novel,
    ground_truth_ranks,
    role_counts,
    state_counts,
    state_labels_from_rank,
    validate_retrieval,
)


PROBE_PLAN_FORMAT = "work2_comparable_retriever_probe_plan_v1"
PROBE_REPORT_FORMAT = "work2_comparable_retriever_probe_report_v1"
QWEN3_RETRIEVAL_FORMAT = "work2_qwen3_retrievals_v1"
PROBE_CONTRACT_FORMAT = "work2_comparable_retriever_contract_v1"

PRIMARY_BASES: Tuple[str, ...] = ("nomic", "qwen3")
QWEN3_MODEL_DEFAULT = "/root/autodl-tmp/hf_models/Qwen3-Embedding-4B"
QWEN3_MAX_LENGTH = 512
QWEN3_EMBED_DIM = 2560
QWEN3_DTYPE = "bfloat16"
QWEN3_POOLING = "last_token_l2"
QWEN3_QUERY_TASK = (
    "Given a natural language query, retrieve the most relevant code snippet "
    "that implements the described functionality"
)
COMPARABILITY_AFR_RATIO_LOW = 0.5
COMPARABILITY_AFR_RATIO_HIGH = 2.0

OPERATOR_NAMES_BY_BASE: Dict[str, Tuple[str, ...]] = {
    "nomic": (
        "dense_next20",
        "qwen3_novel20",
        "unixcoder_novel20",
        "bm25_novel20",
    ),
    "qwen3": (
        "dense_next20",
        "nomic_novel20",
        "unixcoder_novel20",
        "bm25_novel20",
    ),
}


def replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def scalar_string(archive, key: str) -> str:
    if key not in archive:
        raise KeyError(f"Archive missing {key!r}")
    return str(np.asarray(archive[key]).item())


def scalar_bool(archive, key: str) -> bool:
    if key not in archive:
        raise KeyError(f"Archive missing {key!r}")
    return bool(np.asarray(archive[key]).item())


def verify_file_info(row: Mapping[str, Any], label: str) -> Path:
    path = Path(row["path"]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if "size" in row and int(path.stat().st_size) != int(row["size"]):
        raise RuntimeError(f"{label} size changed: {path}")
    if "sha256" in row and file_sha256(path) != str(row["sha256"]):
        raise RuntimeError(f"{label} sha256 changed: {path}")
    return path


def qwen3_query_text(query: str) -> str:
    return f"Instruct: {QWEN3_QUERY_TASK}\nQuery:{query}"


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = bool(
        attention_mask[:, -1].sum().item() == attention_mask.shape[0]
    )
    if left_padding:
        return last_hidden_states[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    return last_hidden_states[
        torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device),
        lengths,
    ]


def load_qwen3(model_path: Path, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    hidden = int(getattr(model.config, "hidden_size", -1))
    if hidden != QWEN3_EMBED_DIM:
        raise RuntimeError(
            f"Qwen3 hidden size {hidden} != frozen embedding dim {QWEN3_EMBED_DIM}."
        )
    return model, tokenizer


def encode_qwen3_batch(
    model,
    tokenizer,
    texts: Sequence[str],
    device: torch.device,
    is_query: bool,
) -> np.ndarray:
    if not texts:
        return np.empty((0, QWEN3_EMBED_DIM), dtype=np.float32)
    prepared = [qwen3_query_text(text) for text in texts] if is_query else list(texts)
    encoded = tokenizer(
        prepared,
        padding=True,
        truncation=True,
        max_length=QWEN3_MAX_LENGTH,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model(**encoded)
        pooled = last_token_pool(output.last_hidden_state, encoded["attention_mask"])
        pooled = F.normalize(pooled.float(), p=2, dim=1)
    out = pooled.cpu().numpy().astype(np.float32, copy=False)
    if out.shape != (len(texts), QWEN3_EMBED_DIM):
        raise RuntimeError(f"Unexpected Qwen3 embedding shape: {out.shape}")
    if not np.isfinite(out).all():
        raise RuntimeError("Qwen3 embeddings contain NaN/Inf.")
    return out


def encode_qwen3_adaptive(
    model,
    tokenizer,
    texts: Sequence[str],
    device: torch.device,
    is_query: bool,
) -> np.ndarray:
    try:
        return encode_qwen3_batch(model, tokenizer, texts, device, is_query)
    except torch.OutOfMemoryError:
        if device.type != "cuda" or len(texts) <= 1:
            raise
        torch.cuda.empty_cache()
        middle = len(texts) // 2
        print(f"Qwen3 OOM on batch={len(texts)}; retrying {middle}+{len(texts)-middle}.")
        left = encode_qwen3_adaptive(
            model, tokenizer, texts[:middle], device, is_query
        )
        right = encode_qwen3_adaptive(
            model, tokenizer, texts[middle:], device, is_query
        )
        return np.concatenate([left, right], axis=0)


def iter_code_rows(path: Path) -> Iterable[Tuple[int, str]]:
    logical_row = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            try:
                code = get_code_text(row)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to extract code text at {path}:{line_number}"
                ) from exc
            yield logical_row, code
            logical_row += 1


def encode_codebase_resumable(
    model,
    tokenizer,
    codebase_path: Path,
    codebase_rows: int,
    final_path: Path,
    device: torch.device,
    batch_size: int,
) -> Path:
    partial_path = final_path.with_name(final_path.name + ".partial.npy")
    progress_path = final_path.with_name(final_path.name + ".progress.json")

    if final_path.is_file():
        cached = np.load(final_path, mmap_mode="r", allow_pickle=False)
        if cached.shape != (codebase_rows, QWEN3_EMBED_DIM):
            raise RuntimeError(f"Existing Qwen3 code cache has wrong shape: {cached.shape}")
        if cached.dtype != np.float32:
            raise RuntimeError(f"Existing Qwen3 code cache has wrong dtype: {cached.dtype}")
        return final_path

    if partial_path.is_file():
        if not progress_path.is_file():
            raise RuntimeError(
                f"Partial Qwen3 code cache exists without progress file: {partial_path}"
            )
        progress = load_json(progress_path)
        completed = int(progress["completed_rows"])
        if int(progress["codebase_rows"]) != codebase_rows:
            raise RuntimeError("Qwen3 code-cache progress has wrong codebase row count.")
        mem = np.lib.format.open_memmap(partial_path, mode="r+")
        if mem.shape != (codebase_rows, QWEN3_EMBED_DIM) or mem.dtype != np.float32:
            raise RuntimeError("Partial Qwen3 code cache shape/dtype mismatch.")
        print(f"Resuming Qwen3 code encoding at row {completed}/{codebase_rows}.")
    else:
        completed = 0
        mem = np.lib.format.open_memmap(
            partial_path,
            mode="w+",
            dtype=np.float32,
            shape=(codebase_rows, QWEN3_EMBED_DIM),
        )
        replace_json(
            progress_path,
            {
                "format": "qwen3_code_embedding_progress_v1",
                "codebase": str(codebase_path.resolve()),
                "codebase_rows": int(codebase_rows),
                "completed_rows": 0,
                "embedding_dim": QWEN3_EMBED_DIM,
                "dtype": "float32",
                "max_length": QWEN3_MAX_LENGTH,
                "pooling": QWEN3_POOLING,
            },
        )

    batch_texts: List[str] = []
    batch_start = completed
    observed_rows = 0

    for row_id, code_text in iter_code_rows(codebase_path):
        observed_rows = row_id + 1
        if row_id < completed:
            continue
        if not batch_texts:
            batch_start = row_id
        batch_texts.append(code_text)
        if len(batch_texts) < int(batch_size):
            continue

        embeddings = encode_qwen3_adaptive(
            model, tokenizer, batch_texts, device=device, is_query=False
        )
        batch_end = batch_start + len(batch_texts)
        mem[batch_start:batch_end] = embeddings
        mem.flush()
        completed = batch_end
        replace_json(
            progress_path,
            {
                "format": "qwen3_code_embedding_progress_v1",
                "codebase": str(codebase_path.resolve()),
                "codebase_rows": int(codebase_rows),
                "completed_rows": int(completed),
                "embedding_dim": QWEN3_EMBED_DIM,
                "dtype": "float32",
                "max_length": QWEN3_MAX_LENGTH,
                "pooling": QWEN3_POOLING,
            },
        )
        if completed % max(int(batch_size) * 50, 1) == 0 or completed == codebase_rows:
            print(f"Qwen3 code encoding: {completed}/{codebase_rows}")
        batch_texts = []

    if batch_texts:
        embeddings = encode_qwen3_adaptive(
            model, tokenizer, batch_texts, device=device, is_query=False
        )
        batch_end = batch_start + len(batch_texts)
        mem[batch_start:batch_end] = embeddings
        mem.flush()
        completed = batch_end
        replace_json(
            progress_path,
            {
                "format": "qwen3_code_embedding_progress_v1",
                "codebase": str(codebase_path.resolve()),
                "codebase_rows": int(codebase_rows),
                "completed_rows": int(completed),
                "embedding_dim": QWEN3_EMBED_DIM,
                "dtype": "float32",
                "max_length": QWEN3_MAX_LENGTH,
                "pooling": QWEN3_POOLING,
            },
        )

    if observed_rows != codebase_rows:
        raise RuntimeError(
            f"Codebase row count changed while encoding: observed={observed_rows}, expected={codebase_rows}."
        )
    if completed != codebase_rows:
        raise RuntimeError(
            f"Qwen3 code encoding incomplete: {completed}/{codebase_rows}."
        )

    del mem
    partial_path.replace(final_path)
    progress_path.unlink(missing_ok=True)
    print(f"Qwen3 code cache complete: {final_path}")
    return final_path


def selected_queries(train_path: Path, source_indices: np.ndarray) -> List[str]:
    selected = read_selected_jsonl(train_path, source_indices)
    queries = []
    for index in source_indices.tolist():
        if int(index) not in selected:
            raise RuntimeError(f"Missing TRAIN source row {index} in {train_path}")
        queries.append(get_query_text(selected[int(index)]))
    return queries


def encode_query_cache(
    model,
    tokenizer,
    queries: Sequence[str],
    output_path: Path,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if output_path.is_file():
        values = np.asarray(np.load(output_path, allow_pickle=False), dtype=np.float32)
        if values.shape != (len(queries), QWEN3_EMBED_DIM):
            raise RuntimeError(f"Existing Qwen3 query cache has wrong shape: {values.shape}")
        return values

    blocks: List[np.ndarray] = []
    for start in range(0, len(queries), int(batch_size)):
        end = min(start + int(batch_size), len(queries))
        block = encode_qwen3_adaptive(
            model,
            tokenizer,
            queries[start:end],
            device=device,
            is_query=True,
        )
        blocks.append(block)
        if end % max(int(batch_size) * 20, 1) == 0 or end == len(queries):
            print(f"Qwen3 query encoding: {end}/{len(queries)}")
    values = np.concatenate(blocks, axis=0).astype(np.float32, copy=False)
    atomic_save_npy(output_path, values)
    return values


def general_bitmask(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels, dtype=bool)
    if values.ndim != 2 or values.shape[1] > 8:
        raise ValueError(f"Unsupported label matrix shape: {values.shape}")
    out = np.zeros(len(values), dtype=np.uint8)
    for operator_id in range(values.shape[1]):
        out |= values[:, operator_id].astype(np.uint8) << np.uint8(operator_id)
    return out


def build_probe_operators(
    base_indices: np.ndarray,
    base_scores: np.ndarray,
    alternate_sources: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_idx = np.asarray(base_indices, dtype=np.int32)
    base_val = np.asarray(base_scores, dtype=np.float32)
    n = len(base_idx)
    operator_count = 1 + len(alternate_sources)
    if base_idx.ndim != 2 or base_idx.shape[1] < BASELINE_K + OPERATOR_BUDGET:
        raise ValueError("Base retrieval is too narrow for DenseNext20.")

    out_idx = np.full((n, operator_count, OPERATOR_BUDGET), -1, dtype=np.int32)
    out_score = np.full(
        (n, operator_count, OPERATOR_BUDGET), np.nan, dtype=np.float32
    )
    out_rank = np.full((n, operator_count, OPERATOR_BUDGET), -1, dtype=np.int16)
    counts = np.zeros((n, operator_count), dtype=np.int16)

    out_idx[:, 0] = base_idx[:, BASELINE_K : BASELINE_K + OPERATOR_BUDGET]
    out_score[:, 0] = base_val[:, BASELINE_K : BASELINE_K + OPERATOR_BUDGET]
    out_rank[:, 0] = np.broadcast_to(
        np.arange(
            BASELINE_K + 1,
            BASELINE_K + OPERATOR_BUDGET + 1,
            dtype=np.int16,
        ),
        (n, OPERATOR_BUDGET),
    )
    counts[:, 0] = OPERATOR_BUDGET

    normalized_sources = [
        (np.asarray(idx, dtype=np.int32), np.asarray(score, dtype=np.float32))
        for idx, score in alternate_sources
    ]
    for idx, score in normalized_sources:
        if idx.shape != base_idx.shape or score.shape != base_val.shape:
            raise ValueError("Alternate retrieval tensors must match base retrieval shape.")

    for row_id in range(n):
        base_top100 = base_idx[row_id, :BASELINE_K]
        for source_id, (source_idx, source_score) in enumerate(
            normalized_sources, start=1
        ):
            ci, cs, cr, cc = first_n_novel(
                source_idx[row_id],
                source_score[row_id],
                base_top100,
                budget=OPERATOR_BUDGET,
            )
            out_idx[row_id, source_id] = ci
            out_score[row_id, source_id] = cs
            out_rank[row_id, source_id] = cr
            counts[row_id, source_id] = cc

    return out_idx, out_score, out_rank, counts


def labels_from_operators(
    operator_indices: np.ndarray,
    ground_truth: np.ndarray,
    states: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    gt = np.asarray(ground_truth, dtype=np.int64).reshape(-1)
    raw = np.any(np.asarray(operator_indices, dtype=np.int64) == gt[:, None, None], axis=2)
    masked = raw & (np.asarray(states, dtype=np.int64).reshape(-1) == 2)[:, None]
    return raw.astype(bool), masked.astype(bool)


def summary_from_labels(
    labels: np.ndarray,
    states: np.ndarray,
    roles: np.ndarray,
    operator_names: Sequence[str],
) -> Dict[str, Any]:
    repair = np.asarray(labels, dtype=bool)
    states = np.asarray(states, dtype=np.int8)
    roles = np.asarray(roles, dtype=np.int8)
    availability = states == 2
    denominator = int(np.sum(availability))
    operator_hits: Dict[str, int] = {}
    for operator_id, name in enumerate(operator_names):
        operator_hits[name] = int(np.sum(availability & repair[:, operator_id]))

    union = np.any(repair, axis=1)
    union_hits = int(np.sum(availability & union))
    strongest_name = max(operator_names, key=lambda name: operator_hits[name])
    strongest_hits = operator_hits[strongest_name]
    hit_count_per_row = repair.sum(axis=1)
    exclusive = {
        name: int(
            np.sum(availability & repair[:, operator_id] & (hit_count_per_row == 1))
        )
        for operator_id, name in enumerate(operator_names)
    }

    strong_cross_name = str(operator_names[1])
    without_strong_cross = np.any(
        np.delete(repair, 1, axis=1), axis=1
    )
    without_strong_cross_hits = int(np.sum(availability & without_strong_cross))
    strong_cross_exclusive = int(
        np.sum(availability & repair[:, 1] & ~without_strong_cross)
    )

    by_role: Dict[str, Any] = {}
    for role_id, role_name in enumerate(ROLE_NAMES):
        mask = availability & (roles == role_id)
        role_den = int(np.sum(mask))
        role_ops = {
            name: {
                "recovered": int(np.sum(mask & repair[:, operator_id])),
                "recovery_rate_percent": 100.0
                * int(np.sum(mask & repair[:, operator_id]))
                / max(role_den, 1),
            }
            for operator_id, name in enumerate(operator_names)
        }
        role_union = int(np.sum(mask & union))
        by_role[role_name] = {
            "availability_failures": role_den,
            "operators": role_ops,
            "oracle_union": {
                "recovered": role_union,
                "recovery_rate_percent": 100.0 * role_union / max(role_den, 1),
            },
        }

    bitmask = general_bitmask(repair[availability])
    return {
        "availability_failures": denominator,
        "operators": {
            name: {
                "recovered": operator_hits[name],
                "recovery_rate_percent": 100.0 * operator_hits[name] / max(denominator, 1),
                "exclusive_recovered": exclusive[name],
            }
            for name in operator_names
        },
        "best_single_operator": {
            "name": strongest_name,
            "recovered": strongest_hits,
            "recovery_rate_percent": 100.0 * strongest_hits / max(denominator, 1),
        },
        "oracle_union": {
            "recovered": union_hits,
            "recovery_rate_percent": 100.0 * union_hits / max(denominator, 1),
            "minus_best_single_pp": 100.0 * (union_hits - strongest_hits) / max(denominator, 1),
        },
        "strong_cross_operator": strong_cross_name,
        "union_without_strong_cross": {
            "recovered": without_strong_cross_hits,
            "recovery_rate_percent": 100.0 * without_strong_cross_hits / max(denominator, 1),
        },
        "strong_cross_exclusive": {
            "recovered": strong_cross_exclusive,
            "recovery_rate_percent": 100.0 * strong_cross_exclusive / max(denominator, 1),
        },
        "bitmask_counts": {
            str(mask): int(np.sum(bitmask == mask))
            for mask in range(1 << len(operator_names))
        },
        "by_role": by_role,
    }


def save_probe_contract(
    output_dir: Path,
    language: str,
    base_name: str,
    source_indices: np.ndarray,
    ground_truth: np.ndarray,
    roles: np.ndarray,
    base_indices: np.ndarray,
    base_scores: np.ndarray,
    alternate_sources: Sequence[Tuple[np.ndarray, np.ndarray]],
    probe_plan_sha: str,
    source_benchmark_retrieval_sha: str,
) -> Dict[str, Any]:
    operator_names = OPERATOR_NAMES_BY_BASE[base_name]
    path = output_dir / f"{language}_train_{base_name}_comparable_probe_v1.npz"

    if path.is_file():
        return summarize_contract(path)

    ranks = ground_truth_ranks(base_indices, ground_truth)
    states = state_labels_from_rank(ranks)
    op_idx, op_score, op_rank, op_count = build_probe_operators(
        base_indices, base_scores, alternate_sources
    )
    raw_labels, labels = labels_from_operators(op_idx, ground_truth, states)
    bitmask = general_bitmask(labels)

    atomic_save_npz(
        path,
        {
            "format": np.asarray(PROBE_CONTRACT_FORMAT),
            "experiment_family": np.asarray(EXPERIMENT_FAMILY),
            "method_family": np.asarray(METHOD_FAMILY),
            "language": np.asarray(language),
            "split": np.asarray("train"),
            "base_retriever": np.asarray(base_name),
            "operator_names": np.asarray(operator_names, dtype="U64"),
            "state_names": np.asarray(STATE_NAMES, dtype="U64"),
            "role_names": np.asarray(ROLE_NAMES, dtype="U64"),
            "source_row_indices": np.asarray(source_indices, dtype=np.int64),
            "ground_truth_code_indices": np.asarray(ground_truth, dtype=np.int32),
            "role_codes": np.asarray(roles, dtype=np.int8),
            "ground_truth_rank_top500_or_501": ranks.astype(np.int32),
            "state_labels": states.astype(np.int8),
            "base_top500_indices": np.asarray(base_indices, dtype=np.int32),
            "base_top500_scores": np.asarray(base_scores, dtype=np.float32),
            "operator_candidate_indices": op_idx.astype(np.int32),
            "operator_candidate_scores": op_score.astype(np.float32),
            "operator_source_ranks_one_based": op_rank.astype(np.int16),
            "operator_candidate_counts": op_count.astype(np.int16),
            "operator_raw_hit_labels": raw_labels.astype(bool),
            "repairability_labels": labels.astype(bool),
            "repairability_bitmask": bitmask.astype(np.uint8),
            "any_repairable": np.any(labels, axis=1).astype(bool),
            "baseline_k": np.asarray(BASELINE_K, dtype=np.int32),
            "retrieval_k": np.asarray(RETRIEVAL_K, dtype=np.int32),
            "operator_budget": np.asarray(OPERATOR_BUDGET, dtype=np.int32),
            "probe_plan_sha256": np.asarray(probe_plan_sha),
            "source_benchmark_retrieval_sha256": np.asarray(
                source_benchmark_retrieval_sha
            ),
            "valid_accessed": np.asarray(False),
            "test_accessed": np.asarray(False),
            "work1_used": np.asarray(False),
        },
    )
    return summarize_contract(path)


def summarize_contract(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as a:
        language = scalar_string(a, "language")
        base_name = scalar_string(a, "base_retriever")
        operator_names = tuple(str(x) for x in a["operator_names"].tolist())
        states = np.asarray(a["state_labels"], dtype=np.int8)
        roles = np.asarray(a["role_codes"], dtype=np.int8)
        labels = np.asarray(a["repairability_labels"], dtype=bool)
    return {
        "language": language,
        "base_retriever": base_name,
        "rows": int(len(states)),
        "role_counts": role_counts(roles),
        "state_counts": state_counts(states),
        "availability_failure_rate_percent": 100.0
        * int(np.sum(states == 2))
        / max(len(states), 1),
        "repairability": summary_from_labels(labels, states, roles, operator_names),
        "contract": str(path.resolve()),
        "contract_sha256": file_sha256(path),
    }


def prepare_plan(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    plan_path = output_dir / "comparable_retriever_probe_train_plan.json"
    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    source_plan_path = benchmark_dir / "repairability_benchmark_train_plan.json"
    source_report_path = benchmark_dir / "repairability_benchmark_train_report.json"
    source_audit_path = benchmark_dir / "repairability_benchmark_train_audit.json"

    source_plan = load_json(source_plan_path)
    source_report = load_json(source_report_path)
    source_audit = load_json(source_audit_path)
    if source_audit.get("audit_passed") is not True:
        raise RuntimeError("Source repairability benchmark did not pass audit.")
    if source_audit.get("valid_accessed") is not False or source_audit.get("test_accessed") is not False:
        raise RuntimeError("Source repairability benchmark claims VALID/TEST access.")
    if source_plan.get("valid_accessed") is not False or source_plan.get("test_accessed") is not False:
        raise RuntimeError("Source plan claims VALID/TEST access.")
    if source_report.get("valid_accessed") is not False or source_report.get("test_accessed") is not False:
        raise RuntimeError("Source report claims VALID/TEST access.")

    qwen3_model_path = Path(args.qwen3_model_path).expanduser().resolve()
    frozen = {
        "languages": list(LANGUAGES),
        "primary_bases": list(PRIMARY_BASES),
        "baseline_k": BASELINE_K,
        "retrieval_k": RETRIEVAL_K,
        "operator_budget": OPERATOR_BUDGET,
        "qwen3_model_path": str(qwen3_model_path),
        "qwen3_max_length": QWEN3_MAX_LENGTH,
        "qwen3_embedding_dim": QWEN3_EMBED_DIM,
        "qwen3_model_dtype": QWEN3_DTYPE,
        "qwen3_cache_dtype": "float32",
        "qwen3_padding_side": "left",
        "qwen3_pooling": QWEN3_POOLING,
        "qwen3_query_task": QWEN3_QUERY_TASK,
        "qwen3_code_instruction": "none; raw code only",
        "requested_encoder_batch_size": int(args.encoder_batch_size),
        "retrieval_query_chunk_size": int(args.query_chunk_size),
        "retrieval_code_chunk_size": int(args.code_chunk_size),
        "operator_names_by_base": {
            key: list(value) for key, value in OPERATOR_NAMES_BY_BASE.items()
        },
        "comparability_afr_ratio_band": [
            COMPARABILITY_AFR_RATIO_LOW,
            COMPARABILITY_AFR_RATIO_HIGH,
        ],
        "source_rows_roles_ground_truth": "reuse exact audited benchmark_train retrieval contracts",
        "valid_accessed": False,
        "test_accessed": False,
    }

    if plan_path.is_file():
        existing = load_json(plan_path)
        if existing.get("format") != PROBE_PLAN_FORMAT:
            raise RuntimeError("Existing comparable probe plan has wrong format.")
        if existing.get("frozen_parameters") != frozen:
            raise RuntimeError("Command/model parameters differ from frozen comparable probe plan.")
        print(f"REUSED_COMPARABLE_PROBE_PLAN_SHA256={file_sha256(plan_path)}")
        return existing

    if any(output_dir.iterdir()):
        raise RuntimeError(
            f"Output directory is non-empty but has no frozen probe plan: {output_dir}"
        )

    source_plan_map = {row["language"]: row for row in source_plan["per_language"]}
    source_report_map = {row["language"]: row for row in source_report["per_language"]}
    per_language = []
    for language in LANGUAGES:
        source_row = source_plan_map[language]
        report_row = source_report_map[language]
        retrieval_path = benchmark_dir / f"{language}_train_retrievals_v1.npz"
        nomic_contract = benchmark_dir / f"{language}_train_nomic_repairability_v1.npz"
        unix_contract = benchmark_dir / f"{language}_train_unixcoder_repairability_v1.npz"
        retrieval_info = checked_file(retrieval_path, f"{language} source retrieval contract")
        nomic_info = checked_file(nomic_contract, f"{language} source Nomic contract")
        unix_info = checked_file(unix_contract, f"{language} source UniXcoder contract")
        if retrieval_info["sha256"] != report_row["retrieval_contract_sha256"]:
            raise RuntimeError(f"{language}: source retrieval hash differs from audited report.")
        base_map = {row["base_retriever"]: row for row in report_row["bases"]}
        if nomic_info["sha256"] != base_map["nomic"]["contract_sha256"]:
            raise RuntimeError(f"{language}: source Nomic contract hash differs from report.")
        if unix_info["sha256"] != base_map["unixcoder"]["contract_sha256"]:
            raise RuntimeError(f"{language}: source UniXcoder contract hash differs from report.")

        train_path = verify_file_info(source_row["train_file"], f"{language} TRAIN")
        codebase_path = verify_file_info(source_row["codebase_file"], f"{language} codebase")
        per_language.append(
            {
                "language": language,
                "train_rows": int(source_row["train_rows"]),
                "codebase_rows": int(source_row["codebase_rows"]),
                "train_file": dict(source_row["train_file"]),
                "codebase_file": dict(source_row["codebase_file"]),
                "source_retrieval_contract": retrieval_info,
                "source_nomic_contract": nomic_info,
                "source_unixcoder_contract": unix_info,
            }
        )
        if train_path != Path(source_row["train_file"]["path"]).resolve():
            raise RuntimeError("Unexpected TRAIN path resolution change.")
        if codebase_path != Path(source_row["codebase_file"]["path"]).resolve():
            raise RuntimeError("Unexpected codebase path resolution change.")

    plan = {
        "format": PROBE_PLAN_FORMAT,
        "experiment_family": EXPERIMENT_FAMILY,
        "method_family": METHOD_FAMILY,
        "split": "train",
        "scientific_role": (
            "Comparable-strength retriever expansion. It tests whether failures of Nomic "
            "and Qwen3-Embedding-4B occupy bidirectionally repairable regions under the "
            "same audited TRAIN query sample and equal candidate budget."
        ),
        "frozen_parameters": frozen,
        "source_benchmark": {
            "directory": str(benchmark_dir),
            "plan": checked_file(source_plan_path, "source repairability plan"),
            "report": checked_file(source_report_path, "source repairability report"),
            "audit": checked_file(source_audit_path, "source repairability audit"),
            "audit_passed": True,
        },
        "qwen3_model": model_identity(qwen3_model_path),
        "operator_contract": {
            "nomic": {
                "dense_next20": "Nomic ranks 101..120",
                "qwen3_novel20": "first 20 Qwen3 candidates absent from Nomic Top100",
                "unixcoder_novel20": "first 20 UniXcoder candidates absent from Nomic Top100",
                "bm25_novel20": "first 20 BM25 candidates absent from Nomic Top100",
            },
            "qwen3": {
                "dense_next20": "Qwen3 ranks 101..120",
                "nomic_novel20": "first 20 Nomic candidates absent from Qwen3 Top100",
                "unixcoder_novel20": "first 20 UniXcoder candidates absent from Qwen3 Top100",
                "bm25_novel20": "first 20 BM25 candidates absent from Qwen3 Top100",
            },
            "all_operators_have_equal_candidate_budget": OPERATOR_BUDGET,
            "operators_never_reorder_or_delete_base_top100": True,
        },
        "implementation_files": {
            "build_comparable_retriever_probe.py": checked_file(
                Path(__file__).resolve(), "comparable retriever probe builder"
            ),
            "audit_comparable_retriever_probe.py": checked_file(
                Path(__file__).resolve().with_name("audit_comparable_retriever_probe.py"),
                "comparable retriever probe audit",
            ),
            "work2_repairability_core.py": checked_file(
                Path(__file__).resolve().with_name("work2_repairability_core.py"),
                "repairability core",
            ),
        },
        "per_language": per_language,
        "valid_accessed": False,
        "test_accessed": False,
        "work1_verifier_used": False,
        "work1_reranker_used": False,
        "rq4_rq5_utility_labels_used": False,
    }
    atomic_write_json(plan_path, plan)
    print(f"FROZEN_COMPARABLE_PROBE_PLAN_SHA256={file_sha256(plan_path)}")
    return plan


def load_source_retrieval(plan_row: Mapping[str, Any]):
    retrieval_path = verify_file_info(
        plan_row["source_retrieval_contract"],
        f"{plan_row['language']} source retrieval contract",
    )
    with np.load(retrieval_path, allow_pickle=False) as r:
        if scalar_string(r, "format") != "work2_repairability_retrievals_v1":
            raise RuntimeError(f"Wrong source retrieval format: {retrieval_path}")
        if scalar_string(r, "split") != "train":
            raise RuntimeError("Comparable probe source retrieval is not TRAIN.")
        if scalar_bool(r, "valid_accessed") or scalar_bool(r, "test_accessed"):
            raise RuntimeError("Source retrieval claims VALID/TEST access.")
        arrays = {
            "source_indices": np.asarray(r["source_row_indices"], dtype=np.int64),
            "ground_truth": np.asarray(r["ground_truth_code_indices"], dtype=np.int32),
            "roles": np.asarray(r["role_codes"], dtype=np.int8),
            "nomic_indices": np.asarray(r["nomic_indices"], dtype=np.int32),
            "nomic_scores": np.asarray(r["nomic_scores"], dtype=np.float32),
            "unixcoder_indices": np.asarray(r["unixcoder_indices"], dtype=np.int32),
            "unixcoder_scores": np.asarray(r["unixcoder_scores"], dtype=np.float32),
            "bm25_indices": np.asarray(r["bm25_indices"], dtype=np.int32),
            "bm25_scores": np.asarray(r["bm25_scores"], dtype=np.float32),
        }
    rows = len(arrays["source_indices"])
    if not all(len(value) == rows for value in arrays.values()):
        raise RuntimeError("Source retrieval arrays have inconsistent row counts.")
    codebase_rows = int(plan_row["codebase_rows"])
    for prefix in ("nomic", "unixcoder", "bm25"):
        validate_retrieval(
            arrays[f"{prefix}_indices"],
            arrays[f"{prefix}_scores"],
            codebase_rows,
            RETRIEVAL_K,
        )
    return retrieval_path, arrays


def build_language(
    args: argparse.Namespace,
    plan_row: Mapping[str, Any],
    output_dir: Path,
    probe_plan_sha: str,
    model,
    tokenizer,
    device: torch.device,
) -> Dict[str, Any]:
    language = str(plan_row["language"])
    print(f"\n===== COMPARABLE RETRIEVER PROBE: {language} / TRAIN =====")
    source_retrieval_path, source = load_source_retrieval(plan_row)
    source_sha = file_sha256(source_retrieval_path)
    source_indices = source["source_indices"]
    ground_truth = source["ground_truth"]
    roles = source["roles"]
    codebase_rows = int(plan_row["codebase_rows"])

    qwen3_retrieval_path = output_dir / f"{language}_train_qwen3_top500_v1.npz"
    nomic_contract_path = output_dir / f"{language}_train_nomic_comparable_probe_v1.npz"
    qwen3_contract_path = output_dir / f"{language}_train_qwen3_comparable_probe_v1.npz"

    code_cache_path = output_dir / f"{language}_qwen3_code_embeddings_f32.npy"
    query_cache_path = output_dir / f"{language}_train_sampled_qwen3_queries_f32.npy"

    if qwen3_retrieval_path.is_file():
        with np.load(qwen3_retrieval_path, allow_pickle=False) as q:
            if scalar_string(q, "format") != QWEN3_RETRIEVAL_FORMAT:
                raise RuntimeError(f"{language}: existing Qwen3 retrieval has wrong format.")
            if scalar_string(q, "probe_plan_sha256") != probe_plan_sha:
                raise RuntimeError(f"{language}: existing Qwen3 retrieval has wrong plan hash.")
            if scalar_string(q, "source_benchmark_retrieval_sha256") != source_sha:
                raise RuntimeError(f"{language}: existing Qwen3 retrieval source hash mismatch.")
            if scalar_bool(q, "valid_accessed") or scalar_bool(q, "test_accessed"):
                raise RuntimeError(f"{language}: Qwen3 retrieval claims VALID/TEST access.")
            if not np.array_equal(q["source_row_indices"], source_indices):
                raise RuntimeError(f"{language}: Qwen3 retrieval source rows changed.")
            if not np.array_equal(q["ground_truth_code_indices"], ground_truth):
                raise RuntimeError(f"{language}: Qwen3 retrieval GT changed.")
            if not np.array_equal(q["role_codes"], roles):
                raise RuntimeError(f"{language}: Qwen3 retrieval roles changed.")
            qwen3_indices = np.asarray(q["qwen3_indices"], dtype=np.int32)
            qwen3_scores = np.asarray(q["qwen3_scores"], dtype=np.float32)
        validate_retrieval(qwen3_indices, qwen3_scores, codebase_rows, RETRIEVAL_K)
        print(f"{language}: reusing completed Qwen3 Top500 retrieval.")
    else:
        train_path = verify_file_info(plan_row["train_file"], f"{language} TRAIN")
        codebase_path = verify_file_info(
            plan_row["codebase_file"], f"{language} codebase"
        )
        queries = selected_queries(train_path, source_indices)
        if len(queries) != len(source_indices):
            raise RuntimeError(f"{language}: query count differs from source rows.")

        encode_codebase_resumable(
            model=model,
            tokenizer=tokenizer,
            codebase_path=codebase_path,
            codebase_rows=codebase_rows,
            final_path=code_cache_path,
            device=device,
            batch_size=args.encoder_batch_size,
        )
        qwen3_code = np.load(code_cache_path, mmap_mode="r", allow_pickle=False)
        qwen3_query = encode_query_cache(
            model=model,
            tokenizer=tokenizer,
            queries=queries,
            output_path=query_cache_path,
            device=device,
            batch_size=args.encoder_batch_size,
        )
        qwen3_scores, qwen3_indices = retrieve_embedding_topk(
            query_embeddings=qwen3_query,
            code_embeddings=qwen3_code,
            device=device,
            top_k=RETRIEVAL_K,
            query_chunk_size=args.query_chunk_size,
            code_chunk_size=args.code_chunk_size,
            description=f"comparable {language} Qwen3 TRAIN top-{RETRIEVAL_K}",
        )
        validate_retrieval(qwen3_indices, qwen3_scores, codebase_rows, RETRIEVAL_K)
        atomic_save_npz(
            qwen3_retrieval_path,
            {
                "format": np.asarray(QWEN3_RETRIEVAL_FORMAT),
                "experiment_family": np.asarray(EXPERIMENT_FAMILY),
                "method_family": np.asarray(METHOD_FAMILY),
                "language": np.asarray(language),
                "split": np.asarray("train"),
                "source_row_indices": source_indices.astype(np.int64),
                "ground_truth_code_indices": ground_truth.astype(np.int32),
                "role_codes": roles.astype(np.int8),
                "qwen3_indices": qwen3_indices.astype(np.int32),
                "qwen3_scores": qwen3_scores.astype(np.float32),
                "probe_plan_sha256": np.asarray(probe_plan_sha),
                "source_benchmark_retrieval_sha256": np.asarray(source_sha),
                "valid_accessed": np.asarray(False),
                "test_accessed": np.asarray(False),
            },
        )
        del qwen3_code, qwen3_query, queries

    nomic_summary = save_probe_contract(
        output_dir=output_dir,
        language=language,
        base_name="nomic",
        source_indices=source_indices,
        ground_truth=ground_truth,
        roles=roles,
        base_indices=source["nomic_indices"],
        base_scores=source["nomic_scores"],
        alternate_sources=(
            (qwen3_indices, qwen3_scores),
            (source["unixcoder_indices"], source["unixcoder_scores"]),
            (source["bm25_indices"], source["bm25_scores"]),
        ),
        probe_plan_sha=probe_plan_sha,
        source_benchmark_retrieval_sha=source_sha,
    )
    qwen3_summary = save_probe_contract(
        output_dir=output_dir,
        language=language,
        base_name="qwen3",
        source_indices=source_indices,
        ground_truth=ground_truth,
        roles=roles,
        base_indices=qwen3_indices,
        base_scores=qwen3_scores,
        alternate_sources=(
            (source["nomic_indices"], source["nomic_scores"]),
            (source["unixcoder_indices"], source["unixcoder_scores"]),
            (source["bm25_indices"], source["bm25_scores"]),
        ),
        probe_plan_sha=probe_plan_sha,
        source_benchmark_retrieval_sha=source_sha,
    )

    row = {
        "language": language,
        "rows": int(len(source_indices)),
        "roles": role_counts(roles),
        "qwen3_code_cache": {
            "path": str(code_cache_path.resolve()),
            "exists": code_cache_path.is_file(),
            "sha256": file_sha256(code_cache_path) if code_cache_path.is_file() else None,
            "shape": [codebase_rows, QWEN3_EMBED_DIM],
            "dtype": "float32",
        },
        "qwen3_query_cache": {
            "path": str(query_cache_path.resolve()),
            "exists": query_cache_path.is_file(),
            "sha256": file_sha256(query_cache_path) if query_cache_path.is_file() else None,
            "shape": [int(len(source_indices)), QWEN3_EMBED_DIM],
            "dtype": "float32",
        },
        "qwen3_retrieval_contract": {
            "path": str(qwen3_retrieval_path.resolve()),
            "sha256": file_sha256(qwen3_retrieval_path),
        },
        "source_benchmark_retrieval_sha256": source_sha,
        "bases": [nomic_summary, qwen3_summary],
    }
    print(json.dumps(row, indent=2, ensure_ascii=False))

    del qwen3_indices, qwen3_scores, source
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def aggregate_base(per_language: Sequence[Mapping[str, Any]], base_name: str) -> Dict[str, Any]:
    operator_names = OPERATOR_NAMES_BY_BASE[base_name]
    rows = 0
    state_total = {name: 0 for name in STATE_NAMES}
    availability = 0
    operator_hits = {name: 0 for name in operator_names}
    operator_exclusive = {name: 0 for name in operator_names}
    union_hits = 0
    without_strong_cross_hits = 0
    strong_cross_exclusive = 0
    role_availability = {name: 0 for name in ROLE_NAMES}
    role_operator_hits = {
        role: {name: 0 for name in operator_names} for role in ROLE_NAMES
    }
    role_union = {name: 0 for name in ROLE_NAMES}

    for language_row in per_language:
        base = next(row for row in language_row["bases"] if row["base_retriever"] == base_name)
        rows += int(base["rows"])
        for state_name in STATE_NAMES:
            state_total[state_name] += int(base["state_counts"][state_name])
        repair = base["repairability"]
        availability += int(repair["availability_failures"])
        union_hits += int(repair["oracle_union"]["recovered"])
        without_strong_cross_hits += int(repair["union_without_strong_cross"]["recovered"])
        strong_cross_exclusive += int(repair["strong_cross_exclusive"]["recovered"])
        for name in operator_names:
            operator_hits[name] += int(repair["operators"][name]["recovered"])
            operator_exclusive[name] += int(repair["operators"][name]["exclusive_recovered"])
        for role in ROLE_NAMES:
            role_row = repair["by_role"][role]
            role_availability[role] += int(role_row["availability_failures"])
            role_union[role] += int(role_row["oracle_union"]["recovered"])
            for name in operator_names:
                role_operator_hits[role][name] += int(role_row["operators"][name]["recovered"])

    best_name = max(operator_names, key=lambda name: operator_hits[name])
    best_hits = operator_hits[best_name]
    return {
        "rows": rows,
        "state_counts": state_total,
        "availability_failures": availability,
        "availability_failure_rate_percent": 100.0 * availability / max(rows, 1),
        "operators": {
            name: {
                "recovered": operator_hits[name],
                "recovery_rate_percent": 100.0 * operator_hits[name] / max(availability, 1),
                "exclusive_recovered": operator_exclusive[name],
                "exclusive_rate_percent": 100.0 * operator_exclusive[name] / max(availability, 1),
            }
            for name in operator_names
        },
        "best_single_operator": {
            "name": best_name,
            "recovered": best_hits,
            "recovery_rate_percent": 100.0 * best_hits / max(availability, 1),
        },
        "oracle_union": {
            "recovered": union_hits,
            "recovery_rate_percent": 100.0 * union_hits / max(availability, 1),
            "minus_best_single_pp": 100.0 * (union_hits - best_hits) / max(availability, 1),
        },
        "strong_cross_operator": operator_names[1],
        "union_without_strong_cross": {
            "recovered": without_strong_cross_hits,
            "recovery_rate_percent": 100.0 * without_strong_cross_hits / max(availability, 1),
        },
        "strong_cross_exclusive": {
            "recovered": strong_cross_exclusive,
            "recovery_rate_percent": 100.0 * strong_cross_exclusive / max(availability, 1),
        },
        "by_role": {
            role: {
                "availability_failures": role_availability[role],
                "operators": {
                    name: {
                        "recovered": role_operator_hits[role][name],
                        "recovery_rate_percent": 100.0
                        * role_operator_hits[role][name]
                        / max(role_availability[role], 1),
                    }
                    for name in operator_names
                },
                "oracle_union": {
                    "recovered": role_union[role],
                    "recovery_rate_percent": 100.0
                    * role_union[role]
                    / max(role_availability[role], 1),
                },
            }
            for role in ROLE_NAMES
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "TRAIN-only comparable-strength retriever probe for the Work2 Repairability Frontier. "
            "Adds Qwen3-Embedding-4B to the already audited Nomic/UniXcoder/BM25 benchmark."
        )
    )
    parser.add_argument(
        "--benchmark-dir",
        default=(
            "/root/autodl-tmp/lora-code-embeddings-main/results/"
            "work2_repairability_frontier_v1/benchmark_train"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/root/autodl-tmp/lora-code-embeddings-main/results/"
            "work2_repairability_frontier_v1/comparable_retriever_probe_train"
        ),
    )
    parser.add_argument("--qwen3-model-path", default=QWEN3_MODEL_DEFAULT)
    parser.add_argument("--encoder-batch-size", type=int, default=8)
    parser.add_argument("--query-chunk-size", type=int, default=16)
    parser.add_argument("--code-chunk-size", type=int, default=8192)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    if args.encoder_batch_size <= 0 or args.query_chunk_size <= 0 or args.code_chunk_size <= 0:
        raise ValueError("Batch/chunk sizes must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = prepare_plan(args, output_dir)
    plan_path = output_dir / "comparable_retriever_probe_train_plan.json"
    plan_sha = file_sha256(plan_path)

    device = torch.device(args.device)
    model, tokenizer = load_qwen3(
        Path(args.qwen3_model_path).expanduser().resolve(), device
    )

    per_language = []
    for plan_row in plan["per_language"]:
        per_language.append(
            build_language(
                args=args,
                plan_row=plan_row,
                output_dir=output_dir,
                probe_plan_sha=plan_sha,
                model=model,
                tokenizer=tokenizer,
                device=device,
            )
        )

    del model, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    aggregate = {base: aggregate_base(per_language, base) for base in PRIMARY_BASES}
    nomic_afr = float(aggregate["nomic"]["availability_failure_rate_percent"])
    qwen3_afr = float(aggregate["qwen3"]["availability_failure_rate_percent"])
    ratio = qwen3_afr / nomic_afr if nomic_afr > 0 else float("inf")
    quality_comparison = {
        "nomic_availability_failure_rate_percent": nomic_afr,
        "qwen3_availability_failure_rate_percent": qwen3_afr,
        "qwen3_to_nomic_afr_ratio": ratio,
        "frozen_comparability_band": [
            COMPARABILITY_AFR_RATIO_LOW,
            COMPARABILITY_AFR_RATIO_HIGH,
        ],
        "inside_comparability_band": bool(
            COMPARABILITY_AFR_RATIO_LOW <= ratio <= COMPARABILITY_AFR_RATIO_HIGH
        ),
    }

    report = {
        "format": PROBE_REPORT_FORMAT,
        "experiment_family": EXPERIMENT_FAMILY,
        "method_family": METHOD_FAMILY,
        "split": "train",
        "source_plan": str(plan_path),
        "source_plan_sha256": plan_sha,
        "quality_comparison": quality_comparison,
        "aggregate": aggregate,
        "per_language": per_language,
        "valid_accessed": False,
        "test_accessed": False,
        "work1_verifier_used": False,
        "work1_reranker_used": False,
        "rq4_rq5_utility_labels_used": False,
    }
    report_path = output_dir / "comparable_retriever_probe_train_report.json"
    if report_path.is_file():
        existing = load_json(report_path)
        if existing != report:
            raise RuntimeError("Existing comparable probe report differs from recomputed report.")
    else:
        atomic_write_json(report_path, report)

    print("\n===== COMPARABLE RETRIEVER PROBE SUMMARY =====")
    print(json.dumps(quality_comparison, indent=2, ensure_ascii=False))
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    print(f"saved: {report_path}")
    print("WORK2_COMPARABLE_RETRIEVER_PROBE_BUILT_PENDING_AUDIT")


if __name__ == "__main__":
    main()
