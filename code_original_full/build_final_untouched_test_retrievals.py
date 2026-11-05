from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from build_comparable_retriever_probe import (
    QWEN3_EMBED_DIM,
    encode_query_cache as encode_qwen3_query_cache,
    load_qwen3,
)
from build_repairability_benchmark import (
    build_bm25_matrix,
    encode_unixcoder_texts,
    load_unixcoder,
    retrieve_bm25_topk,
    retrieve_embedding_topk,
    validate_unixcoder_code_cache,
)
from eval_nomic_embed import (
    TextDataset,
    encode_texts as encode_nomic_texts,
    get_code_text,
    get_query_text,
    load_nomic_model,
)
from work2_frontier_characterization_core import (
    BASES,
    BUDGETS,
    LANGUAGES,
    concatenate_bundles,
    compute_min_budget_bundle,
    leave_one_language_out,
    stratified_bootstrap_frontier,
    summarize_bundle,
)
from work2_repairability_core import (
    atomic_save_npy,
    atomic_save_npz,
    atomic_write_json,
    checked_file,
    file_sha256,
    validate_retrieval,
)


ROOT = Path("/root/autodl-tmp/lora-code-embeddings-main").resolve()
DATA_ROOT = ROOT / "data" / "CSN"
RESULT_ROOT = ROOT / "results" / "work2_repairability_frontier_v1"

PROTOCOL_DIR = RESULT_ROOT / "final_untouched_test_protocol_v1"
PROTOCOL_PATH = PROTOCOL_DIR / "final_untouched_test_protocol.json"
INTERFACE_LOCK_PATH = PROTOCOL_DIR / "final_test_interface_lock.json"
EXPECTED_PROTOCOL_SHA256 = (
    "6b68efed7814e5ef71e470647b65c6e24a7ca6dde98857bf16290ca8101b6e3c"
)

BENCHMARK_DIR = RESULT_ROOT / "benchmark_train"
BENCHMARK_PLAN_PATH = BENCHMARK_DIR / "repairability_benchmark_train_plan.json"
BENCHMARK_AUDIT_PATH = BENCHMARK_DIR / "repairability_benchmark_train_audit.json"

COMPARABLE_DIR = RESULT_ROOT / "comparable_retriever_probe_train"
COMPARABLE_REPORT_PATH = COMPARABLE_DIR / "comparable_retriever_probe_train_report.json"
COMPARABLE_AUDIT_PATH = COMPARABLE_DIR / "comparable_retriever_probe_train_audit.json"

CHAR_DIR = RESULT_ROOT / "repairability_frontier_characterization_v1"
CHAR_AUDIT_PATH = CHAR_DIR / "repairability_frontier_characterization_train_audit.json"

OUTPUT_DIR_DEFAULT = RESULT_ROOT / "final_untouched_test_v1"

EXECUTION_PLAN_FORMAT = "work2_final_untouched_test_execution_plan_v1"
STAGE_FORMAT = "work2_final_test_retrieval_stage_v1"
RETRIEVAL_FORMAT = "work2_final_test_retrievals_v1"
REPORT_FORMAT = "work2_final_untouched_test_report_v1"

RETRIEVAL_K = 500
BASELINE_K = 100
PRIMARY_BUDGET = 20
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260912

NOMIC_BATCH_SIZE = 64
QWEN3_BATCH_SIZE = 8
UNIXCODER_BATCH_SIZE = 64
DENSE_QUERY_CHUNK = 128
DENSE_CODE_CHUNK = 8192
BM25_QUERY_CHUNK = 256
UNIXCODER_CACHE_VALIDATION_ROWS = 8
UNIXCODER_CACHE_MIN_COSINE = 0.995


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def scalar_string(archive, key: str) -> str:
    if key not in archive:
        raise KeyError(key)
    return str(np.asarray(archive[key]).item())


def scalar_bool(archive, key: str) -> bool:
    if key not in archive:
        raise KeyError(key)
    return bool(np.asarray(archive[key]).item())


def verify_file_info(row: Mapping[str, Any], label: str) -> Path:
    path = Path(row["path"]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if "size" in row and int(path.stat().st_size) != int(row["size"]):
        raise RuntimeError(f"{label}: size mismatch: {path}")
    if "sha256" in row and file_sha256(path) != str(row["sha256"]):
        raise RuntimeError(f"{label}: SHA256 mismatch: {path}")
    return path


def checked_path(path: Path, label: str) -> Dict[str, Any]:
    return checked_file(path.expanduser().resolve(), label)


def index_by_language(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    out: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        language = str(row["language"])
        if language in out:
            raise RuntimeError(f"Duplicate language row: {language}")
        out[language] = row
    if set(out) != set(LANGUAGES):
        raise RuntimeError(f"Language coverage mismatch: {sorted(out)}")
    return out


def read_test_queries_and_ground_truth(
    test_path: Path,
    codebase_path: Path,
) -> Tuple[List[str], List[str], np.ndarray]:
    """Read TEST queries and resolve the single annotated target by URL.

    This matches the historical CodeSearchNet evaluator contract, but it is used
    only after the final TEST protocol has been frozen.
    """
    code_texts: List[str] = []
    url_to_code: Dict[str, int] = {}

    with codebase_path.open("r", encoding="utf-8") as handle:
        for logical_row, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            code_texts.append(get_code_text(row))
            url = row.get("url")
            if not url:
                raise RuntimeError(
                    f"Missing codebase URL at logical row {len(code_texts)-1}: {codebase_path}"
                )
            url = str(url)
            if url in url_to_code:
                raise RuntimeError(f"Duplicate codebase URL in {codebase_path}: {url}")
            url_to_code[url] = len(code_texts) - 1

    queries: List[str] = []
    ground_truth: List[int] = []
    with test_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            queries.append(get_query_text(row))
            url = row.get("url")
            if not url:
                raise RuntimeError(f"Missing TEST URL at {test_path}:{line_number}")
            url = str(url)
            if url not in url_to_code:
                raise RuntimeError(
                    f"TEST target URL is absent from codebase at {test_path}:{line_number}: {url}"
                )
            ground_truth.append(int(url_to_code[url]))

    if not queries:
        raise RuntimeError(f"No TEST queries found: {test_path}")
    if len(queries) != len(ground_truth):
        raise RuntimeError("TEST query/ground-truth length mismatch.")
    if len(code_texts) != len(url_to_code):
        raise RuntimeError("Codebase URL mapping is not one-to-one.")

    return queries, code_texts, np.asarray(ground_truth, dtype=np.int32)


def load_or_encode_nomic_queries(
    model,
    tokenizer,
    queries: Sequence[str],
    output_path: Path,
    device: torch.device,
    query_prefix: str,
    max_length: int,
) -> np.ndarray:
    if output_path.is_file():
        values = np.asarray(np.load(output_path, allow_pickle=False), dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != len(queries):
            raise RuntimeError(f"Existing Nomic TEST query cache has wrong shape: {values.shape}")
        return values

    loader = DataLoader(
        TextDataset(list(queries)),
        batch_size=NOMIC_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    values = encode_nomic_texts(
        model,
        tokenizer,
        loader,
        device,
        max_length=int(max_length),
        prefix=str(query_prefix),
        desc="Nomic TEST query encoding",
    ).numpy().astype(np.float32, copy=False)
    if values.shape[0] != len(queries) or not np.isfinite(values).all():
        raise RuntimeError(f"Invalid Nomic TEST query embeddings: {values.shape}")
    atomic_save_npy(output_path, values)
    return values


def load_or_encode_unixcoder_queries(
    model,
    tokenizer,
    queries: Sequence[str],
    output_path: Path,
    device: torch.device,
    max_length: int,
) -> np.ndarray:
    if output_path.is_file():
        values = np.asarray(np.load(output_path, allow_pickle=False), dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != len(queries):
            raise RuntimeError(
                f"Existing UniXcoder TEST query cache has wrong shape: {values.shape}"
            )
        return values

    values = encode_unixcoder_texts(
        model,
        tokenizer,
        texts=queries,
        device=device,
        batch_size=UNIXCODER_BATCH_SIZE,
        max_length=int(max_length),
    ).astype(np.float32, copy=False)
    if values.shape[0] != len(queries) or not np.isfinite(values).all():
        raise RuntimeError(f"Invalid UniXcoder TEST query embeddings: {values.shape}")
    atomic_save_npy(output_path, values)
    return values


def load_stage(
    path: Path,
    language: str,
    retriever: str,
    rows: int,
    codebase_rows: int,
    execution_plan_sha: str,
) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as a:
        if scalar_string(a, "format") != STAGE_FORMAT:
            raise RuntimeError(f"Wrong stage format: {path}")
        if scalar_string(a, "language") != language:
            raise RuntimeError(f"Stage language mismatch: {path}")
        if scalar_string(a, "retriever") != retriever:
            raise RuntimeError(f"Stage retriever mismatch: {path}")
        if scalar_string(a, "split") != "test":
            raise RuntimeError(f"Non-TEST stage: {path}")
        if scalar_string(a, "execution_plan_sha256") != execution_plan_sha:
            raise RuntimeError(f"Stage execution-plan mismatch: {path}")
        if scalar_bool(a, "work1_used"):
            raise RuntimeError(f"Work1 flag in stage: {path}")
        indices = np.asarray(a["indices"], dtype=np.int32)
        scores = np.asarray(a["scores"], dtype=np.float32)
    if indices.shape[0] != rows:
        raise RuntimeError(f"Stage row mismatch: {path}: {indices.shape}")
    validate_retrieval(indices, scores, codebase_rows, RETRIEVAL_K)
    return scores, indices


def save_stage(
    path: Path,
    language: str,
    retriever: str,
    scores: np.ndarray,
    indices: np.ndarray,
    execution_plan_sha: str,
    protocol_sha: str,
    test_sha: str,
    codebase_sha: str,
) -> None:
    if path.is_file():
        raise FileExistsError(path)
    atomic_save_npz(
        path,
        {
            "format": np.asarray(STAGE_FORMAT),
            "language": np.asarray(language),
            "split": np.asarray("test"),
            "retriever": np.asarray(retriever),
            "scores": np.asarray(scores, dtype=np.float32),
            "indices": np.asarray(indices, dtype=np.int32),
            "execution_plan_sha256": np.asarray(execution_plan_sha),
            "frozen_protocol_sha256": np.asarray(protocol_sha),
            "test_file_sha256": np.asarray(test_sha),
            "codebase_file_sha256": np.asarray(codebase_sha),
            "test_accessed": np.asarray(True),
            "test_outcomes_observed": np.asarray(True),
            "work1_used": np.asarray(False),
        },
    )


# -----------------------------------------------------------------------------
# Frozen execution plan
# -----------------------------------------------------------------------------

def prepare_execution_plan(output_dir: Path) -> Dict[str, Any]:
    plan_path = output_dir / "final_test_execution_plan.json"

    protocol_sha = file_sha256(PROTOCOL_PATH)
    if protocol_sha != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            f"Frozen protocol changed: expected={EXPECTED_PROTOCOL_SHA256} actual={protocol_sha}"
        )
    protocol = load_json(PROTOCOL_PATH)
    interface = load_json(INTERFACE_LOCK_PATH)
    if interface.get("test_outcomes_observed") is not False:
        raise RuntimeError("Interface lock already claims TEST outcomes were observed.")
    if interface.get("work1_used") is not False:
        raise RuntimeError("Interface lock claims Work1 use.")

    benchmark_plan = load_json(BENCHMARK_PLAN_PATH)
    benchmark_audit = load_json(BENCHMARK_AUDIT_PATH)
    comparable_report = load_json(COMPARABLE_REPORT_PATH)
    comparable_audit = load_json(COMPARABLE_AUDIT_PATH)
    char_audit = load_json(CHAR_AUDIT_PATH)

    if benchmark_audit.get("audit_passed") is not True:
        raise RuntimeError("Source benchmark audit did not pass.")
    if comparable_audit.get("audit_passed") is not True:
        raise RuntimeError("Comparable-retriever audit did not pass.")
    if char_audit.get("audit_passed") is not True:
        raise RuntimeError("Frontier-characterization audit did not pass.")
    if char_audit.get("decision") != "READY_FOR_FROZEN_TEST_PROTOCOL":
        raise RuntimeError("TRAIN evidence was not ready for frozen TEST protocol.")

    benchmark_map = index_by_language(benchmark_plan["per_language"])
    comparable_map = index_by_language(comparable_report["per_language"])

    interface_map = interface["languages"]
    if set(interface_map) != set(LANGUAGES):
        raise RuntimeError("Interface-lock language coverage changed.")

    per_language = []
    total_test_rows = 0
    for language in LANGUAGES:
        locked = interface_map[language]
        test_files = locked["test_files"]
        codebase_files = locked["codebase_files"]
        if len(test_files) != 1 or len(codebase_files) != 1:
            raise RuntimeError(
                f"{language}: final TEST requires exactly one TEST and one codebase file."
            )
        test_path = Path(test_files[0]["path"]).resolve()
        codebase_path = Path(codebase_files[0]["path"]).resolve()
        test_info = checked_path(test_path, f"{language} TEST")
        codebase_info = checked_path(codebase_path, f"{language} codebase")
        test_rows = int(test_files[0]["rows"])
        codebase_rows = int(codebase_files[0]["rows"])
        total_test_rows += test_rows

        b = benchmark_map[language]
        if file_sha256(Path(b["codebase_file"]["path"])) != codebase_info["sha256"]:
            raise RuntimeError(f"{language}: codebase differs from audited TRAIN benchmark.")
        nomic_code = verify_file_info(b["nomic_code_cache"], f"{language} Nomic code cache")
        unix_code = verify_file_info(b["unixcoder_code_cache"], f"{language} UniXcoder code cache")

        qrow = comparable_map[language]["qwen3_code_cache"]
        qwen_code = Path(qrow["path"]).resolve()
        if not qwen_code.is_file():
            raise FileNotFoundError(qwen_code)
        if file_sha256(qwen_code) != str(qrow["sha256"]):
            raise RuntimeError(f"{language}: Qwen3 code-cache hash changed.")
        qshape = tuple(np.load(qwen_code, mmap_mode="r", allow_pickle=False).shape)
        if qshape != (codebase_rows, 2560):
            raise RuntimeError(f"{language}: Qwen3 code-cache shape mismatch: {qshape}")

        per_language.append(
            {
                "language": language,
                "test_rows": test_rows,
                "codebase_rows": codebase_rows,
                "test_file": test_info,
                "codebase_file": codebase_info,
                "nomic_code_cache": checked_path(nomic_code, f"{language} Nomic code cache"),
                "unixcoder_code_cache": checked_path(unix_code, f"{language} UniXcoder code cache"),
                "qwen3_code_cache": checked_path(qwen_code, f"{language} Qwen3 code cache"),
            }
        )

    frozen = {
        "retrieval_k": RETRIEVAL_K,
        "baseline_k": BASELINE_K,
        "budget_grid": list(BUDGETS),
        "primary_budget": PRIMARY_BUDGET,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "nomic_batch_size": NOMIC_BATCH_SIZE,
        "qwen3_batch_size": QWEN3_BATCH_SIZE,
        "unixcoder_batch_size": UNIXCODER_BATCH_SIZE,
        "dense_query_chunk": DENSE_QUERY_CHUNK,
        "dense_code_chunk": DENSE_CODE_CHUNK,
        "bm25_query_chunk": BM25_QUERY_CHUNK,
        "unixcoder_cache_validation_rows": UNIXCODER_CACHE_VALIDATION_ROWS,
        "unixcoder_cache_min_cosine": UNIXCODER_CACHE_MIN_COSINE,
        "nomic_model": protocol["base_retrievers"]["nomic"],
        "qwen3_model": protocol["base_retrievers"]["qwen3"],
        "unixcoder_model": protocol["controls"]["unixcoder"],
        "bm25": protocol["controls"]["bm25"],
        "primary_operator_family": protocol["primary_operator_family"],
        "primary_union": protocol["primary_union"],
        "full_union": protocol["full_union"],
    }

    implementation_paths = [
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("audit_final_untouched_test.py"),
        Path(__file__).resolve().with_name("work2_frontier_characterization_core.py"),
        Path(__file__).resolve().with_name("work2_repairability_core.py"),
        Path(__file__).resolve().with_name("build_repairability_benchmark.py"),
        Path(__file__).resolve().with_name("build_comparable_retriever_probe.py"),
        Path(__file__).resolve().with_name("eval_nomic_embed.py"),
    ]
    implementation = {
        p.name: checked_path(p, f"implementation {p.name}") for p in implementation_paths
    }

    plan = {
        "format": EXECUTION_PLAN_FORMAT,
        "experiment_family": "work2_repairability_frontier",
        "method_family": "failure_repairability_frontier",
        "scientific_role": (
            "One-shot execution of the already frozen final CodeSearchNet TEST protocol. "
            "No model/operator/budget/metric selection is allowed from TEST outcomes."
        ),
        "frozen_protocol": checked_path(PROTOCOL_PATH, "frozen final TEST protocol"),
        "interface_lock": checked_path(INTERFACE_LOCK_PATH, "final TEST interface lock"),
        "source_audits": {
            "benchmark": checked_path(BENCHMARK_AUDIT_PATH, "benchmark audit"),
            "comparable": checked_path(COMPARABLE_AUDIT_PATH, "comparable audit"),
            "characterization": checked_path(CHAR_AUDIT_PATH, "characterization audit"),
        },
        "frozen_parameters": frozen,
        "per_language": per_language,
        "total_test_rows": int(total_test_rows),
        "implementation_files": implementation,
        "test_interface_accessed": True,
        "test_outcomes_observed_before_execution": False,
        "work1_used": False,
        "post_test_tuning_allowed": False,
    }

    if plan_path.is_file():
        existing = load_json(plan_path)
        if existing != plan:
            raise RuntimeError(
                "Existing final TEST execution plan differs from current source bytes or frozen inputs."
            )
        print(f"REUSED_FROZEN_FINAL_TEST_EXECUTION_PLAN_SHA256={file_sha256(plan_path)}")
        return existing

    if any(output_dir.iterdir()):
        raise RuntimeError(
            f"Output directory must be empty before first execution-plan freeze: {output_dir}"
        )
    atomic_write_json(plan_path, plan)
    print(f"FROZEN_FINAL_TEST_EXECUTION_PLAN_SHA256={file_sha256(plan_path)}")
    return plan


# -----------------------------------------------------------------------------
# Retrieval execution
# -----------------------------------------------------------------------------

def build_language(
    plan_row: Mapping[str, Any],
    output_dir: Path,
    execution_plan_sha: str,
    protocol: Mapping[str, Any],
    protocol_sha: str,
    device: torch.device,
    nomic_model,
    nomic_tokenizer,
    qwen3_model,
    qwen3_tokenizer,
    unix_model,
    unix_tokenizer,
) -> Dict[str, Any]:
    language = str(plan_row["language"])
    print(f"\n{'='*90}\nFINAL UNTOUCHED TEST: {language}\n{'='*90}")

    test_path = verify_file_info(plan_row["test_file"], f"{language} TEST")
    codebase_path = verify_file_info(plan_row["codebase_file"], f"{language} codebase")
    test_rows = int(plan_row["test_rows"])
    codebase_rows = int(plan_row["codebase_rows"])

    queries, code_texts, ground_truth = read_test_queries_and_ground_truth(
        test_path, codebase_path
    )
    if len(queries) != test_rows:
        raise RuntimeError(f"{language}: TEST row count changed: {len(queries)} != {test_rows}")
    if len(code_texts) != codebase_rows:
        raise RuntimeError(
            f"{language}: codebase row count changed: {len(code_texts)} != {codebase_rows}"
        )

    source_rows = np.arange(test_rows, dtype=np.int64)
    test_sha = str(plan_row["test_file"]["sha256"])
    codebase_sha = str(plan_row["codebase_file"]["sha256"])

    stage_paths = {
        name: output_dir / f"{language}_test_{name}_top500_v1.npz"
        for name in ("nomic", "qwen3", "unixcoder", "bm25")
    }

    # -------------------------- Nomic --------------------------
    if stage_paths["nomic"].is_file():
        nomic_scores, nomic_indices = load_stage(
            stage_paths["nomic"], language, "nomic", test_rows, codebase_rows, execution_plan_sha
        )
        print(f"{language}: reusing completed Nomic TEST Top500.")
    else:
        nomic_query_path = output_dir / f"{language}_test_nomic_queries_f32.npy"
        nomic_query = load_or_encode_nomic_queries(
            nomic_model,
            nomic_tokenizer,
            queries,
            nomic_query_path,
            device,
            query_prefix=str(protocol["base_retrievers"]["nomic"]["query_prefix"]),
            max_length=int(protocol["base_retrievers"]["nomic"]["max_length"]),
        )
        nomic_code_path = verify_file_info(
            plan_row["nomic_code_cache"], f"{language} Nomic code cache"
        )
        nomic_code = np.load(nomic_code_path, mmap_mode="r", allow_pickle=False)
        if len(nomic_code) != codebase_rows:
            raise RuntimeError(f"{language}: Nomic code-cache row mismatch.")
        nomic_scores, nomic_indices = retrieve_embedding_topk(
            query_embeddings=nomic_query,
            code_embeddings=nomic_code,
            device=device,
            top_k=RETRIEVAL_K,
            query_chunk_size=DENSE_QUERY_CHUNK,
            code_chunk_size=DENSE_CODE_CHUNK,
            description=f"final {language} Nomic TEST top-{RETRIEVAL_K}",
        )
        validate_retrieval(nomic_indices, nomic_scores, codebase_rows, RETRIEVAL_K)
        save_stage(
            stage_paths["nomic"], language, "nomic", nomic_scores, nomic_indices,
            execution_plan_sha, protocol_sha, test_sha, codebase_sha,
        )
        del nomic_query, nomic_code
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # -------------------------- Qwen3 --------------------------
    if stage_paths["qwen3"].is_file():
        qwen3_scores, qwen3_indices = load_stage(
            stage_paths["qwen3"], language, "qwen3", test_rows, codebase_rows, execution_plan_sha
        )
        print(f"{language}: reusing completed Qwen3 TEST Top500.")
    else:
        qwen_query_path = output_dir / f"{language}_test_qwen3_queries_f32.npy"
        qwen_query = encode_qwen3_query_cache(
            model=qwen3_model,
            tokenizer=qwen3_tokenizer,
            queries=queries,
            output_path=qwen_query_path,
            device=device,
            batch_size=QWEN3_BATCH_SIZE,
        )
        qwen_code_path = verify_file_info(
            plan_row["qwen3_code_cache"], f"{language} Qwen3 code cache"
        )
        qwen_code = np.load(qwen_code_path, mmap_mode="r", allow_pickle=False)
        if tuple(qwen_code.shape) != (codebase_rows, QWEN3_EMBED_DIM):
            raise RuntimeError(f"{language}: Qwen3 code-cache shape changed: {qwen_code.shape}")
        qwen3_scores, qwen3_indices = retrieve_embedding_topk(
            query_embeddings=qwen_query,
            code_embeddings=qwen_code,
            device=device,
            top_k=RETRIEVAL_K,
            query_chunk_size=DENSE_QUERY_CHUNK,
            code_chunk_size=DENSE_CODE_CHUNK,
            description=f"final {language} Qwen3 TEST top-{RETRIEVAL_K}",
        )
        validate_retrieval(qwen3_indices, qwen3_scores, codebase_rows, RETRIEVAL_K)
        save_stage(
            stage_paths["qwen3"], language, "qwen3", qwen3_scores, qwen3_indices,
            execution_plan_sha, protocol_sha, test_sha, codebase_sha,
        )
        del qwen_query, qwen_code
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------ UniXcoder ------------------------
    unix_cache_validation: Mapping[str, Any]
    if stage_paths["unixcoder"].is_file():
        unix_scores, unix_indices = load_stage(
            stage_paths["unixcoder"], language, "unixcoder", test_rows, codebase_rows, execution_plan_sha
        )
        unix_cache_validation = {"status": "reused_completed_stage"}
        print(f"{language}: reusing completed UniXcoder TEST Top500.")
    else:
        unix_code_path = verify_file_info(
            plan_row["unixcoder_code_cache"], f"{language} UniXcoder code cache"
        )
        unix_code = np.load(unix_code_path, mmap_mode="r", allow_pickle=False)
        if len(unix_code) != codebase_rows:
            raise RuntimeError(f"{language}: UniXcoder code-cache row mismatch.")
        unix_cache_validation = validate_unixcoder_code_cache(
            unix_model,
            unix_tokenizer,
            code_texts=code_texts,
            cached_embeddings=unix_code,
            device=device,
            batch_size=UNIXCODER_BATCH_SIZE,
            validation_rows=UNIXCODER_CACHE_VALIDATION_ROWS,
            minimum_cosine=UNIXCODER_CACHE_MIN_COSINE,
        )
        print(f"{language}: UniXcoder code-cache validation {unix_cache_validation}")
        unix_query_path = output_dir / f"{language}_test_unixcoder_queries_f32.npy"
        unix_query = load_or_encode_unixcoder_queries(
            unix_model,
            unix_tokenizer,
            queries,
            unix_query_path,
            device,
            max_length=int(protocol["controls"]["unixcoder"]["max_length"]),
        )
        unix_scores, unix_indices = retrieve_embedding_topk(
            query_embeddings=unix_query,
            code_embeddings=unix_code,
            device=device,
            top_k=RETRIEVAL_K,
            query_chunk_size=DENSE_QUERY_CHUNK,
            code_chunk_size=DENSE_CODE_CHUNK,
            description=f"final {language} UniXcoder TEST top-{RETRIEVAL_K}",
        )
        validate_retrieval(unix_indices, unix_scores, codebase_rows, RETRIEVAL_K)
        save_stage(
            stage_paths["unixcoder"], language, "unixcoder", unix_scores, unix_indices,
            execution_plan_sha, protocol_sha, test_sha, codebase_sha,
        )
        del unix_query, unix_code
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --------------------------- BM25 --------------------------
    bm25_stats: Mapping[str, Any]
    if stage_paths["bm25"].is_file():
        bm25_scores, bm25_indices = load_stage(
            stage_paths["bm25"], language, "bm25", test_rows, codebase_rows, execution_plan_sha
        )
        bm25_stats = {"status": "reused_completed_stage"}
        print(f"{language}: reusing completed pure-BM25 TEST Top500.")
    else:
        bm25_cfg = protocol["controls"]["bm25"]
        vectorizer, bm25_matrix, bm25_stats = build_bm25_matrix(
            code_texts,
            max_features=int(bm25_cfg["max_features"]),
            min_df=int(bm25_cfg["min_df"]),
            k1=float(bm25_cfg["k1"]),
            b=float(bm25_cfg["b"]),
        )
        bm25_scores, bm25_indices = retrieve_bm25_topk(
            vectorizer,
            bm25_matrix,
            queries=queries,
            top_k=RETRIEVAL_K,
            query_chunk_size=BM25_QUERY_CHUNK,
            description=f"final {language} pure BM25 TEST top-{RETRIEVAL_K}",
        )
        validate_retrieval(bm25_indices, bm25_scores, codebase_rows, RETRIEVAL_K)
        save_stage(
            stage_paths["bm25"], language, "bm25", bm25_scores, bm25_indices,
            execution_plan_sha, protocol_sha, test_sha, codebase_sha,
        )
        del vectorizer, bm25_matrix
        gc.collect()

    # ----------------------- final contract --------------------
    final_path = output_dir / f"{language}_test_retrievals_v1.npz"
    if final_path.is_file():
        with np.load(final_path, allow_pickle=False) as a:
            if scalar_string(a, "format") != RETRIEVAL_FORMAT:
                raise RuntimeError(f"Wrong final retrieval format: {final_path}")
            if scalar_string(a, "execution_plan_sha256") != execution_plan_sha:
                raise RuntimeError(f"Final retrieval plan mismatch: {final_path}")
            if not np.array_equal(a["ground_truth_code_indices"], ground_truth):
                raise RuntimeError(f"{language}: final retrieval GT changed.")
        print(f"{language}: reusing completed final TEST retrieval contract.")
    else:
        atomic_save_npz(
            final_path,
            {
                "format": np.asarray(RETRIEVAL_FORMAT),
                "experiment_family": np.asarray("work2_repairability_frontier"),
                "method_family": np.asarray("failure_repairability_frontier"),
                "language": np.asarray(language),
                "split": np.asarray("test"),
                "source_row_indices": source_rows,
                "ground_truth_code_indices": ground_truth.astype(np.int32),
                "nomic_indices": nomic_indices.astype(np.int32),
                "nomic_scores": nomic_scores.astype(np.float32),
                "qwen3_indices": qwen3_indices.astype(np.int32),
                "qwen3_scores": qwen3_scores.astype(np.float32),
                "unixcoder_indices": unix_indices.astype(np.int32),
                "unixcoder_scores": unix_scores.astype(np.float32),
                "bm25_indices": bm25_indices.astype(np.int32),
                "bm25_scores": bm25_scores.astype(np.float32),
                "execution_plan_sha256": np.asarray(execution_plan_sha),
                "frozen_protocol_sha256": np.asarray(protocol_sha),
                "test_file_sha256": np.asarray(test_sha),
                "codebase_file_sha256": np.asarray(codebase_sha),
                "test_accessed": np.asarray(True),
                "test_outcomes_observed": np.asarray(True),
                "work1_used": np.asarray(False),
                "post_test_tuning_allowed": np.asarray(False),
            },
        )

    summary = {
        "language": language,
        "test_rows": test_rows,
        "codebase_rows": codebase_rows,
        "retrieval_contract": checked_path(final_path, f"{language} final TEST retrieval"),
        "stage_contracts": {
            name: checked_path(path, f"{language} {name} TEST stage")
            for name, path in stage_paths.items()
        },
        "query_caches": {
            "nomic": checked_path(
                output_dir / f"{language}_test_nomic_queries_f32.npy",
                f"{language} Nomic TEST query cache",
            ),
            "qwen3": checked_path(
                output_dir / f"{language}_test_qwen3_queries_f32.npy",
                f"{language} Qwen3 TEST query cache",
            ),
            "unixcoder": checked_path(
                output_dir / f"{language}_test_unixcoder_queries_f32.npy",
                f"{language} UniXcoder TEST query cache",
            ),
        },
        "unixcoder_code_cache_validation": dict(unix_cache_validation),
        "bm25_stats": dict(bm25_stats),
    }

    del queries, code_texts, ground_truth
    del nomic_scores, nomic_indices, qwen3_scores, qwen3_indices
    del unix_scores, unix_indices, bm25_scores, bm25_indices
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


# -----------------------------------------------------------------------------
# Final frozen TEST report
# -----------------------------------------------------------------------------

def load_final_retrieval(path: Path, language: str, execution_plan_sha: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as a:
        if scalar_string(a, "format") != RETRIEVAL_FORMAT:
            raise RuntimeError(f"Wrong final retrieval format: {path}")
        if scalar_string(a, "language") != language or scalar_string(a, "split") != "test":
            raise RuntimeError(f"Wrong language/split in final retrieval: {path}")
        if scalar_string(a, "execution_plan_sha256") != execution_plan_sha:
            raise RuntimeError(f"Execution-plan mismatch in final retrieval: {path}")
        if not scalar_bool(a, "test_accessed") or not scalar_bool(a, "test_outcomes_observed"):
            raise RuntimeError(f"Final retrieval lacks TEST outcome flags: {path}")
        if scalar_bool(a, "work1_used"):
            raise RuntimeError(f"Work1 flag in final retrieval: {path}")
        return {
            "ground_truth": np.asarray(a["ground_truth_code_indices"], dtype=np.int32),
            "nomic_indices": np.asarray(a["nomic_indices"], dtype=np.int32),
            "nomic_scores": np.asarray(a["nomic_scores"], dtype=np.float32),
            "qwen3_indices": np.asarray(a["qwen3_indices"], dtype=np.int32),
            "qwen3_scores": np.asarray(a["qwen3_scores"], dtype=np.float32),
            "unixcoder_indices": np.asarray(a["unixcoder_indices"], dtype=np.int32),
            "unixcoder_scores": np.asarray(a["unixcoder_scores"], dtype=np.float32),
            "bm25_indices": np.asarray(a["bm25_indices"], dtype=np.int32),
            "bm25_scores": np.asarray(a["bm25_scores"], dtype=np.float32),
        }


def compute_final_report(
    execution_plan: Mapping[str, Any],
    output_dir: Path,
    execution_plan_sha: str,
    protocol_sha: str,
) -> Dict[str, Any]:
    by_base_language: Dict[str, Dict[str, Mapping[str, Any]]] = {
        base: {} for base in BASES
    }
    per_language = []

    for plan_row in execution_plan["per_language"]:
        language = str(plan_row["language"])
        final_path = output_dir / f"{language}_test_retrievals_v1.npz"
        arrays = load_final_retrieval(final_path, language, execution_plan_sha)
        codebase_rows = int(plan_row["codebase_rows"])
        for prefix in ("nomic", "qwen3", "unixcoder", "bm25"):
            validate_retrieval(
                arrays[f"{prefix}_indices"],
                arrays[f"{prefix}_scores"],
                codebase_rows,
                RETRIEVAL_K,
            )

        lang_row: Dict[str, Any] = {
            "language": language,
            "rows": int(len(arrays["ground_truth"])),
            "bases": {},
        }
        for base in BASES:
            if base == "nomic":
                base_idx = arrays["nomic_indices"]
                peer_idx = arrays["qwen3_indices"]
            else:
                base_idx = arrays["qwen3_indices"]
                peer_idx = arrays["nomic_indices"]
            bundle = compute_min_budget_bundle(
                base_indices=base_idx,
                ground_truth=arrays["ground_truth"],
                peer_indices=peer_idx,
                unixcoder_indices=arrays["unixcoder_indices"],
                bm25_indices=arrays["bm25_indices"],
            )
            by_base_language[base][language] = bundle
            lang_row["bases"][base] = summarize_bundle(bundle)
        per_language.append(lang_row)

    aggregate: Dict[str, Any] = {}
    bootstrap: Dict[str, Any] = {}
    loo: Dict[str, Any] = {}
    for base in BASES:
        agg_bundle = concatenate_bundles(
            [by_base_language[base][language] for language in LANGUAGES]
        )
        aggregate[base] = summarize_bundle(agg_bundle)
        bootstrap[base] = stratified_bootstrap_frontier(
            by_base_language[base], reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED
        )
        loo[base] = leave_one_language_out(
            by_base_language[base], budget=PRIMARY_BUDGET
        )

    # Frozen confirmatory endpoints.
    h1: Dict[str, Any] = {}
    for base in BASES:
        point = float(
            aggregate[base]["budgets"][str(PRIMARY_BUDGET)]["primary_union"][
                "minus_best_single_pp"
            ]
        )
        ci = bootstrap[base][str(PRIMARY_BUDGET)]
        h1[base] = {
            "point_delta_pp": point,
            "ci95_low_pp": float(ci["ci95_low_pp"]),
            "ci95_high_pp": float(ci["ci95_high_pp"]),
            "pass": bool(point > 0.0 and float(ci["ci95_low_pp"]) > 0.0),
        }

    lang_map = {row["language"]: row for row in per_language}
    h2: Dict[str, Any] = {}
    for base in BASES:
        counts = {}
        qualifying = []
        for language in LANGUAGES:
            count = int(
                lang_map[language]["bases"][base]["budgets"][str(PRIMARY_BUDGET)][
                    "operators"
                ]["peer_strong"]["full_exclusive_recovered"]
            )
            counts[language] = count
            if count > 0:
                qualifying.append(language)
        h2[base] = {
            "per_language_counts": counts,
            "languages_with_peer_full_exclusive_recovery": qualifying,
            "count": len(qualifying),
            "pass": len(qualifying) >= 5,
        }

    h3: Dict[str, Any] = {}
    for base in BASES:
        deltas = {
            omitted: float(row["primary_union_minus_best_pp"])
            for omitted, row in loo[base].items()
        }
        h3[base] = {
            "leave_one_language_out_deltas_pp": deltas,
            "minimum_delta_pp": float(min(deltas.values())),
            "pass": all(value > 0.0 for value in deltas.values()),
        }

    h1_pass = all(row["pass"] for row in h1.values())
    h2_pass = all(row["pass"] for row in h2.values())
    h3_pass = all(row["pass"] for row in h3.values())
    overall = bool(h1_pass and h2_pass and h3_pass)

    return {
        "format": REPORT_FORMAT,
        "experiment_family": "work2_repairability_frontier",
        "method_family": "failure_repairability_frontier",
        "frozen_protocol_sha256": protocol_sha,
        "execution_plan_sha256": execution_plan_sha,
        "split": "test",
        "test_rows": int(execution_plan["total_test_rows"]),
        "per_language": per_language,
        "aggregate": aggregate,
        "bootstrap_primary_union_minus_best": bootstrap,
        "leave_one_language_out_B20": loo,
        "confirmatory_endpoints": {
            "H1_primary_union_advantage": h1,
            "H2_bidirectional_peer_complementarity": h2,
            "H3_language_robustness": h3,
        },
        "decision": "FINAL_TEST_CONFIRMED" if overall else "FINAL_TEST_NOT_CONFIRMED",
        "test_accessed": True,
        "test_outcomes_observed": True,
        "valid_accessed": False,
        "work1_used": False,
        "post_test_tuning_allowed": False,
        "predictor_claim": (
            "No deployable low-budget selective-repair claim is made; OCRE-v1 was NO-GO "
            "and D-OCRE was CLOSED before this final TEST evaluation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the frozen one-shot CodeSearchNet TEST Repairability Frontier protocol."
        )
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR_DEFAULT))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Both final builder and audit must already exist before the execution plan is frozen.
    audit_source = Path(__file__).resolve().with_name("audit_final_untouched_test.py")
    if not audit_source.is_file():
        raise FileNotFoundError(
            "Place audit_final_untouched_test.py beside this builder before first execution."
        )

    execution_plan = prepare_execution_plan(output_dir)
    execution_plan_path = output_dir / "final_test_execution_plan.json"
    execution_plan_sha = file_sha256(execution_plan_path)
    protocol_sha = file_sha256(PROTOCOL_PATH)
    protocol = load_json(PROTOCOL_PATH)

    device = torch.device(args.device)
    print(f"device={device}")

    # Load all three dense encoders once. On the RTX 4090 this stays comfortably
    # below available VRAM while avoiding repeated model reloads per language.
    nomic_model_path = str(protocol["base_retrievers"]["nomic"]["model"])
    qwen3_model_path = Path(protocol["base_retrievers"]["qwen3"]["model"]).resolve()
    unix_model_path = Path(protocol["controls"]["unixcoder"]["model"]).resolve()

    nomic_model, nomic_tokenizer = load_nomic_model(nomic_model_path, device)
    qwen3_model, qwen3_tokenizer = load_qwen3(qwen3_model_path, device)
    unix_model, unix_tokenizer = load_unixcoder(unix_model_path, device)

    per_language_execution = []
    for plan_row in execution_plan["per_language"]:
        per_language_execution.append(
            build_language(
                plan_row=plan_row,
                output_dir=output_dir,
                execution_plan_sha=execution_plan_sha,
                protocol=protocol,
                protocol_sha=protocol_sha,
                device=device,
                nomic_model=nomic_model,
                nomic_tokenizer=nomic_tokenizer,
                qwen3_model=qwen3_model,
                qwen3_tokenizer=qwen3_tokenizer,
                unix_model=unix_model,
                unix_tokenizer=unix_tokenizer,
            )
        )

    del nomic_model, nomic_tokenizer, qwen3_model, qwen3_tokenizer, unix_model, unix_tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    execution_manifest = {
        "format": "work2_final_test_execution_manifest_v1",
        "execution_plan_sha256": execution_plan_sha,
        "frozen_protocol_sha256": protocol_sha,
        "per_language": per_language_execution,
        "test_accessed": True,
        "test_outcomes_observed": True,
        "valid_accessed": False,
        "work1_used": False,
        "post_test_tuning_allowed": False,
    }
    manifest_path = output_dir / "final_test_execution_manifest.json"
    if manifest_path.is_file():
        existing = load_json(manifest_path)
        if existing != execution_manifest:
            raise RuntimeError("Existing execution manifest differs from deterministic rebuild.")
    else:
        atomic_write_json(manifest_path, execution_manifest)

    report = compute_final_report(
        execution_plan=execution_plan,
        output_dir=output_dir,
        execution_plan_sha=execution_plan_sha,
        protocol_sha=protocol_sha,
    )
    report_path = output_dir / "final_untouched_test_report.json"
    if report_path.is_file():
        existing = load_json(report_path)
        if existing != report:
            raise RuntimeError("Existing final TEST report differs from deterministic recomputation.")
    else:
        atomic_write_json(report_path, report)

    print("\n===== FINAL UNTOUCHED TEST SUMMARY =====")
    print("decision =", report["decision"])
    print("test_rows =", report["test_rows"])
    for base in BASES:
        print(f"\nBASE = {base}")
        agg = report["aggregate"][base]
        print(
            f"AF={agg['availability_failures']} "
            f"({agg['availability_failure_rate_percent']:.4f}%)"
        )
        for budget in BUDGETS:
            row = agg["budgets"][str(budget)]
            print(
                f"B={budget:>3} local={row['operators']['local_dense']['recovery_rate_percent']:.4f}% "
                f"peer={row['operators']['peer_strong']['recovery_rate_percent']:.4f}% "
                f"bm25={row['operators']['bm25']['recovery_rate_percent']:.4f}% "
                f"union={row['primary_union']['recovery_rate_percent']:.4f}% "
                f"union-best={row['primary_union']['minus_best_single_pp']:.4f}pp"
            )
    print("\nCONFIRMATORY ENDPOINTS")
    print(json.dumps(report["confirmatory_endpoints"], indent=2, ensure_ascii=False))
    print(f"saved: {report_path}")
    print("WORK2_FINAL_UNTOUCHED_TEST_COMPLETED_PENDING_AUDIT")


if __name__ == "__main__":
    main()
