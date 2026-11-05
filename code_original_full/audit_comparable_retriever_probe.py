from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from build_comparable_retriever_probe import (
    COMPARABILITY_AFR_RATIO_HIGH,
    COMPARABILITY_AFR_RATIO_LOW,
    OPERATOR_NAMES_BY_BASE,
    PRIMARY_BASES,
    PROBE_CONTRACT_FORMAT,
    PROBE_PLAN_FORMAT,
    PROBE_REPORT_FORMAT,
    QWEN3_EMBED_DIM,
    QWEN3_RETRIEVAL_FORMAT,
    aggregate_base,
    build_probe_operators,
    general_bitmask,
    labels_from_operators,
    summary_from_labels,
)
from work2_repairability_core import (
    BASELINE_K,
    EXPERIMENT_FAMILY,
    LANGUAGES,
    METHOD_FAMILY,
    OPERATOR_BUDGET,
    RETRIEVAL_K,
    ROLE_NAMES,
    STATE_NAMES,
    atomic_write_json,
    file_sha256,
    ground_truth_ranks,
    role_counts,
    state_counts,
    state_labels_from_rank,
    validate_retrieval,
)


PROBE_AUDIT_FORMAT = "work2_comparable_retriever_probe_audit_v1"


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


def require_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def require_array_equal(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    if not np.array_equal(np.asarray(actual), np.asarray(expected)):
        raise RuntimeError(f"Array mismatch: {label}")


def require_array_close(
    actual: np.ndarray,
    expected: np.ndarray,
    label: str,
    atol: float = 1e-6,
) -> None:
    if not np.allclose(
        np.asarray(actual),
        np.asarray(expected),
        atol=atol,
        rtol=0.0,
        equal_nan=True,
    ):
        raise RuntimeError(f"Array mismatch: {label}")


def verify_file_info(row: Mapping[str, Any], label: str) -> Path:
    path = Path(row["path"]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if "size" in row:
        require_equal(int(path.stat().st_size), int(row["size"]), f"{label} size")
    if "sha256" in row:
        require_equal(file_sha256(path), str(row["sha256"]), f"{label} sha256")
    return path


def load_source_retrieval(plan_row: Mapping[str, Any]):
    path = verify_file_info(
        plan_row["source_retrieval_contract"],
        f"{plan_row['language']} source benchmark retrieval",
    )
    with np.load(path, allow_pickle=False) as r:
        require_equal(
            scalar_string(r, "format"),
            "work2_repairability_retrievals_v1",
            f"{plan_row['language']} source retrieval format",
        )
        require_equal(scalar_string(r, "split"), "train", "source split")
        if scalar_bool(r, "valid_accessed") or scalar_bool(r, "test_accessed"):
            raise RuntimeError("Source benchmark retrieval claims VALID/TEST access.")
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
    return path, arrays


def audit_contract(
    path: Path,
    language: str,
    base_name: str,
    source_indices: np.ndarray,
    ground_truth: np.ndarray,
    roles: np.ndarray,
    base_indices: np.ndarray,
    base_scores: np.ndarray,
    alternate_sources: Sequence[Tuple[np.ndarray, np.ndarray]],
    probe_plan_sha: str,
    source_retrieval_sha: str,
) -> Dict[str, Any]:
    operator_names = OPERATOR_NAMES_BY_BASE[base_name]
    expected_ranks = ground_truth_ranks(base_indices, ground_truth)
    expected_states = state_labels_from_rank(expected_ranks)
    expected_idx, expected_score, expected_rank, expected_count = build_probe_operators(
        base_indices, base_scores, alternate_sources
    )
    expected_raw, expected_labels = labels_from_operators(
        expected_idx, ground_truth, expected_states
    )
    expected_bitmask = general_bitmask(expected_labels)

    with np.load(path, allow_pickle=False) as a:
        require_equal(scalar_string(a, "format"), PROBE_CONTRACT_FORMAT, f"{language}/{base_name} format")
        require_equal(scalar_string(a, "experiment_family"), EXPERIMENT_FAMILY, f"{language}/{base_name} experiment")
        require_equal(scalar_string(a, "method_family"), METHOD_FAMILY, f"{language}/{base_name} method")
        require_equal(scalar_string(a, "language"), language, f"{language}/{base_name} language")
        require_equal(scalar_string(a, "split"), "train", f"{language}/{base_name} split")
        require_equal(scalar_string(a, "base_retriever"), base_name, f"{language}/{base_name} base")
        require_equal(scalar_string(a, "probe_plan_sha256"), probe_plan_sha, f"{language}/{base_name} plan hash")
        require_equal(
            scalar_string(a, "source_benchmark_retrieval_sha256"),
            source_retrieval_sha,
            f"{language}/{base_name} source retrieval hash",
        )
        if scalar_bool(a, "valid_accessed") or scalar_bool(a, "test_accessed") or scalar_bool(a, "work1_used"):
            raise RuntimeError(f"{language}/{base_name}: access/leakage flag violation.")
        require_equal(int(np.asarray(a["baseline_k"]).item()), BASELINE_K, "baseline_k")
        require_equal(int(np.asarray(a["retrieval_k"]).item()), RETRIEVAL_K, "retrieval_k")
        require_equal(int(np.asarray(a["operator_budget"]).item()), OPERATOR_BUDGET, "operator_budget")
        require_array_equal(a["operator_names"], np.asarray(operator_names, dtype="U64"), f"{language}/{base_name} operator names")
        require_array_equal(a["state_names"], np.asarray(STATE_NAMES, dtype="U64"), f"{language}/{base_name} state names")
        require_array_equal(a["role_names"], np.asarray(ROLE_NAMES, dtype="U64"), f"{language}/{base_name} role names")
        require_array_equal(a["source_row_indices"], source_indices, f"{language}/{base_name} source rows")
        require_array_equal(a["ground_truth_code_indices"], ground_truth, f"{language}/{base_name} GT")
        require_array_equal(a["role_codes"], roles, f"{language}/{base_name} roles")
        require_array_equal(a["ground_truth_rank_top500_or_501"], expected_ranks, f"{language}/{base_name} ranks")
        require_array_equal(a["state_labels"], expected_states, f"{language}/{base_name} states")
        require_array_equal(a["base_top500_indices"], base_indices, f"{language}/{base_name} base indices")
        require_array_close(a["base_top500_scores"], base_scores, f"{language}/{base_name} base scores")
        require_array_equal(a["operator_candidate_indices"], expected_idx, f"{language}/{base_name} operator indices")
        require_array_close(a["operator_candidate_scores"], expected_score, f"{language}/{base_name} operator scores")
        require_array_equal(a["operator_source_ranks_one_based"], expected_rank, f"{language}/{base_name} source ranks")
        require_array_equal(a["operator_candidate_counts"], expected_count, f"{language}/{base_name} operator counts")
        require_array_equal(a["operator_raw_hit_labels"], expected_raw, f"{language}/{base_name} raw labels")
        require_array_equal(a["repairability_labels"], expected_labels, f"{language}/{base_name} labels")
        require_array_equal(a["repairability_bitmask"], expected_bitmask, f"{language}/{base_name} bitmask")
        require_array_equal(a["any_repairable"], np.any(expected_labels, axis=1), f"{language}/{base_name} any repairable")

    return {
        "language": language,
        "base_retriever": base_name,
        "rows": int(len(expected_states)),
        "role_counts": role_counts(roles),
        "state_counts": state_counts(expected_states),
        "availability_failure_rate_percent": 100.0
        * int(np.sum(expected_states == 2))
        / max(len(expected_states), 1),
        "repairability": summary_from_labels(
            expected_labels, expected_states, roles, operator_names
        ),
        "contract": str(path.resolve()),
        "contract_sha256": file_sha256(path),
        "labels_recomputed": True,
    }


def audit_language(
    plan_row: Mapping[str, Any],
    output_dir: Path,
    probe_plan_sha: str,
    report_row: Mapping[str, Any],
) -> Dict[str, Any]:
    language = str(plan_row["language"])
    print(f"\n===== AUDIT COMPARABLE PROBE: {language} =====")
    verify_file_info(plan_row["train_file"], f"{language} TRAIN")
    verify_file_info(plan_row["codebase_file"], f"{language} codebase")
    verify_file_info(plan_row["source_nomic_contract"], f"{language} source Nomic contract")
    verify_file_info(plan_row["source_unixcoder_contract"], f"{language} source UniXcoder contract")
    source_retrieval_path, source = load_source_retrieval(plan_row)
    source_sha = file_sha256(source_retrieval_path)
    source_indices = source["source_indices"]
    ground_truth = source["ground_truth"]
    roles = source["roles"]
    codebase_rows = int(plan_row["codebase_rows"])

    qwen3_retrieval_path = output_dir / f"{language}_train_qwen3_top500_v1.npz"
    if not qwen3_retrieval_path.is_file():
        raise FileNotFoundError(qwen3_retrieval_path)
    report_qret = report_row["qwen3_retrieval_contract"]
    require_equal(file_sha256(qwen3_retrieval_path), report_qret["sha256"], f"{language} Qwen3 retrieval report hash")

    with np.load(qwen3_retrieval_path, allow_pickle=False) as q:
        require_equal(scalar_string(q, "format"), QWEN3_RETRIEVAL_FORMAT, f"{language} Qwen3 retrieval format")
        require_equal(scalar_string(q, "experiment_family"), EXPERIMENT_FAMILY, f"{language} Qwen3 experiment")
        require_equal(scalar_string(q, "method_family"), METHOD_FAMILY, f"{language} Qwen3 method")
        require_equal(scalar_string(q, "language"), language, f"{language} Qwen3 language")
        require_equal(scalar_string(q, "split"), "train", f"{language} Qwen3 split")
        require_equal(scalar_string(q, "probe_plan_sha256"), probe_plan_sha, f"{language} Qwen3 plan hash")
        require_equal(scalar_string(q, "source_benchmark_retrieval_sha256"), source_sha, f"{language} Qwen3 source hash")
        if scalar_bool(q, "valid_accessed") or scalar_bool(q, "test_accessed"):
            raise RuntimeError(f"{language}: Qwen3 retrieval claims VALID/TEST access.")
        require_array_equal(q["source_row_indices"], source_indices, f"{language} Qwen3 source rows")
        require_array_equal(q["ground_truth_code_indices"], ground_truth, f"{language} Qwen3 GT")
        require_array_equal(q["role_codes"], roles, f"{language} Qwen3 roles")
        qwen3_indices = np.asarray(q["qwen3_indices"], dtype=np.int32)
        qwen3_scores = np.asarray(q["qwen3_scores"], dtype=np.float32)
    validate_retrieval(qwen3_indices, qwen3_scores, codebase_rows, RETRIEVAL_K)

    code_cache_path = Path(report_row["qwen3_code_cache"]["path"])
    query_cache_path = Path(report_row["qwen3_query_cache"]["path"])
    if not code_cache_path.is_file() or not query_cache_path.is_file():
        raise FileNotFoundError(f"{language}: Qwen3 embedding cache missing.")
    require_equal(file_sha256(code_cache_path), report_row["qwen3_code_cache"]["sha256"], f"{language} code cache sha256")
    require_equal(file_sha256(query_cache_path), report_row["qwen3_query_cache"]["sha256"], f"{language} query cache sha256")
    code_cache = np.load(code_cache_path, mmap_mode="r", allow_pickle=False)
    query_cache = np.load(query_cache_path, mmap_mode="r", allow_pickle=False)
    require_equal(tuple(code_cache.shape), (codebase_rows, QWEN3_EMBED_DIM), f"{language} code cache shape")
    require_equal(tuple(query_cache.shape), (len(source_indices), QWEN3_EMBED_DIM), f"{language} query cache shape")
    require_equal(str(code_cache.dtype), "float32", f"{language} code cache dtype")
    require_equal(str(query_cache.dtype), "float32", f"{language} query cache dtype")
    del code_cache, query_cache

    nomic_path = output_dir / f"{language}_train_nomic_comparable_probe_v1.npz"
    qwen3_path = output_dir / f"{language}_train_qwen3_comparable_probe_v1.npz"
    if not nomic_path.is_file() or not qwen3_path.is_file():
        raise FileNotFoundError(f"{language}: comparable probe base contract missing.")

    nomic_summary = audit_contract(
        path=nomic_path,
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
        source_retrieval_sha=source_sha,
    )
    qwen3_summary = audit_contract(
        path=qwen3_path,
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
        source_retrieval_sha=source_sha,
    )

    audited_bases = [nomic_summary, qwen3_summary]
    report_bases = {row["base_retriever"]: row for row in report_row["bases"]}
    for audited in audited_bases:
        reported = report_bases[audited["base_retriever"]]
        for key in (
            "rows",
            "role_counts",
            "state_counts",
            "availability_failure_rate_percent",
            "repairability",
            "contract",
            "contract_sha256",
        ):
            require_equal(audited[key], reported[key], f"{language}/{audited['base_retriever']} report {key}")

    return {
        "language": language,
        "rows": int(len(source_indices)),
        "roles": role_counts(roles),
        "qwen3_code_cache": dict(report_row["qwen3_code_cache"]),
        "qwen3_query_cache": dict(report_row["qwen3_query_cache"]),
        "qwen3_retrieval_contract": dict(report_row["qwen3_retrieval_contract"]),
        "source_benchmark_retrieval_sha256": source_sha,
        "bases": audited_bases,
        "retrieval_arrays_validated": True,
        "repairability_labels_recomputed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the TRAIN-only Nomic <-> Qwen3 comparable retriever probe. "
            "The audit recomputes failure states, equal-budget repair candidates, labels, "
            "bitmasks and summaries from stored rankings; it does not rerun Qwen3 encoding."
        )
    )
    parser.add_argument(
        "--probe-dir",
        default=(
            "/root/autodl-tmp/lora-code-embeddings-main/results/"
            "work2_repairability_frontier_v1/comparable_retriever_probe_train"
        ),
    )
    args = parser.parse_args()

    probe_dir = Path(args.probe_dir).expanduser().resolve()
    plan_path = probe_dir / "comparable_retriever_probe_train_plan.json"
    report_path = probe_dir / "comparable_retriever_probe_train_report.json"
    plan = load_json(plan_path)
    report = load_json(report_path)

    require_equal(plan.get("format"), PROBE_PLAN_FORMAT, "plan format")
    require_equal(report.get("format"), PROBE_REPORT_FORMAT, "report format")
    require_equal(plan.get("experiment_family"), EXPERIMENT_FAMILY, "plan experiment")
    require_equal(report.get("experiment_family"), EXPERIMENT_FAMILY, "report experiment")
    require_equal(plan.get("method_family"), METHOD_FAMILY, "plan method")
    require_equal(report.get("method_family"), METHOD_FAMILY, "report method")
    require_equal(plan.get("split"), "train", "plan split")
    require_equal(report.get("split"), "train", "report split")
    if plan.get("valid_accessed") is not False or plan.get("test_accessed") is not False:
        raise RuntimeError("Probe plan claims VALID/TEST access.")
    if report.get("valid_accessed") is not False or report.get("test_accessed") is not False:
        raise RuntimeError("Probe report claims VALID/TEST access.")
    if plan.get("work1_verifier_used") is not False or plan.get("work1_reranker_used") is not False:
        raise RuntimeError("Probe plan uses Work1 components.")
    if report.get("work1_verifier_used") is not False or report.get("work1_reranker_used") is not False:
        raise RuntimeError("Probe report uses Work1 components.")
    if plan.get("rq4_rq5_utility_labels_used") is not False or report.get("rq4_rq5_utility_labels_used") is not False:
        raise RuntimeError("Probe leaks old RQ4/RQ5 utility labels.")

    probe_plan_sha = file_sha256(plan_path)
    require_equal(report.get("source_plan_sha256"), probe_plan_sha, "report plan hash")

    for name, row in plan.get("implementation_files", {}).items():
        verify_file_info(row, f"implementation {name}")
    source_audit_info = plan["source_benchmark"]["audit"]
    source_audit_path = verify_file_info(source_audit_info, "source repairability audit")
    source_audit = load_json(source_audit_path)
    if source_audit.get("audit_passed") is not True:
        raise RuntimeError("Source repairability audit is not passed.")
    if source_audit.get("valid_accessed") is not False or source_audit.get("test_accessed") is not False:
        raise RuntimeError("Source repairability audit claims VALID/TEST access.")

    plan_languages = tuple(str(row["language"]) for row in plan["per_language"])
    require_equal(plan_languages, LANGUAGES, "plan language order")
    report_map = {row["language"]: row for row in report["per_language"]}
    require_equal(tuple(report_map.keys()), LANGUAGES, "report language order")

    per_language = [
        audit_language(row, probe_dir, probe_plan_sha, report_map[row["language"]])
        for row in plan["per_language"]
    ]
    aggregate = {base: aggregate_base(per_language, base) for base in PRIMARY_BASES}
    require_equal(aggregate, report["aggregate"], "aggregate report")

    nomic_afr = float(aggregate["nomic"]["availability_failure_rate_percent"])
    qwen3_afr = float(aggregate["qwen3"]["availability_failure_rate_percent"])
    ratio = qwen3_afr / nomic_afr if nomic_afr > 0 else float("inf")
    expected_quality = {
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
    require_equal(expected_quality, report["quality_comparison"], "quality comparison")

    audit = {
        "format": PROBE_AUDIT_FORMAT,
        "experiment_family": EXPERIMENT_FAMILY,
        "method_family": METHOD_FAMILY,
        "split": "train",
        "plan": str(plan_path),
        "plan_sha256": probe_plan_sha,
        "report": str(report_path),
        "report_sha256": file_sha256(report_path),
        "quality_comparison": expected_quality,
        "aggregate": aggregate,
        "per_language": per_language,
        "source_benchmark_audit_reverified": True,
        "qwen3_embedding_cache_hashes_verified": True,
        "qwen3_retrieval_arrays_validated": True,
        "failure_states_recomputed": True,
        "operator_candidates_recomputed_from_stored_rankings": True,
        "repairability_labels_recomputed": True,
        "qwen3_encoder_rerun": False,
        "valid_accessed": False,
        "test_accessed": False,
        "work1_used": False,
        "audit_passed": True,
    }
    audit_path = probe_dir / "comparable_retriever_probe_train_audit.json"
    atomic_write_json(audit_path, audit)

    print("\n===== COMPARABLE RETRIEVER PROBE AUDIT =====")
    print(json.dumps(expected_quality, indent=2, ensure_ascii=False))
    print(
        json.dumps(
            {
                "source_benchmark_audit_reverified": True,
                "qwen3_embedding_cache_hashes_verified": True,
                "qwen3_retrieval_arrays_validated": True,
                "failure_states_recomputed": True,
                "operator_candidates_recomputed_from_stored_rankings": True,
                "repairability_labels_recomputed": True,
                "qwen3_encoder_rerun": False,
                "valid_accessed": False,
                "test_accessed": False,
                "audit_passed": True,
            },
            indent=2,
        )
    )
    print(f"saved: {audit_path}")
    print("WORK2_COMPARABLE_RETRIEVER_PROBE_AUDIT_PASSED")


if __name__ == "__main__":
    main()
