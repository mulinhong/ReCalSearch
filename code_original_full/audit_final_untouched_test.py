from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from eval_nomic_embed import get_code_text
from work2_frontier_characterization_core import (
    BASES,
    LANGUAGES,
    concatenate_bundles,
    compute_min_budget_bundle,
    leave_one_language_out,
    stratified_bootstrap_frontier,
    summarize_bundle,
)
from work2_repairability_core import (
    atomic_write_json,
    file_sha256,
    validate_retrieval,
)


ROOT = Path("/root/autodl-tmp/lora-code-embeddings-main").resolve()
RESULT_ROOT = ROOT / "results" / "work2_repairability_frontier_v1"
PROTOCOL_DIR = RESULT_ROOT / "final_untouched_test_protocol_v1"
PROTOCOL_PATH = PROTOCOL_DIR / "final_untouched_test_protocol.json"
EXPECTED_PROTOCOL_SHA256 = (
    "6b68efed7814e5ef71e470647b65c6e24a7ca6dde98857bf16290ca8101b6e3c"
)
OUTPUT_DIR_DEFAULT = RESULT_ROOT / "final_untouched_test_v1"

EXECUTION_PLAN_FORMAT = "work2_final_untouched_test_execution_plan_v1"
STAGE_FORMAT = "work2_final_test_retrieval_stage_v1"
RETRIEVAL_FORMAT = "work2_final_test_retrievals_v1"
REPORT_FORMAT = "work2_final_untouched_test_report_v1"
AUDIT_FORMAT = "work2_final_untouched_test_audit_v1"

RETRIEVAL_K = 500
PRIMARY_BUDGET = 20
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260912


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


def require_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def verify_info(row: Mapping[str, Any], label: str) -> Path:
    path = Path(row["path"]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if int(path.stat().st_size) != int(row["size"]):
        raise RuntimeError(f"{label}: size changed: {path}")
    if file_sha256(path) != str(row["sha256"]):
        raise RuntimeError(f"{label}: SHA256 changed: {path}")
    return path


def recompute_ground_truth(test_path: Path, codebase_path: Path) -> np.ndarray:
    url_to_code: Dict[str, int] = {}
    code_rows = 0
    with codebase_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # Re-extract code to ensure the exact source row is parseable under the
            # frozen evaluator's code-text contract, even though retrieval is not rerun.
            get_code_text(row)
            url = row.get("url")
            if not url:
                raise RuntimeError(f"Missing codebase URL at {codebase_path}:{line_number}")
            url = str(url)
            if url in url_to_code:
                raise RuntimeError(f"Duplicate codebase URL: {url}")
            url_to_code[url] = code_rows
            code_rows += 1

    gt = []
    with test_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            url = row.get("url")
            if not url:
                raise RuntimeError(f"Missing TEST URL at {test_path}:{line_number}")
            url = str(url)
            if url not in url_to_code:
                raise RuntimeError(f"TEST URL absent from codebase: {url}")
            gt.append(url_to_code[url])
    return np.asarray(gt, dtype=np.int32)


def load_stage(
    path: Path,
    language: str,
    retriever: str,
    execution_plan_sha: str,
    codebase_rows: int,
) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as a:
        require_equal(scalar_string(a, "format"), STAGE_FORMAT, f"{language}/{retriever} stage format")
        require_equal(scalar_string(a, "language"), language, f"{language}/{retriever} language")
        require_equal(scalar_string(a, "retriever"), retriever, f"{language}/{retriever} retriever")
        require_equal(scalar_string(a, "split"), "test", f"{language}/{retriever} split")
        require_equal(
            scalar_string(a, "execution_plan_sha256"),
            execution_plan_sha,
            f"{language}/{retriever} execution plan",
        )
        if not scalar_bool(a, "test_accessed") or not scalar_bool(a, "test_outcomes_observed"):
            raise RuntimeError(f"{language}/{retriever}: missing TEST-outcome flags")
        if scalar_bool(a, "work1_used"):
            raise RuntimeError(f"{language}/{retriever}: Work1 flag")
        indices = np.asarray(a["indices"], dtype=np.int32)
        scores = np.asarray(a["scores"], dtype=np.float32)
    validate_retrieval(indices, scores, codebase_rows, RETRIEVAL_K)
    return {"indices": indices, "scores": scores}


def load_final_contract(
    path: Path,
    language: str,
    execution_plan_sha: str,
    protocol_sha: str,
    codebase_rows: int,
) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as a:
        require_equal(scalar_string(a, "format"), RETRIEVAL_FORMAT, f"{language} final format")
        require_equal(scalar_string(a, "language"), language, f"{language} final language")
        require_equal(scalar_string(a, "split"), "test", f"{language} final split")
        require_equal(
            scalar_string(a, "execution_plan_sha256"),
            execution_plan_sha,
            f"{language} final execution plan",
        )
        require_equal(
            scalar_string(a, "frozen_protocol_sha256"),
            protocol_sha,
            f"{language} final protocol",
        )
        if not scalar_bool(a, "test_accessed") or not scalar_bool(a, "test_outcomes_observed"):
            raise RuntimeError(f"{language}: final contract lacks TEST-outcome flags")
        if scalar_bool(a, "work1_used"):
            raise RuntimeError(f"{language}: Work1 flag")
        if scalar_bool(a, "post_test_tuning_allowed"):
            raise RuntimeError(f"{language}: post-TEST tuning flag violation")

        arrays = {
            "source_rows": np.asarray(a["source_row_indices"], dtype=np.int64),
            "ground_truth": np.asarray(a["ground_truth_code_indices"], dtype=np.int32),
        }
        for prefix in ("nomic", "qwen3", "unixcoder", "bm25"):
            arrays[f"{prefix}_indices"] = np.asarray(a[f"{prefix}_indices"], dtype=np.int32)
            arrays[f"{prefix}_scores"] = np.asarray(a[f"{prefix}_scores"], dtype=np.float32)

    for prefix in ("nomic", "qwen3", "unixcoder", "bm25"):
        validate_retrieval(
            arrays[f"{prefix}_indices"],
            arrays[f"{prefix}_scores"],
            codebase_rows,
            RETRIEVAL_K,
        )
    return arrays


def independently_recompute_report(
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
        codebase_rows = int(plan_row["codebase_rows"])
        final_path = output_dir / f"{language}_test_retrievals_v1.npz"
        arrays = load_final_contract(
            final_path,
            language,
            execution_plan_sha,
            protocol_sha,
            codebase_rows,
        )

        test_path = verify_info(plan_row["test_file"], f"{language} TEST")
        codebase_path = verify_info(plan_row["codebase_file"], f"{language} codebase")
        gt = recompute_ground_truth(test_path, codebase_path)
        require_equal(len(gt), int(plan_row["test_rows"]), f"{language} TEST rows")
        if not np.array_equal(arrays["source_rows"], np.arange(len(gt), dtype=np.int64)):
            raise RuntimeError(f"{language}: source-row identity changed")
        if not np.array_equal(arrays["ground_truth"], gt):
            raise RuntimeError(f"{language}: ground truth does not recompute from TEST/codebase URLs")

        # Stage contracts must be byte-level equivalent in retrieval content to the
        # final combined contract.
        for prefix in ("nomic", "qwen3", "unixcoder", "bm25"):
            stage_path = output_dir / f"{language}_test_{prefix}_top500_v1.npz"
            stage = load_stage(
                stage_path,
                language,
                prefix,
                execution_plan_sha,
                codebase_rows,
            )
            if not np.array_equal(stage["indices"], arrays[f"{prefix}_indices"]):
                raise RuntimeError(f"{language}/{prefix}: stage/final index mismatch")
            if not np.array_equal(stage["scores"], arrays[f"{prefix}_scores"]):
                raise RuntimeError(f"{language}/{prefix}: stage/final score mismatch")

        lang_row: Dict[str, Any] = {
            "language": language,
            "rows": int(len(gt)),
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
                ground_truth=gt,
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
        aggregate_bundle = concatenate_bundles(
            [by_base_language[base][language] for language in LANGUAGES]
        )
        aggregate[base] = summarize_bundle(aggregate_bundle)
        bootstrap[base] = stratified_bootstrap_frontier(
            by_base_language[base],
            reps=BOOTSTRAP_REPS,
            seed=BOOTSTRAP_SEED,
        )
        loo[base] = leave_one_language_out(
            by_base_language[base], budget=PRIMARY_BUDGET
        )

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

    overall = bool(
        all(row["pass"] for row in h1.values())
        and all(row["pass"] for row in h2.values())
        and all(row["pass"] for row in h3.values())
    )

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
            "Audit the frozen final CodeSearchNet TEST Repairability Frontier. "
            "This audit recomputes GT mapping, retrieval validity, budget/depth/union "
            "metrics, bootstrap intervals and confirmatory endpoints from stored TEST "
            "Top500 arrays, but does not rerun neural or BM25 retrieval."
        )
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR_DEFAULT))
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    plan_path = output_dir / "final_test_execution_plan.json"
    manifest_path = output_dir / "final_test_execution_manifest.json"
    report_path = output_dir / "final_untouched_test_report.json"

    protocol_sha = file_sha256(PROTOCOL_PATH)
    require_equal(protocol_sha, EXPECTED_PROTOCOL_SHA256, "frozen protocol SHA256")
    execution_plan = load_json(plan_path)
    manifest = load_json(manifest_path)
    report = load_json(report_path)

    require_equal(execution_plan.get("format"), EXECUTION_PLAN_FORMAT, "execution-plan format")
    require_equal(report.get("format"), REPORT_FORMAT, "report format")
    if execution_plan.get("test_outcomes_observed_before_execution") is not False:
        raise RuntimeError("Execution plan does not prove pre-outcome freeze.")
    if execution_plan.get("work1_used") is not False:
        raise RuntimeError("Execution plan claims Work1 use.")
    if execution_plan.get("post_test_tuning_allowed") is not False:
        raise RuntimeError("Execution plan allows post-TEST tuning.")

    execution_plan_sha = file_sha256(plan_path)
    require_equal(
        manifest.get("execution_plan_sha256"), execution_plan_sha, "manifest execution-plan SHA"
    )
    require_equal(
        manifest.get("frozen_protocol_sha256"), protocol_sha, "manifest protocol SHA"
    )
    if manifest.get("test_accessed") is not True or manifest.get("test_outcomes_observed") is not True:
        raise RuntimeError("Execution manifest lacks TEST-outcome flags.")
    if manifest.get("valid_accessed") is not False or manifest.get("work1_used") is not False:
        raise RuntimeError("Execution manifest leakage flag.")
    if manifest.get("post_test_tuning_allowed") is not False:
        raise RuntimeError("Execution manifest allows post-TEST tuning.")

    # Verify all source and implementation bytes recorded before retrieval outcomes.
    verify_info(execution_plan["frozen_protocol"], "frozen protocol")
    verify_info(execution_plan["interface_lock"], "interface lock")
    for name, row in execution_plan["source_audits"].items():
        verify_info(row, f"source audit {name}")
    for name, row in execution_plan["implementation_files"].items():
        verify_info(row, f"implementation {name}")
    for row in execution_plan["per_language"]:
        language = str(row["language"])
        for key in (
            "test_file",
            "codebase_file",
            "nomic_code_cache",
            "unixcoder_code_cache",
            "qwen3_code_cache",
        ):
            verify_info(row[key], f"{language}/{key}")

    # Verify execution-manifest artifacts and query-cache identity.
    manifest_map = {row["language"]: row for row in manifest["per_language"]}
    if set(manifest_map) != set(LANGUAGES):
        raise RuntimeError("Execution-manifest language coverage mismatch.")
    for language in LANGUAGES:
        row = manifest_map[language]
        verify_info(row["retrieval_contract"], f"{language} final retrieval")
        for name, info in row["stage_contracts"].items():
            verify_info(info, f"{language} {name} stage")
        expected_query_rows = int(row["test_rows"])
        for name, info in row["query_caches"].items():
            path = verify_info(info, f"{language} {name} query cache")
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            if arr.ndim != 2 or arr.shape[0] != expected_query_rows:
                raise RuntimeError(f"{language}/{name}: query-cache shape mismatch {arr.shape}")
            if str(arr.dtype) != "float32":
                raise RuntimeError(f"{language}/{name}: query-cache dtype changed {arr.dtype}")

    recomputed = independently_recompute_report(
        execution_plan=execution_plan,
        output_dir=output_dir,
        execution_plan_sha=execution_plan_sha,
        protocol_sha=protocol_sha,
    )
    require_equal(recomputed, report, "full final TEST report recomputation")

    audit = {
        "format": AUDIT_FORMAT,
        "experiment_family": "work2_repairability_frontier",
        "method_family": "failure_repairability_frontier",
        "frozen_protocol_sha256": protocol_sha,
        "execution_plan_sha256": execution_plan_sha,
        "execution_manifest_sha256": file_sha256(manifest_path),
        "report_sha256": file_sha256(report_path),
        "pre_outcome_protocol_freeze_verified": True,
        "source_artifact_hashes_verified": True,
        "implementation_hashes_verified": True,
        "ground_truth_recomputed_from_test_and_codebase": True,
        "retrieval_arrays_validated": True,
        "stage_final_retrieval_identity_verified": True,
        "budget_frontiers_recomputed": True,
        "failure_depth_frontiers_recomputed": True,
        "operator_overlap_and_exclusive_recovery_recomputed": True,
        "leave_one_language_out_recomputed": True,
        "bootstrap_intervals_recomputed": True,
        "confirmatory_endpoints_recomputed": True,
        "neural_retrieval_rerun": False,
        "bm25_retrieval_rerun": False,
        "valid_accessed": False,
        "test_accessed": True,
        "test_outcomes_observed": True,
        "work1_used": False,
        "post_test_tuning_allowed": False,
        "decision": report["decision"],
        "audit_passed": True,
    }

    audit_path = output_dir / "final_untouched_test_audit.json"
    if audit_path.is_file():
        existing = load_json(audit_path)
        require_equal(existing, audit, "existing final TEST audit")
    else:
        atomic_write_json(audit_path, audit)

    print("\n===== FINAL UNTOUCHED TEST AUDIT =====")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print("WORK2_FINAL_UNTOUCHED_TEST_AUDIT_PASSED")


if __name__ == "__main__":
    main()
