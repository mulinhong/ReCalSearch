from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch

import build_final_untouched_test_retrievals as b


CORRECTION_FORMAT = "work2_final_test_runtime_correction_v1"
CORRECTION_REASON = (
    "The frozen final TEST builder loaded Nomic, Qwen3-Embedding-4B, and "
    "UniXcoder simultaneously on one RTX 4090. The first Nomic TEST query "
    "forward pass OOMed before any retrieval stage or TEST metric was produced. "
    "This correction changes only model residency/scheduling: dense encoders are "
    "loaded and released sequentially. Frozen models, prompts, pooling, cache "
    "sources, batch sizes, Top500 retrieval, operators, budgets, metrics, "
    "bootstrap, and confirmatory endpoints are unchanged."
)


def _cleanup_cuda(*objects) -> None:
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _read_language_inputs(plan_row: Mapping[str, Any]):
    language = str(plan_row["language"])
    test_path = b.verify_file_info(plan_row["test_file"], f"{language} TEST")
    codebase_path = b.verify_file_info(plan_row["codebase_file"], f"{language} codebase")
    queries, code_texts, ground_truth = b.read_test_queries_and_ground_truth(
        test_path, codebase_path
    )
    test_rows = int(plan_row["test_rows"])
    codebase_rows = int(plan_row["codebase_rows"])
    if len(queries) != test_rows:
        raise RuntimeError(
            f"{language}: TEST row count changed: {len(queries)} != {test_rows}"
        )
    if len(code_texts) != codebase_rows:
        raise RuntimeError(
            f"{language}: codebase row count changed: {len(code_texts)} != {codebase_rows}"
        )
    return queries, code_texts, ground_truth


def _stage_path(output_dir: Path, language: str, retriever: str) -> Path:
    return output_dir / f"{language}_test_{retriever}_top500_v1.npz"


def _write_correction_record(
    output_dir: Path,
    execution_plan_path: Path,
    launcher_path: Path,
) -> None:
    path = output_dir / "final_test_runtime_correction_oomsafe.json"
    payload = {
        "format": CORRECTION_FORMAT,
        "scientific_protocol_changed": False,
        "test_outcomes_observed_before_correction": False,
        "reason": CORRECTION_REASON,
        "execution_plan": {
            "path": str(execution_plan_path.resolve()),
            "sha256": b.file_sha256(execution_plan_path),
        },
        "frozen_builder": {
            "path": str(Path(b.__file__).resolve()),
            "sha256": b.file_sha256(Path(b.__file__).resolve()),
        },
        "oom_safe_launcher": {
            "path": str(launcher_path.resolve()),
            "sha256": b.file_sha256(launcher_path),
        },
        "changed_runtime_behavior": [
            "load Nomic only; finish/reuse all Nomic TEST stages; release it",
            "load Qwen3 only; finish/reuse all Qwen3 TEST stages; release it",
            "load UniXcoder only; finish/reuse all UniXcoder TEST stages; release it",
            "run/reuse pure BM25 stages without dense encoders resident",
            "assemble the same final contracts and frozen report",
        ],
        "unchanged": [
            "model identities and model weights",
            "query serialization / prefixes / instructions",
            "pooling and normalization",
            "max sequence lengths",
            "frozen batch sizes",
            "split-independent code caches",
            "retrieval TopK=500",
            "candidate budgets",
            "repair operators",
            "bootstrap settings",
            "confirmatory endpoints",
            "no Work1 use",
            "no post-TEST tuning",
        ],
    }

    if path.is_file():
        existing = b.load_json(path)
        if existing != payload:
            raise RuntimeError(
                "Existing OOM-safe correction record differs from current launcher or plan."
            )
        print(
            "REUSED_FINAL_TEST_RUNTIME_CORRECTION_SHA256="
            + b.file_sha256(path)
        )
    else:
        b.atomic_write_json(path, payload)
        print(
            "FINAL_TEST_RUNTIME_CORRECTION_SHA256="
            + b.file_sha256(path)
        )


def build_nomic_pass(
    execution_plan: Mapping[str, Any],
    output_dir: Path,
    execution_plan_sha: str,
    protocol: Mapping[str, Any],
    protocol_sha: str,
    device: torch.device,
) -> None:
    print("\n" + "=" * 92)
    print("OOM-SAFE PASS 1/4: NOMIC")
    print("=" * 92)

    model_path = str(protocol["base_retrievers"]["nomic"]["model"])
    model, tokenizer = b.load_nomic_model(model_path, device)

    try:
        for plan_row in execution_plan["per_language"]:
            language = str(plan_row["language"])
            stage = _stage_path(output_dir, language, "nomic")

            if stage.is_file():
                b.load_stage(
                    stage,
                    language,
                    "nomic",
                    int(plan_row["test_rows"]),
                    int(plan_row["codebase_rows"]),
                    execution_plan_sha,
                )
                print(f"{language}: reusing completed Nomic TEST Top500.")
                continue

            print(f"\n--- Nomic TEST: {language} ---")
            queries, code_texts, ground_truth = _read_language_inputs(plan_row)
            del code_texts, ground_truth

            query_path = output_dir / f"{language}_test_nomic_queries_f32.npy"
            query = b.load_or_encode_nomic_queries(
                model,
                tokenizer,
                queries,
                query_path,
                device,
                query_prefix=str(
                    protocol["base_retrievers"]["nomic"]["query_prefix"]
                ),
                max_length=int(
                    protocol["base_retrievers"]["nomic"]["max_length"]
                ),
            )

            code_path = b.verify_file_info(
                plan_row["nomic_code_cache"],
                f"{language} Nomic code cache",
            )
            code = np.load(code_path, mmap_mode="r", allow_pickle=False)
            if len(code) != int(plan_row["codebase_rows"]):
                raise RuntimeError(f"{language}: Nomic code-cache row mismatch.")

            scores, indices = b.retrieve_embedding_topk(
                query_embeddings=query,
                code_embeddings=code,
                device=device,
                top_k=b.RETRIEVAL_K,
                query_chunk_size=b.DENSE_QUERY_CHUNK,
                code_chunk_size=b.DENSE_CODE_CHUNK,
                description=f"final {language} Nomic TEST top-{b.RETRIEVAL_K}",
            )
            b.validate_retrieval(
                indices,
                scores,
                int(plan_row["codebase_rows"]),
                b.RETRIEVAL_K,
            )

            b.save_stage(
                stage,
                language,
                "nomic",
                scores,
                indices,
                execution_plan_sha,
                protocol_sha,
                str(plan_row["test_file"]["sha256"]),
                str(plan_row["codebase_file"]["sha256"]),
            )

            del queries, query, code, scores, indices
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        del model, tokenizer
        _cleanup_cuda()


def build_qwen3_pass(
    execution_plan: Mapping[str, Any],
    output_dir: Path,
    execution_plan_sha: str,
    protocol: Mapping[str, Any],
    protocol_sha: str,
    device: torch.device,
) -> None:
    print("\n" + "=" * 92)
    print("OOM-SAFE PASS 2/4: QWEN3")
    print("=" * 92)

    model_path = Path(protocol["base_retrievers"]["qwen3"]["model"]).resolve()
    model, tokenizer = b.load_qwen3(model_path, device)

    try:
        for plan_row in execution_plan["per_language"]:
            language = str(plan_row["language"])
            stage = _stage_path(output_dir, language, "qwen3")

            if stage.is_file():
                b.load_stage(
                    stage,
                    language,
                    "qwen3",
                    int(plan_row["test_rows"]),
                    int(plan_row["codebase_rows"]),
                    execution_plan_sha,
                )
                print(f"{language}: reusing completed Qwen3 TEST Top500.")
                continue

            print(f"\n--- Qwen3 TEST: {language} ---")
            queries, code_texts, ground_truth = _read_language_inputs(plan_row)
            del code_texts, ground_truth

            query_path = output_dir / f"{language}_test_qwen3_queries_f32.npy"
            query = b.encode_qwen3_query_cache(
                model=model,
                tokenizer=tokenizer,
                queries=queries,
                output_path=query_path,
                device=device,
                batch_size=b.QWEN3_BATCH_SIZE,
            )

            code_path = b.verify_file_info(
                plan_row["qwen3_code_cache"],
                f"{language} Qwen3 code cache",
            )
            code = np.load(code_path, mmap_mode="r", allow_pickle=False)
            expected = (int(plan_row["codebase_rows"]), b.QWEN3_EMBED_DIM)
            if tuple(code.shape) != expected:
                raise RuntimeError(
                    f"{language}: Qwen3 code-cache shape changed: "
                    f"{tuple(code.shape)} != {expected}"
                )

            scores, indices = b.retrieve_embedding_topk(
                query_embeddings=query,
                code_embeddings=code,
                device=device,
                top_k=b.RETRIEVAL_K,
                query_chunk_size=b.DENSE_QUERY_CHUNK,
                code_chunk_size=b.DENSE_CODE_CHUNK,
                description=f"final {language} Qwen3 TEST top-{b.RETRIEVAL_K}",
            )
            b.validate_retrieval(
                indices,
                scores,
                int(plan_row["codebase_rows"]),
                b.RETRIEVAL_K,
            )

            b.save_stage(
                stage,
                language,
                "qwen3",
                scores,
                indices,
                execution_plan_sha,
                protocol_sha,
                str(plan_row["test_file"]["sha256"]),
                str(plan_row["codebase_file"]["sha256"]),
            )

            del queries, query, code, scores, indices
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        del model, tokenizer
        _cleanup_cuda()


def build_unixcoder_pass(
    execution_plan: Mapping[str, Any],
    output_dir: Path,
    execution_plan_sha: str,
    protocol: Mapping[str, Any],
    protocol_sha: str,
    device: torch.device,
) -> Dict[str, Dict[str, Any]]:
    print("\n" + "=" * 92)
    print("OOM-SAFE PASS 3/4: UNIXCODER")
    print("=" * 92)

    model_path = Path(protocol["controls"]["unixcoder"]["model"]).resolve()
    model, tokenizer = b.load_unixcoder(model_path, device)
    validation_by_language: Dict[str, Dict[str, Any]] = {}

    try:
        for plan_row in execution_plan["per_language"]:
            language = str(plan_row["language"])
            stage = _stage_path(output_dir, language, "unixcoder")

            if stage.is_file():
                b.load_stage(
                    stage,
                    language,
                    "unixcoder",
                    int(plan_row["test_rows"]),
                    int(plan_row["codebase_rows"]),
                    execution_plan_sha,
                )
                validation_by_language[language] = {
                    "status": "reused_completed_stage"
                }
                print(f"{language}: reusing completed UniXcoder TEST Top500.")
                continue

            print(f"\n--- UniXcoder TEST: {language} ---")
            queries, code_texts, ground_truth = _read_language_inputs(plan_row)
            del ground_truth

            code_path = b.verify_file_info(
                plan_row["unixcoder_code_cache"],
                f"{language} UniXcoder code cache",
            )
            code = np.load(code_path, mmap_mode="r", allow_pickle=False)
            if len(code) != int(plan_row["codebase_rows"]):
                raise RuntimeError(
                    f"{language}: UniXcoder code-cache row mismatch."
                )

            validation = b.validate_unixcoder_code_cache(
                model,
                tokenizer,
                code_texts=code_texts,
                cached_embeddings=code,
                device=device,
                batch_size=b.UNIXCODER_BATCH_SIZE,
                validation_rows=b.UNIXCODER_CACHE_VALIDATION_ROWS,
                minimum_cosine=b.UNIXCODER_CACHE_MIN_COSINE,
            )
            validation_by_language[language] = dict(validation)
            print(
                f"{language}: UniXcoder code-cache validation {validation}"
            )

            query_path = output_dir / f"{language}_test_unixcoder_queries_f32.npy"
            query = b.load_or_encode_unixcoder_queries(
                model,
                tokenizer,
                queries,
                query_path,
                device,
                max_length=int(
                    protocol["controls"]["unixcoder"]["max_length"]
                ),
            )

            scores, indices = b.retrieve_embedding_topk(
                query_embeddings=query,
                code_embeddings=code,
                device=device,
                top_k=b.RETRIEVAL_K,
                query_chunk_size=b.DENSE_QUERY_CHUNK,
                code_chunk_size=b.DENSE_CODE_CHUNK,
                description=f"final {language} UniXcoder TEST top-{b.RETRIEVAL_K}",
            )
            b.validate_retrieval(
                indices,
                scores,
                int(plan_row["codebase_rows"]),
                b.RETRIEVAL_K,
            )

            b.save_stage(
                stage,
                language,
                "unixcoder",
                scores,
                indices,
                execution_plan_sha,
                protocol_sha,
                str(plan_row["test_file"]["sha256"]),
                str(plan_row["codebase_file"]["sha256"]),
            )

            del queries, code_texts, query, code, scores, indices
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        del model, tokenizer
        _cleanup_cuda()

    return validation_by_language


def build_bm25_and_finalize(
    execution_plan: Mapping[str, Any],
    output_dir: Path,
    execution_plan_sha: str,
    protocol: Mapping[str, Any],
    protocol_sha: str,
    device: torch.device,
    unix_validation_by_language: Mapping[str, Mapping[str, Any]],
) -> None:
    print("\n" + "=" * 92)
    print("OOM-SAFE PASS 4/4: PURE BM25 + FINAL CONTRACTS")
    print("=" * 92)

    per_language_execution = []

    for plan_row in execution_plan["per_language"]:
        language = str(plan_row["language"])
        print(f"\n--- BM25/finalize TEST: {language} ---")
        queries, code_texts, ground_truth = _read_language_inputs(plan_row)

        test_rows = int(plan_row["test_rows"])
        codebase_rows = int(plan_row["codebase_rows"])
        test_sha = str(plan_row["test_file"]["sha256"])
        codebase_sha = str(plan_row["codebase_file"]["sha256"])

        stages = {
            name: _stage_path(output_dir, language, name)
            for name in ("nomic", "qwen3", "unixcoder", "bm25")
        }

        # All dense stages must already exist.
        dense = {}
        for name in ("nomic", "qwen3", "unixcoder"):
            scores, indices = b.load_stage(
                stages[name],
                language,
                name,
                test_rows,
                codebase_rows,
                execution_plan_sha,
            )
            dense[name] = {
                "scores": scores,
                "indices": indices,
            }

        bm25_stats: Mapping[str, Any]
        if stages["bm25"].is_file():
            bm25_scores, bm25_indices = b.load_stage(
                stages["bm25"],
                language,
                "bm25",
                test_rows,
                codebase_rows,
                execution_plan_sha,
            )
            bm25_stats = {"status": "reused_completed_stage"}
            print(f"{language}: reusing completed pure-BM25 TEST Top500.")
        else:
            cfg = protocol["controls"]["bm25"]
            vectorizer, matrix, bm25_stats = b.build_bm25_matrix(
                code_texts,
                max_features=int(cfg["max_features"]),
                min_df=int(cfg["min_df"]),
                k1=float(cfg["k1"]),
                b=float(cfg["b"]),
            )
            bm25_scores, bm25_indices = b.retrieve_bm25_topk(
                vectorizer,
                matrix,
                queries=queries,
                top_k=b.RETRIEVAL_K,
                query_chunk_size=b.BM25_QUERY_CHUNK,
                description=f"final {language} pure BM25 TEST top-{b.RETRIEVAL_K}",
            )
            b.validate_retrieval(
                bm25_indices,
                bm25_scores,
                codebase_rows,
                b.RETRIEVAL_K,
            )
            b.save_stage(
                stages["bm25"],
                language,
                "bm25",
                bm25_scores,
                bm25_indices,
                execution_plan_sha,
                protocol_sha,
                test_sha,
                codebase_sha,
            )
            del vectorizer, matrix
            gc.collect()

        final_path = output_dir / f"{language}_test_retrievals_v1.npz"
        source_rows = np.arange(test_rows, dtype=np.int64)

        if final_path.is_file():
            with np.load(final_path, allow_pickle=False) as a:
                if b.scalar_string(a, "format") != b.RETRIEVAL_FORMAT:
                    raise RuntimeError(
                        f"Wrong final retrieval format: {final_path}"
                    )
                if (
                    b.scalar_string(a, "execution_plan_sha256")
                    != execution_plan_sha
                ):
                    raise RuntimeError(
                        f"Final retrieval plan mismatch: {final_path}"
                    )
                if not np.array_equal(
                    a["ground_truth_code_indices"],
                    ground_truth,
                ):
                    raise RuntimeError(
                        f"{language}: final retrieval GT changed."
                    )
            print(
                f"{language}: reusing completed final TEST retrieval contract."
            )
        else:
            b.atomic_save_npz(
                final_path,
                {
                    "format": np.asarray(b.RETRIEVAL_FORMAT),
                    "experiment_family": np.asarray(
                        "work2_repairability_frontier"
                    ),
                    "method_family": np.asarray(
                        "failure_repairability_frontier"
                    ),
                    "language": np.asarray(language),
                    "split": np.asarray("test"),
                    "source_row_indices": source_rows,
                    "ground_truth_code_indices": ground_truth.astype(np.int32),
                    "nomic_indices": dense["nomic"]["indices"].astype(np.int32),
                    "nomic_scores": dense["nomic"]["scores"].astype(np.float32),
                    "qwen3_indices": dense["qwen3"]["indices"].astype(np.int32),
                    "qwen3_scores": dense["qwen3"]["scores"].astype(np.float32),
                    "unixcoder_indices": dense["unixcoder"]["indices"].astype(np.int32),
                    "unixcoder_scores": dense["unixcoder"]["scores"].astype(np.float32),
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

        # Query caches must exist after the three dense passes.
        query_caches = {
            "nomic": b.checked_path(
                output_dir / f"{language}_test_nomic_queries_f32.npy",
                f"{language} Nomic TEST query cache",
            ),
            "qwen3": b.checked_path(
                output_dir / f"{language}_test_qwen3_queries_f32.npy",
                f"{language} Qwen3 TEST query cache",
            ),
            "unixcoder": b.checked_path(
                output_dir / f"{language}_test_unixcoder_queries_f32.npy",
                f"{language} UniXcoder TEST query cache",
            ),
        }

        per_language_execution.append(
            {
                "language": language,
                "test_rows": test_rows,
                "codebase_rows": codebase_rows,
                "retrieval_contract": b.checked_path(
                    final_path,
                    f"{language} final TEST retrieval",
                ),
                "stage_contracts": {
                    name: b.checked_path(
                        path,
                        f"{language} {name} TEST stage",
                    )
                    for name, path in stages.items()
                },
                "query_caches": query_caches,
                "unixcoder_code_cache_validation": dict(
                    unix_validation_by_language.get(
                        language,
                        {"status": "unknown"},
                    )
                ),
                "bm25_stats": dict(bm25_stats),
            }
        )

        del (
            queries,
            code_texts,
            ground_truth,
            dense,
            bm25_scores,
            bm25_indices,
        )
        gc.collect()

    manifest = {
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
        existing = b.load_json(manifest_path)
        if existing != manifest:
            raise RuntimeError(
                "Existing execution manifest differs from deterministic rebuild."
            )
    else:
        b.atomic_write_json(manifest_path, manifest)

    report = b.compute_final_report(
        execution_plan=execution_plan,
        output_dir=output_dir,
        execution_plan_sha=execution_plan_sha,
        protocol_sha=protocol_sha,
    )
    report_path = output_dir / "final_untouched_test_report.json"
    if report_path.is_file():
        existing = b.load_json(report_path)
        if existing != report:
            raise RuntimeError(
                "Existing final TEST report differs from deterministic recomputation."
            )
    else:
        b.atomic_write_json(report_path, report)

    print("\n===== FINAL UNTOUCHED TEST SUMMARY =====")
    print("decision =", report["decision"])
    print("test_rows =", report["test_rows"])
    for base in b.BASES:
        print(f"\nBASE = {base}")
        agg = report["aggregate"][base]
        print(
            f"AF={agg['availability_failures']} "
            f"({agg['availability_failure_rate_percent']:.4f}%)"
        )
        for budget in b.BUDGETS:
            row = agg["budgets"][str(budget)]
            print(
                f"B={budget:>3} "
                f"local={row['operators']['local_dense']['recovery_rate_percent']:.4f}% "
                f"peer={row['operators']['peer_strong']['recovery_rate_percent']:.4f}% "
                f"bm25={row['operators']['bm25']['recovery_rate_percent']:.4f}% "
                f"union={row['primary_union']['recovery_rate_percent']:.4f}% "
                f"union-best={row['primary_union']['minus_best_single_pp']:.4f}pp"
            )

    print("\nCONFIRMATORY ENDPOINTS")
    print(
        json.dumps(
            report["confirmatory_endpoints"],
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"saved: {report_path}")
    print("WORK2_FINAL_UNTOUCHED_TEST_COMPLETED_PENDING_AUDIT")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "OOM-safe runtime scheduler for the already-frozen final TEST "
            "Repairability Frontier execution plan."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=str(b.OUTPUT_DIR_DEFAULT),
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: original frozen builder remains unchanged. This reuses the
    # already-frozen execution plan. It must not regenerate or alter it.
    execution_plan = b.prepare_execution_plan(output_dir)
    execution_plan_path = output_dir / "final_test_execution_plan.json"
    execution_plan_sha = b.file_sha256(execution_plan_path)

    protocol_sha = b.file_sha256(b.PROTOCOL_PATH)
    protocol = b.load_json(b.PROTOCOL_PATH)
    device = torch.device(args.device)

    print(f"device={device}")
    print(f"execution_plan_sha256={execution_plan_sha}")
    print("scientific_protocol_changed=false")
    print("runtime_fix=sequential_model_residency")

    _write_correction_record(
        output_dir,
        execution_plan_path,
        Path(__file__).resolve(),
    )

    # Make allocator behavior more robust, but do not change model/math.
    if device.type == "cuda":
        torch.cuda.empty_cache()

    unix_validation = {}

    build_nomic_pass(
        execution_plan,
        output_dir,
        execution_plan_sha,
        protocol,
        protocol_sha,
        device,
    )

    build_qwen3_pass(
        execution_plan,
        output_dir,
        execution_plan_sha,
        protocol,
        protocol_sha,
        device,
    )

    unix_validation = build_unixcoder_pass(
        execution_plan,
        output_dir,
        execution_plan_sha,
        protocol,
        protocol_sha,
        device,
    )

    build_bm25_and_finalize(
        execution_plan,
        output_dir,
        execution_plan_sha,
        protocol,
        protocol_sha,
        device,
        unix_validation,
    )


if __name__ == "__main__":
    main()
