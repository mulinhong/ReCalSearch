from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(
    "/root/autodl-tmp/lora-code-embeddings-main/results/"
    "work2_repairability_frontier_v1"
)

CHAR_DIR = ROOT / "repairability_frontier_characterization_v1"

CHAR_REPORT = (
    CHAR_DIR / "repairability_frontier_characterization_train_report.json"
)

CHAR_AUDIT = (
    CHAR_DIR / "repairability_frontier_characterization_train_audit.json"
)

OUTPUT_DIR = ROOT / "final_untouched_test_protocol_v1"

OUTPUT_FILE = OUTPUT_DIR / "final_untouched_test_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    report = load_json(CHAR_REPORT)
    audit = load_json(CHAR_AUDIT)

    if report.get("decision") != "READY_FOR_FROZEN_TEST_PROTOCOL":
        raise RuntimeError(
            "TRAIN characterization is not ready for frozen TEST."
        )

    if audit.get("audit_passed") is not True:
        raise RuntimeError(
            "TRAIN characterization audit did not pass."
        )

    if audit.get("decision") != "READY_FOR_FROZEN_TEST_PROTOCOL":
        raise RuntimeError(
            "Audited decision is not READY_FOR_FROZEN_TEST_PROTOCOL."
        )

    if audit.get("valid_accessed") is not False:
        raise RuntimeError("VALID-access flag violation.")

    if audit.get("test_accessed") is not False:
        raise RuntimeError(
            "TEST was accessed before final protocol freeze."
        )

    if audit.get("work1_used") is not False:
        raise RuntimeError("Work1 leakage flag violation.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        existing = load_json(OUTPUT_FILE)
        print(
            "REUSED_FROZEN_FINAL_UNTOUCHED_TEST_PROTOCOL_SHA256="
            + sha256(OUTPUT_FILE)
        )
        print(json.dumps(existing, indent=2, ensure_ascii=False))
        return

    protocol = {
        "format": "work2_final_untouched_test_protocol_v1",

        "experiment_family":
            "work2_repairability_frontier",

        "method_family":
            "failure_repairability_frontier",

        "scientific_role": (
            "Single final confirmatory evaluation on the previously "
            "untouched CodeSearchNet TEST split. All hypotheses, "
            "retrievers, operators, candidate budgets, statistics and "
            "interpretation rules are frozen before TEST access."
        ),

        "frozen_after_train_evidence": {
            "characterization_report": str(
                CHAR_REPORT.resolve()
            ),
            "characterization_report_sha256":
                sha256(CHAR_REPORT),

            "characterization_audit": str(
                CHAR_AUDIT.resolve()
            ),
            "characterization_audit_sha256":
                sha256(CHAR_AUDIT),

            "train_decision":
                report["decision"],
        },

        "test_access_policy": {
            "test_access_before_protocol_freeze": False,

            "single_final_test_evaluation": True,

            "no_hyperparameter_tuning_on_test": True,

            "no_operator_selection_from_test_results": True,

            "no_budget_selection_from_test_results": True,

            "no_model_selection_from_test_results": True,

            "no_predictor_training": True,

            "no_ocre_or_docre_as_primary_method": True,

            "allowed_rerun": (
                "Only deterministic continuation after interruption "
                "or correction of an implementation bug that does not "
                "alter the frozen scientific protocol. Any such event "
                "must be documented."
            ),
        },

        "dataset": {
            "name": "CodeSearchNet",
            "split": "TEST",

            "languages": [
                "ruby",
                "javascript",
                "php",
                "python",
                "java",
                "go",
            ],

            "annotation_caution": (
                "The benchmark provides an annotated target. "
                "Availability failure therefore means that the "
                "annotated target is absent from base Top100; "
                "it does not prove that no other semantically "
                "relevant implementation exists."
            ),
        },

        "base_retrievers": {
            "nomic": {
                "model":
                    "/root/autodl-tmp/hf_models/nomic-embed-code",

                "max_length": 512,

                "query_prefix": (
                    "Represent this query for searching relevant code: "
                ),

                "code_prefix": "",

                "pooling":
                    "last valid token + L2 normalization",
            },

            "qwen3": {
                "model": (
                    "/root/autodl-tmp/hf_models/"
                    "Qwen3-Embedding-4B"
                ),

                "dtype": "bfloat16",

                "max_length": 512,

                "padding_side": "left",

                "pooling":
                    "last token + L2 normalization",

                "query_instruction": (
                    "Given a natural language query, retrieve "
                    "the most relevant code snippet that implements "
                    "the described functionality"
                ),

                "query_format":
                    "Instruct: {task}\\nQuery:{query}",

                "code_format":
                    "raw code without instruction",
            },
        },

        "controls": {
            "unixcoder": {
                "model":
                    "/root/autodl-tmp/hf_models/unixcoder-base",

                "max_length": 512,

                "pooling":
                    "masked mean + L2 normalization",

                "role":
                    "legacy weaker dense-retriever control",
            },

            "bm25": {
                "type":
                    "fresh pure BM25",

                "k1": 1.5,

                "b": 0.75,

                "max_features": 50000,

                "min_df": 2,

                "role":
                    "sparse repair operator",
            },
        },

        "retrieval_contract": {
            "base_top_k": 100,

            "retrieval_depth": 500,

            "candidate_budgets": [
                5,
                10,
                20,
                50,
                100,
            ],

            "primary_budget": 20,
        },

        "failure_states": {
            "stable":
                "annotated target at base rank 1",

            "ranking_failure":
                "annotated target at base ranks 2..100",

            "availability_failure":
                "annotated target absent from base Top100",
        },

        "failure_depth_bins": {
            "rank101_120":
                "base rank 101..120",

            "rank121_200":
                "base rank 121..200",

            "rank201_500":
                "base rank 201..500",

            "rank_gt500":
                "annotated target absent from base Top500",
        },

        "operators": {
            "local_dense": (
                "base ranks 101..(100+B)"
            ),

            "peer_strong": {
                "nomic_base":
                    "first B Qwen3 candidates absent from "
                    "Nomic Top100",

                "qwen3_base":
                    "first B Nomic candidates absent from "
                    "Qwen3 Top100",
            },

            "unixcoder_control": (
                "first B UniXcoder candidates absent from "
                "base Top100"
            ),

            "bm25": (
                "first B pure-BM25 candidates absent from "
                "base Top100"
            ),

            "equal_budget": True,

            "never_reorder_or_delete_base_top100": True,
        },

        "primary_operator_family": [
            "local_dense",
            "peer_strong",
            "bm25",
        ],

        "legacy_control_operator": [
            "unixcoder_control",
        ],

        "primary_union": [
            "local_dense",
            "peer_strong",
            "bm25",
        ],

        "full_union": [
            "local_dense",
            "peer_strong",
            "unixcoder_control",
            "bm25",
        ],

        "primary_confirmatory_endpoint": {
            "budget": 20,

            "metric": (
                "PrimaryUnion Recovery@20 minus "
                "BestPrimarySingle Recovery@20"
            ),

            "evaluated_separately_for": [
                "Nomic base",
                "Qwen3 base",
            ],

            "statistical_test": (
                "paired query bootstrap stratified by "
                "programming language"
            ),

            "bootstrap_repetitions": 2000,

            "bootstrap_seed": 20260912,

            "success_condition": (
                "point estimate > 0 and 95% CI lower bound > 0 "
                "for both base retrievers"
            ),
        },

        "secondary_confirmatory_endpoints": {
            "peer_exclusive_repair": {
                "budget": 20,

                "criterion": (
                    "peer-strong has at least one full-exclusive "
                    "recovery in at least 5 of 6 languages "
                    "for each base direction"
                ),
            },

            "leave_one_language_out": {
                "budget": 20,

                "criterion": (
                    "PrimaryUnion minus BestPrimarySingle remains "
                    "positive after removing each language, "
                    "for both base retrievers"
                ),
            },
        },

        "secondary_characterization": {
            "budget_frontier": [
                5,
                10,
                20,
                50,
                100,
            ],

            "failure_depth_frontier": [
                "rank101_120",
                "rank121_200",
                "rank201_500",
                "rank_gt500",
            ],

            "report_per_language": True,

            "report_aggregate": True,

            "report_operator_overlap": True,

            "report_full_exclusive_recovery": True,

            "report_primary_union": True,

            "report_full_union": True,
        },

        "final_test_interpretation": {
            "H1_primary_union_advantage": (
                "CONFIRMED only if the frozen primary endpoint "
                "passes independently for both Nomic and Qwen3."
            ),

            "H2_bidirectional_peer_complementarity": (
                "CONFIRMED only if the frozen peer-exclusive "
                "criterion passes in both directions."
            ),

            "H3_language_robustness": (
                "CONFIRMED only if every leave-one-language-out "
                "aggregate retains a positive union advantage."
            ),

            "predictor_claim": (
                "No claim of deployable low-budget selective repair "
                "is made. OCRE-v1 was NO-GO and D-OCRE was CLOSED."
            ),

            "no_post_test_redefinition": True,
        },

        "valid_accessed_before_freeze": False,
        "test_accessed_before_freeze": False,
        "work1_used": False,
    }

    protocol["implementation"] = {
        "freeze_script": str(
            Path(__file__).resolve()
        ),

        "freeze_script_sha256":
            sha256(Path(__file__).resolve()),
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            protocol,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "FROZEN_FINAL_UNTOUCHED_TEST_PROTOCOL_SHA256="
        + sha256(OUTPUT_FILE)
    )

    print(
        "saved:",
        OUTPUT_FILE,
    )

    print(
        "WORK2_FINAL_UNTOUCHED_TEST_PROTOCOL_FROZEN"
    )


if __name__ == "__main__":
    main()