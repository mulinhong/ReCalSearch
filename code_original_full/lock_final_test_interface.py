from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ============================================================
# Frozen protocol
# ============================================================

ROOT = Path(
    "/root/autodl-tmp/lora-code-embeddings-main"
).resolve()

TEXT2CODE = ROOT / "text2code"

RESULT_ROOT = (
    ROOT
    / "results"
    / "work2_repairability_frontier_v1"
)

PROTOCOL_DIR = (
    RESULT_ROOT
    / "final_untouched_test_protocol_v1"
)

PROTOCOL_PATH = (
    PROTOCOL_DIR
    / "final_untouched_test_protocol.json"
)

EXPECTED_PROTOCOL_SHA256 = (
    "6b68efed7814e5ef71e470647b65c6e24"
    "a7ca6dde98857bf16290ca8101b6e3c"
)

OUTPUT_PATH = (
    PROTOCOL_DIR
    / "final_test_interface_lock.json"
)

DATA_ROOT = (
    ROOT
    / "data"
    / "CSN"
)

LANGUAGES = (
    "ruby",
    "javascript",
    "php",
    "python",
    "java",
    "go",
)

# Known historical cache/artifact roots.
NOMIC_CACHE_ROOT = (
    ROOT
    / "data"
    / "rerank"
    / "coderank_cache_oldcsn"
)

UNIXCODER_CACHE_ROOT = (
    RESULT_ROOT
    / "external_candidates"
    / "unixcoder_base_cache"
)

# The actual UniXcoder cache may also live under the older
# candidate-recovery result tree.
UNIXCODER_ALT_ROOT = (
    ROOT
    / "results"
    / "work2_candidate_recovery_oldcsn"
    / "external_candidates"
    / "unixcoder_base_cache"
)

COMPARABLE_DIR = (
    RESULT_ROOT
    / "comparable_retriever_probe_train"
)

OLD_BM25_ROOT = (
    ROOT
    / "results"
    / "bm25_nomic_hybrid_oldcsn"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def count_jsonl(path: Path) -> int:
    count = 0

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for line in handle:
            if line.strip():
                count += 1

    return count


def inspect_first_jsonl_row(
    path: Path,
) -> Dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:

        for line in handle:
            if not line.strip():
                continue

            row = json.loads(line)

            if not isinstance(row, dict):
                return {
                    "type": type(row).__name__,
                }

            summary: Dict[str, Any] = {
                "keys": sorted(row.keys()),
            }

            # Only schema/type/length information.
            # Do not print actual TEST content.
            fields = {}

            for key, value in row.items():
                item = {
                    "type": type(value).__name__,
                }

                if isinstance(value, str):
                    item["length"] = len(value)

                elif isinstance(value, list):
                    item["length"] = len(value)

                fields[key] = item

            summary["fields"] = fields

            return summary

    return {
        "keys": [],
        "fields": {},
    }


def find_test_files(
    language_dir: Path,
) -> List[Path]:
    if not language_dir.is_dir():
        return []

    candidates: List[Path] = []

    preferred = [
        "test.jsonl",
        "test.json",
        "test.parquet",
        "test.csv",
    ]

    for name in preferred:
        path = language_dir / name

        if path.is_file():
            candidates.append(path)

    # Do not recursively wander outside the language directory.
    for path in sorted(language_dir.iterdir()):
        if not path.is_file():
            continue

        lower = path.name.lower()

        if "test" not in lower:
            continue

        if path not in candidates:
            candidates.append(path)

    return candidates


def find_codebase_files(
    language_dir: Path,
) -> List[Path]:
    if not language_dir.is_dir():
        return []

    names = [
        "codebase.jsonl",
        "codebase.json",
        "codebase.parquet",
    ]

    rows = []

    for name in names:
        path = language_dir / name

        if path.is_file():
            rows.append(path)

    return rows


def safe_npy_shape(
    path: Path,
) -> Dict[str, Any]:
    try:
        arr = np.load(
            path,
            mmap_mode="r",
            allow_pickle=False,
        )

        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }

    except Exception as exc:
        return {
            "error": repr(exc),
        }


def inventory_numpy_files(
    root: Path,
    language: str,
    require_test: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    if not root.is_dir():
        return []

    rows = []

    language_lower = language.lower()

    for path in sorted(root.rglob("*.npy")):
        name = path.name.lower()

        if language_lower not in name:
            continue

        has_test = "test" in name

        if require_test is True and not has_test:
            continue

        if require_test is False and has_test:
            continue

        item = {
            "path": str(path.resolve()),
            "name": path.name,
            "size_bytes": int(
                path.stat().st_size
            ),
        }

        item.update(
            safe_npy_shape(path)
        )

        rows.append(item)

    return rows


def inventory_npz_files(
    root: Path,
    language: str,
) -> List[Dict[str, Any]]:
    if not root.is_dir():
        return []

    rows = []

    language_lower = language.lower()

    for path in sorted(root.rglob("*.npz")):
        name = path.name.lower()

        if language_lower not in name:
            continue

        rows.append(
            {
                "path": str(path.resolve()),
                "name": path.name,
                "size_bytes": int(
                    path.stat().st_size
                ),
            }
        )

    return rows


def inventory_old_test_artifacts(
    root: Path,
    language: str,
) -> List[str]:
    if not root.is_dir():
        return []

    rows = []

    language_lower = language.lower()

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        name = path.name.lower()

        if (
            language_lower in name
            and "test" in name
        ):
            rows.append(
                str(path.resolve())
            )

    return rows


def inspect_dataset_file(
    path: Path,
) -> Dict[str, Any]:
    suffix = path.suffix.lower()

    result = {
        "path": str(path.resolve()),
        "size_bytes": int(
            path.stat().st_size
        ),
        "suffix": suffix,
    }

    if suffix == ".jsonl":
        result["rows"] = count_jsonl(path)
        result["first_row_schema"] = (
            inspect_first_jsonl_row(path)
        )

    elif suffix == ".json":
        data = load_json(path)

        if isinstance(data, list):
            result["rows"] = len(data)

            if data and isinstance(
                data[0],
                dict,
            ):
                result[
                    "first_row_schema"
                ] = {
                    "keys": sorted(
                        data[0].keys()
                    ),
                }

        else:
            result["json_type"] = (
                type(data).__name__
            )

    else:
        result["rows"] = None

    return result


def main() -> None:
    # --------------------------------------------------------
    # 1. Verify frozen protocol bytes.
    # --------------------------------------------------------

    if not PROTOCOL_PATH.is_file():
        raise FileNotFoundError(
            PROTOCOL_PATH
        )

    actual_protocol_sha = sha256(
        PROTOCOL_PATH
    )

    if (
        actual_protocol_sha
        != EXPECTED_PROTOCOL_SHA256
    ):
        raise RuntimeError(
            "Frozen protocol SHA mismatch.\n"
            f"expected={EXPECTED_PROTOCOL_SHA256}\n"
            f"actual  ={actual_protocol_sha}"
        )

    protocol = load_json(
        PROTOCOL_PATH
    )

    if (
        protocol.get(
            "test_accessed_before_freeze"
        )
        is not False
    ):
        raise RuntimeError(
            "Protocol does not prove untouched "
            "TEST before freeze."
        )

    # --------------------------------------------------------
    # 2. Interface-only TEST inspection.
    #
    # No retrieval.
    # No GT-rank computation.
    # No repairability computation.
    # No metric computation.
    # --------------------------------------------------------

    report: Dict[str, Any] = {
        "format":
            "work2_final_test_interface_lock_v1",

        "experiment_family":
            "work2_repairability_frontier",

        "frozen_protocol": {
            "path":
                str(
                    PROTOCOL_PATH.resolve()
                ),

            "sha256":
                actual_protocol_sha,
        },

        "interface_lock_scope": {
            "test_content_used_for_metrics":
                False,

            "test_retrieval_executed":
                False,

            "test_ground_truth_ranks_computed":
                False,

            "test_repairability_computed":
                False,

            "test_metrics_computed":
                False,

            "purpose": (
                "Read-only schema, row-count, "
                "codebase and cache inventory "
                "after final TEST protocol freeze."
            ),
        },

        "dataset_root":
            str(
                DATA_ROOT.resolve()
            ),

        "languages": {},

        "reuse_policy": {
            "qwen3_train_code_embedding_cache": (
                "Potentially reusable because codebase "
                "is split-independent; exact cache/source "
                "alignment must be checked before final "
                "TEST builder."
            ),

            "qwen3_test_query_embeddings": (
                "Must be encoded fresh unless an exact "
                "frozen-protocol-compatible TEST query "
                "cache is explicitly discovered."
            ),

            "nomic_test_query_cache": (
                "May be reused only if exact TEST cache "
                "and frozen encoding contract match."
            ),

            "unixcoder_test_query_cache": (
                "May be reused only if exact TEST cache "
                "and frozen encoding contract match."
            ),

            "old_bm25_hybrid_test_artifacts": (
                "FORBIDDEN as primary TEST repair operator "
                "because frozen protocol requires fresh "
                "pure BM25, not historical BM25-Nomic "
                "hybrid outputs."
            ),

            "work1_test_outputs": (
                "FORBIDDEN. Work1 verifier/reranker/fusion "
                "is outside the frozen Work2 TEST protocol."
            ),
        },

        "valid_accessed":
            False,

        "test_interface_accessed":
            True,

        "test_outcomes_observed":
            False,

        "work1_used":
            False,
    }

    for language in LANGUAGES:
        print(
            "\n"
            + "=" * 80
        )
        print(
            "TEST INTERFACE:",
            language,
        )
        print(
            "=" * 80
        )

        language_dir = (
            DATA_ROOT
            / language
        )

        language_row: Dict[
            str,
            Any,
        ] = {
            "language_dir":
                str(
                    language_dir.resolve()
                ),

            "language_dir_exists":
                language_dir.is_dir(),
        }

        test_files = (
            find_test_files(
                language_dir
            )
        )

        codebase_files = (
            find_codebase_files(
                language_dir
            )
        )

        language_row[
            "test_files"
        ] = [
            inspect_dataset_file(
                path
            )
            for path in test_files
        ]

        language_row[
            "codebase_files"
        ] = [
            inspect_dataset_file(
                path
            )
            for path in codebase_files
        ]

        # Nomic cache inventory.
        language_row[
            "nomic_test_npy_candidates"
        ] = (
            inventory_numpy_files(
                NOMIC_CACHE_ROOT,
                language,
                require_test=True,
            )
        )

        # UniXcoder cache inventory:
        # inspect both known possible roots.
        unix_rows = []

        for root in (
            UNIXCODER_CACHE_ROOT,
            UNIXCODER_ALT_ROOT,
        ):
            if not root.is_dir():
                continue

            unix_rows.extend(
                inventory_numpy_files(
                    root,
                    language,
                    require_test=True,
                )
            )

        language_row[
            "unixcoder_test_npy_candidates"
        ] = unix_rows

        # Qwen3 train-time code cache inventory.
        # We only inspect shape/path.
        qwen_code_candidates = []

        if COMPARABLE_DIR.is_dir():
            for path in sorted(
                COMPARABLE_DIR.glob(
                    f"{language}_qwen3_code_embeddings*.npy"
                )
            ):
                item = {
                    "path":
                        str(
                            path.resolve()
                        ),

                    "size_bytes":
                        int(
                            path.stat().st_size
                        ),
                }

                item.update(
                    safe_npy_shape(
                        path
                    )
                )

                qwen_code_candidates.append(
                    item
                )

        language_row[
            "qwen3_code_cache_candidates"
        ] = qwen_code_candidates

        # Any existing Qwen3 TEST caches?
        qwen_test_candidates = []

        if RESULT_ROOT.is_dir():
            for path in sorted(
                RESULT_ROOT.rglob(
                    "*.npy"
                )
            ):
                lower = (
                    path.name.lower()
                )

                if (
                    language in lower
                    and "qwen3" in lower
                    and "test" in lower
                ):
                    item = {
                        "path":
                            str(
                                path.resolve()
                            ),

                        "size_bytes":
                            int(
                                path.stat().st_size
                            ),
                    }

                    item.update(
                        safe_npy_shape(
                            path
                        )
                    )

                    qwen_test_candidates.append(
                        item
                    )

        language_row[
            "qwen3_test_cache_candidates"
        ] = qwen_test_candidates

        # Historical BM25 test outputs:
        # inventory only, never use them here.
        language_row[
            "old_bm25_hybrid_test_artifacts"
        ] = (
            inventory_old_test_artifacts(
                OLD_BM25_ROOT,
                language,
            )
        )

        report[
            "languages"
        ][language] = language_row

        print(
            "test files:"
        )

        for row in language_row[
            "test_files"
        ]:
            print(
                " ",
                row["path"],
                "rows=",
                row.get(
                    "rows"
                ),
            )

        print(
            "codebase files:"
        )

        for row in language_row[
            "codebase_files"
        ]:
            print(
                " ",
                row["path"],
                "rows=",
                row.get(
                    "rows"
                ),
            )

        print(
            "Nomic TEST cache candidates:",
            len(
                language_row[
                    "nomic_test_npy_candidates"
                ]
            ),
        )

        for row in language_row[
            "nomic_test_npy_candidates"
        ]:
            print(
                " ",
                row["path"],
                "shape=",
                row.get(
                    "shape"
                ),
                "dtype=",
                row.get(
                    "dtype"
                ),
            )

        print(
            "UniXcoder TEST cache candidates:",
            len(
                language_row[
                    "unixcoder_test_npy_candidates"
                ]
            ),
        )

        for row in language_row[
            "unixcoder_test_npy_candidates"
        ]:
            print(
                " ",
                row["path"],
                "shape=",
                row.get(
                    "shape"
                ),
                "dtype=",
                row.get(
                    "dtype"
                ),
            )

        print(
            "Qwen3 code cache candidates:",
            len(
                language_row[
                    "qwen3_code_cache_candidates"
                ]
            ),
        )

        for row in language_row[
            "qwen3_code_cache_candidates"
        ]:
            print(
                " ",
                row["path"],
                "shape=",
                row.get(
                    "shape"
                ),
                "dtype=",
                row.get(
                    "dtype"
                ),
            )

        print(
            "Qwen3 TEST cache candidates:",
            len(
                language_row[
                    "qwen3_test_cache_candidates"
                ]
            ),
        )

        for row in language_row[
            "qwen3_test_cache_candidates"
        ]:
            print(
                " ",
                row["path"],
                "shape=",
                row.get(
                    "shape"
                ),
                "dtype=",
                row.get(
                    "dtype"
                ),
            )

        print(
            "Old BM25 hybrid TEST artifacts:",
            len(
                language_row[
                    "old_bm25_hybrid_test_artifacts"
                ]
            ),
        )

    # --------------------------------------------------------
    # 3. Freeze interface inventory.
    # --------------------------------------------------------

    if OUTPUT_PATH.exists():
        existing = load_json(
            OUTPUT_PATH
        )

        if existing != report:
            raise RuntimeError(
                "Existing TEST interface lock differs "
                "from current read-only inventory."
            )

        print(
            "\nREUSED_FINAL_TEST_INTERFACE_LOCK_SHA256="
            + sha256(
                OUTPUT_PATH
            )
        )

    else:
        OUTPUT_PATH.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(
            "\nFINAL_TEST_INTERFACE_LOCK_SHA256="
            + sha256(
                OUTPUT_PATH
            )
        )

    print(
        "saved:",
        OUTPUT_PATH,
    )

    print(
        "WORK2_FINAL_TEST_INTERFACE_LOCK_DONE"
    )


if __name__ == "__main__":
    main()