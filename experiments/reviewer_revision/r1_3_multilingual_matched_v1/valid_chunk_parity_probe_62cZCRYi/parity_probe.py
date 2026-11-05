from pathlib import Path
import os
import sys
import json
import hashlib
import subprocess
import traceback
import runpy

import numpy as np

ROOT = Path(
    "/root/autodl-tmp/lora-code-embeddings-main/text2code"
)

BASE = ROOT / (
    "results/reviewer_revision/"
    "r1_3_multilingual_matched_v1"
)

OUT = Path(os.environ["R13_PARITY_OUT"])

EVAL = BASE / (
    "full_valid_multi5k_seed13_v1/"
    "eval_valid_only.py"
)

FROZEN = ROOT / (
    "results/reviewer_revision/"
    "recal_valid_fresh_20260911/"
    "recal_main_frozen_config.json"
)

POOL = BASE / (
    "validation_pool_adapter_v2/"
    "candidate_pools/ruby_valid_nomic_top100.npz"
)

CANON = ROOT / (
    "results/reviewer_revision/"
    "recal_valid_fresh_20260911/"
    "validation_grid/candidate_pools/"
    "ruby_valid_nomic_top100.npz"
)

MODEL = BASE / "models/multi5k_seed13_v5"

CTRL = BASE / (
    "mono5k_training_controller_v1/"
    "run_mono5k_serial_v2.py"
)

HASHES = {
    str(EVAL):
        "08692a72ab8b6e46a1ce99f83d762cea02a933db4bd7cb54840c5877d59163bb",
    str(ROOT / "eval_llm_verifier.py"):
        "a3156c285728d54180ce0da1c65255a92adfe15f583dc987e47ad8a97b8e665a",
    str(FROZEN):
        "70789e10518f32a97f5990b234833fb0443fbcbe426e8a8c3515ab93e6b11b0f",
    str(POOL):
        "fc60c7012e7c2019f034e8d46171aa3220cf20c9e7c5afc3728aa56efa5c85ba",
    str(MODEL / "adapter_model.safetensors"):
        "c02e0fa78a351bff68fd63e6e28128e112a83a42850cbe47ef606168e045cc59",
}

def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(block)

    return h.hexdigest()

def need(condition, message):
    if not condition:
        raise RuntimeError(
            "FAIL_STOP: " + str(message)
        )

def analyze_case(folder, n):
    result = json.loads(
        (folder / "results.json").read_text()
    )

    need(
        "alpha=1" in result
        and "ruby" in result["alpha=1"],
        f"Missing alpha=1 / ruby in {folder.name}"
    )

    entry = result["alpha=1"]["ruby"]

    need(
        int(entry["n_queries"]) == n,
        f"Query count mismatch: {folder.name}"
    )

    score_files = list(
        (folder / "score_cache").glob("*.scores.npy")
    )

    rank_files = list(
        (folder / "ranks").glob("*.ranks.npy")
    )

    need(
        len(score_files) == 1,
        f"Unexpected score files: {folder.name}"
    )

    need(
        len(rank_files) == 1,
        f"Unexpected rank files: {folder.name}"
    )

    scores = np.load(
        score_files[0],
        allow_pickle=False
    )

    ranks = np.load(
        rank_files[0],
        allow_pickle=False
    )

    need(
        scores.shape == (n, 15),
        f"Score shape mismatch: {folder.name}"
    )

    need(
        ranks.shape == (n,),
        f"Rank shape mismatch: {folder.name}"
    )

    need(
        np.isfinite(scores).all(),
        f"Nonfinite score: {folder.name}"
    )

    need(
        np.issubdtype(ranks.dtype, np.integer)
        and np.all(ranks >= 0),
        f"Invalid rank: {folder.name}"
    )

    expected_mrr = (
        100.0 * np.mean(
            1.0 / (ranks.astype(np.float64) + 1.0)
        )
    )

    need(
        abs(expected_mrr - float(entry["mrr"]))
        <= 0.000051,
        f"MRR/rank mismatch: {folder.name}"
    )

    meta = {
        "score_sha256": sha(score_files[0]),
        "rank_sha256": sha(rank_files[0]),
        "result_sha256": sha(folder / "results.json"),
        "mrr": float(entry["mrr"]),
    }

    return scores, ranks, meta

def main():
    print("=" * 78)
    print("WORK1 R1-3 VALID NUMERICAL CHUNK PARITY")
    print("=" * 78)

    print("\n[1] FROZEN SHA256 GATE")

    for name, expected in HASHES.items():
        path = Path(name)

        need(
            path.is_file(),
            f"Missing frozen file: {path}"
        )

        need(
            sha(path) == expected,
            f"SHA256 mismatch: {path}"
        )

        print(
            "FROZEN_SHA256_PASS=",
            path,
            flush=True
        )

    print("\n[2] FROZEN VALID PROTOCOL")

    need(
        CANON.is_file(),
        "Canonical VALID pool missing"
    )

    cfg = json.loads(FROZEN.read_text())
    frozen = cfg["config"]

    need(
        cfg["selected_on_split"] == "valid",
        "Frozen configuration split mismatch"
    )

    need(
        int(frozen["verify_top_k"]) == 15,
        "Frozen verify depth mismatch"
    )

    need(
        float(frozen["fusion_weight"]) == 1.0,
        "Frozen fusion weight mismatch"
    )

    need(
        frozen["normalization"] == "zscore",
        "Frozen normalization mismatch"
    )

    with np.load(
        POOL, allow_pickle=False
    ) as pool, np.load(
        CANON, allow_pickle=False
    ) as canonical:

        need(
            pool["indices"].shape == (2209, 100),
            "Ruby pool shape mismatch"
        )

        need(
            canonical["gt_indices"].shape == (2209,),
            "Canonical GT shape mismatch"
        )

        need(
            np.array_equal(
                pool["indices"][:8],
                canonical["candidate_indices"][:8]
            ),
            "Candidate indices mismatch"
        )

        need(
            np.array_equal(
                pool["scores"][:8],
                canonical["candidate_scores"][:8]
            ),
            "Retrieval scores mismatch"
        )

        need(
            np.array_equal(
                pool["base_ranks"][:8],
                canonical["base_ranks"][:8]
            ),
            "Baseline ranks mismatch"
        )

    print("FROZEN_VALID_PROTOCOL=PASS")
    print("CANONICAL_GT_IDENTITY=PASS")

    print("\n[3] GPU AND STORAGE PREFLIGHT")

    controller = runpy.run_path(
        str(CTRL),
        run_name="r13_valid_gpu_probe_preflight"
    )

    controller["gpu_gate"]()
    controller["disk_gate"]()

    print("GPU_STORAGE_PREFLIGHT=PASS")

    print("\n[4] FULL VERSUS CHUNKED SCORING")

    cases = [
        ("full8", 0, 8),
        ("left4", 0, 4),
        ("right4", 4, 4),
    ]

    env = os.environ.copy()

    env["PYTHONPATH"] = (
        str(ROOT)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )

    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OMP_NUM_THREADS"] = "4"

    manifest = {}

    for name, offset, count in cases:
        directory = OUT / name

        directory.mkdir(exist_ok=False)
        (directory / "score_cache").mkdir()
        (directory / "ranks").mkdir()

        cmd = [
            sys.executable,
            "-u",
            "-B",
            str(EVAL),
            "--languages", "ruby",
            "--split", "valid",
            "--data-root",
            str(ROOT.parent / "data/CSN"),
            "--verifier-model",
            str(MODEL),
            "--verifier-base-model",
            "/root/autodl-tmp/hf_models/Qwen2.5-Coder-7B-Instruct",
            "--verifier-dtype", "float16",
            "--verifier-device-map", "cuda",
            "--verifier-mode", "yesno",
            "--verifier-batch-size", "1",
            "--max-length", "1024",
            "--max-query-chars", "512",
            "--max-code-chars", "3000",
            "--candidate-top-k", "100",
            "--candidate-pool-pattern",
            str(
                BASE /
                "validation_pool_adapter_v2/"
                "candidate_pools/"
                "{language}_{split}_nomic_top100.npz"
            ),
            "--candidate-pool-tag",
            "r13_validation_v2",
            "--frozen-config",
            str(FROZEN),
            "--query-offset", str(offset),
            "--max-queries", str(count),
            "--score-cache-dir",
            str(directory / "score_cache"),
            "--rank-output-dir",
            str(directory / "ranks"),
            "--json-output",
            str(directory / "results.json"),
            "--experiment-name",
            "r13_parity_" + name,
        ]

        (directory / "command.json").write_text(
            json.dumps(cmd, indent=2) + "\n"
        )

        print(
            f"CASE_BEGIN={name} "
            f"OFFSET={offset} COUNT={count}",
            flush=True
        )

        with (directory / "eval.log").open("x") as log:
            result = subprocess.run(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False
            )

        (directory / "exit_code.txt").write_text(
            str(result.returncode) + "\n"
        )

        need(
            result.returncode == 0,
            f"Evaluator failed: {name}; "
            f"see {directory / 'eval.log'}"
        )

        scores, ranks, meta = analyze_case(
            directory, count
        )

        manifest[name] = {
            "offset": offset,
            "count": count,
            **meta,
        }

        print(
            f"CASE_PASS={name} "
            f"MRR={meta['mrr']}",
            flush=True
        )

        del scores, ranks

    print("\n[5] NUMERICAL PARITY")

    full_scores, full_ranks, _ = analyze_case(
        OUT / "full8", 8
    )

    left_scores, left_ranks, _ = analyze_case(
        OUT / "left4", 4
    )

    right_scores, right_ranks, _ = analyze_case(
        OUT / "right4", 4
    )

    joined_scores = np.concatenate(
        [left_scores, right_scores],
        axis=0
    )

    joined_ranks = np.concatenate(
        [left_ranks, right_ranks],
        axis=0
    )

    score_equal = bool(
        np.array_equal(
            full_scores,
            joined_scores
        )
    )

    rank_equal = bool(
        np.array_equal(
            full_ranks,
            joined_ranks
        )
    )

    max_diff = float(
        np.max(
            np.abs(
                full_scores.astype(np.float64)
                - joined_scores.astype(np.float64)
            )
        )
    )

    full_mrr = float(
        np.mean(
            1.0 /
            (full_ranks.astype(np.float64) + 1.0)
        )
    )

    joined_mrr = float(
        np.mean(
            1.0 /
            (joined_ranks.astype(np.float64) + 1.0)
        )
    )

    mrr_equal = full_mrr == joined_mrr

    passed = (
        score_equal
        and rank_equal
        and mrr_equal
    )

    report = {
        "status": "PASS" if passed else "FAIL_STOP",
        "split": "valid",
        "query_count": 8,
        "chunk_ranges": [[0, 4], [4, 8]],
        "scores_bitwise_equal": score_equal,
        "ranks_bitwise_equal": rank_equal,
        "mrr_equal": mrr_equal,
        "max_absolute_score_difference": max_diff,
        "full_mrr_percent": full_mrr * 100,
        "joined_mrr_percent": joined_mrr * 100,
        "cases": manifest,
        "frozen_evaluator_modified": False,
        "test_accessed": False,
    }

    (OUT / "parity_report.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ) + "\n"
    )

    print(
        "SCORE_BITWISE_PARITY=",
        "PASS" if score_equal else "FAIL",
        flush=True
    )

    print(
        "RANK_BITWISE_PARITY=",
        "PASS" if rank_equal else "FAIL",
        flush=True
    )

    print(
        "MRR_PARITY=",
        "PASS" if mrr_equal else "FAIL",
        flush=True
    )

    print(
        "MAX_ABS_SCORE_DIFF=",
        max_diff,
        flush=True
    )

    need(
        passed,
        "Numerical chunk parity failed"
    )

    print("R13_VALID_CHUNK_PARITY_GATE=PASS")
    print("TEST_ACCESSED=NO")

if __name__ == "__main__":
    try:
        main()

    except Exception:
        traceback.print_exc()

        print(
            "R13_VALID_CHUNK_PARITY_GATE=FAIL_STOP",
            flush=True
        )

        sys.exit(2)
