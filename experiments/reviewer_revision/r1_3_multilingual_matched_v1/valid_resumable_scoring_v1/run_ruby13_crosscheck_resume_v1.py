from pathlib import Path
import os
import sys
import json
import hashlib
import subprocess
import runpy
import traceback

import numpy as np

ROOT = Path(
    "/root/autodl-tmp/lora-code-embeddings-main/text2code"
)
BASE = ROOT / (
    "results/reviewer_revision/"
    "r1_3_multilingual_matched_v1"
)
WORK = BASE / "valid_resumable_scoring_v1"

JOB = "mono5k_seed13_ruby"
JOBDIR = WORK / JOB

WORKER = WORK / "worker.py"

EVAL = BASE / (
    "full_valid_multi5k_seed13_v1/"
    "eval_valid_only.py"
)

MODEL = BASE / "models/mono5k_ruby_seed13_v1"

FROZEN = ROOT / (
    "results/reviewer_revision/"
    "recal_valid_fresh_20260911/"
    "recal_main_frozen_config.json"
)

POOL_DIR = BASE / (
    "validation_pool_adapter_v2/"
    "candidate_pools"
)

CONTROLLER = BASE / (
    "mono5k_training_controller_v1/"
    "run_mono5k_serial_v2.py"
)

CROSS = WORK / "independent_mono5k_ruby13_first8_v1"
RESUME_LOG = WORK / "ruby13_remaining4_v1.log"
RESUME_PID = WORK / "ruby13_remaining4_v1.pid"

BLOCK = JOBDIR / "chunks/q0000000_0000512"

EXPECTED_EVAL_SHA = (
    "08692a72ab8b6e46a1ce99f83d762cea02a933db4bd7cb54840c5877d59163bb"
)

EXPECTED_FIRST_BLOCK_SHA = (
    "19b235b6c24ade38b564ef818496012faea8fb79c9e824b2138a42adb95d23f1"
)

EXPECTED_FINGERPRINT = (
    "c680b34cf4f24a5c0bdc8ff138c5be4c6ae386830721340f3ac5d134dc600511"
)


def need(condition, message):
    if not condition:
        raise RuntimeError("FAIL_STOP: " + message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, obj):
    need(not path.exists(), "Output already exists: " + str(path))

    with path.open("x", encoding="utf-8") as f:
        json.dump(
            obj, f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True
        )
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def get_one(folder, pattern):
    files = list(folder.glob(pattern))

    need(
        len(files) == 1,
        f"Expected exactly one file: {folder}/{pattern}"
    )

    return files[0]


def check_process_conflicts():
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue

        if int(proc.name) == os.getpid():
            continue

        try:
            parts = [
                x.decode(errors="replace")
                for x in (proc / "cmdline")
                    .read_bytes().split(b"\0")
                if x
            ]
        except OSError:
            continue

        if not parts:
            continue

        if "python" not in Path(parts[0]).name.lower():
            continue

        command = " ".join(parts[1:])

        conflict = any(
            x in command
            for x in (
                "valid_resumable_scoring_v1/worker.py",
                "eval_valid_only.py",
                "train_qwen_verifier.py",
                "run_mono5k_serial_v2.py",
            )
        )

        need(
            not conflict,
            f"Concurrent GPU job PID={proc.name}: "
            + command[:200]
        )


def main():
    print("=" * 78)
    print("R1-3 INDEPENDENT MONO-RUBY VALID PARITY AND RESUME")
    print("=" * 78)

    print("\n[1] IMMUTABLE INPUT GATE", flush=True)

    need(
        WORKER.is_file()
        and EVAL.is_file()
        and MODEL.is_dir()
        and FROZEN.is_file(),
        "Required source/model/config missing"
    )

    need(
        sha(EVAL) == EXPECTED_EVAL_SHA,
        "Frozen evaluator changed"
    )

    need(
        not CROSS.exists(),
        "Independent crosscheck directory already exists"
    )

    need(
        not RESUME_LOG.exists()
        and not RESUME_PID.exists(),
        "Resume process was already requested"
    )

    check_process_conflicts()

    need(
        BLOCK.is_dir(),
        "First committed 512-query block missing"
    )

    meta = json.loads(
        (BLOCK / "commit.json").read_text()
    )

    need(
        meta["fingerprint"] == EXPECTED_FINGERPRINT,
        "Job fingerprint mismatch"
    )

    need(
        meta["start"] == 0
        and meta["stop"] == 512,
        "Block query range mismatch"
    )

    score_file = BLOCK / "scores.npy"
    rank_file = BLOCK / "ranks.npy"

    need(
        sha(score_file) == EXPECTED_FIRST_BLOCK_SHA,
        "Frozen first block score SHA256 mismatch"
    )

    need(
        meta["scores_sha256"] == sha(score_file)
        and meta["ranks_sha256"] == sha(rank_file),
        "Checkpoint SHA256 mismatch"
    )

    scores512 = np.load(
        score_file, allow_pickle=False
    )

    ranks512 = np.load(
        rank_file, allow_pickle=False
    )

    need(
        scores512.shape == (512, 15)
        and scores512.dtype == np.float32,
        "Invalid first-block score array"
    )

    need(
        ranks512.shape == (512,)
        and ranks512.dtype == np.int32,
        "Invalid first-block rank array"
    )

    print("FIRST_512_CHECKPOINT_SHA256=PASS", flush=True)

    print("\n[2] INDEPENDENT CHECKPOINT REPLAY", flush=True)

    env = os.environ.copy()
    env["R13_VALID_WORK_ROOT"] = str(WORK)
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    check = subprocess.run(
        [
            sys.executable,
            "-u",
            "-B",
            str(WORKER),
            "--job-id", JOB,
            "--verify-only",
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    check_log = WORK / "ruby13_cpu_recheck_v1.log"

    with check_log.open("x", encoding="utf-8") as f:
        f.write(check.stdout)

    need(
        check.returncode == 0,
        "Worker checkpoint replay failed: "
        + str(check_log)
    )

    need(
        "REUSED_VERIFIED_BLOCK=0:512"
        in check.stdout,
        "First block was not verified"
    )

    need(
        "R13_VALID_PARTIAL_CHECKPOINT=PASS"
        in check.stdout,
        "Partial checkpoint verification missing"
    )

    print("FROZEN_RERANK_REPLAY=PASS", flush=True)

    print("\n[3] GPU PREFLIGHT", flush=True)

    controller = runpy.run_path(
        str(CONTROLLER),
        run_name="r13_ruby13_independent_preflight"
    )

    controller["gpu_gate"]()
    controller["disk_gate"]()

    print("GPU_AND_STORAGE_GATE=PASS", flush=True)

    print("\n[4] INDEPENDENT 8-QUERY FROZEN EVALUATION", flush=True)

    CROSS.mkdir(mode=0o700)

    (CROSS / "score_cache").mkdir()
    (CROSS / "ranks").mkdir()

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
        str(POOL_DIR / "{language}_{split}_nomic_top100.npz"),

        "--candidate-pool-tag",
        "r13_validation_v2",

        "--frozen-config",
        str(FROZEN),

        "--query-offset", "0",
        "--max-queries", "8",

        "--score-cache-dir",
        str(CROSS / "score_cache"),

        "--rank-output-dir",
        str(CROSS / "ranks"),

        "--json-output",
        str(CROSS / "results.json"),

        "--experiment-name",
        "r13_independent_mono5k_ruby13_first8",
    ]

    atomic_json(CROSS / "command.json", cmd)

    with (CROSS / "eval.log").open(
        "x", encoding="utf-8"
    ) as log:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )

    atomic_json(
        CROSS / "exit_code.json",
        {"returncode": result.returncode},
    )

    need(
        result.returncode == 0,
        "Independent frozen evaluator failed; "
        "see " + str(CROSS / "eval.log")
    )

    independent_score_path = get_one(
        CROSS / "score_cache", "*.scores.npy"
    )

    independent_rank_path = get_one(
        CROSS / "ranks", "*.ranks.npy"
    )

    independent_scores = np.load(
        independent_score_path,
        allow_pickle=False
    )

    independent_ranks = np.load(
        independent_rank_path,
        allow_pickle=False
    )

    need(
        independent_scores.shape == (8, 15)
        and independent_ranks.shape == (8,),
        "Independent array shape mismatch"
    )

    result_json = json.loads(
        (CROSS / "results.json").read_text()
    )

    need(
        int(
            result_json["alpha=1"]["ruby"]["n_queries"]
        ) == 8,
        "Independent evaluated query count mismatch"
    )

    print("INDEPENDENT_FROZEN_EVALUATOR=PASS", flush=True)

    print("\n[5] EXACT ARRAY PARITY", flush=True)

    score_equal = bool(
        np.array_equal(
            scores512[:8],
            independent_scores,
        )
    )

    rank_equal = bool(
        np.array_equal(
            ranks512[:8],
            independent_ranks,
        )
    )

    max_diff = float(
        np.max(
            np.abs(
                scores512[:8].astype(np.float64)
                - independent_scores.astype(np.float64)
            )
        )
    )

    report = {
        "status": (
            "PASS"
            if score_equal and rank_equal
            else "FAIL_STOP"
        ),
        "job_id": JOB,
        "split": "valid",
        "comparison_queries": 8,
        "score_bitwise_equal": score_equal,
        "rank_bitwise_equal": rank_equal,
        "max_absolute_score_difference": max_diff,
        "original_block_score_sha256":
            sha(score_file),
        "independent_score_sha256":
            sha(independent_score_path),
        "independent_rank_sha256":
            sha(independent_rank_path),
        "worker_sha256": sha(WORKER),
        "frozen_evaluator_sha256": sha(EVAL),
        "test_accessed": False,
    }

    atomic_json(CROSS / "comparison.json", report)

    print(
        "INDEPENDENT_SCORE_BITWISE_PARITY=",
        "PASS" if score_equal else "FAIL",
        flush=True,
    )

    print(
        "INDEPENDENT_RANK_BITWISE_PARITY=",
        "PASS" if rank_equal else "FAIL",
        flush=True,
    )

    print(
        "MAX_ABS_SCORE_DIFFERENCE=",
        max_diff,
        flush=True,
    )

    need(
        score_equal and rank_equal,
        "Independent scoring mismatch: "
        "do not resume full evaluation"
    )

    atomic_json(CROSS / "PASS.json", report)

    print("R13_RUBY13_INDEPENDENT_GATE=PASS", flush=True)

    print("\n[6] RESUME REMAINING RUBY CHUNKS", flush=True)

    check_process_conflicts()

    need(
        not RESUME_LOG.exists()
        and not RESUME_PID.exists(),
        "Resume log or PID file already exists"
    )

    # Launch the original frozen-function worker unchanged.
    # It revalidates the completed first block and
    # processes only the four missing chunks.

    with RESUME_LOG.open(
        "x", encoding="utf-8"
    ) as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-B",
                str(WORKER),
                "--job-id", JOB,
                "--max-new-chunks", "0",
            ],
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    with RESUME_PID.open(
        "x", encoding="utf-8"
    ) as f:
        f.write(str(proc.pid) + "\n")

    print("RUBY13_RESUME=LAUNCH_REQUESTED", flush=True)
    print("RESUME_PID=", proc.pid, flush=True)
    print("RESUME_LOG=", RESUME_LOG, flush=True)
    print("NEW_MODEL_TRAINING=NO", flush=True)
    print("FROZEN_EVALUATOR_MODIFIED=NO", flush=True)
    print("TEST_ACCESSED=NO", flush=True)
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print(
            "R13_RUBY13_INDEPENDENT_GATE=FAIL_STOP",
            flush=True,
        )
        sys.exit(2)
