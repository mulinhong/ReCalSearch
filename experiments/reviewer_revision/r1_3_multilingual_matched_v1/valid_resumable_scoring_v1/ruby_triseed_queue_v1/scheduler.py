from pathlib import Path
import os
import sys
import json
import hashlib
import subprocess
import fcntl
import traceback
import time

import numpy as np

ROOT = Path(
    "/root/autodl-tmp/lora-code-embeddings-main/text2code"
)
BASE = ROOT / (
    "results/reviewer_revision/r1_3_multilingual_matched_v1"
)
WORK = BASE / "valid_resumable_scoring_v1"
QUEUE = WORK / "ruby_triseed_queue_v1"
PLAN = Path(
    "/root/autodl-tmp/recal_r13_valid_plan_MiFZlVso/"
    "evaluation_jobs.json"
)
WORKER = WORK / "worker.py"

JOBS = [
    "mono5k_seed42_ruby",
    "multi5k_seed42_ruby",
    "mono5k_seed87_ruby",
    "multi5k_seed87_ruby",
]

BASELINE = (
    BASE / "full_valid_multi5k_seed13_v1/"
    "languages/ruby/results.json"
)
MONO13 = (
    WORK / "mono5k_seed13_ruby/final/commit.json"
)


def need(ok, why):
    if not ok:
        raise RuntimeError("FAIL_STOP: " + why)


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for v in iter(
            lambda: f.read(4 * 1024 * 1024), b""
        ):
            h.update(v)
    return h.hexdigest()


def running_conflicts():
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue

        if int(p.name) == os.getpid():
            continue

        try:
            args = [
                x.decode(errors="replace")
                for x in (
                    p / "cmdline"
                ).read_bytes().split(b"\0")
                if x
            ]
        except OSError:
            continue

        if not args:
            continue

        if "python" not in Path(
            args[0]
        ).name.lower():
            continue

        cmd = " ".join(args[1:])

        if any(w in cmd for w in (
            "run_mono5k_serial_v2.py",
            "train_qwen_verifier.py",
            "eval_valid_only.py",
            "valid_resumable_scoring_v1/worker.py",
            "speed_audit.py",
        )):
            raise RuntimeError(
                "FAIL_STOP: conflicting PID "
                + p.name + " " + cmd[:240]
            )


def finalized(job, expected_n=2209):
    final = WORK / job / "final"

    need(
        final.is_dir(),
        "Missing final: " + job
    )

    c = json.loads(
        (final / "commit.json").read_text()
    )

    need(
        c["job_id"] == job
        and int(c["n_queries"]) == expected_n,
        "Final commit identity: " + job
    )

    for field, filename in [
        ("scores_sha256", "scores.npy"),
        ("ranks_sha256", "ranks.npy"),
        ("result_sha256", "results.json"),
    ]:
        need(
            sha(final / filename) == c[field],
            "Final SHA256 mismatch: "
            + job + " " + filename
        )

    r = np.load(
        final / "ranks.npy",
        allow_pickle=False
    )
    s = np.load(
        final / "scores.npy",
        allow_pickle=False,
        mmap_mode="r"
    )

    need(
        r.shape == (expected_n,)
        and s.shape == (expected_n, 15)
        and np.isfinite(s).all()
        and np.issubdtype(r.dtype, np.integer)
        and (r >= 0).all(),
        "Final array invalid: " + job
    )

    d = json.loads(
        (final / "results.json").read_text()
    )["alpha=1"]["ruby"]

    mrr = 100.0 * (
        1.0 / (r.astype(np.float64) + 1)
    ).mean()

    need(
        abs(mrr - float(d["mrr"])) <= 0.00005001,
        "MRR replay mismatch: " + job
    )

    need(
        int(d["n_queries"]) == expected_n,
        "Query count mismatch: " + job
    )

    return {
        "job_id": job,
        "mrr": float(mrr),
        "result_sha256": c["result_sha256"],
        "rank_sha256": c["ranks_sha256"],
    }


def main():
    print(
        "R13_RUBY_THREE_SEED_QUEUE=v1",
        flush=True
    )

    need(
        PLAN.is_file()
        and WORKER.is_file()
        and BASELINE.is_file()
        and MONO13.is_file(),
        "Prerequisite files absent"
    )

    rows = json.loads(
        PLAN.read_text()
    )

    need(
        len(rows) == 54,
        "Frozen 54-job plan mismatch"
    )

    mapping = {
        r["job_id"]: r
        for r in rows
    }

    need(
        all(
            j in mapping
            and mapping[j]["language"] == "ruby"
            and int(
                mapping[j]["expected_queries"]
            ) == 2209
            for j in JOBS
        ),
        "Ruby four-cell plan mismatch"
    )

    need(
        int(
            json.loads(
                BASELINE.read_text()
            )["alpha=1"]["ruby"]["n_queries"]
        ) == 2209,
        "Frozen Multi-5k Ruby count"
    )

    need(
        json.loads(
            MONO13.read_text()
        )["job_id"] == "mono5k_seed13_ruby",
        "Frozen Mono-5k Ruby commit"
    )

    running_conflicts()

    worker_digest = sha(WORKER)
    plan_digest = sha(PLAN)

    manifest = {
        "jobs": JOBS,
        "worker_sha256": worker_digest,
        "plan_sha256": plan_digest,
        "split": "valid",
        "max_new_chunks": 0,
        "scoring_algorithm":
            "frozen_original_batch1",
        "test_accessed": False,
    }

    with (
        QUEUE / "queue_manifest.json"
    ).open("x") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

    print(
        "FROZEN_WORKER_SHA256=" + worker_digest,
        flush=True
    )
    print(
        "FROZEN_54_JOB_PLAN_SHA256=" + plan_digest,
        flush=True
    )
    print("QUEUE_JOBS=4", flush=True)

    env = os.environ.copy()
    env["R13_VALID_WORK_ROOT"] = str(WORK)

    # Check all four jobs without GPU inference.
    for i, j in enumerate(JOBS, 1):
        need(
            sha(WORKER) == worker_digest
            and sha(PLAN) == plan_digest,
            "Source changed before preflight"
        )

        logpath = (
            QUEUE / f"preflight_{i}_{j}.log"
        )

        with logpath.open("x") as f:
            p = subprocess.run(
                [
                    sys.executable,
                    "-u", "-B",
                    str(WORKER),
                    "--job-id", j,
                    "--verify-only",
                ],
                cwd=ROOT,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
            )

        need(
            p.returncode == 0,
            "Preflight failed: " + j
        )

        print(
            "JOB_PREFLIGHT_PASS=" + j,
            flush=True
        )

    print(
        "ALL_FOUR_JOB_PREFLIGHT=PASS",
        flush=True
    )

    # Execute the four jobs strictly serially.
    for i, j in enumerate(JOBS, 1):
        need(
            sha(WORKER) == worker_digest
            and sha(PLAN) == plan_digest,
            "Source changed during queue"
        )

        print(
            f"QUEUE_POSITION={i}/4 JOB={j} "
            f"START={time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
            flush=True
        )

        logpath = QUEUE / f"job_{i}_{j}.log"

        with logpath.open("x") as f:
            p = subprocess.run(
                [
                    sys.executable,
                    "-u", "-B",
                    str(WORKER),
                    "--job-id", j,
                    "--max-new-chunks", "0",
                ],
                cwd=ROOT,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
            )

        need(
            p.returncode == 0,
            "Worker failed: " + j
            + " log " + str(logpath)
        )

        result = finalized(j)

        print(
            "JOB_FINAL_PASS=" + j
            + " MRR="
            + format(result["mrr"], ".8f")
            + " RESULT_SHA256="
            + result["result_sha256"],
            flush=True
        )

    records = [
        finalized(j)
        for j in JOBS
    ]

    report = {
        "status": "PASS",
        "jobs": records,
        "worker_sha256": worker_digest,
        "plan_sha256": plan_digest,
        "test_accessed": False,
    }

    with (
        QUEUE / "final_report.json"
    ).open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

    print(
        "R13_RUBY_THREE_SEED_QUEUE=PASS",
        flush=True
    )
    print(
        "GPU_TRAINING_STARTED=NO TEST_ACCESSED=NO",
        flush=True
    )


if __name__ == "__main__":
    try:
        with (
            QUEUE / "scheduler.lock"
        ).open("a+b") as lock:
            fcntl.flock(
                lock,
                fcntl.LOCK_EX | fcntl.LOCK_NB
            )
            main()

    except Exception:
        traceback.print_exc()
        print(
            "R13_RUBY_THREE_SEED_QUEUE=FAIL_STOP",
            flush=True
        )
        sys.exit(2)
