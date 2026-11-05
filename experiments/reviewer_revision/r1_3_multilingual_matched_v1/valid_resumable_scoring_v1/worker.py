#!/usr/bin/env python3
"""R1-3 VALID-only resumable verifier scoring worker."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import runpy
import sys
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(
    "/root/autodl-tmp/lora-code-embeddings-main/text2code"
)
BASE = ROOT / (
    "results/reviewer_revision/r1_3_multilingual_matched_v1"
)
PLAN = Path(
    "/root/autodl-tmp/recal_r13_valid_plan_MiFZlVso"
)
EVAL = BASE / (
    "full_valid_multi5k_seed13_v1/eval_valid_only.py"
)
FROZEN = ROOT / (
    "results/reviewer_revision/recal_valid_fresh_20260911/"
    "recal_main_frozen_config.json"
)
CANON_DIR = ROOT / (
    "results/reviewer_revision/recal_valid_fresh_20260911/"
    "validation_grid/candidate_pools"
)
MODEL_BASE = Path(
    "/root/autodl-tmp/hf_models/Qwen2.5-Coder-7B-Instruct"
)
DATA = ROOT.parent / "data/CSN"
CONTROLLER = BASE / (
    "mono5k_training_controller_v1/"
    "run_mono5k_serial_v2.py"
)
OUTPUT = Path(os.environ["R13_VALID_WORK_ROOT"])

CHUNK_SIZE = 512

EXPECTED_EVAL_SHA = (
    "08692a72ab8b6e46a1ce99f83d762cea02a933db4bd7cb54840c5877d59163bb"
)
EXPECTED_FROZEN_SHA = (
    "70789e10518f32a97f5990b234833fb0443fbcbe426e8a8c3515ab93e6b11b0f"
)

LANG_COUNTS = {
    "ruby": 2209,
    "javascript": 8253,
    "php": 26015,
    "python": 23107,
    "java": 15328,
    "go": 14242,
}


def need(ok, msg):
    if not ok:
        raise RuntimeError("FAIL_STOP: " + msg)


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(8 * 1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def canonical_bytes(obj):
    return (
        json.dumps(
            obj,
            sort_keys=True,
            ensure_ascii=False,
            indent=2,
        ) + "\n"
    ).encode("utf-8")


def write_new_json(path, obj):
    with path.open("xb") as f:
        f.write(canonical_bytes(obj))
        f.flush()
        os.fsync(f.fileno())


def load_source():
    need(
        EVAL.is_file()
        and sha(EVAL) == EXPECTED_EVAL_SHA,
        "Frozen evaluator SHA256 mismatch",
    )

    spec = importlib.util.spec_from_file_location(
        "r13_frozen_valid_eval",
        EVAL,
    )

    need(
        spec is not None and spec.loader is not None,
        "Frozen evaluator import failed",
    )

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


def verify_block(
    folder, start, stop, fingerprint, mod, arrays
):
    need(
        folder.is_dir(),
        "Block directory missing: " + str(folder),
    )

    cp = folder / "commit.json"
    sp = folder / "scores.npy"
    rp = folder / "ranks.npy"

    need(
        cp.is_file() and sp.is_file() and rp.is_file(),
        "Incomplete block: " + str(folder),
    )

    meta = json.loads(cp.read_text())

    expected = {
        "fingerprint": fingerprint,
        "start": start,
        "stop": stop,
        "version": "r13_valid_block_v1",
    }

    for key, value in expected.items():
        need(
            meta.get(key) == value,
            "Block identity mismatch: " + key,
        )

    need(
        meta["scores_sha256"] == sha(sp),
        "Block score SHA256 mismatch",
    )
    need(
        meta["ranks_sha256"] == sha(rp),
        "Block rank SHA256 mismatch",
    )

    scores = np.load(sp, allow_pickle=False)
    ranks = np.load(rp, allow_pickle=False)

    need(
        scores.dtype == np.float32
        and scores.shape == (stop - start, 15)
        and np.isfinite(scores).all(),
        "Invalid checkpoint scores",
    )

    need(
        ranks.dtype == np.int32
        and ranks.shape == (stop - start,)
        and np.all(ranks >= 0),
        "Invalid checkpoint ranks",
    )

    replay = mod.rerank_with_verifier(
        arrays["indices"][start:stop],
        arrays["scores"][start:stop],
        scores,
        arrays["gt"][start:stop],
        arrays["base_ranks"][start:stop],
        15,
        1.0,
        "zscore",
    )

    need(
        np.array_equal(replay, ranks),
        "Frozen reranking replay mismatch",
    )

    return scores, ranks


def commit_block(
    work, start, stop, fingerprint, scores, ranks
):
    chunks = work / "chunks"
    staging = work / "staging"

    final = chunks / f"q{start:07d}_{stop:07d}"

    need(
        not final.exists(),
        "Committed block already exists",
    )

    stage = staging / (
        final.name + "_" + uuid.uuid4().hex + ".pending"
    )

    stage.mkdir(mode=0o700)

    with (stage / "scores.npy").open("xb") as f:
        np.save(
            f,
            scores.astype(np.float32, copy=False),
            allow_pickle=False,
        )
        f.flush()
        os.fsync(f.fileno())

    with (stage / "ranks.npy").open("xb") as f:
        np.save(
            f,
            ranks.astype(np.int32, copy=False),
            allow_pickle=False,
        )
        f.flush()
        os.fsync(f.fileno())

    meta = {
        "version": "r13_valid_block_v1",
        "fingerprint": fingerprint,
        "start": start,
        "stop": stop,
        "scores_sha256": sha(stage / "scores.npy"),
        "ranks_sha256": sha(stage / "ranks.npy"),
    }

    write_new_json(stage / "commit.json", meta)

    fd = os.open(stage, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

    os.rename(stage, final)

    fd = os.open(chunks, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

    return meta


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--job-id", required=True)
    parser.add_argument(
        "--max-new-chunks",
        type=int,
        default=1,
        help="0 runs all remaining chunks",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
    )

    args = parser.parse_args()

    need(
        args.max_new_chunks >= 0,
        "Invalid chunk limit",
    )
    need(
        args.job_id.replace("_", "").isalnum(),
        "Unsafe job identifier",
    )

    print("R13_VALID_RESUMABLE_WORKER=v1", flush=True)
    print("JOB_ID=", args.job_id, flush=True)
    print("VERIFY_ONLY=", args.verify_only, flush=True)

    need(
        PLAN.is_dir() and MODEL_BASE.is_dir(),
        "Missing plan or base model",
    )
    need(
        FROZEN.is_file()
        and sha(FROZEN) == EXPECTED_FROZEN_SHA,
        "Frozen configuration SHA mismatch",
    )

    frozen = json.loads(FROZEN.read_text())

    need(
        frozen.get("selected_on_split") == "valid",
        "Invalid frozen split",
    )

    fc = frozen["config"]

    need(
        int(fc["verify_top_k"]) == 15
        and float(fc["fusion_weight"]) == 1.0
        and fc["normalization"] == "zscore",
        "Frozen fusion settings mismatch",
    )

    protocol = json.loads(
        (PLAN / "protocol.json").read_text()
    )

    need(
        protocol["split"] == "VALID_ONLY"
        and protocol["verify_top_k"] == 15
        and protocol["candidate_top_k"] == 100
        and protocol["test_access_permitted"] is False,
        "Evaluation protocol mismatch",
    )

    jobs = json.loads(
        (PLAN / "evaluation_jobs.json").read_text()
    )

    candidates = [
        x for x in jobs
        if x["job_id"] == args.job_id
    ]

    need(
        len(jobs) == 54 and len(candidates) == 1,
        "Evaluation job not uniquely identified",
    )

    job = candidates[0]
    lang = job["language"]
    n = LANG_COUNTS[lang]
    seed = int(job["seed"])
    group = job["group"]

    need(
        group in ("mono5k", "multi5k", "multi30k")
        and seed in (13, 42, 87)
        and lang in LANG_COUNTS,
        "Invalid evaluation coordinates",
    )

    need(
        args.job_id == f"{group}_seed{seed}_{lang}",
        "Job ID mismatch",
    )
    need(
        int(job["expected_queries"]) == n,
        "Query count mismatch",
    )

    pools = json.loads(
        (PLAN / "valid_pools_manifest.json").read_text()
    )

    pool_path = Path(pools[lang]["path"])

    need(
        pool_path.is_file()
        and sha(pool_path) == job["pool_sha256"]
        and sha(pool_path) == pools[lang]["sha256"],
        "Frozen VALID pool SHA mismatch",
    )

    model = Path(job["model_path"])
    cfg = model / "verifier_config.json"
    adapter = model / "adapter_model.safetensors"
    adapter_cfg = model / "adapter_config.json"

    need(
        cfg.is_file()
        and sha(cfg) == job["model_config_sha256"],
        "Verifier model config SHA mismatch",
    )
    need(
        adapter.is_file() and adapter_cfg.is_file(),
        "Adapter files missing",
    )

    model_cfg = json.loads(cfg.read_text())

    need(
        int(model_cfg.get("seed", -1)) == seed,
        "Training seed mismatch",
    )

    expected_rows = (
        270000 if group == "multi30k" else 45000
    )

    need(
        int(
            model_cfg.get(
                "num_training_examples", -1
            )
        ) == expected_rows,
        "Training example count mismatch",
    )

    audit = runpy.run_path(
        str(CONTROLLER),
        run_name="r13_valid_frozen_model_verifier",
    )

    audit["preflight_parent"]()

    if group == "mono5k":
        run, checked_model = audit["dirs"](lang, seed)

        need(
            checked_model.resolve() == model.resolve(),
            "Training controller model mismatch",
        )

        audit["validate_completed"](
            lang, seed, run, model
        )

    valid_data = DATA / lang / "valid.jsonl"
    code_data = DATA / lang / "codebase.jsonl"
    canonical = (
        CANON_DIR / f"{lang}_valid_nomic_top100.npz"
    )

    need(
        valid_data.is_file()
        and code_data.is_file()
        and canonical.is_file(),
        "Missing VALID-only data",
    )

    data_sha = {
        "valid": sha(valid_data),
        "codebase": sha(code_data),
        "canonical": sha(canonical),
    }

    source = load_source()

    query_rows = source.read_jsonl(valid_data)
    code_rows = source.read_jsonl(code_data)

    need(
        len(query_rows) == n and len(code_rows) > 0,
        "Dataset row count mismatch",
    )

    queries = [
        source.get_query_text(row)
        for row in query_rows
    ]
    codes = [
        source.get_code_text(row)
        for row in code_rows
    ]

    with np.load(
        pool_path, allow_pickle=False
    ) as p, np.load(
        canonical, allow_pickle=False
    ) as c:

        indices = np.array(
            p["indices"], dtype=np.int64, copy=True
        )
        retrieval = np.array(
            p["scores"], dtype=np.float32, copy=True
        )
        base_ranks = np.array(
            p["base_ranks"], dtype=np.int64, copy=True
        )
        gt = np.array(
            c["gt_indices"], dtype=np.int64, copy=True
        )

        need(
            np.array_equal(
                indices, c["candidate_indices"]
            ),
            "Candidate index alignment mismatch",
        )
        need(
            np.array_equal(
                retrieval,
                c["candidate_scores"].astype(np.float32),
            ),
            "Candidate score alignment mismatch",
        )
        need(
            np.array_equal(
                base_ranks, c["base_ranks"]
            ),
            "Baseline rank alignment mismatch",
        )

    need(
        indices.shape == (n, 100)
        and retrieval.shape == (n, 100)
        and base_ranks.shape == (n,)
        and gt.shape == (n,),
        "VALID array shape mismatch",
    )

    need(
        np.all(indices >= 0)
        and np.all(indices < len(codes))
        and np.all(gt >= 0)
        and np.all(gt < len(codes)),
        "Code index outside valid range",
    )
    need(
        np.isfinite(retrieval).all(),
        "Nonfinite retrieval scores",
    )

    arrays = {
        "indices": indices,
        "scores": retrieval,
        "base_ranks": base_ranks,
        "gt": gt,
    }

    identity = {
        "version": "r13_valid_job_identity_v1",
        "job_id": args.job_id,
        "group": group,
        "seed": seed,
        "language": lang,
        "n_queries": n,
        "model_path": str(model),
        "model_config_sha256": sha(cfg),
        "adapter_sha256": sha(adapter),
        "adapter_config_sha256": sha(adapter_cfg),
        "base_model_path": str(MODEL_BASE),
        "evaluator_sha256": sha(EVAL),
        "frozen_config_sha256": sha(FROZEN),
        "pool_sha256": sha(pool_path),
        "data_sha256": data_sha,
        "chunk_size": CHUNK_SIZE,
        "candidate_top_k": 100,
        "verify_top_k": 15,
        "fusion_alpha": 1.0,
        "normalization": "zscore",
        "prompt": "yesno_v1",
        "batch_size": 1,
        "max_length": 1024,
        "max_query_chars": 512,
        "max_code_chars": 3000,
        "split": "valid",
    }

    fingerprint = hashlib.sha256(
        canonical_bytes(identity)
    ).hexdigest()

    work = OUTPUT / args.job_id
    work.mkdir(exist_ok=True)

    for dirname in ("chunks", "staging"):
        (work / dirname).mkdir(exist_ok=True)

    global_lock = (
        OUTPUT / "gpu_evaluation.global.lock"
    )

    with global_lock.open("a+b") as lock:
        fcntl.flock(
            lock,
            fcntl.LOCK_EX | fcntl.LOCK_NB,
        )

        identity_path = work / "identity.json"

        if identity_path.is_file():
            need(
                identity_path.read_bytes()
                == canonical_bytes(identity),
                "Stored job identity changed",
            )
        else:
            write_new_json(
                identity_path, identity
            )

        print(
            "SOURCE_AND_INPUT_IDENTITY=PASS",
            flush=True,
        )
        print(
            "JOB_FINGERPRINT=",
            fingerprint,
            flush=True,
        )

        ranges = [
            (x, min(x + CHUNK_SIZE, n))
            for x in range(0, n, CHUNK_SIZE)
        ]

        missing = []
        verified = 0

        for start, stop in ranges:
            dest = (
                work / "chunks" /
                f"q{start:07d}_{stop:07d}"
            )

            if dest.exists():
                verify_block(
                    dest, start, stop,
                    fingerprint, source, arrays
                )
                verified += 1

                print(
                    f"REUSED_VERIFIED_BLOCK={start}:{stop}",
                    flush=True,
                )
            else:
                missing.append((start, stop))

        print(
            "COMMITTED_VERIFIED=",
            verified,
            "TOTAL_CHUNKS=",
            len(ranges),
            flush=True,
        )

        if not args.verify_only and missing:
            audit["gpu_gate"]()
            audit["disk_gate"]()

            import torch

            device = torch.device("cuda")

            selected = (
                missing if args.max_new_chunks == 0
                else missing[:args.max_new_chunks]
            )

            print(
                "GPU_MODEL_LOAD_ONCE_BEGIN",
                flush=True,
            )

            verifier, tokenizer = source.load_verifier(
                str(model),
                str(MODEL_BASE),
                "float16",
                device,
                False,
                "cuda",
            )

            yes_id = source.single_token_id(
                tokenizer,
                [" Yes", "Yes", " yes", "yes"],
                "yes",
            )
            no_id = source.single_token_id(
                tokenizer,
                [" No", "No", " no", "no"],
                "no",
            )

            print(
                "GPU_MODEL_LOAD_ONCE_PASS",
                flush=True,
            )

            for start, stop in selected:
                prompts = [
                    source.build_yesno_prompt(
                        queries[i],
                        codes[int(code_idx)],
                        512,
                        3000,
                    )
                    for i in range(start, stop)
                    for code_idx in indices[i, :15]
                ]

                scores = source.score_yesno_prompts(
                    verifier,
                    tokenizer,
                    prompts,
                    yes_id,
                    no_id,
                    device,
                    1,
                    1024,
                ).reshape(stop - start, 15)

                need(
                    np.isfinite(scores).all(),
                    "Nonfinite verifier logits",
                )

                ranks = source.rerank_with_verifier(
                    indices[start:stop],
                    retrieval[start:stop],
                    scores,
                    gt[start:stop],
                    base_ranks[start:stop],
                    15,
                    1.0,
                    "zscore",
                )

                need(
                    ranks.shape == (stop - start,)
                    and np.all(ranks >= 0),
                    "Invalid reranked results",
                )

                meta = commit_block(
                    work,
                    start,
                    stop,
                    fingerprint,
                    scores,
                    ranks,
                )

                verify_block(
                    work / "chunks" /
                    f"q{start:07d}_{stop:07d}",
                    start,
                    stop,
                    fingerprint,
                    source,
                    arrays,
                )

                print(
                    f"BLOCK_COMMIT_PASS={start}:{stop} "
                    f"SHA256={meta['scores_sha256']}",
                    flush=True,
                )

            del verifier, tokenizer
            torch.cuda.empty_cache()

        done = 0
        all_scores = []
        all_ranks = []

        for start, stop in ranges:
            dest = (
                work / "chunks" /
                f"q{start:07d}_{stop:07d}"
            )

            if dest.exists():
                scores, ranks = verify_block(
                    dest,
                    start,
                    stop,
                    fingerprint,
                    source,
                    arrays,
                )

                done += 1
                all_scores.append(scores)
                all_ranks.append(ranks)

        print(
            "R13_JOB_VERIFIED_CHUNKS=",
            done,
            "/",
            len(ranges),
            flush=True,
        )

        if done < len(ranges):
            print(
                "R13_VALID_PARTIAL_CHECKPOINT=PASS",
                flush=True,
            )
            print("TEST_ACCESSED=NO", flush=True)
            return

        score_all = np.concatenate(
            all_scores, axis=0
        )
        rank_all = np.concatenate(
            all_ranks, axis=0
        )

        need(
            score_all.shape == (n, 15)
            and rank_all.shape == (n,),
            "Final merged shape mismatch",
        )

        metrics = source.compute_metrics_from_ranks(
            rank_all
        )

        final = work / "final"

        if final.exists():
            need(
                final.is_dir(),
                "Final output type mismatch",
            )

            report = json.loads(
                (final / "commit.json").read_text()
            )

            need(
                report["fingerprint"] == fingerprint
                and report["n_queries"] == n,
                "Final fingerprint mismatch",
            )

            for key, filename in [
                ("scores_sha256", "scores.npy"),
                ("ranks_sha256", "ranks.npy"),
                ("result_sha256", "results.json"),
            ]:
                need(
                    report[key] == sha(final / filename),
                    "Final SHA256 mismatch: " + filename,
                )

            need(
                np.array_equal(
                    np.load(
                        final / "scores.npy",
                        allow_pickle=False,
                    ),
                    score_all,
                ),
                "Final score mismatch",
            )

            need(
                np.array_equal(
                    np.load(
                        final / "ranks.npy",
                        allow_pickle=False,
                    ),
                    rank_all,
                ),
                "Final rank mismatch",
            )

            result = json.loads(
                (final / "results.json").read_text()
            )

            need(
                result["alpha=1"][lang]["mrr"]
                == metrics["mrr"],
                "Final MRR mismatch",
            )

            print(
                "R13_VALID_JOB_FINAL=PASS_REUSED",
                flush=True,
            )
            print("TEST_ACCESSED=NO", flush=True)
            return

        if args.verify_only:
            print(
                "VERIFY_ONLY_CHECKPOINTS_COMPLETE=PASS",
                flush=True,
            )
            print(
                "NO_FINAL_OUTPUT_WRITTEN=TRUE",
                flush=True,
            )
            return

        stage = (
            work / "staging" /
            ("final_" + uuid.uuid4().hex + ".pending")
        )
        stage.mkdir(mode=0o700)

        with (stage / "scores.npy").open("xb") as f:
            np.save(
                f,
                score_all.astype(np.float32),
                allow_pickle=False,
            )
            f.flush()
            os.fsync(f.fileno())

        with (stage / "ranks.npy").open("xb") as f:
            np.save(
                f,
                rank_all.astype(np.int32),
                allow_pickle=False,
            )
            f.flush()
            os.fsync(f.fileno())

        result = {
            "alpha=1": {
                lang: {
                    **metrics,
                    "fusion_alpha": 1.0,
                    "candidate_top_k": 100,
                    "verify_top_k": 15,
                    "candidate_pool": "r13_validation_v2",
                }
            }
        }

        write_new_json(
            stage / "results.json", result
        )

        report = {
            "version": "r13_valid_final_v1",
            "fingerprint": fingerprint,
            "job_id": args.job_id,
            "scores_sha256": sha(stage / "scores.npy"),
            "ranks_sha256": sha(stage / "ranks.npy"),
            "result_sha256": sha(stage / "results.json"),
            "n_queries": n,
            "mrr_percent": metrics["mrr"],
            "test_accessed": False,
        }

        write_new_json(
            stage / "commit.json", report
        )

        fd = os.open(stage, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

        os.rename(stage, final)

        fd = os.open(work, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

        print(
            "R13_VALID_JOB_FINAL=PASS",
            flush=True,
        )
        print(
            "MRR_PERCENT=",
            metrics["mrr"],
            flush=True,
        )
        print("TEST_ACCESSED=NO", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        print(
            "R13_VALID_RESUMABLE_WORKER=FAIL_STOP",
            flush=True,
        )
        sys.exit(2)
