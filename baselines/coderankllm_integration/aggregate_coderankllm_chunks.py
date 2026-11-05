import argparse
import json
from pathlib import Path

import numpy as np


def metrics(ranks):
    ranks = np.asarray(ranks, dtype=np.int64)

    return {
        "r1": float(np.mean(ranks < 1)),
        "r5": float(np.mean(ranks < 5)),
        "r10": float(np.mean(ranks < 10)),
        "mrr": float(np.mean(1.0 / (ranks + 1))),
        "ndcg@10": float(
            np.mean(
                np.where(
                    ranks < 10,
                    1.0 / np.log2(ranks + 2),
                    0.0,
                )
            )
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True)
    ap.add_argument(
        "--run-root",
        default="/root/autodl-tmp/cornstack-official/"
                "runs_coderankllm_full",
    )
    ap.add_argument(
        "--pool-root",
        default="/root/autodl-tmp/"
                "lora-code-embeddings-main/text2code/"
                "results/reviewer_revision/pools/nomic_test",
    )
    args = ap.parse_args()

    lang_root = Path(args.run_root) / args.language

    pool_file = (
        Path(args.pool_root) /
        f"{args.language}_test_nomic_top100.npz"
    )

    pool = np.load(pool_file, mmap_mode="r")

    total_queries = int(pool["base_ranks"].shape[0])

    chunks = []

    for cdir in sorted(lang_root.glob("chunk_*")):
        done = cdir / "DONE"
        summary = cdir / "fullmetrics.json"
        ranks = cdir / "fullranks.npy"

        if not done.exists():
            continue

        if not summary.exists() or not ranks.exists():
            raise RuntimeError(
                f"Incomplete DONE chunk: {cdir}"
            )

        meta = json.load(summary.open())

        chunks.append({
            "dir": cdir,
            "offset": int(meta["source_offset"]),
            "end": int(meta["source_end"]),
            "queries": int(meta["queries"]),
            "summary": meta,
            "ranks_file": ranks,
        })

    if not chunks:
        raise RuntimeError(
            f"No completed chunks for {args.language}"
        )

    chunks.sort(key=lambda x: x["offset"])

    expected_offset = 0

    all_ranks = []
    total_wall = 0
    total_inside = 0
    total_outside = 0

    weighted_gpu_util_num = 0.0
    weighted_gpu_util_den = 0

    peak_vram = 0.0

    for c in chunks:

        if c["offset"] != expected_offset:
            raise RuntimeError(
                f"Chunk gap/overlap: expected offset "
                f"{expected_offset}, got {c['offset']}"
            )

        if c["end"] - c["offset"] != c["queries"]:
            raise RuntimeError(
                f"Bad chunk range: {c['dir']}"
            )

        ranks = np.load(c["ranks_file"])

        if len(ranks) != c["queries"]:
            raise RuntimeError(
                f"Rank count mismatch: {c['dir']}"
            )

        all_ranks.append(
            np.asarray(ranks, dtype=np.int32)
        )

        m = c["summary"]

        total_wall += int(m["wall_seconds"])
        total_inside += int(m["inside_top100"])
        total_outside += int(
            m["outside_top100_preserve_dense_rank"]
        )

        q = int(m["queries"])

        if m.get("mean_gpu_util_percent") is not None:
            weighted_gpu_util_num += (
                float(m["mean_gpu_util_percent"]) * q
            )
            weighted_gpu_util_den += q

        if m.get("peak_gpu_memory_MiB") is not None:
            peak_vram = max(
                peak_vram,
                float(m["peak_gpu_memory_MiB"]),
            )

        expected_offset = c["end"]

    combined = np.concatenate(all_ranks)

    base_ranks = np.asarray(
        pool["base_ranks"][:expected_offset],
        dtype=np.int64,
    )

    if len(combined) != expected_offset:
        raise RuntimeError(
            "Combined rank length mismatch"
        )

    base_metrics = metrics(base_ranks)
    coderank_metrics = metrics(combined)

    summary = {
        "protocol": "recal_full_rank_v1_chunk_aggregate",
        "language": args.language,

        "completed_queries": int(expected_offset),
        "total_queries": total_queries,

        "complete_language": (
            expected_offset == total_queries
        ),

        "chunks_completed": len(chunks),

        "inside_top100": total_inside,
        "outside_top100_preserve_dense_rank":
            total_outside,

        "nomic_base": base_metrics,
        "coderankllm": coderank_metrics,

        "mrr_absolute_gain": (
            coderank_metrics["mrr"] -
            base_metrics["mrr"]
        ),

        "mrr_gain_points": (
            coderank_metrics["mrr"] -
            base_metrics["mrr"]
        ) * 100.0,

        "total_wall_seconds": total_wall,

        "seconds_per_query": (
            total_wall / expected_offset
            if expected_offset else None
        ),

        "estimated_gpu_hours_so_far":
            total_wall / 3600.0,

        "peak_gpu_memory_MiB": peak_vram,

        "query_weighted_mean_gpu_util_percent": (
            weighted_gpu_util_num /
            weighted_gpu_util_den
            if weighted_gpu_util_den else None
        ),
    }

    np.save(
        lang_root / f"{args.language}_coderankllm_fullranks.npy",
        combined,
    )

    out = (
        lang_root /
        f"{args.language}_coderankllm_aggregate.json"
    )

    with out.open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

    if summary["complete_language"]:
        print()
        print("FULL LANGUAGE VALIDATION PASSED")
    else:
        print()
        print(
            "PARTIAL LANGUAGE VALIDATION PASSED "
            f"({expected_offset}/{total_queries})"
        )


if __name__ == "__main__":
    main()
