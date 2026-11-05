#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def describe(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "p10": float(np.quantile(x, 0.10)),
        "p25": float(np.quantile(x, 0.25)),
        "p50": float(np.quantile(x, 0.50)),
        "p75": float(np.quantile(x, 0.75)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(x.max()),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--reference-nomic-groups", required=True)
    ap.add_argument("--source-groups", required=True)
    ap.add_argument("--output-file", required=True)
    ap.add_argument("--report-file", required=True)
    ap.add_argument("--num-negatives", type=int, default=7)
    ap.add_argument("--mae-max", type=float, default=0.02)
    ap.add_argument("--mean-shift-max", type=float, default=0.02)

    args = ap.parse_args()

    refs = load_jsonl(args.reference_nomic_groups)
    srcs = load_jsonl(args.source_groups)

    ref_by_qid = {x["query_id"]: x for x in refs}
    src_by_qid = {x["query_id"]: x for x in srcs}

    if set(ref_by_qid) != set(src_by_qid):
        raise RuntimeError("Reference/source query_id sets differ")

    qids = [x["query_id"] for x in refs]

    if len(qids) != len(set(qids)):
        raise RuntimeError("Duplicate query IDs")

    target_all = []
    matched_all = []
    error_all = []
    query_mae_all = []
    source_rank_all = []

    source_overlap_before = 0
    excluded_reference_pairs = 0

    below_target_max = 0
    below_target_min = 0
    mae_gt_005 = 0
    mae_gt_010 = 0

    eligible_counts = []

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fout:

        for qid in qids:

            ref = ref_by_qid[qid]
            src = src_by_qid[qid]

            if ref["query"] != src["query"]:
                raise RuntimeError(f"Query mismatch: {qid}")

            ref_pos = int(ref["positive"]["code_index"])
            src_pos = int(src["positive"]["code_index"])

            if ref_pos != src_pos:
                raise RuntimeError(f"Positive mismatch: {qid}")

            target = np.asarray(
                ref["negative_retrieval_scores"][:args.num_negatives],
                dtype=np.float64,
            )

            ref_ids = set(
                int(x)
                for x in ref["negative_code_indices"][:args.num_negatives]
            )

            src_ids = np.asarray(
                src["negative_code_indices"],
                dtype=np.int64,
            )

            src_dep = np.asarray(
                src["negative_deployed_nomic_scores"],
                dtype=np.float64,
            )

            src_rank = np.asarray(
                src["negative_retrieval_ranks"],
                dtype=np.int64,
            )

            src_score = np.asarray(
                src["negative_retrieval_scores"],
                dtype=np.float64,
            )

            if not (
                len(src_ids)
                == len(src_dep)
                == len(src_rank)
                == len(src_score)
            ):
                raise RuntimeError(f"Source field length mismatch: {qid}")

            overlap_mask = np.asarray(
                [int(x) in ref_ids for x in src_ids],
                dtype=bool,
            )

            source_overlap_before += int(overlap_mask.sum())
            excluded_reference_pairs += int(overlap_mask.sum())

            eligible_pos = np.flatnonzero(~overlap_mask)
            eligible_counts.append(len(eligible_pos))

            if len(eligible_pos) < args.num_negatives:
                raise RuntimeError(
                    f"{qid}: only {len(eligible_pos)} "
                    "source-disjoint candidates"
                )

            eligible_dep = src_dep[eligible_pos]

            cost = np.abs(
                target[:, None]
                - eligible_dep[None, :]
            )

            rows, local_cols = linear_sum_assignment(cost)

            order = np.argsort(rows)
            rows = rows[order]
            local_cols = local_cols[order]

            selected_pos = eligible_pos[local_cols]

            selected_ids = src_ids[selected_pos]
            selected_dep = src_dep[selected_pos]
            selected_rank = src_rank[selected_pos]
            selected_score = src_score[selected_pos]

            if any(int(x) in ref_ids for x in selected_ids):
                raise RuntimeError(
                    f"Source-disjoint violation: {qid}"
                )

            err = np.abs(target - selected_dep)
            qmae = float(err.mean())

            target_all.extend(target.tolist())
            matched_all.extend(selected_dep.tolist())
            error_all.extend(err.tolist())
            query_mae_all.append(qmae)
            source_rank_all.extend(selected_rank.tolist())

            if eligible_dep.max() < target.max():
                below_target_max += 1

            if eligible_dep.max() < target.min():
                below_target_min += 1

            if qmae > 0.05:
                mae_gt_005 += 1

            if qmae > 0.10:
                mae_gt_010 += 1

            out = {
                "format":
                    "grouped_indices_v4_"
                    "dense_source_disjoint_"
                    "nomic_difficulty_matched",
                "mining":
                    "unixcoder_base_"
                    "source_disjoint_"
                    "difficulty_matched",
                "mining_retriever":
                    "microsoft/unixcoder-base",
                "difficulty_reference":
                    "deployed_nomic_similarity",
                "matching":
                    "hungarian_min_absolute_error",
                "source_disjoint_from_reference_top7":
                    True,
                "query_id": qid,
                "query": ref["query"],
                "language": ref["language"],
                "positive": {
                    "code_index": ref_pos,
                    "code": None,
                    "retrieval_rank": None,
                    "retrieval_score": None,
                },
                "negative_code_indices":
                    selected_ids.astype(int).tolist(),
                "negative_retrieval_ranks":
                    selected_rank.astype(int).tolist(),
                "negative_retrieval_scores":
                    selected_score.astype(float).tolist(),
                "negative_deployed_nomic_scores":
                    selected_dep.astype(float).tolist(),
                "matched_reference_nomic_scores":
                    target.astype(float).tolist(),
            }

            fout.write(
                json.dumps(out, ensure_ascii=False) + "\n"
            )

    target_all = np.asarray(target_all, dtype=np.float64)
    matched_all = np.asarray(matched_all, dtype=np.float64)
    error_all = np.asarray(error_all, dtype=np.float64)
    query_mae_all = np.asarray(query_mae_all, dtype=np.float64)
    source_rank_all = np.asarray(source_rank_all, dtype=np.float64)
    eligible_counts = np.asarray(eligible_counts, dtype=np.int64)

    mae = float(error_all.mean())
    shift = float(matched_all.mean() - target_all.mean())

    accepted = (
        mae <= args.mae_max
        and abs(shift) <= args.mean_shift_max
    )

    report = {
        "protocol":
            "reviewer_r1_1_dense_source_disjoint_match_v1",
        "queries": len(qids),
        "pairs": int(len(error_all)),
        "source_disjoint_from_reference_top7": True,
        "source_overlap_before_exclusion_pairs":
            int(source_overlap_before),
        "excluded_reference_pairs":
            int(excluded_reference_pairs),
        "eligible_candidates_per_query":
            describe(eligible_counts),
        "target_nomic_similarity":
            describe(target_all),
        "matched_nomic_similarity":
            describe(matched_all),
        "absolute_similarity_error":
            describe(error_all),
        "per_query_mae":
            describe(query_mae_all),
        "mean_similarity_shift":
            shift,
        "queries_candidate_max_below_target_max":
            below_target_max,
        "queries_candidate_max_below_target_min":
            below_target_min,
        "queries_mae_gt_0_05":
            mae_gt_005,
        "queries_mae_gt_0_10":
            mae_gt_010,
        "matched_source_ranks":
            describe(source_rank_all),
        "acceptance_thresholds": {
            "mae_max": args.mae_max,
            "absolute_mean_shift_max": args.mean_shift_max,
        },
        "difficulty_match_accepted":
            accepted,
    }

    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(json.dumps(report, indent=2))

    print(
        "MATCHED_GROUP_SHA256 =",
        sha256_file(out_path),
    )

    print(
        "REPORT_SHA256 =",
        sha256_file(report_path),
    )

    print(
        "SOURCE_DISJOINT_MATCH_AUDIT=PASSED"
    )

    print(
        "DIFFICULTY_MATCH_ACCEPTED =",
        accepted,
    )


if __name__ == "__main__":
    main()
