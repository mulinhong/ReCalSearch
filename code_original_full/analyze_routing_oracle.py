import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from evaluate import calculate_ndcg


def load_ranks(path: str) -> np.ndarray:
    ranks = np.load(path)
    if ranks.ndim != 1:
        raise ValueError(f"Expected 1D rank array in {path}, got shape={ranks.shape}")
    return ranks.astype(np.int64)


def metrics_from_ranks(ranks: np.ndarray) -> Dict[str, float]:
    ndcgs = []
    for rank in ranks:
        rel = [0] * 10
        if rank < 10:
            rel[int(rank)] = 1
        ndcgs.append(calculate_ndcg(rel, k=10))
    return {
        "r1": round(100.0 * float(np.mean(ranks < 1)), 2),
        "r5": round(100.0 * float(np.mean(ranks < 5)), 2),
        "r10": round(100.0 * float(np.mean(ranks < 10)), 2),
        "mrr": round(100.0 * float(np.mean(1.0 / (ranks + 1))), 2),
        "ndcg@10": round(100.0 * float(np.mean(ndcgs)), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute oracle routing upper bound from per-query rank files.")
    parser.add_argument(
        "--rank-file",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        required=True,
        help="Route name and .npy rank file. Repeat for each route.",
    )
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    route_names: List[str] = []
    rank_arrays: List[np.ndarray] = []
    expected_len = None
    for name, path in args.rank_file:
        ranks = load_ranks(path)
        if expected_len is None:
            expected_len = len(ranks)
        elif len(ranks) != expected_len:
            raise ValueError(f"{name} has {len(ranks)} ranks, expected {expected_len}")
        route_names.append(name)
        rank_arrays.append(ranks)

    rank_matrix = np.stack(rank_arrays, axis=1)
    best_route_ids = np.argmin(rank_matrix, axis=1)
    oracle_ranks = rank_matrix[np.arange(rank_matrix.shape[0]), best_route_ids]

    results = {
        "routes": {
            name: metrics_from_ranks(ranks)
            for name, ranks in zip(route_names, rank_arrays)
        },
        "oracle": metrics_from_ranks(oracle_ranks),
        "oracle_route_share": {
            name: round(100.0 * float(np.mean(best_route_ids == i)), 2)
            for i, name in enumerate(route_names)
        },
        "num_queries": int(rank_matrix.shape[0]),
    }

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved oracle analysis to: {output_path}")


if __name__ == "__main__":
    main()
