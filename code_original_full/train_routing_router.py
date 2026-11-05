import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import DATA_ROOT
from evaluate import calculate_ndcg


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def cache_paths(
    cache_dir: str,
    language: str,
    split: str,
    model_name: str,
    max_seq_length: int,
    query_prefix: str,
    code_prefix: str,
) -> Dict[str, Path]:
    root = Path(cache_dir)
    model_key = sanitize_name(model_name)
    cache_key = json.dumps(
        {
            "model_name": model_name,
            "backend": "nomic_transformers",
            "query_prefix": query_prefix,
            "code_prefix": code_prefix,
            "max_seq_length": max_seq_length,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:10]
    prefix = root / f"{language}_{model_key}_msl{max_seq_length}_{digest}"
    return {
        "query": root / f"{language}_{split}_{model_key}_msl{max_seq_length}_{digest}.query.npy",
        "code": prefix.with_suffix(".code.npy"),
    }


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_query_text(example: Dict) -> str:
    for key in ["docstring", "doc", "query", "nl", "text"]:
        if key in example and example[key]:
            value = example[key]
            if isinstance(value, list):
                return " ".join(map(str, value))
            return str(value)
    if "docstring_tokens" in example and example["docstring_tokens"]:
        value = example["docstring_tokens"]
        if isinstance(value, list):
            return " ".join(map(str, value))
        return str(value)
    raise KeyError(f"Could not find query text field in example keys: {list(example.keys())}")


def load_ranks(path: str) -> np.ndarray:
    ranks = np.load(path)
    if ranks.ndim != 1:
        raise ValueError(f"Expected 1D ranks in {path}, got shape={ranks.shape}")
    return ranks.astype(np.int64)


def load_route_feature_file(path: str) -> Tuple[np.ndarray, List[str]]:
    data = np.load(path, allow_pickle=True)
    if "features" not in data or "feature_names" not in data:
        raise ValueError(f"Route feature file must contain features and feature_names: {path}")
    features = np.asarray(data["features"], dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features in {path}, got shape={features.shape}")
    feature_names = [str(x) for x in data["feature_names"].tolist()]
    if len(feature_names) != features.shape[1]:
        raise ValueError(
            f"Feature-name count mismatch in {path}: "
            f"{len(feature_names)} names for {features.shape[1]} columns"
        )
    return features, feature_names


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


@torch.no_grad()
def compute_nomic_top_scores(
    query_embeds,
    code_embeds,
    device: torch.device,
    top_k: int,
    query_chunk_size: int,
    code_chunk_size: int,
) -> np.ndarray:
    n_queries = query_embeds.shape[0]
    top_scores_all = []

    for q_start in tqdm(range(0, n_queries, query_chunk_size), desc=f"Nomic top{top_k} scores"):
        q_end = min(q_start + query_chunk_size, n_queries)
        q_chunk = torch.from_numpy(np.asarray(query_embeds[q_start:q_end])).to(device).float()
        q_chunk = F.normalize(q_chunk, dim=1)
        q_len = q_end - q_start

        best_scores = torch.full((q_len, top_k), -1e9, dtype=torch.float32, device=device)
        for c_start in range(0, code_embeds.shape[0], code_chunk_size):
            c_end = min(c_start + code_chunk_size, code_embeds.shape[0])
            c_chunk = torch.from_numpy(np.asarray(code_embeds[c_start:c_end])).to(device).float()
            c_chunk = F.normalize(c_chunk, dim=1)
            scores = q_chunk @ c_chunk.t()
            chunk_top = torch.topk(scores, k=min(top_k, scores.size(1)), dim=1).values
            merged = torch.cat([best_scores, chunk_top], dim=1)
            best_scores = torch.topk(merged, k=top_k, dim=1).values

            del c_chunk, scores, chunk_top, merged
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        top_scores_all.append(best_scores.cpu().numpy())
        del q_chunk, best_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.concatenate(top_scores_all, axis=0)


def softmax_entropy(scores: np.ndarray) -> np.ndarray:
    centered = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(centered)
    probs = exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1)
    return entropy


def build_features(top_scores: np.ndarray, eval_rows: List[Dict]) -> Tuple[np.ndarray, List[str]]:
    if top_scores.shape[1] < 10:
        raise ValueError("--feature-top-k must be at least 10")

    queries = [get_query_text(row) for row in eval_rows]
    char_len = np.array([len(q) for q in queries], dtype=np.float32)
    word_len = np.array([len(q.split()) for q in queries], dtype=np.float32)

    s = top_scores.astype(np.float32)
    features = [
        s[:, 0],
        s[:, 1],
        s[:, 4],
        s[:, 9],
        s[:, 0] - s[:, 1],
        s[:, 0] - s[:, 4],
        s[:, 0] - s[:, 9],
        s[:, :5].mean(axis=1),
        s[:, :10].mean(axis=1),
        s[:, :5].std(axis=1),
        s[:, :10].std(axis=1),
        softmax_entropy(s[:, :5]),
        softmax_entropy(s[:, :10]),
        char_len,
        word_len,
        np.log1p(char_len),
        np.log1p(word_len),
    ]
    names = [
        "nomic_top1",
        "nomic_top2",
        "nomic_top5",
        "nomic_top10",
        "margin_1_2",
        "margin_1_5",
        "margin_1_10",
        "top5_mean",
        "top10_mean",
        "top5_std",
        "top10_std",
        "top5_entropy",
        "top10_entropy",
        "query_chars",
        "query_words",
        "log_query_chars",
        "log_query_words",
    ]
    return np.stack(features, axis=1).astype(np.float32), names


def standardize_train_val(x_train: np.ndarray, x_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x_train - mean) / std, (x_val - mean) / std, mean.squeeze(0), std.squeeze(0)


def standardize_with_stats(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean.reshape(1, -1)) / std.reshape(1, -1)


def fit_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    lr: float,
    epochs: int,
    l2: float,
    seed: int,
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.01, size=x.shape[1]).astype(np.float32)
    b = 0.0

    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    pos_weight = neg / max(pos, 1.0)
    weights = np.where(y == 1, pos_weight, 1.0).astype(np.float32)
    weights = weights / np.mean(weights)

    for _ in range(epochs):
        logits = x @ w + b
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
        grad = (probs - y) * weights
        grad_w = (x.T @ grad) / x.shape[0] + l2 * w
        grad_b = float(np.mean(grad))
        w -= lr * grad_w
        b -= lr * grad_b

    return w, b


def predict_probs(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    logits = x @ w + b
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))


def route_by_probs(
    probs: np.ndarray,
    direct_ranks: np.ndarray,
    fusion_ranks: np.ndarray,
    threshold: float,
) -> np.ndarray:
    use_fusion = probs >= threshold
    return np.where(use_fusion, fusion_ranks, direct_ranks)


def select_threshold(
    probs: np.ndarray,
    direct_ranks: np.ndarray,
    fusion_ranks: np.ndarray,
    num_thresholds: int,
    cost_lambda: float,
) -> float:
    thresholds = np.quantile(probs, np.linspace(0.0, 1.0, num_thresholds))
    best_threshold = 0.5
    best_score = -1e9
    for threshold in thresholds:
        use_fusion = probs >= threshold
        ranks = np.where(use_fusion, fusion_ranks, direct_ranks)
        mrr = 100.0 * float(np.mean(1.0 / (ranks + 1)))
        score = mrr - cost_lambda * 100.0 * float(use_fusion.mean())
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate a lightweight adaptive router with cross-validation.")
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--split", type=str, default="valid", choices=["valid", "test"])
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--cache-dir", type=str, default="")
    parser.add_argument("--model-path", type=str, default="/root/autodl-tmp/hf_models/nomic-embed-code")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--query-prefix", type=str, default="Represent this query for searching relevant code: ")
    parser.add_argument("--code-prefix", type=str, default="")
    parser.add_argument("--direct-ranks", type=str, required=True)
    parser.add_argument("--fusion-ranks", type=str, required=True)
    parser.add_argument(
        "--feature-file",
        type=str,
        default="",
        help="Optional .npz route feature file exported by eval_reranker.py.",
    )
    parser.add_argument(
        "--include-cache-features",
        action="store_true",
        help="Append Nomic top-k cache features even when --feature-file is provided.",
    )
    parser.add_argument("--feature-top-k", type=int, default=10)
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument("--code-chunk-size", type=int, default=4096)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--num-thresholds", type=int, default=101)
    parser.add_argument(
        "--cost-lambda",
        type=float,
        default=0.0,
        help="Penalty per 1 percent routed to fusion when selecting thresholds.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--apply-split", type=str, default="", choices=["", "valid", "test"])
    parser.add_argument("--apply-feature-file", type=str, default="")
    parser.add_argument("--apply-direct-ranks", type=str, default="")
    parser.add_argument("--apply-fusion-ranks", type=str, default="")
    parser.add_argument("--apply-output-ranks", type=str, default="")
    args = parser.parse_args()

    eval_rows = read_jsonl(Path(args.data_root) / args.language / f"{args.split}.jsonl")
    direct_ranks = load_ranks(args.direct_ranks)
    fusion_ranks = load_ranks(args.fusion_ranks)
    if len(direct_ranks) != len(fusion_ranks) or len(direct_ranks) != len(eval_rows):
        raise ValueError("Rank files and eval split have different lengths.")

    feature_parts = []
    feature_names = []
    if args.feature_file:
        route_features, route_feature_names = load_route_feature_file(args.feature_file)
        if route_features.shape[0] != len(eval_rows):
            raise ValueError(
                f"Route feature file has {route_features.shape[0]} rows, expected {len(eval_rows)}"
            )
        feature_parts.append(route_features)
        feature_names.extend([f"route:{name}" for name in route_feature_names])

    if not args.feature_file or args.include_cache_features:
        if not args.cache_dir:
            raise ValueError("--cache-dir is required when cache features are enabled.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for router cache feature extraction.")
        device = torch.device("cuda")

        caches = cache_paths(
            cache_dir=args.cache_dir,
            language=args.language,
            split=args.split,
            model_name=args.model_path,
            max_seq_length=args.max_length,
            query_prefix=args.query_prefix,
            code_prefix=args.code_prefix,
        )
        query_embeds = np.load(caches["query"], mmap_mode="r")
        code_embeds = np.load(caches["code"], mmap_mode="r")
        if query_embeds.shape[0] != len(eval_rows):
            raise ValueError(f"Query cache has {query_embeds.shape[0]} rows, expected {len(eval_rows)}")

        top_scores = compute_nomic_top_scores(
            query_embeds=query_embeds,
            code_embeds=code_embeds,
            device=device,
            top_k=args.feature_top_k,
            query_chunk_size=args.query_chunk_size,
            code_chunk_size=args.code_chunk_size,
        )
        cache_x, cache_feature_names = build_features(top_scores, eval_rows)
        feature_parts.append(cache_x)
        feature_names.extend([f"cache:{name}" for name in cache_feature_names])

    if not feature_parts:
        raise ValueError("No router features were loaded or computed.")
    x = np.concatenate(feature_parts, axis=1).astype(np.float32)
    y = (fusion_ranks < direct_ranks).astype(np.float32)

    indices = list(range(len(eval_rows)))
    random.Random(args.seed).shuffle(indices)
    folds = np.array_split(np.array(indices, dtype=np.int64), args.folds)

    routed_ranks = np.empty_like(direct_ranks)
    routed_probs = np.zeros(len(eval_rows), dtype=np.float32)
    routed_use_fusion = np.zeros(len(eval_rows), dtype=bool)
    fold_results = []

    for fold_id in range(args.folds):
        val_idx = folds[fold_id]
        train_idx = np.concatenate([folds[i] for i in range(args.folds) if i != fold_id])
        x_train, x_val, _, _ = standardize_train_val(x[train_idx], x[val_idx])
        y_train = y[train_idx]

        w, b = fit_logistic_regression(
            x=x_train,
            y=y_train,
            lr=args.lr,
            epochs=args.epochs,
            l2=args.l2,
            seed=args.seed + fold_id,
        )
        train_probs = predict_probs(x_train, w, b)
        threshold = select_threshold(
            probs=train_probs,
            direct_ranks=direct_ranks[train_idx],
            fusion_ranks=fusion_ranks[train_idx],
            num_thresholds=args.num_thresholds,
            cost_lambda=args.cost_lambda,
        )
        val_probs = predict_probs(x_val, w, b)
        val_use_fusion = val_probs >= threshold
        val_ranks = np.where(val_use_fusion, fusion_ranks[val_idx], direct_ranks[val_idx])

        routed_ranks[val_idx] = val_ranks
        routed_probs[val_idx] = val_probs
        routed_use_fusion[val_idx] = val_use_fusion
        fold_results.append(
            {
                "fold": fold_id,
                "threshold": float(threshold),
                "fusion_rate": round(100.0 * float(val_use_fusion.mean()), 2),
                **metrics_from_ranks(val_ranks),
            }
        )

    oracle_ranks = np.minimum(direct_ranks, fusion_ranks)
    results = {
        "direct": metrics_from_ranks(direct_ranks),
        "fusion": metrics_from_ranks(fusion_ranks),
        "oracle": metrics_from_ranks(oracle_ranks),
        "learned_router_cv": metrics_from_ranks(routed_ranks),
        "learned_router_fusion_rate": round(100.0 * float(routed_use_fusion.mean()), 2),
        "oracle_fusion_label_rate": round(100.0 * float(np.mean(y == 1)), 2),
        "fold_results": fold_results,
        "feature_names": feature_names,
    }

    if args.apply_split:
        if not args.feature_file or args.include_cache_features:
            raise ValueError("--apply-split currently expects training with --feature-file and without --include-cache-features.")
        missing = [
            name
            for name, value in [
                ("--apply-feature-file", args.apply_feature_file),
                ("--apply-direct-ranks", args.apply_direct_ranks),
                ("--apply-fusion-ranks", args.apply_fusion_ranks),
            ]
            if not value
        ]
        if missing:
            raise ValueError(f"{', '.join(missing)} are required when --apply-split is set.")

        apply_rows = read_jsonl(Path(args.data_root) / args.language / f"{args.apply_split}.jsonl")
        apply_x, apply_feature_names = load_route_feature_file(args.apply_feature_file)
        apply_feature_names = [f"route:{name}" for name in apply_feature_names]
        if apply_feature_names != feature_names:
            raise ValueError("Apply feature names do not match training feature names.")

        apply_direct_ranks = load_ranks(args.apply_direct_ranks)
        apply_fusion_ranks = load_ranks(args.apply_fusion_ranks)
        if (
            apply_x.shape[0] != len(apply_rows)
            or len(apply_direct_ranks) != len(apply_rows)
            or len(apply_fusion_ranks) != len(apply_rows)
        ):
            raise ValueError("Apply feature/rank files and apply split have different lengths.")

        x_train, _, mean, std = standardize_train_val(x, x)
        w, b = fit_logistic_regression(
            x=x_train,
            y=y,
            lr=args.lr,
            epochs=args.epochs,
            l2=args.l2,
            seed=args.seed,
        )
        train_probs = predict_probs(x_train, w, b)
        threshold = select_threshold(
            probs=train_probs,
            direct_ranks=direct_ranks,
            fusion_ranks=fusion_ranks,
            num_thresholds=args.num_thresholds,
            cost_lambda=args.cost_lambda,
        )

        apply_probs = predict_probs(standardize_with_stats(apply_x, mean, std), w, b)
        apply_use_fusion = apply_probs >= threshold
        apply_ranks = np.where(apply_use_fusion, apply_fusion_ranks, apply_direct_ranks)
        results["apply"] = {
            "split": args.apply_split,
            "threshold": float(threshold),
            "fusion_rate": round(100.0 * float(apply_use_fusion.mean()), 2),
            "direct": metrics_from_ranks(apply_direct_ranks),
            "fusion": metrics_from_ranks(apply_fusion_ranks),
            "oracle": metrics_from_ranks(np.minimum(apply_direct_ranks, apply_fusion_ranks)),
            "router": metrics_from_ranks(apply_ranks),
        }
        if args.apply_output_ranks:
            output_ranks = Path(args.apply_output_ranks)
            output_ranks.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_ranks, apply_ranks.astype(np.int32))
            results["apply"]["output_ranks"] = str(output_ranks)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved learned router analysis to: {output_path}")


if __name__ == "__main__":
    main()
