"""Unified evaluation script for debiased text2code retrieval.

Usage:
  # Evaluate a trained adapter on all languages
  python evaluate.py --adapter-path trained_adapters/baseline_seed42

  # Evaluate base model (no adapter)
  python evaluate.py --no-adapter

  # Evaluate on a single language
  python evaluate.py --adapter-path trained_adapters/baseline_seed42 --languages python
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer

from config import DATA_ROOT, LANGUAGES, RESULTS_DIR, resolve_model_path
from data_utils import create_dataset, create_loader
from train_utils import build_pooling


def read_adapter_eval_config(adapter_path):
    if adapter_path is None:
        return {}

    adapter_dir = Path(adapter_path)
    config_path = adapter_dir / "pooling_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    meta_path = adapter_dir / "experiment_meta.txt"
    if not meta_path.exists():
        return {}

    config = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.strip().split(":", 1)
            key = key.strip()
            value = value.strip()
            if key == "pooling":
                config["pooling_mode"] = value
    return config


def resolve_eval_options(adapter_path, pooling_mode, query_prefix, code_prefix, max_length, pooling_hidden_dim):
    adapter_config = read_adapter_eval_config(adapter_path)

    if pooling_mode == "auto":
        pooling_mode = adapter_config.get("pooling_mode", "mean")
    if query_prefix is None:
        query_prefix = adapter_config.get("query_prefix", "")
    if code_prefix is None:
        code_prefix = adapter_config.get("code_prefix", "")
    if max_length is None:
        max_length = int(adapter_config.get("max_length", 512))
    pooling_hidden_dim = int(adapter_config.get("pooling_hidden_dim", pooling_hidden_dim))

    if pooling_mode == "query_aware":
        raise ValueError(
            "query_aware pooling makes code embeddings query-dependent, so this bi-encoder "
            "evaluator cannot precompute one code embedding matrix. Use mean/last_token/attention "
            "for retrieval evaluation, or add a separate exact query-aware reranker."
        )

    return pooling_mode, query_prefix, code_prefix, max_length, pooling_hidden_dim


def attach_pooling_head(model, adapter_path, pooling_mode, pooling_hidden_dim):
    pooling = build_pooling(pooling_mode, pooling_hidden_dim)
    state_path = Path(adapter_path) / "pooling_head.pt" if adapter_path is not None else None

    if pooling_mode == "attention":
        if state_path is None or not state_path.exists():
            raise FileNotFoundError(
                f"attention pooling requires a trained pooling_head.pt, but it was not found at {state_path}"
            )
        pooling.load_state_dict(torch.load(state_path, map_location="cpu"))

    model.pooling_head = pooling
    return model


@torch.no_grad()
def get_feats(model, tokenizer, data_loader, max_length, device, pooling_mode="mean", desc="Get feats", text_prefix=""):
    embeds = []
    for text in tqdm(data_loader, total=len(data_loader), desc=desc):
        if text_prefix:
            text = [text_prefix + t for t in text]
        text_input = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        ids = text_input["input_ids"]
        mask = text_input["attention_mask"]
        hidden = model(ids, attention_mask=mask)[0]

        pooling_head = getattr(model, "pooling_head", None)
        if pooling_head is not None:
            pooled = pooling_head(hidden, mask)
        elif pooling_mode == "last_token":
            seq_lengths = mask.sum(dim=1) - 1
            batch_idx = torch.arange(hidden.size(0), device=hidden.device)
            pooled = hidden[batch_idx, seq_lengths.long()]
        else:
            in_mask = mask.unsqueeze(-1).expand(hidden.size()).float()
            pooled = torch.sum(hidden * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-6)

        embeds.append(pooled)
    return torch.cat(embeds, dim=0)


def calculate_dcg(relevance_scores, k):
    relevance_scores = np.array(relevance_scores[:k])
    return np.sum(relevance_scores / np.log2(np.arange(2, relevance_scores.size + 2)))


def calculate_ndcg(relevance_scores, k):
    dcg = calculate_dcg(relevance_scores, k)
    ideal = sorted(relevance_scores, reverse=True)
    idcg = calculate_dcg(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def _text2code_to_tensor(text2code, n_queries, device):
    if isinstance(text2code, torch.Tensor):
        return text2code.to(device)
    if isinstance(text2code, dict):
        return torch.tensor([text2code[i] for i in range(n_queries)], dtype=torch.long, device=device)
    return torch.tensor(list(text2code), dtype=torch.long, device=device)


@torch.no_grad()
def compute_retrieval_metrics(
    text_embeds,
    code_embeds,
    text2code,
    device,
    query_chunk_size=256,
    code_chunk_size=4096,
):
    text_embeds = torch.nn.functional.normalize(text_embeds, dim=1).to(device)
    code_embeds = torch.nn.functional.normalize(code_embeds, dim=1).to(device)
    n_queries = text_embeds.shape[0]
    n_codes = code_embeds.shape[0]

    text2code_t = _text2code_to_tensor(text2code, n_queries, device)

    ranks = torch.full((n_queries,), -1, dtype=torch.long, device=device)
    ndcgs_list = []

    for q_start in tqdm(range(0, n_queries, query_chunk_size), desc="Retrieval metrics"):
        q_end = min(q_start + query_chunk_size, n_queries)
        q_chunk = text_embeds[q_start:q_end]
        q_len = q_end - q_start
        chunk_scores = torch.empty((q_len, n_codes), dtype=torch.float32, device=device)

        for c_start in range(0, n_codes, code_chunk_size):
            c_end = min(c_start + code_chunk_size, n_codes)
            chunk_scores[:, c_start:c_end] = q_chunk @ code_embeds[c_start:c_end].t()

        sorted_indices = torch.argsort(chunk_scores, dim=1, descending=True)

        targets = text2code_t[q_start:q_end].unsqueeze(1)
        match_mask = sorted_indices == targets
        rank_positions = match_mask.nonzero(as_tuple=False)[:, 1]
        ranks[q_start:q_end] = rank_positions

        for local_i in range(q_len):
            r = int(rank_positions[local_i].item())
            rel = [0] * min(10, n_codes)
            if r < 10:
                rel[r] = 1
            ndcgs_list.append(calculate_ndcg(rel, k=10))

        del q_chunk, chunk_scores, sorted_indices, match_mask, rank_positions, targets
        torch.cuda.empty_cache()

    ranks_np = ranks.cpu().numpy()
    r1 = 100.0 * np.mean(ranks_np < 1)
    r5 = 100.0 * np.mean(ranks_np < 5)
    r10 = 100.0 * np.mean(ranks_np < 10)
    mrr = 100.0 * np.mean(1.0 / (ranks_np + 1))
    ndcg10 = 100.0 * np.mean(ndcgs_list)

    del text_embeds, code_embeds, text2code_t, ranks
    torch.cuda.empty_cache()

    return {
        "r1": round(r1, 2),
        "r5": round(r5, 2),
        "r10": round(r10, 2),
        "mrr": round(mrr, 2),
        "ndcg@10": round(ndcg10, 2),
    }


def load_model(model_name, adapter_path=None, pooling_mode="mean", pooling_hidden_dim=768):
    model_path = resolve_model_path(model_name)
    print(f"Loading base model from: {model_path}")
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    base_model = RobertaModel.from_pretrained(model_path)

    if adapter_path is not None:
        print(f"Loading adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        model.eval()
    else:
        model = base_model

    model = attach_pooling_head(model, adapter_path, pooling_mode, pooling_hidden_dim)
    return model, tokenizer


def evaluate_language(
    model,
    tokenizer,
    language,
    data_root,
    pooling_mode="auto",
    query_prefix="",
    code_prefix="",
    max_length=512,
    query_chunk_size=256,
    code_chunk_size=4096,
):
    print(f"\n--- Evaluating {language} ---")
    _, _, test_dataset, code_dataset = create_dataset(str(data_root), language)

    test_loader, code_loader = create_loader(
        [test_dataset, code_dataset],
        [None, None],
        batch_size=[256, 256],
        num_workers=[4, 4],
        is_trains=[False, False],
        collate_fns=[None, None],
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation.")
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    text_embeds = get_feats(model, tokenizer, test_loader, max_length, device, pooling_mode, "Text feats", query_prefix)
    code_embeds = get_feats(model, tokenizer, code_loader, max_length, device, pooling_mode, "Code feats", code_prefix)
    result = compute_retrieval_metrics(
        text_embeds,
        code_embeds,
        test_loader.dataset.text2code,
        device=device,
        query_chunk_size=query_chunk_size,
        code_chunk_size=code_chunk_size,
    )
    print(f"  {language}: {result}")
    return result


def evaluate_all(
    model_name,
    adapter_path=None,
    languages=None,
    data_root=None,
    output_file=None,
    pooling_mode="auto",
    experiment_name="",
    query_prefix=None,
    code_prefix=None,
    max_length=None,
    pooling_hidden_dim=768,
    query_chunk_size=256,
    code_chunk_size=4096,
):
    languages = languages or LANGUAGES
    data_root = data_root or str(DATA_ROOT)

    pooling_mode, query_prefix, code_prefix, max_length, pooling_hidden_dim = resolve_eval_options(
        adapter_path, pooling_mode, query_prefix, code_prefix, max_length, pooling_hidden_dim
    )

    model, tokenizer = load_model(model_name, adapter_path, pooling_mode, pooling_hidden_dim)

    all_results = {}
    for lang in languages:
        result = evaluate_language(
            model,
            tokenizer,
            lang,
            data_root,
            pooling_mode,
            query_prefix,
            code_prefix,
            max_length,
            query_chunk_size,
            code_chunk_size,
        )
        all_results[lang] = result

    avg_mrr = np.mean([r["mrr"] for r in all_results.values()])
    avg_r1 = np.mean([r["r1"] for r in all_results.values()])
    avg_r5 = np.mean([r["r5"] for r in all_results.values()])
    avg_r10 = np.mean([r["r10"] for r in all_results.values()])
    avg_ndcg = np.mean([r["ndcg@10"] for r in all_results.values()])

    all_results["average"] = {
        "r1": round(avg_r1, 2),
        "r5": round(avg_r5, 2),
        "r10": round(avg_r10, 2),
        "mrr": round(avg_mrr, 2),
        "ndcg@10": round(avg_ndcg, 2),
    }

    print(f"\n{'=' * 60}")
    print(f"Average: {all_results['average']}")
    print(f"{'=' * 60}")

    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Adapter: {adapter_path}\n")
            f.write(f"Pooling: {pooling_mode}\n")
            f.write(f"Max length: {max_length}\n")
            f.write(f"Query prefix: {query_prefix!r}\n")
            f.write(f"Code prefix: {code_prefix!r}\n")
            f.write(f"{'-' * 40}\n")
            for lang in languages:
                f.write(f"  {lang}: {all_results[lang]}\n")
            f.write(f"  average: {all_results['average']}\n")
            f.write(f"{'=' * 60}\n")
        print(f"Results appended to {out_path}")

    json_path = Path(output_file or "results.txt").with_suffix(".json")
    json_path = json_path.parent / f"{experiment_name}_{json_path.name}" if experiment_name else json_path
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"experiment": experiment_name, "results": all_results}, f, indent=2)
    print(f"JSON results saved to {json_path}")

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate text2code retrieval")
    parser.add_argument("--model-name", type=str, default="microsoft/unixcoder-base")
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--languages", type=str, nargs="+", default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--pooling", type=str, default="auto", choices=["auto", "mean", "last_token", "attention", "query_aware"])
    parser.add_argument("--pooling-hidden-dim", type=int, default=768)
    parser.add_argument("--query-prefix", type=str, default=None)
    parser.add_argument("--code-prefix", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default="")
    parser.add_argument("--query-chunk-size", type=int, default=256)
    parser.add_argument("--code-chunk-size", type=int, default=4096)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.output_file is None:
        args.output_file = str(RESULTS_DIR / "eval_results.txt")

    adapter_path = None if args.no_adapter else args.adapter_path

    evaluate_all(
        model_name=args.model_name,
        adapter_path=adapter_path,
        languages=args.languages,
        data_root=args.data_root,
        output_file=args.output_file,
        pooling_mode=args.pooling,
        experiment_name=args.experiment_name,
        query_prefix=args.query_prefix,
        code_prefix=args.code_prefix,
        max_length=args.max_length,
        pooling_hidden_dim=args.pooling_hidden_dim,
        query_chunk_size=args.query_chunk_size,
        code_chunk_size=args.code_chunk_size,
    )


if __name__ == "__main__":
    main()
