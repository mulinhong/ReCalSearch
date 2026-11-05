#!/usr/bin/env python3
"""Reviewer-revision wrapper around the official CoRNStack llm-reranker.

Changes versus the upstream cornstack/src/rerank.py are deliberately narrow:
1) run all six CodeSearchNet languages rather than hard-coded Ruby only;
2) accept a local CodeRankLLM model path;
3) fail closed if the llm-reranker checkout / frozen rank.tsv inputs are absent;
4) preserve the published listwise settings top_k=100, window=10, step=5 by default.

The actual listwise ranking implementation remains upstream llm-reranker/scripts/rerank_llm.py.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def evaluate_results(eval_dir: str, dataset_name: str, qrels_path: str, results_path: str):
    with open(qrels_path, encoding="utf-8") as f:
        qrels = {}
        for i, line in enumerate(f):
            if i == 0 and line.lower().startswith("query-id"):
                continue
            qid, docid, score = line.rstrip("\n").split("\t")
            qrels.setdefault(qid, {})[docid] = int(score)

    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    metrics = [1, 3, 5, 10, 20, 100]
    mrr_at_k = {}
    for k in metrics:
        mrr_sum = 0.0
        num_queries = 0
        for qid in qrels:
            if qid not in results:
                continue
            sorted_docs = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)[:k]
            for rank, (doc_id, _) in enumerate(sorted_docs, start=1):
                if doc_id in qrels[qid] and qrels[qid][doc_id] > 0:
                    mrr_sum += 1.0 / rank
                    break
            num_queries += 1
        mrr_at_k[k] = mrr_sum / num_queries if num_queries else 0.0

    out = Path(eval_dir) / "eval_results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{dataset_name}_eval.json").write_text(json.dumps(mrr_at_k, indent=2), encoding="utf-8")
    return mrr_at_k


def run_checked(cmd, dry_run=False):
    print("+", " ".join(map(str, cmd)), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--model", default="/root/autodl-tmp/hf_models/CodeRankLLM")
    ap.add_argument("--languages", nargs="+", default=["ruby", "javascript", "php", "python", "java", "go"])
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--window_size", type=int, default=10)
    ap.add_argument("--step_size", type=int, default=5)
    ap.add_argument("--context_size", type=int, default=32768)
    ap.add_argument("--do_batched", type=int, default=1, choices=[0, 1])
    ap.add_argument("--llm_reranker_dir", default="", help="Defaults to ../llm-reranker relative to this file")
    ap.add_argument("--skip_convert", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if args.top_k != 100 or args.window_size != 10 or args.step_size != 5:
        print("[WARN] Non-published CodeRankLLM inference settings requested. Main paper baseline should use top_k=100, window=10, step=5.", file=sys.stderr)

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Local CodeRankLLM model does not exist: {model_path}")

    here = Path(__file__).resolve().parent
    reranker_dir = Path(args.llm_reranker_dir).resolve() if args.llm_reranker_dir else (here.parent / "llm-reranker").resolve()
    convert_script = reranker_dir / "scripts" / "convert_results.py"
    rerank_script = reranker_dir / "scripts" / "rerank_llm.py"
    if not convert_script.exists() or not rerank_script.exists():
        raise FileNotFoundError(
            f"Official llm-reranker checkout is missing. Expected {convert_script} and {rerank_script}. "
            "Clone gangiswag/llm-reranker into cornstack-official/llm-reranker first."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.eval_dir, exist_ok=True)
    code_datasets_dir = Path(args.dataset_dir) / "code_datasets"

    for lang in args.languages:
        dataset_name = f"csn_{lang}"
        dataset_path = code_datasets_dir / dataset_name
        rank_tsv = Path(args.output_dir) / dataset_name / "rank.tsv"
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"Missing BEIR dataset directory: {dataset_path}")
        if not rank_tsv.exists():
            raise FileNotFoundError(f"Missing frozen first-stage rank file: {rank_tsv}")

        if not args.skip_convert:
            run_checked([
                sys.executable, str(convert_script),
                "--dataset", dataset_name,
                "--output_dir", args.output_dir,
                "--data_type", "codedataset",
                "--data_dir", str(code_datasets_dir),
                "--top_k", str(args.top_k),
                "--rerank_type", "code",
            ], dry_run=args.dry_run)

        expected_rank_json = Path(args.output_dir) / dataset_name / f"rank_{args.top_k}.json"
        if not args.dry_run and not expected_rank_json.exists():
            raise FileNotFoundError(f"Conversion did not create {expected_rank_json}")

        run_checked([
            sys.executable, str(rerank_script),
            "--model", str(model_path),
            "--dataset", dataset_name,
            "--output_dir", args.output_dir,
            "--data_type", "codedataset",
            "--data_dir", str(code_datasets_dir),
            "--use_logits", "0",
            "--use_alpha", "0",
            "--context_size", str(args.context_size),
            "--llm_top_k", str(args.top_k),
            "--window_size", str(args.window_size),
            "--step_size", str(args.step_size),
            "--do_batched", str(args.do_batched),
            "--rerank_type", "code",
            "--code_prompt_type", "docstring",
        ], dry_run=args.dry_run)

        if args.dry_run:
            continue
        qrels_path = dataset_path / "qrels" / "test.tsv"
        results_path = Path(args.output_dir) / "code_datasets" / dataset_name / f"rerank_{args.top_k}_llm_gen_num.json"
        if not results_path.exists():
            raise FileNotFoundError(f"Official reranker did not create expected output: {results_path}")
        metrics = evaluate_results(args.eval_dir, dataset_name, str(qrels_path), str(results_path))
        print(f"[{dataset_name}] " + " ".join(f"MRR@{k}={v:.6f}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
