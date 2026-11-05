import json, csv, argparse
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from evaluate import calculate_ndcg

def read_jsonl(p):
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def toks(row, kind):
    key = "code_tokens" if kind == "code" else "docstring_tokens"
    if key in row:
        return [str(x).lower() for x in row[key]]
    raw = row.get("code" if kind == "code" else "docstring", "")
    return str(raw).lower().split()

def metrics(ranks):
    ranks = np.asarray(ranks)
    ndcgs = []
    for r in ranks:
        rel = [0] * 10
        if r < 10:
            rel[int(r)] = 1
        ndcgs.append(calculate_ndcg(rel, k=10))
    return {
        "r1": round(100 * float(np.mean(ranks < 1)), 2),
        "r5": round(100 * float(np.mean(ranks < 5)), 2),
        "r10": round(100 * float(np.mean(ranks < 10)), 2),
        "mrr": round(100 * float(np.mean(1.0 / (ranks + 1))), 2),
        "ndcg@10": round(100 * float(np.mean(ndcgs)), 2),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", nargs="+", required=True)
    ap.add_argument("--data-root", default="/root/autodl-tmp/lora-code-embeddings-main/data/CSN")
    ap.add_argument("--split", default="test")
    ap.add_argument("--output-dir", default="results/bm25_oldcsn")
    ap.add_argument("--k1", type=float, default=1.2)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--query-batch-size", type=int, default=256)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for lang in args.languages:
        print(f"\n--- BM25 evaluating {lang} / {args.split} ---", flush=True)
        root = Path(args.data_root) / lang
        queries = read_jsonl(root / f"{args.split}.jsonl")
        codebase = read_jsonl(root / "codebase.jsonl")
        url_to_idx = {r.get("url"): i for i, r in enumerate(codebase)}

        code_tokens = [toks(r, "code") for r in codebase]
        query_tokens = [toks(r, "query") for r in queries]

        vec = CountVectorizer(
            analyzer=lambda x: x,
            lowercase=False,
            token_pattern=None,
            dtype=np.float32,
        )
        X = vec.fit_transform(code_tokens).tocsr().astype(np.float32)
        Q = vec.transform(query_tokens).tocsr().astype(np.float32)

        n_docs = X.shape[0]
        doc_len = np.asarray(X.sum(axis=1)).ravel()
        avgdl = float(doc_len.mean())
        df = np.diff(X.tocsc().indptr).astype(np.float32)
        idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)

        row_ids = np.repeat(np.arange(n_docs), np.diff(X.indptr))
        denom = X.data + args.k1 * (1.0 - args.b + args.b * doc_len[row_ids] / avgdl)
        X.data = X.data * (args.k1 + 1.0) / denom * idf[X.indices]

        ranks = []
        for s in tqdm(range(0, Q.shape[0], args.query_batch_size), desc=f"{lang} BM25"):
            e = min(s + args.query_batch_size, Q.shape[0])
            S = (Q[s:e] @ X.T).tocsr()
            for local_i, qrow in enumerate(queries[s:e]):
                gold = url_to_idx.get(qrow.get("url"))
                if gold is None:
                    raise RuntimeError(f"Cannot find gold url for {lang}: {qrow.get('url')}")
                row = S.getrow(local_i)
                pos = np.where(row.indices == gold)[0]
                gold_score = float(row.data[pos[0]]) if len(pos) else 0.0
                rank = int(np.sum(row.data > gold_score))
                ranks.append(rank)

        ranks = np.asarray(ranks, dtype=np.int32)
        res = metrics(ranks)
        all_results[lang] = res
        print(f"  {lang}: {res}", flush=True)
        np.save(out_dir / f"{lang}_{args.split}_bm25.ranks.npy", ranks)

    avg = {k: round(float(np.mean([v[k] for v in all_results.values()])), 2)
           for k in ["r1", "r5", "r10", "mrr", "ndcg@10"]}
    all_results["average"] = avg

    print("\nAverage:", avg, flush=True)
    with open(out_dir / f"bm25_{args.split}_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    with open(out_dir / f"bm25_{args.split}_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["language", "r1", "r5", "r10", "mrr", "ndcg@10"])
        w.writeheader()
        for lang, res in all_results.items():
            w.writerow({"language": lang, **res})

if __name__ == "__main__":
    main()
