import argparse, csv, json, hashlib, re
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from evaluate import calculate_ndcg

def sanitize_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")

def cache_paths(cache_dir, language, split, model_name, max_seq_length, query_prefix, code_prefix):
    root = Path(cache_dir)
    model_key = sanitize_name(model_name)
    cache_key = json.dumps({
        "model_name": model_name,
        "backend": "nomic_transformers",
        "query_prefix": query_prefix,
        "code_prefix": code_prefix,
        "max_seq_length": max_seq_length,
    }, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:10]
    prefix = root / f"{language}_{model_key}_msl{max_seq_length}_{digest}"
    return {
        "query": root / f"{language}_{split}_{model_key}_msl{max_seq_length}_{digest}.query.npy",
        "code": prefix.with_suffix(".code.npy"),
    }

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

def zscore(x):
    x = np.asarray(x, dtype=np.float32)
    s = float(x.std())
    if s < 1e-8:
        return x * 0.0
    return (x - float(x.mean())) / s

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
    ap.add_argument("--cache-dir", default="data/rerank/coderank_cache_oldcsn")
    ap.add_argument("--model-path", default="/root/autodl-tmp/hf_models/nomic-embed-code")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--query-prefix", default="Represent this query for searching relevant code: ")
    ap.add_argument("--code-prefix", default="")
    ap.add_argument("--candidate-top-k", type=int, default=100)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0, 0.25, 0.5, 1, 2, 5])
    ap.add_argument("--query-batch-size", type=int, default=64)
    ap.add_argument("--code-chunk-size", type=int, default=65536)
    ap.add_argument("--output-dir", default="results/bm25_nomic_hybrid_oldcsn")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Nomic top-k scoring.")
    device = torch.device("cuda")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {f"alpha={a:g}": {} for a in args.alphas}

    for lang in args.languages:
        print(f"\n--- Hybrid BM25+Nomic evaluating {lang} / {args.split} ---", flush=True)
        root = Path(args.data_root) / lang
        queries = read_jsonl(root / f"{args.split}.jsonl")
        codebase = read_jsonl(root / "codebase.jsonl")
        url_to_idx = {r.get("url"): i for i, r in enumerate(codebase)}
        gold = np.array([url_to_idx[q.get("url")] for q in queries], dtype=np.int64)

        # BM25 matrix.
        code_tokens = [toks(r, "code") for r in codebase]
        query_tokens = [toks(r, "query") for r in queries]
        vec = CountVectorizer(analyzer=lambda x: x, lowercase=False, token_pattern=None, dtype=np.float32)
        X = vec.fit_transform(code_tokens).tocsr().astype(np.float32)
        Q = vec.transform(query_tokens).tocsr().astype(np.float32)
        n_docs = X.shape[0]
        doc_len = np.asarray(X.sum(axis=1)).ravel()
        avgdl = float(doc_len.mean())
        df = np.diff(X.tocsc().indptr).astype(np.float32)
        idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
        row_ids = np.repeat(np.arange(n_docs), np.diff(X.indptr))
        denom = X.data + 1.2 * (1.0 - 0.75 + 0.75 * doc_len[row_ids] / avgdl)
        X.data = X.data * 2.2 / denom * idf[X.indices]

        caches = cache_paths(args.cache_dir, lang, args.split, args.model_path, args.max_length, args.query_prefix, args.code_prefix)
        q_emb = np.load(caches["query"], mmap_mode="r")
        c_emb = np.load(caches["code"], mmap_mode="r")

        ranks_by_alpha = {a: [] for a in args.alphas}

        for s in tqdm(range(0, len(queries), args.query_batch_size), desc=f"{lang} hybrid"):
            e = min(s + args.query_batch_size, len(queries))
            q = torch.from_numpy(np.asarray(q_emb[s:e])).to(device).float()
            q = F.normalize(q, dim=1)

            top_scores = torch.full((e - s, args.candidate_top_k), -1e9, device=device)
            top_indices = torch.full((e - s, args.candidate_top_k), -1, dtype=torch.long, device=device)

            for cs in range(0, c_emb.shape[0], args.code_chunk_size):
                ce = min(cs + args.code_chunk_size, c_emb.shape[0])
                c = torch.from_numpy(np.asarray(c_emb[cs:ce])).to(device).float()
                c = F.normalize(c, dim=1)
                sim = q @ c.t()
                vals, idx = torch.topk(sim, k=min(args.candidate_top_k, sim.shape[1]), dim=1)
                idx = idx + cs
                merged_scores = torch.cat([top_scores, vals], dim=1)
                merged_indices = torch.cat([top_indices, idx], dim=1)
                vals2, pos = torch.topk(merged_scores, k=args.candidate_top_k, dim=1)
                top_indices = torch.gather(merged_indices, 1, pos)
                top_scores = vals2
                del c, sim, vals, idx, merged_scores, merged_indices, vals2, pos

            cand_idx = top_indices.cpu().numpy()
            nomic_scores = top_scores.cpu().numpy()
            del q, top_scores, top_indices
            torch.cuda.empty_cache()

            bm25_rows = Q[s:e] @ X.T
            for i in range(e - s):
                candidates = cand_idx[i]
                dense = nomic_scores[i]
                bm25_row = bm25_rows.getrow(i)
                bm25_map = dict(zip(bm25_row.indices.tolist(), bm25_row.data.tolist()))
                bm25 = np.array([bm25_map.get(int(c), 0.0) for c in candidates], dtype=np.float32)
                g = gold[s + i]

                for a in args.alphas:
                    final = zscore(dense) + a * zscore(bm25)
                    order = np.argsort(-final, kind="mergesort")
                    sorted_candidates = candidates[order]
                    hit = np.where(sorted_candidates == g)[0]
                    if len(hit):
                        rank = int(hit[0])
                    else:
                        rank = int(args.candidate_top_k + np.sum(dense > -1e8))
                    ranks_by_alpha[a].append(rank)

        for a in args.alphas:
            ranks = np.asarray(ranks_by_alpha[a], dtype=np.int32)
            res = metrics(ranks)
            res["fusion_alpha"] = float(a)
            res["candidate_top_k"] = args.candidate_top_k
            key = f"alpha={a:g}"
            all_results[key][lang] = res
            np.save(out_dir / f"{lang}_{args.split}_bm25_nomic_hybrid_top{args.candidate_top_k}_alpha={a:g}.ranks.npy", ranks)
            print(f"  {lang} / {key}: {res}", flush=True)

    for key in all_results:
        avg = {m: round(float(np.mean([all_results[key][l][m] for l in args.languages])), 2)
               for m in ["r1", "r5", "r10", "mrr", "ndcg@10"]}
        all_results[key]["average"] = avg
        print(f"{key} Average: {avg}", flush=True)

    out_json = out_dir / f"bm25_nomic_hybrid_{args.split}_results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("saved:", out_json)

if __name__ == "__main__":
    main()
