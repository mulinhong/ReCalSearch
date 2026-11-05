#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_query_text(example: Dict[str, Any]) -> str:
    for key in ['docstring', 'doc', 'query', 'nl', 'text']:
        value = example.get(key)
        if value:
            if isinstance(value, list):
                return ' '.join(map(str, value))
            return str(value)
    value = example.get('docstring_tokens')
    if value:
        if isinstance(value, list):
            return ' '.join(map(str, value))
        return str(value)
    raise KeyError(f'Could not find query text in keys={list(example.keys())}')


def get_code_text(example: Dict[str, Any]) -> str:
    value = example.get('code')
    if value:
        return str(value)
    for key in ['code_tokens', 'function_tokens']:
        value = example.get(key)
        if value:
            if isinstance(value, list):
                return ' '.join(map(str, value))
            return str(value)
    raise KeyError(f'Could not find code text in keys={list(example.keys())}')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            'Convert a frozen ReCalSearch Nomic candidate pool into the BEIR + rank.tsv layout '
            'used by the official CoRNStack/llm-reranker pipeline. Ground truth is taken ONLY '
            'from pool["gt_indices"], which was exported from create_dataset(...).text2code.'
        )
    )
    ap.add_argument('--language', required=True)
    ap.add_argument('--split', default='test', choices=['valid', 'test'])
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--candidate-pool', required=True)
    ap.add_argument('--dataset-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--top-k', type=int, default=100)
    ap.add_argument('--query-offset', type=int, default=0)
    ap.add_argument('--max-queries', type=int, default=0)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    data_root = Path(args.data_root)
    pool_path = Path(args.candidate_pool)
    dataset_root = Path(args.dataset_dir)
    output_root = Path(args.output_dir)
    dataset_name = f'csn_{args.language}'

    split_path = data_root / args.language / f'{args.split}.jsonl'
    codebase_path = data_root / args.language / 'codebase.jsonl'

    for p in (split_path, codebase_path, pool_path):
        if not p.exists():
            raise FileNotFoundError(p)

    queries_all = read_jsonl(split_path)
    codebase = read_jsonl(codebase_path)

    pool = np.load(pool_path, allow_pickle=False)

    required = [
        k for k in ('indices', 'scores', 'base_ranks', 'gt_indices')
        if k not in pool
    ]
    if required:
        raise ValueError(f'Candidate pool missing arrays: {required}')

    indices_all = np.asarray(pool['indices'], dtype=np.int64)
    scores_all = np.asarray(pool['scores'], dtype=np.float32)
    base_ranks_all = np.asarray(pool['base_ranks'], dtype=np.int64)
    gt_indices_all = np.asarray(pool['gt_indices'], dtype=np.int64)

    n_queries = len(queries_all)

    if indices_all.ndim != 2 or scores_all.shape != indices_all.shape:
        raise ValueError(
            f'Invalid pool shapes: indices={indices_all.shape}, '
            f'scores={scores_all.shape}'
        )

    if indices_all.shape[0] != n_queries:
        raise ValueError(
            f'Split rows ({n_queries}) != candidate pool rows '
            f'({indices_all.shape[0]})'
        )

    if len(base_ranks_all) != n_queries or len(gt_indices_all) != n_queries:
        raise ValueError('base_ranks/gt_indices length mismatch')

    if indices_all.shape[1] < args.top_k:
        raise ValueError(
            f'Pool width {indices_all.shape[1]} < requested top-k {args.top_k}'
        )

    start = int(args.query_offset)

    if start < 0 or start >= n_queries:
        raise ValueError(
            f'query-offset {start} outside [0,{n_queries - 1}]'
        )

    stop = (
        n_queries
        if args.max_queries <= 0
        else min(n_queries, start + int(args.max_queries))
    )

    queries_src = queries_all[start:stop]
    indices = indices_all[start:stop, :args.top_k]
    scores = scores_all[start:stop, :args.top_k]
    base_ranks = base_ranks_all[start:stop]
    gt_indices = gt_indices_all[start:stop]

    if np.any(indices < 0) or np.any(indices >= len(codebase)):
        raise ValueError('Candidate pool contains out-of-range code indices')

    if np.any(gt_indices < 0) or np.any(gt_indices >= len(codebase)):
        raise ValueError('gt_indices contains out-of-range code indices')

    code_texts = [get_code_text(row) for row in codebase]

    beir_dir = dataset_root / 'code_datasets' / dataset_name
    qrels_dir = beir_dir / 'qrels'
    rank_dir = output_root / dataset_name

    for d in (beir_dir, qrels_dir, rank_dir):
        d.mkdir(parents=True, exist_ok=True)

    targets = [
        beir_dir / 'corpus.jsonl',
        beir_dir / 'queries.jsonl',
        qrels_dir / 'test.tsv',
        rank_dir / 'rank.tsv',
    ]

    if not args.overwrite:
        existing = [str(p) for p in targets if p.exists()]
        if existing:
            raise FileExistsError(
                'Refusing to overwrite existing files without --overwrite: '
                + ', '.join(existing)
            )

    with (beir_dir / 'corpus.jsonl').open(
        'w', encoding='utf-8'
    ) as f:
        for i, code in enumerate(code_texts):
            f.write(
                json.dumps(
                    {'_id': str(i), 'title': '', 'text': code},
                    ensure_ascii=False
                ) + '\n'
            )

    with (beir_dir / 'queries.jsonl').open(
        'w', encoding='utf-8'
    ) as f:
        for local_qid, row in enumerate(queries_src):
            f.write(
                json.dumps(
                    {
                        '_id': str(local_qid),
                        'text': get_query_text(row)
                    },
                    ensure_ascii=False
                ) + '\n'
            )

    with (qrels_dir / 'test.tsv').open(
        'w', encoding='utf-8'
    ) as f:
        f.write('query-id\tcorpus-id\tscore\n')

        for local_qid, gt in enumerate(gt_indices):
            f.write(
                f'{local_qid}\t{int(gt)}\t1\n'
            )

    with (rank_dir / 'rank.tsv').open(
        'w', encoding='utf-8'
    ) as f:
        f.write('query-id\tcorpus-id\tscore\n')

        for local_qid in range(len(queries_src)):
            for j in range(args.top_k):
                f.write(
                    f'{local_qid}\t'
                    f'{int(indices[local_qid, j])}\t'
                    f'{float(scores[local_qid, j]):.9g}\n'
                )

    selected_base_mrr = (
        float(
            np.mean(
                1.0 / (
                    base_ranks.astype(np.float64) + 1.0
                )
            )
        )
        if len(base_ranks)
        else 0.0
    )

    manifest = {
        'format': 'coderankllm_official_beir_v2_recal_gt',
        'gt_source':
            'candidate_pool.gt_indices exported from '
            'create_dataset(...).text2code',
        'language': args.language,
        'source_split': args.split,
        'dataset_name': dataset_name,
        'source_query_offset': start,
        'queries_selected': len(queries_src),
        'queries_total': n_queries,
        'corpus': len(codebase),
        'top_k': args.top_k,
        'candidate_pool': str(pool_path.resolve()),
        'candidate_pool_sha256': sha256_file(pool_path),
        'split_file': str(split_path.resolve()),
        'codebase_file': str(codebase_path.resolve()),
        'base_mrr_selected_from_pool': selected_base_mrr,
        'beir_dir': str(beir_dir.resolve()),
        'rank_tsv': str(
            (rank_dir / 'rank.tsv').resolve()
        ),
    }

    (rank_dir / 'input_manifest.json').write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False
        ),
        encoding='utf-8'
    )

    print(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False
        )
    )


if __name__ == '__main__':
    main()
