import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import DATA_ROOT
from eval_nomic_embed import get_code_text, get_query_text, read_jsonl


def canonicalize_code(code: str) -> str:
    return re.sub(r"\s+", " ", code.strip())


def load_query_indices(path: str, language: str) -> Optional[List[int]]:
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return [int(x) for x in payload]
    if "languages" in payload:
        entry = payload["languages"].get(language)
        if entry is None:
            raise KeyError(f"Language {language!r} not found in query manifest {path}")
        values = entry.get("indices", entry) if isinstance(entry, dict) else entry
        return [int(x) for x in values]
    values = payload.get(language)
    if values is None:
        raise KeyError(f"Language {language!r} not found in query manifest {path}")
    return [int(x) for x in values]


def resolve_positive_code_index(
    example: Dict[str, Any],
    url_to_code: Dict[str, int],
    canon_to_indices: Dict[str, List[int]],
) -> Optional[int]:
    url = example.get("url")
    if url is not None and url in url_to_code:
        return int(url_to_code[url])

    try:
        code = get_code_text(example)
    except Exception:
        return None
    candidates = canon_to_indices.get(canonicalize_code(code), [])
    return int(candidates[0]) if candidates else None


def sample_negative_indices(
    rng: random.Random,
    n_codes: int,
    num_negatives: int,
    used: set,
    pos_canon: str,
    canon_code_texts: List[str],
    exclude_canonical_duplicates: bool,
) -> List[int]:
    negatives: List[int] = []
    max_attempts = max(1000, num_negatives * 100)

    for _ in range(max_attempts):
        if len(negatives) >= num_negatives:
            break
        idx = rng.randrange(n_codes)
        if idx in used:
            continue
        if exclude_canonical_duplicates and canon_code_texts[idx] == pos_canon:
            used.add(idx)
            continue
        negatives.append(idx)
        used.add(idx)

    if len(negatives) < num_negatives:
        start = rng.randrange(n_codes)
        for offset in range(n_codes):
            if len(negatives) >= num_negatives:
                break
            idx = (start + offset) % n_codes
            if idx in used:
                continue
            if exclude_canonical_duplicates and canon_code_texts[idx] == pos_canon:
                continue
            negatives.append(idx)
            used.add(idx)

    return negatives


def make_group(
    language: str,
    row_idx: int,
    example: Dict[str, Any],
    pos_code_index: Optional[int],
    negative_indices: List[int],
) -> Dict[str, Any]:
    query = get_query_text(example)
    pos_code = get_code_text(example)
    return {
        "format": "grouped_indices_v1",
        "mining": "random",
        "mining_retriever": "random",
        "query_id": f"{language}:{row_idx}",
        "query": query,
        "language": language,
        "positive": {
            "code_index": pos_code_index,
            "code": pos_code if pos_code_index is None else None,
            "retrieval_rank": None,
            "retrieval_score": None,
        },
        "negative_code_indices": negative_indices,
        "negative_retrieval_ranks": [],
        "negative_retrieval_scores": [],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build grouped random-negative reranker/verifier training data."
    )
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"])
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--num-negatives", type=int, default=31)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--query-indices-file", type=str, default="", help="Immutable query manifest used to keep every negative-source control on identical training queries.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-canonical-duplicates",
        action="store_true",
        help="Allow sampled negatives whose normalized code equals the positive code.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    examples = read_jsonl(data_root / args.language / f"{args.split}.jsonl")
    codebase_examples = read_jsonl(data_root / args.language / "codebase.jsonl")
    print(f"Read {len(examples)} data from {data_root / args.language / f'{args.split}.jsonl'}")
    print(f"Read {len(codebase_examples)} data from {data_root / args.language / 'codebase.jsonl'}")

    original_examples = examples
    selected_indices = load_query_indices(args.query_indices_file, args.language)
    if selected_indices is not None:
        if args.query_offset or args.max_samples is not None:
            raise ValueError("Use either --query-indices-file or --query-offset/--max-samples, not both.")
        if any(i < 0 or i >= len(original_examples) for i in selected_indices):
            raise IndexError("Query manifest contains out-of-range indices.")
        examples = [original_examples[i] for i in selected_indices]
        print(f"Using immutable query manifest {args.query_indices_file}: rows={len(examples)}")
    else:
        end = None if args.max_samples is None else args.query_offset + args.max_samples
        if args.query_offset or args.max_samples is not None:
            examples = original_examples[args.query_offset:end]
        print(f"Using query slice offset={args.query_offset}, rows={len(examples)}")

    code_texts = [get_code_text(row) for row in codebase_examples]
    canon_code_texts = [canonicalize_code(code) for code in code_texts]
    canon_to_indices: Dict[str, List[int]] = {}
    for idx, canon in enumerate(canon_code_texts):
        canon_to_indices.setdefault(canon, []).append(idx)
    url_to_code = {row["url"]: idx for idx, row in enumerate(codebase_examples) if "url" in row}

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved = 0
    skipped = 0
    missing_pos = 0
    exclude_dupes = not args.allow_canonical_duplicates

    with output_path.open("w", encoding="utf-8") as f:
        for local_idx, example in enumerate(examples):
            row_idx = selected_indices[local_idx] if selected_indices is not None else args.query_offset + local_idx
            pos_code_index = resolve_positive_code_index(
                example,
                url_to_code=url_to_code,
                canon_to_indices=canon_to_indices,
            )
            if pos_code_index is None:
                missing_pos += 1

            pos_canon = canonicalize_code(get_code_text(example))
            used = set()
            if pos_code_index is not None:
                used.add(int(pos_code_index))

            rng = random.Random(args.seed + row_idx)
            negatives = sample_negative_indices(
                rng=rng,
                n_codes=len(codebase_examples),
                num_negatives=args.num_negatives,
                used=used,
                pos_canon=pos_canon,
                canon_code_texts=canon_code_texts,
                exclude_canonical_duplicates=exclude_dupes,
            )
            if not negatives:
                skipped += 1
                continue

            group = make_group(
                language=args.language,
                row_idx=row_idx,
                example=example,
                pos_code_index=pos_code_index,
                negative_indices=negatives,
            )
            f.write(json.dumps(group, ensure_ascii=False) + "\n")
            saved += 1

    print(f"Saved {saved} random-negative groups to: {output_path}")
    if missing_pos:
        print(f"[WARN] {missing_pos} examples have no resolved codebase index; positive code was stored inline.")
    if skipped:
        print(f"[WARN] Skipped {skipped} groups without negatives.")


if __name__ == "__main__":
    main()
