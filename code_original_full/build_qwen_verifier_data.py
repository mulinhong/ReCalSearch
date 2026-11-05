import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from config import DATA_ROOT
from eval_llm_verifier import build_yesno_prompt
from eval_nomic_embed import get_code_text


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_codebase_codes(data_root: Path, language: str) -> List[str]:
    codebase_path = data_root / language / "codebase.jsonl"
    codebase_rows = read_jsonl(codebase_path)
    return [get_code_text(row) for row in codebase_rows]


def resolve_positive_code(row: Dict[str, Any], codebase_codes: List[str]) -> Optional[str]:
    positive = row.get("positive") or {}
    code_index = positive.get("code_index")
    if code_index is not None:
        try:
            return codebase_codes[int(code_index)]
        except (IndexError, ValueError, TypeError):
            pass
    code = positive.get("code")
    return str(code) if code else None


def label_to_answer(label: int) -> str:
    return " Yes" if int(label) == 1 else " No"


def build_examples(args) -> List[Dict[str, Any]]:
    rows = read_jsonl(Path(args.input_file))
    if args.max_groups is not None:
        rows = rows[args.query_offset:args.query_offset + args.max_groups]
    elif args.query_offset:
        rows = rows[args.query_offset:]

    codebase_codes = load_codebase_codes(Path(args.data_root), args.language)
    examples: List[Dict[str, Any]] = []
    skipped = 0

    for group_idx, row in enumerate(rows):
        query = str(row.get("query", ""))
        if not query:
            skipped += 1
            continue

        query_id = row.get("query_id", f"{args.language}:{args.query_offset + group_idx}")
        positive_code = resolve_positive_code(row, codebase_codes)
        if positive_code is None:
            skipped += 1
            continue

        positive_prompt = build_yesno_prompt(
            query=query,
            code=positive_code,
            max_query_chars=args.max_query_chars,
            max_code_chars=args.max_code_chars,
        )
        for repeat_idx in range(args.positive_repeat):
            examples.append(
                {
                    "query_id": query_id,
                    "language": args.language,
                    "label": 1,
                    "answer": "Yes",
                    "target": label_to_answer(1),
                    "prompt": positive_prompt,
                    "source": "positive",
                    "mining": row.get("mining", row.get("mining_retriever", "unknown")),
                    "repeat": repeat_idx,
                }
            )

        negative_indices = row.get("negative_code_indices") or []
        negative_ranks = row.get("negative_retrieval_ranks") or []
        negative_scores = row.get("negative_retrieval_scores") or []
        max_negatives = min(args.negatives_per_query, len(negative_indices))
        for neg_pos in range(max_negatives):
            try:
                code_index = int(negative_indices[neg_pos])
                code = codebase_codes[code_index]
            except (IndexError, ValueError, TypeError):
                continue

            prompt = build_yesno_prompt(
                query=query,
                code=code,
                max_query_chars=args.max_query_chars,
                max_code_chars=args.max_code_chars,
            )
            source_rank = negative_ranks[neg_pos] if neg_pos < len(negative_ranks) else None
            source_score = negative_scores[neg_pos] if neg_pos < len(negative_scores) else None
            examples.append(
                {
                    "query_id": query_id,
                    "language": args.language,
                    "label": 0,
                    "answer": "No",
                    "target": label_to_answer(0),
                    "prompt": prompt,
                    "source": "negative",
                    "mining": row.get("mining", row.get("mining_retriever", "unknown")),
                    "code_index": code_index,
                    "source_rank": source_rank,
                    "source_score": source_score,
                    "deployed_nomic_score": ((row.get("negative_deployed_nomic_scores") or [])[neg_pos] if neg_pos < len(row.get("negative_deployed_nomic_scores") or []) else None),
                }
            )

    rng = random.Random(args.seed)
    if args.shuffle:
        rng.shuffle(examples)

    positives = sum(1 for row in examples if int(row["label"]) == 1)
    negatives = len(examples) - positives
    print(
        json.dumps(
            {
                "input_groups": len(rows),
                "saved_examples": len(examples),
                "positives": positives,
                "negatives": negatives,
                "skipped_groups": skipped,
                "positive_repeat": args.positive_repeat,
                "positive_examples_per_group": args.positive_repeat,
                "negatives_per_query": args.negatives_per_query,
                "expected_examples_per_complete_group": args.positive_repeat + args.negatives_per_query,
            },
            indent=2,
        )
    )
    return examples


def main():
    parser = argparse.ArgumentParser(description="Build supervised Yes/No Qwen verifier data from grouped rerank data.")
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--negatives-per-query", type=int, default=7)
    parser.add_argument("--positive-repeat", type=int, default=2)
    parser.add_argument("--max-query-chars", type=int, default=512)
    parser.add_argument("--max-code-chars", type=int, default=3000)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()

    examples = build_examples(args)
    saved = write_jsonl(Path(args.output_file), examples)
    print(f"Saved {saved} verifier examples to: {args.output_file}")


if __name__ == "__main__":
    main()
