import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from config import DATA_ROOT
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


class CodebaseCache:
    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.codes_by_language: Dict[str, List[str]] = {}

    def codes(self, language: str) -> List[str]:
        if language not in self.codes_by_language:
            codebase_path = self.data_root / language / "codebase.jsonl"
            rows = read_jsonl(codebase_path)
            self.codes_by_language[language] = [get_code_text(row) for row in rows]
            print(f"Loaded {len(rows)} code snippets for {language} from {codebase_path}")
        return self.codes_by_language[language]


def resolve_positive_code(group: Dict[str, Any], codebase_codes: List[str]) -> str:
    positive = group.get("positive") or {}
    code_index = positive.get("code_index")
    if code_index is not None:
        return codebase_codes[int(code_index)]
    code = positive.get("code")
    if code:
        return str(code)
    raise ValueError(f"Group has no resolvable positive code: {group.get('query_id')}")


def materialize_group(
    group: Dict[str, Any],
    codebase_cache: CodebaseCache,
    max_negatives: int,
) -> List[Dict[str, Any]]:
    language = str(group.get("language") or str(group.get("query_id", "")).split(":", 1)[0])
    if not language:
        raise ValueError(f"Cannot infer language for query_id={group.get('query_id')}")

    codebase_codes = codebase_cache.codes(language)
    query_id = group.get("query_id")
    query = str(group["query"])
    rows = [
        {
            "query_id": query_id,
            "language": language,
            "query": query,
            "code": resolve_positive_code(group, codebase_codes),
            "code_index": (group.get("positive") or {}).get("code_index"),
            "label": 1,
            "source": group.get("mining", "grouped"),
            "retrieval_rank": (group.get("positive") or {}).get("retrieval_rank"),
            "retrieval_score": (group.get("positive") or {}).get("retrieval_score"),
        }
    ]

    negative_indices = group.get("negative_code_indices") or []
    negative_ranks = group.get("negative_retrieval_ranks") or []
    negative_scores = group.get("negative_retrieval_scores") or []
    if max_negatives > 0:
        negative_indices = negative_indices[:max_negatives]

    for pos, code_index in enumerate(negative_indices):
        code_index = int(code_index)
        rows.append(
            {
                "query_id": query_id,
                "language": language,
                "query": query,
                "code": codebase_codes[code_index],
                "code_index": code_index,
                "label": 0,
                "source": group.get("mining", "grouped"),
                "retrieval_rank": negative_ranks[pos] if pos < len(negative_ranks) else pos + 1,
                "retrieval_score": negative_scores[pos] if pos < len(negative_scores) else None,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Materialize grouped code_index rerank data into inline-code pair rows."
    )
    parser.add_argument("--input-files", type=str, nargs="+", required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--max-negatives-per-query", type=int, default=31)
    args = parser.parse_args()

    codebase_cache = CodebaseCache(Path(args.data_root))
    output_rows = []
    skipped = 0
    groups = 0
    for input_file in args.input_files:
        for group in read_jsonl(Path(input_file)):
            groups += 1
            try:
                rows = materialize_group(
                    group,
                    codebase_cache=codebase_cache,
                    max_negatives=args.max_negatives_per_query,
                )
            except Exception as exc:
                skipped += 1
                print(f"[WARN] Skipping group {group.get('query_id')}: {exc}")
                continue
            output_rows.extend(rows)

    saved = write_jsonl(Path(args.output_file), output_rows)
    print(
        json.dumps(
            {
                "input_groups": groups,
                "skipped_groups": skipped,
                "saved_rows": saved,
                "output_file": args.output_file,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
