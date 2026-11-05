import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from config import RESULTS_DIR


METRIC_FIELDS = ["r1", "r5", "r10", "mrr", "ndcg@10"]


def parse_labeled_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, path = value.split("=", 1)
    return label, Path(path)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_average_rows(label: str, path: Path) -> Iterable[Dict[str, Any]]:
    data = load_json(path)
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        avg = value.get("average")
        if not isinstance(avg, dict):
            continue
        row = {"source": label, "setting": key, "path": str(path)}
        for field in METRIC_FIELDS:
            row[field] = avg.get(field)
        yield row


def collect_default_json_inputs() -> List[Tuple[str, Path]]:
    candidates = [
        ("random-negative Qwen verifier", RESULTS_DIR / "oldcsn_random_negative_qwen_verify15_alpha_sweep_eval_results.json"),
        ("UniXcoder cross-encoder reranker", RESULTS_DIR / "oldcsn_cross_encoder_unixcoder_hn31_rerank100_eval_results.json"),
        ("Qwen zero-shot listwise", RESULTS_DIR / "oldcsn_qwen25coder7b_zeroshot_listwise_verify5_eval_results.json"),
    ]
    return [(label, path) for label, path in candidates if path.exists()]


def collect_default_runtime_inputs() -> List[Tuple[str, Path]]:
    runtime_dir = Path("results/jss_rescue_oldcsn/runtime")
    if not runtime_dir.exists():
        return []
    return [(path.stem, path) for path in sorted(runtime_dir.glob("*.csv"))]


def read_runtime_csv(label: str, path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = dict(row)
            row["source"] = label
            row["path"] = str(path)
            yield row


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Summarize JSS rescue experiment result JSON/CSV files.")
    parser.add_argument(
        "--json-result",
        action="append",
        default=[],
        help="Label and JSON path as LABEL=PATH. May be repeated.",
    )
    parser.add_argument(
        "--runtime-csv",
        action="append",
        default=[],
        help="Label and runtime CSV path as LABEL=PATH. May be repeated.",
    )
    parser.add_argument("--output-dir", type=str, default="results/jss_rescue_oldcsn")
    args = parser.parse_args()

    json_inputs = [parse_labeled_path(value) for value in args.json_result]
    if not json_inputs:
        json_inputs = collect_default_json_inputs()

    metric_rows: List[Dict[str, Any]] = []
    for label, path in json_inputs:
        if not path.exists():
            print(f"[WARN] Missing JSON result: {path}")
            continue
        metric_rows.extend(iter_average_rows(label, path))

    output_dir = Path(args.output_dir)
    if metric_rows:
        write_csv(
            output_dir / "jss_rescue_metric_summary.csv",
            metric_rows,
            ["source", "setting", *METRIC_FIELDS, "path"],
        )
        for row in metric_rows:
            print(row)
    else:
        print("[WARN] No metric JSON rows found.")

    runtime_inputs = [parse_labeled_path(value) for value in args.runtime_csv]
    if not runtime_inputs:
        runtime_inputs = collect_default_runtime_inputs()

    runtime_rows: List[Dict[str, Any]] = []
    for label, path in runtime_inputs:
        if not path.exists():
            print(f"[WARN] Missing runtime CSV: {path}")
            continue
        runtime_rows.extend(read_runtime_csv(label, path))

    if runtime_rows:
        fieldnames = ["source", *[k for k in runtime_rows[0].keys() if k != "source"]]
        write_csv(output_dir / "jss_rescue_runtime_summary.csv", runtime_rows, fieldnames)
    else:
        print("[WARN] No runtime CSV rows found.")


if __name__ == "__main__":
    main()
