import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    RobertaModel,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
)

from config import DATA_ROOT


def resolve_model_path(model_name: str) -> str:
    local_root = Path("/root/autodl-tmp/hf_models")
    model_dir_map = {
        "microsoft/codebert-base": local_root / "codebert-base",
        "microsoft/graphcodebert-base": local_root / "graphcodebert-base",
        "microsoft/unixcoder-base": local_root / "unixcoder-base",
    }
    local_path = model_dir_map.get(model_name)
    if local_path is not None and local_path.exists():
        return str(local_path)
    return model_name


def get_code_text(example: Dict[str, Any]) -> str:
    if "code" in example and example["code"]:
        return str(example["code"])
    if "code_tokens" in example and example["code_tokens"]:
        value = example["code_tokens"]
        if isinstance(value, list):
            return " ".join(map(str, value))
        return str(value)
    if "function_tokens" in example and example["function_tokens"]:
        value = example["function_tokens"]
        if isinstance(value, list):
            return " ".join(map(str, value))
        return str(value)
    raise KeyError(f"Could not find code field in example keys: {list(example.keys())}")


def read_codebase_codes(path: Path) -> List[str]:
    codes = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                codes.append(get_code_text(json.loads(line)))
    return codes


class RerankDataset(Dataset):
    def __init__(
        self,
        path: str,
        codebase_codes: Optional[List[str]] = None,
        max_negatives_per_query: Optional[int] = 31,
        hard_keep: int = 8,
        seed: int = 42,
    ):
        self.codebase_codes = codebase_codes
        self.max_negatives_per_query = max_negatives_per_query
        self.hard_keep = hard_keep
        self.seed = seed

        rows: List[Dict] = []
        groups: List[Dict] = []
        compact_format: Optional[bool] = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    if compact_format is None:
                        compact_format = "negative_code_indices" in row or row.get("format") == "grouped_indices_v1"
                    if compact_format:
                        group = self._make_index_group(row)
                        if group is not None:
                            groups.append(group)
                    else:
                        rows.append(row)

        self.groups = groups if compact_format else self._build_groups(rows)
        if not self.groups:
            raise ValueError(f"No valid query groups found in {path}")

    @staticmethod
    def _sort_key(row: Dict):
        rank = row.get("retrieval_rank")
        if rank is None:
            rank = 10**9
        return int(rank)

    def _resolve_code(self, row: Dict) -> str:
        code = row.get("code")
        if code:
            return str(code)

        code_index = row.get("code_index")
        if code_index is None:
            raise KeyError(f"Candidate has neither code nor code_index: {row}")
        if self.codebase_codes is None:
            raise ValueError("This reranker file stores code_index values; pass --codebase-file or --language/--data-root.")
        return self.codebase_codes[int(code_index)]

    def _make_index_group(self, row: Dict) -> Optional[Dict]:
        positive = row.get("positive", {})
        negative_indices = row.get("negative_code_indices", [])
        if not negative_indices:
            return None

        negative_teacher_scores = row.get("negative_teacher_scores", [])
        positive_teacher_score = row.get("positive_teacher_score")
        if positive_teacher_score is None:
            positive_teacher_score = positive.get("teacher_score")

        return {
            "query_id": row.get("query_id", str(len(negative_indices))),
            "query": row["query"],
            "positive": positive,
            "positive_teacher_score": positive_teacher_score,
            "negative_code_indices": [int(x) for x in negative_indices],
            "negative_retrieval_ranks": [int(x) for x in row.get("negative_retrieval_ranks", [])],
            "negative_retrieval_scores": [
                float(x) if x is not None else None
                for x in row.get("negative_retrieval_scores", [])
            ],
            "negative_teacher_scores": [float(x) for x in negative_teacher_scores],
        }

    def _make_group(self, rows: List[Dict]) -> Optional[Dict]:
        positives = [r for r in rows if int(r.get("label", 0)) == 1]
        if not positives:
            return None

        positive = positives[0]
        negatives = [r for r in rows if int(r.get("label", 0)) == 0]
        if not negatives:
            return None

        negatives.sort(key=self._sort_key)
        return {
            "query_id": positive.get("query_id", str(len(rows))),
            "query": positive["query"],
            "positive": positive,
            "positive_teacher_score": positive.get("teacher_score"),
            "negatives": negatives,
        }

    def _build_groups(self, rows: List[Dict]) -> List[Dict]:
        if rows and any("negative_code_indices" in row or row.get("format") == "grouped_indices_v1" for row in rows[:10]):
            groups = [self._make_index_group(row) for row in rows]
            return [g for g in groups if g is not None]

        if any("query_id" in row for row in rows):
            grouped = {}
            order = []
            for row in rows:
                query_id = row.get("query_id")
                if query_id is None:
                    query_id = f"legacy:{len(order)}"
                if query_id not in grouped:
                    grouped[query_id] = []
                    order.append(query_id)
                grouped[query_id].append(row)
            groups = [self._make_group(grouped[qid]) for qid in order]
            return [g for g in groups if g is not None]

        groups = []
        current = []
        for row in rows:
            if int(row.get("label", 0)) == 1 and current:
                group = self._make_group(current)
                if group is not None:
                    groups.append(group)
                current = []
            current.append(row)

        if current:
            group = self._make_group(current)
            if group is not None:
                groups.append(group)

        return groups

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]
        if "negative_code_indices" in group:
            ranks = group.get("negative_retrieval_ranks", [])
            scores = group.get("negative_retrieval_scores", [])
            teacher_scores = group.get("negative_teacher_scores", [])
            negatives = [
                {
                    "code_index": code_index,
                    "retrieval_rank": ranks[i] if i < len(ranks) else i + 1,
                    "retrieval_score": scores[i] if i < len(scores) else None,
                    "teacher_score": teacher_scores[i] if i < len(teacher_scores) else None,
                }
                for i, code_index in enumerate(group["negative_code_indices"])
            ]
        else:
            negatives = list(group["negatives"])

        if self.max_negatives_per_query is not None and self.max_negatives_per_query > 0:
            max_negs = self.max_negatives_per_query
            if len(negatives) > max_negs:
                hard = negatives[: min(self.hard_keep, max_negs)]
                rest = negatives[len(hard):]
                need = max_negs - len(hard)
                rng = random.Random(self.seed + idx)
                sampled = hard + rng.sample(rest, k=min(need, len(rest)))
                sampled.sort(key=self._sort_key)
                negatives = sampled

        candidates = [
            {
                "code": self._resolve_code(group["positive"]),
                "label": 1,
                "teacher_score": group.get("positive_teacher_score"),
            }
        ]
        candidates.extend({"code": self._resolve_code(neg), "label": 0} for neg in negatives)
        for cand, neg in zip(candidates[1:], negatives):
            cand["teacher_score"] = neg.get("teacher_score")

        return {
            "query_id": group["query_id"],
            "query": group["query"],
            "candidates": candidates,
            "positive_index": 0,
        }


class RerankerModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        score_dim: int = 1,
        distill_weight: float = 0.0,
        distill_temperature: float = 1.0,
    ):
        super().__init__()
        model_path = resolve_model_path(model_name)
        self.encoder = RobertaModel.from_pretrained(model_path)
        self.score_dim = score_dim
        self.distill_weight = float(distill_weight)
        self.distill_temperature = float(distill_temperature)
        target_modules = target_modules or ["query", "value"]

        for param in self.encoder.parameters():
            param.requires_grad = False
            if param.ndim == 1:
                param.data = param.data.to(torch.float32)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            task_type="FEATURE_EXTRACTION",
        )
        self.encoder = get_peft_model(self.encoder, lora_config)

        # 关键：配合 gradient checkpointing，保证输入可求梯度
        if hasattr(self.encoder, "enable_input_require_grads"):
            self.encoder.enable_input_require_grads()

        # 关闭 cache，避免 checkpointing 下的不兼容
        if hasattr(self.encoder, "config"):
            self.encoder.config.use_cache = False

        self.classifier = nn.Linear(self.encoder.config.hidden_size, score_dim)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            if gradient_checkpointing_kwargs is not None:
                self.encoder.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                )
            else:
                self.encoder.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        if hasattr(self.encoder, "gradient_checkpointing_disable"):
            self.encoder.gradient_checkpointing_disable()

    def forward(self, input_ids=None, attention_mask=None, labels=None, group_sizes=None, teacher_scores=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        raw_logits = self.classifier(cls)
        logits = raw_logits.squeeze(-1) if raw_logits.shape[-1] == 1 else raw_logits

        loss = None
        if labels is not None and group_sizes is not None and raw_logits.shape[-1] == 1:
            scores = raw_logits.squeeze(-1)
            offset = 0
            losses = []
            distill_losses = []
            temperature = max(self.distill_temperature, 1e-6)
            for group_idx, group_size in enumerate(group_sizes.tolist()):
                group_scores = scores[offset:offset + group_size].unsqueeze(0)
                group_label = labels[group_idx].view(1)
                losses.append(F.cross_entropy(group_scores, group_label))

                if teacher_scores is not None and self.distill_weight > 0.0:
                    group_teacher = teacher_scores[offset:offset + group_size]
                    if torch.isfinite(group_teacher).all():
                        teacher_probs = F.softmax(group_teacher.unsqueeze(0) / temperature, dim=1)
                        student_log_probs = F.log_softmax(group_scores / temperature, dim=1)
                        distill_losses.append(
                            F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
                        )
                offset += group_size

            loss = torch.stack(losses).mean() if losses else scores.sum() * 0.0
            if distill_losses:
                loss = loss + self.distill_weight * torch.stack(distill_losses).mean()
        elif labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}


def tokenize_pairs(tokenizer, queries, codes, max_length=512, max_query_length=96):
    special_tokens = tokenizer.num_special_tokens_to_add(pair=True)
    query_budget = max(1, min(max_query_length, max_length - special_tokens - 1))
    code_budget = max(1, max_length - query_budget - special_tokens)

    query_enc = tokenizer(
        queries,
        add_special_tokens=False,
        truncation=True,
        max_length=query_budget,
        padding=False,
        return_attention_mask=False,
        verbose=False,
    )
    code_enc = tokenizer(
        codes,
        add_special_tokens=False,
        truncation=True,
        max_length=code_budget,
        padding=False,
        return_attention_mask=False,
        verbose=False,
    )

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    if cls_id is None:
        cls_id = tokenizer.bos_token_id
    if sep_id is None:
        sep_id = tokenizer.eos_token_id
    if cls_id is None or sep_id is None:
        raise ValueError(
            f"Tokenizer {tokenizer.__class__.__name__} is missing cls/sep token ids."
        )

    input_ids = [
        [int(cls_id)] + list(query_ids) + [int(sep_id), int(sep_id)] + list(code_ids) + [int(sep_id)]
        for query_ids, code_ids in zip(query_enc["input_ids"], code_enc["input_ids"])
    ]
    max_batch_length = max(len(ids) for ids in input_ids)
    pad_id = tokenizer.pad_token_id
    padded_ids = []
    attention_mask = []
    for ids in input_ids:
        pad_len = max_batch_length - len(ids)
        padded_ids.append(ids + [pad_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)

    return {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def collate_fn(batch, tokenizer, max_length=512, max_query_length=96):
    queries = []
    codes = []
    group_sizes = []
    labels = []
    teacher_scores = []
    has_teacher_scores = False

    for group in batch:
        candidates = group["candidates"]
        group_sizes.append(len(candidates))
        labels.append(int(group.get("positive_index", 0)))
        for item in candidates:
            queries.append("query: " + group["query"])
            codes.append("code: " + item["code"])
            teacher_score = item.get("teacher_score")
            if teacher_score is None:
                teacher_scores.append(float("nan"))
            else:
                teacher_scores.append(float(teacher_score))
                has_teacher_scores = True

    enc = tokenize_pairs(tokenizer, queries, codes, max_length=max_length, max_query_length=max_query_length)
    enc["group_sizes"] = torch.tensor(group_sizes, dtype=torch.long)
    enc["labels"] = torch.tensor(labels, dtype=torch.long)
    if has_teacher_scores:
        enc["teacher_scores"] = torch.tensor(teacher_scores, dtype=torch.float32)
    return enc


def main():
    parser = argparse.ArgumentParser(description="Train a lightweight reranker for text2code retrieval.")
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="microsoft/unixcoder-base")
    parser.add_argument("--language", type=str, default="python")
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--codebase-file", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-query-length", type=int, default=96)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--target-modules", type=str, nargs="+", default=["query", "value"])
    parser.add_argument("--max-negatives-per-query", type=int, default=31)
    parser.add_argument("--hard-keep", type=int, default=8)
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument("--distill-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to a DeepSpeed config JSON file.")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1, help="Injected by DeepSpeed/torchrun.")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true", default=False)
    checkpoint_group.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_path = resolve_model_path(args.model_name)
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = RerankerModel(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        score_dim=1,
        distill_weight=args.distill_weight,
        distill_temperature=args.distill_temperature,
    )

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    codebase_file = Path(args.codebase_file) if args.codebase_file else Path(args.data_root) / args.language / "codebase.jsonl"
    codebase_codes = None
    if codebase_file.exists():
        print(f"Loading codebase codes from: {codebase_file}")
        codebase_codes = read_codebase_codes(codebase_file)
        print(f"Loaded {len(codebase_codes)} codebase codes")

    max_negatives = args.max_negatives_per_query if args.max_negatives_per_query > 0 else None
    dataset = RerankDataset(
        args.train_file,
        codebase_codes=codebase_codes,
        max_negatives_per_query=max_negatives,
        hard_keep=args.hard_keep,
        seed=args.seed,
    )
    print(f"Loaded {len(dataset)} query groups from {args.train_file}")
    candidates_per_group = 1 + (args.max_negatives_per_query if args.max_negatives_per_query > 0 else 0)
    print(
        "Training batch shape: "
        f"{args.batch_size} query groups/step, about {args.batch_size * candidates_per_group} pairs/forward, "
        f"effective batch {args.batch_size * args.gradient_accumulation_steps} query groups"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda batch: collate_fn(batch, tokenizer, args.max_length, args.max_query_length),
    )

    trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    with (output_dir / "reranker_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "language": args.language,
                "data_root": args.data_root,
                "codebase_file": str(codebase_file),
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "target_modules": args.target_modules,
                "score_dim": 1,
                "loss_type": "groupwise_cross_entropy",
                "max_length": args.max_length,
                "max_query_length": args.max_query_length,
                "max_negatives_per_query": args.max_negatives_per_query,
                "hard_keep": args.hard_keep,
                "distill_weight": args.distill_weight,
                "distill_temperature": args.distill_temperature,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "dataloader_num_workers": args.dataloader_num_workers,
                "deepspeed": args.deepspeed,
                "seed": args.seed,
            },
            f,
            indent=2,
        )

    print(f"Reranker saved to: {output_dir}")


if __name__ == "__main__":
    main()
