import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from revision_utils import write_environment_manifest


class SingleGroupTrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is None:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(
                [{"params": trainable_params, "weight_decay": self.args.weight_decay}],
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        return self.optimizer


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def target_text(row: Dict[str, Any]) -> str:
    if "target" in row and row["target"]:
        return str(row["target"])
    label = int(row["label"])
    return " Yes" if label == 1 else " No"


class VerifierDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int, max_samples: int = None, seed: int = 42):
        rows = read_jsonl(Path(path))
        if max_samples is not None and max_samples > 0 and max_samples < len(rows):
            rng = random.Random(seed)
            rows = rng.sample(rows, max_samples)
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        prompt = str(row["prompt"])
        answer = target_text(row)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        answer_ids = self.tokenizer(answer, add_special_tokens=False).input_ids
        if not answer_ids:
            raise ValueError(f"Empty answer tokenization for row {idx}: {answer!r}")

        max_prompt_len = max(1, self.max_length - len(answer_ids))
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class VerifierCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_len = ((max_len + multiple - 1) // multiple) * multiple

        input_ids = []
        attention_mask = []
        labels = []
        pad_id = self.tokenizer.pad_token_id
        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def dtype_from_name(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


def cast_trainable_parameters(model, dtype_name: str):
    dtype = dtype_from_name(dtype_name)
    cast_count = 0
    for param in model.parameters():
        if param.requires_grad and param.dtype != dtype:
            param.data = param.data.to(dtype=dtype)
            cast_count += 1
    if cast_count:
        print(f"Cast {cast_count} trainable parameter tensors to {dtype_name}")


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"

    load_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": dtype_from_name(args.dtype),
        "low_cpu_mem_usage": True,
        "use_safetensors": True,
    }
    if args.device_map == "cuda":
        load_kwargs["device_map"] = {"": 0}
    elif args.device_map == "auto":
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    if args.device_map == "none":
        model = model.cuda()

    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, lora_config)
    cast_trainable_parameters(model, args.trainable_dtype)
    model.print_trainable_parameters()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train a Qwen LoRA Yes/No verifier for code search reranking.")
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-name-or-path", type=str, default="/root/autodl-tmp/hf_models/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--trainable-dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="Dtype for LoRA trainable parameters. Keep float32 with fp16 base models to avoid GradScaler FP16-gradient errors.",
    )
    parser.add_argument("--device-map", type=str, default="cuda", choices=["cuda", "auto", "none"])

    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_torch",
        choices=["adamw_torch", "adamw_torch_fused"],
        help="Optimizer implementation. R1-1 reviewer controls explicitly use historical adamw_torch.",
    )
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", help="Request deterministic kernels where possible; may reduce speed.")
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
    parser.add_argument(
        "--single-optimizer-group",
        action="store_true",
        help="Put all LoRA trainable parameters in one AdamW parameter group. Useful with DeepSpeed on older Trainer stacks.",
    )

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen verifier training.")

    model, tokenizer = load_model_and_tokenizer(args)
    dataset = VerifierDataset(
        args.train_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    print(f"Loaded {len(dataset)} verifier examples from {args.train_file}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        optim=args.optim,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        fp16=args.dtype == "float16",
        bf16=args.dtype == "bfloat16",
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
        optim="adamw_torch",
        seed=args.seed,
    )

    trainer_cls = SingleGroupTrainer if args.single_optimizer_group else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=VerifierCollator(tokenizer),
    )
    trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    with (output_dir / "verifier_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model": args.model_name_or_path,
                "train_file": args.train_file,
                "loss_type": "causal_lm_yes_no_completion",
                "max_length": args.max_length,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "target_modules": args.target_modules,
                "trainable_dtype": args.trainable_dtype,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "seed": args.seed,
                "deterministic": args.deterministic,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "warmup_ratio": args.warmup_ratio,
                "optim": "adamw_torch",
                "optim": args.optim,
                "max_train_samples": args.max_train_samples,
                "num_training_examples": len(dataset),
            },
            f,
            indent=2,
        )
    write_environment_manifest(
        output_dir / "environment_manifest.json",
        extra={
            "role": "qwen_verifier_training",
            "seed": args.seed,
            "train_file": str(Path(args.train_file).resolve()),
            "output_dir": str(output_dir.resolve()),
            "base_model": args.model_name_or_path,
            "num_training_examples": len(dataset),
        },
    )
    print(f"Verifier adapter saved to: {output_dir}")


if __name__ == "__main__":
    main()
