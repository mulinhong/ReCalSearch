"""Unified training entry point for debiased text2code retrieval experiments."""

import argparse
import json
import os
import random
import time
import warnings
from bisect import bisect_right
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from peft import LoraConfig, PeftModel
from torch.utils.data import Dataset
from transformers import RobertaModel, RobertaTokenizer, TrainingArguments, TrainerCallback

from config import ExperimentConfig, LANGUAGES, resolve_model_path
from train_utils import DebiasedContrastiveTrainer, build_pooling, collate_fn


def _register_numpy_safe_globals() -> None:
    """Allow PyTorch 2.6+ weights_only loader to deserialize common NumPy RNG objects."""
    try:
        np_core = np._core if hasattr(np, "_core") else np.core
        safe_globals = [
            np_core.multiarray._reconstruct,
            np.ndarray,
            np.dtype,
            type(np.dtype(np.uint32)),
            type(np.dtype(np.int64)),
            type(np.dtype(np.float32)),
        ]
        torch.serialization.add_safe_globals(safe_globals)
    except Exception as exc:
        warnings.warn(f"Could not register torch safe globals for NumPy RNG state: {exc}")


class SaveExtraArtifactsCallback(TrainerCallback):
    """Save pooling artifacts into every checkpoint so intermediate checkpoints are directly evaluable."""

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        model = kwargs.get("model", None)
        if model is None or not checkpoint_dir.exists():
            return control

        if hasattr(model, "pooling_head"):
            torch.save(model.pooling_head.state_dict(), checkpoint_dir / "pooling_head.pt")

        with open(checkpoint_dir / "pooling_config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "pooling_mode": self.cfg.pooling_mode,
                    "pooling_hidden_dim": self.cfg.pooling_hidden_dim,
                    "max_length": self.cfg.max_length,
                    "query_prefix": self.cfg.query_prefix,
                    "code_prefix": self.cfg.code_prefix,
                },
                f,
                indent=2,
            )

        with open(checkpoint_dir / "experiment_meta.txt", "w", encoding="utf-8") as f:
            f.write(f"experiment: {self.cfg.experiment_name}\n")
            f.write(f"model: {self.cfg.model_name}\n")
            f.write(f"seed: {self.cfg.seed}\n")
            f.write(f"negative_mode: {self.cfg.negative_mode}\n")
            f.write(f"dual_view: {self.cfg.use_dual_view}\n")
            f.write(f"fn_filter: {self.cfg.use_fn_filter}\n")
            f.write(f"hard_neg_file_suffix: {self.cfg.hard_neg_file_suffix}\n")
            f.write(f"pooling: {self.cfg.pooling_mode}\n")
            f.write(f"temperature: {self.cfg.temperature}\n")
            f.write(f"init_adapter: {self.cfg.init_adapter}\n")
            f.write(f"hn_loss_mode: {self.cfg.hn_loss_mode}\n")
            f.write(f"lambda_hn: {self.cfg.lambda_hn}\n")
            f.write(f"warmup_hn_ratio: {self.cfg.warmup_hn_ratio}\n")
            f.write(f"lora_r: {self.cfg.lora_r}\n")
            f.write(f"train_batch: {self.cfg.per_device_train_batch_size}\n")
            f.write(f"grad_accum: {self.cfg.gradient_accumulation_steps}\n")
            f.write(f"gradient_checkpointing: {self.cfg.gradient_checkpointing}\n")
            f.write(f"query_prefix: {self.cfg.query_prefix!r}\n")
            f.write(f"code_prefix: {self.cfg.code_prefix!r}\n")

        return control


class SafeResumeTrainer(DebiasedContrastiveTrainer):
    """Trainer with PyTorch 2.6+ compatible RNG-state resume."""

    def _load_rng_state(self, checkpoint: str) -> None:
        rng_file = os.path.join(checkpoint, "rng_state.pth")
        if not os.path.isfile(rng_file):
            return

        checkpoint_rng_state = None

        try:
            checkpoint_rng_state = torch.load(rng_file, map_location="cpu")
        except Exception as exc:
            warnings.warn(
                "Safe RNG-state load failed. Retrying with weights_only=False for a trusted local checkpoint. "
                f"Original error: {exc}"
            )
            try:
                checkpoint_rng_state = torch.load(rng_file, map_location="cpu", weights_only=False)
            except Exception as exc2:
                warnings.warn(
                    "Failed to restore RNG state even with fallback; training will resume without exact RNG "
                    f"continuity. Error: {exc2}"
                )
                return

        if "python" in checkpoint_rng_state:
            random.setstate(checkpoint_rng_state["python"])
        if "numpy" in checkpoint_rng_state:
            np.random.set_state(checkpoint_rng_state["numpy"])
        if "cpu" in checkpoint_rng_state:
            torch.random.set_rng_state(checkpoint_rng_state["cpu"])

        if torch.cuda.is_available() and "cuda" in checkpoint_rng_state:
            try:
                torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])
            except Exception:
                try:
                    torch.cuda.random.set_rng_state(checkpoint_rng_state["cuda"])
                except Exception as exc:
                    warnings.warn(f"Could not restore CUDA RNG state exactly: {exc}")


class JsonlLazyDataset(Dataset):
    def __init__(self, path: Path):
        self.path = Path(path)
        self.offsets = self._build_offsets(self.path)
        self._fp = None

    @staticmethod
    def _build_offsets(path: Path) -> List[int]:
        offsets = []
        with path.open("rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)
        return offsets

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fp"] = None
        return state

    def _file(self):
        if self._fp is None:
            self._fp = self.path.open("rb")
        return self._fp

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        f = self._file()
        f.seek(self.offsets[idx])
        return json.loads(f.readline().decode("utf-8"))


class MultiJsonlLazyDataset(Dataset):
    def __init__(self, paths: Sequence[Path]):
        self.datasets = [JsonlLazyDataset(path) for path in paths]
        self.cumulative = []
        total = 0
        for ds in self.datasets:
            total += len(ds)
            self.cumulative.append(total)

    def __len__(self) -> int:
        return self.cumulative[-1] if self.cumulative else 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ds_idx = bisect_right(self.cumulative, idx)
        prev = self.cumulative[ds_idx - 1] if ds_idx > 0 else 0
        return self.datasets[ds_idx][idx - prev]


def resolve_split_path(
    data_root: str,
    lang: str,
    split: str,
    negative_mode: str,
    use_fn_filter: bool = False,
    hard_neg_file_suffix: str = "",
) -> Path:
    root = Path(data_root) / lang

    if negative_mode != "inbatch" and split == "train":
        suffixes_to_try: List[str] = []
        if hard_neg_file_suffix:
            if use_fn_filter:
                suffixes_to_try.append(f"train_{negative_mode}_filtered_{hard_neg_file_suffix}.jsonl")
            suffixes_to_try.append(f"train_{negative_mode}_{hard_neg_file_suffix}.jsonl")
        else:
            if use_fn_filter:
                suffixes_to_try.append(f"train_{negative_mode}_filtered.jsonl")
            suffixes_to_try.append(f"train_{negative_mode}.jsonl")

        for suffix in suffixes_to_try:
            candidate = root / suffix
            if candidate.exists():
                return candidate

        attempted = ", ".join(str(root / suffix) for suffix in suffixes_to_try)
        raise FileNotFoundError(
            f"No hard-negative train file found for {lang}/{negative_mode}. "
            f"Attempted: {attempted}. Run mine_hard_negatives.py first or use --negative-mode inbatch."
        )

    path = root / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {path}")
    return path


def get_dataset(
    data_root: str,
    languages: List[str],
    split: str,
    negative_mode: str = "inbatch",
    use_fn_filter: bool = False,
    hard_neg_file_suffix: str = "",
) -> MultiJsonlLazyDataset:
    paths = []
    for lang in languages:
        data_path = resolve_split_path(data_root, lang, split, negative_mode, use_fn_filter, hard_neg_file_suffix)
        print(f"  Indexing {data_path}")
        paths.append(data_path)
    return MultiJsonlLazyDataset(paths)


def get_model(cfg: ExperimentConfig):
    model_path = resolve_model_path(cfg.model_name)
    print(f"Loading model from: {model_path}")

    model = RobertaModel.from_pretrained(model_path)
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg.init_adapter:
        for param in model.parameters():
            param.requires_grad = False
            if param.ndim == 1:
                param.data = param.data.to(torch.float32)

        print(f"Loading trainable init adapter from: {cfg.init_adapter}")
        model = PeftModel.from_pretrained(model, cfg.init_adapter, is_trainable=True)
        if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        elif hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        model.config.use_cache = False
        model.config.chunk_size_feed_forward = 0
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        return model, tokenizer

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules,
        lora_dropout=cfg.lora_dropout,
        task_type="FEATURE_EXTRACTION",
    )

    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = False
    model.config.chunk_size_feed_forward = 0

    adapter_name = cfg.experiment_name
    model.add_adapter(lora_config, adapter_name=adapter_name)
    model.set_adapter(adapter_name)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    return model, tokenizer


def print_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable} || all params: {total} || trainable%: {100 * trainable / total:.4f}")


def run_training(cfg: ExperimentConfig):
    print(f"\n{'=' * 60}")
    print(f"Experiment: {cfg.experiment_name}")
    print(f"  model:          {cfg.model_name}")
    print(f"  seed:           {cfg.seed}")
    print(f"  negatives:      {cfg.negative_mode}")
    print(f"  dual-view:      {cfg.use_dual_view}")
    print(f"  fn-filter:      {cfg.use_fn_filter}")
    print(f"  hn file suffix: {cfg.hard_neg_file_suffix}")
    print(f"  pooling:        {cfg.pooling_mode}")
    print(f"  temperature:    {cfg.temperature}")
    print(f"  init adapter:   {cfg.init_adapter}")
    print(f"  hn loss mode:   {cfg.hn_loss_mode}")
    print(f"  train batch:    {cfg.per_device_train_batch_size}")
    print(f"  grad accum:     {cfg.gradient_accumulation_steps}")
    print(f"  grad ckpt:      {cfg.gradient_checkpointing}")
    print(f"{'=' * 60}\n")

    _register_numpy_safe_globals()

    if cfg.pooling_mode == "query_aware":
        print(
            "[WARN] query_aware pooling is query-dependent and is not compatible "
            "with the standard precomputed-code retrieval evaluator."
        )

    model, tokenizer = get_model(cfg)

    num_hn = cfg.num_hard_negatives if cfg.negative_mode != "inbatch" else 0

    model.pooling_head = build_pooling(cfg.pooling_mode, cfg.pooling_hidden_dim)
    if torch.cuda.is_available():
        model.pooling_head = model.pooling_head.cuda()
    for p in model.pooling_head.parameters():
        p.requires_grad = True
    print_trainable_parameters(model)

    print("\nLoading training data lazily...")
    train_dataset = get_dataset(
        cfg.data_root,
        cfg.languages,
        "train",
        cfg.negative_mode,
        cfg.use_fn_filter,
        cfg.hard_neg_file_suffix,
    )
    print(f"  train size: {len(train_dataset)}")

    print("Loading validation data lazily...")
    val_dataset = get_dataset(cfg.data_root, cfg.languages, "valid", "inbatch")
    print(f"  valid size: {len(val_dataset)}")

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        warmup_steps=cfg.warmup_steps,
        logging_steps=cfg.logging_steps,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=cfg.gradient_checkpointing,
        fp16=cfg.fp16,
        seed=cfg.seed,
        data_seed=cfg.seed,
        dataloader_num_workers=0,
    )

    collate = partial(
        collate_fn,
        tokenizer=tokenizer,
        use_dual_view=cfg.use_dual_view,
        num_hard_negatives=num_hn,
        max_length=cfg.max_length,
        query_prefix=cfg.query_prefix,
        code_prefix=cfg.code_prefix,
    )

    trainer = SafeResumeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate,
        pooling_module=model.pooling_head,
        temperature=cfg.temperature,
        num_hard_negatives=num_hn,
        lambda_hn=cfg.lambda_hn,
        use_dual_view=cfg.use_dual_view,
        lambda_norm=cfg.lambda_norm,
        lambda_consistency=cfg.lambda_consistency,
        warmup_hn_ratio=cfg.warmup_hn_ratio,
        hn_loss_mode=cfg.hn_loss_mode,
    )
    trainer.add_callback(SaveExtraArtifactsCallback(cfg))

    start_time = time.time()
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s ({elapsed / 60:.1f}min)")

    save_dir = Path(cfg.adapter_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    if hasattr(model, "pooling_head"):
        torch.save(model.pooling_head.state_dict(), save_dir / "pooling_head.pt")
    tokenizer.save_pretrained(save_dir)

    with open(save_dir / "pooling_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "pooling_mode": cfg.pooling_mode,
                "pooling_hidden_dim": cfg.pooling_hidden_dim,
                "max_length": cfg.max_length,
                "query_prefix": cfg.query_prefix,
                "code_prefix": cfg.code_prefix,
            },
            f,
            indent=2,
        )
    print(f"Adapter saved to: {save_dir}")

    meta_path = save_dir / "experiment_meta.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"experiment: {cfg.experiment_name}\n")
        f.write(f"model: {cfg.model_name}\n")
        f.write(f"seed: {cfg.seed}\n")
        f.write(f"negative_mode: {cfg.negative_mode}\n")
        f.write(f"dual_view: {cfg.use_dual_view}\n")
        f.write(f"fn_filter: {cfg.use_fn_filter}\n")
        f.write(f"hard_neg_file_suffix: {cfg.hard_neg_file_suffix}\n")
        f.write(f"pooling: {cfg.pooling_mode}\n")
        f.write(f"temperature: {cfg.temperature}\n")
        f.write(f"init_adapter: {cfg.init_adapter}\n")
        f.write(f"hn_loss_mode: {cfg.hn_loss_mode}\n")
        f.write(f"lambda_hn: {cfg.lambda_hn}\n")
        f.write(f"warmup_hn_ratio: {cfg.warmup_hn_ratio}\n")
        f.write(f"lora_r: {cfg.lora_r}\n")
        f.write(f"train_batch: {cfg.per_device_train_batch_size}\n")
        f.write(f"grad_accum: {cfg.gradient_accumulation_steps}\n")
        f.write(f"gradient_checkpointing: {cfg.gradient_checkpointing}\n")
        f.write(f"query_prefix: {cfg.query_prefix!r}\n")
        f.write(f"code_prefix: {cfg.code_prefix!r}\n")
        f.write(f"train_time_seconds: {elapsed:.1f}\n")
        f.write(f"trainable_params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")
        f.write(f"total_params: {sum(p.numel() for p in model.parameters())}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Debiased text2code retrieval training")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name")
    parser.add_argument("--model-name", type=str, default="microsoft/unixcoder-base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--languages", type=str, nargs="+", default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--negative-mode", type=str, default="inbatch", choices=["inbatch", "bm25", "dense", "hybrid"])
    parser.add_argument("--num-hard-negatives", type=int, default=5)
    parser.add_argument("--lambda-hn", type=float, default=0.25)
    parser.add_argument("--warmup-hn-ratio", type=float, default=0.0)
    parser.add_argument("--hn-loss-mode", type=str, default="separate", choices=["separate", "unified"])
    parser.add_argument("--init-adapter", type=str, default=None)
    parser.add_argument("--use-fn-filter", action="store_true")
    parser.add_argument("--hard-neg-file-suffix", type=str, default="")
    parser.add_argument("--use-dual-view", action="store_true")
    parser.add_argument("--lambda-norm", type=float, default=1.0)
    parser.add_argument("--lambda-consistency", type=float, default=0.5)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "last_token", "attention", "query_aware"])
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--query-prefix", type=str, default="")
    parser.add_argument("--code-prefix", type=str, default="")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=10000)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=None)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    batch_size = args.batch_size if args.batch_size is not None else 16
    grad_accum = args.grad_accum if args.grad_accum is not None else 1
    gradient_checkpointing = args.gradient_checkpointing
    if gradient_checkpointing is None:
        gradient_checkpointing = False

    cfg = ExperimentConfig(
        experiment_name=f"{args.experiment}_seed{args.seed}",
        model_name=args.model_name,
        seed=args.seed,
        languages=args.languages or LANGUAGES,
        data_root=args.data_root or str(ExperimentConfig().data_root),
        negative_mode=args.negative_mode,
        num_hard_negatives=args.num_hard_negatives,
        lambda_hn=args.lambda_hn,
        warmup_hn_ratio=args.warmup_hn_ratio,
        hn_loss_mode=args.hn_loss_mode,
        init_adapter=args.init_adapter,
        use_fn_filter=args.use_fn_filter,
        hard_neg_file_suffix=args.hard_neg_file_suffix,
        use_dual_view=args.use_dual_view,
        lambda_norm=args.lambda_norm,
        lambda_consistency=args.lambda_consistency,
        pooling_mode=args.pooling,
        temperature=args.temperature,
        query_prefix=args.query_prefix,
        code_prefix=args.code_prefix,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        resume_from_checkpoint=args.resume_from_checkpoint,
        fp16=args.fp16,
        gradient_checkpointing=gradient_checkpointing,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()