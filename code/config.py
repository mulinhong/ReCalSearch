"""Centralized experiment configuration for text2code retrieval."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data" / "CSN"
RESULTS_DIR = PROJECT_ROOT / "results"

LOCAL_MODEL_ROOT = Path(os.environ.get("HF_MODEL_ROOT", "/root/autodl-tmp/hf_models"))
MODEL_DIR_MAP = {
    "microsoft/codebert-base": LOCAL_MODEL_ROOT / "codebert-base",
    "microsoft/graphcodebert-base": LOCAL_MODEL_ROOT / "graphcodebert-base",
    "microsoft/unixcoder-base": LOCAL_MODEL_ROOT / "unixcoder-base",
}
LANGUAGES = ["ruby", "go", "php", "python", "java", "javascript"]


def resolve_model_path(model_name: str) -> str:
    local_path = MODEL_DIR_MAP.get(model_name)
    if local_path is not None and local_path.exists():
        return str(local_path)
    return model_name


@dataclass
class ExperimentConfig:
    # --- model ---
    model_name: str = "microsoft/unixcoder-base"
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["query", "value"])
    init_adapter: Optional[str] = None

    # --- training ---
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 1
    learning_rate: float = 5e-5
    warmup_steps: int = 1000
    logging_steps: int = 200
    save_steps: int = 10000
    save_total_limit: int = 3
    seed: int = 42
    fp16: bool = True
    gradient_checkpointing: bool = False
    resume_from_checkpoint: Optional[str] = None

    # --- data ---
    languages: List[str] = field(default_factory=lambda: LANGUAGES.copy())
    data_root: str = str(DATA_ROOT)
    max_length: int = 512
    language_specific: bool = False
    query_prefix: str = ""
    code_prefix: str = ""

    # --- hard negatives ---
    negative_mode: str = "inbatch"  # inbatch | bm25 | dense | hybrid
    num_hard_negatives: int = 5
    lambda_hn: float = 0.25
    warmup_hn_ratio: float = 0.0
    hn_loss_mode: str = "separate"  # separate | unified
    hard_neg_file_suffix: str = ""

    # --- false-negative filtering ---
    use_fn_filter: bool = False
    fn_filter_token_overlap: float = 0.8
    fn_filter_min_edit_distance: int = 10

    # --- dual-view ---
    use_dual_view: bool = False
    lambda_norm: float = 1.0
    lambda_consistency: float = 0.5

    # --- pooling ---
    pooling_mode: str = "mean"  # mean | last_token | attention | query_aware
    pooling_hidden_dim: int = 768

    # --- temperature ---
    temperature: float = 0.07

    # --- output ---
    experiment_name: str = "baseline"
    output_dir: Optional[str] = None
    adapter_save_dir: Optional[str] = None

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = str(SCRIPT_DIR / "runs" / self.experiment_name)
        if self.adapter_save_dir is None:
            self.adapter_save_dir = str(SCRIPT_DIR / "trained_adapters" / self.experiment_name)

    def build_tag(self) -> str:
        parts = [
            self.experiment_name,
            f"r{self.lora_r}",
            f"seed{self.seed}",
            f"neg_{self.negative_mode}",
            f"hn{self.num_hard_negatives}",
            f"pool_{self.pooling_mode}",
            f"temp{self.temperature:g}",
        ]
        if self.use_dual_view:
            parts.append("dualview")
        if self.use_fn_filter:
            parts.append("fnfilter")
        if self.hard_neg_file_suffix:
            parts.append(f"hnsrc_{self.hard_neg_file_suffix}")
        if self.query_prefix:
            parts.append("qprefix")
        if self.code_prefix:
            parts.append("cprefix")
        if self.init_adapter:
            parts.append("warmstart")
        return "_".join(parts)