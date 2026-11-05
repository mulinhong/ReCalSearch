import importlib.util
import hashlib
import sys
from pathlib import Path

ROOT = Path(
    "/root/autodl-tmp/lora-code-embeddings-main/text2code"
)

ORIGINAL = ROOT / "train_qwen_verifier.py"

EXPECTED_SHA = (
    "ec0a1307a2b1d5e2a283fa3527582b487761e388fbda26fa6a1b5b758d3e2c07"
)

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1048576),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()

def require(ok, message):
    if not ok:
        raise RuntimeError("FAIL_STOP: " + message)

require(
    ORIGINAL.is_file(),
    "Frozen original training script missing",
)

require(
    sha256(ORIGINAL) == EXPECTED_SHA,
    "Frozen original training script SHA256 mismatch",
)

# Import the original module without executing main().
spec = importlib.util.spec_from_file_location(
    "r13_frozen_qwen_trainer",
    ORIGINAL,
)

require(
    spec is not None and spec.loader is not None,
    "Unable to load frozen training module",
)

module = importlib.util.module_from_spec(spec)

sys.modules[spec.name] = module
spec.loader.exec_module(module)

require(
    hasattr(module, "TrainingArguments"),
    "Original module does not expose TrainingArguments",
)

require(
    hasattr(module, "main")
    and callable(module.main),
    "Original training main() missing",
)

OriginalTrainingArguments = module.TrainingArguments

class FrozenHistoricalTrainingArguments(
    OriginalTrainingArguments
):
    def __init__(self, *args, **kwargs):

        # Override keyword values at the constructor
        # boundary, whether or not the original source
        # explicitly passed these two parameters.
        kwargs["optim"] = "adamw_torch"
        kwargs["data_seed"] = None

        super().__init__(*args, **kwargs)

        actual_optim = str(
            getattr(self.optim, "value", self.optim)
        )

        require(
            actual_optim == "adamw_torch",
            "Actual optimizer differs from historical",
        )

        require(
            self.data_seed is None,
            "Actual data_seed differs from historical",
        )

        require(
            int(self.gradient_accumulation_steps) == 16,
            "Gradient accumulation mismatch",
        )

        require(
            self.per_device_train_batch_size == 1,
            "Micro batch size mismatch",
        )

        require(
            abs(float(self.learning_rate) - 1e-4)
            < 1e-12,
            "Learning rate mismatch",
        )

        require(
            self.lr_scheduler_type.value == "linear",
            "LR scheduler mismatch",
        )

        require(
            abs(float(self.weight_decay)) < 1e-12,
            "Weight decay mismatch",
        )

        print("=" * 85, flush=True)
        print(
            "R13_RUNTIME_OPTIMIZER=adamw_torch",
            flush=True,
        )
        print(
            "R13_RUNTIME_DATA_SEED=None",
            flush=True,
        )
        print(
            "R13_RUNTIME_TRAINING_ARGUMENTS=PASS",
            flush=True,
        )
        print("=" * 85, flush=True)

# Replace only the module's TrainingArguments
# class reference in memory.
module.TrainingArguments = (
    FrozenHistoricalTrainingArguments
)

if __name__ == "__main__":
    module.main()
