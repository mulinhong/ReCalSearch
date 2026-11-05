from pathlib import Path
import hashlib
import importlib.util
import json
import ast
import sys

import torch
from safetensors import safe_open

ROOT = Path(
    "/root/autodl-tmp/lora-code-embeddings-main/text2code"
)

BASE = ROOT / (
    "results/reviewer_revision/"
    "r1_3_multilingual_matched_v1"
)

RUN = BASE / "train_multi5k_seed13_v5"
MODEL = BASE / "models/multi5k_seed13_v5"
OUT = BASE / "posttrain_closure_seed13_v1"

def gate(ok, msg):
    if not ok:
        raise RuntimeError("FAIL_STOP: " + msg)

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1048576),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

print("=" * 95)
print("WORK1 R1-3 MULTI5K SEED13 POSTTRAIN CLOSURE V1")
print("=" * 95)
print("READ_ONLY_MODEL = TRUE")
print("RETRAINING = FALSE")
print("GPU_INFERENCE = FALSE")

print()
print("[1] SOURCE PROVENANCE")

original = ROOT / "train_qwen_verifier.py"
data = BASE / "multi5k_nomic_seed20261013.jsonl"
wrapper = RUN / "run_frozen_trainer.py"
preflight = RUN / "preflight_manifest.json"
log = RUN / "train_seed13.log"
exit_file = RUN / "exit_code.txt"

expected_source = (
    "ec0a1307a2b1d5e2a283fa3527582b487761e388fbda26fa6a1b5b758d3e2c07"
)

expected_data = (
    "bcbd58d005cf227a52c9afed0912983340ce1b30888bd32c42643a214eb4bf3b"
)

for p in (
    original, data, wrapper, preflight, log, exit_file
):
    gate(p.is_file(), f"Missing source: {p}")

gate(
    sha256(original) == expected_source,
    "Original trainer changed",
)

gate(
    sha256(data) == expected_data,
    "Multi5k dataset changed",
)

pre = read_json(preflight)

gate(
    pre["original_trainer_sha256"] == expected_source,
    "Preflight original source mismatch",
)

gate(
    pre["dataset_sha256"] == expected_data,
    "Preflight dataset mismatch",
)

gate(
    pre["wrapper_sha256"] == sha256(wrapper),
    "Runtime wrapper changed",
)

gate(
    pre["optimizer"] == "adamw_torch"
    and pre["data_seed"] is None
    and pre["seed"] == 13,
    "Preflight training parameters mismatch",
)

print("ORIGINAL_TRAINER_SHA256 = PASS")
print("MULTI5K_DATA_SHA256 = PASS")
print("RUNTIME_WRAPPER_SHA256 = PASS")

print()
print("[2] TRAINING LOG CLOSURE")

text = log.read_text(
    encoding="utf-8",
    errors="replace",
)

gate(
    "R13_RUNTIME_OPTIMIZER=adamw_torch" in text,
    "Missing runtime optimizer verification",
)

gate(
    "R13_RUNTIME_DATA_SEED=None" in text,
    "Missing data_seed verification",
)

gate(
    "R13_RUNTIME_TRAINING_ARGUMENTS=PASS" in text,
    "Missing runtime training gate",
)

gate(
    "Loaded 45000 verifier examples" in text,
    "Incorrect training dataset size",
)

gate(
    "2813/2813" in text,
    "Training progress did not reach 2813 steps",
)

gate(
    "Verifier adapter saved to:" in text,
    "Missing final adapter save evidence",
)

metrics = []

for line in text.splitlines():
    line = line.strip()

    if not line.startswith("{'train_runtime':"):
        continue

    try:
        item = ast.literal_eval(line)
    except (SyntaxError, ValueError):
        continue

    if isinstance(item, dict):
        metrics.append(item)

gate(
    len(metrics) >= 1,
    "Final Trainer metrics not found",
)

final = metrics[-1]

gate(
    abs(float(final["epoch"]) - 1.0) < 1e-6,
    "Final epoch not equal to 1",
)

gate(
    float(final["train_runtime"]) > 0,
    "Invalid training runtime",
)

gate(
    int(exit_file.read_text().strip()) == 1,
    "Expected original post-audit exit status 1",
)

gate(
    "FrozenHistoricalTrainingArguments" in text
    and "AttributeError" in text,
    "Unexpected V5 failure mechanism",
)

print("TRAINING_REACHED_FINAL_STEP = PASS")
print("TRAINING_EPOCH = 1")
print("TRAINING_RUNTIME_SECONDS =", final["train_runtime"])
print("FINAL_TRAIN_LOSS =", final["train_loss"])
print("ORIGINAL_POSTAUDIT_EXIT_CODE = 1")

print()
print("[3] MODEL FILE INTEGRITY")

required = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "verifier_config.json",
    "training_args.bin",
    "environment_manifest.json",
]

artifact_hashes = {}

for name in required:
    path = MODEL / name

    gate(path.is_file(), f"Missing model artifact: {name}")

    gate(
        path.stat().st_size > 0,
        f"Empty model artifact: {name}",
    )

    artifact_hashes[name] = {
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }

    print(
        name,
        "SHA256=",
        artifact_hashes[name]["sha256"],
    )

cfg = read_json(MODEL / "verifier_config.json")
adapter = read_json(MODEL / "adapter_config.json")

gate(
    cfg["num_training_examples"] == 45000,
    "Saved number of examples mismatch",
)

gate(cfg["seed"] == 13, "Saved seed mismatch")
gate(cfg["max_length"] == 1024, "Max length mismatch")

gate(
    cfg["gradient_accumulation_steps"] == 16,
    "Gradient accumulation mismatch",
)

gate(
    abs(float(cfg["learning_rate"]) - 1e-4) < 1e-12,
    "Learning rate mismatch",
)

gate(
    float(cfg["epochs"]) == 1.0,
    "Epoch configuration mismatch",
)

if "optim" in cfg:
    gate(
        cfg["optim"] == "adamw_torch",
        "Saved optimizer summary mismatch",
    )

gate(adapter["r"] == 8, "LoRA rank mismatch")

gate(
    adapter["lora_alpha"] == 16,
    "LoRA alpha mismatch",
)

gate(
    abs(float(adapter["lora_dropout"]) - 0.05) < 1e-12,
    "LoRA dropout mismatch",
)

gate(
    set(adapter["target_modules"])
    == {"q_proj", "k_proj", "v_proj", "o_proj"},
    "LoRA target modules mismatch",
)

weights_path = MODEL / "adapter_model.safetensors"

num_tensors = 0
num_parameters = 0

with safe_open(
    str(weights_path),
    framework="pt",
    device="cpu",
) as f:

    keys = list(f.keys())

    gate(len(keys) > 0, "Adapter tensor set empty")

    for key in keys:
        tensor = f.get_tensor(key)

        gate(
            torch.isfinite(tensor).all().item(),
            f"Nonfinite adapter tensor: {key}",
        )

        num_tensors += 1
        num_parameters += tensor.numel()

gate(
    num_parameters == 5046272,
    "LoRA trainable parameter count mismatch",
)

print("ADAPTER_TENSORS =", num_tensors)
print("ADAPTER_PARAMETERS =", num_parameters)
print("ADAPTER_ALL_FINITE = PASS")
print("MODEL_CONFIGURATION = PASS")

print()
print("[4] CUSTOM TRAINING_ARGUMENTS COMPATIBILITY")

# Load the exact frozen runtime wrapper under
# a non-main module name. This does NOT call
# the wrapper's main() and does NOT train.
spec = importlib.util.spec_from_file_location(
    "r13_v5_runtime_audit",
    str(wrapper),
)

gate(
    spec is not None and spec.loader is not None,
    "Cannot import frozen runtime wrapper",
)

module = importlib.util.module_from_spec(spec)

sys.modules[spec.name] = module
spec.loader.exec_module(module)

gate(
    hasattr(module, "FrozenHistoricalTrainingArguments"),
    "Saved TrainingArguments class not found",
)

# The checkpoint was created while the wrapper
# was executed as __main__. Register its class
# in this audit process's __main__ namespace.
main_module = sys.modules["__main__"]

setattr(
    main_module,
    "FrozenHistoricalTrainingArguments",
    module.FrozenHistoricalTrainingArguments,
)

# Local checkpoint was created by this project.
# weights_only=False is needed for this trusted,
# custom-class TrainingArguments pickle.
args = torch.load(
    MODEL / "training_args.bin",
    map_location="cpu",
    weights_only=False,
)

actual_optimizer = str(
    getattr(args.optim, "value", args.optim)
)

scheduler = str(
    getattr(
        args.lr_scheduler_type,
        "value",
        args.lr_scheduler_type,
    )
)

gate(
    actual_optimizer == "adamw_torch",
    "Actual optimizer mismatch",
)

gate(
    args.data_seed is None,
    "Actual data_seed mismatch",
)

gate(args.seed == 13, "Actual seed mismatch")

gate(
    args.per_device_train_batch_size == 1,
    "Actual micro batch mismatch",
)

gate(
    args.gradient_accumulation_steps == 16,
    "Actual accumulation mismatch",
)

gate(
    abs(float(args.learning_rate) - 1e-4) < 1e-12,
    "Actual learning rate mismatch",
)

gate(
    float(args.num_train_epochs) == 1.0,
    "Actual epoch mismatch",
)

gate(
    scheduler == "linear",
    "Actual LR scheduler mismatch",
)

gate(
    args.gradient_checkpointing is True,
    "Gradient checkpointing mismatch",
)

gate(
    args.fp16 is True and args.bf16 is False,
    "Mixed precision mismatch",
)

print("ACTUAL_OPTIMIZER =", actual_optimizer)
print("ACTUAL_DATA_SEED =", args.data_seed)
print("ACTUAL_SCHEDULER =", scheduler)
print("TRAINING_ARGUMENTS_LOAD = PASS")
print("HISTORICAL_TRAINING_PARITY = PASS")

report = {
    "experiment": "R1-3 Multi5k seed13",
    "source_run": "train_multi5k_seed13_v5",
    "classification":
        "TRAINING_COMPLETED_POSTAUDIT_PICKLE_FAILURE",
    "training_steps": 2813,
    "epochs": 1,
    "train_metrics": final,
    "training_loss": float(final["train_loss"]),
    "original_launcher_exit_code": 1,
    "posttrain_audit": "PASS",
    "model_dir": str(MODEL),
    "model_artifacts": artifact_hashes,
    "adapter_tensors": num_tensors,
    "adapter_parameters": num_parameters,
    "adapter_finite": True,
    "optimizer": actual_optimizer,
    "data_seed": args.data_seed,
    "lr_scheduler": scheduler,
    "source_wrapper_sha256": sha256(wrapper),
    "original_trainer_sha256": sha256(original),
    "training_data_sha256": sha256(data),
    "preflight_sha256": sha256(preflight),
    "training_log_sha256": sha256(log),
    "training_args_pickle_portability":
        "Requires custom class resolution",
    "validation_evaluation_completed": False,
    "test_evaluation_completed": False,
    "historical_artifacts_modified": False,
}

path = OUT / "posttrain_closure.json"

gate(
    not path.exists(),
    "Closure report already exists",
)

path.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    ) + "\n",
    encoding="utf-8",
)

print()
print("=" * 95)
print("R1_3_MULTI5K_SEED13_POSTTRAIN_CLOSURE = PASS")
print("TRAINING_COMPLETED = TRUE")
print("FINAL_MODEL_SAVED_AND_VERIFIED = TRUE")
print("ORIGINAL_LAUNCH_EXIT_CODE_REMAINS = 1")
print("VALIDATION_MRR = NOT_YET_EVALUATED")
print("FROZEN_ARTIFACTS_UNCHANGED = TRUE")
print("CLOSURE_SHA256 =", sha256(path))
print("=" * 95)
