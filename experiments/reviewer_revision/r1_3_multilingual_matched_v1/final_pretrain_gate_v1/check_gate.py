import sys
import json
import hashlib
import inspect
from pathlib import Path

import numpy as np

ROOT = Path(
    "/root/autodl-tmp/lora-code-embeddings-main/text2code"
)

OUT = ROOT / (
    "results/reviewer_revision/"
    "r1_3_multilingual_matched_v1/"
    "final_pretrain_gate_v1"
)

R13 = ROOT / (
    "results/reviewer_revision/"
    "r1_3_multilingual_matched_v1"
)

ADAPTER = R13 / "validation_pool_adapter_v2"

DATA = ROOT.parent / "data/CSN"

LANGS = [
    "ruby", "javascript", "php",
    "python", "java", "go",
]

SEEDS = [13, 42, 87]

def require(ok, message):
    if not ok:
        raise RuntimeError("FAIL_STOP: " + str(message))

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def check_language(lang):
    from data_utils import create_dataset

    manifest_path = ADAPTER / "conversion_manifest.json"

    require(
        sha256(manifest_path)
        == "558902f9d89b0ec9ee3ba61e1227ad70285f87f3f5217b5418d26ad40da0ee49",
        "Conversion manifest SHA256 mismatch",
    )

    manifest = read_json(manifest_path)

    original = Path(
        manifest["source_pools"][lang]["path"]
    )

    converted = Path(
        manifest["converted_pools"][lang]["path"]
    )

    require(
        sha256(original)
        == manifest["source_pools"][lang]["sha256"],
        "Original validation pool changed",
    )

    require(
        sha256(converted)
        == manifest["converted_pools"][lang]["sha256"],
        "Converted validation pool changed",
    )

    with np.load(original, allow_pickle=False) as z:
        original_gt = np.asarray(
            z["gt_indices"], dtype=np.int64
        )
        original_indices = z["candidate_indices"]
        original_scores = z["candidate_scores"]
        original_ranks = z["base_ranks"]

    with np.load(converted, allow_pickle=False) as z:
        require(
            np.array_equal(
                z["indices"], original_indices
            ),
            "Converted candidate indices differ",
        )
        require(
            np.array_equal(
                z["scores"], original_scores
            ),
            "Converted candidate scores differ",
        )
        require(
            np.array_equal(
                z["base_ranks"], original_ranks
            ),
            "Converted base ranks differ",
        )

    # Exact same GT source as eval_llm_verifier.py.
    _, valid_dataset, _, code_dataset = create_dataset(
        str(DATA), lang
    )

    mapping = valid_dataset.text2code

    if isinstance(mapping, dict):
        canonical_gt = np.asarray(
            [mapping[i] for i in range(len(valid_dataset))],
            dtype=np.int64,
        )
    elif hasattr(mapping, "detach"):
        canonical_gt = (
            mapping.detach().cpu().numpy()
            .astype(np.int64)
        )
    else:
        canonical_gt = np.asarray(
            list(mapping), dtype=np.int64
        )

    require(
        canonical_gt.shape == original_gt.shape,
        f"{lang}: canonical GT shape mismatch",
    )

    differences = np.flatnonzero(
        canonical_gt != original_gt
    )

    result = {
        "language": lang,
        "valid_queries": int(len(valid_dataset)),
        "codebase_rows": int(len(code_dataset)),
        "canonical_gt_sha256": hashlib.sha256(
            canonical_gt.tobytes()
        ).hexdigest(),
        "original_gt_sha256": hashlib.sha256(
            original_gt.tobytes()
        ).hexdigest(),
        "gt_different_rows": int(len(differences)),
        "first_differences": [
            {
                "row": int(i),
                "canonical": int(canonical_gt[i]),
                "saved": int(original_gt[i]),
            }
            for i in differences[:10]
        ],
        "gt_identity_pass": len(differences) == 0,
        "pool_array_parity_pass": True,
        "original_pool_sha256": sha256(original),
        "converted_pool_sha256": sha256(converted),
    }

    path = OUT / f"{lang}_canonical_gt.json"

    require(not path.exists(), f"Output exists: {path}")

    path.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        lang,
        "N=", len(valid_dataset),
        "GT_DIFFERENCES=", len(differences),
        "GT_GATE=",
        "PASS" if not len(differences) else "FAIL",
        flush=True,
    )

    require(
        not len(differences),
        f"{lang}: canonical GT differs from frozen pool",
    )

def summarize():
    from transformers import TrainingArguments

    report = {
        "phase": "R1-3 final pretrain audit",
        "canonical_gt": {},
        "historical_training": {},
        "training_authorized": False,
    }

    for lang in LANGS:
        path = OUT / f"{lang}_canonical_gt.json"

        require(path.is_file(), f"Missing GT audit: {lang}")

        item = read_json(path)

        require(
            item["gt_identity_pass"]
            and item["pool_array_parity_pass"],
            f"{lang}: GT/parity gate failed",
        )

        report["canonical_gt"][lang] = item

    trainer = ROOT / "train_qwen_verifier.py"

    require(
        sha256(trainer)
        == "ec0a1307a2b1d5e2a283fa3527582b487761e388fbda26fa6a1b5b758d3e2c07",
        "Training script SHA256 changed",
    )

    model_root = (
        ROOT / "trained_verifiers/"
        "reviewer_r1_1_hn_source_ablation"
    )

    expected = {
        "max_length": 1024,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 0.0001,
        "warmup_ratio": 0.03,
        "epochs": 1.0,
        "optim": "adamw_torch",
        "deterministic": False,
        "trainable_dtype": "float32",
        "num_training_examples": 270000,
    }

    for seed in SEEDS:
        cfg_path = (
            model_root
            / f"nomic_seed{seed}"
            / "verifier_config.json"
        )

        require(
            cfg_path.is_file(),
            f"Historical model config missing: {seed}",
        )

        cfg = read_json(cfg_path)

        for key, value in expected.items():
            require(
                cfg.get(key) == value,
                f"Historical seed{seed} mismatch: {key}",
            )

        require(
            int(cfg.get("seed", -1)) == seed,
            f"Historical seed mismatch: {seed}",
        )

        require(
            set(cfg.get("target_modules", []))
            == {"q_proj", "k_proj", "v_proj", "o_proj"},
            f"Historical LoRA target mismatch: {seed}",
        )

        report["historical_training"][str(seed)] = {
            "config_sha256": sha256(cfg_path),
            "config": cfg,
        }

    # Inspect installed Transformers API without
    # instantiating a model or using CUDA.
    signature = inspect.signature(TrainingArguments)

    require(
        "optim" in signature.parameters,
        "Current TrainingArguments lacks optim parameter",
    )

    optimizer_default = (
        signature.parameters["optim"].default
    )

    actual_optimizer = getattr(
        optimizer_default,
        "value",
        str(optimizer_default),
    )

    report["current_transformers_default_optimizer"] = (
        str(actual_optimizer)
    )

    report["historical_optimizer"] = "adamw_torch"

    optimizer_match = (
        str(actual_optimizer) == "adamw_torch"
    )

    report["optimizer_default_matches_historical"] = (
        optimizer_match
    )

    report["budget"] = {
        "multi5k_examples": 45000,
        "mono5k_examples": 45000,
        "multi30k_examples": 270000,
        "mono30k_exposure_matched_examples": 270000,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "nominal_steps_5k": 2813,
        "nominal_steps_30k": 16875,
        "exact_token_budget_parity": "NOT_YET_CHECKED",
        "historical_dataset_optimizer_alignment":
            "REQUIRES_FROZEN_TRAINING_CLI",
    }

    print()
    print("=" * 95)
    print("FINAL PRETRAIN GATES")
    print("=" * 95)

    print("CANONICAL_GT_ALL_SIX = PASS")
    print("CONVERTED_POOL_PARITY = PASS")
    print("HISTORICAL_TRAINING_CONFIG = PASS")
    print(
        "CURRENT_OPTIMIZER_DEFAULT =",
        actual_optimizer,
    )
    print("HISTORICAL_OPTIMIZER = adamw_torch")

    if optimizer_match:
        print("OPTIMIZER_DEFAULT_COMPATIBILITY = PASS")
    else:
        print(
            "OPTIMIZER_DEFAULT_COMPATIBILITY = "
            "REQUIRES_EXPLICIT_OPTIMIZER"
        )

    print("EXACT_TOKEN_BUDGET = PENDING")
    print("GPU_TRAINING_AUTHORIZED = FALSE")
    print("NO_FROZEN_ARTIFACT_MODIFICATION = TRUE")

    path = OUT / "final_pretrain_summary.json"

    require(not path.exists(), "Summary already exists")

    path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print("SUMMARY_SHA256 =", sha256(path))

if __name__ == "__main__":
    require(
        len(sys.argv) == 2,
        "Usage: check_gate.py <language|summary>",
    )

    if sys.argv[1] == "summary":
        summarize()
    else:
        require(
            sys.argv[1] in LANGS,
            "Unknown language",
        )
        check_language(sys.argv[1])
