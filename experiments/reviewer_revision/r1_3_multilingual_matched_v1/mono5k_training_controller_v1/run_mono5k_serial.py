#!/usr/bin/env python3

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(
    "/root/autodl-tmp/lora-code-embeddings-main/text2code"
)
REV = ROOT / "results/reviewer_revision"
BASE = REV / "r1_3_multilingual_matched_v1"
CTRL = BASE / "mono5k_training_controller_v1"

LANGS = (
    "ruby", "javascript", "php",
    "python", "java", "go"
)
SEEDS = (13, 42, 87)

PARENT_RUN = BASE / "train_mono5k_ruby_seed13_v1"
PARENT_MODEL = BASE / "models/mono5k_ruby_seed13_v1"
PARENT_AUDIT = (
    PARENT_RUN / "posttrain_artifact_audit.json"
)

DATA_ROOT = REV / "hn_source_ablation/qwen_data"

FROZEN_SOURCE = ROOT / "train_qwen_verifier.py"
MULTI5 = BASE / "multi5k_nomic_seed20261013.jsonl"
MULTI30 = (
    DATA_ROOT / "combined_nomic_hn7_30k.jsonl"
)

EXPECTED = {
    "trainer":
        "ec0a1307a2b1d5e2a283fa3527582b487761e388fbda26fa6a1b5b758d3e2c07",
    "multi5":
        "bcbd58d005cf227a52c9afed0912983340ce1b30888bd32c42643a214eb4bf3b",
    "multi30":
        "2240eede7a3d50bf9846562ca0eefb4b0236c8619d40c7110212e947fea16503",
    "parent_audit":
        "6aa9179212ebe1a66a2ac22224a0f265837423b01c502ff99fdac53576c992be",
}

REQUIRED_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "verifier_config.json",
    "training_args.bin",
    "environment_manifest.json",
)


def now():
    return datetime.now().astimezone().isoformat(
        timespec="seconds"
    )


def emit(message):
    print(message, flush=True)


def need(ok, message):
    if not ok:
        raise RuntimeError("FAIL_STOP: " + message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(4 * 1024 * 1024),
            b""
        ):
            h.update(block)
    return h.hexdigest()


def verify(path, digest, name):
    need(path.is_file(), name + ": missing")
    need(
        sha(path) == digest,
        name + ": SHA256 mismatch"
    )
    emit(name + "=PASS")


def read_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def write_new(path, content):
    with path.open("x", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())


def write_json_new(path, data):
    write_new(
        path,
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        ) + "\n"
    )


def flag_value(argv, name):
    need(
        argv.count(name) == 1,
        "Invalid argument: " + name
    )
    i = argv.index(name) + 1
    need(i < len(argv), "Missing value: " + name)
    return argv[i], i


def dirs(lang, seed):
    label = f"mono5k_{lang}_seed{seed}_v1"

    run = BASE / f"train_{label}"
    model = BASE / "models" / label

    return run, model


def data_path(lang):
    return DATA_ROOT / f"{lang}_nomic_hn7_5k.jsonl"


def validate_completed(lang, seed, run, model):
    """
    Accept existing completed jobs only after independent
    artifact/hash checks. Never overwrite existing results.
    """
    for name in (
        "train_exit_code.txt",
        "posttrain_exit_code.txt",
    ):
        p = run / name
        need(
            p.is_file() and p.read_text().strip() == "0",
            f"Existing {lang}/{seed}: {name} not zero"
        )

    report_path = run / "posttrain_artifact_audit.json"

    need(
        report_path.is_file(),
        f"Existing {lang}/{seed}: audit missing"
    )

    if lang == "ruby" and seed == 13:
        need(
            sha(report_path) ==
            EXPECTED["parent_audit"],
            "Ruby Seed13 frozen audit changed"
        )

    report = read_json(report_path)

    need(
        report.get("status") == "PASS",
        f"Existing {lang}/{seed}: audit not PASS"
    )
    need(
        int(report.get("seed", -1)) == seed,
        "Existing seed mismatch"
    )
    need(
        report.get("language") == lang,
        "Existing language mismatch"
    )
    need(
        int(report.get("training_examples", -1)) == 45000,
        "Existing training count mismatch"
    )
    need(
        report.get("all_weights_finite") is True,
        "Existing finite-weight evidence missing"
    )
    need(
        report.get("config_parity") is True,
        "Existing config parity missing"
    )

    recorded = report.get("files_sha256", {})

    for name in REQUIRED_FILES:
        p = model / name
        need(
            name in recorded
            and p.is_file()
            and sha(p) == recorded[name],
            f"Existing artifact mismatch: {p}"
        )

    cfg = read_json(model / "verifier_config.json")

    need(int(cfg["seed"]) == seed, "Existing config seed")
    need(
        int(cfg["num_training_examples"]) == 45000,
        "Existing config examples"
    )

    expected_data = data_path(lang)

    need(
        Path(cfg["train_file"]).resolve()
        == expected_data.resolve(),
        "Existing training data path mismatch"
    )

    need(
        sha(expected_data) == report["data_sha256"],
        "Existing training data content drift"
    )

    log_file = run / "run.log"
    success_marker = (
        f"MONO5K_{lang.upper()}_SEED{seed}=PASS"
    )

    need(
        log_file.is_file()
        and success_marker in log_file.read_text(
            encoding="utf-8",
            errors="replace"
        ),
        "Existing final PASS marker missing"
    )

    emit(
        f"SKIP_VERIFIED_COMPLETED={lang}/seed{seed}"
    )


def preflight_parent():
    emit("========== GLOBAL FROZEN PREFLIGHT ==========")

    verify(
        FROZEN_SOURCE,
        EXPECTED["trainer"],
        "FROZEN_TRAINER"
    )
    verify(
        MULTI5,
        EXPECTED["multi5"],
        "FROZEN_MULTI5_DATA"
    )
    verify(
        MULTI30,
        EXPECTED["multi30"],
        "FROZEN_MULTI30_DATA"
    )
    verify(
        PARENT_AUDIT,
        EXPECTED["parent_audit"],
        "RUBY13_PARENT_AUDIT"
    )

    # Ruby Seed13 must be fully intact.
    validate_completed(
        "ruby", 13, PARENT_RUN, PARENT_MODEL
    )

    manifest_path = (
        PARENT_RUN / "preflight_manifest.json"
    )

    need(
        manifest_path.is_file(),
        "Ruby13 parent manifest missing"
    )

    manifest = read_json(manifest_path)
    argv = list(manifest["training_argv"])

    need(
        sha(FROZEN_SOURCE)
        == manifest["original_trainer_sha256"],
        "Parent trainer hash mismatch"
    )

    wrapper = Path(argv[1]).resolve()

    need(wrapper.is_file(), "Runtime wrapper missing")
    need(
        sha(wrapper)
        == manifest["runtime_wrapper_sha256"],
        "Frozen runtime wrapper drift"
    )

    need(
        Path(argv[0]).resolve()
        == Path(sys.executable).resolve(),
        "Training Python environment drift"
    )

    train_value, _ = flag_value(argv, "--train-file")
    out_value, _ = flag_value(argv, "--output-dir")
    seed_value, _ = flag_value(argv, "--seed")

    need(
        Path(train_value).resolve()
        == data_path("ruby").resolve(),
        "Parent training input"
    )
    need(
        Path(out_value).resolve()
        == PARENT_MODEL.resolve(),
        "Parent model output"
    )
    need(seed_value == "13", "Parent seed")

    expected_params = {
        "--max-length": 1024,
        "--lora-r": 8,
        "--lora-alpha": 16,
        "--lora-dropout": 0.05,
        "--batch-size": 1,
        "--gradient-accumulation-steps": 16,
        "--lr": 1e-4,
        "--epochs": 1,
    }

    for flag, expected in expected_params.items():
        actual, _ = flag_value(argv, flag)
        need(
            abs(float(actual) - float(expected))
            < 1e-12,
            "Frozen parameter mismatch: " + flag
        )

    need(
        flag_value(argv, "--dtype")[0]
        == "float16",
        "Base model dtype"
    )
    need(
        flag_value(argv, "--trainable-dtype")[0]
        == "float32",
        "Trainable dtype"
    )

    need(
        "--deterministic" in argv,
        "Deterministic training flag missing"
    )
    need(
        "--gradient-checkpointing" in argv,
        "Gradient checkpointing missing"
    )

    cfg = read_json(
        PARENT_MODEL / "verifier_config.json"
    )

    need(
        set(cfg["target_modules"])
        == {"q_proj", "k_proj", "v_proj", "o_proj"},
        "LoRA target modules mismatch"
    )

    need(
        int(cfg["num_training_examples"]) == 45000,
        "Parent example count"
    )

    emit("GLOBAL_FROZEN_PREFLIGHT=PASS")

    return argv, cfg, wrapper, manifest_path


def disk_gate():
    s = os.statvfs(BASE)
    gib = 1024 ** 3

    free = s.f_bfree * s.f_frsize / gib
    available = s.f_bavail * s.f_frsize / gib

    emit(f"FREE_BLOCKS_GIB={free:.3f}")
    emit(f"AVAILABLE_GIB={available:.3f}")

    need(
        free >= 50,
        "Insufficient actual free blocks"
    )

    probe = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=BASE,
            prefix=".mono_controller_probe_",
            delete=False,
        ) as f:
            probe = Path(f.name)
            block = b"M" * (1024 * 1024)

            for _ in range(16):
                f.write(block)

            f.flush()
            os.fsync(f.fileno())

        need(
            probe.stat().st_size
            == 16 * 1024 * 1024,
            "Write test size mismatch"
        )

        with probe.open("rb") as f:
            need(
                f.read(1024 * 1024) == block,
                "Write readback mismatch"
            )

    finally:
        if probe is not None:
            probe.unlink(missing_ok=True)

    emit("PERSISTENT_WRITE_GATE=PASS")


def gpu_gate():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True
    )

    for line in result.stdout.splitlines():
        pid = line.strip().split(",")[0].strip()

        if not pid.isdecimal():
            continue

        try:
            cmd = (
                Path(f"/proc/{pid}/cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
            )
        except OSError:
            raise RuntimeError(
                "Cannot verify GPU process PID=" + pid
            )

        need(
            "work1_gpu_keepalive.py" in cmd,
            f"GPU conflict PID={pid}: {cmd[:200]}"
        )

    emit("GPU_CONFLICT_GATE=PASS")


def count_rows(path):
    with path.open("rb") as f:
        return sum(bool(line.strip()) for line in f)


def build_argv(reference, lang, seed, model):
    argv = list(reference)

    train_old, data_idx = flag_value(
        argv, "--train-file"
    )
    output_old, out_idx = flag_value(
        argv, "--output-dir"
    )
    seed_old, seed_idx = flag_value(
        argv, "--seed"
    )

    need(seed_old == "13", "Invalid parent seed")

    data = data_path(lang)

    argv[data_idx] = str(data.resolve())
    argv[out_idx] = str(model.resolve())
    argv[seed_idx] = str(seed)

    changed = {
        i for i, (old, new)
        in enumerate(zip(reference, argv))
        if old != new
    }

    allowed = {out_idx, seed_idx}

    if lang != "ruby":
        allowed.add(data_idx)

    need(
        changed == allowed,
        "Command differs beyond expected data/seed/output"
    )

    return argv


def posttrain_audit(
    lang, seed, run, model,
    data, data_sha, reference_cfg
):
    import numpy as np
    from safetensors import safe_open

    for name in REQUIRED_FILES:
        p = model / name
        need(
            p.is_file() and p.stat().st_size > 0,
            "Missing posttrain file: " + name
        )

    cfg = read_json(
        model / "verifier_config.json"
    )

    need(int(cfg["seed"]) == seed, "Saved seed")
    need(
        int(cfg["num_training_examples"]) == 45000,
        "Saved training example count"
    )
    need(
        Path(cfg["train_file"]).resolve()
        == data.resolve(),
        "Saved training file"
    )

    normalized = dict(cfg)
    normalized["seed"] = reference_cfg["seed"]
    normalized["train_file"] = (
        reference_cfg["train_file"]
    )

    need(
        normalized == reference_cfg,
        "Posttrain frozen parameter parity"
    )

    count = 0
    parameters = 0

    with safe_open(
        str(model / "adapter_model.safetensors"),
        framework="pt",
        device="cpu",
    ) as f:
        for key in f.keys():
            x = f.get_tensor(key)
            need(
                bool(np.isfinite(
                    x.float().numpy()
                ).all()),
                "Nonfinite tensor: " + key
            )
            count += 1
            parameters += x.numel()

    need(count == 224, "Adapter tensor count")
    need(
        parameters == 5046272,
        "Adapter parameter count"
    )

    log = (run / "train.log").read_text(
        encoding="utf-8",
        errors="replace"
    )

    for marker in (
        "2813/2813",
        "Verifier adapter saved to:",
        "R13_RUNTIME_OPTIMIZER=adamw_torch",
        "R13_RUNTIME_DATA_SEED=None",
        "R13_RUNTIME_TRAINING_ARGUMENTS=PASS",
    ):
        need(
            marker in log,
            "Training log marker missing: " + marker
        )

    need(
        sha(data) == data_sha,
        "Training data changed during run"
    )

    report = {
        "experiment":
            f"R1_3_MONO5K_{lang.upper()}_SEED{seed}",
        "status": "PASS",
        "seed": seed,
        "language": lang,
        "training_examples": 45000,
        "config_parity": True,
        "adapter_tensors": count,
        "adapter_parameters": parameters,
        "all_weights_finite": True,
        "data_sha256": data_sha,
        "files_sha256": {
            name: sha(model / name)
            for name in REQUIRED_FILES
        },
        "validation_mrr": None,
    }

    audit_path = (
        run / "posttrain_artifact_audit.json"
    )
    write_json_new(audit_path, report)

    return sha(audit_path)


def run_job(
    lang, seed, parent_argv,
    reference_cfg, wrapper, parent_manifest
):
    run, model = dirs(lang, seed)

    emit("")
    emit("=" * 76)
    emit(
        f"MONO5K_JOB={lang}/seed{seed}"
    )
    emit("=" * 76)

    if run.exists() or model.exists():
        need(
            run.is_dir() and model.is_dir(),
            f"{lang}/{seed}: partial existing output"
        )

        validate_completed(
            lang, seed, run, model
        )
        return

    data = data_path(lang)

    need(
        data.is_file() and not data.is_symlink(),
        "Mono training data missing: " + str(data)
    )

    need(
        count_rows(data) == 45000,
        "Mono training row count mismatch"
    )

    data_sha = sha(data)

    argv = build_argv(
        parent_argv, lang, seed, model
    )

    gpu_gate()
    disk_gate()

    need(
        not run.exists() and not model.exists(),
        "Output appeared during preflight"
    )

    run.mkdir(parents=False, exist_ok=False)

    manifest = {
        "experiment": f"R1_3_MONO5K_{lang.upper()}_SEED{seed}",
        "language": lang,
        "seed": seed,
        "training_examples": 45000,
        "train_file": str(data.resolve()),
        "train_file_sha256": data_sha,
        "training_argv": argv,
        "parent_manifest_sha256":
            sha(parent_manifest),
        "frozen_trainer_sha256":
            sha(FROZEN_SOURCE),
        "frozen_wrapper_sha256":
            sha(wrapper),
        "multi5_sha256": sha(MULTI5),
        "multi30_sha256": sha(MULTI30),
        "optimizer": "adamw_torch",
        "data_seed": None,
        "test_access": False,
        "frozen_assets_modified": False,
    }

    write_json_new(
        run / "preflight_manifest.json",
        manifest
    )

    with (run / "run.log").open(
        "x", encoding="utf-8"
    ) as runlog:

        def record(s):
            runlog.write(s + "\n")
            runlog.flush()
            os.fsync(runlog.fileno())
            emit(s)

        record("TRAIN_START=" + now())
        record(f"LANGUAGE={lang}")
        record(f"SEED={seed}")
        record("TRAINING_DATA_SHA256=" + data_sha)

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = (
            str(ROOT)
            + os.pathsep
            + env.get("PYTHONPATH", "")
        )

        with (run / "train.log").open(
            "x", encoding="utf-8"
        ) as f:

            rc = subprocess.run(
                argv,
                cwd=ROOT,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode

        write_new(
            run / "train_exit_code.txt",
            str(rc) + "\n"
        )

        record("TRAIN_END=" + now())
        record("TRAIN_RETURN_CODE=" + str(rc))

        need(
            rc == 0,
            f"{lang}/seed{seed}: training failed"
        )

        try:
            audit_sha = posttrain_audit(
                lang, seed, run, model,
                data, data_sha, reference_cfg
            )

        except BaseException:
            write_new(
                run / "posttrain_exit_code.txt",
                "1\n"
            )
            record(
                f"MONO5K_{lang.upper()}_SEED{seed}"
                "=POSTTRAIN_FAIL_STOP"
            )
            raise

        write_new(
            run / "posttrain_exit_code.txt",
            "0\n"
        )

        record("POSTTRAIN_CONFIG_PARITY=PASS")
        record("ADAPTER_TENSORS=224")
        record("ADAPTER_PARAMETERS=5046272")
        record("ALL_WEIGHTS_FINITE=PASS")
        record(
            "POSTTRAIN_AUDIT_SHA256=" + audit_sha
        )
        record(
            f"MONO5K_{lang.upper()}_SEED{seed}=PASS"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        choices=SEEDS
    )
    parser.add_argument(
        "--language",
        choices=LANGS
    )
    parser.add_argument(
        "--all",
        action="store_true"
    )

    args = parser.parse_args()

    need(
        not (
            args.all
            and (
                args.seed is not None
                or args.language is not None
            )
        ),
        "--all cannot be combined with other selectors"
    )

    if args.all:
        jobs = [
            (lang, seed)
            for seed in SEEDS
            for lang in LANGS
        ]
    else:
        need(
            args.seed is not None,
            "Specify --seed or --all"
        )

        if args.language is None:
            jobs = [
                (lang, args.seed)
                for lang in LANGS
            ]
        else:
            jobs = [
                (args.language, args.seed)
            ]

    CTRL.mkdir(parents=True, exist_ok=True)

    # Prevent concurrent controller sessions.
    lock_path = CTRL / "mono5k_gpu_serial.lock"

    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(
                lock.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError:
            raise RuntimeError(
                "FAIL_STOP: Another Mono5k controller "
                "is already running"
            )

        emit("MONO5K_CONTROLLER=START")
        emit("START_TIME=" + now())
        emit("QUEUE_JOBS=" + str(len(jobs)))
        emit("EXECUTION_MODE=SERIAL")
        emit("NO_PARALLEL_GPU_TRAINING=TRUE")

        parent_argv, cfg, wrapper, parent_manifest = (
            preflight_parent()
        )

        for i, (lang, seed) in enumerate(jobs, 1):
            emit(
                f"QUEUE_POSITION={i}/{len(jobs)}"
            )

            run_job(
                lang, seed,
                parent_argv,
                cfg,
                wrapper,
                parent_manifest
            )

        emit("")
        emit("MONO5K_SELECTED_QUEUE=PASS")
        emit("END_TIME=" + now())


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        emit("")
        emit("MONO5K_CONTROLLER=FAIL_STOP")
        traceback.print_exc()
        sys.exit(2)
