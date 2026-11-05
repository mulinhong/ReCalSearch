#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _jsonable(x: Any) -> Any:
    """
    Convert common Python / NumPy / Path objects to JSON-safe values.
    """
    if isinstance(x, Path):
        return str(x)

    if isinstance(x, dict):
        return {
            str(k): _jsonable(v)
            for k, v in x.items()
        }

    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]

    # NumPy scalars and similar objects.
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass

    if isinstance(x, (str, int, float, bool)) or x is None:
        return x

    return str(x)


def _atomic_json_dump(path: str | Path, payload: dict) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(
            _jsonable(payload),
            f,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        f.write("\n")

    os.replace(tmp, path)


def load_json(path: str | Path) -> dict:
    """
    Load a JSON file and require a top-level object.
    """
    path = Path(path).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"JSON file does not exist: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(
            f"Expected top-level JSON object in {path}, "
            f"got {type(obj).__name__}"
        )

    return obj


def save_frozen_config(
    path: str | Path,
    *,
    method: str,
    selected_on_split: str,
    selected_metric: str,
    config: dict,
    validation_result: Any,
) -> None:
    """
    Save an immutable-by-protocol configuration selected ONLY on validation.

    This does not technically make the file filesystem-immutable; rather, it
    records the provenance needed by the evaluation protocol.
    """
    if selected_on_split != "valid":
        raise ValueError(
            "Frozen paper configuration must be selected on validation, "
            f"not {selected_on_split!r}."
        )

    if not isinstance(config, dict) or not config:
        raise ValueError(
            "Frozen config must contain a non-empty config dictionary."
        )

    payload = {
        "schema_version": 1,
        "method": method,
        "selected_on_split": selected_on_split,
        "selected_metric": selected_metric,
        "config": _jsonable(config),
        "validation_result": _jsonable(validation_result),
        "selection_protocol": "validation_only_then_test_once",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    _atomic_json_dump(path, payload)


def enforce_frozen_test_protocol(
    split: str,
    selected_values,
    frozen_config: str | Path,
    hyperparameter_name: str,
) -> None:
    """
    Enforce reviewer-safe test evaluation.

    For test:
      1. a frozen config must be supplied;
      2. it must have been selected on valid;
      3. only ONE value may be evaluated;
      4. that value must exactly match the frozen value.

    This prevents accidental test-set hyperparameter sweeps.
    """
    if split != "test":
        return

    if not frozen_config:
        raise ValueError(
            "Test evaluation requires --frozen-config. "
            "Select hyperparameters on validation first. "
            "For historical reproduction only, use the evaluator's "
            "--unsafe-allow-test-without-frozen-config flag."
        )

    frozen = load_json(frozen_config)

    if frozen.get("selected_on_split") != "valid":
        raise ValueError(
            "Frozen configuration was not selected on validation."
        )

    cfg = frozen.get("config")

    if not isinstance(cfg, dict):
        raise ValueError(
            "Frozen configuration is missing its 'config' object."
        )

    if hyperparameter_name not in cfg:
        raise ValueError(
            f"Frozen configuration does not contain "
            f"{hyperparameter_name!r}."
        )

    values = list(selected_values)

    if len(values) != 1:
        raise ValueError(
            "Test evaluation must use exactly one frozen "
            f"{hyperparameter_name}; got {values}. "
            "Parameter sweeps on test are prohibited."
        )

    observed = values[0]
    expected = cfg[hyperparameter_name]

    try:
        equal = math.isclose(
            float(observed),
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    except (TypeError, ValueError):
        equal = observed == expected

    if not equal:
        raise ValueError(
            f"Test {hyperparameter_name}={observed!r} does not match "
            f"validation-frozen value {expected!r}."
        )


def _run_capture(cmd: list[str]) -> str | None:
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
            check=False,
        )
        text = result.stdout.strip()
        return text if text else None
    except Exception:
        return None


def _module_version(name: str) -> str | None:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return None


def write_environment_manifest(
    path: str | Path,
    *,
    extra: dict | None = None,
) -> None:
    """
    Write a reproducibility manifest for reviewer artifacts.
    """
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version.replace("\n", " "),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "hostname": socket.gethostname(),
        },
        "working_directory": os.getcwd(),
        "packages": {
            "torch": _module_version("torch"),
            "transformers": _module_version("transformers"),
            "peft": _module_version("peft"),
            "numpy": _module_version("numpy"),
        },
    }

    # Torch / CUDA information.
    try:
        import torch

        manifest["cuda"] = {
            "torch_cuda_version": torch.version.cuda,
            "available": bool(torch.cuda.is_available()),
            "device_count": (
                int(torch.cuda.device_count())
                if torch.cuda.is_available()
                else 0
            ),
        }

        if torch.cuda.is_available():
            manifest["cuda"]["devices"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_bytes": int(
                        torch.cuda.get_device_properties(i).total_memory
                    ),
                }
                for i in range(torch.cuda.device_count())
            ]
    except Exception as e:
        manifest["cuda"] = {
            "error": repr(e)
        }

    git_commit = _run_capture(
        ["git", "rev-parse", "HEAD"]
    )
    git_status = _run_capture(
        ["git", "status", "--porcelain"]
    )

    manifest["git"] = {
        "commit": git_commit,
        "dirty": bool(git_status),
        "status_porcelain": git_status or "",
    }

    manifest["nvidia_smi"] = _run_capture([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    ])

    if extra is not None:
        manifest["experiment"] = _jsonable(extra)

    _atomic_json_dump(path, manifest)

def write_json(path, payload):
    """Write JSON with UTF-8, indentation, and parent-directory creation."""
    import json
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")

