#!/usr/bin/env python3
"""Export a restricted version-only environment summary (not a pip freeze)."""
import argparse
import importlib.metadata as im
import json
import platform
from pathlib import Path

PACKAGES = (
    'torch', 'transformers', 'peft', 'accelerate', 'numpy', 'scipy',
    'tokenizers', 'safetensors', 'scikit-learn', 'sentence-transformers',
    'tqdm', 'vllm', 'tevatron',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = im.version(pkg)
        except im.PackageNotFoundError:
            versions[pkg] = None
    summary = {
        'purpose': 'new_local_environment_inventory; not historical freeze',
        'python_version': platform.python_version(),
        'platform': platform.system(),
        'packages': versions,
        'torch_cuda_runtime': None,
        'note': ('CUDA not probed; matching package versions alone does not prove '
                 'exact research reproducibility.'),
    }
    out = json.dumps(summary, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out, encoding='utf-8')
        print('OUTPUT=' + str(args.output))
    else:
        print(out, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
