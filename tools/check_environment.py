#!/usr/bin/env python3
"""Dependency presence check without downloads, training, inference or GPU access."""
import argparse
import importlib.metadata as im
import json
import sys
from pathlib import Path

PROFILE = {
    'bootstrap': ('numpy',),
    'training': (
        'numpy', 'scipy', 'scikit-learn', 'torch', 'transformers',
        'peft', 'accelerate', 'safetensors', 'sentence-transformers', 'tqdm',
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILE, default='bootstrap')
    parser.add_argument('--compare-observed', action='store_true',
                        help='Print archival-version differences; do not fail on them')
    args = parser.parse_args()
    print('PROFILE=' + args.profile)
    print('PYTHON=' + sys.version.split()[0])
    missing = []
    for pkg in PROFILE[args.profile]:
        try:
            ver = im.version(pkg)
            print(f'PACKAGE={pkg} VERSION={ver} STATUS=FOUND')
        except im.PackageNotFoundError:
            print(f'PACKAGE={pkg} STATUS=MISSING')
            missing.append(pkg)
    if args.compare_observed:
        snapshot_path = Path(__file__).resolve().parents[1] / 'metadata/environment_current.json'
        obj = json.loads(snapshot_path.read_text(encoding='utf-8'))
        for name, frozen_version in obj['current_packages'].items():
            try:
                here = im.version(name)
            except im.PackageNotFoundError:
                here = '(missing)'
            print(f'OBSERVED={name} ARCHIVED={frozen_version} HERE={here} '
                  f'MATCH={str(here == frozen_version).upper()}')
        print('OBSERVED_NOTE=Current Work1 snapshot is not a historical run lock')
    print('STATUS=' + ('PASS_DEPENDENCIES_PRESENT' if not missing else 'MISSING_DEPENDENCIES'))
    return 0 if not missing else 2


if __name__ == '__main__':
    raise SystemExit(main())
