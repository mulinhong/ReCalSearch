#!/usr/bin/env python3
"""Verify the relative SHA256SUMS manifest of a release, without running experiments."""
from pathlib import Path
import hashlib,sys
root=Path(__file__).resolve().parents[1]
manifest=root/'SHA256SUMS'
lines=manifest.read_text(encoding='utf8').splitlines()
count=0
for line in lines:
    if not line.strip(): continue
    expected,rel=line.split('  ',1)
    path=(root/rel).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise SystemExit('MISSING/OUTSIDE '+rel)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for piece in iter(lambda:f.read(4*1024*1024),b''): h.update(piece)
    if h.hexdigest()!=expected:raise SystemExit('SHA_MISMATCH '+rel)
    count+=1
print('RELEASE_SHA256_PASS',count)
