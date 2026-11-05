#!/usr/bin/env python3
"""Collect existing ReCalSearch research scripts/results into a GitHub release ZIP.

Run on the author's AutoDL machine AFTER copying/unzipping the GitHub candidate.
Read-only for research inputs; writes only a new output directory. Never launches
training/inference. Explicitly excludes checkpoints, model weights and raw datasets.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import zipfile

DEFAULT_W = Path('/root/autodl-tmp/work1_isolated')
SKIP_DIRS = {
    '.git', '__pycache__', '.cache', 'env', 'venv', '.venv', 'site-packages',
    'node_modules', 'data', 'CSN', 'raw_data', 'datasets', 'codebase', 'corpus',
    'hf_models', 'models', 'weights', 'checkpoints', 'checkpoint',
    'trained_verifiers', 'trained_adapters', 'model_cache', 'embedding_cache',
    'jina_cache', 'cache', 'cached', 'pools', 'logs', 'log', 'per_query',
}
SCRIPT_EXT = {'.py','.sh','.md','.toml','.yaml','.yml','.ini','.cfg'}
RESULT_EXT = {'.json','.csv','.tsv','.tex'}
RESULT_RE = re.compile(r'bootstrap|summary|metrics|protocol|closure|crossover|interaction|hardness|manifest|paired|final',re.I)
SECRET_RE = [
    re.compile(rb'(?:ghp_|github_pat_|hf_|sk-)[A-Za-z0-9_\-]{20,}'),
    re.compile(rb'AKIA[0-9A-Z]{16}'),
    re.compile(rb'-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----'),
]


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def add_source(src:Path, dest:Path, manifest:list, category:str, root:Path, max_bytes:int=5_000_000):
    if not src.is_file() or src.is_symlink(): return False
    if src.stat().st_size > max_bytes: return False
    if not src.resolve().is_relative_to(root.resolve()): return False
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(src,dest)
    if sha256(src)!=sha256(dest): raise RuntimeError('COPY_SHA_MISMATCH: '+str(src))
    manifest.append({'category':category,'relative_path':str(dest),'bytes':src.stat().st_size})
    return True


def walk_files(root:Path):
    if not root.is_dir():return
    for base,dirs,files in os.walk(root,followlinks=False):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.')]
        for name in files:
            p=Path(base)/name
            if not p.is_symlink() and p.is_file():yield p


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--work1',default=str(DEFAULT_W),help='Work1 root')
    ap.add_argument('--include-legacy',action='store_true',help='Also collect text sources from original non-isolated AutoDL tree')
    ap.add_argument('--output-root',default='',help='Where to create an isolated new release directory')
    args=ap.parse_args()
    W=Path(args.work1).resolve()
    R=W/'data/lora-code-embeddings-main/text2code'
    if not R.is_dir():raise SystemExit('FAIL_STOP: Work1 source directory not found '+str(R))

    original=Path(__file__).resolve().parents[1]
    if not (original/'README.md').is_file():raise SystemExit('FAIL_STOP: run this tool from extracted candidate GitHub source tree')
    result_root=Path(args.output_root).resolve() if args.output_root else W/'diagnostics'
    result_root.mkdir(parents=True,exist_ok=True)
    out=result_root/('ReCalSearch_GitHub_Release_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(exist_ok=False)
    target=out/'ReCalSearch'
    shutil.copytree(original,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git','.DS_Store'))
    (target/'SHA256SUMS').unlink(missing_ok=True)
    (target/'RELEASE_INFO.json').unlink(missing_ok=True)

    manifest=[]
    counts={}
    base_count=sum(1 for p in target.rglob('*') if p.is_file())
    print('PHASE=1/5 BASE_INPUT_FILES',base_count,flush=True)

    # Research source files in the primary Work1 tree. Only Python and shell/code docs.
    targets=[
        ('work1_core',R,target/'code_original_full',SCRIPT_EXT),
        ('work1_reviewer',R/'results/reviewer_revision',target/'experiments/reviewer_revision',SCRIPT_EXT),
    ]
    if args.include_legacy:
        legacy=Path('/root/autodl-tmp/lora-code-embeddings-main/text2code')
        if legacy.is_dir() and legacy.resolve()!=R.resolve():
            targets.append(('legacy_core',legacy,target/'experiments/legacy_original',SCRIPT_EXT))
    for category,src_root,dst_root,extensions in targets:
        n=0
        if src_root==R:
            # Root source scripts plus scripts/ and paper methods; no model cache/datasets.
            selected=list(R.glob('*.py'))+list(R.glob('*.sh'))
            for sub in ('scripts','paper'):
                selected+=list(walk_files(R/sub) or [])
        elif category=='legacy_core':
            selected=list(src_root.glob('*.py'))+list(src_root.glob('*.sh'))
            for sub in ('scripts',):selected+=list(walk_files(src_root/sub) or [])
        else:selected=walk_files(src_root) or []
        for p in selected:
            if p.suffix.lower() not in extensions: continue
            if add_source(p,dst_root/p.relative_to(src_root),manifest,category,src_root):n+=1
        counts[category]=n
    print('PHASE=2/5 SOURCE_SCRIPTS',counts,flush=True)

    # Frozen result summaries; large inference arrays and raw query data excluded.
    results_root=R/'results/reviewer_revision'
    n=0
    if results_root.is_dir():
        for p in walk_files(results_root) or []:
            if p.suffix.lower() not in RESULT_EXT or not RESULT_RE.search(p.name):continue
            if add_source(p,target/'experiments/reviewer_revision_results'/p.relative_to(results_root),manifest,'reviewer_results',results_root):n+=1
    counts['reviewer_results']=n
    # Author-provided CoRNStack integration patch only, not upstream third-party repository.
    patch_roots=[W/'data/cornstack-official/recal_patch',Path('/root/autodl-tmp/cornstack-official/recal_patch')]
    seen=[]
    for patch in patch_roots:
        if not patch.is_dir():continue
        if patch.resolve() in seen:continue
        seen.append(patch.resolve())
        for p in walk_files(patch) or []:
            if p.suffix.lower() not in SCRIPT_EXT:continue
            if add_source(p,target/'baselines/coderankllm_integration'/p.relative_to(patch),manifest,'coderank_patch',patch):
                counts['coderank_patch']=counts.get('coderank_patch',0)+1

    # Historical CodeRankLLM frozen compact result files (no full per-query code/metadata).
    res_roots=[W/'data/cornstack-official/results_coderankllm_stats_final_frozen',Path('/root/autodl-tmp/cornstack-official/results_coderankllm_stats_final_frozen')]
    seen=[]
    for cr in res_roots:
        if not cr.is_dir() or cr.resolve() in seen:continue
        seen.append(cr.resolve())
        for p in cr.iterdir():
            if p.suffix.lower() not in RESULT_EXT or not p.is_file():continue
            if add_source(p,target/'experiments/coderankllm_frozen_results'/p.name,manifest,'coderank_results',cr):
                counts['coderank_results']=counts.get('coderank_results',0)+1
    print('PHASE=3/5 RESULT_RECORDS',counts,flush=True)

    # Look for original crossover analysis sources in isolated/legacy trees. Not proof of closure.
    codes=[p.name for p in (target/'experiments').rglob('*.py')] + [p.name for p in (target/'experiments').rglob('*.sh')]
    presence={
        'source_ablation_script_names':sum(bool(re.search('hard.neg|negative|hn_source|nomic|jina',n,re.I)) for n in codes),
        'crossover_script_names':sum('crossover' in n.lower() for n in codes),
        'coderank_patch_scripts':counts.get('coderank_patch',0),
        'coderank_result_files':counts.get('coderank_results',0),
    }

    # Strict scan for obvious access credentials before packaging. Keep private log outside ZIP.
    flagged=[]
    for p in target.rglob('*'):
        if not p.is_file():continue
        if p.suffix.lower() in {'.py','.sh','.txt','.md','.yaml','.json','.toml','.ini','.cfg','.csv'} and p.stat().st_size<5_000_000:
            b=p.read_bytes()
            if any(rx.search(b) for rx in SECRET_RE):flagged.append(str(p.relative_to(target)))
    if flagged:
        (out/'PRIVATE_REVIEW_REQUIRED.txt').write_text('\n'.join(flagged)+'\n',encoding='utf8')
        raise RuntimeError('FAIL_STOP_POSSIBLE_CREDENTIALS: review PRIVATE_REVIEW_REQUIRED.txt; no public ZIP created')

    infos=[]
    for p in sorted(target.rglob('*')):
        if p.is_file() and p.name not in {'SHA256SUMS','RELEASE_INFO.json'}:
            infos.append({'path':p.relative_to(target).as_posix(),'sha256':sha256(p),'bytes':p.stat().st_size})
    public_info={
        'status':'SOURCE_COLLECTED_NOT_PUBLISHED',
        'base_file_count':base_count,
        'new_source_and_result_files':len(manifest),
        'category_counts':counts,
        'source_presence_by_filename_only':presence,
        'files_sha256':infos,
        'note':'Does not include datasets, model weights, adapters or full large frozen arrays; check licenses and completeness before publishing.'
    }
    (target/'RELEASE_INFO.json').write_text(json.dumps(public_info,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    lines=[sha256(p)+'  '+p.relative_to(target).as_posix() for p in sorted(target.rglob('*')) if p.is_file() and p.name!='SHA256SUMS']
    (target/'SHA256SUMS').write_text('\n'.join(lines)+'\n',encoding='utf8')

    # Avoid unsupported GitHub object sizes. No model checkpoints should survive.
    files=[p for p in target.rglob('*') if p.is_file()]
    max_size=max((p.stat().st_size for p in files),default=0)
    total_size=sum(p.stat().st_size for p in files)
    if max_size>95_000_000 or total_size>700_000_000:
        raise RuntimeError('FAIL_STOP_REPO_TOO_LARGE max_file='+str(max_size)+' total='+str(total_size))

    zip_path=out/'ReCalSearch_GitHub_Source_and_Experiments.zip'
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for p in files:z.write(p,arcname='ReCalSearch/'+p.relative_to(target).as_posix())
    print('PHASE=4/5 SHA256_AND_SECRET_SCAN PASS',flush=True)
    print('PHASE=5/5 OUTPUT',flush=True)
    print('STATUS=PASS_SOURCE_COLLECTED_NOT_PUBLISHED',flush=True)
    print('REPO_FILES=',len(files),flush=True)
    print('SOURCE_COLLECTION_COUNTS=',json.dumps(counts,ensure_ascii=False),flush=True)
    print('PRESENCE_HEURISTICS=',json.dumps(presence,ensure_ascii=False),flush=True)
    print('OUTPUT_ZIP=',zip_path,flush=True)
    print('OUTPUT_SHA256=',sha256(zip_path),flush=True)
    print('NEW_GPU_EXPERIMENT=NO',flush=True)
    print('WORK2_MODIFIED=NO',flush=True)

if __name__=='__main__':main()
