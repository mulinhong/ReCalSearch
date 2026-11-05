from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
import traceback
import runpy

import numpy as np

ROOT = Path('/root/autodl-tmp/lora-code-embeddings-main/text2code')
BASE = ROOT / 'results/reviewer_revision/r1_3_multilingual_matched_v1'
OUT = Path(os.environ['R13_SINGLELOAD_OUT'])

EVAL = BASE / 'full_valid_multi5k_seed13_v1/eval_valid_only.py'
FROZEN = ROOT / (
    'results/reviewer_revision/recal_valid_fresh_20260911/'
    'recal_main_frozen_config.json'
)
POOL = BASE / (
    'validation_pool_adapter_v2/candidate_pools/'
    'ruby_valid_nomic_top100.npz'
)
CANON = ROOT / (
    'results/reviewer_revision/recal_valid_fresh_20260911/'
    'validation_grid/candidate_pools/'
    'ruby_valid_nomic_top100.npz'
)
MODEL = BASE / 'models/multi5k_seed13_v5'
CTRL = BASE / (
    'mono5k_training_controller_v1/'
    'run_mono5k_serial_v2.py'
)
REFERENCE = BASE / 'valid_chunk_parity_probe_62cZCRYi/full8'
BASE_MODEL = Path(
    '/root/autodl-tmp/hf_models/Qwen2.5-Coder-7B-Instruct'
)
DATA = ROOT.parent / 'data/CSN'

EXPECTED = {
    EVAL:
        '08692a72ab8b6e46a1ce99f83d762cea02a933db4bd7cb54840c5877d59163bb',
    ROOT / 'eval_llm_verifier.py':
        'a3156c285728d54180ce0da1c65255a92adfe15f583dc987e47ad8a97b8e665a',
    FROZEN:
        '70789e10518f32a97f5990b234833fb0443fbcbe426e8a8c3515ab93e6b11b0f',
    POOL:
        'fc60c7012e7c2019f034e8d46171aa3220cf20c9e7c5afc3728aa56efa5c85ba',
    MODEL / 'adapter_model.safetensors':
        'c02e0fa78a351bff68fd63e6e28128e112a83a42850cbe47ef606168e045cc59',
}

def need(condition, message):
    if not condition:
        raise RuntimeError('FAIL_STOP: ' + message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(
            lambda: f.read(4 * 1024 * 1024), b''
        ):
            h.update(block)
    return h.hexdigest()

def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.pending')
    need(not tmp.exists(), f'Pending JSON exists: {tmp}')

    with tmp.open('x', encoding='utf-8') as f:
        json.dump(
            obj, f,
            ensure_ascii=False,
            sort_keys=True,
            indent=2
        )
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)

def atomic_npz(path, **arrays):
    tmp = path.with_suffix('.npz.pending')
    need(not tmp.exists(), f'Pending block exists: {tmp}')

    with tmp.open('xb') as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)

def load_reference():
    need(
        REFERENCE.is_dir(),
        f'Missing Full-8 reference: {REFERENCE}'
    )

    score_files = list(
        (REFERENCE / 'score_cache').glob('*.scores.npy')
    )
    rank_files = list(
        (REFERENCE / 'ranks').glob('*.ranks.npy')
    )

    need(
        len(score_files) == 1 and len(rank_files) == 1,
        'Expected exactly one reference score/rank pair'
    )

    scores = np.load(
        score_files[0],
        allow_pickle=False
    )
    ranks = np.load(
        rank_files[0],
        allow_pickle=False
    )

    need(
        scores.shape == (8, 15) and ranks.shape == (8,),
        'Reference array shape mismatch'
    )
    need(
        np.isfinite(scores).all()
        and np.issubdtype(ranks.dtype, np.integer),
        'Invalid reference arrays'
    )

    result_path = REFERENCE / 'results.json'
    result = json.loads(result_path.read_text())

    need(
        int(result['alpha=1']['ruby']['n_queries']) == 8,
        'Reference result count mismatch'
    )

    reference_hashes = {
        'score_sha256': sha(score_files[0]),
        'rank_sha256': sha(rank_files[0]),
        'result_sha256': sha(result_path),
    }

    return scores, ranks, reference_hashes

def main():
    print('=' * 78)
    print('WORK1 R1-3 SINGLE-LOAD PERSISTENT CHUNK PARITY')
    print('=' * 78)

    need(
        not (OUT / 'PASS.json').exists(),
        'Pilot already completed'
    )

    print('\n[1] FROZEN PROVENANCE')

    for path, digest in EXPECTED.items():
        need(
            path.is_file() and sha(path) == digest,
            f'Frozen file SHA256 mismatch: {path}'
        )

    need(
        BASE_MODEL.is_dir() and CANON.is_file(),
        'Missing base model or canonical VALID pool'
    )

    frozen = json.loads(FROZEN.read_text())

    need(
        frozen['selected_on_split'] == 'valid',
        'Frozen split mismatch'
    )

    cfg = frozen['config']

    need(
        int(cfg['verify_top_k']) == 15
        and float(cfg['fusion_weight']) == 1.0
        and cfg['normalization'] == 'zscore',
        'Frozen fusion protocol mismatch'
    )

    ref_scores, ref_ranks, ref_hashes = load_reference()

    print('FROZEN_SOURCE_AND_FULL8_REFERENCE=PASS', flush=True)

    print('\n[2] IMPORT FROZEN SCORING FUNCTIONS')

    spec = importlib.util.spec_from_file_location(
        'r13_frozen_valid_scoring',
        EVAL
    )

    need(
        spec is not None and spec.loader is not None,
        'Frozen evaluator import failed'
    )

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rows = mod.read_jsonl(
        DATA / 'ruby' / 'valid.jsonl'
    )
    code_rows = mod.read_jsonl(
        DATA / 'ruby' / 'codebase.jsonl'
    )

    need(len(rows) == 2209, 'Ruby VALID count mismatch')

    queries = [
        mod.get_query_text(row)
        for row in rows[:8]
    ]
    codes = [
        mod.get_code_text(row)
        for row in code_rows
    ]

    with np.load(
        POOL, allow_pickle=False
    ) as pool, np.load(
        CANON, allow_pickle=False
    ) as canonical:

        need(
            pool['indices'].shape == (2209, 100),
            'Pool shape mismatch'
        )

        candidate_indices = np.array(
            pool['indices'][:8],
            dtype=np.int64,
            copy=True
        )
        candidate_scores = np.array(
            pool['scores'][:8],
            dtype=np.float32,
            copy=True
        )
        base_ranks = np.array(
            pool['base_ranks'][:8],
            dtype=np.int64,
            copy=True
        )
        gt_indices = np.array(
            canonical['gt_indices'][:8],
            dtype=np.int64,
            copy=True
        )

        need(
            np.array_equal(
                candidate_indices,
                canonical['candidate_indices'][:8]
            ),
            'Candidate identity mismatch'
        )
        need(
            np.array_equal(
                candidate_scores,
                np.asarray(
                    canonical['candidate_scores'][:8],
                    dtype=np.float32
                )
            ),
            'Retrieval score mismatch'
        )
        need(
            np.array_equal(
                base_ranks,
                canonical['base_ranks'][:8]
            ),
            'Base rank mismatch'
        )

    need(
        np.all(candidate_indices >= 0)
        and np.all(candidate_indices < len(codes)),
        'Invalid candidate code indices'
    )

    print('CANONICAL_VALID_QUERY_ALIGNMENT=PASS', flush=True)

    print('\n[3] GPU AND STORAGE PREFLIGHT')

    controller = runpy.run_path(
        str(CTRL),
        run_name='r13_singleload_preflight'
    )

    controller['gpu_gate']()
    controller['disk_gate']()

    print('GPU_AND_DISK_PREFLIGHT=PASS', flush=True)

    print('\n[4] LOAD MODEL ONCE')

    import torch

    device = torch.device('cuda')

    model, tokenizer = mod.load_verifier(
        str(MODEL),
        str(BASE_MODEL),
        'float16',
        device,
        False,
        'cuda'
    )

    print('SINGLE_MODEL_LOAD=PASS', flush=True)

    yes_id = mod.single_token_id(
        tokenizer,
        [' Yes', 'Yes', ' yes', 'yes'],
        'yes'
    )
    no_id = mod.single_token_id(
        tokenizer,
        [' No', 'No', ' no', 'no'],
        'no'
    )

    print('\n[5] PROCESS AND COMMIT TWO BLOCKS')

    ranges = [(0, 4), (4, 8)]

    for start, stop in ranges:
        path = OUT / f'block_{start:05d}_{stop:05d}.npz'
        metadata = path.with_suffix('.json')

        if path.exists() or metadata.exists():
            need(
                path.is_file() and metadata.is_file(),
                'Incomplete existing checkpoint pair'
            )

            info = json.loads(metadata.read_text())

            need(
                info['sha256'] == sha(path)
                and info['start'] == start
                and info['stop'] == stop,
                'Checkpoint integrity mismatch'
            )

            with np.load(path, allow_pickle=False) as saved:
                need(
                    saved['scores'].shape == (stop-start, 15)
                    and saved['ranks'].shape == (stop-start,),
                    'Existing checkpoint shape mismatch'
                )
                need(
                    np.isfinite(saved['scores']).all(),
                    'Existing checkpoint has nonfinite scores'
                )

            print(
                f'BLOCK_SKIP_VERIFIED={start}:{stop}',
                flush=True
            )
            continue

        prompts = [
            mod.build_yesno_prompt(
                queries[i],
                codes[int(code_idx)],
                512,
                3000
            )
            for i in range(start, stop)
            for code_idx in candidate_indices[i, :15]
        ]

        verifier_scores = mod.score_yesno_prompts(
            model,
            tokenizer,
            prompts,
            yes_id,
            no_id,
            device,
            1,
            1024
        ).reshape(stop - start, 15)

        ranks = mod.rerank_with_verifier(
            candidate_indices[start:stop],
            candidate_scores[start:stop],
            verifier_scores,
            gt_indices[start:stop],
            base_ranks[start:stop],
            15,
            1.0,
            'zscore'
        )

        need(
            np.isfinite(verifier_scores).all()
            and ranks.shape == (stop-start,),
            'Invalid block scores or ranks'
        )

        atomic_npz(
            path,
            scores=np.asarray(
                verifier_scores, dtype=np.float32
            ),
            ranks=np.asarray(
                ranks, dtype=np.int32
            )
        )

        atomic_json(metadata, {
            'start': start,
            'stop': stop,
            'sha256': sha(path),
            'model_sha256': EXPECTED[
                MODEL / 'adapter_model.safetensors'
            ],
            'evaluator_sha256': EXPECTED[EVAL],
            'pool_sha256': EXPECTED[POOL],
        })

        print(
            f'BLOCK_SAVED={start}:{stop} '
            f'SHA256={sha(path)}',
            flush=True
        )

    del model, tokenizer
    torch.cuda.empty_cache()

    print('\n[6] RELOAD CHECKPOINTS AND COMPARE')

    scores = []
    ranks = []
    block_hashes = {}

    for start, stop in ranges:
        path = OUT / f'block_{start:05d}_{stop:05d}.npz'
        metadata = path.with_suffix('.json')

        info = json.loads(metadata.read_text())

        need(
            info['sha256'] == sha(path),
            'Checkpoint SHA256 changed'
        )

        with np.load(path, allow_pickle=False) as saved:
            scores.append(
                np.array(saved['scores'], copy=True)
            )
            ranks.append(
                np.array(saved['ranks'], copy=True)
            )

        block_hashes[f'{start}:{stop}'] = info['sha256']

    joined_scores = np.concatenate(scores, axis=0)
    joined_ranks = np.concatenate(ranks, axis=0)

    score_equal = bool(
        np.array_equal(joined_scores, ref_scores)
    )
    rank_equal = bool(
        np.array_equal(joined_ranks, ref_ranks)
    )

    mrr_equal = (
        mod.compute_metrics_from_ranks(
            joined_ranks
        )['mrr']
        ==
        mod.compute_metrics_from_ranks(
            ref_ranks
        )['mrr']
    )

    max_diff = float(
        np.max(
            np.abs(
                joined_scores.astype(np.float64)
                - ref_scores.astype(np.float64)
            )
        )
    )

    passed = score_equal and rank_equal and mrr_equal

    report = {
        'status': 'PASS' if passed else 'FAIL_STOP',
        'single_model_load': True,
        'split': 'valid',
        'chunks': block_hashes,
        'reference_sha256': ref_hashes,
        'score_bitwise_parity': score_equal,
        'rank_bitwise_parity': rank_equal,
        'mrr_parity': mrr_equal,
        'max_absolute_score_difference': max_diff,
        'test_evaluated': False,
    }

    atomic_json(
        OUT / 'parity_report.json',
        report
    )

    print(
        'SINGLELOAD_SCORE_BITWISE_PARITY=',
        score_equal,
        flush=True
    )
    print(
        'SINGLELOAD_RANK_BITWISE_PARITY=',
        rank_equal,
        flush=True
    )
    print(
        'SINGLELOAD_MRR_PARITY=',
        mrr_equal,
        flush=True
    )
    print(
        'MAX_ABS_SCORE_DIFF=',
        max_diff,
        flush=True
    )

    need(
        passed,
        'Single-load scoring differs from frozen Full-8'
    )

    atomic_json(OUT / 'PASS.json', report)

    print(
        'R13_SINGLELOAD_CHECKPOINT_PARITY=PASS',
        flush=True
    )
    print('TEST_EVALUATED=NO', flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        print(
            'R13_SINGLELOAD_CHECKPOINT_PARITY=FAIL_STOP',
            flush=True
        )
        sys.exit(2)
