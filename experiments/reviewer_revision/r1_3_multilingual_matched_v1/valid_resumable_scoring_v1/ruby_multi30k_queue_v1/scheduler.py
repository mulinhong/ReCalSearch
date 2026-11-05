from pathlib import Path
import fcntl, hashlib, json, os, re
import subprocess, sys, time, traceback
import numpy as np

ROOT = Path('/root/autodl-tmp/lora-code-embeddings-main/text2code')
BASE = ROOT / 'results/reviewer_revision/r1_3_multilingual_matched_v1'
WORK = BASE / 'valid_resumable_scoring_v1'
QUEUE = WORK / 'ruby_multi30k_queue_v1'
WORKER = WORK / 'worker.py'
PLAN = Path('/root/autodl-tmp/recal_r13_valid_plan_MiFZlVso/evaluation_jobs.json')
STATS = Path('/root/autodl-tmp/recal_r13_ruby3_stats_vWjJWbnb/ruby_three_seed_report.json')
ARCHIVE = Path('/root/autodl-tmp/recal_r13_ruby3_stats_vWjJWbnb/upload_ruby_three_seed_statistics.tar.gz')
POOL = BASE / 'validation_pool_adapter_v2/candidate_pools/ruby_valid_nomic_top100.npz'

JOBS = [f'multi30k_seed{s}_ruby' for s in (13, 42, 87)]

EXPECTED = {
    str(WORKER): 'c34b8666779f316c540453823ef4e50e30823b1f84eda2d1ad9f817ee130e22b',
    str(PLAN): '5d6d038e86e2cca490ceb9fe9f9c4ffcbec95be6f5cd74527ff63da3110fd1fb',
    str(ARCHIVE): 'b7a9356023842feb02b4d2ebace3c3c242a1869b2b10e48650a0ffb2322353c2',
    str(POOL): 'fc60c7012e7c2019f034e8d46171aa3220cf20c9e7c5afc3728aa56efa5c85ba',
}

def need(ok, msg):
    if not ok:
        raise RuntimeError('FAIL_STOP: ' + str(msg))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4194304), b''):
            h.update(block)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, data):
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())

def process_gate():
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = [
                x.decode(errors='replace')
                for x in (proc/'cmdline').read_bytes().split(b'\0')
                if x
            ]
        except OSError:
            continue
        if not args or 'python' not in Path(args[0]).name.lower():
            continue
        command = ' '.join(args[1:])
        if any(token in command for token in (
            'valid_resumable_scoring_v1/worker.py',
            'ruby_triseed_queue_v1/scheduler.py',
            'run_mono5k_serial_v2.py',
            'train_qwen_verifier.py',
            'eval_valid_only.py',
            'speed_audit.py',
        )):
            raise RuntimeError(
                f'FAIL_STOP: conflicting PID={proc.name} {command[:200]}'
            )

def verify_final(job):
    final = WORK / job / 'final'
    need(final.is_dir(), 'Missing final ' + job)
    c = read_json(final/'commit.json')
    need(c['job_id'] == job and c['n_queries'] == 2209,
         'Final identity ' + job)

    for field, name in (
        ('scores_sha256','scores.npy'),
        ('ranks_sha256','ranks.npy'),
        ('result_sha256','results.json'),
    ):
        need(sha(final/name) == c[field],
             'Final file hash ' + job + ' ' + name)

    ranks = np.load(final/'ranks.npy', allow_pickle=False)
    scores = np.load(final/'scores.npy', mmap_mode='r', allow_pickle=False)
    need(ranks.shape == (2209,) and scores.shape == (2209,15),
         'Final array shape ' + job)
    need(np.issubdtype(ranks.dtype,np.integer)
         and (ranks >= 0).all() and np.isfinite(scores).all(),
         'Invalid final arrays ' + job)

    metrics = read_json(final/'results.json')['alpha=1']['ruby']
    mrr = 100.0 * (1.0/(ranks.astype(np.float64)+1)).mean()
    need(abs(mrr - float(metrics['mrr'])) <= 0.00005001,
         'MRR replay mismatch ' + job)

    return {
        'job_id':job,
        'mrr_percent_exact':float(mrr),
        'scores_sha256':c['scores_sha256'],
        'ranks_sha256':c['ranks_sha256'],
        'result_sha256':c['result_sha256'],
    }

def worker_check(job, index, env, phase):
    path = QUEUE / f'{phase}_{index}_{job}.log'
    with path.open('x') as f:
        r = subprocess.run(
            [sys.executable,'-u','-B',str(WORKER),
             '--job-id',job,'--verify-only'],
            cwd=ROOT, env=env,
            stdout=f, stderr=subprocess.STDOUT
        )
    need(r.returncode == 0,
         f'{phase} failed for {job}; see {path}')
    if phase == 'postflight':
        need('R13_VALID_JOB_FINAL=PASS_REUSED'
             in path.read_text(errors='replace'),
             'Final replay marker missing ' + job)
    print(f'{phase.upper()}_PASS={job}',flush=True)

def progress_tail(log):
    try:
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-12000))
            text=f.read().decode(errors='replace')
        hits=re.findall(r'Verifier scoring:[^\r\n]*',text)
        if hits:
            return hits[-1][:120]
        lines=[x for x in text.splitlines() if x.strip()]
        return lines[-1][:120] if lines else 'starting'
    except OSError:
        return 'log unavailable'

def main():
    print('='*72,flush=True)
    print('R1-3 RUBY MULTI30K VALID THREE-SEED QUEUE',flush=True)
    print('='*72,flush=True)

    for path, digest in EXPECTED.items():
        need(Path(path).is_file() and sha(path)==digest,
             'Frozen fingerprint ' + path)

    need(STATS.is_file(), 'Ruby three-seed audit missing')
    old=read_json(STATS)
    need(old.get('status')=='PASS'
         and len(old.get('cells',[]))==6
         and old.get('test_accessed') is False,
         'Existing Ruby statistical audit mismatch')

    rows=read_json(PLAN)
    need(len(rows)==54, 'Frozen evaluation count')
    mapping={x['job_id']:x for x in rows}
    need(len(mapping)==54, 'Duplicate evaluation IDs')

    for seed, job in zip((13,42,87),JOBS):
        need(job in mapping, 'Missing job '+job)
        entry=mapping[job]
        need(entry['group']=='multi30k'
             and entry['language']=='ruby'
             and int(entry['seed'])==seed
             and int(entry['expected_queries'])==2209,
             'Wrong plan cell '+job)

        model=Path(entry['model_path'])
        cfg=model/'verifier_config.json'
        need(cfg.is_file()
             and sha(cfg)==entry['model_config_sha256'],
             'Model config hash '+job)

        config=read_json(cfg)
        need(int(config.get('seed',-1))==seed
             and int(config.get('num_training_examples',-1))==270000,
             'Multi30k seed/row count '+job)

        need((model/'adapter_model.safetensors').is_file()
             and (model/'adapter_config.json').is_file(),
             'Adapter missing '+job)

    process_gate()

    print('FROZEN_SOURCE_AND_RUBY_RESULTS=PASS',flush=True)
    print('MULTI30K_MODEL_PROVENANCE=PASS',flush=True)

    write_json(QUEUE/'manifest.json',{
        'jobs':JOBS,
        'worker_sha256':sha(WORKER),
        'plan_sha256':sha(PLAN),
        'split':'valid',
        'batch_size':1,
        'max_new_chunks':0,
        'test_accessed':False,
    })

    env=os.environ.copy()
    env['R13_VALID_WORK_ROOT']=str(WORK)
    env['CUDA_VISIBLE_DEVICES']='0'
    env['TOKENIZERS_PARALLELISM']='false'
    env['PYTHONDONTWRITEBYTECODE']='1'
    env['OMP_NUM_THREADS']='4'
    env['PYTHONPATH']=str(ROOT)+os.pathsep+env.get('PYTHONPATH','')

    for i,job in enumerate(JOBS,1):
        worker_check(job,i,env,'preflight')

    print('ALL_THREE_JOB_PREFLIGHT=PASS',flush=True)

    results=[]
    for i,job in enumerate(JOBS,1):
        need(sha(WORKER)==EXPECTED[str(WORKER)]
             and sha(PLAN)==EXPECTED[str(PLAN)],
             'Worker/plan changed during queue')

        print(
            f'QUEUE_POSITION={i}/3 JOB={job} '
            f'START={time.strftime("%Y-%m-%dT%H:%M:%S%z")}',
            flush=True
        )

        log=QUEUE/f'job_{i}_{job}.log'

        with log.open('x') as f:
            p=subprocess.Popen(
                [sys.executable,'-u','-B',str(WORKER),
                 '--job-id',job,'--max-new-chunks','0'],
                cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,
                stdout=f,stderr=subprocess.STDOUT
            )

            while p.poll() is None:
                chunkdir=WORK/job/'chunks'
                count=len(list(chunkdir.glob('q*'))) if chunkdir.is_dir() else 0
                print(
                    f'LIVE_JOB={job} BLOCK_DIRS={count}/5 '
                    f'PROGRESS={progress_tail(log)}',
                    flush=True
                )
                time.sleep(30)

            code=p.wait()

        need(code==0,
             f'Worker failed: {job}, rc={code}, log={log}')

        worker_check(job,i,env,'postflight')
        result=verify_final(job)
        results.append(result)

        print(
            f'JOB_FINAL_PASS={job} '
            f'MRR={result["mrr_percent_exact"]:.8f}',
            flush=True
        )

    write_json(QUEUE/'final_report.json',{
        'status':'PASS',
        'jobs':results,
        'test_accessed':False,
        'worker_sha256':sha(WORKER),
        'plan_sha256':sha(PLAN),
    })

    print('R13_RUBY_MULTI30K_THREE_SEED_QUEUE=PASS',flush=True)
    print('RUBY_ALL_THREE_FAMILIES=9/9',flush=True)
    print('TEST_ACCESSED=NO',flush=True)

if __name__=='__main__':
    try:
        with (QUEUE/'scheduler.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            main()
    except Exception:
        import traceback
        traceback.print_exc()
        print('R13_RUBY_MULTI30K_THREE_SEED_QUEUE=FAIL_STOP',flush=True)
        sys.exit(2)
