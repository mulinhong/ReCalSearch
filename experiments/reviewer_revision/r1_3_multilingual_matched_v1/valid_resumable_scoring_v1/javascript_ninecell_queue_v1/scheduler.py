from pathlib import Path
import os, sys, json, hashlib, fcntl, subprocess, time, re, traceback
import numpy as np
ROOT=Path('/root/autodl-tmp/lora-code-embeddings-main/text2code')
BASE=ROOT/'results/reviewer_revision/r1_3_multilingual_matched_v1'
WORK=BASE/'valid_resumable_scoring_v1'
QUEUE=WORK/'javascript_ninecell_queue_v1'
WORKER=WORK/'worker.py'
PLAN=Path('/root/autodl-tmp/recal_r13_valid_plan_MiFZlVso/evaluation_jobs.json')
POOLPLAN=Path('/root/autodl-tmp/recal_r13_valid_plan_MiFZlVso/valid_pools_manifest.json')
EVAL=BASE/'full_valid_multi5k_seed13_v1/eval_valid_only.py'
FROZEN=ROOT/'results/reviewer_revision/recal_valid_fresh_20260911/recal_main_frozen_config.json'
CANON=ROOT/'results/reviewer_revision/recal_valid_fresh_20260911/validation_grid/candidate_pools/javascript_valid_nomic_top100.npz'
RUBY9=Path('/root/autodl-tmp/recal_r13_ruby9_scale_Ap4Jqvhi/upload_ruby_ninecell_scale_audit.tar.gz')
LANG='javascript'; N=8253; PAIRS=N*15; SEEDS=(13,42,87); FAMILIES=('mono5k','multi5k','multi30k')
JOBS=[f'{group}_seed{seed}_{LANG}' for seed in SEEDS for group in FAMILIES]
EXPECTED={
    WORKER:'c34b8666779f316c540453823ef4e50e30823b1f84eda2d1ad9f817ee130e22b',
    PLAN:'5d6d038e86e2cca490ceb9fe9f9c4ffcbec95be6f5cd74527ff63da3110fd1fb',
    EVAL:'08692a72ab8b6e46a1ce99f83d762cea02a933db4bd7cb54840c5877d59163bb',
    FROZEN:'70789e10518f32a97f5990b234833fb0443fbcbe426e8a8c3515ab93e6b11b0f',
    RUBY9:'f7bcaaf7214b123082a516c7fe899e81ab86cfccf0ac2466608cf5765f7f1c96',
}
def need(ok, reason):
    if not ok: raise RuntimeError('FAIL_STOP: '+str(reason))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''): h.update(b)
    return h.hexdigest()
def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
def atomic_json(path, obj):
    temp=path.with_suffix('.tmp')
    need(not path.exists() and not temp.exists(), 'Refuse overwrite: '+str(path))
    with temp.open('x',encoding='utf-8') as f:
        json.dump(obj,f,indent=2,ensure_ascii=False,sort_keys=True)
        f.write('\n');f.flush();os.fsync(f.fileno())
    os.rename(temp,path)
def check_sources():
    for path,digest in EXPECTED.items():
        need(path.is_file() and sha(path)==digest,'Frozen SHA256: '+str(path))
    need(POOLPLAN.is_file() and CANON.is_file(),'Frozen JS pool missing')
    plans=load(PLAN)
    need(len(plans)==54 and len({j['job_id'] for j in plans})==54, '54-job plan')
    mapping={j['job_id']:j for j in plans}
    pools=load(POOLPLAN); p=pools[LANG]
    pool=Path(p['path'])
    need(pool.is_file() and sha(pool)==p['sha256'],'JavaScript pool SHA256')
    need(int(p['expected_queries'])==N,'JavaScript query count')
    with np.load(pool,allow_pickle=False) as a,np.load(CANON,allow_pickle=False) as b:
        need(a['indices'].shape==(N,100),'Pool dimensions')
        need(np.array_equal(a['indices'],b['candidate_indices']),'Canonical indices')
        need(np.array_equal(a['scores'].astype(np.float32),b['candidate_scores'].astype(np.float32)),'Canonical scores')
        need(np.array_equal(a['base_ranks'],b['base_ranks']),'Canonical base ranks')
        need(b['gt_indices'].shape==(N,),'Canonical GT count')
    for seed in SEEDS:
        for family in FAMILIES:
            jid=f'{family}_seed{seed}_{LANG}'
            need(jid in mapping,'Missing job '+jid)
            job=mapping[jid]
            need(job['group']==family and int(job['seed'])==seed and job['language']==LANG,'Job coordinates '+jid)
            need(int(job['expected_queries'])==N and int(job['expected_pair_scores'])==PAIRS,'Job query/pair count '+jid)
            m=Path(job['model_path'])
            need((m/'adapter_model.safetensors').is_file() and (m/'adapter_config.json').is_file(),'Adapter missing '+jid)
            need(sha(m/'verifier_config.json')==job['model_config_sha256'],'Model config SHA256 '+jid)
    return mapping,pool

def conflict_gate():
    tokens=('valid_resumable_scoring_v1/worker.py',
            'ruby_multi30k_queue_v1/scheduler.py','ruby_triseed_queue_v1/scheduler.py',
            'train_qwen_verifier.py','eval_valid_only.py','run_mono5k_serial_v2.py')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            args=[v.decode(errors='replace') for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
        except OSError:continue
        if not args or 'python' not in Path(args[0]).name.lower():continue
        cmd=' '.join(args[1:])
        need(not any(t in cmd for t in tokens), f'Competing PID {proc.name}: {cmd[:180]}')

def verify_job(jid,job,pool):
    folder=WORK/jid; final=folder/'final'
    identity=load(folder/'identity.json'); meta=load(final/'commit.json')
    need(identity['job_id']==jid and identity['split']=='valid' and int(identity['n_queries'])==N,'Job identity '+jid)
    need(identity['pool_sha256']==sha(pool) and identity['evaluator_sha256']==sha(EVAL),'Frozen pool/evaluator identity '+jid)
    need(identity['model_config_sha256']==job['model_config_sha256'],'Frozen model identity '+jid)
    fp=hashlib.sha256((json.dumps(identity,sort_keys=True,ensure_ascii=False,indent=2)+'\n').encode()).hexdigest()
    need(meta['fingerprint']==fp and meta['job_id']==jid and int(meta['n_queries'])==N,'Final fingerprint '+jid)
    for key,name in [('scores_sha256','scores.npy'),('ranks_sha256','ranks.npy'),('result_sha256','results.json')]:
        need(meta[key]==sha(final/name),'Final hash '+jid+' '+name)
    ss=np.load(final/'scores.npy',mmap_mode='r',allow_pickle=False)
    rr=np.load(final/'ranks.npy',allow_pickle=False)
    need(ss.shape==(N,15) and ss.dtype==np.float32 and np.isfinite(ss).all(),'Scores invalid '+jid)
    need(rr.shape==(N,) and np.issubdtype(rr.dtype,np.integer) and (rr>=0).all(),'Ranks invalid '+jid)
    metrics=load(final/'results.json')['alpha=1'][LANG]
    exact=100.0*(1.0/(rr.astype(np.float64)+1)).mean()
    need(int(metrics['n_queries'])==N and metrics['candidate_pool']=='r13_validation_v2','Metric protocol '+jid)
    need(abs(exact-float(metrics['mrr']))<0.00005001,'MRR replay '+jid)
    return {'job_id':jid,'mrr_percent_exact':float(exact),'result_sha256':meta['result_sha256'],'ranks_sha256':meta['ranks_sha256']}

def tail_progress(path):
    try:
        with path.open('rb') as f:
            f.seek(max(0,path.stat().st_size-16384));s=f.read().decode(errors='replace')
        hits=re.findall(r'Verifier scoring:[^\r\n]*',s)
        if hits:return hits[-1][-95:]
        lines=[v for v in s.splitlines() if v.strip()]
        return lines[-1][-100:] if lines else 'STARTING'
    except OSError:return 'LOG_NOT_READY'

def do_worker(jid,kind,i,env):
    fname=QUEUE/f'{kind}_{i:02d}_{jid}.log'
    cmd=[sys.executable,'-u','-B',str(WORKER),'--job-id',jid]
    cmd+=['--verify-only'] if kind!='job' else ['--max-new-chunks','0']
    with fname.open('x') as f:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        if kind=='job':
            while proc.poll() is None:
                chunks=WORK/jid/'chunks'
                count=len([p for p in chunks.glob('q*') if p.is_dir()]) if chunks.is_dir() else 0
                pct=min(100.0,100.0*count/((N+511)//512))
                print(f'LIVE_JOB={jid} COMMITTED_BLOCKS={count}/17 {pct:.1f}% LOG_PROGRESS={tail_progress(fname)}',flush=True)
                time.sleep(30)
        rc=proc.wait()
    need(rc==0, f'{kind} worker failed job={jid} rc={rc}, log={fname}')
    print(f'{kind.upper()}_PASS={jid}',flush=True)

def main():
    print('R13_JS_NINECELL_QUEUE=v1',flush=True)
    print('PHASE=1/3 FROZEN_AND_MODEL_PREFLIGHT',flush=True)
    mapping,pool=check_sources()
    print('FROZEN_SOURCE_AND_JS_CANONICAL_POOL=PASS',flush=True)
    conflict_gate()
    # Zero-byte availability fields can be misleading; require actual durable write.
    check=QUEUE/'.fsync_write_gate.tmp'
    with check.open('xb') as f:
        f.write(b'R13_JS_QUEUE_WRITE_GATE\n');f.flush();os.fsync(f.fileno())
    check.unlink()
    print('CONCURRENT_GPU_PROCESS_AND_PERSISTENT_WRITE=PASS',flush=True)
    atomic_json(QUEUE/'queue_manifest.json',{'language':LANG,'jobs':JOBS,'split':'valid','work_sha256':sha(WORKER),'plan_sha256':sha(PLAN),'pool_sha256':sha(pool),'candidate_top_k':100,'verify_top_k':15,'batch_size':1,'n_queries_each':N,'test_accessed':False})
    env=os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES':'0','TOKENIZERS_PARALLELISM':'false','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','R13_VALID_WORK_ROOT':str(WORK),'PYTHONPATH':str(ROOT)+os.pathsep+env.get('PYTHONPATH','')})
    for i,jid in enumerate(JOBS,1):
        do_worker(jid,'preflight',i,env)
    print('ALL_NINE_JOB_PREFLIGHT=PASS',flush=True)
    print('PHASE=2/3 VALID_ONLY_SERIAL_INFERENCE',flush=True)
    records=[]
    for i,jid in enumerate(JOBS,1):
        for path,digest in EXPECTED.items():
            need(path.is_file() and sha(path)==digest,'Frozen source changed: '+str(path))
        print(f'QUEUE_POSITION={i}/9 JOB={jid} START={time.strftime("%Y-%m-%dT%H:%M:%S%z")}',flush=True)
        do_worker(jid,'job',i,env)
        do_worker(jid,'postflight',i,env)
        row=verify_job(jid,mapping[jid],pool)
        records.append(row)
        print(f'JOB_FINAL_PASS={jid} MRR={row["mrr_percent_exact"]:.8f} RESULT_SHA256={row["result_sha256"]}',flush=True)
    print('PHASE=3/3 RESULTS_AND_EVIDENCE',flush=True)
    atomic_json(QUEUE/'final_report.json',{'status':'PASS','cells':records,'language':LANG,'n_queries':N,'frozen_worker_sha256':sha(WORKER),'pool_sha256':sha(pool),'test_accessed':False})
    print('R13_JS_NINECELL_QUEUE=PASS',flush=True)
    print('TOTAL_VALID_COMPLETE=18/54',flush=True)
    print('TEST_ACCESSED=NO GPU_TRAINING_STARTED=NO',flush=True)

if __name__=='__main__':
    try:
        with (QUEUE/'scheduler.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            main()
    except Exception:
        traceback.print_exc()
        print('R13_JS_NINECELL_QUEUE=FAIL_STOP',flush=True)
        sys.exit(2)
