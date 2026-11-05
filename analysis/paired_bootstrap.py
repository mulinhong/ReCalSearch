from pathlib import Path
import argparse,csv,json
import numpy as np

LANGS=("ruby","javascript","php","python","java","go")

def reciprocal(a):
    result=np.zeros(len(a),dtype=np.float64)
    mask=(a>=0)&(a<100)
    result[mask]=1.0/(a[mask]+1)
    return result

def bootstrap(delta,rng,B):
    n=len(delta)
    samples=np.empty(B,dtype=np.float64)
    for start in range(0,B,12):
        batch=min(12,B-start)
        ix=rng.integers(0,n,size=(batch,n))
        samples[start:start+batch]=100*delta[ix].mean(axis=1)
    return samples

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,
        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--replicates",type=int,default=3000)
    parser.add_argument("--seed",type=int,default=20261104)
    a=parser.parse_args()
    rng=np.random.default_rng(a.seed)
    rows=[]
    draws=[]
    deltas=[]
    for lang in LANGS:
        folder=a.root/"derived_ranks"
        base=np.load(folder/f"{lang}_nomic.ranks.npy",
                     allow_pickle=False)
        rec=np.load(folder/f"{lang}_recal.ranks.npy",
                    allow_pickle=False)
        assert base.shape==rec.shape
        delta=reciprocal(rec)-reciprocal(base)
        bs=bootstrap(delta,rng,a.replicates)
        low,high=np.percentile(bs,[2.5,97.5])
        rows.append(dict(language=lang,queries=len(delta),
            delta_mrr_pp=float(100*delta.mean()),
            ci95_low_pp=float(low),ci95_high_pp=float(high)))
        draws.append(bs)
        deltas.append(delta)
    macro=np.mean(np.stack(draws),axis=0)
    pooled=np.concatenate(deltas)
    micro=bootstrap(pooled,rng,a.replicates)
    result=dict(
        per_language=rows,
        macro_delta_mrr_pp=float(np.mean(
            [x["delta_mrr_pp"] for x in rows])),
        macro_ci95=list(map(float,np.percentile(macro,[2.5,97.5]))),
        micro_delta_mrr_pp=float(100*pooled.mean()),
        micro_ci95=list(map(float,np.percentile(micro,[2.5,97.5]))),
        replicates=a.replicates,
        seed=a.seed)
    print(json.dumps(result,indent=2))
if __name__=="__main__":
    main()
