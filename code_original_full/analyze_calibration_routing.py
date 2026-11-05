#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from revision_utils import write_json


def softmax_entropy(x: np.ndarray) -> np.ndarray:
    z=x-x.max(axis=1,keepdims=True); e=np.exp(z); p=e/np.clip(e.sum(axis=1,keepdims=True),1e-12,None)
    return -(p*np.log(np.clip(p,1e-12,None))).sum(axis=1)


def features(scores:np.ndarray)->Dict[str,np.ndarray]:
    s=np.asarray(scores,dtype=np.float64)
    k=s.shape[1]
    s1=s[:,0]; s2=s[:,min(1,k-1)]; s5=s[:,min(4,k-1)]; s10=s[:,min(9,k-1)]
    top10=s[:,:min(10,k)]; std=np.std(top10,axis=1); rng=np.ptp(top10,axis=1)
    return {
        'neg_margin_1_2':-(s1-s2),
        'neg_margin_1_5':-(s1-s5),
        'neg_margin_1_10':-(s1-s10),
        'normalized_neg_margin_1_2':-(s1-s2)/np.clip(std,1e-8,None),
        'entropy_top10':softmax_entropy(top10),
        'variance_top10':np.var(top10,axis=1),
        'inverse_range_top10':-rng,
    }


def safe_auc(y,score)->Tuple[float,float]:
    if len(np.unique(y))<2: return float('nan'),float('nan')
    return float(roc_auc_score(y,score)),float(average_precision_score(y,score))


def quantile_bins(signal:np.ndarray, outcome:np.ndarray, n_bins:int):
    edges=np.quantile(signal,np.linspace(0,1,n_bins+1)); rows=[]
    for i in range(n_bins):
        lo,hi=edges[i],edges[i+1]
        mask=(signal>=lo)&(signal<=hi) if i==n_bins-1 else (signal>=lo)&(signal<hi)
        if not mask.any(): continue
        rows.append({'bin':i,'lo':float(lo),'hi':float(hi),'n':int(mask.sum()),'mean_signal':float(signal[mask].mean()),'empirical_rate':float(outcome[mask].mean())})
    return rows


def risk_coverage(signal:np.ndarray, correct:np.ndarray, coverages):
    # score high = uncertain, so defer highest-score cases; retained set contains low uncertainty.
    order=np.argsort(signal)
    rows=[]; n=len(signal)
    for cov in coverages:
        keep=max(1,int(round(n*cov)))
        idx=order[:keep]
        acc=float(correct[idx].mean())
        rows.append({'coverage':float(cov),'n_kept':len(idx),'top1_accuracy':acc,'risk':1.0-acc})
    return rows


def main():
    ap=argparse.ArgumentParser(description='Reviewer-requested calibration diagnostics for dense retrieval uncertainty and optional verifier utility.')
    ap.add_argument('--candidate-pool',required=True,help='.npz with scores and base_ranks')
    ap.add_argument('--verifier-ranks',default='',help='Optional final 0-based rank array for full verifier.')
    ap.add_argument('--bins',type=int,default=10)
    ap.add_argument('--output-dir',required=True)
    args=ap.parse_args()
    pool=np.load(args.candidate_pool,allow_pickle=False); scores=np.asarray(pool['scores']); base=np.asarray(pool['base_ranks'],dtype=np.int64)
    if len(scores)!=len(base): raise ValueError('scores/base_ranks length mismatch')
    top1_correct=(base==0).astype(np.int64); top1_error=1-top1_correct
    feats=features(scores)
    report={'n_queries':len(base),'top1_accuracy':float(top1_correct.mean()),'signals':{},'risk_coverage':{}}
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    for name,signal in feats.items():
        auroc,auprc=safe_auc(top1_error,signal)
        bins=quantile_bins(signal,top1_error,args.bins)
        report['signals'][name]={'auroc_error':auroc,'auprc_error':auprc,'bins':bins}
        report['risk_coverage'][name]=risk_coverage(signal,top1_correct,[.2,.3,.4,.5,.6,.7,.8,.9,1.0])
    # Margin reliability explicitly requested by reviewer: report top-1 accuracy as positive margin grows.
    margin=scores[:,0]-scores[:,1]
    report['margin_reliability']=quantile_bins(margin,top1_correct,args.bins)

    if args.verifier_ranks:
        vr=np.load(args.verifier_ranks).astype(np.int64)
        if vr.shape!=base.shape: raise ValueError(f'verifier/base rank shape mismatch {vr.shape} vs {base.shape}')
        dense_rr=1.0/(base+1.0); ver_rr=1.0/(vr+1.0); delta=ver_rr-dense_rr
        beneficial=(delta>0).astype(np.int64); harmful=(delta<0).astype(np.int64)
        report['verification_utility']={'mean_delta_rr':float(delta.mean()),'beneficial_rate':float(beneficial.mean()),'harmful_rate':float(harmful.mean()),'unchanged_rate':float((delta==0).mean()),'signals':{}}
        for name,signal in feats.items():
            # For utility prediction, uncertainty may or may not imply benefit; report empirical discrimination without forcing sign.
            a1,p1=safe_auc(beneficial,signal); a2,p2=safe_auc(harmful,signal)
            report['verification_utility']['signals'][name]={'auroc_benefit':a1,'auprc_benefit':p1,'auroc_harm':a2,'auprc_harm':p2}
        np.savez_compressed(out/'utility_labels.npz',delta_rr=delta.astype(np.float32),beneficial=beneficial,harmful=harmful,base_ranks=base,verifier_ranks=vr,**{k:v.astype(np.float32) for k,v in feats.items()})

    write_json(out/'calibration_report.json',report)
    with (out/'margin_reliability.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['bin','lo','hi','n','mean_signal','empirical_rate']); w.writeheader(); w.writerows(report['margin_reliability'])
    print(json.dumps({k:v for k,v in report.items() if k not in ('signals','risk_coverage')},indent=2))
    print(f'Saved calibration report to {out}')

if __name__=='__main__': main()
