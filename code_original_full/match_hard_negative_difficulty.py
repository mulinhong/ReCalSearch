#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from revision_utils import write_json


def read_jsonl(path:Path)->List[Dict[str,Any]]:
    rows=[]
    with path.open('r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line: rows.append(json.loads(line))
    return rows


def describe(x:List[float])->Dict[str,float]:
    a=np.asarray(x,dtype=np.float64)
    if a.size==0: return {'n':0}
    q=np.quantile(a,[0.1,0.25,0.5,0.75,0.9])
    return {'n':int(a.size),'mean':float(a.mean()),'std':float(a.std()),'p10':float(q[0]),'p25':float(q[1]),'p50':float(q[2]),'p75':float(q[3]),'p90':float(q[4])}


def main():
    ap=argparse.ArgumentParser(description='Difficulty-match cross-retriever hard negatives to deployed Nomic hard negatives using Nomic similarity.')
    ap.add_argument('--reference-nomic-groups',required=True)
    ap.add_argument('--source-groups',required=True)
    ap.add_argument('--num-negatives',type=int,default=7)
    ap.add_argument('--output-file',required=True)
    ap.add_argument('--report-file',required=True)
    args=ap.parse_args()

    ref={r['query_id']:r for r in read_jsonl(Path(args.reference_nomic_groups))}
    src={r['query_id']:r for r in read_jsonl(Path(args.source_groups))}
    common=sorted(set(ref)&set(src))
    out=[]; target_all=[]; matched_all=[]; abs_err=[]; dropped=0
    for qid in common:
        rr=ref[qid]; ss=src[qid]
        target=[float(x) for x in (rr.get('negative_retrieval_scores') or [])[:args.num_negatives]]
        dep=[float(x) for x in (ss.get('negative_deployed_nomic_scores') or [])]
        ids=list(ss.get('negative_code_indices') or [])
        ranks=list(ss.get('negative_retrieval_ranks') or [])
        source_scores=list(ss.get('negative_retrieval_scores') or [])
        if len(target)<args.num_negatives or len(dep)<args.num_negatives:
            dropped+=1; continue
        available=set(range(len(dep))); chosen=[]
        for t in target:
            j=min(available,key=lambda k:abs(dep[k]-t)); available.remove(j); chosen.append(j)
        row=dict(ss)
        row['format']='grouped_indices_v2_matched'
        row['difficulty_reference']='nomic_deployed_similarity'
        row['difficulty_reference_file']=str(Path(args.reference_nomic_groups).resolve())
        row['negative_code_indices']=[int(ids[j]) for j in chosen]
        row['negative_retrieval_ranks']=[int(ranks[j]) if j<len(ranks) else j+1 for j in chosen]
        row['negative_retrieval_scores']=[float(source_scores[j]) if j<len(source_scores) else None for j in chosen]
        row['negative_deployed_nomic_scores']=[float(dep[j]) for j in chosen]
        row['matched_reference_nomic_scores']=target
        out.append(row)
        target_all.extend(target); matched_all.extend(row['negative_deployed_nomic_scores'])
        abs_err.extend(abs(a-b) for a,b in zip(target,row['negative_deployed_nomic_scores']))

    op=Path(args.output_file); op.parent.mkdir(parents=True,exist_ok=True)
    with op.open('w',encoding='utf-8') as f:
        for r in out: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    report={
        'protocol_version':'recal_hn_difficulty_match_v1',
        'reference_file':str(Path(args.reference_nomic_groups).resolve()),
        'source_file':str(Path(args.source_groups).resolve()),
        'common_queries':len(common),'saved_queries':len(out),'dropped_queries':dropped,
        'num_negatives_per_query':args.num_negatives,
        'reference_nomic_similarity':describe(target_all),
        'matched_source_nomic_similarity':describe(matched_all),
        'absolute_similarity_error':describe(abs_err),
    }
    write_json(args.report_file,report)
    print(json.dumps(report,indent=2))
    print(f'Saved matched groups: {op}')

if __name__=='__main__':
    main()
