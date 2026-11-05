#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import create_dataset
from eval_llm_verifier import (
    average_metric_results, pooled_metric_results, compute_metrics_from_ranks,
    normalize_scores, read_jsonl, retrieve_topk_from_cache, text2code_to_array,
)
from eval_nomic_embed import coderank_cache_paths, get_code_text, get_query_text, load_cache_array
from revision_utils import enforce_frozen_test_protocol, load_json, save_frozen_config, write_environment_manifest

DEFAULT_T2C_INSTRUCTION='Given a natural language programming task, retrieve code that correctly solves or implements the task.'
PREFIX='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def format_instruction(instruction:str,query:str,doc:str)->str:
    return f'<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}'


def load_model(path:str,dtype:str,device_map:str,trust:bool):
    tok=AutoTokenizer.from_pretrained(path,padding_side='left',trust_remote_code=trust,use_fast=False)
    dt={'float16':torch.float16,'bfloat16':torch.bfloat16,'float32':torch.float32}[dtype]
    kwargs={'torch_dtype':dt,'trust_remote_code':trust,'low_cpu_mem_usage':True}
    if device_map=='cuda': kwargs['device_map']={'':0}
    elif device_map=='auto': kwargs['device_map']='auto'
    model=AutoModelForCausalLM.from_pretrained(path,**kwargs).eval()
    if device_map=='none': model=model.cuda()
    return model,tok


@torch.no_grad()
def score_pairs(model,tok,pairs:List[str],batch_size:int,max_length:int)->np.ndarray:
    false_id=tok.convert_tokens_to_ids('no'); true_id=tok.convert_tokens_to_ids('yes')
    if false_id is None or true_id is None: raise ValueError('Qwen3 reranker tokenizer lacks yes/no tokens')
    pre=tok.encode(PREFIX,add_special_tokens=False); suf=tok.encode(SUFFIX,add_special_tokens=False)
    out=[]
    for st in tqdm(range(0,len(pairs),batch_size),desc='Qwen3 reranker scoring'):
        texts=pairs[st:st+batch_size]
        enc=tok(texts,padding=False,truncation='longest_first',return_attention_mask=False,max_length=max_length-len(pre)-len(suf))
        for i,ids in enumerate(enc['input_ids']): enc['input_ids'][i]=pre+ids+suf
        enc=tok.pad(enc,padding=True,return_tensors='pt',max_length=max_length)
        enc={k:v.to(model.device) for k,v in enc.items()}
        logits=model(**enc).logits[:,-1,:].float()
        # Raw logit difference is monotonic with yes probability and matches the official reranker ranking interface.
        out.append((logits[:,true_id]-logits[:,false_id]).cpu().numpy().astype(np.float32))
        del enc,logits
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return np.concatenate(out)


def rerank(indices,scores,rr_scores,gt,base_ranks,topk,fusion_weight,normalization):
    final=base_ranks.copy()
    for i in range(len(indices)):
        cand=indices[i]; d=scores[i,:topk]; r=rr_scores[i,:topk]
        fs=r if fusion_weight<=0 else normalize_scores(r,normalization)+fusion_weight*normalize_scores(d,normalization)
        order=np.argsort(-fs); ranked=list(cand[:topk][order])+list(cand[topk:])
        pos=np.where(np.asarray(ranked)==int(gt[i]))[0]
        if len(pos): final[i]=int(pos[0])
    return final


def eval_language(args,lang,model,tok,device):
    _,val,test,_=create_dataset(args.data_root,lang); ds=val if args.split=='valid' else test
    gt=text2code_to_array(ds.text2code,len(ds)); rows=read_jsonl(Path(args.data_root)/lang/f'{args.split}.jsonl'); codes_rows=read_jsonl(Path(args.data_root)/lang/'codebase.jsonl')
    queries=[get_query_text(x) for x in rows]; codes=[get_code_text(x) for x in codes_rows]
    cache=coderank_cache_paths(args.nomic_cache_dir,lang,args.split,args.nomic_model_path,args.nomic_max_length,'nomic_transformers',args.nomic_query_prefix,args.nomic_code_prefix)
    qe=load_cache_array(cache['query'],len(ds),'Nomic query'); ce=load_cache_array(cache['code'],len(codes),'Nomic code')
    top_scores,top_idx,base=retrieve_topk_from_cache(qe,ce,gt,device,args.candidate_top_k,args.query_chunk_size,args.code_chunk_size)
    k=min(args.rerank_top_k,args.candidate_top_k)
    score_dir=Path(args.score_cache_dir); score_dir.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha1(json.dumps({'model':args.reranker_model,'lang':lang,'split':args.split,'k':k,'instruction':args.instruction,'max_length':args.max_length},sort_keys=True).encode()).hexdigest()[:10]
    cache_path=score_dir/f'{lang}_{args.split}_qwen3reranker_top{k}_{digest}.npy'
    if args.reuse_score_cache and cache_path.exists(): rr=np.load(cache_path)
    else:
        pairs=[format_instruction(args.instruction,q,codes[int(ci)]) for q,cands in zip(queries,top_idx) for ci in cands[:k]]
        rr=score_pairs(model,tok,pairs,args.batch_size,args.max_length).reshape(len(queries),k); np.save(cache_path,rr)
    results={}
    for fw in args.fusion_weights:
        ranks=rerank(top_idx,top_scores,rr,gt,base,k,float(fw),args.normalization); res=compute_metrics_from_ranks(ranks); res.update({'fusion_weight':float(fw),'rerank_top_k':k,'model':args.reranker_model}); results[f'weight={fw:g}']=res
        if args.rank_output_dir:
            rd=Path(args.rank_output_dir); rd.mkdir(parents=True,exist_ok=True); np.save(rd/f'{lang}_{args.split}_{Path(args.reranker_model).name}_top{k}_weight{fw:g}.ranks.npy',ranks.astype(np.int32))
    return results


def main():
    ap=argparse.ArgumentParser(description='Evaluate Qwen3-Reranker/CoREB-Reranker on the exact Nomic candidate pool used by ReCalSearch.')
    ap.add_argument('--reranker-model',required=True)
    ap.add_argument('--languages',nargs='+',default=['ruby','javascript','php','python','java','go'])
    ap.add_argument('--split',default='valid',choices=['valid','test'])
    ap.add_argument('--data-root',required=True)
    ap.add_argument('--nomic-model-path',default='/root/autodl-tmp/hf_models/nomic-embed-code'); ap.add_argument('--nomic-cache-dir',default='data/rerank/coderank_cache_oldcsn'); ap.add_argument('--nomic-max-length',type=int,default=512); ap.add_argument('--nomic-query-prefix',default='Represent this query for searching relevant code: '); ap.add_argument('--nomic-code-prefix',default='')
    ap.add_argument('--candidate-top-k',type=int,default=100); ap.add_argument('--rerank-top-k',type=int,default=15); ap.add_argument('--batch-size',type=int,default=4); ap.add_argument('--max-length',type=int,default=8192); ap.add_argument('--query-chunk-size',type=int,default=64); ap.add_argument('--code-chunk-size',type=int,default=4096)
    ap.add_argument('--instruction',default=DEFAULT_T2C_INSTRUCTION); ap.add_argument('--dtype',choices=['float16','bfloat16','float32'],default='bfloat16'); ap.add_argument('--device-map',choices=['cuda','auto','none'],default='cuda'); ap.add_argument('--trust-remote-code',action='store_true')
    ap.add_argument('--fusion-weights',type=float,nargs='+',default=[0.0]); ap.add_argument('--normalization',choices=['none','zscore','minmax','rank','softmax'],default='zscore')
    ap.add_argument('--score-cache-dir',default='results/reviewer_revision/sota_scores'); ap.add_argument('--reuse-score-cache',action='store_true'); ap.add_argument('--rank-output-dir',default='results/reviewer_revision/sota_ranks'); ap.add_argument('--output-json',required=True)
    ap.add_argument('--frozen-config',default=''); ap.add_argument('--save-frozen-config',default=''); ap.add_argument('--selection-metric',choices=['mrr','r1','ndcg@10'],default='mrr'); ap.add_argument('--unsafe-allow-test-without-frozen-config',action='store_true')
    args=ap.parse_args()
    if args.frozen_config:
        fr=load_json(args.frozen_config); cfg=fr['config']; args.fusion_weights=[float(cfg['fusion_weight'])]; args.normalization=cfg.get('normalization',args.normalization); args.rerank_top_k=int(cfg.get('rerank_top_k',args.rerank_top_k))
    if args.split=='test' and not args.unsafe_allow_test_without_frozen_config: enforce_frozen_test_protocol(args.split,args.fusion_weights,args.frozen_config,'fusion_weight')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    model,tok=load_model(args.reranker_model,args.dtype,args.device_map,args.trust_remote_code); device=torch.device('cuda')
    allr={f'weight={w:g}':{} for w in args.fusion_weights}
    for lang in args.languages:
        lr=eval_language(args,lang,model,tok,device)
        for k,v in lr.items(): allr[k][lang]=v
    for k,d in allr.items():
        per=list(d.values()); d['macro_average']=average_metric_results(per); d['pooled_micro']=pooled_metric_results(per); d['average']=d['macro_average']
    if args.split=='valid' and args.save_frozen_config:
        best=max(((float(allr[f'weight={w:g}']['macro_average'][args.selection_metric]),float(w)) for w in args.fusion_weights),key=lambda x:x[0])[1]
        save_frozen_config(args.save_frozen_config,method='qwen3_family_reranker',selected_on_split='valid',selected_metric=args.selection_metric,config={'fusion_weight':best,'normalization':args.normalization,'rerank_top_k':args.rerank_top_k,'candidate_top_k':args.candidate_top_k,'model':args.reranker_model},validation_result=allr[f'weight={best:g}']['macro_average'])
    op=Path(args.output_json); op.parent.mkdir(parents=True,exist_ok=True); op.write_text(json.dumps(allr,indent=2,ensure_ascii=False)); write_environment_manifest(op.with_suffix(op.suffix+'.environment.json'),extra={'role':'sota_qwen3_family_reranker','model':args.reranker_model,'split':args.split,'instruction':args.instruction,'frozen_config':args.frozen_config})
    print(json.dumps({k:{'macro':v['macro_average'],'micro':v['pooled_micro']} for k,v in allr.items()},indent=2))

if __name__=='__main__': main()
