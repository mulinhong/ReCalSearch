from pathlib import Path
import os,json,csv,hashlib,sys,traceback
import numpy as np

A2=Path(os.environ["R16_A2"])
A3=Path(os.environ["R16_A3"])
O=Path(os.environ["R16_OUT"])

LANGS=("ruby","javascript","php","python","java","go")
N=(2279,6483,28391,22176,26909,14291)

EXPECTED={
 "ruby":(66.27,68.75),
 "javascript":(55.68,59.14),
 "php":(40.45,43.47),
 "python":(67.52,71.33),
 "java":(56.43,59.28),
 "go":(71.72,72.70)
}

A3_SHA="7b568b558684bf1bf466ff0f374915eb6c769c96a1eb03ff176979b52bac1db3"
B=3000
SEED=20261104

def emit(k,v):
 print(f"{k}={v}",flush=True)

def need(x,msg):
 if not x:
  raise RuntimeError("FAIL_STOP_"+msg)

def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(4194304),b""):
   h.update(b)
 return h.hexdigest()

def ranks(p,n,expected_sha):
 need(p.is_file(),"RANK_FILE_"+p.name)
 need(sha(p)==expected_sha,"RANK_SHA_"+p.name)
 a=np.load(p,allow_pickle=False)
 need(a.ndim==1 and len(a)==n,"RANK_SHAPE_"+p.name)
 need(np.issubdtype(a.dtype,np.integer),
      "RANK_DTYPE_"+p.name)
 return a.astype(np.int64)

def reciprocal(rank):
 ans=np.zeros(len(rank),dtype=np.float64)
 mask=(rank>=0)&(rank<100)
 ans[mask]=1.0/(rank[mask]+1)
 return ans

def boot_means(delta,rng):
 out=np.empty(B,dtype=np.float64)
 n=len(delta)
 for st in range(0,B,12):
  k=min(12,B-st)
  ix=rng.integers(0,n,size=(k,n))
  out[st:st+k]=100*delta[ix].mean(axis=1)
 return out

def writecsv(name,rows):
 with (O/name).open("w",newline="",encoding="utf8") as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]))
  writer.writeheader()
  writer.writerows(rows)

try:
 emit("PHASE","1/5 VERIFY PREVIOUS EVIDENCE")

 arc=Path(str(A3)+".tar.gz")
 need(arc.is_file() and sha(arc)==A3_SHA,"A3_ARCHIVE_SHA")

 report=json.loads((A3/"report.json").read_text())
 need(
  report["status"]=="BLOCKED_CANONICAL_NOMIC_SOURCE",
  "EXPECTED_A3_STATUS"
 )
 need(report["rank_convention"]=="ZERO_BASED",
      "RANK_CONVENTION")

 index=json.loads(
  (A2/"candidate_rank_sources.json").read_text()
 )

 emit("FROZEN_INPUT","PASS")
 emit("RANK_CONVENTION","ZERO_BASED")

 emit("PHASE","2/5 LOAD SIX PAIRED RANK ARRAYS")

 pairs=[]
 pins=[]

 for lang,n in zip(LANGS,N):
  rec_items=[
   x for x in index
   if x["path"].endswith(
    f"/{lang}_test_llm_verifier_dense_cand100_verify15_alpha=1.ranks.npy"
   )
   and "/recal_valid_fresh_20260911/frozen_test/ranks/" in x["path"]
  ]

  need(len(rec_items)==1,"RECAL_SOURCE_"+lang)
  rec_item=rec_items[0]
  rec_path=Path(rec_item["path"])

  base_items=report["candidates"].get(lang,[])
  need(len(base_items)==1,"NOMIC_SOURCE_"+lang)

  base_item=base_items[0]
  base_path=Path(base_item["path"])
  need(
   base_path.name==f"{lang}_test_nomic_direct.ranks.npy",
   "NOMIC_IDENTITY_"+lang
  )

  rec=ranks(rec_path,n,rec_item["sha256"])
  base=ranks(base_path,n,base_item["sha256"])

  pairs.append((
   lang,n,reciprocal(base),reciprocal(rec)
  ))

  pins.append({
   "language":lang,
   "nomic_path":str(base_path),
   "nomic_sha256":base_item["sha256"],
   "recal_path":str(rec_path),
   "recal_sha256":rec_item["sha256"],
   "query_count":n
  })

  emit("PAIRED_RANKS",f"{lang} N={n} SHA_PASS")

 emit("PHASE","3/5 COMPUTE PAIRED BOOTSTRAP")

 rng=np.random.default_rng(SEED)
 rows=[]
 draws=[]
 deltas=[]
 base_means=[]
 rec_means=[]

 for lang,n,base,rec in pairs:
  delta=rec-base
  res=boot_means(delta,rng)

  low,high=np.percentile(res,[2.5,97.5])
  base_mrr=100*base.mean()
  rec_mrr=100*rec.mean()
  gain=100*delta.mean()

  exp_base,exp_rec=EXPECTED[lang]

  rows.append({
   "language":lang,
   "queries":n,
   "nomic_mrr_percent":base_mrr,
   "recal_mrr_percent":rec_mrr,
   "delta_mrr_pp":gain,
   "ci95_low_pp":float(low),
   "ci95_high_pp":float(high),
   "paper_nomic_mrr_percent":exp_base,
   "paper_recal_mrr_percent":exp_rec,
   "nomic_minus_paper_pp":base_mrr-exp_base,
   "recal_minus_paper_pp":rec_mrr-exp_rec
  })

  draws.append(res)
  deltas.append(delta)
  base_means.append(base_mrr)
  rec_means.append(rec_mrr)

  emit(
   "CI95",
   f"{lang} DELTA={gain:.4f} "
   f"LOW={low:.4f} HIGH={high:.4f}"
  )

 writecsv("per_language_paired_ci.csv",rows)

 emit("PHASE","4/5 COMPUTE MACRO AND MICRO")

 macro_draws=np.mean(np.stack(draws,axis=0),axis=0)
 macro_low,macro_high=np.percentile(
  macro_draws,[2.5,97.5]
 )

 macro_base=float(np.mean(base_means))
 macro_rec=float(np.mean(rec_means))
 macro_delta=macro_rec-macro_base

 pooled=np.concatenate(deltas)
 pooled_draws=boot_means(pooled,rng)
 micro_low,micro_high=np.percentile(
  pooled_draws,[2.5,97.5]
 )

 pooled_base=np.concatenate([x[2] for x in pairs])
 pooled_rec=np.concatenate([x[3] for x in pairs])

 micro_base=100*pooled_base.mean()
 micro_rec=100*pooled_rec.mean()
 micro_delta=100*pooled.mean()

 summary=[
  {
   "aggregation":"Macro_equal_language",
   "queries":sum(N),
   "nomic_mrr_percent":macro_base,
   "recal_mrr_percent":macro_rec,
   "delta_mrr_pp":macro_delta,
   "ci95_low_pp":float(macro_low),
   "ci95_high_pp":float(macro_high)
  },
  {
   "aggregation":"Micro_pooled_query",
   "queries":sum(N),
   "nomic_mrr_percent":micro_base,
   "recal_mrr_percent":micro_rec,
   "delta_mrr_pp":micro_delta,
   "ci95_low_pp":float(micro_low),
   "ci95_high_pp":float(micro_high)
  }
 ]

 writecsv("macro_micro_paired_ci.csv",summary)

 emit(
  "MACRO_CI95",
  f"NOMIC={macro_base:.4f} "
  f"RECAL={macro_rec:.4f} "
  f"DELTA={macro_delta:.4f} "
  f"LOW={macro_low:.4f} "
  f"HIGH={macro_high:.4f}"
 )

 emit(
  "MICRO_CI95",
  f"NOMIC={micro_base:.4f} "
  f"RECAL={micro_rec:.4f} "
  f"DELTA={micro_delta:.4f} "
  f"LOW={micro_low:.4f} "
  f"HIGH={micro_high:.4f}"
 )

 emit("PHASE","5/5 SAVE REPORT AND PAPER TABLE")

 lines=[
  r"\begin{table}[t]",
  r"\centering",
  r"\caption{Paired bootstrap 95\% confidence intervals "
  r"for MRR differences on frozen CodeSearchNet test rankings.}",
  r"\label{tab:paired_mrr_ci}",
  r"\resizebox{\columnwidth}{!}{%",
  r"\begin{tabular}{lrrr}",
  r"\toprule",
  r"Language & $\Delta$MRR (pp) & 95\% CI & Queries \\",
  r"\midrule"
 ]

 for x in rows:
  lines.append(
   f"{x['language'].capitalize()} & "
   f"{x['delta_mrr_pp']:.3f} & "
   f"[{x['ci95_low_pp']:.3f}, "
   f"{x['ci95_high_pp']:.3f}] & "
   f"{x['queries']} "+r"\\"
  )

 lines.append(r"\midrule")

 for x in summary:
  title=(
   "Macro" if x["aggregation"].startswith("Macro")
   else "Micro"
  )
  lines.append(
   f"{title} & {x['delta_mrr_pp']:.3f} & "
   f"[{x['ci95_low_pp']:.3f}, "
   f"{x['ci95_high_pp']:.3f}] & "
   f"{x['queries']} "+r"\\"
  )

 lines.extend([
  r"\bottomrule",
  r"\end{tabular}}",
  r"\end{table}"
 ])

 (O/"R16_paired_bootstrap_table.tex").write_text(
  "\n".join(lines)+"\n",encoding="utf8"
 )

 (O/"source_manifest.json").write_text(
  json.dumps(pins,indent=2)+"\n",encoding="utf8"
 )

 final={
  "status":"PASS_R16_A3F1_FROZEN_PAIRED_BOOTSTRAP",
  "bootstrap_replicates":B,
  "bootstrap_seed":SEED,
  "ranking_convention":"ZERO_BASED",
  "per_language":rows,
  "aggregate":summary,
  "source_manifest":pins,
  "table4_reconstruction_note":
    "Original paper rounded MRR values and the "
    "frozen rank arrays differ slightly; both "
    "values are preserved without alteration.",
  "php_training":"V1",
  "frozen_test_rank_postprocessing":True,
  "new_model_inference":False,
  "new_training":False,
  "work2_modified":False
 }

 (O/"report.json").write_text(
  json.dumps(final,indent=2)+"\n",encoding="utf8"
 )

 emit("STATUS","PASS_R16_A3F1_FROZEN_PAIRED_BOOTSTRAP")
 emit("LANGUAGES","6/6")
 emit("TOTAL_QUERIES",sum(N))
 emit("NEW_GPU_EXPERIMENT","NO")
 emit("WORK2_MODIFIED","NO")

except Exception as e:
 emit("STATUS","FAIL_STOP_R16_A3F1")
 emit("ERROR",repr(e))
 traceback.print_exc()
 sys.exit(2)
