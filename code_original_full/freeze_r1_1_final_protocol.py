import json
import hashlib
from pathlib import Path

LANGS = [
    "ruby",
    "javascript",
    "php",
    "python",
    "java",
    "go",
]

ROOT = Path(
    "/root/autodl-tmp/lora-code-embeddings-main/text2code"
)

OLD_GROUPS = ROOT / "results/jss_rescue_oldcsn/groups"

OUT = (
    ROOT
    / "results/reviewer_revision"
    / "hn_source_ablation"
    / "final_protocol"
)

OUT.mkdir(parents=True, exist_ok=True)


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


manifest = {
    "format": "r1_1_final_train_query_manifest_v1",
    "split": "train",
    "languages": {},
}

alignment = {
    "format": "r1_1_existing_nomic_random_alignment_v1",
    "languages": {},
}

all_pass = True


for lang in LANGS:

    nomic_file = (
        OLD_GROUPS
        / f"{lang}_train_nomic_hn31_5000.jsonl"
    )

    random_file = (
        OLD_GROUPS
        / f"{lang}_train_random31_5000.jsonl"
    )

    if not nomic_file.is_file():
        raise FileNotFoundError(nomic_file)

    if not random_file.is_file():
        raise FileNotFoundError(random_file)

    nomic = load_jsonl(nomic_file)
    random_rows = load_jsonl(random_file)

    if len(nomic) != 5000:
        raise RuntimeError(
            f"{lang}: Nomic rows={len(nomic)}, expected 5000"
        )

    if len(random_rows) != 5000:
        raise RuntimeError(
            f"{lang}: Random rows={len(random_rows)}, expected 5000"
        )

    nqids = [str(x["query_id"]) for x in nomic]
    rqids = [str(x["query_id"]) for x in random_rows]

    qid_equal = nqids == rqids

    query_equal = all(
        str(a.get("query", ""))
        == str(b.get("query", ""))
        for a, b in zip(nomic, random_rows)
    )

    positive_equal = all(
        a.get("positive", {}).get("code_index")
        ==
        b.get("positive", {}).get("code_index")
        for a, b in zip(nomic, random_rows)
    )

    unique_qids = len(set(nqids)) == 5000

    prefix = f"{lang}:"
    indices = []

    for qid in nqids:
        if not qid.startswith(prefix):
            raise RuntimeError(
                f"{lang}: unexpected query_id={qid}"
            )

        indices.append(
            int(qid[len(prefix):])
        )

    unique_indices = len(set(indices)) == 5000

    nomic_ge7 = all(
        len(row.get("negative_code_indices", [])) >= 7
        for row in nomic
    )

    random_ge7 = all(
        len(row.get("negative_code_indices", [])) >= 7
        for row in random_rows
    )

    lang_pass = all([
        qid_equal,
        query_equal,
        positive_equal,
        unique_qids,
        unique_indices,
        nomic_ge7,
        random_ge7,
    ])

    all_pass = all_pass and lang_pass

    manifest["languages"][lang] = {
        "indices": indices,
        "n_queries": 5000,
    }

    alignment["languages"][lang] = {
        "qid_equal": qid_equal,
        "query_equal": query_equal,
        "positive_equal": positive_equal,
        "unique_qids": unique_qids,
        "unique_indices": unique_indices,
        "nomic_ge7": nomic_ge7,
        "random_ge7": random_ge7,
        "nomic_file": str(nomic_file),
        "random_file": str(random_file),
        "nomic_sha256": sha256(nomic_file),
        "random_sha256": sha256(random_file),
        "pass": lang_pass,
    }

    print(
        f"{lang:<11} "
        f"qid={qid_equal} "
        f"query={query_equal} "
        f"positive={positive_equal} "
        f"unique={unique_indices} "
        f"nomic>=7={nomic_ge7} "
        f"random>=7={random_ge7} "
        f"PASS={lang_pass}"
    )


manifest_file = OUT / "train_query_manifest.json"

with manifest_file.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        manifest,
        f,
        indent=2,
        ensure_ascii=False,
    )
    f.write("\n")


alignment["all_pass"] = all_pass
alignment["query_manifest"] = str(manifest_file)
alignment["query_manifest_sha256"] = sha256(
    manifest_file
)

alignment_file = (
    OUT / "existing_nomic_random_alignment.json"
)

with alignment_file.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        alignment,
        f,
        indent=2,
        ensure_ascii=False,
    )
    f.write("\n")


protocol = {
    "format": "reviewer_r1_1_final_protocol_v1",

    "reviewer_item": "R1-1",

    "research_question": (
        "Distinguish generic hard-negative benefit, "
        "lexical/dense hard-negative benefit, and "
        "evidence for deployed-retriever alignment."
    ),

    "negative_sources": [
        "random",
        "bm25",
        "jina",
        "nomic",
    ],

    "deployed_retriever":
        "nomic-embed-code",

    "alternative_dense_retriever":
        "jina-code-embeddings-0.5b",

    "no_further_model_shopping": True,

    "languages": LANGS,

    "queries_per_language": 5000,
    "total_queries": 30000,

    "negatives_per_query": 7,
    "positive_repeat": 2,

    "examples_per_query": 9,
    "examples_per_source": 270000,

    "query_manifest": str(manifest_file),

    "query_manifest_sha256": sha256(
        manifest_file
    ),

    "rank_control": (
        "source-relative highest-ranked "
        "7 valid negatives"
    ),

    "similarity_policy": (
        "Raw similarity values are not forced "
        "to match across heterogeneous retrievers. "
        "Deployed-Nomic hardness distributions "
        "will be reported separately."
    ),

    "training": {
        "base_model":
            "Qwen2.5-Coder-7B-Instruct",

        "max_length": 1024,

        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,

        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],

        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "lr": 1e-4,
        "epochs": 1,
        "fp16": True,
        "optimizer": "adamw_torch",

        "seeds": [13, 42, 87],
    },

    "source_ablation": {
        "deployment_retriever": "nomic",
        "candidate_pool": 100,
        "verify_k": 15,
        "normalization": "zscore",
        "fusion_weight": 1.0,
    },

    "crossover": {
        "candidate_retrievers": [
            "nomic",
            "jina",
        ],

        "verifiers": [
            "nomic_hn",
            "jina_hn",
        ],

        "candidate_pool": 100,
        "rerank_depth": 15,

        "ranking_score":
            "verifier_logit_difference_only",

        "dense_fusion": False,

        "shared_hit_subset": (
            "GT contained in both "
            "Nomic top100 and Jina top100"
        ),

        "bootstrap_samples": 10000,

        "interaction_analysis": True,
    },

    "claim_gate": {
        "retain_retriever_aligned_claim_only_if": (
            "Nomic-HN beats Jina-HN on Nomic candidates; "
            "Jina-HN beats Nomic-HN on Jina candidates; "
            "the same direction holds on the shared-hit "
            "subset; and the paired-bootstrap interaction "
            "95% CI excludes zero."
        ),

        "otherwise":
            "narrow the retriever-specific claim",
    },

    "test_used_for_protocol_selection": False,
}


protocol_file = (
    OUT / "r1_1_final_protocol.json"
)

with protocol_file.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        protocol,
        f,
        indent=2,
        ensure_ascii=False,
    )
    f.write("\n")


print()
print("=" * 78)

print(
    "MANIFEST_FILE=",
    manifest_file,
)

print(
    "MANIFEST_SHA256=",
    sha256(manifest_file),
)

print(
    "ALIGNMENT_FILE=",
    alignment_file,
)

print(
    "PROTOCOL_FILE=",
    protocol_file,
)

print(
    "PROTOCOL_SHA256=",
    sha256(protocol_file),
)

print(
    "EXISTING_ALIGNMENT_PASS=",
    all_pass,
)

if not all_pass:
    raise SystemExit(2)

print("R1_1_FINAL_PROTOCOL_FREEZE_PASS=True")
