#!/usr/bin/env python3

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(
    "/root/autodl-tmp/lora-code-embeddings-main/text2code"
)

DATA_ROOT = Path(
    "/root/autodl-tmp/lora-code-embeddings-main/data/CSN"
)

MANIFEST = (
    ROOT
    / "results/reviewer_revision"
    / "hn_source_ablation"
    / "final_protocol"
    / "train_query_manifest.json"
)

EXPECTED_MANIFEST_SHA = (
    "fd948a05bda3d45d72bae336590b43a8"
    "1f3f1ec543faaa190a746dad522ec354"
)

OLD_GROUPS = (
    ROOT
    / "results/jss_rescue_oldcsn/groups"
)

BM25_ROOT = (
    ROOT
    / "results/reviewer_revision"
    / "hn_source_ablation/groups"
)

JINA_ROOT = (
    ROOT
    / "results/reviewer_revision"
    / "hn_source_ablation/final_groups"
)

OUT = (
    ROOT
    / "results/reviewer_revision"
    / "hn_source_ablation/final_protocol"
    / "four_source_group_alignment_audit.json"
)

LANGS = [
    "ruby",
    "javascript",
    "php",
    "python",
    "java",
    "go",
]

SOURCES = [
    "random",
    "bm25",
    "jina",
    "nomic",
]

N_NEG = 7


def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def load_jsonl(path):
    rows = []

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def get_code_text(row):
    if row.get("code"):
        return str(row["code"])

    for key in [
        "code_tokens",
        "function_tokens",
    ]:
        value = row.get(key)

        if value:
            if isinstance(value, list):
                return " ".join(
                    map(str, value)
                )

            return str(value)

    raise KeyError(
        f"Cannot resolve code text: {list(row.keys())}"
    )


def canonicalize(text):
    return re.sub(
        r"\s+",
        " ",
        str(text).strip(),
    )


def canonical_hash(text):
    return hashlib.sha1(
        canonicalize(text).encode(
            "utf-8"
        )
    ).hexdigest()


def group_path(source, lang):

    if source == "random":
        return (
            OLD_GROUPS
            / (
                f"{lang}_train_"
                "random31_5000.jsonl"
            )
        )

    if source == "nomic":
        return (
            OLD_GROUPS
            / (
                f"{lang}_train_"
                "nomic_hn31_5000.jsonl"
            )
        )

    if source == "bm25":
        return (
            BM25_ROOT
            / (
                f"{lang}_train_"
                "bm25_top100_5000.jsonl"
            )
        )

    if source == "jina":
        return (
            JINA_ROOT
            / (
                f"{lang}_train_"
                "jina_top100_5000.jsonl"
            )
        )

    raise KeyError(source)


def collect_needed_code_hashes(
    codebase_path,
    groups_by_source,
):
    needed = set()

    for source in SOURCES:

        for row in groups_by_source[source]:

            pos = row.get(
                "positive", {}
            ).get(
                "code_index"
            )

            if pos is None:
                raise RuntimeError(
                    f"{source}: missing positive code_index"
                )

            needed.add(
                int(pos)
            )

            negatives = row.get(
                "negative_code_indices",
                [],
            )

            if len(negatives) < N_NEG:
                raise RuntimeError(
                    f"{source}: fewer than "
                    f"{N_NEG} negatives"
                )

            for idx in negatives[:N_NEG]:
                needed.add(
                    int(idx)
                )

    hashes = {}

    with Path(codebase_path).open(
        "r",
        encoding="utf-8",
    ) as f:

        for idx, line in enumerate(f):

            if idx not in needed:
                continue

            row = json.loads(line)

            hashes[idx] = canonical_hash(
                get_code_text(row)
            )

            if len(hashes) == len(needed):
                break

    missing = needed - set(hashes)

    if missing:
        raise RuntimeError(
            "Missing codebase indices: "
            f"{sorted(missing)[:20]}"
        )

    return hashes


def nomic_cache_paths(lang):

    tag = (
        "root_autodl-tmp_hf_models_"
        "nomic-embed-code_"
        "msl512_14805c3aa7"
    )

    root = (
        ROOT
        / "data/rerank"
        / "coderank_cache_oldcsn"
    )

    return {
        "query":
            root
            / f"{lang}_train_{tag}.query.npy",

        "code":
            root
            / f"{lang}_{tag}.code.npy",
    }


@torch.no_grad()
def deployed_nomic_hardness(
    lang,
    train_indices,
    neg_matrix,
):
    paths = nomic_cache_paths(
        lang
    )

    for p in paths.values():
        if not p.is_file():
            raise FileNotFoundError(
                p
            )

    q_np = np.load(
        paths["query"],
        mmap_mode="r",
    )

    c_np = np.load(
        paths["code"],
        mmap_mode="r",
    )

    train_indices = np.asarray(
        train_indices,
        dtype=np.int64,
    )

    neg_matrix = np.asarray(
        neg_matrix,
        dtype=np.int64,
    )

    if neg_matrix.shape != (
        5000,
        N_NEG,
    ):
        raise RuntimeError(
            f"{lang}: bad negative "
            f"matrix shape {neg_matrix.shape}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    output = np.empty(
        (
            len(train_indices),
            N_NEG,
        ),
        dtype=np.float32,
    )

    chunk_size = 256

    for start in range(
        0,
        len(train_indices),
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            len(train_indices),
        )

        q = torch.from_numpy(
            np.asarray(
                q_np[
                    train_indices[
                        start:end
                    ]
                ],
                dtype=np.float32,
            )
        ).to(device)

        q = F.normalize(
            q,
            dim=1,
        )

        ids = neg_matrix[
            start:end
        ]

        c = torch.from_numpy(
            np.asarray(
                c_np[
                    ids.reshape(-1)
                ],
                dtype=np.float32,
            )
        ).to(device)

        c = c.reshape(
            end - start,
            N_NEG,
            -1,
        )

        c = F.normalize(
            c,
            dim=2,
        )

        scores = torch.einsum(
            "bd,bkd->bk",
            q,
            c,
        )

        output[start:end] = (
            scores
            .float()
            .cpu()
            .numpy()
        )

        del q
        del c
        del scores

    return output


def stats(values):

    x = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    return {
        "n":
            int(x.size),

        "mean":
            float(np.mean(x)),

        "std":
            float(np.std(x)),

        "min":
            float(np.min(x)),

        "p10":
            float(
                np.percentile(
                    x,
                    10,
                )
            ),

        "p25":
            float(
                np.percentile(
                    x,
                    25,
                )
            ),

        "median":
            float(
                np.percentile(
                    x,
                    50,
                )
            ),

        "p75":
            float(
                np.percentile(
                    x,
                    75,
                )
            ),

        "p90":
            float(
                np.percentile(
                    x,
                    90,
                )
            ),

        "max":
            float(np.max(x)),
    }


manifest_sha = sha256(
    MANIFEST
)

print(
    "MANIFEST_SHA256=",
    manifest_sha,
)

if manifest_sha != EXPECTED_MANIFEST_SHA:
    raise RuntimeError(
        "FROZEN MANIFEST SHA CHANGED"
    )


with MANIFEST.open(
    "r",
    encoding="utf-8",
) as f:
    manifest = json.load(f)


report = {
    "format":
        "r1_1_four_source_group_alignment_v1",

    "manifest":
        str(MANIFEST),

    "manifest_sha256":
        manifest_sha,

    "sources":
        SOURCES,

    "first_n_negatives":
        N_NEG,

    "languages":
        {},

    "pooled_nomic_hardness":
        {},
}


pooled = {
    source: []
    for source in SOURCES
}

all_pass = True


for lang in LANGS:

    print()
    print(
        "=" * 78
    )

    print(
        f"AUDIT LANGUAGE: {lang}"
    )

    print(
        "=" * 78
    )

    indices = [
        int(x)
        for x in
        manifest[
            "languages"
        ][lang]["indices"]
    ]

    if len(indices) != 5000:
        raise RuntimeError(
            f"{lang}: manifest "
            f"count={len(indices)}"
        )

    expected_qids = [
        f"{lang}:{idx}"
        for idx in indices
    ]

    groups = {}

    for source in SOURCES:

        path = group_path(
            source,
            lang,
        )

        if not path.is_file():
            raise FileNotFoundError(
                path
            )

        rows = load_jsonl(
            path
        )

        if len(rows) != 5000:
            raise RuntimeError(
                f"{lang}/{source}: "
                f"rows={len(rows)}"
            )

        groups[
            source
        ] = rows

    # Nomic = identity reference only.
    ref = groups[
        "nomic"
    ]

    ref_qids = [
        str(x["query_id"])
        for x in ref
    ]

    if ref_qids != expected_qids:
        raise RuntimeError(
            f"{lang}: Nomic vs "
            "frozen manifest mismatch"
        )

    code_hashes = (
        collect_needed_code_hashes(
            DATA_ROOT
            / lang
            / "codebase.jsonl",
            groups,
        )
    )

    lang_report = {
        "sources": {},
    }

    lang_pass = True


    for source in SOURCES:

        rows = groups[
            source
        ]

        qid_ok = (
            [
                str(x["query_id"])
                for x in rows
            ]
            ==
            expected_qids
        )

        query_ok = all(
            str(a.get("query", ""))
            ==
            str(b.get("query", ""))
            for a, b in zip(
                rows,
                ref,
            )
        )

        positive_ok = all(
            int(
                a[
                    "positive"
                ][
                    "code_index"
                ]
            )
            ==
            int(
                b[
                    "positive"
                ][
                    "code_index"
                ]
            )
            for a, b in zip(
                rows,
                ref,
            )
        )

        count7_ok = True
        unique7_ok = True
        positive_index_excluded = True
        positive_canonical_excluded = True
        rank_order_ok = True

        neg_matrix = []

        for row in rows:

            pos = int(
                row[
                    "positive"
                ][
                    "code_index"
                ]
            )

            negatives = [
                int(x)
                for x in
                row.get(
                    "negative_code_indices",
                    [],
                )[:N_NEG]
            ]

            if len(negatives) != N_NEG:
                count7_ok = False
                continue

            if (
                len(set(negatives))
                != N_NEG
            ):
                unique7_ok = False

            if pos in negatives:
                positive_index_excluded = (
                    False
                )

            pos_hash = code_hashes[
                pos
            ]

            for neg in negatives:

                if (
                    code_hashes[
                        neg
                    ]
                    ==
                    pos_hash
                ):
                    positive_canonical_excluded = (
                        False
                    )

            neg_matrix.append(
                negatives
            )

            if source in [
                "nomic",
                "bm25",
                "jina",
            ]:

                ranks = row.get(
                    "negative_retrieval_ranks",
                    [],
                )

                if len(ranks) < N_NEG:
                    rank_order_ok = False

                else:

                    first7_ranks = [
                        int(x)
                        for x in
                        ranks[:N_NEG]
                    ]

                    if not all(
                        first7_ranks[i]
                        <
                        first7_ranks[i + 1]
                        for i in range(
                            N_NEG - 1
                        )
                    ):
                        rank_order_ok = (
                            False
                        )

        neg_matrix_ok = (
            len(neg_matrix)
            == 5000
        )

        source_pass = all([
            qid_ok,
            query_ok,
            positive_ok,
            count7_ok,
            unique7_ok,
            positive_index_excluded,
            positive_canonical_excluded,
            neg_matrix_ok,
            rank_order_ok,
        ])

        if not source_pass:
            lang_pass = False

        hardness = (
            deployed_nomic_hardness(
                lang,
                indices,
                neg_matrix,
            )
        )

        hardness_stats = stats(
            hardness
        )

        pooled[
            source
        ].append(
            hardness.reshape(-1)
        )

        path = group_path(
            source,
            lang,
        )

        lang_report[
            "sources"
        ][source] = {
            "path":
                str(path),

            "sha256":
                sha256(path),

            "rows":
                len(rows),

            "qid_order":
                qid_ok,

            "query_equal":
                query_ok,

            "positive_equal":
                positive_ok,

            "first7_count":
                count7_ok,

            "first7_unique":
                unique7_ok,

            "positive_index_excluded":
                positive_index_excluded,

            "positive_canonical_excluded":
                positive_canonical_excluded,

            "source_relative_rank_order":
                (
                    rank_order_ok
                    if source != "random"
                    else "not_applicable"
                ),

            "deployed_nomic_hardness":
                hardness_stats,

            "pass":
                source_pass,
        }

        print(
            f"{source:<8} "
            f"qid={qid_ok} "
            f"query={query_ok} "
            f"positive={positive_ok} "
            f"n7={count7_ok} "
            f"unique7={unique7_ok} "
            f"noGT={positive_index_excluded} "
            f"noCanonGT={positive_canonical_excluded} "
            f"rank={rank_order_ok} "
            f"NomicSimMean="
            f"{hardness_stats['mean']:.6f} "
            f"PASS={source_pass}"
        )

    lang_report[
        "pass"
    ] = bool(
        lang_pass
    )

    report[
        "languages"
    ][lang] = (
        lang_report
    )

    all_pass = (
        all_pass
        and lang_pass
    )

    print(
        f"{lang} "
        f"FOUR_SOURCE_LANGUAGE_PASS="
        f"{lang_pass}"
    )


for source in SOURCES:

    values = np.concatenate(
        pooled[
            source
        ]
    )

    report[
        "pooled_nomic_hardness"
    ][source] = stats(
        values
    )


report[
    "all_pass"
] = bool(
    all_pass
)


OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUT.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        indent=2,
        ensure_ascii=False,
    )

    f.write("\n")


print()
print(
    "=" * 78
)

print(
    "POOLED DEPLOYED-NOMIC HARDNESS"
)

print(
    "=" * 78
)


for source in SOURCES:

    s = report[
        "pooled_nomic_hardness"
    ][source]

    print(
        f"{source:<8} "
        f"n={s['n']} "
        f"mean={s['mean']:.6f} "
        f"std={s['std']:.6f} "
        f"p10={s['p10']:.6f} "
        f"p25={s['p25']:.6f} "
        f"median={s['median']:.6f} "
        f"p75={s['p75']:.6f} "
        f"p90={s['p90']:.6f}"
    )


print()
print(
    "AUDIT_FILE=",
    OUT,
)

print(
    "FOUR_SOURCE_GROUP_ALIGNMENT_PASS=",
    all_pass,
)

if not all_pass:
    raise SystemExit(2)
