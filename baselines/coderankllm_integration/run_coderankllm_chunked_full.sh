#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/cornstack-official"
PATCH="${ROOT}/recal_patch"
LLM_RERANKER="${ROOT}/llm-reranker"
MODEL="/root/autodl-tmp/hf_models/CodeRankLLM"

TEXT2CODE="/root/autodl-tmp/lora-code-embeddings-main/text2code"
DATA_ROOT="/root/autodl-tmp/lora-code-embeddings-main/data/CSN"
POOL_ROOT="${TEXT2CODE}/results/reviewer_revision/pools/nomic_test"

RUN_ROOT="${ROOT}/runs_coderankllm_full"

CHUNK_SIZE="${CHUNK_SIZE:-1000}"
LANGUAGES="${LANGUAGES:-ruby javascript php python java go}"
MAX_CHUNKS="${MAX_CHUNKS:-0}"

mkdir -p "${RUN_ROOT}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export CODERANKLLM_GPU_MEMORY_UTILIZATION=0.90
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------------------
# Freeze global reproducibility information once.
# ---------------------------------------------------------------------

META="${RUN_ROOT}/reproducibility"

if [[ ! -f "${META}.txt" ]]; then
{
    echo "date=$(date -Iseconds)"
    echo
    echo "===== GPU ====="
    nvidia-smi
    echo
    echo "===== PYTHON ====="
    python --version
    echo
    echo "===== PACKAGES ====="
    python - <<'PY'
import importlib.metadata as m
for pkg in ["torch", "vllm", "transformers"]:
    try:
        print(pkg, m.version(pkg))
    except Exception as e:
        print(pkg, "UNKNOWN", e)
PY
    echo
    echo "===== CORNSTACK COMMIT ====="
    git -C "${ROOT}" rev-parse HEAD || true
    echo
    echo "===== LLM-RERANKER COMMIT ====="
    git -C "${LLM_RERANKER}" rev-parse HEAD || true
    echo
    echo "===== ENVIRONMENT ====="
    echo "VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER}"
    echo "CODERANKLLM_GPU_MEMORY_UTILIZATION=${CODERANKLLM_GPU_MEMORY_UTILIZATION}"
    echo "PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF}"
    echo "top_k=100"
    echo "window_size=10"
    echo "step_size=5"
    echo "context_size=32768"
    echo "do_batched=1"
    echo "rerank_type=code"
    echo "code_prompt_type=docstring"
    echo "use_logits=0"
    echo "use_alpha=0"
} > "${META}.txt"

    sha256sum \
      "${LLM_RERANKER}/scripts/convert_results.py" \
      "${LLM_RERANKER}/scripts/utils/rank_listwise_os_llm.py" \
      > "${META}_patched_files.sha256"
fi


for LANG in ${LANGUAGES}; do

    POOL="${POOL_ROOT}/${LANG}_test_nomic_top100.npz"

    if [[ ! -f "${POOL}" ]]; then
        echo "ERROR: missing frozen pool: ${POOL}"
        exit 1
    fi

    LANG_ROOT="${RUN_ROOT}/${LANG}"
    mkdir -p "${LANG_ROOT}"

    sha256sum "${POOL}" > "${LANG_ROOT}/candidate_pool.sha256"

    N=$(python - "${POOL}" <<'PY'
import sys
import numpy as np
p = np.load(sys.argv[1], mmap_mode="r")
print(p["indices"].shape[0])
PY
)

    echo
    echo "============================================================"
    echo "LANGUAGE=${LANG}"
    echo "TOTAL_QUERIES=${N}"
    echo "CHUNK_SIZE=${CHUNK_SIZE}"
    echo "============================================================"

    CHUNK_COUNTER=0

    for ((OFFSET=0; OFFSET<N; OFFSET+=CHUNK_SIZE)); do

        if [[ "${MAX_CHUNKS}" -gt 0 && \
              "${CHUNK_COUNTER}" -ge "${MAX_CHUNKS}" ]]; then
            break
        fi

        END=$((OFFSET + CHUNK_SIZE))
        if [[ "${END}" -gt "${N}" ]]; then
            END="${N}"
        fi

        TAKE=$((END - OFFSET))

        TAG=$(printf "%08d_%08d" "${OFFSET}" "${END}")

        CROOT="${LANG_ROOT}/chunk_${TAG}"
        DD="${CROOT}/dataset"
        OD="${CROOT}/output"
        ED="${CROOT}/evaluation"

        LOG="${CROOT}/rerank.log"
        GPU_LOG="${CROOT}/gpu.csv"
        META_LOG="${CROOT}/runtime.txt"

        SUMMARY="${CROOT}/fullmetrics.json"
        RANKS="${CROOT}/fullranks.npy"
        DONE="${CROOT}/DONE"

        if [[ -f "${DONE}" && \
              -s "${SUMMARY}" && \
              -s "${RANKS}" ]]; then

            echo "[SKIP] ${LANG} ${OFFSET}:${END}"
            CHUNK_COUNTER=$((CHUNK_COUNTER + 1))
            continue
        fi

        echo
        echo "------------------------------------------------------------"
        echo "RUN ${LANG} source queries ${OFFSET}:${END}"
        echo "------------------------------------------------------------"

        rm -rf "${CROOT}"
        mkdir -p "${DD}" "${OD}" "${ED}"

        # -------------------------------------------------------------
        # 1. Export exactly this frozen query slice into BEIR format.
        # Local qids are 0..TAKE-1; source offset is preserved separately.
        # -------------------------------------------------------------

        python "${PATCH}/prepare_coderankllm_beir.py" \
          --language "${LANG}" \
          --split test \
          --data-root "${DATA_ROOT}" \
          --candidate-pool "${POOL}" \
          --dataset-dir "${DD}" \
          --output-dir "${OD}" \
          --top-k 100 \
          --query-offset "${OFFSET}" \
          --max-queries "${TAKE}" \
          --overwrite \
          > "${CROOT}/prepare.log" 2>&1

        # -------------------------------------------------------------
        # 2. GPU monitor.
        # -------------------------------------------------------------

        echo \
"timestamp,memory_used_MiB,memory_total_MiB,gpu_util_percent,power_W" \
          > "${GPU_LOG}"

        (
          while true; do
            TS=$(date '+%Y-%m-%dT%H:%M:%S%z')

            S=$(nvidia-smi \
              --query-gpu=memory.used,memory.total,utilization.gpu,power.draw \
              --format=csv,noheader,nounits | head -1)

            echo "${TS},${S}" >> "${GPU_LOG}"
            sleep 1
          done
        ) &

        MONITOR_PID=$!

        START_EPOCH=$(date +%s)
        START_ISO=$(date -Iseconds)

        {
          echo "language=${LANG}"
          echo "source_offset=${OFFSET}"
          echo "source_end=${END}"
          echo "queries=${TAKE}"
          echo "start_iso=${START_ISO}"
          echo "start_epoch=${START_EPOCH}"
        } > "${META_LOG}"

        # -------------------------------------------------------------
        # 3. Official CodeRankLLM listwise reranking.
        # -------------------------------------------------------------

        set +e
        set -o pipefail

        python "${PATCH}/rerank.py" \
          --dataset_dir "${DD}" \
          --output_dir "${OD}" \
          --eval_dir "${ED}" \
          --model "${MODEL}" \
          --llm_reranker_dir "${LLM_RERANKER}" \
          --languages "${LANG}" \
          --top_k 100 \
          --window_size 10 \
          --step_size 5 \
          --context_size 32768 \
          --do_batched 1 \
          2>&1 | tee "${LOG}"

        STATUS=${PIPESTATUS[0]}

        set -e

        END_EPOCH=$(date +%s)
        END_ISO=$(date -Iseconds)
        ELAPSED=$((END_EPOCH - START_EPOCH))

        kill "${MONITOR_PID}" 2>/dev/null || true
        wait "${MONITOR_PID}" 2>/dev/null || true

        {
          echo "end_iso=${END_ISO}"
          echo "end_epoch=${END_EPOCH}"
          echo "elapsed_seconds=${ELAPSED}"
          echo "exit_status=${STATUS}"
        } >> "${META_LOG}"

        if [[ "${STATUS}" -ne 0 ]]; then
            echo "ERROR: CodeRankLLM failed: ${LANG} ${OFFSET}:${END}"
            exit "${STATUS}"
        fi

        if grep -qiE \
          'out of memory|memory allocation failed|CUDA out of memory' \
          "${LOG}"; then

            echo "ERROR: OOM warning detected: ${LANG} ${OFFSET}:${END}"
            exit 1
        fi

        # -------------------------------------------------------------
        # 4. Fail-closed qid audit + ReCal full-codebase rank mapping.
        # -------------------------------------------------------------

        python - \
          "${POOL}" \
          "${OD}" \
          "${DD}" \
          "${LANG}" \
          "${OFFSET}" \
          "${TAKE}" \
          "${RANKS}" \
          "${SUMMARY}" \
          "${ELAPSED}" \
          "${GPU_LOG}" <<'PY'

import csv
import json
import sys
from pathlib import Path

import numpy as np

(
    pool_file,
    output_dir,
    dataset_dir,
    lang,
    offset,
    take,
    ranks_out,
    summary_out,
    elapsed,
    gpu_log,
) = sys.argv[1:]

offset = int(offset)
take = int(take)
elapsed = int(elapsed)

pool = np.load(pool_file, mmap_mode="r")

indices = np.asarray(
    pool["indices"][offset:offset + take]
)

base_ranks = np.asarray(
    pool["base_ranks"][offset:offset + take],
    dtype=np.int64
)

gt_indices = np.asarray(
    pool["gt_indices"][offset:offset + take],
    dtype=np.int64
)

qrels_path = Path(
    dataset_dir
) / "code_datasets" / f"csn_{lang}" / "qrels" / "test.tsv"

result_path = Path(
    output_dir
) / "code_datasets" / f"csn_{lang}" / \
    "rerank_100_llm_gen_num.json"

if not result_path.exists():
    raise RuntimeError(
        f"Missing reranker output: {result_path}"
    )

qrel_qids = set()

with qrels_path.open() as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader)
    for row in reader:
        qrel_qids.add(str(row[0]))

results = json.load(result_path.open())
result_qids = set(map(str, results.keys()))

expected = {str(i) for i in range(take)}

if qrel_qids != expected:
    raise RuntimeError(
        f"QREL qid mismatch: "
        f"expected={len(expected)}, actual={len(qrel_qids)}"
    )

if result_qids != expected:
    raise RuntimeError(
        f"RESULT qid mismatch: "
        f"missing={sorted(expected-result_qids)[:20]}, "
        f"extra={sorted(result_qids-expected)[:20]}"
    )

final_ranks = base_ranks.copy()

inside_top100 = 0
outside_top100 = 0

for local_i in range(take):

    qid = str(local_i)
    gt = int(gt_indices[local_i])

    docs = sorted(
        results[qid].items(),
        key=lambda x: x[1],
        reverse=True,
    )

    reranked_docids = [
        int(docid)
        for docid, _ in docs
    ]

    if len(reranked_docids) != 100:
        raise RuntimeError(
            f"qid={qid}: expected 100 reranked docs, "
            f"got {len(reranked_docids)}"
        )

    if len(set(reranked_docids)) != 100:
        raise RuntimeError(
            f"qid={qid}: duplicate document IDs"
        )

    if gt in reranked_docids:
        final_ranks[local_i] = reranked_docids.index(gt)
        inside_top100 += 1
    else:
        frozen_candidates = set(
            map(int, indices[local_i])
        )

        if gt in frozen_candidates:
            raise RuntimeError(
                f"qid={qid}: GT existed in frozen top100 "
                f"but disappeared after reranking"
            )

        final_ranks[local_i] = base_ranks[local_i]
        outside_top100 += 1


def metrics(ranks):
    ranks = np.asarray(ranks, dtype=np.int64)

    return {
        "r1": float(np.mean(ranks < 1)),
        "r5": float(np.mean(ranks < 5)),
        "r10": float(np.mean(ranks < 10)),
        "mrr": float(np.mean(1.0 / (ranks + 1))),
        "ndcg@10": float(
            np.mean(
                np.where(
                    ranks < 10,
                    1.0 / np.log2(ranks + 2),
                    0.0,
                )
            )
        ),
    }


# GPU summary
mem = []
util = []
power = []

with open(gpu_log) as f:
    for row in csv.DictReader(f):
        try:
            mem.append(float(row["memory_used_MiB"]))
            util.append(float(row["gpu_util_percent"]))
            power.append(float(row["power_W"]))
        except Exception:
            pass

base = metrics(base_ranks)
coderank = metrics(final_ranks)

summary = {
    "protocol": "recal_full_rank_v1",
    "language": lang,
    "source_offset": offset,
    "source_end": offset + take,
    "queries": take,

    "inside_top100": inside_top100,
    "outside_top100_preserve_dense_rank": outside_top100,

    "nomic_base": base,
    "coderankllm": coderank,

    "mrr_absolute_gain": (
        coderank["mrr"] - base["mrr"]
    ),
    "mrr_gain_points": (
        coderank["mrr"] - base["mrr"]
    ) * 100.0,

    "wall_seconds": elapsed,
    "seconds_per_query": (
        elapsed / take if take else None
    ),

    "peak_gpu_memory_MiB": (
        max(mem) if mem else None
    ),
    "mean_gpu_memory_MiB": (
        sum(mem) / len(mem) if mem else None
    ),
    "peak_gpu_util_percent": (
        max(util) if util else None
    ),
    "mean_gpu_util_percent": (
        sum(util) / len(util) if util else None
    ),
    "peak_power_W": (
        max(power) if power else None
    ),
    "mean_power_W": (
        sum(power) / len(power) if power else None
    ),
}

np.save(
    ranks_out,
    final_ranks.astype(np.int32)
)

with open(summary_out, "w") as f:
    json.dump(
        summary,
        f,
        indent=2
    )

print(json.dumps(summary, indent=2))
print("CHUNK VALIDATION PASSED")

PY

        touch "${DONE}"

        echo "[DONE] ${LANG} ${OFFSET}:${END}"

        CHUNK_COUNTER=$((CHUNK_COUNTER + 1))

    done
done

echo
echo "ALL REQUESTED CHUNKS FINISHED"
