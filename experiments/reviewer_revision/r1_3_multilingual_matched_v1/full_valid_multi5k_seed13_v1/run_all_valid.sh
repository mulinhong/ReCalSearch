#!/usr/bin/env bash

set +e
set +o pipefail

cd /root/autodl-tmp/lora-code-embeddings-main/text2code || exit 2

BASE="$PWD/results/reviewer_revision/r1_3_multilingual_matched_v1"
OUT="$BASE/full_valid_multi5k_seed13_v1"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

mkdir -p "$OUT/languages"

echo "START=$(date -Is)"
echo "SPLIT=valid"
echo "LANGUAGES=ruby javascript php python java go"
echo "VALIDATION_ONLY=TRUE"
echo "TEST_EVALUATION=FALSE"

for LANG in ruby javascript php python java go
do
    echo
    echo "LANGUAGE_START=$LANG $(date -Is)"

    LOUT="$OUT/languages/$LANG"

    if [ -e "$LOUT" ]; then
        echo "FAIL_STOP: Language output already exists: $LANG"
        echo 3 > "$OUT/exit_code.txt"
        exit 3
    fi

    mkdir -p "$LOUT"

    python -u "$OUT/eval_valid_only.py" \
        --languages "$LANG" \
        --split valid \
        --data-root /root/autodl-tmp/lora-code-embeddings-main/data/CSN \
        --verifier-model "$BASE/models/multi5k_seed13_v5" \
        --verifier-base-model /root/autodl-tmp/hf_models/Qwen2.5-Coder-7B-Instruct \
        --verifier-dtype float16 \
        --verifier-device-map cuda \
        --verifier-mode yesno \
        --verifier-batch-size 1 \
        --max-length 1024 \
        --max-query-chars 512 \
        --max-code-chars 3000 \
        --candidate-top-k 100 \
        --candidate-pool-pattern \
          "$BASE/validation_pool_adapter_v2/candidate_pools/{language}_{split}_nomic_top100.npz" \
        --candidate-pool-tag r13_validation_v2 \
        --frozen-config \
          results/reviewer_revision/recal_valid_fresh_20260911/recal_main_frozen_config.json \
        --score-cache-dir "$LOUT/score_cache" \
        --rank-output-dir "$LOUT/ranks" \
        --json-output "$LOUT/results.json" \
        --experiment-name "r13_multi5k_seed13_valid_${LANG}" \
        > "$LOUT/eval.log" 2>&1

    RC=$?

    if [ "$RC" -eq 0 ]; then

        python - "$LOUT" "$LANG" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
lang = sys.argv[2]

expected = {
    "ruby": 2209,
    "javascript": 8253,
    "php": 26015,
    "python": 23107,
    "java": 15328,
    "go": 14242,
}[lang]

d = json.loads((out / "results.json").read_text())

assert "alpha=1" in d
assert lang in d["alpha=1"]
assert d["alpha=1"][lang]["n_queries"] == expected

scores = list((out / "score_cache").glob("*.scores.npy"))
ranks = list((out / "ranks").glob("*.ranks.npy"))

assert len(scores) == 1
assert len(ranks) == 1

s = np.load(scores[0], mmap_mode="r", allow_pickle=False)
r = np.load(ranks[0], allow_pickle=False)

assert s.shape == (expected, 15)
assert r.shape == (expected,)
assert np.isfinite(s).all()
assert (r >= 0).all()

print("LANGUAGE_OUTPUT_INTEGRITY=PASS")
print("LANGUAGE=", lang)
print("N_QUERIES=", expected)
print("MRR=", d["alpha=1"][lang]["mrr"])
PY

        CHECK_RC=$?

        if [ "$CHECK_RC" -ne 0 ]; then
            RC=$CHECK_RC
        fi
    fi

    echo "$RC" > "$LOUT/exit_code.txt"

    if [ "$RC" -ne 0 ]; then
        echo "LANGUAGE_FAIL_STOP=$LANG"
        tail -n 20 "$LOUT/eval.log"
        echo "$RC" > "$OUT/exit_code.txt"
        exit "$RC"
    fi

    echo "LANGUAGE_PASS=$LANG $(date -Is)"
    sha256sum "$LOUT/results.json" \
        > "$LOUT/results.sha256"

done

python - "$OUT" <<'PY'
from pathlib import Path
import json
import sys

out = Path(sys.argv[1])

langs = [
    "ruby", "javascript", "php",
    "python", "java", "go",
]

rows = {}
total_n = 0
total_rr = 0.0

for lang in langs:
    p = out / "languages" / lang / "results.json"
    d = json.loads(p.read_text())

    entry = d["alpha=1"][lang]

    rows[lang] = entry

    total_n += int(entry["n_queries"])
    total_rr += float(entry["_sum_rr"])

assert total_n == 89154

macro = sum(float(rows[x]["mrr"]) for x in langs) / 6
micro = 100.0 * total_rr / total_n

report = {
    "experiment": "r1_3_multi5k_seed13_full_valid",
    "split": "valid",
    "languages": rows,
    "total_queries": total_n,
    "macro_mrr_percent": macro,
    "micro_mrr_percent": micro,
    "test_evaluated": False,
    "all_six_complete": True,
}

p = out / "full_valid_summary.json"
assert not p.exists()
p.write_text(json.dumps(report, indent=2) + "\n")

print("ALL_SIX_VALID=PASS")
print("TOTAL_QUERIES=", total_n)
print("MACRO_MRR_PERCENT=", round(macro, 6))
print("MICRO_MRR_PERCENT=", round(micro, 6))
print("FULL_VALID_SUMMARY=", p)
PY

RC=$?

echo "$RC" > "$OUT/exit_code.txt"

if [ "$RC" -eq 0 ]; then
    echo "R1_3_MULTI5K_SEED13_FULL_VALID=PASS"
else
    echo "R1_3_MULTI5K_SEED13_FULL_VALID=FAIL_STOP"
fi

echo "END=$(date -Is)"
exit "$RC"
