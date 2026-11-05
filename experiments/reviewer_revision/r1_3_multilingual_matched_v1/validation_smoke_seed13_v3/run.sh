#!/usr/bin/env bash
set +e
set +o pipefail

cd /root/autodl-tmp/lora-code-embeddings-main/text2code || exit 2

BASE="$PWD/results/reviewer_revision/r1_3_multilingual_matched_v1"
OUT="$BASE/validation_smoke_seed13_v3"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

echo "START=$(date -Is)"
echo "EXPERIMENT=R1-3 MULTI5K SEED13 VALIDATION SMOKE"
echo "SPLIT=valid"
echo "QUERY_COUNT=8"
echo "CANDIDATE_TOP_K=100"
echo "VERIFY_TOP_K=15"
echo "FUSION_ALPHA=1"
echo "NORMALIZATION=zscore"

python -u \
    "$OUT/eval_llm_verifier_external_pool.py" \
    --languages ruby \
    --split valid \
    --data-root \
      /root/autodl-tmp/lora-code-embeddings-main/data/CSN \
    --verifier-model \
      "$BASE/models/multi5k_seed13_v5" \
    --verifier-base-model \
      /root/autodl-tmp/hf_models/Qwen2.5-Coder-7B-Instruct \
    --verifier-dtype float16 \
    --verifier-device-map cuda \
    --verifier-mode yesno \
    --verifier-batch-size 1 \
    --max-length 1024 \
    --candidate-top-k 100 \
    --candidate-pool-pattern \
      "$BASE/validation_pool_adapter_v2/candidate_pools/{language}_{split}_nomic_top100.npz" \
    --candidate-pool-tag r13_validation_v2 \
    --frozen-config \
      results/reviewer_revision/recal_valid_fresh_20260911/recal_main_frozen_config.json \
    --max-queries 8 \
    --query-offset 0 \
    --score-cache-dir "$OUT/score_cache" \
    --rank-output-dir "$OUT/ranks" \
    --json-output "$OUT/results.json" \
    --experiment-name r13_multi5k_seed13_valid_smoke

RC=$?

if [ "$RC" -eq 0 ]; then

python - "$OUT" <<'PY'
from pathlib import Path
import json
import numpy as np
import sys

out = Path(sys.argv[1])

result = out / "results.json"
assert result.is_file(), "Missing evaluation JSON"

data = json.loads(result.read_text())
assert "alpha=1" in data, "Frozen alpha result missing"

ranks = list((out / "ranks").glob("*.ranks.npy"))
scores = list((out / "score_cache").glob("*.scores.npy"))

assert len(ranks) == 1, "Expected one rank file"
assert len(scores) == 1, "Expected one score cache"

r = np.load(ranks[0], allow_pickle=False)
s = np.load(scores[0], allow_pickle=False)

assert r.shape == (8,), f"Rank shape: {r.shape}"
assert s.shape == (8, 15), f"Score shape: {s.shape}"
assert np.isfinite(s).all(), "Nonfinite verifier scores"

print("VALIDATION_OUTPUT_INTEGRITY=PASS")
print("SCORES_SHAPE=(8,15)")
print("RANKS_SHAPE=(8,)")
PY

    CHECK_RC=$?

    if [ "$CHECK_RC" -ne 0 ]; then
        RC=$CHECK_RC
    fi

fi

echo "END=$(date -Is)"
echo "VALIDATION_RETURN_CODE=$RC"

if [ "$RC" -eq 0 ]; then
    echo "R1_3_VALIDATION_SMOKE_V3=PASS"
else
    echo "R1_3_VALIDATION_SMOKE_V3=FAIL_STOP"
fi

echo "$RC" > "$OUT/exit_code.txt"
exit "$RC"
