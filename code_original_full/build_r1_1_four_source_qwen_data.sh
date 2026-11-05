#!/usr/bin/env bash

set +e
set +u
set +o pipefail

ROOT="/root/autodl-tmp/lora-code-embeddings-main/text2code"
DATA_ROOT="/root/autodl-tmp/lora-code-embeddings-main/data/CSN"

OUT_ROOT="$ROOT/results/reviewer_revision/hn_source_ablation/qwen_data"

LANGS=(
  ruby
  javascript
  php
  python
  java
  go
)

SOURCES=(
  random
  bm25
  jina
  nomic
)

NEGATIVES=7
POSITIVE_REPEAT=2
DATA_SEED=42

mkdir -p "$OUT_ROOT"

group_path() {
    local source="$1"
    local lang="$2"

    case "$source" in

        random)
            echo \
"$ROOT/results/jss_rescue_oldcsn/groups/${lang}_train_random31_5000.jsonl"
            ;;

        bm25)
            echo \
"$ROOT/results/reviewer_revision/hn_source_ablation/groups/${lang}_train_bm25_top100_5000.jsonl"
            ;;

        jina)
            echo \
"$ROOT/results/reviewer_revision/hn_source_ablation/final_groups/${lang}_train_jina_top100_5000.jsonl"
            ;;

        nomic)
            echo \
"$ROOT/results/jss_rescue_oldcsn/groups/${lang}_train_nomic_hn31_5000.jsonl"
            ;;

        *)
            echo "UNKNOWN SOURCE: $source" >&2
            return 2
            ;;
    esac
}


for source in "${SOURCES[@]}"
do
    echo
    echo "======================================================================"
    echo "BUILD QWEN DATA: $source"
    echo "======================================================================"

    COMBINED="$OUT_ROOT/combined_${source}_hn7_30k.jsonl"

    rm -f "$COMBINED"

    source_failed=0

    for lang in "${LANGS[@]}"
    do
        GROUP="$(group_path "$source" "$lang")"

        OUT="$OUT_ROOT/${lang}_${source}_hn7_5k.jsonl"

        echo
        echo "SOURCE=$source LANGUAGE=$lang"
        echo "GROUP=$GROUP"
        echo "OUTPUT=$OUT"

        if [ ! -f "$GROUP" ]; then
            echo "MISSING_GROUP=$GROUP"
            source_failed=1
            break
        fi

        python build_qwen_verifier_data.py \
          --input-file "$GROUP" \
          --output-file "$OUT" \
          --language "$lang" \
          --data-root "$DATA_ROOT" \
          --negatives-per-query "$NEGATIVES" \
          --positive-repeat "$POSITIVE_REPEAT" \
          --max-query-chars 512 \
          --max-code-chars 3000 \
          --shuffle \
          --seed "$DATA_SEED"

        RC=$?

        if [ "$RC" -ne 0 ]; then
            echo "${source}_${lang}_BUILD_FAILED=True"
            source_failed=1
            break
        fi

        N="$(wc -l < "$OUT")"

        echo "${source}_${lang}_ROWS=$N"

        if [ "$N" -ne 45000 ]; then
            echo "${source}_${lang}_BAD_ROW_COUNT=True"
            source_failed=1
            break
        fi

        cat "$OUT" >> "$COMBINED"
    done

    if [ "$source_failed" -ne 0 ]; then
        echo "SOURCE_${source}_FAILED=True"
        echo "STOPPING=True"
        exit 2
    fi

    TOTAL="$(wc -l < "$COMBINED")"

    echo
    echo "${source}_COMBINED_ROWS=$TOTAL"

    if [ "$TOTAL" -ne 270000 ]; then
        echo "${source}_COMBINED_BAD_COUNT=True"
        exit 2
    fi

    sha256sum "$COMBINED"

    echo "SOURCE_${source}_270K_BUILD_PASS=True"
done

echo
echo "======================================================================"
echo "ALL FOUR 270K DATASETS BUILT"
echo "======================================================================"

echo "R1_1_FOUR_SOURCE_270K_BUILD_PASS=True"
