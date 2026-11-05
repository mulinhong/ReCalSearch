#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/cornstack-official
cd "$ROOT"
rm -rf llm-reranker
# cornstack's .gitmodules names llm-reranker, but the checked-out commit has no gitlink,
# so git submodule update cannot populate it. Clone the official dependency explicitly.
git clone https://ghfast.top/https://github.com/gangiswag/llm-reranker.git llm-reranker
cd llm-reranker
git log -1 --oneline
ls -lah scripts/rerank_llm.py scripts/convert_results.py scripts/utils/rank_listwise_os_llm.py
