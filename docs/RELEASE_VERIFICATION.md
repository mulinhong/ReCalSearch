# Release verification — 2026-11-06

The checks below are retained from the reviewed source package. The JSS/public-scope documentation update did not rerun experiments or add missing per-query inputs. Current release coverage is described in [REPRODUCTION_SCOPE.md](REPRODUCTION_SCOPE.md).

- Source: author-uploaded `ReCalSearch_GitHub_Source_and_Experiments.zip`, SHA256 `9abb02963ae5cf1d8271d45e9840b017cefbabf48d34a440b9679e21ad9d3395`.
- Original source collector: **398 files**, `SHA256SUMS` **397/397 PASS**, no zip path traversal, no symbolic links.
- Source collector issue: one historical file, `code_original_full/finetune_cosqa.py`, has pre-existing IndentationError at line 76; `code_original_full/run_experiments.sh` has DOS-CRLF shell syntax issues. These have not been edited; both are excluded from reviewed release.
- Curated historical source selection: **217 preserved unchanged non-documentation files**, 175 other original-full files excluded (unrelated or broken).
- Curated source parse: **112/112 Python AST syntax PASS**; **13/13 Bash syntax PASS**.
- Automated screen for common exposed private keys, GitHub/OpenAI/AWS tokens, URL-embedded credentials and simple password assignments found no hits. This is not a full security audit.
- Original archive's only self-contained model-free numerical replay, `analysis/paired_bootstrap.py`, reproduced main Nomic vs ReCalSearch Macro CI **[2.6043, 2.9280]** and Micro CI **[2.7713, 2.9767]** with 3,000 replicates (seed 20261104).
- The source codes for Nomic/Jina 2×2 crossover and four-source negative construction were located in the original collector despite the collector's file-name heuristics reporting zero.
- These status checks do **not** establish that raw full-inference runs can be rerun without external datasets, checkpoints, upstream dependencies and historical path configuration.
