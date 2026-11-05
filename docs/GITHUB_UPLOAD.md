# Publish the reviewed ReCalSearch source repository

The ZIP is a GitHub-ready source tree, **not yet published**. It intentionally has no `.git` history.

1. Review source, history, credit, and licensing. Only choose/add a license if you have the rights to do so. In particular, inspect `baselines/coderankllm_integration/` and all third-party dependencies.
2. Unzip and `cd ReCalSearch` (the directory containing this README and SHA256SUMS).
3. Create an empty GitHub repository (e.g. `ReCalSearch`). Do not generate a README in the GitHub UI if using an empty initial branch.
4. Run the commands below, substituting **your own GitHub account**:

```bash
git init
git add .
git commit -m "Release ReCalSearch research source and frozen results"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/ReCalSearch.git
git push -u origin main
```

Never put your GitHub token into command text, scripts, or git remote URLs. The normal git credential manager/browser authorization is preferable.

`SHA256SUMS` can be checked from the repository root with `sha256sum -c SHA256SUMS` on Linux.

**Before publishing**: inspect `.gitignore`, selected `code_original_full/` files and JSON manifests for local paths and dataset disclosure; decide whether history from third-party projects is required. Some original scripts have AutoDL-specific absolute paths and need path configuration to rerun on another machine.
