# ReCalSearch environments and installation

This repository has **two supported environment scopes**: (a) CPU-only replay of
released Nomic-versus-ReCalSearch rankings; (b) training/inference code requiring
a GPU and external models/data. The CodeRankLLM upstream implementation has its
**own separate** vLLM/Tevatron environment.

## 1. Scope and provenance

`metadata/environment_current.json` records the **current Work1 environment**
observed when the research archive was built. It is NOT an exact lock for every
historical run. The historical scripts may have been run on other images and at
other times. Source code is preserved unmodified; the environment files supplied
here are new packaging documentation and are not frozen scientific evidence.

| Package | Observed Work1 version |
|---|---|
| Python | 3.10.20 |
| PyTorch | `2.12.0.dev20260408+cu128` |
| Transformers | 5.10.2 |
| PEFT | 0.19.1 |
| Accelerate | 1.13.0 |
| NumPy | 2.2.6 |
| Tokenizers | 0.22.2 |
| Safetensors | 0.7.0 |
| SciPy | 1.15.3 |

No verified versions were available for several other imported packages, so
`requirements.txt` uses **proposed broad installation ranges** for these rather
than inventing historical pins. In particular the observed PyTorch dev+CUDA build
may no longer be available from a normal public package index. We have not
installed/tested all possible combinations in `requirements.txt`; this is a
starting point for building a new machine's environment, not a guarantee of
numerically identical inference.

## 2. CPU-only statistics: self-contained and recommended first

From the repository root on Linux/macOS:

```bash
python3 -m venv .venv-bootstrap
source .venv-bootstrap/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-bootstrap.txt
python tools/check_environment.py --profile bootstrap
python analysis/paired_bootstrap.py --replicates 3000 --seed 20261104
```

Windows PowerShell activation: `.venv-bootstrap\Scripts\Activate.ps1`.
This uses frozen `derived_ranks/` and needs no GPU, model weights or data download.
The output is the Nomic Direct versus ReCalSearch paired query Bootstrap **only**.

## 3. New GPU training/inference environment (proposed portable profile)

On a Linux host with an NVIDIA CUDA GPU, install compatible GPU drivers first.

```bash
conda env create -f environment.yml
conda activate recal-search
python -m pip install --upgrade pip
# IMPORTANT: install an appropriate torch wheel FIRST using the official selector:
# https://pytorch.org/get-started/locally/
# Use that site's actual command for your NVIDIA driver/CUDA situation.
python -m pip install -r requirements.txt
python tools/check_environment.py --profile training
```

The `requirements.txt` file intentionally does not pin CUDA or PyTorch. A
CUDA-enabled PyTorch build should be selected to match the new machine. The
repository's historical Work1 torch build was a nightly/development wheel, so
a plain `pip install torch==2.12.0.dev20260408+cu128` may fail.

If you have an archive/cache of the exact Work1 wheels, a closer reconstruction
can use `requirements-work1-snapshot.txt` **after** installing the corresponding
torch wheel; note that package availability and transitive requirements may
prevent installation on other machines. That file is an observation profile and
not an authoritative lock of the September/October research experiments.

The Qwen2.5-Coder-7B-Instruct model, Nomic Embed Code, Jina model, CodeSearchNet,
CoSQA, LoRA adapters and original cached candidates/ranks for all experiments
are **not** distributed in this ZIP. Users must obtain authorized copies and
adjust hard-coded historic AutoDL paths on a separate working copy. No command
in this environment guide runs a test evaluation or retrains the published
models automatically.

## 4. What each dependency is for

| Package | Scope |
|---|---|
| `torch` | Model training, embeddings, GPU inference (install separately) |
| `transformers`, `peft`, `accelerate`, `safetensors` | Qwen verifier / LoRA |
| `numpy`, `scipy`, `scikit-learn` | Rankings, sparse BM25 controls, metrics |
| `sentence-transformers` | Optional teacher/Jina-style encoder and rerank helpers |
| `tqdm` | Progress indicators |
| `tokenizers` | Pulled by Transformers (snapshot separately records its version) |

Some scripts in `code_original_full/` have dependencies on **legacy Work2
modules not included here**. Those historical exploratory entry points are not
part of the documented ReCalSearch runnable path; do not claim every source file
works standalone with these pip dependencies.

## 5. Separate upstream CodeRankLLM environment

The CodeRankLLM integration scripts invoke an external upstream repository's
listwise-reranking implementation. Its own environment requires **vLLM and
Tevatron**, and that environment must be built according to the specific
upstream commit's requirements and CUDA/PyTorch compatibility.

Keep it separate from the ReCalSearch LoRA environment. Do **not** blindly
`pip install vllm` into the same environment as the above training packages.
See `baselines/coderankllm_integration/README.md` and the upstream project.
This release does not vendor third-party upstream code or its lockfile.

## 6. Inspect the local environment without training

```bash
python tools/check_environment.py --profile bootstrap
python tools/check_environment.py --profile training
python tools/check_environment.py --profile training --compare-observed
python tools/export_environment_summary.py --output my_environment_summary.json
```

These commands use package metadata and Python stdlib only; the export file is a
summary of **the environment you ran them in**, never a historical-run claim.
`--compare-observed` compares versions against the archived snapshot and prints
warnings; it does not imply parity or rerun research tests.

## 7. Integrity and release scope

Check `SHA256SUMS` (all repository files except that manifest). The environment
add-on changes only documentation, install metadata, and helper scripts. Main
historical training/evaluation source bytes and frozen numeric evidence are not
modified. CPU bootstrap is tested against the distributed frozen rank arrays;
GPU training/inference portability and CodeRankLLM upstream execution are **not**
validated by this environment add-on.
