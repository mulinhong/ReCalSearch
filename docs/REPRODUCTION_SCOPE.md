# Public reproduction scope

The ReCalSearch manuscript is prepared for submission to the Journal of Systems and Software (JSS). The repository is publicly accessible at https://github.com/mulinhong/ReCalSearch.

## Shared availability statement

The public release provides implementation sources, experiment scripts, frozen statistical summaries, and six-language paired Nomic/ReCalSearch rank arrays. These arrays and the CPU bootstrap script support self-contained statistical replay of the Nomic--ReCalSearch confidence intervals. Full training and inference reruns require external datasets, model weights or adapters, upstream dependencies, experiment-specific inputs, and path configuration. Per-query inputs for the crossover and CodeRankLLM bootstrap intervals are not included in this release.

## Coverage of the distributed materials

| Experiment or material | Included | Reproduction supported by the repository alone |
|---|---|---|
| Nomic Direct versus ReCalSearch paired statistical comparison | Twelve rank arrays covering six languages, paired bootstrap implementation, and reference summaries | CPU statistical replay with Python and NumPy, using 3,000 replicates and seed 20261104 |
| Three-seed VALID study | Validation seed summaries and table inputs | Inspection of archived summaries; model training requires external inputs and checkpoints |
| Four-source negative-mining study | Construction and experiment scripts, frozen protocols and summaries | Inspection of archived protocols and results; complete training/inference requires external inputs and checkpoints |
| Nomic/Jina verifier-only crossover | Statistical entry point and frozen 10,000-resample summary | Summary inspection; the complete per-query inputs for independent recomputation are not bundled |
| ReCalSearch versus CodeRankLLM | Integration wrappers, statistical scripts, and frozen 10,000-resample summary | Summary inspection; per-query reranking outputs and upstream runtime are separate dependencies |
| Full retrieval, verifier training, and inference | Implementation and experiment scripts | Requires the original data, checkpoint/model versions, candidate pools or caches, dependencies, and path configuration |

The repository does not distribute raw CodeSearchNet/CoSQA datasets, model weights or adapters, all ranking caches and candidate pools, or the upstream CodeRankLLM runtime. Installation profiles, current observed versions, and the separate CodeRankLLM environment are documented in [ENVIRONMENT.md](ENVIRONMENT.md). The environment metadata describes a collected environment snapshot; it does not lock every historical experiment environment. The portable GPU profile is an installation starting point and does not add the external experiment inputs omitted from this release.

## Statistical provenance

The main-paper aggregate MRR, the Nomic/ReCalSearch frozen-rank reconstruction, the direct CodeRankLLM comparison, and the four-source controls use distinct statistical inputs. Their values must retain their original experiment labels. The 3,000-resample Nomic/ReCalSearch replay does not independently recompute the 10,000-resample crossover or CodeRankLLM intervals. Public release of the code does not itself constitute a new training, inference, or timing experiment.

The README, manuscript replication appendix, and manuscript Data availability statement use this same coverage. The manuscript and its submission/reviewer files are maintained separately from the public code repository.
