# Public repository access and updates

The reviewed ReCalSearch package is now public at https://github.com/mulinhong/ReCalSearch. It is independent of the author's Work2 repository and accompanies the manuscript prepared for submission to the Expert Systems with Applications (ESWA).

```bash
git clone https://github.com/mulinhong/ReCalSearch.git
cd ReCalSearch
```

The exact included materials and supported reproduction scope are described in [REPRODUCTION_SCOPE.md](REPRODUCTION_SCOPE.md). Complete training and inference require separately obtained inputs, checkpoints, dependencies, and path configuration. The public repository includes the paired Nomic/ReCalSearch rank arrays and CPU bootstrap implementation; the per-query inputs for the crossover and CodeRankLLM bootstrap comparisons are not bundled.

`SHA256SUMS` records all tracked release files except itself. Documentation updates refresh this manifest while preserving implementation sources and frozen experimental data. `RELEASE_INFO.json` retains the hashes and provenance of the original preserved files.

Do not place credentials in scripts, commit messages, or remote URLs. Use GitHub's normal authentication flow for maintainer updates. Keep generated training outputs, local datasets, model weights, and private caches outside tracked release content.
