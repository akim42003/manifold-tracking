# Cluster-size confound paper — package contents

## Files

- **`paper.tex`** — main source. Compiles with plain `pdflatex` (uses standard packages: `amsmath`, `booktabs`, `graphicx`, `hyperref`, `xcolor`, `float`, `enumitem`, `microtype`, `natbib`).
- **`paper.pdf`** — compiled output, 20 pages including references and Appendix A.
- **`make_figures.py`** — reproduces all five figures from the underlying cluster JSONs and logs. Requires `matplotlib` and `numpy`.
- **`figures/`** — PDF and PNG versions of each figure, matched to the `\includegraphics` paths in `paper.tex`.
- **`data/`** — the underlying cluster analyses referenced by the paper:
  - `per_cluster_svd.json` — Gemma-3 4B, Euclidean k-means, K=256
  - `per_cluster_svd_spherical.json` — Gemma-3 4B, spherical k-means++ (the primary Gemma analysis)
  - `per_cluster_svd_spherical_nltk.json` — Gemma-3 4B, spherical random-init
  - `3896b_per_cluster_svd_spherical_nltk.log` — OLMo-2 7B final checkpoint
  - `olmo2_trajectory.json` / `olmo2_trajectory_spherical_nltk.json` — trajectory summaries

## Compiling

```
cd <extracted>
pdflatex paper.tex
pdflatex paper.tex   # second pass for cross-references
```

## Regenerating figures

```
python3 make_figures.py
```

The script reads the JSONs and log in `data/` (you'll need to symlink or adjust the `UPLOADS` path at the top of the script). It writes to `figures/`.

## Paper summary

**Central finding**: in weight-space offset clusters, $d_{50}$ is primarily a cluster-size statistic. Pearson $r(n, d_{50})$ is between 0.50 and 0.88 across three clustering methods, and coherent vs. noise clusters show indistinguishable size scaling. The widely-cited "5–22D pass rate" metric is therefore cluster-size confounded.

**Secondary findings that survive**:
- Content coherence (≥1 repeated source-target pair per cluster) is a useful diagnostic. Amber 7B has 65% pass rate but <1% coherence — a clear "fake pass".
- OLMo-2 shows a sharp content-coherence emergence between 214B and 911B training tokens, independent of pass-rate dynamics.
- Euclidean k-means on unit-normalized offsets produces trajectory artifacts not present under spherical k-means.

**Retraction**: a prior draft claimed OLMo-2 implements relations at "systematically higher dimensionality" than Gemma. This is not supported. 83% of OLMo-2 coherent clusters fall in the 5–22D range vs. Gemma's 55%; the apparent shift was a cluster-size selection artifact. Appendix A documents the error and its source.

## Figures

1. **fig1_d50_vs_size_gemma** — the central figure. Scatter of $d_{50}$ vs. cluster size on Gemma-3 4B, showing coherent and noise clusters trace nearly identical curves.
2. **fig2_size_bin_comparison** — median $d_{50}$ by cluster-size bin on Gemma and OLMo-2, split by coherence.
3. **fig3_olmo_trajectory** — seven-checkpoint OLMo-2 trajectory overlaying Euclidean pass rate, spherical pass rate, and content-coherence rate.
4. **fig4_crossmodel** — cross-model bar chart highlighting the Amber fake-pass.
5. **fig5_d50_distribution** — $d_{50}$ histograms for Gemma and OLMo-2 with coherent vs. noise overlays.
