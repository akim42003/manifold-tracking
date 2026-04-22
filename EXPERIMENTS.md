# Running Experiments

Practical guide to running every analysis in this repo.  Each section is
self-contained: you can jump directly to the experiment you want without
reading the others.

---

## Prerequisites

```bash
pip install numpy scikit-learn matplotlib pandas umap-learn nltk
```

LARQL binary (already built):
```
~/manifold-tracking/larql/target/release/larql
```

Set a convenience alias for the session:
```bash
alias larql=~/manifold-tracking/larql/target/release/larql
```

---

## Reproducibility note

Extraction results depend on the LARQL binary version.  Record it before
running any experiments:

```bash
larql --version          # currently: larql 0.1.0
git -C ~/manifold-tracking/larql rev-parse --short HEAD
```

Pin both the binary version and the git SHA alongside the `vindex_hash` in any
results you intend to compare across runs.

---

## Step 0 — Extract a vindex

One-time per model.  Produces the `.vindex` directory all experiments read from.

```bash
larql extract-index <model_path_or_hf_id> \
    -o <name>.vindex \
    --level browse \
    --f16
```

`--level browse` writes `gate_vectors.bin`, `embeddings.bin`, and
`down_meta.bin`.  This is sufficient for all experiments in this repo.

**Gemma 3 4B** (cached locally):
```bash
larql extract-index \
    ~/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767 \
    -o gemma3-4b.vindex \
    --level browse --f16
# ~10 min, ~4 GB
```

**Any HuggingFace model**:
```bash
larql extract-index google/gemma-3-4b-it -o gemma3-4b.vindex --level browse --f16
```

**Resuming an interrupted extraction**:
```bash
larql extract-index <model> -o <name>.vindex --level browse --f16 --resume
```

Verify completion — all four files must be present:
```bash
ls <name>.vindex/
# Required: index.json  embeddings.bin  gate_vectors.bin  down_meta.bin  tokenizer.json
```

---

## Step 1 — Generate relation clusters

Produces `relation_clusters.json` inside the vindex.  Required before any
analysis experiment.

```bash
# TinyLlama (1.1B, 22 layers)
python generate_clusters.py tinyllama.vindex --k 256

# Gemma 3 4B (34 layers)
python generate_clusters.py gemma3-4b.vindex --k 512

# Options
#   vindex       path to .vindex directory  (required, positional)
#   --k INT      number of clusters (default 256)
#   --seed INT   random seed (default 42)
```

**Note**: if `relation_clusters.json` is already present after extraction,
`generate_clusters.py` will overwrite it.  If you rerun with a different `k`,
the old file is silently replaced — there is no versioning.  To preserve
multiple runs, copy the file before rerunning:
```bash
cp tinyllama.vindex/relation_clusters.json tinyllama.vindex/relation_clusters_k256.json
python generate_clusters.py tinyllama.vindex --k 512
```

**Choosing k**: the heuristic of k=256 (≤2B), k=512 (4–8B), k=1024 (larger)
is a starting point only.  TinyLlama with k=256 produced near-zero silhouette
scores, meaning the clusters don't correspond to real structure at that scale.
Always validate k by running Experiment 2's cluster quality section (which
tests k=64, 128, 256) and looking for an elbow in inertia and a peak in
silhouette before trusting the centroid-based results from Experiment 1.

**Expected output** (TinyLlama, current behavior):
```
Reading vindex: tinyllama.vindex
  Embeddings: (32000, 2048)  embed_scale=1.00
  Whole-word vocab: 15898 tokens
  Knowledge-layer features: 45056
  Gate vectors (knowledge layers): (45056, 2048)
  Finding gate tokens via cosine NN over 15898 whole-word tokens ...
  Valid offset directions: 44798
  After degenerate filtering: 44798
  Fitting k-means (k=256) ...
  Done.  Inertia: 115580.2
Wrote tinyllama.vindex/relation_clusters.json  (11.7 MB, k=256)
```

The degenerate filter (near-zero norm after normalization) removed 0 vectors
for TinyLlama.  This step exists primarily for models where many features have
output ≈ input embeddings.

**Runtime**: ~3 min for TinyLlama, ~10–15 min for Gemma 3 4B.

---

## Experiment 1 — Raw direction SVD (primary replication)

**Script**: `analyze_raw_directions.py`

This is the primary replication test.  It runs SVD on the full ~44k raw offset
directions before any clustering, then measures cluster quality at multiple k.

```bash
python analyze_raw_directions.py --vindex tinyllama.vindex

# Gemma
python analyze_raw_directions.py --vindex gemma3-4b.vindex

# Save output
python analyze_raw_directions.py --vindex gemma3-4b.vindex \
    | tee outputs/gemma3-4b/raw_analysis.txt
```

### Section 1 output — SVD on raw directions

**Passing case** (what success looks like on a model with clean relation structure):
```
── 1. SVD on 44,798 raw offset directions ──
   Top-10 singular values: 4.21  3.89  3.71  3.50  1.12  1.08  ...
   50% variance:    8D  ✓ (target: 5-22D)
   90% variance:   42D
   Spectral gap: component 5 (ratio 3.8×)
```
A sharp elbow in singular values (large values followed by a sudden drop),
a spectral gap ratio > 2×, and 50%-variance in single digits = genuine
low-dimensional manifold.

**Failing case** (TinyLlama 1.1B, observed):
```
── 1. SVD on 44,798 raw offset directions ──
   Top-10 singular values: 2.70  2.67  2.54  1.96  1.85  1.63  1.61  ...
   50% variance:   68D  ✗ (target: 5-22D)
   90% variance:  188D
   Spectral gap: component 3 (ratio 1.3×)
```
Nearly equal singular values (smooth decay, no elbow), gap ratio barely above
1×, 50%-variance at 68D.  This means either the proxy is too noisy for this
model scale, or genuine low-dimensional structure is absent at 1.1B.

The spectral gap ratio is the sharpest single indicator.  A ratio of 1.3×
at component 3 means no meaningful discontinuity in the spectrum.

### Section 2 output — Cluster quality

```
── 2. Cluster quality ──
   k= 64:  inertia/pt=0.9201  silhouette=0.0812  davies-bouldin=3.21
   k=128:  inertia/pt=0.8944  silhouette=0.0631  davies-bouldin=3.87
   k=256:  inertia/pt=0.8703  silhouette=0.0512  davies-bouldin=4.11
```

- **Silhouette > 0.1**: clusters have real geometric structure
- **Silhouette ≤ 0.05 or negative**: clusters are noise — the centroid SVD
  in Experiment 2 will not be interpretable
- **Inertia plateau**: if inertia/pt barely drops from k=64 to k=256, the
  data has fewer than 64 natural clusters; increasing k just splits noise

### Section 3 output — PC extreme clusters

```
── 3. Top-3 PC extreme clusters (k=64) ──
   PC1
     + extreme: c3[France,Germany→Paris,Berlin]  c12[Japan,Korea→Tokyo,Seoul]
     - extreme: c41[past,old→was,had]  c55[many,other→are,were]
   PC2
     + extreme: c7[music,song→composed,wrote]  c22[film,movie→directed,starred]
     - extreme: c8[north,south→located,situated]
```

Each cluster shows `[most-common-input-tokens → most-common-output-tokens]`.
Reading the extremes tells you what semantic axis each PC captures.  If the
tokens at both extremes of a PC look like random noise (e.g.,
`c3[##ing,##ed→the,a]`), that PC is not encoding a semantic relation.

---

## Experiment 2 — Centroid SVD + UMAP (visualization)

**Script**: `replicate_relation_offsets.py`

Runs SVD on the k-means cluster centroids and produces three plots.  The
centroid SVD is a discretized summary of the direction distribution — useful
for visualization but can misrepresent dimensionality if k is large relative
to true cluster count.  Use Experiment 1 for the primary dimensionality verdict;
use this experiment for plots and publication figures.

```bash
python replicate_relation_offsets.py \
    --vindex tinyllama.vindex \
    --out outputs/tinyllama \
    --bootstrap 200

python replicate_relation_offsets.py \
    --vindex gemma3-4b.vindex \
    --out outputs/gemma3-4b \
    --bootstrap 200

# All options
#   --vindex PATH    path to .vindex directory (default: gpt2.vindex)
#   --out PATH       output directory (default: outputs/relation_offsets)
#   --bootstrap N    bootstrap resamples for CI bands (default: 200)
#   --seed INT       random seed (default: 42)
```

**Output files**:

| File | Contents |
|---|---|
| `spectrum.png` | Log-scale singular values + bootstrap CI ribbon (left); cumulative variance with threshold lines at 50/90/95/99% and spectral gap marker (right) |
| `umap_by_cluster.png` | 2D UMAP of centroids coloured by cluster label |
| `umap_by_cluster_id.png` | Same UMAP coloured by continuous cluster ID — use this to spot density structure |
| `results.json` | Full dimensionality table, spectral gap, top-10 singular values, pass/fail verdict |

**Reading `spectrum.png`**: a sharp elbow followed by a flat tail means
low-dimensional.  A smooth, gradual decay means variance is spread broadly.
The vertical line on the right panel marks the spectral gap.

**Reading the UMAP**: tight, well-separated blobs = real cluster structure.
A smooth continuous cloud = k-means found noise, and the centroid SVD is
not meaningful.  If the UMAP looks like a cloud, rely on Experiment 1 instead.

---

## Experiment 3 — Random baseline

**Script**: `baseline_random.py`

Generates random unit vectors in the same ambient space as a vindex and runs
the identical SVD + clustering pipeline.  Run this alongside every real-model
experiment to establish what "no structure" looks like numerically.

```bash
python baseline_random.py --vindex tinyllama.vindex
python baseline_random.py --vindex gemma3-4b.vindex

# Or specify size manually
python baseline_random.py --n 44798 --d 2048
```

**Interpreting the output**: for truly isotropic unit vectors in `d`-dimensional
space, the spectrum is nearly flat — all singular values similar, 50%-variance
requires roughly `d/2` components.  For d=2048 that is ~1024D, well above the
80-component truncation of `randomized_svd`, so the script will report
50%-variance as ">80 (truncated)".

The operational comparison is:

| Metric | Random baseline | Real data (structure present) | Real data (noise) |
|---|---|---|---|
| 50%-var dim | >80 (truncated) | 5–22D | >80 |
| Silhouette | ~0.00 | > 0.05 | ~0.00 |
| Spectral gap ratio | ~1.0 | > 2× | ~1.0 |

If real-data silhouette matches the baseline silhouette (~0.00), the structure
is not distinguishable from random regardless of what the SVD says.

---

## Experiment 3b — Per-cluster SVD (correct 5–22D replication)

**Script**: `per_cluster_svd.py`

The 5–22D finding is a claim about **within-relation dimensionality**: take all
offset vectors assigned to one relation cluster, run SVD on those, and the
50%-variance dim is 5–22D.  Pooled SVD (Experiment 1) measures the union of all
relation subspaces, which is naturally much higher-dimensional and is the wrong
object.

Unlike the old `analyze_raw_directions.py`, this script recomputes the full
offset direction pipeline from the raw vindex binary files without needing
`relation_clusters.json`.

```bash
# Recommended invocation for Gemma 3 4B
python per_cluster_svd.py --vindex gemma3-4b.vindex \
    --gate-cos-percentile 85 --k 256 --n-bootstrap 100

# Key options
#   --k INT                      number of clusters (default 64)
#   --min-cluster-size INT        skip clusters below this size (default 30)
#   --gate-cos-percentile FLOAT  keep top N% by gate cosine (e.g. 85 → top 15%)
#   --gate-cos-threshold F        fixed threshold fallback (default 0.15)
#   --same-concept-threshold F   drop pairs with cos(E[out],E[in]) > this (0.9)
#   --n-bootstrap INT             bootstrap resamples for d50 CI (default 200)
#   --clustering {euclidean,spherical}  algorithm (default euclidean)
```

**Pass/fail criterion**: ≥50% of clusters have d50 in 5–22D.

**Output**:

```
── Per-cluster 50%-variance dimensionality (60 clusters, 4 skipped) ──
  Stat          d50    d90  gap_ratio
  ──────────────────────────────────────
  median       54.0  212.0       1.48
  ...
  Clusters with d50 in 5–22D: 20/60 (33%)

  ✗ REPLICATION FAILS

── Passing clusters (20) — sorted by d50 ──
  c  n    d50  ci90   d90  top pairs
  ...
```

The passing clusters section is the interpretive core: if they contain
recognisable semantic relations (e.g. `France→Paris`, `sing→sang`,
`dog→dogs`), you have replicated the Hernandez finding.

**Choosing k**: lower k means more points per cluster, more reliable SVD.
Start at k=64.  If ~20 clusters pass at k=64, try k=32 and k=16.

**Gate threshold**: use `--gate-cos-percentile 85` for cross-model comparisons
(model-agnostic, always keeps top 15% of features).  Use `--gate-cos-threshold
0.15` for Gemma, `0.3` for TinyLlama-class models if you prefer a fixed value.

**Output files**:
- `outputs/<vindex>/per_cluster_svd.json` — Euclidean run results
- `outputs/<vindex>/per_cluster_svd_spherical.json` — spherical run results
- `outputs/<vindex>/offsets_cached.npz` — cached vectors + assignments (Euclidean)
- `outputs/<vindex>/offsets_cached_spherical.npz` — cached vectors (spherical)

---

## Experiment 3c — Spherical k-means comparison

**Scripts**: `per_cluster_svd.py --clustering spherical` + `compare_clustering_methods.py`

**Motivation**: offset directions are L2-normalised unit vectors living on the
unit sphere.  MiniBatchKMeans minimises squared Euclidean distance and lets
centroids drift off the sphere.  Spherical k-means renormalises centroids after
each update, respecting the geometry.  The question is whether this algorithmic
difference produces meaningfully different partitions on Gemma's offset data.

```bash
# 1. Run Euclidean (may already be done from 3b)
python per_cluster_svd.py --vindex gemma3-4b.vindex \
    --gate-cos-percentile 85 --k 256 --n-bootstrap 100 \
    --clustering euclidean

# 2. Run spherical
python per_cluster_svd.py --vindex gemma3-4b.vindex \
    --gate-cos-percentile 85 --k 256 --n-bootstrap 100 \
    --clustering spherical

# 3. Compare
python compare_clustering_methods.py --vindex gemma3-4b.vindex
```

**What the comparison produces**:

```
── Summary statistics ──────────────────────────────────────────────
  Metric                       Euclidean     Spherical
  Pass rate (% d50 in 5-22D)      XX.X%         XX.X%
  d50 median                        XX.X          XX.X
  ...

── Cluster size distributions ──────────────────────────────────────
  Bin             Euclidean     Spherical
  n<10                  ...           ...
  ...

── Jaccard overlap (euclidean→spherical) ───────────────────────────
  Jaccard ≥ 0.8: N/256 (XX%)
  Jaccard ≥ 0.5: N/256 (XX%)
  Median Jaccard: 0.XXX

── Content comparison (known probe pairs) ──────────────────────────
  yourself→you:  eu=c42(yourself→you)  sph=c71(yourself→you)  jaccard=0.XXX
  ...
```

**Interpreting results**:
- Median Jaccard > 0.7 + pass rates within a few percent → methods agree;
  MiniBatchKMeans is faster, keep it as default.
- Meaningful pass-rate improvement or much higher Jaccard on known tight
  clusters → adopt spherical as default.
- Spherical worse → investigate before concluding; verify implementation on
  synthetic data first (ARI should be > 0.9 on 3-cluster toy data).

---

## Experiment 4 — Cross-model comparison

**Script**: `compare_models.py`

After running Experiment 2 on multiple models, compare results:

```bash
# Explicit paths
python compare_models.py \
    --results outputs/tinyllama/results.json \
    --results outputs/gemma3-4b/results.json \
    --out outputs/comparison.csv

# Auto-discover all results.json under outputs/
python compare_models.py --auto outputs/
```

**Output** (stdout table + CSV):
```
Model                  50%D   90%D   95%D  Gap@   Ratio    S[1]   Pass
────────────────────────────────────────────────────────────────────────
TinyLlama-1.1B-Chat      68    188    212   PC3    1.30    2.700     ✗
Gemma-3-4B-IT            ???   ???    ???   ???    ???      ???      ???
────────────────────────────────────────────────────────────────────────
Target: 50%D in 5–22D, spectral gap ratio > 2×
```

The CSV at `--out` preserves the full schema and is easy to extend with
additional columns (e.g., baseline comparison, PC interpretation notes)
as more models are added.

---

## Running on a new model

Complete sequence:

```bash
MODEL="path/or/hf-id"
NAME="mymodel"

# 0. Record LARQL version before extracting
larql --version
git -C ~/manifold-tracking/larql rev-parse --short HEAD

# 1. Extract
larql extract-index "$MODEL" -o "${NAME}.vindex" --level browse --f16

# 2. Verify
ls "${NAME}.vindex/"
# Required: index.json  embeddings.bin  gate_vectors.bin  down_meta.bin  tokenizer.json

# 3. Inspect model size to choose starting k
jq '{layers: .num_layers, hidden: .hidden_size, intermediate: .intermediate_size}' \
    "${NAME}.vindex/index.json"
# k=256 for ≤2B, k=512 for 4–8B, k=1024 for larger

# 4. Random baseline (establishes what "no structure" looks like)
python baseline_random.py --vindex "${NAME}.vindex"

# 5. Per-cluster SVD — primary 5-22D replication test
mkdir -p "outputs/${NAME}"
python per_cluster_svd.py --vindex "${NAME}.vindex" \
    --gate-cos-percentile 85 --k 256 --n-bootstrap 100
# For smaller models use --gate-cos-threshold 0.3 instead of percentile

# 6. Visualization
python visualize_manifold.py --vindex "${NAME}.vindex" --k 256

# 7. Spherical k-means comparison (optional, adds ~30 min)
python per_cluster_svd.py --vindex "${NAME}.vindex" \
    --gate-cos-percentile 85 --k 256 --n-bootstrap 100 --clustering spherical
python compare_clustering_methods.py --vindex "${NAME}.vindex"
```

---

## What to do if Gemma also fails

Before results land, the contingency tree:

**If Gemma shows the same pattern as TinyLlama** (50%-var dim >40D, silhouette
near zero, spectral gap <1.5×):

1. **Check the extraction against LARQL source**: verify that the knowledge
   layer range is correct for Gemma's 34-layer architecture by reading
   `layer_bands` from `index.json`.  If `layer_bands` is populated, use those
   bounds instead of the hardcoded L14–L28 heuristic.

2. **Try another model family**: run Llama-3 8B, Qwen-2 7B, or Mistral 7B.
   The failure may be Gemma-family specific (e.g., due to Gemma's
   pre-FFN-layernorm architecture, which changes what gate vectors represent).

3. **Filter to high-confidence features**: modify `generate_clusters.py` to
   only include features where the gate cosine similarity to the nearest
   whole-word token exceeds a threshold (e.g., 0.3).  Low-confidence gate
   assignments are most likely to produce spurious offset directions.

4. **Widen the whole-word filter**: try removing the alphabetic-only restriction
   to include numeric tokens and proper nouns (capitalized tokens), which are
   common in factual relations.

5. **Abandon the FFN-proxy approach**: the proxy only works if FFN features
   cleanly encode source→target pairs.  If superposition dominates at all tested
   scales, the right approach is to extract offset directions from actual
   entity-token pairs using a Wikidata lookup — feed (subject, object) token
   pairs directly into `E[object] − E[subject]` without using the FFN
   activation structure.

**If Gemma passes** (50%-var dim ≤22D, silhouette > 0.05, spectral gap >2×):

This confirms the finding is scale-sensitive.  Document the threshold between
TinyLlama (fails) and Gemma 3 4B (passes), then try an intermediate model
(Llama-3 1B or Gemma-3 1B) to narrow the crossing point.  The crossing point
is the empirical evidence for a Platonic Representation Hypothesis scaling
effect on relation manifold dimensionality.

---

## Troubleshooting

**`relation_clusters.json not found`**
→ Run `generate_clusters.py` first (Step 1).

**`down_meta.bin` missing after extraction**
→ Extraction is still running or was interrupted.  Track progress with
`ls -lh <name>.vindex/` — files are written sequentially: `embeddings.bin`,
`gate_vectors.bin`, then `down_meta.bin`.  Resume with `--resume` if interrupted.

**`gate_vectors.bin` present but very small**
→ Extraction failed partway through.  Delete and re-extract.

**Gate NN search hangs or runs out of memory**
→ `per_cluster_svd.py` runs a batched matmul of shape `(n_kl × ww_vocab_size)`
— for Gemma this is ~20k × 16k = ~6 GB in f32.  The batch size is set to 2048
rows; reduce it in `compute_directions()` if memory-constrained.

**Silhouette score computation hangs**
→ Capped at 5000 sample points internally.  If still slow, the issue is the
cosine metric at k=256.  Reduce to k=64 or use `metric="euclidean"` for speed.

**All silhouette scores near zero or negative**
→ K-means is not finding real cluster structure.  This is expected for smaller
models (<2B params) where the relation manifold is less crystallized — TinyLlama
1.1B showed this.  The centroid SVD in Experiment 2 will not be reliable; rely
on Experiment 1's raw-direction SVD and spectral gap instead.  If the spectral
gap ratio is also below 1.5×, the model may be genuinely below the scale where
relation manifolds compress cleanly (Platonic Representation Hypothesis scaling
effect).  See "What to do if Gemma also fails" above.

**`umap-learn` not installed**
→ `pip install umap-learn`.  Only `visualize_manifold.py` needs it;
`per_cluster_svd.py` and `baseline_random.py` do not.

**Spherical k-means is slow at K=256**
→ `NLTKSphericalKMeans` runs `n_init=10` restarts sequentially; each restart
is O(n × K × iter).  For Gemma with ~20k directions and K=256 this takes
roughly 20–40 min.  Lower `n_init` to 3 for quick iteration:
```python
km = SphericalKMeans(n_clusters=256, n_init=3, random_state=42)
```

**Spherical k-means ARI < 0.8 on synthetic data**
→ Run the sanity check in ARCHITECTURE.md §4 to verify the NLTK backend is
working.  ARI should be > 0.9 on 3-cluster toy data in 100D.  If it's not,
NLTK may have changed its `cosine_distance` convention — check
`nltk.cluster.util.cosine_distance` returns 0 for identical vectors.
