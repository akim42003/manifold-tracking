# manifold-tracking — Architecture & Experiments

This document explains how the repository uses LARQL, what every Python file
does, and the detailed mechanics of each experiment.

---

## 1. What LARQL is and what this repo does with it

LARQL ("Lazarus Query Language") decompiles transformer model weights into a
**vindex** — a directory of memory-mapped binary files that can be queried like
a graph database.  The core idea: the model *is* the database; its weights
encode facts as geometric structure in embedding space.

This repo uses LARQL as a **measurement instrument**.  It does not train
models or modify weights.  It asks: *what is the intrinsic dimensionality of
the relation manifold inside a transformer?*  LARQL exposes the low-level
weight geometry needed to answer that question without running inference.

The specific claim being investigated: **relation offset vectors
`normalize(E[target] − E[source])` span a 5–22 dimensional subspace** (50%
variance), even though the ambient embedding space is 768–8192D.  If true,
factual knowledge lives on a surprisingly thin sheet inside the model.

### The vindex on disk

Running `larql extract-index <model> -o model.vindex --level browse` produces:

| File | Contents |
|---|---|
| `index.json` | Metadata: num_layers, hidden_size, vocab_size, layer offsets |
| `embeddings.bin` | Token embedding matrix `[vocab, d_model]`, f16 or f32 |
| `gate_vectors.bin` | FFN gate row vectors `[total_features, d_model]`, stacked across all layers |
| `down_meta.bin` | Per-feature top-K output token IDs and logit scores (binary, see §3) |
| `tokenizer.json` | HuggingFace BPE tokenizer |
| `relation_clusters.json` | K-means centroids of relation offset directions (written by non-streaming extraction path, or by `generate_clusters.py`) |

Browse-level extraction takes ~5 minutes for a 4B model and produces ~4 GB on
disk.

### How LARQL's knowledge pipeline works

During extraction, for every FFN feature in the "knowledge layers" (L14–L28):

1. **Gate token**: find the whole-word vocabulary token whose embedding has the
   highest cosine similarity with the feature's gate row vector.  This is the
   concept that would most activate the feature in a forward pass.
2. **Output token**: the token whose logit the feature's down column most boosts
   (top entry in `down_meta`).
3. **Offset direction**: `normalize(E[output_token] − E[gate_token_avg])`.  If
   gate = "France" and output = "Paris", the direction encodes "capital-of".
4. **K-means**: cluster all offset directions.  Each centroid is a canonical
   relation direction.

The resulting `relation_clusters.json` contains the centroid matrix
`centres [k, d_model]` — that matrix is the primary input to this repo's
analysis pipeline.

### Why "knowledge layers" are L14–L28

Empirically, factual recall in Transformer language models concentrates in the
middle-to-late layers.  Early layers handle syntax and surface form; the final
layers handle output distribution shaping.  L14–L28 is the window where
subject–relation–object triples are most cleanly encoded as linear structure in
the residual stream.  For a model with fewer than 28 layers the range is capped
at `min(28, num_layers)`.

**Caveat**: this band is not a fixed empirical constant across architectures.
For Gemma 3 4B (34 layers), LARQL's own TRACE examples suggest the phase
transition for factual recall sits closer to L20–L24, so the effective knowledge
window may be L20–L30.  The right practice is to read `layer_bands` from
`index.json` if LARQL populated it during extraction; `generate_clusters.py` and
`analyze_raw_directions.py` fall back to the L14–L28 heuristic only when
`layer_bands` is absent.

---

## 2. Python files

### `generate_clusters.py`

**Purpose**: generate `relation_clusters.json` for a vindex that was extracted
with `--level browse` (the streaming path, which skips the knowledge pipeline).

The non-streaming extraction path writes `relation_clusters.json`
automatically; the streaming path does not.  This script replicates what LARQL
does internally:

1. Loads `embeddings.bin` and `gate_vectors.bin` from the vindex.
2. Builds a "whole-word" vocabulary subset: tokens that start with the
   SentencePiece word-boundary marker `▁` (U+2581) and contain only alphabetic
   characters.  This filters out subword fragments and punctuation, leaving
   ~16 000 clean lexical tokens for Gemma and TinyLlama.  **Tokenizer
   portability**: models using GPT-2-style BPE (e.g., GPT-2, CodeLlama) use `Ġ`
   (U+0120) as the word-boundary marker instead.  Some tokenizer families have no
   explicit marker at all.  The current implementation is correct for SentencePiece
   models; extending to other families requires detecting the tokenizer type from
   `tokenizer.json` and choosing the appropriate marker.
3. For each feature in knowledge layers (L14–`min(28, num_layers)`), finds the
   nearest whole-word token to its gate vector via batch cosine similarity
   (batch matmul of size 2048).
4. Computes `normalize(E[output_token] − E[input_token])` for each feature,
   filtering cases where the output token equals the input token (no direction
   to encode).
5. Runs MiniBatch K-means (k=256 default, 300 iterations, 5 inits) on the
   direction matrix.
6. Writes `relation_clusters.json` with fields `k`, `centres`, `labels`,
   `counts`, `top_tokens`.

**Key design decision**: restricting to whole-word tokens and knowledge layers
is what makes the offset directions semantically meaningful.  Using all layers
or the full vocabulary introduces noise from syntactic features and subword
fragments that dilute the relation signal.

### `replicate_relation_offsets.py`

**Purpose**: run the full replication experiment for the 5–22D finding and
produce publication-quality plots.

Pipeline:

1. `check_vindex` — validates the vindex directory exists and contains
   `relation_clusters.json`.
2. `check_deps` — validates `umap-learn` and `matplotlib` are installed.
3. `load_relation_offsets` (via `manifold_tools.directions`) — reads the
   cluster centroid matrix as a `DirectionBundle`.
4. `svd_with_bootstrap` — SVD of the centroid matrix with 200 bootstrap
   resamples (90% CI), bootstrap using percentile method.
5. Reports dimensionality at four variance thresholds (50 / 90 / 95 / 99%)
   and flags whether 50% variance falls in the 5–22D target range.
6. `umap_project` — 2D UMAP projection of the centroid matrix with cosine
   metric.
7. Saves `spectrum.png`, `umap_by_cluster.png`, `umap_by_cluster_id.png`,
   and `results.json` to `outputs/relation_offsets/`.

**Outputs and interpretation**:

- `spectrum.png`: two-panel plot.  Left: log-scale singular values with
  bootstrap ribbon.  Right: cumulative variance with threshold lines.  A sharp
  "elbow" and large spectral gap at component k means the data is k-dimensional.
- `umap_by_cluster.png`: each point is a cluster centroid, coloured by label.
  Tight separated blobs indicate real semantic clusters.  A smooth diffuse
  cloud indicates noise.
- `results.json`: machine-readable replication verdict and full dimensionality
  table for downstream analysis.

### `analyze_raw_directions.py`

**Purpose**: diagnostic script that runs the four-part analysis described
below on the ~44 000 *raw* offset direction vectors (pre-clustering), not just
the 256 cluster centroids.  This is the apples-to-apples comparison with
LARQL's published dimensionality finding, since the published result was
measured on raw directions, not centroids.

Four diagnostics:

1. **SVD on raw directions**: recomputes the offset direction pipeline from
   scratch (same logic as `generate_clusters.py`), then runs randomized SVD
   (`sklearn.utils.extmath.randomized_svd`, 80 components) on the centered
   44k × d_model matrix.  Reports dimensionality at 50/90/95% variance and
   the spectral gap.  This is the primary replication check.

2. **Cluster quality**: fits K-means at k=64, 128, 256 and reports:
   - *Inertia per point*: lower is more compact clustering.  A plateau between
     k values indicates the model has fewer natural clusters than k.
   - *Silhouette score* (sampled, cosine): ranges −1 to +1.  >0.1 indicates
     meaningful cluster structure; ≤0.05 is noise.
   - *Davies–Bouldin index*: lower is better; measures ratio of within-cluster
     scatter to between-cluster separation.

3. **Top-PC extreme centroids**: projects k=64 cluster centroids onto the top
   3 principal components and reports which centroids are most extreme along
   each axis.  Examining the top-token labels of extreme centroids reveals what
   semantic dimension each PC captures (e.g., "geographic relations" vs
   "temporal relations").

4. **Summary verdict**: passes/fails the 5–22D criterion on raw directions.

### `manifold_tools/`

The core analysis library.  All functions return structured dataclasses with
provenance attached so results remain reproducible and self-describing.

#### `_types.py`

Three dataclasses that flow through the whole pipeline:

- **`DirectionBundle`**: pairs `vectors: np.ndarray [n, d_model]` with
  `metadata: pd.DataFrame` (one row per vector), a `vindex_hash` for
  reproducibility, and a free-form `config` dict.  Has `.n`, `.d_model`,
  and `.filter(mask)` convenience properties.
- **`SVDResult`**: output of `svd_with_bootstrap`.  Contains `singular_values`,
  `cumulative_variance`, `bootstrap_lower`/`upper` CI bands, `spectral_gap_idx`
  and `spectral_gap_ratio`, plus `provenance` recording all hyperparameters.
- **`ProjectionResult`**: output of `umap_project` and `isomap_project`.
  Contains `coords [n, n_components]`, the original `metadata`, `metric`, and
  `provenance`.

#### `directions/__init__.py`

Three loaders, each returning a `DirectionBundle`:

- **`load_relation_offsets(vindex_path)`**: reads `relation_clusters.json`
  directly.  No LARQL Python bindings needed — the JSON serialises the full
  centroid matrix.  This is the primary data source for replication.
- **`load_gates(vindex_path, layers, sample_per_layer)`**: calls
  `larql.load()` and `vindex.gate_vectors(layer=n)`.  Returns gate row
  vectors for selected layers.  Needs LARQL Python bindings installed.
- **`load_embeddings(vindex_path, sample)`**: calls
  `vindex.embedding_matrix()`.  Returns the token embedding matrix.

Metric guidance in the docstrings: use cosine for offsets and residuals
(direction only), euclidean or dot for gates (magnitude encodes selectivity).

#### `manifolds/__init__.py`

- **`svd_with_bootstrap`**: centers the data matrix, runs full SVD
  (`np.linalg.svd`) for small datasets or randomized SVD
  (`sklearn.utils.extmath.randomized_svd`) for datasets >50 000 rows, then
  runs `n_bootstrap` percentile-bootstrap resamples to estimate uncertainty
  bands on the singular value spectrum.

  *Known bias*: percentile bootstrap of singular values is biased low because
  bootstrap resamples contain ~37% duplicate rows, which deflates the rank and
  hence the singular values.  The observed S typically lies above the upper CI
  band.  This is not a bug — the CI still shows uncertainty structure — but the
  test suite documents it explicitly rather than asserting coverage.

- **`umap_project`**: thin wrapper around `umap.UMAP` that attaches provenance
  and returns a `ProjectionResult`.
- **`isomap_project`**, **`procrustes_align`**, **`grassmannian_distance`**:
  additional geometric primitives for comparing manifolds across checkpoints
  (finetuning tracking use case).

#### `visualize/__init__.py`

- **`spectrum_plot`**: two-panel figure.  Left: log-scale singular values +
  bootstrap ribbon.  Right: cumulative variance with labelled threshold lines
  at 50/90/95/99% and a vertical line marking the spectral gap.
- **`projection_plot`**: UMAP or Isomap scatter.  `color_by` can be a metadata
  column name; handles both categorical (tab20 palette) and continuous
  (viridis) coloring automatically.
- **`trajectory_plot`**, **`procrustes_heatmap`**: planned for finetuning
  trajectory visualization (not yet exercised).

### `larql/scripts/`

Exploratory scripts from LARQL's own development.  Not part of the
manifold-tracking experiment pipeline, but useful as reference for how to
drive LARQL from Python.

| Script | What it does |
|---|---|
| `attention_residual.py` | Approximates residual stream contributions from OV-circuit projections (no forward pass) for a set of entity tokens, then projects against gate vectors to predict feature activations |
| `attention_trace.py` | Full forward pass via MLX, capturing FFN gate×up activations at specified layers; compares dynamic activations against static extraction |
| `circuit_summary.py` | Reads per-layer `L{n}_circuits.json` files and prints a table of circuit type distributions (projector / transform / suppressor / identity / inverter) |
| `compare_inference.py` | Runs LARQL inference and MLX F32 reference side-by-side on the same prompt; reports top-k token agreements to validate the LARQL forward pass |
| `debug_gemma.py` / `debug_gemma_l0.py` / `debug_gemma_l1.py` / `debug_gemma_attn.py` / `debug_gemma_norms.py` | Layer-by-layer numerical debugging of the Gemma forward pass against MLX; used to track down attention and norm implementation discrepancies |
| `edge_discover_fast.py` | Batch cosine-similarity edge discovery: for each gate feature in a layer, finds which embedding vectors are close in output space and writes `(layer, feature, token, similarity)` to JSONL |
| `edges_to_larql.py` | Converts the JSONL edge files from `edge_discover_fast.py` into a `.larql.json` graph format for querying |
| `probe_residuals.py` | Uses `larql residuals` to capture actual post-attention residual vectors during inference, then projects through gate matrices to identify which features fire; matches (entity, feature output) against Wikidata triples |
| `probe_with_attention.py` | Full prompted forward pass for 32 relation templates × N entities; captures gate activations at L14–27 and matches against Wikidata for precision/recall reporting |
| `validate_graph.py` | Cross-validates a `.larql.json` knowledge graph against live model inference: for each entity, runs a template prompt, then checks whether the graph's edges for that entity match the model's top prediction |

---

## 3. Experiments

### Experiment 0 — vindex extraction

**Script**: `larql extract-index` CLI  
**Input**: HuggingFace model weights (safetensors)  
**Output**: `.vindex` directory

The extraction pipeline:

1. Loads all safetensors shards, detects model architecture (Gemma, Llama,
   Mistral, etc.) from `config.json`, maps weight keys to canonical names.
2. Writes `embeddings.bin` and `gate_vectors.bin` (the embedding matrix and
   all gate rows stacked, in f16).
3. For each layer, multiplies the embedding matrix by the layer's down
   projection matrix (`W_down ∈ ℝ^{d_model × d_ffn}`) to get per-feature
   output logit scores, then keeps the top-K token IDs and logits
   per feature.  Writes to `down_meta.bin`.
4. For knowledge layers (L14–L28), runs the offset direction pipeline and
   K-means (see §2 above), writing `relation_clusters.json`.

`down_meta.bin` format (little-endian binary):
```
Header: magic=0x444D4554  version  num_layers  top_k   (4×u32 = 16 bytes)
Per layer:
  num_features  (u32)
  Per feature (entry_size = 8 + top_k×8 bytes, fixed):
    top_token_id  (u32)
    c_score       (f32)
    top_k × { token_id (u32), logit (f32) }
```

Because every feature entry is exactly `entry_size` bytes, any feature can be
accessed in O(1) by seeking to `16 + layer_offset + feature_idx × entry_size`
— no sequential scan needed.  This random-access property is the reason for the
fixed-width layout (rather than, say, variable-length JSONL).

### Experiment 1 — Relation offset dimensionality (primary replication)

**Scripts**: `generate_clusters.py` → `replicate_relation_offsets.py`  
**Scientific claim**: relation offset vectors are 5–22D at 50% variance

**Design**: The 5–22D number comes from LARQL's internal experiments on
Gemma 3 4B, not from an earlier paper.  The linear-relation finding it builds
on (Hernandez et al. 2023 "Linearity of Relation Decoding"; Chanin et al.
follow-up on low-rank structure) was measured on GPT-J and Llama2-7B and
reports rank ~200 for Llama2-7B — proportionally consistent but a different
measurement protocol.  Do not conflate the two.

This repo replicates the geometric structure using FFN feature activations as a
proxy for entity pairs: for each FFN feature in a knowledge layer, its gate
token approximates the "source entity" and its output token approximates the
"target entity".  The offset direction `E[target] − E[source]` then represents
the relation.

**Why this proxy is reasonable — and where it breaks**: the gate row vector of
an FFN feature points in the direction of the residual-stream representation
that maximally activates it.  The nearest embedding vector in cosine space is
the input concept most likely to trigger the feature.  Similarly,
`E^T W_down[:, feature]` gives the logits the feature would add to the output
distribution; the argmax is the most-boosted token.

The critical limitation is **feature superposition**: a single FFN feature
can represent a superposition of multiple unrelated concepts simultaneously
(this is why gate vectors are observed to be full-rank rather than
low-dimensional).  When superposition dominates, the gate-nearest-token and
down-argmax-token can reflect *different* component concepts, producing offset
directions that do not correspond to any coherent relation.  The negative
silhouette scores and ~68D dimensionality observed on TinyLlama 1.1B are
consistent with this noise dominating the signal.  Whether the proxy works on a
given model is itself a test result, not an assumption.

**Procedure**:

1. For each feature in knowledge layers, compute the offset direction (see §2).
2. Filter: drop features where output == input token (no relation) or where the
   direction has near-zero norm after normalization (output ≈ input in embedding
   space).
3. Cluster remaining directions with MiniBatch K-means (k=256, 300 iterations).
4. Write centroids as `relation_clusters.json`.
5. Load centroids as a `DirectionBundle` (256 × d_model matrix).
6. Run SVD with 200 bootstrap resamples.
7. Check: does 50% variance land in [5, 22]D?

**Replication status**: ongoing.  TinyLlama 1.1B currently shows ~68D at 50%
variance using cluster centroids.  The `analyze_raw_directions.py` script tests
whether the *raw* directions (before clustering) show the low-dimensional
structure, which is the correct apples-to-apples comparison.

**Expected sensitivity**: the finding is expected to be *strongest on larger,
better-trained models*.  Gemma 3 4B is the primary target — it is the model
LARQL's own finding was measured on, and larger models tend to encode factual
associations more cleanly per the Platonic Representation Hypothesis scaling
argument.  TinyLlama 1.1B at 22 layers is a weaker candidate: it has a
narrower knowledge band (only 8 layers fall in L14–L22), and at 1.1B parameters
factual representations are noisier and more entangled with syntactic ones.
GPT-2 (125M) would likely produce results similar to or worse than TinyLlama.

### Experiment 2 — Raw direction SVD (primary replication check)

**Script**: `analyze_raw_directions.py`  
**Purpose**: apples-to-apples replication of the 5–22D finding on raw offset
directions, not centroid summaries

**Note on naming**: `replicate_relation_offsets.py` runs SVD on the 256
cluster *centroids*, which is a discretized summary of the direction
distribution.  This is useful for visualization but can inflate or deflate
apparent dimensionality depending on k and cluster quality.  The *correct*
replication test is SVD on the ~44k raw directions computed before clustering —
that is what this script does.  If these two analyses give conflicting results,
trust this one.

Running SVD directly on the ~44 000 raw directions (rather than 256 centroids)
tests whether the cluster-based analysis is hiding or distorting low-rank
structure.  K-means centroids can artificially inflate apparent dimensionality
if (a) the true clusters are not convex or (b) k is much larger than the true
number of relation types.

The script uses `sklearn.utils.extmath.randomized_svd` (Halko et al., 2009)
with 80 components, which is efficient for the 44k × 2048 matrix without
computing the full decomposition.

Cluster quality metrics (silhouette, Davies–Bouldin, inertia vs k) help
distinguish two failure modes:

- **Good clusters, high dimensionality**: the proxy computation is working
  correctly but TinyLlama's relation manifold is genuinely broader than GPT-2's.
- **Bad clusters, high dimensionality**: the proxy is noisy; the SVD on
  raw directions may still show low-dimensional structure that the centroid
  averaging is washing out.

### Experiment 3 — Gemma 3 4B replication

**Vindex**: `gemma3-4b.vindex`  
**Purpose**: repeat Experiments 1–2 on a larger, better-supported model

Gemma 3 4B (46 layers, 3072D hidden) is the primary model in LARQL's own
experiments and performance benchmarks.  Knowledge layers L14–28 cover 14
layers × intermediate_size features ≈ 3× more features than TinyLlama.  The
larger hidden dimension may produce cleaner or noisier offset directions
depending on how much of the 3072D space is actually used for factual encoding.

After extraction completes, run:
```
python generate_clusters.py gemma3-4b.vindex --k 512
python analyze_raw_directions.py --vindex gemma3-4b.vindex
python replicate_relation_offsets.py --vindex gemma3-4b.vindex
```

k=512 is appropriate for a 4B model (more features → more distinct relation
types discoverable); the `replicate_relation_offsets.py` default of k=256 is
calibrated for GPT-2–class models.

---

## 4. Data flow summary

```
HuggingFace weights
        │
        ▼
larql extract-index                  (Rust; ~5 min for 4B)
        │
        ├─ embeddings.bin
        ├─ gate_vectors.bin
        ├─ down_meta.bin
        └─ relation_clusters.json  ← if non-streaming extraction
                │
                │  if missing:
                ▼
        generate_clusters.py         (Python; ~3 min for 1B)
                │
                └─ relation_clusters.json
                        │
        ┌───────────────┴──────────────┐
        ▼                              ▼
replicate_relation_offsets.py    analyze_raw_directions.py
  (centroid SVD + UMAP)            (raw SVD + cluster QA)
        │                              │
        └─── outputs/relation_offsets/ outputs to stdout
               spectrum.png
               umap_by_cluster.png
               results.json
```

---

## 5. Running everything

```bash
# 0. Extract (one-time, ~5–10 min)
/path/to/larql extract-index <model_path> -o model.vindex --level browse --f16

# 1. Generate clusters (if relation_clusters.json is missing)
python generate_clusters.py model.vindex --k 256

# 2. Replication check
python replicate_relation_offsets.py --vindex model.vindex --bootstrap 200

# 3. Deep diagnostics on raw directions
python analyze_raw_directions.py --vindex model.vindex

# 4. Run tests
cd tests && python -m pytest -v
```

Dependencies: `numpy`, `scikit-learn`, `umap-learn`, `matplotlib`, `pandas`.
LARQL Python bindings (`larql`) are only needed for `load_gates` and
`load_embeddings`; the relation offset analysis path is LARQL-free.
