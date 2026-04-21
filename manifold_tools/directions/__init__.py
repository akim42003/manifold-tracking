"""Direction loaders — thin wrappers over LARQL Python bindings.

Each loader returns a DirectionBundle pairing a float32 numpy matrix with a
pandas DataFrame of per-vector metadata.  All reads are lazy where possible:
embeddings and gate vectors are already mmap'd inside the larql Vindex object.

Loaders that require only browse-level data (load_gates, load_embeddings,
load_relation_offsets) work on any vindex.  load_down_rows and
load_unembeddings need an all-level vindex and are not implemented for MVP.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from manifold_tools._types import DirectionBundle


# ── helpers ──────────────────────────────────────────────────────────────────

def _read_index_json(vindex_path: str) -> dict:
    path = os.path.join(vindex_path, "index.json")
    with open(path) as f:
        return json.load(f)


def _vindex_hash(index_data: dict) -> str:
    checksums = index_data.get("checksums", {})
    return checksums.get("gate_vectors.bin", "")


# ── public API ────────────────────────────────────────────────────────────────

def load_relation_offsets(vindex_path: str) -> DirectionBundle:
    """Load relation-offset cluster centers from a vindex.

    Reads relation_clusters.json directly — no forward pass needed.
    Returns one vector per cluster (k ≈ 512 for a typical vindex).

    Metadata columns: cluster_id, cluster_label, count, top_tokens.
    """
    clusters_path = os.path.join(vindex_path, "relation_clusters.json")
    if not os.path.exists(clusters_path):
        raise FileNotFoundError(
            f"relation_clusters.json not found in {vindex_path}. "
            "Re-extract the vindex with the knowledge pipeline enabled."
        )

    with open(clusters_path) as f:
        data = json.load(f)

    if "centres" not in data:
        raise KeyError(
            "relation_clusters.json has no 'centres' key. "
            "The vindex was likely built without clustering."
        )

    centres = np.array(data["centres"], dtype=np.float32)   # [k, d_model]
    k = centres.shape[0]

    labels = data.get("labels", [f"cluster_{i}" for i in range(k)])
    counts = data.get("counts", [0] * k)
    top_tokens_raw = data.get("top_tokens", [[] for _ in range(k)])

    metadata = pd.DataFrame({
        "cluster_id": np.arange(k, dtype=np.int32),
        "cluster_label": labels,
        "count": counts,
        "top_tokens": [" / ".join(t[:3]) if t else "" for t in top_tokens_raw],
    })

    index_data = _read_index_json(vindex_path)

    return DirectionBundle(
        vectors=centres,
        metadata=metadata,
        vindex_hash=_vindex_hash(index_data),
        config={
            "loader": "load_relation_offsets",
            "vindex": vindex_path,
            "model": index_data.get("model", ""),
            "k": k,
        },
    )


def load_gates(
    vindex_path: str,
    layers: Optional[Sequence[int]] = None,
    sample_per_layer: int = 0,
    random_state: int = 42,
) -> DirectionBundle:
    """Load gate vectors from selected layers.

    Parameters
    ----------
    vindex_path:
        Path to .vindex directory.
    layers:
        Layer indices to load. Defaults to the model's knowledge band
        (from layer_bands in index.json), or all layers if absent.
    sample_per_layer:
        If > 0, randomly sample this many features per layer (useful for
        large models where full gate matrices exceed memory budgets).
    random_state:
        Seed for reproducible sampling.

    Metadata columns: layer, feature_idx, top_token, c_score.
    """
    import larql

    vindex = larql.load(vindex_path)
    index_data = _read_index_json(vindex_path)

    if layers is None:
        bands = vindex.layer_bands()
        if bands and "knowledge" in bands:
            start, end = bands["knowledge"]
            layers = list(range(start, end + 1))
        else:
            layers = list(range(vindex.num_layers))

    rng = np.random.default_rng(random_state)

    all_vecs: list[np.ndarray] = []
    all_meta: list[dict] = []

    for layer in layers:
        n_feat = vindex.num_features(layer)
        if n_feat == 0:
            continue

        gates = np.array(vindex.gate_vectors(layer=layer), dtype=np.float32)

        if sample_per_layer > 0 and n_feat > sample_per_layer:
            idx = rng.choice(n_feat, sample_per_layer, replace=False)
            gates = gates[idx]
            feat_indices = idx.tolist()
        else:
            feat_indices = list(range(n_feat))

        all_vecs.append(gates)
        for feat_idx in feat_indices:
            meta = vindex.feature_meta(layer, feat_idx)
            all_meta.append({
                "layer": layer,
                "feature_idx": feat_idx,
                "top_token": meta.top_token if meta else "",
                "c_score": meta.c_score if meta else 0.0,
            })

    if not all_vecs:
        raise ValueError(f"No gate vectors found in {vindex_path} for layers {layers}")

    vectors = np.vstack(all_vecs)
    metadata = pd.DataFrame(all_meta)

    return DirectionBundle(
        vectors=vectors,
        metadata=metadata,
        vindex_hash=_vindex_hash(index_data),
        config={
            "loader": "load_gates",
            "vindex": vindex_path,
            "model": index_data.get("model", ""),
            "layers": list(layers),
            "sample_per_layer": sample_per_layer,
        },
    )


def load_embeddings(
    vindex_path: str,
    sample: int = 0,
    random_state: int = 42,
) -> DirectionBundle:
    """Load the full token embedding matrix.

    Parameters
    ----------
    vindex_path:
        Path to .vindex directory.
    sample:
        If > 0, randomly sample this many token embeddings.

    Metadata columns: token_id, token.
    """
    import larql

    vindex = larql.load(vindex_path)
    index_data = _read_index_json(vindex_path)

    matrix = np.array(vindex.embedding_matrix(), dtype=np.float32)  # [vocab, d_model]
    vocab_size = matrix.shape[0]

    if sample > 0 and vocab_size > sample:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(vocab_size, sample, replace=False)
        matrix = matrix[idx]
        token_ids = idx.tolist()
    else:
        token_ids = list(range(vocab_size))

    tokens = [vindex.decode([tid]) for tid in token_ids]

    metadata = pd.DataFrame({
        "token_id": token_ids,
        "token": tokens,
    })

    return DirectionBundle(
        vectors=matrix,
        metadata=metadata,
        vindex_hash=_vindex_hash(index_data),
        config={
            "loader": "load_embeddings",
            "vindex": vindex_path,
            "model": index_data.get("model", ""),
            "sample": sample,
        },
    )
