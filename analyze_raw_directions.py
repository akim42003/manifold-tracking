#!/usr/bin/env python3
"""Analyze raw offset directions from a vindex — four diagnostics:

  1. SVD on raw ~44k directions (apples-to-apples with LARQL 5-22D finding)
  2. Cluster quality: silhouette sample + Davies-Bouldin + inertia vs k
  3. Top-PC interpretation: which cluster centroids are extreme along PC1-3
  4. Summary verdict

Usage:
    python analyze_raw_directions.py --vindex tinyllama.vindex
"""

import argparse
import json
import os
import re
import struct
import sys
import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd

ASCII_WORD = re.compile(r"^\u2581[A-Za-z]{3,}$")


# ── helpers (mirrors generate_clusters.py) ────────────────────────────────────

def read_index(p):
    return json.load(open(os.path.join(p, "index.json")))


def read_embeddings(p, idx):
    dtype = np.float16 if idx.get("dtype") == "f16" else np.float32
    hidden, vocab = idx["hidden_size"], idx.get("vocab_size")
    raw = np.fromfile(os.path.join(p, "embeddings.bin"), dtype=dtype)
    n = vocab if vocab else raw.size // hidden
    return raw.reshape(n, hidden).astype(np.float32)


def read_down_meta(p):
    MAGIC = 0x444D4554
    triples = []
    with open(os.path.join(p, "down_meta.bin"), "rb") as f:
        magic, _, num_layers, top_k = struct.unpack("<IIII", f.read(16))
        if magic != MAGIC:
            raise ValueError(f"Bad magic 0x{magic:08X}")
        entry_size = 4 + 4 + top_k * 8
        for layer in range(num_layers):
            (nf,) = struct.unpack("<I", f.read(4))
            raw = f.read(nf * entry_size)
            for i in range(nf):
                (tid,) = struct.unpack_from("<I", raw, i * entry_size)
                triples.append((layer, i, tid))
    return triples


def build_ww_vocab(tok_path, embed):
    """ASCII-Latin whole-word tokens: ▁ + ≥3 Latin letters, no special/added tokens."""
    tok_json = json.load(open(tok_path))
    vocab = tok_json["model"]["vocab"]
    special_ids = {t["id"] for t in tok_json.get("added_tokens", [])}
    ids = np.array(
        [tid for token, tid in vocab.items()
         if ASCII_WORD.match(token) and tid < embed.shape[0] and tid not in special_ids],
        dtype=np.int32,
    )
    return ids, embed[ids], special_ids


def read_gate_vectors(p, idx):
    dtype = np.float16 if idx.get("dtype") == "f16" else np.float32
    hidden = idx["hidden_size"]
    raw = np.fromfile(os.path.join(p, "gate_vectors.bin"), dtype=dtype)
    return raw.reshape(raw.size // hidden, hidden).astype(np.float32)


def compute_raw_directions(vindex_path, gate_cos_threshold=0.3, same_concept_threshold=0.9):
    """Run the offset direction pipeline. Returns (dirs, valid_out_ids, valid_in_ids, embed)."""
    idx = read_index(vindex_path)
    num_layers = idx["num_layers"]

    layer_bands = idx.get("layer_bands", {})
    if "knowledge" in layer_bands:
        kl_min, kl_max = layer_bands["knowledge"][0], layer_bands["knowledge"][1] + 1
        print(f"  Knowledge layers from layer_bands: {kl_min}–{kl_max - 1}")
    else:
        kl_min, kl_max = min(14, num_layers), min(28, num_layers)
        print(f"  Knowledge layers (heuristic): {kl_min}–{kl_max - 1}")

    print("  Loading embeddings ...")
    embed = read_embeddings(vindex_path, idx)

    print("  Building whole-word vocab (ASCII-Latin only) ...")
    ww_ids, ww_embed, special_ids = build_ww_vocab(
        os.path.join(vindex_path, "tokenizer.json"), embed)
    ww_norm = ww_embed / (np.linalg.norm(ww_embed, axis=1, keepdims=True) + 1e-8)
    print(f"  Whole-word vocab: {len(ww_ids):,} tokens  (special excluded: {len(special_ids):,})")

    print("  Reading down_meta ...")
    triples = read_down_meta(vindex_path)
    layers_info = idx.get("layers", [])
    layer_of = np.empty(len(triples), dtype=np.int32)
    feat_global = 0
    for li in layers_info:
        n = li["num_features"]
        layer_of[feat_global:feat_global + n] = li["layer"]
        feat_global += n

    kl_mask = (layer_of >= kl_min) & (layer_of < kl_max)
    kl_idx = np.where(kl_mask)[0]
    kl_out_ids = np.array([triples[i][2] for i in kl_idx], dtype=np.int32)
    print(f"  Knowledge-layer features: {len(kl_idx):,}")

    print("  Loading gate vectors ...")
    all_gates = read_gate_vectors(vindex_path, idx)
    kl_gates = all_gates[kl_idx]

    print(f"  Gate NN search ({len(kl_idx):,} × {len(ww_ids):,} ww-tokens) ...")
    batch = 2048
    n_kl = len(kl_idx)
    in_ids = np.zeros(n_kl, dtype=np.int32)
    gate_best_cos = np.zeros(n_kl, dtype=np.float32)
    n_batches = (n_kl + batch - 1) // batch
    t_nn = time.perf_counter()
    for bi, s in enumerate(range(0, n_kl, batch)):
        e = min(s + batch, n_kl)
        g = kl_gates[s:e]
        g_norm = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-8)
        scores = g_norm @ ww_norm.T
        best = scores.argmax(axis=1)
        in_ids[s:e] = ww_ids[best]
        gate_best_cos[s:e] = scores[np.arange(len(best)), best]
        if (bi + 1) % 10 == 0 or bi == n_batches - 1:
            elapsed = time.perf_counter() - t_nn
            eta = elapsed / (bi + 1) * (n_batches - bi - 1)
            print(f"    batch {bi+1}/{n_batches}  {elapsed:.0f}s elapsed  {eta:.0f}s eta",
                  end="\r", flush=True)
    print()

    # Filter 1: valid output (not special/unused, in vocab range)
    valid = (
        (kl_out_ids > 0) &
        (~np.isin(kl_out_ids, list(special_ids))) &
        (kl_out_ids < embed.shape[0])
    )
    print(f"  After output-token filter:     {valid.sum():>8,}")

    # Filter 2: gate confidence
    valid &= (gate_best_cos >= gate_cos_threshold)
    print(f"  After gate cosine ≥{gate_cos_threshold}:       {valid.sum():>8,}")

    # Filter 3: in ≠ out
    valid &= (in_ids != kl_out_ids)
    print(f"  After in≠out:                  {valid.sum():>8,}")

    out_vecs = embed[kl_out_ids[valid]]
    in_vecs  = embed[in_ids[valid]]

    # Filter 4: same-concept cosine
    dot = np.einsum("ij,ij->i",
                    out_vecs / (np.linalg.norm(out_vecs, axis=1, keepdims=True) + 1e-8),
                    in_vecs  / (np.linalg.norm(in_vecs,  axis=1, keepdims=True) + 1e-8))
    diff_mask = dot <= same_concept_threshold
    out_vecs = out_vecs[diff_mask]
    in_vecs  = in_vecs[diff_mask]
    valid_indices = np.where(valid)[0][diff_mask]
    print(f"  After cosine(out,in) ≤{same_concept_threshold}:    {diff_mask.sum():>8,}")

    dirs = (out_vecs - in_vecs).astype(np.float32)
    dirs = normalize(dirs, norm="l2")
    keep = np.linalg.norm(dirs, axis=1) > 0.1
    dirs = dirs[keep]
    valid_indices = valid_indices[keep]
    print(f"  After degenerate filter:       {len(dirs):>8,}")

    return dirs, kl_out_ids[valid_indices], in_ids[valid_indices], embed


# ── 1. SVD on raw directions ──────────────────────────────────────────────────

def svd_raw(dirs, n_components=80):
    print(f"\n── 1. SVD on {dirs.shape[0]:,} raw offset directions ──")
    t0 = time.perf_counter()
    X = dirs - dirs.mean(axis=0)
    U, S, Vt = randomized_svd(X, n_components=n_components, random_state=42)
    print(f"   Done ({time.perf_counter() - t0:.1f}s)")

    total_var = np.sum(X ** 2)
    cum_var = np.cumsum(S ** 2) / total_var

    print(f"\n   Top-10 singular values: " + "  ".join(f"{s:.3f}" for s in S[:10]))
    print(f"\n   Dimensionality (raw directions):")
    for t in [0.50, 0.90, 0.95]:
        crossings = np.where(cum_var >= t)[0]
        d = int(crossings[0]) + 1 if len(crossings) else f">{n_components}"
        in_range = (5 <= d <= 22) if t == 0.50 and isinstance(d, int) else False
        mark = f"  {'✓' if in_range else '✗'} (target: 5-22D)" if t == 0.50 else ""
        print(f"   {int(t*100):3d}% variance: {str(d):>4}D{mark}")

    ratios = S[:-1] / S[1:]
    gap_idx = int(np.argmax(ratios))
    print(f"\n   Spectral gap: component {gap_idx + 1} (ratio {ratios[gap_idx]:.2f}×)")

    return S, cum_var, Vt


# ── 2. Cluster quality ────────────────────────────────────────────────────────

def cluster_quality(dirs, ks=(64, 128, 256), seed=42):
    print(f"\n── 2. Cluster quality ──")
    results = {}
    for k in ks:
        t0 = time.perf_counter()
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096,
                             max_iter=300, n_init=5)
        labels = km.fit_predict(dirs)
        inertia_per_point = km.inertia_ / len(dirs)

        sample_n = min(5000, len(dirs))
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(len(dirs), sample_n, replace=False)
        sil = silhouette_score(dirs[sample_idx], labels[sample_idx],
                               metric="cosine", sample_size=None)
        db = davies_bouldin_score(dirs[sample_idx], labels[sample_idx])

        print(f"   k={k:3d}:  inertia/pt={inertia_per_point:.4f}  "
              f"silhouette={sil:.4f}  davies-bouldin={db:.4f}  ({time.perf_counter()-t0:.0f}s)")
        results[k] = {"inertia_per_point": inertia_per_point,
                      "silhouette": sil, "davies_bouldin": db}

    sils = [results[k]["silhouette"] for k in ks]
    print(f"\n   Silhouette interpretation: "
          + ("clusters are meaningful (>0.1)" if max(sils) > 0.1
             else "clusters are mostly noise (≤0.1, structure may be absent)"))
    return results


# ── 3. Top-PC extreme centroids ───────────────────────────────────────────────

def top_pc_extremes(dirs, Vt, vindex_path, valid_out_ids, valid_in_ids, embed,
                    k=64, seed=42, n_extreme=5):
    print(f"\n── 3. Top-3 PC extreme clusters (k={k}) ──")

    tok_path = os.path.join(vindex_path, "tokenizer.json")
    try:
        vocab_inv = {}
        for token, tid in json.load(open(tok_path))["model"]["vocab"].items():
            vocab_inv[tid] = token.replace("\u2581", " ").strip()
    except Exception:
        vocab_inv = {}

    def decode_token(tid):
        return vocab_inv.get(int(tid), f"#{tid}") if vocab_inv else f"#{tid}"

    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096,
                         max_iter=300, n_init=5)
    labels = km.fit_predict(dirs)
    centres = normalize(km.cluster_centers_, norm="l2").astype(np.float32)
    n_use = len(dirs)

    for pc_idx in range(3):
        pc = Vt[pc_idx]
        proj = centres @ pc
        order_pos = np.argsort(-proj)[:n_extreme]
        order_neg = np.argsort(proj)[:n_extreme]

        def cluster_tokens(cids):
            parts = []
            for ci in cids:
                mask = labels == ci
                if not mask.any():
                    parts.append(f"c{ci}(empty)")
                    continue
                out_s = valid_out_ids[mask]
                in_s  = valid_in_ids[mask]
                top_out = [decode_token(t) for t in
                           np.unique(out_s)[np.argsort(
                               -np.bincount(out_s, minlength=embed.shape[0])[np.unique(out_s)])[:3]]]
                top_in  = [decode_token(t) for t in
                           np.unique(in_s)[np.argsort(
                               -np.bincount(in_s, minlength=embed.shape[0])[np.unique(in_s)])[:3]]]
                parts.append(f"c{ci}[{','.join(top_in)}→{','.join(top_out)}]")
            return "  ".join(parts)

        print(f"\n   PC{pc_idx + 1}")
        print(f"     + extreme: {cluster_tokens(order_pos)}")
        print(f"     - extreme: {cluster_tokens(order_neg)}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vindex", default="tinyllama.vindex")
    args = parser.parse_args()

    if not os.path.isdir(args.vindex):
        sys.exit(f"Not a directory: {args.vindex}")

    print(f"Computing raw offset directions from {args.vindex} ...")
    t0 = time.perf_counter()
    dirs, valid_out_ids, valid_in_ids, embed = compute_raw_directions(args.vindex)
    print(f"  Done ({time.perf_counter() - t0:.1f}s total)")

    S, cum_var, Vt = svd_raw(dirs, n_components=80)
    cluster_quality(dirs)
    top_pc_extremes(dirs, Vt, args.vindex, valid_out_ids, valid_in_ids, embed)


if __name__ == "__main__":
    main()
