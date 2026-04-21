#!/usr/bin/env python3
"""Per-cluster SVD — the correct replication of the 5-22D finding.

The 5-22D claim is about within-relation dimensionality: take all offset
vectors assigned to one relation cluster, run SVD on those, report the
50%-variance dim. The range across clusters is the 5-22D range.

Pooled SVD (what replicate_relation_offsets.py does) measures the union of
all relation subspaces, which is naturally much higher-dimensional and not
the right object.

Usage:
    python per_cluster_svd.py --vindex gemma3-4b.vindex
    python per_cluster_svd.py --vindex gemma3-4b.vindex --min-cluster-size 30
"""

import argparse
import json
import os
import re
import struct
import sys
import time

import numpy as np
from sklearn.preprocessing import normalize

ASCII_WORD = re.compile(r"^\u2581[A-Za-z]{3,}$")


# ── data loading (mirrors generate_clusters.py) ───────────────────────────────

def read_index(p): return json.load(open(os.path.join(p, "index.json")))

def read_embeddings(p, idx):
    dtype = np.float16 if idx.get("dtype") == "f16" else np.float32
    hidden, vocab = idx["hidden_size"], idx.get("vocab_size")
    raw = np.fromfile(os.path.join(p, "embeddings.bin"), dtype=dtype)
    return raw.reshape(vocab if vocab else raw.size // hidden, hidden).astype(np.float32)

def read_down_meta(p):
    MAGIC = 0x444D4554
    triples = []
    with open(os.path.join(p, "down_meta.bin"), "rb") as f:
        magic, _, num_layers, top_k = struct.unpack("<IIII", f.read(16))
        if magic != MAGIC: raise ValueError(f"Bad magic")
        entry_size = 4 + 4 + top_k * 8
        for layer in range(num_layers):
            (nf,) = struct.unpack("<I", f.read(4))
            raw = f.read(nf * entry_size)
            for i in range(nf):
                (tid,) = struct.unpack_from("<I", raw, i * entry_size)
                triples.append((layer, i, tid))
    return triples

def build_ww_vocab(tok_path, embed):
    tok_json = json.load(open(tok_path))
    special_ids = {t["id"] for t in tok_json.get("added_tokens", [])}
    ids = np.array([tid for t, tid in tok_json["model"]["vocab"].items()
                    if ASCII_WORD.match(t) and tid < embed.shape[0] and tid not in special_ids],
                   dtype=np.int32)
    return ids, embed[ids], special_ids

def read_gate_vectors(p, idx):
    dtype = np.float16 if idx.get("dtype") == "f16" else np.float32
    hidden = idx["hidden_size"]
    raw = np.fromfile(os.path.join(p, "gate_vectors.bin"), dtype=dtype)
    return raw.reshape(raw.size // hidden, hidden).astype(np.float32)


def compute_directions(vindex_path, gate_cos_threshold, same_concept_threshold,
                       gate_cos_percentile=None):
    idx = read_index(vindex_path)
    bands = idx.get("layer_bands", {})
    kl_min = bands["knowledge"][0] if "knowledge" in bands else min(14, idx["num_layers"])
    kl_max = bands["knowledge"][1] + 1 if "knowledge" in bands else min(28, idx["num_layers"])
    print(f"  Knowledge layers: {kl_min}–{kl_max - 1}")

    embed = read_embeddings(vindex_path, idx)
    ww_ids, ww_embed, special_ids = build_ww_vocab(
        os.path.join(vindex_path, "tokenizer.json"), embed)
    ww_norm = ww_embed / (np.linalg.norm(ww_embed, axis=1, keepdims=True) + 1e-8)
    print(f"  Whole-word vocab: {len(ww_ids):,}  special excluded: {len(special_ids):,}")

    triples = read_down_meta(vindex_path)
    layers_info = idx.get("layers", [])
    layer_of = np.empty(len(triples), dtype=np.int32)
    fg = 0
    for li in layers_info:
        n = li["num_features"]; layer_of[fg:fg+n] = li["layer"]; fg += n

    kl_idx = np.where((layer_of >= kl_min) & (layer_of < kl_max))[0]
    kl_out_ids = np.array([triples[i][2] for i in kl_idx], dtype=np.int32)
    print(f"  Knowledge-layer features: {len(kl_idx):,}")

    all_gates = read_gate_vectors(vindex_path, idx)
    kl_gates = all_gates[kl_idx]

    print(f"  Gate NN search ({len(kl_idx):,} × {len(ww_ids):,}) ...")
    batch, n_kl = 2048, len(kl_idx)
    in_ids = np.zeros(n_kl, dtype=np.int32)
    gate_cos = np.zeros(n_kl, dtype=np.float32)
    n_batches = (n_kl + batch - 1) // batch
    t0 = time.perf_counter()
    for bi, s in enumerate(range(0, n_kl, batch)):
        e = min(s + batch, n_kl)
        g = kl_gates[s:e]
        g_norm = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-8)
        scores = g_norm @ ww_norm.T
        best = scores.argmax(axis=1)
        in_ids[s:e] = ww_ids[best]
        gate_cos[s:e] = scores[np.arange(len(best)), best]
        if (bi + 1) % 10 == 0 or bi == n_batches - 1:
            elapsed = time.perf_counter() - t0
            print(f"    {bi+1}/{n_batches}  {elapsed:.0f}s  eta {elapsed/(bi+1)*(n_batches-bi-1):.0f}s",
                  end="\r", flush=True)
    print()

    # Always print percentile table — needed for threshold calibration across models
    print(f"  Gate cosine distribution (all {n_kl:,} features):")
    for p in [50, 75, 85, 90, 95, 99]:
        print(f"    p{p:2d}: {np.percentile(gate_cos, p):.4f}")

    # Resolve threshold: percentile-based overrides fixed value
    if gate_cos_percentile is not None:
        gate_cos_threshold = float(np.percentile(gate_cos, gate_cos_percentile))
        keep_pct = 100 - gate_cos_percentile
        print(f"  Percentile p{gate_cos_percentile} → threshold {gate_cos_threshold:.4f} "
              f"(keeping top {keep_pct}% of features)")
    else:
        keep_n = (gate_cos >= gate_cos_threshold).sum()
        keep_pct = keep_n / n_kl * 100
        print(f"  Fixed threshold {gate_cos_threshold} → keeping {keep_n:,} / {n_kl:,} "
              f"features ({keep_pct:.1f}%)")

    valid = ((kl_out_ids > 0) & (~np.isin(kl_out_ids, list(special_ids))) &
             (kl_out_ids < embed.shape[0]) & (gate_cos >= gate_cos_threshold) &
             (in_ids != kl_out_ids))

    out_vecs = embed[kl_out_ids[valid]]
    in_vecs  = embed[in_ids[valid]]
    dot = np.einsum("ij,ij->i",
                    out_vecs / (np.linalg.norm(out_vecs, axis=1, keepdims=True) + 1e-8),
                    in_vecs  / (np.linalg.norm(in_vecs,  axis=1, keepdims=True) + 1e-8))
    diff_mask = dot <= same_concept_threshold
    out_vecs, in_vecs = out_vecs[diff_mask], in_vecs[diff_mask]
    valid_idx = np.where(valid)[0][diff_mask]

    dirs = normalize((out_vecs - in_vecs).astype(np.float32), norm="l2")
    keep = np.linalg.norm(dirs, axis=1) > 0.1
    print(f"  Directions after all filters: {keep.sum():,}  "
          f"(gate≥{gate_cos_threshold:.4f}: {valid.sum():,}  "
          f"cosine≤{same_concept_threshold}: {diff_mask.sum():,})")

    return dirs[keep], kl_out_ids[valid_idx][keep], in_ids[valid_idx][keep], embed


# ── per-cluster SVD ───────────────────────────────────────────────────────────

def build_vocab_inv(vindex_path):
    tok_path = os.path.join(vindex_path, "tokenizer.json")
    try:
        vocab = json.load(open(tok_path))["model"]["vocab"]
        return {tid: tok.replace("\u2581", " ").strip() for tok, tid in vocab.items()}
    except Exception:
        return {}


def top_pairs(out_ids_c, in_ids_c, vocab_inv, n=8):
    from collections import Counter
    pairs = list(zip(in_ids_c.tolist(), out_ids_c.tolist()))
    common = Counter(pairs).most_common(n)
    def dec(tid): return vocab_inv.get(tid, f"#{tid}")
    return [{"in": dec(inp), "out": dec(out), "count": cnt} for (inp, out), cnt in common]


def per_cluster_svd(dirs, assignments, out_ids, in_ids, vocab_inv, min_size=30):
    k = int(assignments.max()) + 1
    results = []
    skipped = 0

    for c in range(k):
        mask = assignments == c
        n = mask.sum()
        if n < min_size:
            skipped += 1
            continue

        X = dirs[mask].astype(np.float64)
        X -= X.mean(axis=0)
        S = np.linalg.svd(X, compute_uv=False)
        var = S ** 2
        total = var.sum()
        if total < 1e-12:
            skipped += 1
            continue
        cum = np.cumsum(var) / total

        d50 = int(np.searchsorted(cum, 0.50)) + 1
        d90 = int(np.searchsorted(cum, 0.90)) + 1
        gap_idx = int(np.argmax(S[:-1] / (S[1:] + 1e-12)))
        gap_ratio = float(S[gap_idx] / (S[gap_idx + 1] + 1e-12)) if len(S) > 1 else 1.0

        pairs = top_pairs(out_ids[mask], in_ids[mask], vocab_inv)

        results.append({"cluster": c, "n": int(n), "d50": d50, "d90": d90,
                        "spectral_gap_idx": gap_idx, "spectral_gap_ratio": gap_ratio,
                        "top_pairs": pairs})

    return results, skipped


def report(results, skipped, k):
    if not results:
        print(f"  No clusters passed the min-size filter (skipped {skipped}/{k}).")
        return

    d50s = np.array([r["d50"] for r in results])
    d90s = np.array([r["d90"] for r in results])
    gaps = np.array([r["spectral_gap_ratio"] for r in results])

    in_range = ((d50s >= 5) & (d50s <= 22)).sum()
    pct = in_range / len(d50s) * 100

    print(f"\n── Per-cluster 50%-variance dimensionality ({len(results)} clusters, {skipped} skipped) ──")
    print(f"  {'Stat':<12} {'d50':>6} {'d90':>6} {'gap_ratio':>10}")
    print(f"  {'─'*38}")
    for label, arr in [("median", np.median), ("mean", np.mean),
                        ("p5", lambda x: np.percentile(x, 5)),
                        ("p25", lambda x: np.percentile(x, 25)),
                        ("p75", lambda x: np.percentile(x, 75)),
                        ("p95", lambda x: np.percentile(x, 95))]:
        print(f"  {label:<12} {arr(d50s):>6.1f} {arr(d90s):>6.1f} {arr(gaps):>10.2f}")

    print(f"\n  Range: d50 = {d50s.min()}–{d50s.max()}D  (target 5-22D)")
    print(f"  Clusters with d50 in 5–22D: {in_range}/{len(results)} ({pct:.0f}%)")

    verdict = "✓ REPLICATION PASSES" if pct >= 50 else "✗ REPLICATION FAILS"
    print(f"\n  {verdict}")
    print(f"  Criterion: ≥50% of clusters have d50 in 5–22D (got {pct:.0f}%)")

    # ── Passing clusters: sorted by d50, with token pairs
    passing = sorted([r for r in results if 5 <= r["d50"] <= 22], key=lambda r: r["d50"])
    print(f"\n── Passing clusters ({len(passing)}) — sorted by d50 ──")
    print(f"  {'c':>4} {'n':>6} {'d50':>5} {'d90':>5}  top pairs")
    for r in passing:
        pairs_str = "  ".join(f"{p['in']}→{p['out']}({p['count']})" for p in r["top_pairs"][:5])
        print(f"  {r['cluster']:>4} {r['n']:>6} {r['d50']:>5} {r['d90']:>5}  {pairs_str}")

    # ── Failing large clusters: top-10 by size among d50 > 22
    failing = sorted([r for r in results if r["d50"] > 22], key=lambda r: -r["n"])
    print(f"\n── Failing clusters ({len(failing)}) — top-10 by size ──")
    print(f"  {'c':>4} {'n':>6} {'d50':>5} {'d90':>5}  top pairs")
    for r in failing[:10]:
        pairs_str = "  ".join(f"{p['in']}→{p['out']}({p['count']})" for p in r["top_pairs"][:5])
        print(f"  {r['cluster']:>4} {r['n']:>6} {r['d50']:>5} {r['d90']:>5}  {pairs_str}")

    return {"n_clusters_analyzed": len(results), "skipped": skipped,
            "d50_median": float(np.median(d50s)), "d50_range": [int(d50s.min()), int(d50s.max())],
            "pct_in_target": float(pct), "passes": bool(pct >= 50),
            "n_passing": int(in_range), "n_failing": int(len(results) - in_range)}


def main():
    parser = argparse.ArgumentParser(
        description="Per-cluster SVD — correct replication of 5-22D finding.")
    parser.add_argument("--vindex", default="gemma3-4b.vindex")
    parser.add_argument("--k", type=int, default=64,
                        help="Number of clusters for k-means (default 64)")
    parser.add_argument("--min-cluster-size", type=int, default=30,
                        help="Skip clusters smaller than this (default 30)")
    parser.add_argument("--gate-cos-threshold", type=float, default=0.15,
                        help="Fixed gate cosine threshold (default 0.15). Ignored if "
                             "--gate-cos-percentile is set.")
    parser.add_argument("--gate-cos-percentile", type=float, default=None,
                        help="Keep features above this percentile of the gate cosine "
                             "distribution (e.g. 85 keeps top 15%%). Overrides "
                             "--gate-cos-threshold. Recommended for cross-model comparisons.")
    parser.add_argument("--same-concept-threshold", type=float, default=0.9)
    args = parser.parse_args()

    if not os.path.isdir(args.vindex):
        sys.exit(f"Not a directory: {args.vindex}")

    print(f"Computing offset directions from {args.vindex} ...")
    dirs, out_ids, in_ids, embed = compute_directions(
        args.vindex, args.gate_cos_threshold, args.same_concept_threshold,
        gate_cos_percentile=args.gate_cos_percentile)

    if len(dirs) < args.k * args.min_cluster_size:
        print(f"\nWarning: {len(dirs):,} directions / k={args.k} = "
              f"{len(dirs)//args.k} per cluster avg — "
              f"below min_cluster_size={args.min_cluster_size}. "
              f"Consider lowering --gate-cos-threshold or --k.")

    from sklearn.cluster import MiniBatchKMeans
    print(f"\nFitting k={args.k} clusters ...")
    km = MiniBatchKMeans(n_clusters=args.k, random_state=42, batch_size=4096,
                         max_iter=300, n_init=5)
    assignments = km.fit_predict(dirs)
    print(f"  Inertia: {km.inertia_:.1f}  "
          f"smallest cluster: {np.bincount(assignments).min()}")

    vocab_inv = build_vocab_inv(args.vindex)

    print(f"\nRunning per-cluster SVD (min_size={args.min_cluster_size}) ...")
    results, skipped = per_cluster_svd(dirs, assignments, out_ids, in_ids, vocab_inv,
                                       args.min_cluster_size)
    summary = report(results, skipped, args.k)

    # Save
    out_dir = f"outputs/{os.path.basename(args.vindex.rstrip('/'))}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "per_cluster_svd.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "clusters": results,
                   "params": vars(args)}, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Cache filtered offsets so visualize_manifold.py can skip the 77-second gate search
    cache_path = os.path.join(out_dir, "offsets_cached.npz")
    np.savez(cache_path, dirs=dirs, out_ids=out_ids, in_ids=in_ids, assignments=assignments)
    print(f"Cached: {cache_path}")


if __name__ == "__main__":
    main()
