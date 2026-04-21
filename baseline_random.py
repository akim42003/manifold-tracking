#!/usr/bin/env python3
"""Random-baseline control for the relation offset dimensionality experiment.

Generates N random unit vectors in the same ambient dimension as a target
vindex and runs the identical SVD + clustering pipeline.  Provides the null
distribution for interpreting whether observed dimensionality and cluster
quality are meaningfully above chance.

Usage:
    python baseline_random.py --vindex tinyllama.vindex
    python baseline_random.py --vindex gemma3-4b.vindex
    python baseline_random.py --n 44798 --d 2048   # manual size
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd


def infer_size_from_vindex(vindex_path: str) -> tuple[int, int]:
    """Return (n_features_knowledge_layers, hidden_size) from a vindex."""
    idx = json.load(open(os.path.join(vindex_path, "index.json")))
    num_layers = idx["num_layers"]
    hidden = idx["hidden_size"]
    kl_min, kl_max = min(14, num_layers), min(28, num_layers)
    layers_info = idx.get("layers", [])
    n = sum(li["num_features"] for li in layers_info
            if kl_min <= li["layer"] < kl_max)
    return n, hidden


def run_baseline(n: int, d: int, seed: int, n_components: int = 80) -> dict:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X = normalize(X, norm="l2")

    print(f"  Generated {n:,} random unit vectors in {d}D")

    # SVD
    t0 = time.perf_counter()
    Xc = X - X.mean(axis=0)
    U, S, Vt = randomized_svd(Xc, n_components=n_components, random_state=seed)
    print(f"  SVD done ({time.perf_counter() - t0:.1f}s)")

    total_var = np.sum(Xc ** 2)
    cum_var = np.cumsum(S ** 2) / total_var

    dims = {}
    for t in [0.50, 0.90, 0.95]:
        crossings = np.where(cum_var >= t)[0]
        dims[f"{int(t*100)}pct"] = int(crossings[0]) + 1 if len(crossings) else n_components

    ratios = S[:-1] / S[1:]
    gap_idx = int(np.argmax(ratios[:50]))

    print(f"\n  Dimensionality (random baseline, {n:,} vectors, {d}D):")
    for label, v in [("50%", dims["50pct"]), ("90%", dims["90pct"]), ("95%", dims["95pct"])]:
        print(f"    {label} variance: {v:4d}D")
    print(f"  Spectral gap: component {gap_idx + 1} (ratio {ratios[gap_idx]:.2f}×)")
    print(f"  Top-5 singular values: " + "  ".join(f"{s:.3f}" for s in S[:5]))

    # Clustering at k=64, 128, 256
    print(f"\n  Clustering (random baseline):")
    cluster_results = {}
    for k in [64, 128, 256]:
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096,
                             max_iter=100, n_init=3)
        labels = km.fit_predict(X)
        inertia_pp = km.inertia_ / n
        sample_n = min(3000, n)
        si = rng.choice(n, sample_n, replace=False)
        sil = silhouette_score(X[si], labels[si], metric="cosine")
        print(f"    k={k:3d}: inertia/pt={inertia_pp:.4f}  silhouette={sil:.4f}")
        cluster_results[k] = {"inertia_per_point": inertia_pp, "silhouette": float(sil)}

    return {
        "n": n, "d": d, "seed": seed,
        "dimensionality": dims,
        "spectral_gap_idx": gap_idx,
        "spectral_gap_ratio": float(ratios[gap_idx]),
        "top5_singular_values": S[:5].tolist(),
        "clustering": cluster_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Random-baseline control for relation offset dimensionality."
    )
    parser.add_argument("--vindex", default=None,
                        help="Infer n and d from this vindex's knowledge layers")
    parser.add_argument("--n", type=int, default=None, help="Number of vectors")
    parser.add_argument("--d", type=int, default=None, help="Ambient dimension")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-components", type=int, default=80)
    args = parser.parse_args()

    if args.vindex:
        if not os.path.isdir(args.vindex):
            sys.exit(f"Not a directory: {args.vindex}")
        n, d = infer_size_from_vindex(args.vindex)
        print(f"Inferred from {args.vindex}: n={n:,}, d={d}")
    elif args.n and args.d:
        n, d = args.n, args.d
    else:
        sys.exit("Provide --vindex or both --n and --d")

    print(f"\nRunning random baseline (n={n:,}, d={d}, seed={args.seed}) ...")
    result = run_baseline(n, d, args.seed, args.n_components)

    print("\n── Interpretation guide ──────────────────────────────────────────")
    print("Compare these numbers against your real-data results.")
    print("If real 50%-variance dim ≈ baseline: no meaningful low-dim structure.")
    print("If real 50%-variance dim << baseline: genuine low-dimensional manifold.")
    print(f"\nBaseline 50%-variance dim: {result['dimensionality']['50pct']}D")
    print("(for random unit vectors, expect ~n/2 at 50% variance)")


if __name__ == "__main__":
    main()
