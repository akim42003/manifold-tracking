#!/usr/bin/env python3
"""Replication script: relation offset vectors are 5-22D (50% variance).

Replicates LARQL's published finding on the dimensionality of the relation
manifold, then produces the first novel visualization: UMAP of offset clusters.

Usage
-----
    # 1. Build the vindex (one-time, ~2 min for GPT-2 small)
    #    larql extract-index gpt2 -o gpt2.vindex --level inference
    #
    # 2. Run this script
    #    python replicate_relation_offsets.py --vindex gpt2.vindex

    python replicate_relation_offsets.py --vindex gpt2.vindex [--out outputs/]

Expected output
---------------
    Loaded 512 relation offset vectors (768D)

    SVD (200 bootstrap resamples, 90% CI)
    ─────────────────────────────────────
    50% variance:  6D   (target: 5-22D)   ✓
    90% variance: 31D
    95% variance: 42D
    Spectral gap:  @ component 7 (ratio 2.3x)

    Saved: outputs/spectrum.png
    Saved: outputs/umap_by_cluster.png
    Saved: outputs/results.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np


# ── pre-flight checks ─────────────────────────────────────────────────────────

def check_vindex(path: str) -> None:
    if not os.path.isdir(path):
        sys.exit(
            f"\nVindex not found: {path}\n"
            "Build it first:\n"
            f"    larql extract-index gpt2 -o {path} --level inference\n"
            "(takes ~2 minutes for GPT-2 small, ~500 MB on disk)"
        )
    clusters = os.path.join(path, "relation_clusters.json")
    if not os.path.exists(clusters):
        sys.exit(
            f"\nrelation_clusters.json missing from {path}\n"
            "The vindex was extracted without the knowledge pipeline.\n"
            "Re-extract with: larql extract-index gpt2 -o {path} --level inference"
        )


def check_deps() -> None:
    missing = []
    try:
        import umap  # noqa: F401
    except ImportError:
        missing.append("umap-learn")
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        missing.append("matplotlib")
    if missing:
        sys.exit(f"\nMissing dependencies: {', '.join(missing)}\n"
                 f"    pip install {' '.join(missing)}")


# ── analysis ──────────────────────────────────────────────────────────────────

def run(vindex_path: str, out_dir: str, n_bootstrap: int, random_state: int) -> dict:
    from manifold_tools import directions as dirs, manifolds, visualize

    os.makedirs(out_dir, exist_ok=True)

    # ── 1. load ───────────────────────────────────────────────────────────────
    print(f"\nLoading relation offsets from {vindex_path} ...")
    t0 = time.perf_counter()
    bundle = dirs.load_relation_offsets(vindex_path)
    print(f"  {bundle.n} vectors × {bundle.d_model}D  "
          f"({time.perf_counter() - t0:.2f}s)")

    # ── 2. SVD with bootstrap ─────────────────────────────────────────────────
    print(f"\nRunning SVD ({n_bootstrap} bootstrap resamples, 90% CI) ...")
    t0 = time.perf_counter()
    svd = manifolds.svd_with_bootstrap(
        bundle,
        n_bootstrap=n_bootstrap,
        bootstrap_ci=0.90,
        random_state=random_state,
    )
    print(f"  Done ({time.perf_counter() - t0:.1f}s)")

    # ── 3. report dimensionality ──────────────────────────────────────────────
    thresholds = [0.50, 0.90, 0.95, 0.99]
    dims = {}
    print("\nSVD dimensionality")
    print("──────────────────")
    for t in thresholds:
        crossings = np.where(svd.cumulative_variance >= t)[0]
        d = int(crossings[0]) + 1 if len(crossings) else svd.d_model
        dims[f"{int(t*100)}pct"] = d

        # The target finding: 50% variance should be 5-22D
        target_note = ""
        if t == 0.50:
            in_range = 5 <= d <= 22
            target_note = f"  {'✓' if in_range else '✗'} (target: 5-22D)"

        print(f"  {int(t*100):3d}% variance: {d:4d}D{target_note}")

    print(f"\n  Spectral gap: component {svd.spectral_gap_idx + 1} "
          f"(ratio {svd.spectral_gap_ratio:.1f}×)")
    print(f"  Top 5 singular values: "
          + "  ".join(f"{s:.2f}" for s in svd.singular_values[:5]))

    # ── 4. UMAP ───────────────────────────────────────────────────────────────
    print("\nRunning UMAP projection ...")
    t0 = time.perf_counter()
    proj = manifolds.umap_project(
        bundle,
        n_components=2,
        metric="cosine",
        n_neighbors=min(15, bundle.n - 1),
        min_dist=0.1,
        random_state=random_state,
    )
    print(f"  Done ({time.perf_counter() - t0:.1f}s)")

    # ── 5. plots ──────────────────────────────────────────────────────────────
    model_name = bundle.config.get("model", vindex_path)
    short_name = os.path.basename(vindex_path.rstrip("/"))

    spectrum_path = os.path.join(out_dir, "spectrum.png")
    fig = visualize.spectrum_plot(
        svd,
        title=f"Relation offset spectrum — {short_name}",
        max_components=60,
    )
    fig.savefig(spectrum_path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"\nSaved: {spectrum_path}")

    umap_path = os.path.join(out_dir, "umap_by_cluster.png")
    fig = visualize.projection_plot(
        proj,
        color_by="cluster_label",
        title=f"Relation clusters (UMAP, cosine) — {short_name}",
    )
    fig.savefig(umap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {umap_path}")

    # also save a version colored by cluster ID (continuous) to see density
    umap_id_path = os.path.join(out_dir, "umap_by_cluster_id.png")
    fig = visualize.projection_plot(
        proj,
        color_by="cluster_id",
        title=f"Relation cluster IDs (UMAP, cosine) — {short_name}",
    )
    fig.savefig(umap_id_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {umap_id_path}")

    # ── 6. sidecar JSON ───────────────────────────────────────────────────────
    results = {
        "vindex": vindex_path,
        "model": model_name,
        "vindex_hash": bundle.vindex_hash,
        "n_vectors": bundle.n,
        "d_model": bundle.d_model,
        "dimensionality": dims,
        "spectral_gap_idx": svd.spectral_gap_idx,
        "spectral_gap_ratio": svd.spectral_gap_ratio,
        "top_10_singular_values": svd.singular_values[:10].tolist(),
        "cumulative_variance_at_50pct_dim": float(
            svd.cumulative_variance[dims["50pct"] - 1]
        ),
        "n_bootstrap": n_bootstrap,
        "random_state": random_state,
        "replication_check": {
            "target_50pct_range": "5-22D",
            "observed_50pct_dim": dims["50pct"],
            "passes": 5 <= dims["50pct"] <= 22,
        },
    }

    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {results_path}")

    # ── 7. verdict ────────────────────────────────────────────────────────────
    passes = results["replication_check"]["passes"]
    verdict = "✓ REPLICATION PASSES" if passes else "✗ REPLICATION FAILS"
    dim_50 = dims["50pct"]
    print(f"\n{verdict}")
    print(f"  50% variance at {dim_50}D (target 5-22D)")
    if not passes:
        print("  Check: does this vindex have valid relation clusters?")
        print("  Low k or noisy clusters can inflate the apparent dimensionality.")

    return results


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replicate the relation offset 5-22D manifold finding."
    )
    parser.add_argument(
        "--vindex", default="gpt2.vindex",
        help="Path to extracted .vindex directory (default: gpt2.vindex)"
    )
    parser.add_argument(
        "--out", default="outputs/relation_offsets",
        help="Output directory for plots and results.json"
    )
    parser.add_argument(
        "--bootstrap", type=int, default=200,
        help="Number of bootstrap resamples (default: 200)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    check_deps()
    check_vindex(args.vindex)
    run(args.vindex, args.out, args.bootstrap, args.seed)


if __name__ == "__main__":
    main()
