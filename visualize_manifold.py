#!/usr/bin/env python3
"""Visualize per-cluster SVD results: spectrum, relation heatmap, UMAP.

Requires offsets_cached.npz produced by per_cluster_svd.py (run that first).

Usage:
    python visualize_manifold.py --vindex gemma3-4b.vindex --k 256
    python visualize_manifold.py --vindex gemma3-4b.vindex --k 256 --only umap
    python visualize_manifold.py --vindex gemma3-4b.vindex --k 256 --max-clusters-in-heatmap 50
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from manifold_tools._types import ProjectionResult
from manifold_tools.directions import bundle_from_vectors
from manifold_tools.manifolds import svd_with_bootstrap
from manifold_tools.visualize import projection_plot, spectrum_plot as _spectrum_plot


# ── helpers ───────────────────────────────────────────────────────────────────

def load_cache(out_dir, vindex_arg):
    path = os.path.join(out_dir, "offsets_cached.npz")
    if not os.path.exists(path):
        sys.exit(
            f"Cache not found: {path}\n"
            "Run per_cluster_svd.py first to generate it."
        )
    d = np.load(path, allow_pickle=True)
    if "vindex_path" in d:
        cached = str(d["vindex_path"])
        expected = os.path.abspath(vindex_arg)
        if cached != expected:
            print(f"WARNING: cache was built from {cached!r}, "
                  f"but --vindex is {expected!r}. "
                  f"Re-run per_cluster_svd.py if this is wrong.")
    return d["dirs"], d["out_ids"], d["in_ids"], d["assignments"]


def load_results(out_dir):
    path = os.path.join(out_dir, "per_cluster_svd.json")
    if not os.path.exists(path):
        sys.exit(f"Results not found: {path}")
    with open(path) as f:
        return json.load(f)


def cluster_label(r):
    """Short label: use stored label field if available, else derive from top_pairs."""
    if "label" in r:
        return r["label"][:20]
    if r.get("top_pairs"):
        p = r["top_pairs"][0]
        return f"{p['in']}→{p['out']}"[:20]
    return f"c{r['cluster']}"


# ── Figure 1: spectrum ────────────────────────────────────────────────────────

def figure_spectrum(dirs, assignments, clusters, args, out_dir, model_name):
    print("Generating spectrum plot ...")
    min_size = args.min_cluster_size
    d50_thresh = args.d50_threshold

    analyzed = [r for r in clusters if r["n"] >= min_size]
    if not analyzed:
        print("  No clusters passed min-size — skipping spectrum.")
        return

    d50s = np.array([r["d50"] for r in analyzed])
    vmin = d50s.min()
    vmax = max(d50s.max(), vmin + 1)
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10, 6))

    for r in analyzed:
        c = r["cluster"]
        mask = assignments == c
        n = mask.sum()
        if n < min_size:
            continue
        X = dirs[mask].astype(np.float64)
        X -= X.mean(axis=0)
        S = np.linalg.svd(X, compute_uv=False)
        var = S ** 2
        total = var.sum()
        if total < 1e-12:
            continue
        cum = np.cumsum(var) / total
        xs = np.arange(1, len(cum) + 1)
        color = cmap(norm(r["d50"]))
        ax.plot(xs, cum, color=color, alpha=0.4, linewidth=0.8)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, label="50% variance (d50)")
    ax.axvline(5,  color="steelblue", linestyle="--", linewidth=0.8, alpha=0.7, label="d=5")
    ax.axvline(22, color="tomato",    linestyle="--", linewidth=0.8, alpha=0.7, label="d=22")

    ax.set_xscale("log")
    ax.set_xlim(1, None)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Component index (log scale)")
    ax.set_ylabel("Cumulative variance fraction")
    ax.set_title(f"{model_name}  k={args.k}  —  cumulative variance per cluster")
    ax.legend(loc="lower right", fontsize=8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="d50")

    for ext in ("png", "svg"):
        p = os.path.join(out_dir, f"spectrum.{ext}")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)

    # ── Reference: package spectrum_plot on the cluster nearest median d50 ──
    rep_clusters = [r for r in analyzed if r.get("d50") is not None]
    if rep_clusters:
        d50_arr = np.array([r["d50"] for r in rep_clusters])
        med_d50 = float(np.median(d50_arr))
        rep = min(rep_clusters, key=lambda r: abs(r["d50"] - med_d50))
        c = rep["cluster"]
        rep_mask = assignments == c
        n_rep = int(rep_mask.sum())
        if n_rep >= min_size:
            X_rep = dirs[rep_mask]
            meta_rep = pd.DataFrame({"cluster_id": np.full(n_rep, c, dtype=np.int32)})
            bundle_rep = bundle_from_vectors(
                vectors=X_rep,
                metadata=meta_rep,
                config={"cluster_id": c, "source": "visualize_manifold"},
            )
            try:
                svd_rep = svd_with_bootstrap(
                    bundle_rep, n_components=min(80, n_rep - 1), n_bootstrap=50
                )
                ref_title = (
                    f"{model_name}  c{c}  (d50={rep['d50']}, n={n_rep}, "
                    f"near median d50={med_d50:.0f})"
                )
                fig_ref = _spectrum_plot(svd_rep, title=ref_title)
                for ext in ("png", "svg"):
                    p = os.path.join(out_dir, f"spectrum_reference.{ext}")
                    fig_ref.savefig(p, dpi=150, bbox_inches="tight")
                    print(f"  Saved: {p}")
                plt.close(fig_ref)
            except ValueError:
                print("  Skipped reference spectrum (degenerate cluster).")


# ── Figure 2: relation-relation similarity heatmap ───────────────────────────

def figure_heatmap(dirs, assignments, clusters, args, out_dir, model_name):
    try:
        from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
        from scipy.spatial.distance import squareform
    except ImportError:
        sys.exit("scipy is required for the heatmap. pip install scipy")

    print("Generating relation similarity heatmap ...")
    d50_thresh = args.d50_threshold
    min_size = args.min_cluster_size

    passing = [r for r in clusters if r["d50"] <= d50_thresh and r["n"] >= min_size]
    if not passing:
        print("  No passing clusters — skipping heatmap.")
        return

    # Limit heatmap size
    if args.max_clusters_in_heatmap and len(passing) > args.max_clusters_in_heatmap:
        passing = sorted(passing, key=lambda r: r["d50"])[: args.max_clusters_in_heatmap]
        print(f"  Limiting heatmap to top {args.max_clusters_in_heatmap} clusters by d50.")

    # Build centroid matrix
    centroids = []
    for r in passing:
        mask = assignments == r["cluster"]
        vecs = dirs[mask]
        mean = vecs.mean(axis=0)
        norm_val = np.linalg.norm(mean)
        centroids.append(mean / (norm_val + 1e-8))
    C = np.array(centroids, dtype=np.float32)
    sim = C @ C.T
    np.clip(sim, -1, 1, out=sim)

    # Hierarchical clustering for reordering
    dist = squareform(1 - sim, checks=False)
    dist = np.clip(dist, 0, None)
    Z = linkage(dist, method="average")
    order = leaves_list(Z)

    sim_ord = sim[np.ix_(order, order)]
    labels_ord = [cluster_label(passing[i]) for i in order]

    n = len(passing)
    cell = max(0.18, min(0.6, 10.0 / n))
    fig_w = max(8, n * cell + 2.5)
    fig_h = max(6, n * cell + 2.0)

    fig = plt.figure(figsize=(fig_w, fig_h))
    # Layout: [dendrogram left | heatmap | colorbar]
    ax_dend = fig.add_axes([0.02, 0.10, 0.10, 0.80])
    ax_heat = fig.add_axes([0.13, 0.10, 0.72, 0.80])
    ax_cbar = fig.add_axes([0.87, 0.10, 0.02, 0.80])

    dendrogram(Z, ax=ax_dend, orientation="left", color_threshold=0,
               above_threshold_color="gray", no_labels=True)
    ax_dend.axis("off")

    im = ax_heat.imshow(sim_ord, cmap="coolwarm", vmin=-1, vmax=1,
                        interpolation="nearest", aspect="auto")
    ax_heat.set_xticks(range(n))
    ax_heat.set_xticklabels(labels_ord, rotation=90, fontsize=max(4, min(8, 120 // n)))
    ax_heat.set_yticks(range(n))
    ax_heat.set_yticklabels(labels_ord, fontsize=max(4, min(8, 120 // n)))
    ax_heat.set_title(
        f"{model_name}  —  pairwise cosine similarity of relation centroids "
        f"(passing clusters only, d50≤{d50_thresh})\n"
        "Diagonal = 1.0 (trivially self-similar)",
        fontsize=9
    )

    fig.colorbar(im, cax=ax_cbar, label="cosine similarity")

    for ext in ("png", "svg"):
        p = os.path.join(out_dir, f"relation_similarity.{ext}")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


# ── Figure 3: UMAP ────────────────────────────────────────────────────────────

def figure_umap(dirs, assignments, clusters, args, out_dir, model_name):
    try:
        import umap
    except ImportError:
        sys.exit(
            "umap-learn is required for UMAP.\n"
            "pip install umap-learn --break-system-packages"
        )

    print(f"Running UMAP on {len(dirs):,} offset vectors ...")
    reducer = umap.UMAP(
        n_components=args.umap_components, n_neighbors=15, min_dist=0.1,
        metric="cosine", random_state=42, verbose=False
    )
    emb = reducer.fit_transform(dirs)
    print("  UMAP done.")

    cid_to_d50 = {r["cluster"]: r["d50"] for r in clusters}
    point_d50 = np.array([cid_to_d50.get(int(a), -1) for a in assignments], dtype=np.int32)

    meta_df = pd.DataFrame({
        "cluster_id": assignments.astype(np.int32),
        "d50": point_d50,
    })
    proj_result = ProjectionResult(
        coords=emb[:, :2].astype(np.float32),
        metadata=meta_df,
        n_components=2,
        metric="cosine",
        provenance={"method": "umap", "source": "visualize_manifold.py",
                    "model": model_name, "k": args.k},
    )

    fig_cluster = projection_plot(
        proj_result,
        color_by="cluster_id",
        title=f"{model_name}  k={args.k}  —  UMAP colored by cluster ID",
    )
    for ext in ("png", "svg"):
        p = os.path.join(out_dir, f"umap_cluster.{ext}")
        fig_cluster.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig_cluster)

    fig_d50 = projection_plot(
        proj_result,
        color_by="d50",
        title=f"{model_name}  k={args.k}  —  UMAP colored by per-cluster d50",
    )
    for ext in ("png", "svg"):
        p = os.path.join(out_dir, f"umap_d50.{ext}")
        fig_d50.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig_d50)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize per-cluster SVD results (spectrum, heatmap, UMAP).")
    parser.add_argument("--vindex", default="gemma3-4b.vindex",
                        help="Vindex directory name (default: gemma3-4b.vindex)")
    parser.add_argument("--k", type=int, default=None,
                        help="K value to show in titles; inferred from JSON if omitted")
    parser.add_argument("--min-cluster-size", type=int, default=30)
    parser.add_argument("--d50-threshold", type=float, default=22,
                        help="d50 threshold for 'passing' (default 22)")
    parser.add_argument("--only", choices=["spectrum", "heatmap", "umap"], default=None,
                        help="Generate only one figure")
    parser.add_argument("--max-clusters-in-heatmap", type=int, default=None,
                        dest="max_clusters_in_heatmap",
                        help="Limit heatmap to top-N passing clusters by d50")
    parser.add_argument("--umap-components", type=int, default=2,
                        dest="umap_components",
                        help="Number of UMAP components (default 2)")
    args = parser.parse_args()

    vindex_name = os.path.basename(args.vindex.rstrip("/"))
    out_dir = os.path.join("outputs", vindex_name)

    dirs, out_ids, in_ids, assignments = load_cache(out_dir, args.vindex)
    data = load_results(out_dir)
    clusters = data["clusters"]

    # Infer k from JSON params if not provided
    if args.k is None:
        args.k = data.get("params", {}).get("k", int(assignments.max()) + 1)

    model_name = vindex_name

    want = args.only
    if want is None or want == "spectrum":
        figure_spectrum(dirs, assignments, clusters, args, out_dir, model_name)
    if want is None or want == "heatmap":
        figure_heatmap(dirs, assignments, clusters, args, out_dir, model_name)
    if want is None or want == "umap":
        figure_umap(dirs, assignments, clusters, args, out_dir, model_name)

    print("\nDone.")


if __name__ == "__main__":
    main()
