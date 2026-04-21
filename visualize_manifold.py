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
from sklearn.preprocessing import normalize


# ── helpers ───────────────────────────────────────────────────────────────────

def load_cache(out_dir):
    path = os.path.join(out_dir, "offsets_cached.npz")
    if not os.path.exists(path):
        sys.exit(
            f"Cache not found: {path}\n"
            "Run per_cluster_svd.py first to generate it."
        )
    d = np.load(path)
    return d["dirs"], d["out_ids"], d["in_ids"], d["assignments"]


def load_results(out_dir):
    path = os.path.join(out_dir, "per_cluster_svd.json")
    if not os.path.exists(path):
        sys.exit(f"Results not found: {path}")
    with open(path) as f:
        return json.load(f)


def cluster_label(r):
    """Short label from top (in, out) pair."""
    if r["top_pairs"]:
        p = r["top_pairs"][0]
        label = f"{p['in']}→{p['out']}"
        return label[:20]
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

    is_3d = emb.shape[1] >= 3
    if is_3d:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    d50_thresh = args.d50_threshold
    min_size = args.min_cluster_size

    # Build per-point d50 and passing mask
    cid_to_d50 = {r["cluster"]: r["d50"] for r in clusters}
    cid_to_n   = {r["cluster"]: r["n"]   for r in clusters}
    point_d50  = np.array([cid_to_d50.get(int(a), 999) for a in assignments], dtype=float)
    point_pass = np.array(
        [(cid_to_d50.get(int(a), 999) <= d50_thresh and
          cid_to_n.get(int(a), 0) >= min_size)
         for a in assignments], dtype=bool
    )

    k = int(assignments.max()) + 1
    if k <= 20:
        cmap_ids = plt.get_cmap("tab20")
        cluster_colors = np.array([cmap_ids(i / 20) for i in range(k)])
    else:
        cmap_ids = plt.get_cmap("gist_ncar")
        cluster_colors = np.array([cmap_ids(i / k) for i in range(k)])

    subplot_kw = {"projection": "3d"} if is_3d else {}
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 7),
                                      subplot_kw=subplot_kw,
                                      **({} if is_3d else {"sharex": True, "sharey": True}))
    alpha = 0.4
    s = max(1.5, min(8, 30000 / len(dirs)))

    def scatter(ax, mask, c, **kw):
        if is_3d:
            ax.scatter(emb[mask, 0], emb[mask, 1], emb[mask, 2], c=c, **kw)
        else:
            ax.scatter(emb[mask, 0], emb[mask, 1], c=c, **kw)

    def set_labels(ax, title):
        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        if is_3d:
            ax.set_zlabel("UMAP-3")

    # Left: colored by cluster ID
    colors_left = cluster_colors[assignments.astype(int)]
    scatter(ax_l, slice(None), colors_left, s=s, alpha=alpha, linewidths=0)
    set_labels(ax_l, "Colored by cluster ID (legend suppressed)")

    # Right: gray for failing, viridis gradient for passing by d50
    passing_d50s = point_d50[point_pass]
    d50_min = passing_d50s.min() if point_pass.any() else 0
    d50_max = passing_d50s.max() if point_pass.any() else 1
    d50_norm = mcolors.Normalize(vmin=d50_min, vmax=max(d50_max, d50_min + 1))

    scatter(ax_r, ~point_pass, "lightgray", s=s, alpha=alpha * 0.6, linewidths=0)
    if point_pass.any():
        if is_3d:
            sc = ax_r.scatter(emb[point_pass, 0], emb[point_pass, 1], emb[point_pass, 2],
                               c=passing_d50s, cmap="viridis", norm=d50_norm,
                               s=s * 1.5, alpha=alpha, linewidths=0)
        else:
            sc = ax_r.scatter(emb[point_pass, 0], emb[point_pass, 1],
                               c=passing_d50s, cmap="viridis", norm=d50_norm,
                               s=s * 1.5, alpha=alpha, linewidths=0)
        fig.colorbar(sc, ax=ax_r, label="d50", shrink=0.6 if is_3d else 1.0)
    set_labels(ax_r, f"Colored by d50  (passing ≤ {d50_thresh}: viridis, failing: gray)")

    # Annotate passing cluster centroids (2D only — 3D text is too cluttered)
    if not is_3d:
        passing_clusters = [r for r in clusters
                            if r["d50"] <= d50_thresh and r["n"] >= min_size and r["top_pairs"]]
        passing_clusters.sort(key=lambda r: r["d50"])
        annotate_n = min(10, len(passing_clusters))
        step = max(1, len(passing_clusters) // annotate_n)
        for r in passing_clusters[::step][:annotate_n]:
            mask = assignments == r["cluster"]
            if mask.sum() == 0:
                continue
            cx, cy = emb[mask, 0].mean(), emb[mask, 1].mean()
            label = cluster_label(r)
            for ax in (ax_l, ax_r):
                ax.annotate(label, (cx, cy), fontsize=6,
                            ha="center", va="bottom",
                            bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.6, ec="none"),
                            xytext=(0, 3), textcoords="offset points")

    ndim = emb.shape[1]
    fig.suptitle(
        f"{model_name}  k={args.k}  —  UMAP ({ndim}D) of filtered offset vectors\n"
        f"Note: UMAP is a nonlinear projection; distances in {ndim}D don't directly "
        f"reflect distances in {dirs.shape[1]}D.",
        fontsize=9
    )
    fig.tight_layout()

    for ext in ("png", "svg"):
        p = os.path.join(out_dir, f"umap.{ext}")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


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

    dirs, out_ids, in_ids, assignments = load_cache(out_dir)
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
