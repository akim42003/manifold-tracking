#!/usr/bin/env python3
"""Compare Euclidean vs. spherical k-means clustering outputs.

Loads per_cluster_svd.json (euclidean) and per_cluster_svd_spherical.json
from the same vindex output directory and produces:
  - Side-by-side summary statistics
  - Cluster size distribution histograms
  - Jaccard overlap between the two partitions
  - Content comparison on specific known clusters

Usage:
    python compare_clustering_methods.py --vindex gemma3-4b.vindex
"""

import argparse
import json
import os
import sys

import numpy as np
from collections import Counter


# ── Loading ───────────────────────────────────────────────────────────────────

def load_output(out_dir, suffix=""):
    fname = f"per_cluster_svd{suffix}.json"
    path = os.path.join(out_dir, fname)
    if not os.path.exists(path):
        return None, path
    with open(path) as f:
        return json.load(f), path


def load_assignments(out_dir, suffix=""):
    fname = f"offsets_cached{suffix}.npz"
    path = os.path.join(out_dir, fname)
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    return data["assignments"]


# ── Statistics ────────────────────────────────────────────────────────────────

def summary_stats(data):
    s = data.get("summary") or {}
    clusters = data.get("clusters", [])
    d50s = np.array([r["d50"] for r in clusters]) if clusters else np.array([])
    return {
        "pass_rate": s.get("pct_in_target", float("nan")),
        "passes": s.get("passes", None),
        "n_analyzed": s.get("n_clusters_analyzed", len(clusters)),
        "n_skipped": s.get("n_skipped", 0),
        "d50_median": float(np.median(d50s)) if len(d50s) else float("nan"),
        "d50_range": [int(d50s.min()), int(d50s.max())] if len(d50s) else [0, 0],
        "n_passing": s.get("n_passing", int(((d50s >= 5) & (d50s <= 22)).sum())),
    }


def size_histogram(assignments):
    counts = np.bincount(assignments)
    bins = [
        ("n<10",    (counts < 10).sum()),
        ("10-29",   ((counts >= 10) & (counts < 30)).sum()),
        ("30-99",   ((counts >= 30) & (counts < 100)).sum()),
        ("100-499", ((counts >= 100) & (counts < 500)).sum()),
        ("≥500",    (counts >= 500).sum()),
    ]
    return bins, counts


# ── Jaccard ───────────────────────────────────────────────────────────────────

def best_jaccard_per_cluster(assign_a, assign_b):
    """For each cluster in A, return its best Jaccard score against any cluster in B."""
    k_a = int(assign_a.max()) + 1
    k_b = int(assign_b.max()) + 1
    scores = []
    for a in range(k_a):
        mask_a = assign_a == a
        if not mask_a.any():
            continue
        best = 0.0
        for b in range(k_b):
            mask_b = assign_b == b
            inter = (mask_a & mask_b).sum()
            if inter == 0:
                continue
            union = (mask_a | mask_b).sum()
            j = inter / max(union, 1)
            if j > best:
                best = j
        scores.append(best)
    return np.array(scores)


# ── Content comparison ────────────────────────────────────────────────────────

def find_cluster_by_pair(clusters, in_tok, out_tok):
    """Return cluster id whose top pairs contain (in_tok, out_tok)."""
    for r in clusters:
        for p in r.get("top_pairs", []):
            if p["in"] == in_tok and p["out"] == out_tok:
                return r["cluster"]
    return None


def find_cluster_for_label(clusters, label_fragment):
    """Return the first cluster whose label contains label_fragment."""
    for r in clusters:
        if label_fragment.lower() in r.get("label", "").lower():
            return r
    return None


def content_comparison(eu_clusters, sph_clusters, assign_eu, assign_sph):
    """Compare specific known clusters between the two methods."""
    # Probe pairs that should form tight clusters in Gemma
    probes = [
        ("yourself", "you"),
        ("Japanese", "Japan"),
        ("French", "France"),
        ("German", "Germany"),
        ("swimming", "swim"),
    ]
    lines = []
    for in_tok, out_tok in probes:
        eu_cid = find_cluster_by_pair(eu_clusters, in_tok, out_tok)
        sph_cid = find_cluster_by_pair(sph_clusters, in_tok, out_tok)

        if eu_cid is None and sph_cid is None:
            lines.append(f"  {in_tok}→{out_tok}: not in top pairs of any cluster (both methods)")
            continue

        eu_label = next((r["label"] for r in eu_clusters if r["cluster"] == eu_cid), "?") if eu_cid is not None else "absent"
        sph_label = next((r["label"] for r in sph_clusters if r["cluster"] == sph_cid), "?") if sph_cid is not None else "absent"

        if eu_cid is not None and sph_cid is not None and assign_eu is not None and assign_sph is not None:
            mask_eu = assign_eu == eu_cid
            mask_sph = assign_sph == sph_cid
            inter = (mask_eu & mask_sph).sum()
            union = (mask_eu | mask_sph).sum()
            jaccard = inter / max(union, 1)
            lines.append(
                f"  {in_tok}→{out_tok}:  "
                f"eu=c{eu_cid}({eu_label})  "
                f"sph=c{sph_cid}({sph_label})  "
                f"jaccard={jaccard:.3f}"
            )
        else:
            lines.append(
                f"  {in_tok}→{out_tok}:  "
                f"eu={'c'+str(eu_cid)+'('+eu_label+')' if eu_cid is not None else 'absent'}  "
                f"sph={'c'+str(sph_cid)+'('+sph_label+')' if sph_cid is not None else 'absent'}"
            )
    return lines


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare euclidean vs spherical clustering.")
    parser.add_argument("--vindex", default="gemma3-4b.vindex")
    args = parser.parse_args()

    out_dir = f"outputs/{os.path.basename(args.vindex.rstrip('/'))}"

    eu_data, eu_path = load_output(out_dir, suffix="")
    sph_data, sph_path = load_output(out_dir, suffix="_spherical")

    if eu_data is None:
        sys.exit(f"Euclidean output not found: {eu_path}\nRun per_cluster_svd.py --clustering euclidean first.")
    if sph_data is None:
        sys.exit(f"Spherical output not found: {sph_path}\nRun per_cluster_svd.py --clustering spherical first.")

    print(f"Loaded: {eu_path}")
    print(f"Loaded: {sph_path}")

    assign_eu = load_assignments(out_dir, suffix="")
    assign_sph = load_assignments(out_dir, suffix="_spherical")

    eu_s = summary_stats(eu_data)
    sph_s = summary_stats(sph_data)
    eu_clusters = eu_data.get("clusters", [])
    sph_clusters = sph_data.get("clusters", [])

    # ── 1. Summary stats ──────────────────────────────────────────────────────
    print("\n── Summary statistics ──────────────────────────────────────────────")
    print(f"  {'Metric':<28} {'Euclidean':>12} {'Spherical':>12}")
    print(f"  {'─'*54}")
    rows = [
        ("Pass rate (% d50 in 5-22D)", f"{eu_s['pass_rate']:.1f}%", f"{sph_s['pass_rate']:.1f}%"),
        ("Passes criterion",          str(eu_s['passes']),          str(sph_s['passes'])),
        ("Clusters analyzed",         str(eu_s['n_analyzed']),      str(sph_s['n_analyzed'])),
        ("Clusters skipped",          str(eu_s['n_skipped']),       str(sph_s['n_skipped'])),
        ("d50 median",                f"{eu_s['d50_median']:.1f}",  f"{sph_s['d50_median']:.1f}"),
        ("d50 range",                 f"{eu_s['d50_range'][0]}-{eu_s['d50_range'][1]}",
                                      f"{sph_s['d50_range'][0]}-{sph_s['d50_range'][1]}"),
        ("N clusters passing",        str(eu_s['n_passing']),       str(sph_s['n_passing'])),
    ]
    for label, eu_val, sph_val in rows:
        print(f"  {label:<28} {eu_val:>12} {sph_val:>12}")

    # ── 2. Cluster size distributions ────────────────────────────────────────
    if assign_eu is not None and assign_sph is not None:
        eu_bins, eu_counts = size_histogram(assign_eu)
        sph_bins, sph_counts = size_histogram(assign_sph)

        print("\n── Cluster size distributions ──────────────────────────────────────")
        print(f"  {'Bin':<12} {'Euclidean':>12} {'Spherical':>12}")
        print(f"  {'─'*38}")
        for (label, eu_n), (_, sph_n) in zip(eu_bins, sph_bins):
            print(f"  {label:<12} {eu_n:>12} {sph_n:>12}")
        print(f"  {'min size':<12} {eu_counts.min():>12} {sph_counts.min():>12}")
        print(f"  {'max size':<12} {eu_counts.max():>12} {sph_counts.max():>12}")
        print(f"  {'mean size':<12} {eu_counts.mean():>12.1f} {sph_counts.mean():>12.1f}")

        # ── 3. Jaccard overlap ────────────────────────────────────────────────
        print("\n── Jaccard overlap (euclidean→spherical) ───────────────────────────")
        print("  (For each Euclidean cluster, best Jaccard against any spherical cluster)")
        j_scores = best_jaccard_per_cluster(assign_eu, assign_sph)
        thresholds = [0.8, 0.5, 0.3, 0.1]
        for t in thresholds:
            n = (j_scores >= t).sum()
            pct = n / len(j_scores) * 100
            print(f"  Jaccard ≥ {t:.1f}: {n}/{len(j_scores)} ({pct:.0f}%)")
        print(f"  Median Jaccard: {np.median(j_scores):.3f}")
        print(f"  Mean   Jaccard: {np.mean(j_scores):.3f}")

        # ── 4. Content comparison ─────────────────────────────────────────────
        print("\n── Content comparison (known probe pairs) ──────────────────────────")
        lines = content_comparison(eu_clusters, sph_clusters, assign_eu, assign_sph)
        for l in lines:
            print(l)

        # ── Save JSON ─────────────────────────────────────────────────────────
        comparison = {
            "vindex": args.vindex,
            "euclidean": eu_s,
            "spherical": sph_s,
            "size_distribution": {
                "euclidean": {label: int(n) for label, n in eu_bins},
                "spherical":  {label: int(n) for label, n in sph_bins},
            },
            "jaccard": {
                "median": float(np.median(j_scores)),
                "mean":   float(np.mean(j_scores)),
                "n_ge_0.8": int((j_scores >= 0.8).sum()),
                "n_ge_0.5": int((j_scores >= 0.5).sum()),
                "n_ge_0.3": int((j_scores >= 0.3).sum()),
                "n_total":  int(len(j_scores)),
            },
        }
        comp_path = os.path.join(out_dir, "clustering_comparison.json")
        with open(comp_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nSaved: {comp_path}")
    else:
        print("\nNote: offsets_cached*.npz files not found — skipping size distribution, "
              "Jaccard, and content comparison. Run per_cluster_svd.py for both methods "
              "to generate cache files.")


if __name__ == "__main__":
    main()
