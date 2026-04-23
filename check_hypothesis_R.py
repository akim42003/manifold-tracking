#!/usr/bin/env python3
"""
Empirical check of hypothesis (R) from the invariance_note.

Hypothesis (R) (Remark 4.8):
    The empirical CDF F_N of cluster margins satisfies
        F_N(tau) <= L * tau   for tau in [0, tau_0]
    for some constants L, tau_0.

For each feature f with offset x_f and its assigned cluster index k*(f),
the margin is
    gamma_f = <x_f, c_{k*}> - <x_f, c_{k**}>
where c_{k**} is the second-closest centroid.

This script:
    1. Loads offsets_cached.npz produced by per_cluster_svd.py
    2. Reconstructs centroids as normalized means of assigned offsets
    3. Computes per-feature margins
    4. Plots the empirical CDF F_N(tau) and its behavior near tau=0
    5. Fits a linear bound F_N(tau) <= L*tau on [0, tau_0] and reports L

Run:
    python check_hypothesis_R.py outputs/gemma3-4b.vindex/offsets_cached.npz
    python check_hypothesis_R.py outputs/gemma3-4b.vindex/offsets_cached.npz --k 256
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def reconstruct_centroids(dirs: np.ndarray, assignments: np.ndarray, K: int) -> np.ndarray:
    """Normalized mean of offsets per cluster.
    dirs: (N, d), unit vectors. assignments: (N,), int in [0, K).
    Returns: (K, d) with unit-norm rows. Empty clusters get zero rows.
    """
    d = dirs.shape[1]
    centroids = np.zeros((K, d), dtype=dirs.dtype)
    for k in range(K):
        mask = assignments == k
        if mask.any():
            mean = dirs[mask].mean(axis=0)
            norm = np.linalg.norm(mean)
            if norm > 1e-12:
                centroids[k] = mean / norm
    return centroids


def compute_margins(dirs: np.ndarray, centroids: np.ndarray, assignments: np.ndarray) -> np.ndarray:
    """Per-feature margin: assigned cosine minus second-closest cosine.
    dirs: (N, d). centroids: (K, d). assignments: (N,).
    Returns: (N,) with values in [0, 2].
    """
    sims = dirs @ centroids.T  # (N, K)
    N = dirs.shape[0]
    row_idx = np.arange(N)
    assigned_sim = sims[row_idx, assignments]
    sims_masked = sims.copy()
    sims_masked[row_idx, assignments] = -np.inf  # mask assigned
    second_sim = sims_masked.max(axis=1)
    return assigned_sim - second_sim


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_path", help="Path to offsets_cached.npz")
    ap.add_argument("--k", type=int, default=None,
                    help="Number of clusters (inferred from assignments if not given)")
    ap.add_argument("--tau-max", type=float, default=0.3,
                    help="Upper bound on tau for the near-zero diagnostic")
    ap.add_argument("--fit-up-to", type=float, default=0.1,
                    help="Fit linear bound F_N(tau) <= L*tau on [0, fit_up_to]")
    ap.add_argument("--out", default="hypothesis_R_check",
                    help="Output prefix (writes .png and .json)")
    args = ap.parse_args()

    print(f"Loading {args.npz_path} ...")
    data = np.load(args.npz_path)
    dirs = data["dirs"].astype(np.float32)
    assignments = data["assignments"].astype(np.int64)
    N, d = dirs.shape
    K = int(args.k or (assignments.max() + 1))
    print(f"  {N} features, dim {d}, {K} clusters")

    # Ensure unit norm (should already be)
    norms = np.linalg.norm(dirs, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        print(f"  re-normalizing dirs (min/max norm before: {norms.min():.4f}/{norms.max():.4f})")
        dirs = dirs / norms[:, None]

    print("Reconstructing centroids ...")
    centroids = reconstruct_centroids(dirs, assignments, K)
    empty = (centroids.sum(axis=1) == 0).sum()
    if empty:
        print(f"  {empty} empty clusters (skipped in margin computation)")

    print("Computing margins ...")
    margins = compute_margins(dirs, centroids, assignments)
    assert margins.min() >= -1e-4, f"negative margin: {margins.min()}"
    margins = np.clip(margins, 0.0, None)

    # Basic stats
    print()
    print("Margin distribution statistics:")
    pct = np.percentile(margins, [1, 5, 25, 50, 75, 95, 99])
    print(f"  percentiles 1/5/25/50/75/95/99:")
    for p, val in zip([1, 5, 25, 50, 75, 95, 99], pct):
        print(f"    p{p}: {val:.4f}")
    print(f"  mean: {margins.mean():.4f}    std: {margins.std():.4f}")
    print(f"  fraction < 0.01: {(margins < 0.01).mean():.4f}")
    print(f"  fraction < 0.05: {(margins < 0.05).mean():.4f}")
    print(f"  fraction < 0.10: {(margins < 0.10).mean():.4f}")

    # Fit linear bound F_N(tau) <= L * tau on [0, fit_up_to]
    sorted_margins = np.sort(margins)
    F_N = np.arange(1, N + 1) / N

    fit_mask = sorted_margins <= args.fit_up_to
    if fit_mask.sum() < 10:
        print("\n(Warning: too few points below fit-up-to; linear fit unreliable.)")
        L_fit = None
        worst_L = None
    else:
        # L_fit: least-squares slope (through origin) on the fit region
        taus = sorted_margins[fit_mask]
        Fs = F_N[fit_mask]
        L_fit = float((taus * Fs).sum() / (taus * taus).sum())
        # worst-case L: max over tau in fit region of F_N(tau) / tau
        # (this is the smallest L making F_N(tau) <= L*tau hold on that range)
        positive = taus > 1e-6
        if positive.any():
            worst_L = float((Fs[positive] / taus[positive]).max())
        else:
            worst_L = None

    print()
    print(f"Linear-bound fit on [0, {args.fit_up_to}]:")
    print(f"  least-squares slope L_fit: {L_fit}")
    print(f"  worst-case L (smallest L making F_N(tau) <= L*tau hold): {worst_L}")
    if worst_L is not None:
        print(f"  Interpretation: F_N(tau) <= {worst_L:.2f} * tau for tau in [0, {args.fit_up_to}]")

    # Plot empirical CDF with linear-bound overlay
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Near-zero region
    ax = axes[0]
    ax.plot(sorted_margins, F_N, lw=1.5, color="#1D9E75", label="$F_N(\\tau)$")
    if worst_L is not None:
        tau_grid = np.linspace(0, args.fit_up_to, 100)
        ax.plot(tau_grid, worst_L * tau_grid, "--", color="#AA3333", lw=1.2,
                label=f"$L \\cdot \\tau$, $L={worst_L:.2f}$")
    ax.axvline(args.fit_up_to, color="gray", ls=":", alpha=0.6,
               label=f"$\\tau_0={args.fit_up_to}$")
    ax.set_xlim(0, args.tau_max)
    ax.set_ylim(0, F_N[sorted_margins <= args.tau_max][-1] if (sorted_margins <= args.tau_max).any() else 1)
    ax.set_xlabel("margin $\\tau$")
    ax.set_ylabel("empirical CDF $F_N(\\tau)$")
    ax.set_title(f"Margin CDF near zero (N={N}, K={K})")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Full CDF
    ax = axes[1]
    ax.plot(sorted_margins, F_N, lw=1.5, color="#1D9E75")
    ax.set_xlabel("margin $\\tau$")
    ax.set_ylabel("empirical CDF $F_N(\\tau)$")
    ax.set_title("Full margin CDF")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(sorted_margins.max(), 1.0))

    plt.tight_layout()
    out_png = f"{args.out}.png"
    plt.savefig(out_png, dpi=150)
    print(f"\nSaved plot: {out_png}")

    # Also dump numeric results
    result = {
        "N": int(N),
        "d": int(d),
        "K": int(K),
        "n_empty_clusters": int(empty),
        "margin_percentiles": {f"p{p}": float(v) for p, v in zip([1, 5, 25, 50, 75, 95, 99], pct)},
        "margin_mean": float(margins.mean()),
        "margin_std": float(margins.std()),
        "fraction_below": {"0.01": float((margins < 0.01).mean()),
                           "0.05": float((margins < 0.05).mean()),
                           "0.10": float((margins < 0.10).mean())},
        "fit_up_to": args.fit_up_to,
        "L_fit_least_squares": L_fit,
        "L_worst_case": worst_L,
    }
    out_json = f"{args.out}.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved numeric summary: {out_json}")

    # Verdict
    print()
    print("=" * 64)
    print("VERDICT")
    print("=" * 64)
    if worst_L is None:
        print("  Insufficient data near zero; cannot assess (R).")
    elif worst_L < 5:
        print(f"  (R) holds with modest constant L = {worst_L:.2f} on [0, {args.fit_up_to}].")
        print("  Linear flip-rate bound of Remark 4.8 applies; the quadratic")
        print("  centroid-stability claim of Proposition 4.10 is non-trivial.")
    elif worst_L < 20:
        print(f"  (R) holds with moderate L = {worst_L:.2f} on [0, {args.fit_up_to}].")
        print("  Linear flip-rate bound applies but with non-negligible slope.")
        print("  The quadratic centroid claim holds for epsilon small relative to 1/L.")
    else:
        print(f"  (R) fails or holds only with large L = {worst_L:.2f}.")
        print("  A substantial fraction of features sits near decision boundaries.")
        print("  The linear flip-rate conclusion is not useful at practical epsilon.")


if __name__ == "__main__":
    main()
