#!/usr/bin/env python3
"""
Empirical check of local flip uniformity (assumption A4 of Proposition 4.10).

Assumption A4 says:
    For each cluster k,
        |S_k(theta) triangle S_k(theta')| <= C_loc * p_total * n_k
    where p_total is the global flip fraction, and C_loc is a constant
    of order 1 that characterizes how uniformly flips are spread across clusters.

Procedure:
    1. Load offsets_cached.npz and reconstruct spherical-k-means centroids
       from (dirs, assignments).
    2. Apply iid Gaussian noise of scale sigma to each offset, renormalize
       to the sphere.
    3. Reassign each perturbed offset via argmax cosine against the
       (unperturbed) centroids.
    4. For each cluster k, compute the local flip rate
           p_k = |S_k triangle S_k'| / n_k.
    5. Report p_total, distribution of p_k, and C_loc_emp = max_k p_k / p_total.
    6. Sweep over sigma values to see how C_loc behaves with perturbation scale.

Run:
    python check_local_flip_uniformity.py outputs/gemma3-4b.vindex/offsets_cached_spherical_nltk.npz
    python check_local_flip_uniformity.py ... --sigmas 0.001 0.003 0.01 0.03
    python check_local_flip_uniformity.py ... --seed 0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def reconstruct_centroids(dirs: np.ndarray, assignments: np.ndarray, K: int) -> np.ndarray:
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


def perturb_and_reassign(dirs: np.ndarray, centroids: np.ndarray, sigma: float,
                         rng: np.random.Generator) -> np.ndarray:
    """Add iid Gaussian noise of scale sigma, renormalize, reassign by argmax cosine."""
    noise = rng.normal(scale=sigma, size=dirs.shape).astype(dirs.dtype)
    perturbed = dirs + noise
    norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    perturbed = perturbed / norms
    sims = perturbed @ centroids.T
    return sims.argmax(axis=1)


def local_flip_stats(old_assign: np.ndarray, new_assign: np.ndarray, K: int) -> dict:
    """Compute per-cluster flip statistics and global statistics."""
    N = len(old_assign)
    p_total = float((old_assign != new_assign).mean())

    # For each cluster k, |S_k(old) triangle S_k(new)|
    # = |features that WERE in k but aren't now| + |features that ARE in k but weren't|
    p_locals = np.zeros(K, dtype=np.float64)
    n_ks = np.zeros(K, dtype=np.int64)
    for k in range(K):
        was_k = (old_assign == k)
        is_k = (new_assign == k)
        symdiff = np.logical_xor(was_k, is_k).sum()
        n_k = was_k.sum()
        n_ks[k] = n_k
        if n_k > 0:
            p_locals[k] = symdiff / n_k

    valid = n_ks > 0
    p_locals_valid = p_locals[valid]
    n_ks_valid = n_ks[valid]

    return {
        "p_total": p_total,
        "p_locals": p_locals,
        "n_ks": n_ks,
        "p_local_mean": float(p_locals_valid.mean()) if valid.any() else 0.0,
        "p_local_median": float(np.median(p_locals_valid)) if valid.any() else 0.0,
        "p_local_max": float(p_locals_valid.max()) if valid.any() else 0.0,
        "p_local_p95": float(np.percentile(p_locals_valid, 95)) if valid.any() else 0.0,
        "p_local_p99": float(np.percentile(p_locals_valid, 99)) if valid.any() else 0.0,
        "C_loc_max": float(p_locals_valid.max() / p_total) if p_total > 0 else 0.0,
        "C_loc_p95": float(np.percentile(p_locals_valid, 95) / p_total) if p_total > 0 else 0.0,
        "C_loc_p99": float(np.percentile(p_locals_valid, 99) / p_total) if p_total > 0 else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_path", help="Path to offsets_cached.npz")
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.001, 0.003, 0.01, 0.03, 0.1],
                    help="Noise scales to test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-trials", type=int, default=3,
                    help="Number of independent noise realizations per sigma")
    ap.add_argument("--out", default="local_flip_uniformity",
                    help="Output prefix (writes .png and .json)")
    args = ap.parse_args()

    print(f"Loading {args.npz_path} ...")
    data = np.load(args.npz_path)
    dirs = data["dirs"].astype(np.float32)
    assignments = data["assignments"].astype(np.int64)
    N, d = dirs.shape
    K = int(assignments.max() + 1)
    print(f"  {N} features, dim {d}, {K} clusters")

    norms = np.linalg.norm(dirs, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        dirs = dirs / norms[:, None]

    print("Reconstructing centroids ...")
    centroids = reconstruct_centroids(dirs, assignments, K)

    # Verify reconstructed centroids are consistent with stored assignments
    sims = dirs @ centroids.T
    recon_assign = sims.argmax(axis=1)
    consistency = float((recon_assign == assignments).mean())
    print(f"  reconstructed-vs-stored assignment consistency: {consistency:.4f}")
    if consistency < 0.99:
        print(f"  (Note: {(1-consistency)*100:.1f}% of features have stored assignment")
        print(f"   that differs from argmax cosine vs reconstructed centroids.")
        print(f"   This can happen when clustering hasn't fully converged or used")
        print(f"   a different centroid update rule. Using stored assignments as reference.)")

    rng = np.random.default_rng(args.seed)

    # Sweep over sigma values
    results_by_sigma = {}
    for sigma in args.sigmas:
        per_trial = []
        for trial in range(args.n_trials):
            new_assign = perturb_and_reassign(dirs, centroids, sigma, rng)
            stats = local_flip_stats(assignments, new_assign, K)
            per_trial.append(stats)

        # Aggregate across trials
        agg = {
            "sigma": sigma,
            "p_total_mean": float(np.mean([s["p_total"] for s in per_trial])),
            "p_total_std": float(np.std([s["p_total"] for s in per_trial])),
            "p_local_max_mean": float(np.mean([s["p_local_max"] for s in per_trial])),
            "p_local_p99_mean": float(np.mean([s["p_local_p99"] for s in per_trial])),
            "p_local_p95_mean": float(np.mean([s["p_local_p95"] for s in per_trial])),
            "p_local_median_mean": float(np.mean([s["p_local_median"] for s in per_trial])),
            "C_loc_max_mean": float(np.mean([s["C_loc_max"] for s in per_trial])),
            "C_loc_max_max": float(np.max([s["C_loc_max"] for s in per_trial])),
            "C_loc_p95_mean": float(np.mean([s["C_loc_p95"] for s in per_trial])),
            "C_loc_p99_mean": float(np.mean([s["C_loc_p99"] for s in per_trial])),
            "p_locals_last_trial": per_trial[-1]["p_locals"].tolist(),
            "n_ks": per_trial[-1]["n_ks"].tolist(),
        }
        results_by_sigma[sigma] = agg

        print(f"\nsigma = {sigma}:")
        print(f"  p_total (global flip fraction):        {agg['p_total_mean']:.4f} +/- {agg['p_total_std']:.4f}")
        print(f"  p_local median:                        {agg['p_local_median_mean']:.4f}")
        print(f"  p_local p95:                           {agg['p_local_p95_mean']:.4f}")
        print(f"  p_local p99:                           {agg['p_local_p99_mean']:.4f}")
        print(f"  p_local max:                           {agg['p_local_max_mean']:.4f}")
        print(f"  C_loc (= max p_local / p_total) mean:  {agg['C_loc_max_mean']:.2f}")
        print(f"  C_loc (= max p_local / p_total) max:   {agg['C_loc_max_max']:.2f}")
        print(f"  C_loc (p95 / p_total):                 {agg['C_loc_p95_mean']:.2f}")
        print(f"  C_loc (p99 / p_total):                 {agg['C_loc_p99_mean']:.2f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: C_loc vs sigma
    ax = axes[0]
    sigmas = np.array(args.sigmas)
    C_loc_max = np.array([results_by_sigma[s]["C_loc_max_mean"] for s in args.sigmas])
    C_loc_p95 = np.array([results_by_sigma[s]["C_loc_p95_mean"] for s in args.sigmas])
    C_loc_p99 = np.array([results_by_sigma[s]["C_loc_p99_mean"] for s in args.sigmas])
    ax.semilogx(sigmas, C_loc_max, "o-", color="#AA3333", label="$C_{\\mathrm{loc}}$ (max)", lw=1.5, ms=7)
    ax.semilogx(sigmas, C_loc_p99, "s-", color="#DD8833", label="$C_{\\mathrm{loc}}$ (p99)", lw=1.5, ms=6)
    ax.semilogx(sigmas, C_loc_p95, "^-", color="#1D9E75", label="$C_{\\mathrm{loc}}$ (p95)", lw=1.5, ms=6)
    ax.axhline(1, color="gray", ls=":", alpha=0.6, label="uniformity ($C_{\\mathrm{loc}} = 1$)")
    ax.set_xlabel("perturbation scale $\\sigma$")
    ax.set_ylabel("$C_{\\mathrm{loc}} = p_k / p_{\\mathrm{total}}$")
    ax.set_title("Local flip uniformity constant vs perturbation scale")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Right: p_local distribution at a middle sigma value
    middle_sigma = args.sigmas[len(args.sigmas) // 2]
    ax = axes[1]
    p_locals = np.array(results_by_sigma[middle_sigma]["p_locals_last_trial"])
    n_ks = np.array(results_by_sigma[middle_sigma]["n_ks"])
    valid = n_ks > 0
    p_total_mid = results_by_sigma[middle_sigma]["p_total_mean"]
    ax.hist(p_locals[valid], bins=40, color="#1D9E75", alpha=0.85, edgecolor="white")
    ax.axvline(p_total_mid, color="#AA3333", ls="--", lw=1.5,
               label=f"$p_{{\\mathrm{{total}}}}={p_total_mid:.3f}$")
    ax.set_xlabel(f"local flip fraction $p_k$")
    ax.set_ylabel("cluster count")
    ax.set_title(f"Distribution of $p_k$ across clusters ($\\sigma={middle_sigma}$)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{args.out}.png", dpi=150)
    print(f"\nSaved plot: {args.out}.png")

    with open(f"{args.out}.json", "w") as f:
        json.dump({
            "N": int(N), "d": int(d), "K": int(K),
            "consistency": consistency,
            "results_by_sigma": {str(s): v for s, v in results_by_sigma.items()},
        }, f, indent=2)
    print(f"Saved numeric summary: {args.out}.json")

    # Verdict
    C_loc_estimates = [results_by_sigma[s]["C_loc_max_mean"] for s in args.sigmas]
    typical_C_loc = float(np.median(C_loc_estimates))
    worst_C_loc = float(max(C_loc_estimates))
    print()
    print("=" * 64)
    print("VERDICT")
    print("=" * 64)
    print(f"  Across sigma values tested: C_loc ranges from "
          f"{min(C_loc_estimates):.1f} to {worst_C_loc:.1f}")
    print(f"  Typical C_loc (median across sigma): {typical_C_loc:.1f}")
    if worst_C_loc < 5:
        print("  Assumption (A4) holds with a small constant. Local flips distribute")
        print("  roughly proportionally to cluster size.")
    elif worst_C_loc < 20:
        print("  Assumption (A4) holds but with a non-trivial constant.")
        print("  Some clusters see substantially more flip churn than average.")
    else:
        print("  Assumption (A4) is violated or holds only with very large C_loc.")
        print("  Flips concentrate on specific clusters; Proposition 4.10's per-cluster")
        print("  bound must be stated with a cluster-dependent constant.")


if __name__ == "__main__":
    main()
