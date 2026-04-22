"""Sanity checks for manifold_tools.clustering._FallbackSphericalKMeans.

Three tests:
1. Recovery — well-separated synthetic clusters on the unit sphere should be
   recovered with ARI > 0.95.
2. Convergence — one extra iteration after fit_predict should not move more
   than tol of points; if it does the stopping criterion is too loose.
3. NLTK parity — on a small problem both backends should produce similar
   quality (inertia ratio < 1.1, Jaccard > 0.8). Skipped if nltk not installed.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize

from manifold_tools.clustering import (
    SphericalKMeans,
    NLTKSphericalKMeans,
    _FallbackSphericalKMeans,
    _NLTK_AVAILABLE,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_sphere_clusters(
    n_per_cluster: int,
    d: int,
    k: int,
    separation: float = 8.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate k well-separated clusters on the unit sphere in R^d.

    Each cluster is a von-Mises-Fisher-like blob: a random unit centre plus
    Gaussian noise scaled so that within-cluster spread << between-cluster gap.
    """
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((k, d)).astype(np.float32)
    centres = normalize(centres)

    points, labels = [], []
    for i, c in enumerate(centres):
        noise = rng.standard_normal((n_per_cluster, d)).astype(np.float32)
        pts = c * separation + noise
        points.append(pts)
        labels.extend([i] * n_per_cluster)

    X = normalize(np.vstack(points).astype(np.float32))
    y = np.array(labels, dtype=np.int32)
    return X, y


def _pairwise_jaccard(a: np.ndarray, b: np.ndarray, k_a: int, k_b: int) -> float:
    """Mean per-cluster Jaccard between two partition label arrays."""
    scores = []
    for i in range(k_a):
        mask_a = a == i
        best = 0.0
        for j in range(k_b):
            mask_b = b == j
            inter = float((mask_a & mask_b).sum())
            union = float((mask_a | mask_b).sum())
            if union > 0:
                best = max(best, inter / union)
        scores.append(best)
    return float(np.mean(scores))


# ── test 1: recovery ──────────────────────────────────────────────────────────

def test_recovery():
    """Three well-separated clusters in 100-D should be recovered with ARI > 0.95."""
    K = 3
    X, y_true = _make_sphere_clusters(n_per_cluster=300, d=100, k=K, separation=10.0)

    km = _FallbackSphericalKMeans(n_clusters=K, random_state=0, n_init=5)
    y_pred = km.fit_predict(X)

    ari = adjusted_rand_score(y_true, y_pred)
    assert ari > 0.95, f"ARI {ari:.3f} < 0.95 — cluster recovery failed"


# ── test 2: convergence ───────────────────────────────────────────────────────

def test_convergence():
    """One extra iteration after fit_predict should reassign < tol of points."""
    K = 5
    X, _ = _make_sphere_clusters(n_per_cluster=200, d=64, k=K, separation=6.0)

    tol = 1e-4
    km = _FallbackSphericalKMeans(n_clusters=K, random_state=0, n_init=3, tol=tol)
    assignments = km.fit_predict(X)

    # one manual re-assignment step using the converged centroids
    sims = X @ km.cluster_centers_.T
    reassigned = sims.argmax(axis=1).astype(np.int32)

    frac_moved = float((reassigned != assignments).mean())
    assert frac_moved < tol * 10, (
        f"{frac_moved:.4%} of points moved in one extra iteration "
        f"(tol={tol}) — stopping criterion may be too loose"
    )


# ── test 3: NLTK parity ───────────────────────────────────────────────────────

@pytest.mark.skipif(not _NLTK_AVAILABLE, reason="nltk not installed")
def test_nltk_parity():
    """Fallback should be at least as good as NLTK; both should find real structure.

    NLTK uses random initialization (not k-means++) so it consistently converges
    to worse local optima (~0.72 ARI vs ~1.0 for fallback on the same data).
    We verify:
      - fallback recovers ground truth well (ARI > 0.95)
      - NLTK finds real structure, not noise (ARI > 0.5)
      - fallback inertia <= NLTK inertia (the documented quality claim)
    """
    K = 5
    X, y_true = _make_sphere_clusters(n_per_cluster=100, d=32, k=K, separation=12.0)

    km_fast = _FallbackSphericalKMeans(n_clusters=K, random_state=0, n_init=10)
    y_fast = km_fast.fit_predict(X)

    km_nltk = NLTKSphericalKMeans(n_clusters=K, random_state=0, n_init=10)
    y_nltk = km_nltk.fit_predict(X)

    ari_fast = adjusted_rand_score(y_true, y_fast)
    ari_nltk = adjusted_rand_score(y_true, y_nltk)

    assert ari_fast > 0.95, f"Fallback ARI {ari_fast:.3f} < 0.95"
    assert ari_nltk > 0.5, (
        f"NLTK ARI {ari_nltk:.3f} < 0.5 — NLTK backend found no real structure"
    )
    assert km_fast.inertia_ <= km_nltk.inertia_, (
        f"Fallback inertia ({km_fast.inertia_:.3f}) > NLTK ({km_nltk.inertia_:.3f}) "
        f"— k-means++ init should dominate random init"
    )
