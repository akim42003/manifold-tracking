"""Clustering algorithms tailored to directional (unit-normalized) data.

Euclidean k-means (sklearn's default) treats centroids as arithmetic means,
which drift off the unit sphere over iterations. For offset vectors and other
normalized data, spherical k-means — which uses cosine distance — is the
geometrically correct algorithm.

Default backend: _FallbackSphericalKMeans (vectorized BLAS, k-means++ init).
Optional backend: NLTKSphericalKMeans (pass use_nltk=True to SphericalKMeans);
  available for comparison but uses random init and converges to worse optima.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

try:
    from nltk.cluster import KMeansClusterer
    from nltk.cluster.util import cosine_distance
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False


class NLTKSphericalKMeans:
    """Sklearn-like wrapper around NLTK's cosine k-means."""

    def __init__(
        self,
        n_clusters: int,
        random_state: Optional[int] = None,
        n_init: int = 10,
        max_iter: int = 300,
    ):
        import random as _random
        self.n_clusters = n_clusters
        self._clusterer = KMeansClusterer(
            num_means=n_clusters,
            distance=cosine_distance,
            repeats=n_init,
            avoid_empty_clusters=True,
            rng=_random.Random(random_state) if random_state is not None else None,
        )

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float64)
        assignments = self._clusterer.cluster(list(X), assign_clusters=True)
        centers = np.array(self._clusterer.means(), dtype=np.float64)
        # NLTK returns un-normalized means; normalize before computing cosine inertia
        norms = np.linalg.norm(centers, axis=1, keepdims=True)
        centers = centers / np.where(norms > 1e-12, norms, 1.0)
        self.cluster_centers_ = centers.astype(np.float32)
        sims = X @ centers.T
        max_sims = sims[np.arange(len(X)), np.array(assignments)]
        self.inertia_ = float((1.0 - max_sims).sum())
        self.n_iter_ = None  # NLTK doesn't expose iteration count
        return np.array(assignments, dtype=np.int32)


class _FallbackSphericalKMeans:
    """Hand-rolled spherical k-means. Used when NLTK is unavailable."""

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 300,
        n_init: int = 10,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = random_state

    def _init_centroids(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = len(X)
        centroids = np.empty((self.n_clusters, X.shape[1]), dtype=np.float32)
        idx = rng.integers(n)
        centroids[0] = X[idx]
        for c in range(1, self.n_clusters):
            sims = X @ centroids[:c].T
            max_sims = sims.max(axis=1)
            distances = 1.0 - max_sims
            probs = distances ** 2
            probs_sum = probs.sum()
            if probs_sum < 1e-12:
                idx = rng.integers(n)
            else:
                probs = probs / probs_sum
                idx = rng.choice(n, p=probs)
            centroids[c] = X[idx]
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
        return centroids

    def _single_run(self, X: np.ndarray, rng: np.random.Generator) -> tuple:
        centroids = self._init_centroids(X, rng)
        prev_assignments = None
        n_iter = self.max_iter
        assignments = None
        for iteration in range(self.max_iter):
            sims = X @ centroids.T
            assignments = sims.argmax(axis=1)
            if prev_assignments is not None:
                if (assignments != prev_assignments).mean() < self.tol:
                    n_iter = iteration + 1
                    break
            prev_assignments = assignments
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_clusters):
                mask = assignments == k
                if mask.any():
                    mean_vec = X[mask].mean(axis=0)
                    norm = np.linalg.norm(mean_vec)
                    new_centroids[k] = mean_vec / norm if norm > 1e-12 else X[rng.integers(len(X))]
                else:
                    new_centroids[k] = X[rng.integers(len(X))]
            centroids = new_centroids
        sims = X @ centroids.T
        max_sims = sims[np.arange(len(X)), assignments]
        inertia = float((1.0 - max_sims).sum())
        return centroids, assignments, inertia, n_iter

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            warnings.warn(
                f"Input not unit-normalized (norms in [{norms.min():.4f}, "
                f"{norms.max():.4f}]). Normalizing."
            )
            X = X / (norms[:, None] + 1e-12)
        rng = np.random.default_rng(self.random_state)
        best_inertia = np.inf
        best_centroids = best_assignments = None
        best_iter = 0
        for _ in range(self.n_init):
            centroids, assignments, inertia, n_iter = self._single_run(X, rng)
            if inertia < best_inertia:
                best_inertia, best_centroids, best_assignments, best_iter = (
                    inertia, centroids, assignments, n_iter
                )
        self.cluster_centers_ = best_centroids
        self.inertia_ = best_inertia
        self.n_iter_ = best_iter
        return best_assignments


def SphericalKMeans(
    n_clusters: int,
    random_state: Optional[int] = None,
    n_init: int = 10,
    max_iter: int = 300,
    use_nltk: bool = False,
    **kwargs,
):
    """Factory returning the best spherical k-means implementation.

    Returns _FallbackSphericalKMeans, which uses:
      - vectorized X @ centroids.T for O(n·K·d) assignment via BLAS
      - k-means++ initialization adapted for cosine distance
      - multiple restarts (n_init) keeping the lowest-inertia partition

    NLTKSphericalKMeans is available as an alternative backend for
    verification purposes, but uses random initialization internally
    (via nltk.cluster.KMeansClusterer) which converges to worse local
    optima. Tests confirm the fallback consistently achieves lower
    inertia than NLTK on synthetic data.

    Both expose the same API: fit_predict / cluster_centers_ / inertia_ / n_iter_.
    """
    if use_nltk:
        if not _NLTK_AVAILABLE:
            raise ImportError("NLTK is not installed. Run: pip install nltk")
        return NLTKSphericalKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
        )
    return _FallbackSphericalKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    )
