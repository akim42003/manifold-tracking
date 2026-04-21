"""Analysis primitives for direction bundles.

Each function accepts a DirectionBundle (or raw numpy array) and returns a
structured result with provenance attached.  Every function that does linear
algebra explicitly documents which metric it uses and why.

Metric policy
-------------
- Relation offsets, residual vectors: cosine (direction matters, magnitude does not)
- Gate vectors fed to UMAP: cosine by default, but pass metric="euclidean" if you
  want magnitude preserved (firing strength encodes feature selectivity)
- SVD: metric-free — it operates on the centred data matrix directly
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from manifold_tools._types import DirectionBundle, ProjectionResult, SVDResult

# ── SVD with bootstrap ────────────────────────────────────────────────────────

_RANDOMIZED_SVD_THRESHOLD = 50_000   # rows above which we switch to randomized


def svd_with_bootstrap(
    bundle: DirectionBundle,
    n_components: Optional[int] = None,
    n_bootstrap: int = 200,
    bootstrap_ci: float = 0.90,
    random_state: int = 42,
    center: bool = True,
) -> SVDResult:
    """SVD of a DirectionBundle with bootstrap confidence bands.

    Bootstrap procedure: resample rows with replacement, re-run SVD, collect
    the i-th singular value across resamples.  This gives honest uncertainty
    estimates for small-N datasets where the spectrum looks clean but isn't.

    Parameters
    ----------
    bundle:
        Source data.  Must have float32 vectors.
    n_components:
        Number of singular values to return.  Defaults to min(n, d_model).
    n_bootstrap:
        Bootstrap resamples.  200 is enough for stable 90% bands.
    bootstrap_ci:
        Width of the confidence interval (default 0.90 → 5th/95th percentiles).
    center:
        Subtract the column mean before SVD (standard PCA convention).

    Returns
    -------
    SVDResult
        singular_values, cumulative_variance, bootstrap_lower, bootstrap_upper,
        spectral_gap_idx, spectral_gap_ratio.
    """
    X = bundle.vectors.astype(np.float32)
    n, d = X.shape

    if center:
        X = X - X.mean(axis=0)

    k = n_components if n_components is not None else min(n, d)
    k = min(k, min(n, d))

    # Full SVD on the observed data
    S = _run_svd(X, k)

    variance = S ** 2
    total_var = float(variance.sum())
    if total_var < 1e-12:
        raise ValueError("All singular values are zero — data is constant.")
    cum_var = np.cumsum(variance) / total_var

    # Spectral gap: largest ratio S[i]/S[i+1] in the first min(50, k-1) components
    search_depth = min(50, k - 1)
    if search_depth > 0:
        ratios = S[:search_depth] / np.maximum(S[1:search_depth + 1], 1e-12)
        gap_idx = int(np.argmax(ratios))
        gap_ratio = float(ratios[gap_idx])
    else:
        gap_idx, gap_ratio = 0, 1.0

    # Bootstrap
    rng = np.random.default_rng(random_state)
    alpha = (1.0 - bootstrap_ci) / 2.0
    bootstrap_S = np.zeros((n_bootstrap, k), dtype=np.float64)

    for i in range(n_bootstrap):
        row_idx = rng.integers(0, n, size=n)
        X_boot = X[row_idx]
        if center:
            X_boot = X_boot - X_boot.mean(axis=0)
        bootstrap_S[i] = _run_svd(X_boot, k)

    lower = np.percentile(bootstrap_S, alpha * 100, axis=0)
    upper = np.percentile(bootstrap_S, (1 - alpha) * 100, axis=0)

    return SVDResult(
        singular_values=S,
        cumulative_variance=cum_var,
        bootstrap_lower=lower.astype(np.float32),
        bootstrap_upper=upper.astype(np.float32),
        spectral_gap_idx=gap_idx,
        spectral_gap_ratio=gap_ratio,
        n_vectors=n,
        d_model=d,
        provenance={
            "vindex_hash": bundle.vindex_hash,
            "loader_config": bundle.config,
            "n_bootstrap": n_bootstrap,
            "bootstrap_ci": bootstrap_ci,
            "centered": center,
            "n_components": k,
        },
    )


def _run_svd(X: np.ndarray, k: int) -> np.ndarray:
    """Return the top-k singular values of X."""
    n, d = X.shape
    if n > _RANDOMIZED_SVD_THRESHOLD or d > _RANDOMIZED_SVD_THRESHOLD:
        from sklearn.utils.extmath import randomized_svd
        _, S, _ = randomized_svd(X, n_components=k, random_state=0)
    else:
        _, S, _ = np.linalg.svd(X, full_matrices=False)
        S = S[:k]
    return S.astype(np.float64)


# ── UMAP projection ────────────────────────────────────────────────────────────

def umap_project(
    bundle: DirectionBundle,
    n_components: int = 2,
    metric: str = "cosine",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    **umap_kwargs,
) -> ProjectionResult:
    """Project a DirectionBundle to low-dimensional coordinates via UMAP.

    Parameters
    ----------
    bundle:
        Source data.
    n_components:
        Output dimensionality (2 for scatter plots, 3 for 3D).
    metric:
        Distance metric.  Use 'cosine' for relation offsets and residuals
        (direction-only data).  Consider 'euclidean' for gate vectors if
        firing strength (magnitude) should influence the layout.
    n_neighbors:
        UMAP local neighbourhood size.  Smaller → more local structure.
    min_dist:
        Controls how tightly UMAP packs points in the embedding.

    Returns
    -------
    ProjectionResult
        coords [n, n_components] and the original metadata DataFrame.
    """
    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn is required: pip install umap-learn")

    reducer = umap.UMAP(
        n_components=n_components,
        metric=metric,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        **umap_kwargs,
    )

    coords = reducer.fit_transform(bundle.vectors).astype(np.float32)

    return ProjectionResult(
        coords=coords,
        metadata=bundle.metadata.reset_index(drop=True),
        n_components=n_components,
        metric=metric,
        provenance={
            "vindex_hash": bundle.vindex_hash,
            "loader_config": bundle.config,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "random_state": random_state,
        },
    )


# ── Isomap projection ──────────────────────────────────────────────────────────

def isomap_project(
    bundle: DirectionBundle,
    n_components: int = 2,
    n_neighbors: int = 10,
    metric: str = "cosine",
) -> ProjectionResult:
    """Project via Isomap (geodesic distance manifold unfolding).

    Slower than UMAP but preserves global structure better on small datasets.
    """
    from sklearn.manifold import Isomap

    iso = Isomap(n_components=n_components, n_neighbors=n_neighbors, metric=metric)
    coords = iso.fit_transform(bundle.vectors).astype(np.float32)

    return ProjectionResult(
        coords=coords,
        metadata=bundle.metadata.reset_index(drop=True),
        n_components=n_components,
        metric=metric,
        provenance={
            "vindex_hash": bundle.vindex_hash,
            "loader_config": bundle.config,
            "n_neighbors": n_neighbors,
        },
    )


# ── Procrustes alignment ───────────────────────────────────────────────────────

def procrustes_align(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Orthogonal Procrustes alignment of source onto target subspace.

    Both arrays should be [n, k] matrices of basis vectors (e.g. top-k right
    singular vectors from SVD).  Returns (aligned_source, cosine_similarity).

    Use this to compare subspaces: e.g. the top-10D gate manifold at layer 14
    vs layer 27, or the same relation manifold across two model families.
    """
    from scipy.linalg import orthogonal_procrustes

    R, scale = orthogonal_procrustes(source, target)
    aligned = source @ R

    # Mean cosine similarity between aligned rows and target rows
    norms_a = np.linalg.norm(aligned, axis=1, keepdims=True)
    norms_t = np.linalg.norm(target, axis=1, keepdims=True)
    cos = (aligned / np.maximum(norms_a, 1e-12)) * (target / np.maximum(norms_t, 1e-12))
    similarity = float(cos.sum(axis=1).mean())

    return aligned, similarity


# ── Grassmannian distance ──────────────────────────────────────────────────────

def grassmannian_distance(
    U1: np.ndarray,
    U2: np.ndarray,
) -> float:
    """Geodesic distance between two subspaces on the Grassmannian.

    U1, U2 are orthonormal basis matrices [d_model, k].  Computed via
    principal angles: arcsin of singular values of U1.T @ U2.

    dist = sqrt(sum(theta_i^2)) where theta_i are the principal angles.
    """
    M = U1.T @ U2
    _, sigma, _ = np.linalg.svd(M, full_matrices=False)
    sigma = np.clip(sigma, -1.0, 1.0)
    angles = np.arcsin(np.sqrt(np.maximum(0, 1 - sigma ** 2)))
    return float(np.sqrt(np.sum(angles ** 2)))
