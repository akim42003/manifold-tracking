"""manifold_tools — geometric structure extraction from transformer weights.

Targeted at three geometric primitives from language models:
  - Gate vectors (full-rank, superposition regime)
  - Relation offset vectors (5-22D, the primary manifold analysis target)
  - Residual stream trajectories (cross-layer, requires per-layer normalization)

Depends on LARQL Python bindings for weight loading.  Analysers (SVD, UMAP,
Procrustes) are pure Python/numpy and work without a live vindex.

Quick start
-----------
    from manifold_tools import directions as dirs, manifolds, visualize

    bundle = dirs.load_relation_offsets("gpt2.vindex")
    svd = manifolds.svd_with_bootstrap(bundle)
    proj = manifolds.umap_project(bundle)

    visualize.spectrum_plot(svd).savefig("spectrum.png")
    visualize.projection_plot(proj).savefig("umap.png")
"""

from manifold_tools._types import DirectionBundle, SVDResult, ProjectionResult
from manifold_tools import directions, manifolds, visualize

__version__ = "0.1.0"

__all__ = [
    "DirectionBundle",
    "SVDResult",
    "ProjectionResult",
    "directions",
    "manifolds",
    "visualize",
]
