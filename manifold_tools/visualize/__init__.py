"""Plotting conventions for manifold analysis results.

All functions return matplotlib Figure objects.  Call fig.savefig(...) or
fig.show() yourself — nothing is displayed automatically.

Layout choices are deliberately simple: two-panel spectrum, scatter for UMAP.
The value is in the earlier modules; this one just makes results readable.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from manifold_tools._types import ProjectionResult, SVDResult


# Threshold lines drawn on cumulative variance panel
_VARIANCE_THRESHOLDS = [0.50, 0.90, 0.95, 0.99]

# Colourblind-friendly categorical palette (max ~20 categories)
_DEFAULT_PALETTE = [
    "#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE",
    "#AA3377", "#BBBBBB", "#44AA99", "#DDCC77", "#332288",
    "#882255", "#117733", "#999933", "#88CCEE", "#CC6677",
    "#AA4499", "#44AA88", "#DDDDDD", "#000000", "#88CCDD",
]


def spectrum_plot(
    result: SVDResult,
    title: str = "",
    max_components: int = 50,
) -> "matplotlib.figure.Figure":
    """Two-panel singular value spectrum with bootstrap bands.

    Left panel: log-scale singular values + bootstrap CI ribbon.
    Right panel: cumulative explained variance with threshold annotations
                 and a vertical line at the spectral gap.

    Parameters
    ----------
    result:
        Output of manifolds.svd_with_bootstrap().
    title:
        Optional figure title.
    max_components:
        How many components to show on the x-axis (default 50).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = min(max_components, len(result.singular_values))
    idx = np.arange(1, k + 1)
    S = result.singular_values[:k]
    lower = result.bootstrap_lower[:k]
    upper = result.bootstrap_upper[:k]
    cum = result.cumulative_variance[:k]

    fig, (ax_s, ax_v) = plt.subplots(1, 2, figsize=(12, 5))

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)

    # ── left: singular values (log scale) ──
    ax_s.semilogy(idx, S, "o-", color="#4477AA", ms=4, lw=1.5, label="observed")
    ax_s.fill_between(idx, lower, upper, alpha=0.25, color="#4477AA",
                       label=f"{int(result.provenance.get('bootstrap_ci', 0.9)*100)}% CI")

    # Mark the spectral gap
    gap = result.spectral_gap_idx
    if 0 < gap < k - 1:
        ax_s.axvline(gap + 1, color="#EE6677", ls="--", lw=1,
                     label=f"gap @ {gap + 1} (×{result.spectral_gap_ratio:.1f})")

    ax_s.set_xlabel("Component")
    ax_s.set_ylabel("Singular value (log scale)")
    ax_s.set_xlim(0.5, k + 0.5)
    ax_s.legend(fontsize=9)
    ax_s.set_title(
        f"Spectrum  (n={result.n_vectors}, d={result.d_model})", fontsize=11
    )
    ax_s.grid(True, which="both", ls=":", alpha=0.4)

    # ── right: cumulative variance ──
    ax_v.plot(idx, cum * 100, "o-", color="#228833", ms=4, lw=1.5)

    colors = ["#CCBB44", "#EE6677", "#AA3377", "#332288"]
    for thresh, color in zip(_VARIANCE_THRESHOLDS, colors):
        # find first component that crosses this threshold
        crossings = np.where(cum >= thresh)[0]
        if len(crossings):
            c_idx = int(crossings[0]) + 1
            ax_v.axhline(thresh * 100, color=color, ls=":", lw=1, alpha=0.8)
            ax_v.axvline(c_idx, color=color, ls=":", lw=1, alpha=0.8)
            ax_v.annotate(
                f"{thresh*100:.0f}% @ {c_idx}D",
                xy=(c_idx, thresh * 100),
                xytext=(c_idx + 0.5, thresh * 100 - 4),
                fontsize=8, color=color,
            )

    ax_v.set_xlabel("Component")
    ax_v.set_ylabel("Cumulative variance (%)")
    ax_v.set_xlim(0.5, k + 0.5)
    ax_v.set_ylim(0, 105)
    ax_v.set_title("Cumulative explained variance", fontsize=11)
    ax_v.grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    return fig


def projection_plot(
    result: ProjectionResult,
    color_by: str = "cluster_label",
    label_col: Optional[str] = None,
    size: float = 8.0,
    alpha: float = 0.7,
    title: str = "",
    max_legend_items: int = 20,
) -> "matplotlib.figure.Figure":
    """Scatter plot of a 2D projection colored by a metadata column.

    Parameters
    ----------
    result:
        Output of manifolds.umap_project() or manifolds.isomap_project().
    color_by:
        Column name in result.metadata to use for colouring points.
        Categorical columns get a legend; continuous columns get a colorbar.
    label_col:
        If set, annotate each point with this metadata column's value.
    size:
        Point size.
    alpha:
        Point opacity.
    title:
        Optional plot title.
    max_legend_items:
        Truncate legend if more than this many categories.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if result.n_components < 2:
        raise ValueError("projection_plot requires n_components >= 2")

    coords = result.coords
    meta = result.metadata

    fig, ax = plt.subplots(figsize=(9, 7))

    if color_by in meta.columns:
        col = meta[color_by]
        if pd.api.types.is_numeric_dtype(col) and col.nunique() > max_legend_items:
            # Continuous colormap
            sc = ax.scatter(coords[:, 0], coords[:, 1],
                            c=col.values, cmap="viridis", s=size, alpha=alpha)
            fig.colorbar(sc, ax=ax, label=color_by)
        else:
            # Categorical
            categories = col.unique().tolist()
            cat_to_color = {
                cat: _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)]
                for i, cat in enumerate(categories)
            }
            colors = [cat_to_color[v] for v in col]
            ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=size, alpha=alpha)

            if len(categories) <= max_legend_items:
                from matplotlib.patches import Patch
                handles = [
                    Patch(color=cat_to_color[cat], label=str(cat))
                    for cat in categories
                ]
                ax.legend(handles=handles, fontsize=8, loc="best",
                          ncol=max(1, len(categories) // 12))
            else:
                ax.set_title(
                    f"{title} (too many categories for legend)"
                    if title else f"Too many categories for legend",
                    fontsize=11,
                )
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=size, alpha=alpha, color="#4477AA")

    if label_col and label_col in meta.columns:
        for i, label in enumerate(meta[label_col]):
            ax.annotate(str(label), (coords[i, 0], coords[i, 1]),
                        fontsize=6, alpha=0.6)

    ax.set_xlabel("UMAP 1" if "umap" in str(result.provenance).lower() else "Dim 1")
    ax.set_ylabel("UMAP 2" if "umap" in str(result.provenance).lower() else "Dim 2")

    n = len(coords)
    metric = result.metric
    plot_title = title or f"Projection  (n={n}, metric={metric}, color={color_by})"
    ax.set_title(plot_title, fontsize=11)
    ax.grid(True, ls=":", alpha=0.3)

    fig.tight_layout()
    return fig


def trajectory_plot(
    residuals: "np.ndarray",
    token_labels: Optional[Sequence[str]] = None,
    layer_labels: Optional[Sequence[str]] = None,
    title: str = "",
) -> "matplotlib.figure.Figure":
    """Plot residual stream norms across layers for one or more token positions.

    Parameters
    ----------
    residuals:
        Shape [n_tokens, n_layers, d_model] or [n_layers, d_model] for a
        single token.  Norms are plotted per layer.
    token_labels:
        Legend labels for each token position.
    layer_labels:
        X-axis tick labels.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = np.array(residuals)
    if r.ndim == 2:
        r = r[np.newaxis]   # treat as single token

    n_tokens, n_layers, _ = r.shape
    norms = np.linalg.norm(r, axis=-1)   # [n_tokens, n_layers]

    fig, ax = plt.subplots(figsize=(10, 5))

    for i in range(n_tokens):
        label = token_labels[i] if token_labels else f"token {i}"
        color = _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)]
        ax.plot(range(n_layers), norms[i], "o-", ms=4, lw=1.5,
                color=color, label=label)

    if layer_labels:
        ax.set_xticks(range(len(layer_labels)))
        ax.set_xticklabels(layer_labels, rotation=45, ha="right", fontsize=8)
    else:
        ax.set_xlabel("Layer")

    ax.set_ylabel("Residual ‖x‖")
    ax.set_title(title or "Residual stream norms", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    return fig


def procrustes_heatmap(
    alignment_matrix: "np.ndarray",
    row_labels: Optional[Sequence[str]] = None,
    col_labels: Optional[Sequence[str]] = None,
    title: str = "Procrustes alignment",
) -> "matplotlib.figure.Figure":
    """Heatmap of pairwise Procrustes cosine similarity scores."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(alignment_matrix, vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")
    fig.colorbar(im, ax=ax, label="cosine similarity")

    n = alignment_matrix.shape[0]
    if row_labels:
        ax.set_yticks(range(n))
        ax.set_yticklabels(row_labels, fontsize=8)
    if col_labels:
        ax.set_xticks(range(alignment_matrix.shape[1]))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)

    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig
