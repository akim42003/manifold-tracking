from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DirectionBundle:
    """Paired array of direction vectors and per-vector metadata.

    Every loader produces one of these. Metadata columns vary by loader but
    always include enough to color a scatter plot or filter an analysis.
    """

    vectors: np.ndarray   # [n, d_model], float32
    metadata: pd.DataFrame
    vindex_hash: str = ""   # SHA256 of gate_vectors.bin — provenance anchor
    config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.vectors) != len(self.metadata):
            raise ValueError(
                f"vectors has {len(self.vectors)} rows but metadata has "
                f"{len(self.metadata)} rows"
            )

    @property
    def n(self) -> int:
        return self.vectors.shape[0]

    @property
    def d_model(self) -> int:
        return self.vectors.shape[1]

    def filter(self, mask: np.ndarray) -> "DirectionBundle":
        """Return a new bundle with rows selected by boolean or integer mask."""
        return DirectionBundle(
            vectors=self.vectors[mask],
            metadata=self.metadata.iloc[mask].reset_index(drop=True),
            vindex_hash=self.vindex_hash,
            config=self.config,
        )


@dataclass
class SVDResult:
    singular_values: np.ndarray       # [k] — truncated to min(n, d_model)
    cumulative_variance: np.ndarray   # [k] — fraction of total variance
    bootstrap_lower: np.ndarray       # [k] — 5th percentile over resamples
    bootstrap_upper: np.ndarray       # [k] — 95th percentile over resamples
    d50_distribution: np.ndarray      # [n_bootstrap] — d50 per resample
    spectral_gap_idx: int             # index i where S[i]/S[i+1] is largest
    spectral_gap_ratio: float
    n_vectors: int
    d_model: int
    provenance: dict = field(default_factory=dict)


@dataclass
class ProjectionResult:
    coords: np.ndarray        # [n, n_components]
    metadata: pd.DataFrame    # same rows as input bundle
    n_components: int
    metric: str
    provenance: dict = field(default_factory=dict)
