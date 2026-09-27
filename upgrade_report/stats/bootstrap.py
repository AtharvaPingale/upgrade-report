"""Paired bootstrap confidence intervals."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Cap on index-matrix elements generated at once, to bound memory on large datasets.
_CHUNK_ELEMENTS = 4_000_000


def bootstrap_means(values: np.ndarray, resamples: int, seed: int) -> np.ndarray:
    """Means of `resamples` bootstrap resamples of `values` (fixed seed, reproducible)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return np.full(resamples, np.nan)
    rng = np.random.default_rng(seed)
    out = np.empty(resamples)
    batch = max(1, _CHUNK_ELEMENTS // n)
    for start in range(0, resamples, batch):
        size = min(batch, resamples - start)
        idx = rng.integers(0, n, size=(size, n))
        out[start:start + size] = values[idx].mean(axis=1)
    return out


@dataclass
class Interval:
    mean: float
    low: float
    high: float
    n: int

    @property
    def valid(self) -> bool:
        return self.n > 0 and not math.isnan(self.mean)


def paired_bootstrap_ci(diffs: np.ndarray, resamples: int = 10_000, confidence: float = 0.95,
                        seed: int = 0) -> Interval:
    """Percentile bootstrap CI for the mean of per-example paired differences."""
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n == 0:
        return Interval(math.nan, math.nan, math.nan, 0)
    means = bootstrap_means(diffs, resamples, seed)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha])
    return Interval(float(diffs.mean()), float(low), float(high), n)

