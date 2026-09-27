"""Noise floor from the baseline's own repeats (a free A/A test)."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from .bootstrap import bootstrap_means


@dataclass
class NoiseFloor:
    value: float | None  # at the resolution of an R-repeat mean; None with a single repeat
    single_repeat: float | None  # the same quantity for one repeat vs one repeat
    repeats: int
    pairs: int


def noise_floor(repeat_scores: np.ndarray, resamples: int = 10_000, seed: int = 0,
                quantile: float = 0.95) -> NoiseFloor:
    """Noise floor of a metric from the baseline's R repeats.

    `repeat_scores` has shape (R, n): one row per repeat, NaN where missing.

    1. For every pair of repeats (1 vs 2, 1 vs 3, 2 vs 3), take the per-example
       differences.
    2. Bootstrap the mean of those differences.
    3. Take the `quantile` of the absolute bootstrapped means, pooled over pairs.

    That is the noise of comparing one repeat with another. The candidate delta
    compares means of R repeats per example, whose run-to-run noise is smaller
    by sqrt(R), so the returned `value` is scaled by 1/sqrt(R). Without the
    scaling, a regression as large as the decision margin can hide inside the
    noise floor on a well-powered dataset.
    """
    scores = np.asarray(repeat_scores, dtype=float)
    repeats = scores.shape[0]
    if repeats < 2:
        return NoiseFloor(None, None, repeats, 0)
    pooled = []
    for i, (a, b) in enumerate(itertools.combinations(range(repeats), 2)):
        mask = ~np.isnan(scores[a]) & ~np.isnan(scores[b])
        diffs = scores[a][mask] - scores[b][mask]
        if len(diffs) < 2:
            continue
        pooled.append(np.abs(bootstrap_means(diffs, resamples, seed + 7919 * (i + 1))))
    if not pooled:
        return NoiseFloor(None, None, repeats, 0)
    single = float(np.quantile(np.concatenate(pooled), quantile))
    return NoiseFloor(single / math.sqrt(repeats), single, repeats, len(pooled))
