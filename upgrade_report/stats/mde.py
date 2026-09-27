"""Minimum detectable effect for a paired comparison."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def minimum_detectable_effect(diffs: np.ndarray, alpha: float = 0.05, power: float = 0.8) -> float | None:
    """Smallest true mean difference a two-sided test at `alpha` detects with `power`,
    given the observed spread of per-example paired differences and the sample size.
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n < 2:
        return None
    sd = float(np.std(diffs, ddof=1))
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * sd / math.sqrt(n))
