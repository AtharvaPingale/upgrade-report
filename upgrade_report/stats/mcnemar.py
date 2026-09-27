"""McNemar's test on discordant pass/fail pairs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

EXACT_BELOW = 25


@dataclass
class McNemarResult:
    improved: int  # failed in baseline, passed in candidate
    regressed: int  # passed in baseline, failed in candidate
    p_value: float
    method: str  # "exact", "chi2", or "none" when there are no discordant pairs

    def to_json(self) -> dict:
        return {"improved": self.improved, "regressed": self.regressed, "p_value": self.p_value,
                "method": self.method}


def mcnemar(baseline_pass: np.ndarray, candidate_pass: np.ndarray, exact_below: int = EXACT_BELOW) -> McNemarResult:
    base = np.asarray(baseline_pass, dtype=bool)
    cand = np.asarray(candidate_pass, dtype=bool)
    regressed = int(np.sum(base & ~cand))
    improved = int(np.sum(~base & cand))
    n = regressed + improved
    if n == 0:
        return McNemarResult(improved, regressed, 1.0, "none")
    if n < exact_below:
        p = 2 * stats.binom.cdf(min(regressed, improved), n, 0.5)
        return McNemarResult(improved, regressed, float(min(1.0, p)), "exact")
    statistic = (abs(regressed - improved) - 1) ** 2 / n
    return McNemarResult(improved, regressed, float(stats.chi2.sf(statistic, df=1)), "chi2")
