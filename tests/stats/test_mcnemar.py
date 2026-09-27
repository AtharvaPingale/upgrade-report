import numpy as np
from scipy import stats

from upgrade_report.stats import mcnemar


def _pairs(improved: int, regressed: int, concordant: int = 50):
    base = [False] * improved + [True] * regressed + [True] * concordant
    cand = [True] * improved + [False] * regressed + [True] * concordant
    return np.array(base), np.array(cand)


def test_exact_binomial_below_25_discordant_pairs():
    result = mcnemar(*_pairs(improved=12, regressed=9))
    assert (result.improved, result.regressed, result.method) == (12, 9, "exact")
    assert abs(result.p_value - stats.binomtest(9, 21, 0.5).pvalue) < 1e-9
    assert round(result.p_value, 2) == 0.66  # the example in the plan


def test_chi_square_with_continuity_correction_from_25_pairs():
    result = mcnemar(*_pairs(improved=30, regressed=15))
    assert result.method == "chi2"
    expected = stats.chi2.sf((abs(15 - 30) - 1) ** 2 / 45, df=1)
    assert abs(result.p_value - expected) < 1e-12


def test_no_discordant_pairs():
    result = mcnemar(*_pairs(0, 0))
    assert result.p_value == 1.0 and result.method == "none"


def test_type_one_error_is_controlled():
    rng = np.random.default_rng(0)
    rejections = 0
    trials = 1000
    for _ in range(trials):
        base = rng.random(200) < 0.8
        cand = rng.random(200) < 0.8
        rejections += mcnemar(base, cand).p_value < 0.05
    assert rejections / trials <= 0.06
