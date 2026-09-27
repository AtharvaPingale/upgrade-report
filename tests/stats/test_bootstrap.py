import numpy as np
import pytest

from upgrade_report.stats import bootstrap_means, paired_bootstrap_ci


@pytest.mark.parametrize("true_delta", [0.0, 0.05, 0.3])
def test_ci_coverage_is_near_95_percent(true_delta):
    rng = np.random.default_rng(1)
    trials, n = 400, 200
    covered = 0
    for t in range(trials):
        diffs = true_delta + rng.normal(0, 0.5, size=n)
        ci = paired_bootstrap_ci(diffs, resamples=1000, seed=t)
        covered += ci.low <= true_delta <= ci.high
    assert 0.92 <= covered / trials <= 0.975


def test_paired_design_cancels_example_difficulty():
    rng = np.random.default_rng(2)
    difficulty = rng.uniform(0, 1, size=300)
    base = difficulty + rng.normal(0, 0.02, 300)
    cand = difficulty + 0.01 + rng.normal(0, 0.02, 300)
    paired = paired_bootstrap_ci(cand - base, resamples=2000, seed=0)
    # an unpaired comparison of the same data could not see a 0.01 shift under this spread
    assert paired.low > 0
    assert paired.high - paired.low < 0.01


def test_fixed_seed_is_reproducible():
    diffs = np.random.default_rng(3).normal(size=50)
    assert paired_bootstrap_ci(diffs, seed=7) == paired_bootstrap_ci(diffs, seed=7)


def test_nan_and_empty_inputs():
    ci = paired_bootstrap_ci(np.array([np.nan, 1.0, 1.0]), resamples=200)
    assert ci.n == 2 and ci.low == ci.high == 1.0
    empty = paired_bootstrap_ci(np.array([]), resamples=200)
    assert empty.n == 0 and not empty.valid


def test_chunked_resampling_matches_mean():
    values = np.arange(10_000, dtype=float)
    means = bootstrap_means(values, 1000, seed=0)  # 10M index elements → several chunks
    assert len(means) == 1000
    assert abs(means.mean() - values.mean()) < 50
