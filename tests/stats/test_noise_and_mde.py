import math

import numpy as np

from upgrade_report.stats import minimum_detectable_effect, noise_floor, paired_bootstrap_ci


def _arm(latent: np.ndarray, repeats: int, sigma: float, rng) -> np.ndarray:
    return latent[None, :] + rng.normal(0, sigma, size=(repeats, len(latent)))


def test_aa_differences_fall_within_the_noise_floor():
    rng = np.random.default_rng(0)
    within = 0
    trials = 200
    for t in range(trials):
        latent = rng.uniform(0.3, 0.9, 300)
        base, cand = _arm(latent, 3, 0.1, rng), _arm(latent, 3, 0.1, rng)
        nf = noise_floor(base, resamples=1000, seed=t).value
        delta = cand.mean(axis=0).mean() - base.mean(axis=0).mean()
        within += abs(delta) <= nf
    assert within / trials >= 0.95


def test_noise_floor_is_scaled_to_repeat_average_resolution():
    rng = np.random.default_rng(1)
    base = _arm(rng.uniform(0, 1, 400), 4, 0.1, rng)
    nf = noise_floor(base, resamples=1000, seed=0)
    assert nf.pairs == 6
    assert math.isclose(nf.value, nf.single_repeat / 2)


def test_a_single_repeat_has_no_noise_floor():
    nf = noise_floor(np.ones((1, 10)))
    assert nf.value is None and nf.repeats == 1


def test_deterministic_metric_has_zero_noise_floor():
    assert noise_floor(np.tile(np.linspace(0, 1, 50), (3, 1)), resamples=500).value == 0.0


def test_missing_values_are_ignored_pairwise():
    scores = np.array([[1.0, 0.5, np.nan, 0.2], [1.0, np.nan, 0.3, 0.2], [0.9, 0.5, 0.3, np.nan]])
    assert noise_floor(scores, resamples=500).value is not None


def test_true_effect_equal_to_mde_is_detected_about_80_percent_of_the_time():
    rng = np.random.default_rng(2)
    n, sd = 250, 0.4
    pilot = rng.normal(0, sd, 100_000)  # a large pilot pins down the spread
    mde_n = minimum_detectable_effect(pilot) * math.sqrt(len(pilot) / n)  # MDE at the trial size
    detected = 0
    trials = 400
    for t in range(trials):
        diffs = mde_n + rng.normal(0, sd, n)
        ci = paired_bootstrap_ci(diffs, resamples=1000, seed=t)
        detected += ci.low > 0
    assert 0.72 <= detected / trials <= 0.88


def test_mde_shrinks_with_more_examples():
    rng = np.random.default_rng(3)
    small = minimum_detectable_effect(rng.normal(0, 0.3, 50))
    large = minimum_detectable_effect(rng.normal(0, 0.3, 800))
    assert large < small / 3
    assert minimum_detectable_effect(np.array([0.1])) is None
