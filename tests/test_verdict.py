"""Decision rules on hand-built inputs."""

import numpy as np

from upgrade_report.analysis import MetricResult, compare_metric
from upgrade_report.config import DecisionConfig
from upgrade_report.verdict import CAVEATS, DONT_SHIP, SHIP, decide, recommend

DECISION = DecisionConfig(primary_margin=-0.02, bootstrap_resamples=1000)
EMPTY = {"flags": []}


def metric(role="primary", delta=0.0, ci=(-0.01, 0.01), nf=0.01, mde=0.01, status="within_noise", **kw):
    return MetricResult(name=kw.pop("name", "faithfulness" if role == "primary" else "format"), role=role,
                        kind="continuous", n=300, baseline=0.9, candidate=0.9 + delta, delta=delta, ci=ci,
                        noise_floor=nf, mde=mde, status=status, **kw)


def test_clean_run_ships():
    v = decide([metric(), metric("guardrail")], [], EMPTY, EMPTY, EMPTY, None, DECISION)
    assert v.verdict == SHIP


def test_blocking_metric_means_dont_ship_and_keeps_caveats_as_warnings():
    blocked = metric(delta=-0.05, ci=(-0.07, -0.03), status="fail", blocking=True, reason="primary down",
                     plain_reason="worse")
    behavior = {"flags": [{"name": "length", "message": "answer length +40% (flag: 25%)", "plain": "longer"}]}
    v = decide([blocked], [], behavior, EMPTY, EMPTY, None, DECISION)
    assert v.verdict == DONT_SHIP and v.reasons == ["primary down"] and v.client_reasons == ["worse"]
    assert v.warnings == ["answer length +40% (flag: 25%)"]


def test_mde_larger_than_margin_is_a_caveat():
    v = decide([metric(mde=0.05)], [], EMPTY, EMPTY, EMPTY, None, DECISION)
    assert v.verdict == CAVEATS and "too small" in v.reasons[0]


def test_ci_crossing_the_margin_within_noise_is_a_caveat():
    v = decide([metric(delta=-0.005, ci=(-0.025, 0.015))], [], EMPTY, EMPTY, EMPTY, None, DECISION)
    assert v.verdict == CAVEATS and v.reasons[0].startswith("non-inferiority not established")


def test_pairwise_preference_for_the_baseline_is_a_caveat():
    pairwise = {"n": 75, "preference_rate": 0.35, "ci": [0.25, 0.46]}
    v = decide([metric()], [], EMPTY, EMPTY, EMPTY, pairwise, DECISION)
    assert v.verdict == CAVEATS


def _mats(delta, sigma=0.05, n=300, repeats=3, seed=0):
    rng = np.random.default_rng(seed)
    latent = rng.uniform(0.5, 0.9, n)
    base = latent + rng.normal(0, sigma, (repeats, n))
    cand = latent + delta + rng.normal(0, sigma, (repeats, n))
    return base, cand


def test_primary_blocks_below_margin_beyond_noise():
    r = compare_metric("q", "primary", "continuous", *_mats(-0.05), DECISION)
    assert r.blocking and r.status == "fail" and r.ci[0] < -0.02


def test_primary_improvement_never_blocks_even_with_a_wide_ci():
    r = compare_metric("q", "primary", "continuous", *_mats(0.01, sigma=0.4, n=40), DECISION)
    assert not r.blocking


def test_small_primary_drop_inside_margin_does_not_block():
    r = compare_metric("q", "primary", "continuous", *_mats(-0.008, sigma=0.02), DECISION)
    assert not r.blocking and r.status in ("regressed", "within_noise")


def test_guardrail_uses_noise_floor_or_absolute_tolerance():
    base, cand = _mats(-0.03, sigma=0.05)
    assert compare_metric("g", "guardrail", "continuous", base, cand, DECISION).blocking
    loose = DecisionConfig(primary_margin=-0.02, guardrail_tolerance=0.05, bootstrap_resamples=1000)
    assert not compare_metric("g", "guardrail", "continuous", base, cand, loose).blocking


def test_tracked_metrics_never_block():
    r = compare_metric("t", "tracked", "continuous", *_mats(-0.3), DECISION)
    assert not r.blocking and r.status == "regressed"


def test_binary_metric_reports_flips():
    base = np.ones((3, 100))
    cand = np.ones((3, 100))
    cand[:, :12] = 0
    base[:, 90:] = 0
    r = compare_metric("f", "guardrail", "binary", base, cand, DECISION)
    assert (r.flips.regressed, r.flips.improved) == (12, 10)


def test_recommend_prefers_verdict_then_fewer_concerns():
    def comp(arm, verdict, reasons, delta):
        return {"arm": arm, "verdict": verdict, "verdict_reasons": reasons, "warnings": [],
                "metrics": [{"role": "primary", "delta": delta}], "cost": {"candidate": {"per_query_usd": 1}}}

    arms = [comp("a", CAVEATS, ["x", "y"], 0.05), comp("b", CAVEATS, ["x"], 0.0), comp("c", DONT_SHIP, ["z"], 0.1)]
    assert recommend(arms)["arm"] == "b"
    assert recommend(arms + [comp("d", SHIP, ["ok"], -0.01)])["arm"] == "d"
