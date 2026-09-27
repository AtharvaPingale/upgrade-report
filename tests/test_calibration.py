"""Evidence that the verdicts are right: A/A calibration and injected regressions."""

import fake_target
import numpy as np
import pytest
from conftest import base_config, write_config

from upgrade_report.analysis import compare_metric, compare_slices
from upgrade_report.config import DecisionConfig, load_config
from upgrade_report.dataset import Example
from upgrade_report.pipeline import RunOptions, run
from upgrade_report.verdict import SHIP, decide

NO_FLAGS = {"flags": []}


def _synthetic_trial(rng, n=300, repeats=3, shift=0.0, guardrail_drop=0.0):
    latent = rng.uniform(0.4, 0.9, n)
    p_pass = rng.uniform(0.9, 1.0, n)

    def arm(offset, drop):
        cont = latent[None, :] + offset + rng.normal(0, 0.1, (repeats, n))
        binary = (rng.random((repeats, n)) < (p_pass - drop)[None, :]).astype(float)
        return {"quality": cont, "format": binary}

    return arm(0.0, 0.0), arm(shift, guardrail_drop)


def _examples(n):
    tags = ["billing", "shipping", "account", "returns"]
    return [Example(id=str(i), inputs={}, metadata={"tags": [tags[i % 4]] + (["legal"] if i < 8 else [])})
            for i in range(n)]


def _verdict(base, cand, decision, examples):
    metrics = [compare_metric("quality", "primary", "continuous", base["quality"], cand["quality"], decision),
               compare_metric("format", "guardrail", "binary", base["format"], cand["format"], decision)]
    slices = compare_slices(examples, "tags", [("quality", "primary", "continuous"), ("format", "guardrail", "binary")],
                            base, cand, decision)
    return decide(metrics, slices, NO_FLAGS, NO_FLAGS, NO_FLAGS, None, decision)


@pytest.mark.slow
def test_aa_runs_ship_at_least_95_percent_of_the_time():
    rng = np.random.default_rng(0)
    decision = DecisionConfig(primary_margin=-0.02, bootstrap_resamples=1000)
    examples = _examples(300)
    trials = 200
    verdicts = [_verdict(*_synthetic_trial(rng), decision, examples) for _ in range(trials)]
    ship_rate = sum(v.verdict == SHIP for v in verdicts) / trials
    reasons = [r for v in verdicts if v.verdict != SHIP for r in v.reasons]
    assert ship_rate >= 0.95, reasons[:5]


def test_regression_beyond_the_margin_is_caught_at_the_expected_rate():
    rng = np.random.default_rng(1)
    decision = DecisionConfig(primary_margin=-0.02, bootstrap_resamples=1000)
    examples = _examples(300)
    caught = sum(_verdict(*_synthetic_trial(rng, shift=-0.04), decision, examples).verdict == "dont_ship"
                 for _ in range(60))
    # a drop of twice the margin on 300 examples, noise sd 0.1, 3 repeats: well above the MDE
    assert caught / 60 >= 0.95


def test_guardrail_regression_is_caught():
    rng = np.random.default_rng(2)
    decision = DecisionConfig(primary_margin=-0.02, bootstrap_resamples=1000)
    verdict = _verdict(*_synthetic_trial(rng, guardrail_drop=0.3), decision, _examples(300))
    assert verdict.verdict == "dont_ship"
    assert any("format (guardrail)" in r for r in verdict.reasons)


@pytest.mark.slow
def test_aa_end_to_end_with_fake_target(tmp_path, no_network):
    ships = 0
    trials = 8
    for seed in range(trials):
        project = tmp_path / f"trial{seed}"
        project.mkdir()
        cfg = base_config(project, n=300, resamples=1000, decision={"primary_margin": -0.02})
        fake_target.reset(seed)
        result = run(load_config(write_config(project, cfg)), RunOptions(aa=True, log=lambda m: None))
        ships += result.report["verdict"] == SHIP
        assert result.report["mode"] == "aa"
    assert ships >= trials - 1


@pytest.mark.parametrize("seed", range(4))
def test_injected_ten_percent_failure_is_caught(tmp_path, no_network, seed):
    """A fake candidate that fails a fixed 10% of examples must be caught (plan §17)."""
    cfg = base_config(
        tmp_path, n=300, candidate="fake-regress10", resamples=1000,
        evaluators=["fake_evaluators:correct", "fake_evaluators:format_compliance"],
        metrics={"primary": "correct", "guardrails": ["format_compliance"], "tracked": []},
        decision={"primary_margin": -0.02},
    )
    fake_target.reset(seed)
    result = run(load_config(write_config(tmp_path, cfg)), RunOptions(log=lambda m: None))
    assert result.report["verdict"] == "dont_ship"
    assert result.exit_code == 20
    primary = result.report["metrics"][0]
    assert primary["name"] == "correct" and primary["status"] == "fail"
    regressed = {e["example_id"] for e in result.report["examples"] if e["direction"] == "regressed"}
    injected = {f"q0-{i:04d}" for i in range(300) if fake_target.regressed_example(f"q0-{i:04d}")}
    assert regressed & injected
