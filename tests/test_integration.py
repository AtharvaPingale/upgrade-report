"""End to end with a fake target and a fake judge. No network calls."""

import json

import fake_judge
import fake_target
import pytest
from conftest import base_config, write_config

from upgrade_report.config import load_config
from upgrade_report.errors import JudgeDriftError
from upgrade_report.pipeline import RunOptions, run
from upgrade_report.tracing import NullSink, make_sink

QUIET = RunOptions(log=lambda m: None)

PLAN_KEYS = {"project", "generated_at", "config_hash", "verdict", "verdict_reasons", "metrics", "flips", "slices",
             "behavior", "cost", "latency", "examples", "caveats"}


def _run(tmp_path, **kwargs):
    cfg = load_config(write_config(tmp_path, base_config(tmp_path, **kwargs)))
    return run(cfg, QUIET)


def test_end_to_end_with_judge_and_pairwise(tmp_path, no_network):
    result = _run(tmp_path, use_judge=True, pairwise=True)
    report = result.report
    assert PLAN_KEYS <= set(report)
    assert report["verdict"] in ("ship", "ship_with_caveats")
    assert result.exit_code in (0, 10)
    for kind in ("json", "md", "html"):
        assert result.paths[kind].exists()
    assert json.loads(result.paths["json"].read_text())["project"] == "test-project"
    assert fake_target.STATE["calls"] == 60 * 3 * 2
    assert report["judge"]["calls"] == fake_judge.STATE["calls"] > 0
    names = [m["name"] for m in report["metrics"]]
    assert names == ["correctness", "format_compliance", "quality", "faithfulness"]
    assert report["pairwise"]["n"] == 20
    assert report["cost"]["candidate"]["per_query_usd"] < report["cost"]["baseline"]["per_query_usd"]
    assert report["eval_cost"]["target_calls"] == 360
    assert {s["tag"] for s in report["slices"]} == {"billing", "shipping", "account", "returns", "legal"}
    assert next(s for s in report["slices"] if s["tag"] == "legal")["directional"] is True


def test_second_run_is_served_from_cache(tmp_path, no_network):
    _run(tmp_path, use_judge=True)
    calls, judge_calls = fake_target.STATE["calls"], fake_judge.STATE["calls"]
    result = _run(tmp_path, use_judge=True)
    assert fake_target.STATE["calls"] == calls
    assert fake_judge.STATE["calls"] == judge_calls
    assert result.report["eval_cost"]["target_calls"] == 0
    assert all(r["cached"] == 60 for arm in result.report["arms"].values() for r in arm["runs"])


def test_changing_the_judge_rescores_without_rerunning(tmp_path, no_network):
    _run(tmp_path, use_judge=True)
    calls, judge_calls = fake_target.STATE["calls"], fake_judge.STATE["calls"]
    _run(tmp_path, use_judge=True, judge={"model": "fake-judge-2025-10", "temperature": 0,
                                      "client": "fake_judge:complete", "max_tokens": 2048})
    assert fake_target.STATE["calls"] == calls
    assert fake_judge.STATE["calls"] > judge_calls


def test_only_the_candidate_runs_when_the_baseline_is_cached(tmp_path, no_network):
    _run(tmp_path)
    before = fake_target.STATE["calls"]
    result = _run(tmp_path, candidate="fake-verbose")
    assert fake_target.STATE["calls"] - before == 60 * 3
    assert result.report["arms"]["baseline"]["runs"][0]["executed"] == 0


def test_judge_drift_is_fatal(tmp_path, no_network):
    fake_judge.STATE["answer_model"] = "fake-judge-2026-01"
    with pytest.raises(JudgeDriftError):
        _run(tmp_path, use_judge=True)


def test_behavior_flags(tmp_path, no_network):
    (tmp_path / "v").mkdir()
    verbose = _run(tmp_path / "v", candidate="fake-verbose")
    assert verbose.report["verdict"] == "ship_with_caveats"
    assert any(r.startswith("answer length +") for r in verbose.report["verdict_reasons"])

    (tmp_path / "r").mkdir()
    refuser = _run(tmp_path / "r", candidate="fake-refuser", n=120)
    assert refuser.report["behavior"]["changes"]["refusal_rate_pts"] > 10
    assert any(r.startswith("refusal rate") for r in refuser.report["verdict_reasons"] + refuser.report["warnings"])

    (tmp_path / "p").mkdir()
    pricey = _run(tmp_path / "p", candidate="fake-pricey")
    assert pricey.report["cost"]["change_pct"] > 15
    assert any(r.startswith("cost per query") for r in pricey.report["verdict_reasons"])


def test_target_errors_are_counted_not_hidden(tmp_path, no_network):
    result = _run(tmp_path, candidate="fake-broken")
    report = result.report
    assert report["behavior"]["candidate"]["error_rate"] > 0.2
    reasons = report["verdict_reasons"] + report["warnings"]
    assert any(r.startswith("target error rate") for r in reasons)
    assert any("target calls failed" in c for c in report["caveats"])


def test_failed_calls_are_not_cached_and_are_retried(tmp_path, no_network):
    _run(tmp_path, candidate="fake-broken")
    before = fake_target.STATE["calls"]
    _run(tmp_path, candidate="fake-broken")
    failing = sum(1 for i in range(60) if fake_target.stable(f"q0-{i:04d}", "broken") < 0.3)
    assert fake_target.STATE["calls"] - before == failing * 3


def test_langsmith_is_off_for_local_datasets(tmp_path, no_network):
    cfg = load_config(write_config(tmp_path, base_config(tmp_path)))
    from upgrade_report.dataset import load_dataset

    ds = load_dataset(cfg.dataset, cfg.base_dir)
    assert isinstance(make_sink(None, ds, "p"), NullSink)


def test_identical_candidate_is_trivially_within_noise(tmp_path, no_network):
    result = _run(tmp_path, candidate="fake-good-a")
    assert fake_target.STATE["calls"] == 60 * 3
    assert all(m["delta"] == 0 for m in result.report["metrics"])
    assert any("identical to the baseline" in c for c in result.report["caveats"])


def test_curated_client_examples(tmp_path, no_network):
    result = _run(tmp_path, report={"formats": ["html"], "client_examples": ["q0-0003", "q0-0005"]})
    html = result.paths["html"].read_text()
    assert "What is the value for item 3?" in html and "What is the value for item 5?" in html
    assert "q0-0003" not in html
