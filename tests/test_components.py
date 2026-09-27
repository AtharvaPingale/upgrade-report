"""Config validation, runner, scoring and registry behavior."""

import time
from types import SimpleNamespace

import pytest
from conftest import base_config

from upgrade_report import judges
from upgrade_report.cache import Cache, run_key
from upgrade_report.config import JudgeConfig, parse_config
from upgrade_report.dataset import Example
from upgrade_report.errors import ConfigError, JudgeError
from upgrade_report.judge import Judge, parse_json_reply
from upgrade_report.records import ArmRun, RunRecord
from upgrade_report.runner import Runner, interleave
from upgrade_report.scoring import _normalize, load_evaluator, score_record
from upgrade_report.tracing import NullSink
from upgrade_report.usage import normalize_usage, record_usage


def _cfg(tmp_path, **changes):
    data = base_config(tmp_path)
    for dotted, value in changes.items():
        node = data
        *path, last = dotted.split(".")
        for key in path:
            node = node[key]
        node[last] = value
    return parse_config(data, tmp_path)


@pytest.mark.parametrize("change, message", [
    ({"judge.model": "judge-latest"}, "alias"),
    ({"decision.primary_margin": 0.01}, "must be <= 0"),
    ({"metrics.tracked": ["quality"]}, "more than one role"),
    ({"target": "no_colon"}, "module:function"),
    ({"repeats": 0}, "greater than or equal to 1"),
    ({"surprise": True}, "Extra inputs"),
])
def test_config_validation(tmp_path, change, message):
    with pytest.raises(ConfigError, match=message):
        _cfg(tmp_path, **change)


def test_cache_key_covers_the_plan_tuple():
    k1, parts = run_key("ds1", "model-a", "p1", "t1", 0)
    assert set(parts) == {"dataset_version", "model", "prompt_hash", "target_version", "repeat_index"}
    assert len({k1, run_key("ds2", "model-a", "p1", "t1", 0)[0], run_key("ds1", "model-b", "p1", "t1", 0)[0],
                run_key("ds1", "model-a", "p2", "t1", 0)[0], run_key("ds1", "model-a", "p1", "t2", 0)[0],
                run_key("ds1", "model-a", "p1", "t1", 1)[0]}) == 6


def _examples(n):
    return [Example(id=f"e{i}", inputs={"i": i}) for i in range(n)]


def test_interleaving_alternates_arms_per_example():
    ex = _examples(3)
    base = [ArmRun("baseline", r, f"b{r}", {}) for r in range(2)]
    cand = [ArmRun("candidate", r, f"c{r}", {}) for r in range(2)]
    tasks = interleave({"baseline": [(ar, None, ex) for ar in base], "candidate": [(ar, None, ex) for ar in cand]})
    order = [(t.arm_run.key, t.example.id) for t in tasks]
    assert order[:6] == [("b0", "e0"), ("c0", "e0"), ("c0", "e1"), ("b0", "e1"), ("b0", "e2"), ("c0", "e2")]
    assert len(order) == 12


def test_runner_retries_and_records_contract_errors():
    attempts = {}

    def target(inputs, model, prompt):
        i = inputs["i"]
        attempts[i] = attempts.get(i, 0) + 1
        if i == 0 and attempts[i] == 1:
            raise TimeoutError("transient")
        if i == 1:
            return "not a dict"
        if i == 2:
            return {"text": "no answer key"}
        record_usage(10, 5)
        return {"answer": "ok"}

    arm = SimpleNamespace(model="m", prompt="p")
    ar = ArmRun("baseline", 0, "k", {})
    Runner(target, Cache(None), NullSink(), concurrency=2, retries=1, backoff_s=0.01).run(
        [t for t in interleave({"baseline": [(ar, arm, _examples(4))]})])
    assert ar.records["e0"].ok and ar.records["e0"].attempts == 2
    assert ar.records["e1"].error.startswith("target must return a dict")
    assert ar.records["e2"].ok and ar.records["e2"].contract_error
    assert ar.records["e3"].usage == {"m": {"input_tokens": 10, "output_tokens": 5}}


def test_async_targets_are_supported():
    async def target(inputs, model, prompt):
        return {"answer": "async"}

    ar = ArmRun("baseline", 0, "k", {})
    Runner(target, Cache(None), NullSink(), concurrency=1, retries=0).run(
        interleave({"baseline": [(ar, SimpleNamespace(model="m", prompt="p"), _examples(1))]}))
    assert ar.records["e0"].answer == "async"


def test_usage_formats():
    assert normalize_usage({"prompt_tokens": 3, "completion_tokens": 4}, "m") == {
        "m": {"input_tokens": 3, "output_tokens": 4}}
    assert normalize_usage({"m1": {"input_tokens": 1, "output_tokens": 2}, "emb": {"input_tokens": 9}}, "m") == {
        "m1": {"input_tokens": 1, "output_tokens": 2}, "emb": {"input_tokens": 9, "output_tokens": 0}}
    assert normalize_usage(None, "m") is None


@pytest.mark.parametrize("result, expected", [
    (True, [("f", 1.0, "binary")]),
    (0.25, [("f", 0.25, "continuous")]),
    ({"key": "k", "score": False}, [("k", 0.0, "binary")]),
    ({"results": [{"key": "a", "score": 1}, {"key": "b", "value": 0.5}]}, [("a", 1.0, "continuous"),
                                                                          ("b", 0.5, "continuous")]),
    (SimpleNamespace(key="obj", score=0.75, value=None, comment="c"), [("obj", 0.75, "continuous")]),
    (None, []),
])
def test_evaluator_result_normalization(result, expected):
    assert [(r["key"], r["score"], r["kind"]) for r in _normalize(result, "f")] == expected


def test_evaluator_signatures_and_errors(tmp_path):
    ex = Example(id="e", inputs={"question": "q"}, outputs={"answer": "The answer is 1.", "doc_ids": ["d1", "d2"]})
    rec = RunRecord("e", {"answer": "The answer is 1.", "doc_ids": ["d1", "x", "y"], "citations": ["[1]"],
                          "quality": 0.5})
    evaluators = [load_evaluator(r, tmp_path) for r in (
        "fake_evaluators:correct", "fake_evaluators:legacy_style", "registry:recall@5", "registry:exact_match",
        "registry:refusal_correctness", "fake_evaluators:quality")]
    judge = Judge(JudgeConfig(model="j"), None)
    scores = score_record(evaluators, ex, rec, judge)
    assert scores["correct"]["score"] == 1.0 and scores["correct"]["kind"] == "binary"
    assert scores["has_citation"]["score"] == 1.0
    assert scores["recall@5"]["score"] == 0.5
    assert scores["exact_match"]["score"] == 1.0
    assert scores["refusal_correctness"]["score"] == 1.0
    broken = RunRecord("e", {"answer": "x"})
    assert "quality" in score_record(evaluators, ex, broken, judge)["_errors"]


def test_registry_lookup():
    assert judges.get("recall@3").__name__ == "recall@3"
    with pytest.raises(ConfigError, match="unknown registry judge"):
        judges.get("nope")


def test_judge_reply_parsing():
    assert parse_json_reply('Sure.\n```json\n{"correct": true}\n```') == {"correct": True}
    with pytest.raises(JudgeError):
        parse_json_reply("no json here")


def test_judge_without_client_is_a_config_error():
    with pytest.raises(ConfigError):
        Judge(JudgeConfig(model="j"), None).complete("hi")


def test_cache_does_not_store_failures(tmp_path):
    cache = Cache(tmp_path / "c.sqlite")
    cache.save_record("k", {"model": "m"}, RunRecord("ok", {"answer": "a"}, usage={"m": {"input_tokens": 1,
                                                                                         "output_tokens": 2}}))
    cache.save_record("k", {"model": "m"}, RunRecord("bad", None, error="boom"))
    assert set(cache.load_records("k")) == {"ok"}
    assert cache.average_usage("m") == (1, 2)
    assert Cache(tmp_path / "c.sqlite", read=False).load_records("k") == {}


def test_bootstrap_speed_is_reasonable():
    import numpy as np

    from upgrade_report.stats import paired_bootstrap_ci

    start = time.perf_counter()
    paired_bootstrap_ci(np.random.default_rng(0).normal(size=300), resamples=10_000)
    assert time.perf_counter() - start < 2.0
