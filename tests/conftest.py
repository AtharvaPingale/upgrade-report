from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest
import yaml

FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(FIXTURES))

import fake_judge  # noqa: E402
import fake_target  # noqa: E402

TAGS = ["billing", "shipping", "account", "returns"]


def make_dataset(path: Path, n: int, small_tag_size: int = 6, seed: int = 0) -> Path:
    rows = []
    for i in range(n):
        qid = f"q{seed}-{i:04d}"
        tags = [TAGS[i % len(TAGS)]]
        if i < small_tag_size:
            tags.append("legal")
        truth = f"value-{i}"
        rows.append({"id": qid, "inputs": {"question": f"What is the value for item {i}?", "qid": qid, "truth": truth},
                     "outputs": {"answer": f"The answer is {truth}.", "doc_ids": [f"doc-{qid}"]},
                     "metadata": {"tags": tags}})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


PRICING = {
    "fake-good-a": {"input_per_mtok": 3.0, "output_per_mtok": 15.0},
    "fake-good-b": {"input_per_mtok": 2.5, "output_per_mtok": 10.0},
    "fake-regress10": {"input_per_mtok": 2.5, "output_per_mtok": 10.0},
    "fake-verbose": {"input_per_mtok": 2.5, "output_per_mtok": 10.0},
    "fake-refuser": {"input_per_mtok": 2.5, "output_per_mtok": 10.0},
    "fake-pricey": {"input_per_mtok": 6.0, "output_per_mtok": 30.0},
    "fake-broken": {"input_per_mtok": 2.5, "output_per_mtok": 10.0},
    "fake-judge-2025-10": {"input_per_mtok": 1.0, "output_per_mtok": 5.0},
}


def base_config(tmp: Path, *, n: int = 60, candidate: str = "fake-good-b", use_judge: bool = False,
                pairwise: bool = False, repeats: int = 3, resamples: int = 2000, **overrides) -> dict:
    make_dataset(tmp / "dataset.jsonl", n)
    (tmp / "pricing.yaml").write_text(yaml.safe_dump(PRICING))
    evaluators = ["fake_evaluators:quality", "fake_evaluators:format_compliance"]
    metrics = {"primary": "quality", "guardrails": ["format_compliance"], "tracked": []}
    if use_judge:
        evaluators += ["registry:correctness", "registry:faithfulness"]
        metrics = {"primary": "correctness", "guardrails": ["format_compliance"],
                   "tracked": ["quality", "faithfulness"]}
    cfg = {
        "project": "test-project",
        "dataset": "dataset.jsonl",
        "target": "fake_target:run",
        "baseline": {"model": "fake-good-a", "prompt": "prompts/answer@v1"},
        "candidate": {"model": candidate, "prompt": "prompts/answer@v1"},
        "evaluators": evaluators,
        "repeats": repeats,
        "concurrency": 4,
        "retries": 0,
        "judge": {"model": "fake-judge-2025-10", "temperature": 0,
                  "client": "fake_judge:complete" if use_judge else None},
        "metrics": metrics,
        "decision": {"primary_margin": -0.05, "min_slice_size": 20, "bootstrap_resamples": resamples},
        "behavior": {"pairwise": {"enabled": pairwise, "sample_size": 20}},
        "traffic": {"monthly_queries": 120000},
        "report": {"formats": ["md", "json", "html"]},
        "langsmith": {"enabled": False},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **value}
        else:
            cfg[key] = value
    return cfg


def write_config(tmp: Path, cfg: dict) -> Path:
    path = tmp / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


@pytest.fixture(autouse=True)
def _reset_fakes():
    fake_target.reset(0)
    fake_judge.reset()
    yield


@pytest.fixture
def no_network(monkeypatch):
    """Fail any attempt to open a network connection."""

    def guard(*args, **kwargs):
        raise AssertionError("network access attempted during a test")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    yield
