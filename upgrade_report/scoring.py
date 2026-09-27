"""Evaluator orchestration: load evaluators, score every output, cache scores."""

from __future__ import annotations

import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from . import judges
from .cache import Cache, stable_hash
from .dataset import Dataset, Example
from .errors import ConfigError, JudgeDriftError
from .judge import Judge
from .loader import load_object
from .records import ArmRun, ExampleScores, RunRecord

_KNOWN_ARGS = {"inputs", "outputs", "reference_outputs", "run", "example", "judge", "metadata"}


@dataclass
class Evaluator:
    ref: str
    fn: Callable[..., Any]
    name: str
    params: list[str]
    positional_run_example: bool
    uses_judge: bool
    source_hash: str


def _source_hash(obj: Any, ref: str) -> str:
    try:
        src = inspect.getsource(obj if not hasattr(obj, "evaluate_run") else type(obj))
    except (OSError, TypeError):
        src = ref
    return stable_hash([ref, getattr(obj, "__name__", ""), src], 12)


def load_evaluator(ref: str, base_dir: Path) -> Evaluator:
    if ref.startswith("registry:"):
        name = ref.split(":", 1)[1]
        fn = judges.get(name)
    else:
        fn = load_object(ref, base_dir)
        name = getattr(fn, "__name__", None) or ref.rsplit(":", 1)[1]
    target = fn.evaluate_run if hasattr(fn, "evaluate_run") else fn
    if not callable(target):
        raise ConfigError(f"evaluator {ref!r} is not callable")
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"cannot inspect evaluator {ref!r}: {exc}") from exc
    params = [p.name for p in sig.parameters.values() if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    known = [p for p in params if p in _KNOWN_ARGS]
    positional = hasattr(fn, "evaluate_run") or (not known and len(params) == 2)
    if not positional and not known:
        raise ConfigError(
            f"evaluator {ref!r} must take LangSmith-style arguments "
            "(inputs, outputs, reference_outputs, run, example) or (run, example)"
        )
    return Evaluator(ref=ref, fn=fn, name=name, params=known, positional_run_example=positional,
                     uses_judge="judge" in known, source_hash=_source_hash(fn, ref))


def load_evaluators(refs: list[str], base_dir: Path) -> list[Evaluator]:
    if not refs:
        raise ConfigError("no evaluators configured; add at least one under `evaluators`")
    return [load_evaluator(r, base_dir) for r in refs]


def scorer_key(evaluators: list[Evaluator], judge: Judge) -> str:
    return stable_hash({"judge": judge.cfg.describe(), "evaluators": [(e.ref, e.source_hash) for e in evaluators],
                        "v": 1}, 16)


def _normalize(result: Any, default_key: str) -> list[dict]:
    if result is None:
        return []
    if isinstance(result, bool):
        return [{"key": default_key, "score": float(result), "kind": "binary", "comment": None}]
    if isinstance(result, (int, float)):
        return [{"key": default_key, "score": float(result), "kind": "continuous", "comment": None}]
    if isinstance(result, (list, tuple)):
        return [r for item in result for r in _normalize(item, default_key)]
    if not isinstance(result, dict):
        if hasattr(result, "results"):
            return _normalize(list(result.results), default_key)
        if hasattr(result, "key") or hasattr(result, "score"):
            result = {k: getattr(result, k, None) for k in ("key", "score", "value", "comment")}
        else:
            raise TypeError(f"unsupported evaluator result type {type(result).__name__}")
    if "results" in result and isinstance(result["results"], (list, tuple)):
        return [r for item in result["results"] for r in _normalize(item, default_key)]
    key = result.get("key") or default_key
    score = result.get("score")
    if score is None and isinstance(result.get("value"), (bool, int, float)):
        score = result["value"]
    if score is None:
        return []
    kind = "binary" if isinstance(score, bool) else "continuous"
    return [{"key": key, "score": float(score), "kind": kind, "comment": result.get("comment")}]


def _call(ev: Evaluator, example: Example, record: RunRecord, judge: Judge) -> Any:
    run = SimpleNamespace(id=record.run_id, inputs=example.inputs, outputs=record.output,
                          reference_example_id=example.id, error=record.error)
    ex = SimpleNamespace(id=example.id, inputs=example.inputs, outputs=example.outputs, metadata=example.metadata)
    if ev.positional_run_example:
        fn = ev.fn.evaluate_run if hasattr(ev.fn, "evaluate_run") else ev.fn
        return fn(run, ex)
    available = {"inputs": example.inputs, "outputs": record.output, "reference_outputs": example.outputs,
                 "run": run, "example": ex, "judge": judge, "metadata": example.metadata}
    return ev.fn(**{p: available[p] for p in ev.params})


def score_record(evaluators: list[Evaluator], example: Example, record: RunRecord, judge: Judge) -> dict:
    scores: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for ev in evaluators:
        try:
            for item in _normalize(_call(ev, example, record, judge), ev.name):
                scores[item["key"]] = {"score": item["score"], "kind": item["kind"], "comment": item["comment"]}
        except (JudgeDriftError, ConfigError):
            raise
        except Exception as exc:  # evaluator code belongs to the project
            errors[ev.name] = f"{type(exc).__name__}: {exc}"
    if errors:
        scores["_errors"] = errors
    return scores


class Scorer:
    def __init__(self, evaluators: list[Evaluator], judge: Judge, cache: Cache, sink, *, concurrency: int,
                 log: Callable[[str], None] | None = None):
        self.evaluators = evaluators
        self.judge = judge
        self.cache = cache
        self.sink = sink
        self.concurrency = concurrency
        self.log = log or (lambda msg: None)
        self.key = scorer_key(evaluators, judge)
        self.scored_now = 0

    def score(self, arm_runs: list[ArmRun], dataset: Dataset) -> dict[str, ExampleScores]:
        by_id = dataset.by_id()
        out: dict[str, ExampleScores] = {}
        todo: list[tuple[ArmRun, RunRecord]] = []
        for ar in arm_runs:
            cached = self.cache.load_scores(ar.key, self.key)
            out[ar.key] = {}
            for eid, rec in ar.records.items():
                if not rec.ok:
                    continue
                if eid in cached:
                    out[ar.key][eid] = cached[eid]
                else:
                    todo.append((ar, rec))
        if not todo:
            return out
        if any(e.uses_judge for e in self.evaluators) and not self.judge.available:
            raise ConfigError("an evaluator uses the judge but `judge.client` is not configured")
        self.log(f"Scoring {len(todo)} outputs with {len(self.evaluators)} evaluators")
        lock = threading.Lock()
        done = 0

        def work(item: tuple[ArmRun, RunRecord]) -> None:
            nonlocal done
            ar, rec = item
            scores = score_record(self.evaluators, by_id[rec.example_id], rec, self.judge)
            self.cache.save_scores(ar.key, self.key, rec.example_id, scores)
            self.sink.log_scores(ar, rec.run_id, scores)
            with lock:
                out[ar.key][rec.example_id] = scores
                done += 1
                step = max(1, len(todo) // 10)
                if done % step == 0 or done == len(todo):
                    self.log(f"  scored: {done}/{len(todo)}")

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            for f in [pool.submit(work, item) for item in todo]:
                f.result()
        self.scored_now = len(todo)
        return out
