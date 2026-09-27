"""Execute arms: N repeats per arm, baseline and candidate calls interleaved."""

from __future__ import annotations

import asyncio
import inspect
import random
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from .cache import Cache
from .config import ArmConfig
from .dataset import Example
from .records import ArmRun, RunRecord
from .usage import UsageCollector, activate, deactivate, normalize_usage

Log = Callable[[str], None]


@dataclass
class Task:
    arm_run: ArmRun
    arm: ArmConfig
    example: Example


def interleave(pending: dict[str, list[tuple[ArmRun, ArmConfig, list[Example]]]]) -> list[Task]:
    """Order work so that every arm's k-th pending repeat runs example-by-example
    side by side with the other arms' k-th repeat. A provider slowdown then hits
    all arms at the same time instead of skewing one of them.

    `pending` maps arm name -> [(arm_run, arm_config, examples_to_run), ...].
    """
    arms = list(pending)
    rounds = max((len(v) for v in pending.values()), default=0)
    tasks: list[Task] = []
    for k in range(rounds):
        members = [pending[a][k] for a in arms if k < len(pending[a])]
        by_id = [{e.id: e for e in examples} for _, _, examples in members]
        order: list[str] = []
        seen: set[str] = set()
        for _, _, examples in members:
            for e in examples:
                if e.id not in seen:
                    seen.add(e.id)
                    order.append(e.id)
        n = len(members)
        for i, eid in enumerate(order):
            # rotate which arm goes first so neither is systematically first
            for step in range(n):
                j = (i + step) % n
                if eid in by_id[j]:
                    arm_run, arm_cfg, _ = members[j]
                    tasks.append(Task(arm_run, arm_cfg, by_id[j][eid]))
    return tasks


def _check_contract(output: Any) -> tuple[dict | None, str | None]:
    if not isinstance(output, dict):
        return None, f"target must return a dict, got {type(output).__name__}"
    if not isinstance(output.get("answer"), str):
        return output, "output has no string 'answer'"
    return output, None


def _call_target(target: Callable, example: Example, arm: ArmConfig) -> Any:
    result = target(inputs=example.inputs, model=arm.model, prompt=arm.prompt)
    if inspect.isawaitable(result):
        result = asyncio.run(_await(result))
    return result


async def _await(awaitable):
    return await awaitable


class Runner:
    def __init__(self, target: Callable, cache: Cache, sink, *, concurrency: int, retries: int,
                 backoff_s: float = 1.0, log: Log | None = None):
        self.target = target
        self.cache = cache
        self.sink = sink
        self.concurrency = concurrency
        self.retries = retries
        self.backoff_s = backoff_s
        self.log = log or (lambda msg: None)
        self._lock = threading.Lock()
        self._done = 0

    def _execute(self, task: Task, total: int) -> RunRecord:
        arm_run, arm, example = task.arm_run, task.arm, task.example
        metadata = {"arm": arm_run.arm, "repeat": arm_run.repeat, "model": arm.model, "prompt": arm.prompt}
        record: RunRecord | None = None
        for attempt in range(1, self.retries + 2):
            collector = UsageCollector(arm.model)
            token = activate(collector)
            start = time.perf_counter()
            try:
                with self.sink.trace(arm_run, example, metadata) as trace:
                    raw = _call_target(self.target, example, arm)
                    trace.end(raw)
                latency = time.perf_counter() - start
                output, contract_error = _check_contract(raw)
                if output is None:
                    record = RunRecord(example.id, None, error=contract_error, latency_s=latency,
                                       run_id=trace.run_id, trace_url=trace.url, attempts=attempt)
                    break
                usage = normalize_usage(output.get("usage"), arm.model) or (collector.usage or None)
                record = RunRecord(example.id, output, latency_s=latency, usage=usage, run_id=trace.run_id,
                                   trace_url=trace.url, attempts=attempt, contract_error=contract_error)
                break
            except Exception as exc:  # the target is project code; anything can happen
                err = f"{type(exc).__name__}: {exc}"
                record = RunRecord(example.id, None, error=err, attempts=attempt)
                if attempt <= self.retries:
                    time.sleep(self.backoff_s * 2 ** (attempt - 1) * (1 + random.random() * 0.25))
                    continue
                self.log(f"  {arm_run.label} {example.id}: {err}\n{traceback.format_exc(limit=3)}")
            finally:
                deactivate(token)
        assert record is not None
        self.cache.save_record(arm_run.key, arm_run.key_parts, record)
        with self._lock:
            arm_run.records[example.id] = record
            self._done += 1
            step = max(1, total // 10)
            if self._done % step == 0 or self._done == total:
                self.log(f"  target calls: {self._done}/{total}")
        return record

    def run(self, tasks: list[Task]) -> None:
        if not tasks:
            return
        self._done = 0
        total = len(tasks)
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            # The pool starts tasks in submission order, which preserves interleaving.
            futures = [pool.submit(self._execute, t, total) for t in tasks]
            for f in futures:
                f.result()
