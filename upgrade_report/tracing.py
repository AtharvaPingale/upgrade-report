"""LangSmith experiments: one tracing project per arm per repeat.

Runs are traced with `reference_example_id`, so each (arm, repeat) shows up as
an experiment on the dataset in the LangSmith UI, with evaluator scores as
feedback. The runner calls the target itself (instead of `langsmith.evaluate`)
so that baseline and candidate calls can be interleaved.
"""

from __future__ import annotations

import contextlib
import os
import time
import uuid
from typing import Any, Callable, Iterator

from .dataset import Dataset, Example
from .records import ArmRun


class NullSink:
    enabled = False

    def start_experiment(self, arm_run: ArmRun, metadata: dict) -> None:
        return None

    @contextlib.contextmanager
    def trace(self, arm_run: ArmRun, example: Example, metadata: dict) -> Iterator["_Trace"]:
        yield _Trace(None, None, lambda outputs: None)

    def log_scores(self, arm_run: ArmRun, run_id: str | None, scores: dict) -> None:
        return None

    def token_usage(self, arm_run: ArmRun, run_ids: list[str]) -> dict[str, tuple[int, int]]:
        return {}

    def flush(self) -> None:
        return None


class _Trace:
    def __init__(self, run_id: str | None, url: str | None, end: Callable[[Any], None]):
        self.run_id = run_id
        self.url = url
        self._end = end

    def end(self, outputs: Any) -> None:
        self._end(outputs)


class LangSmithSink:
    enabled = True

    def __init__(self, dataset: Dataset, prefix: str):
        from langsmith import Client

        self.client = Client()
        self.dataset = dataset
        self.prefix = prefix
        self._projects: dict[str, Any] = {}
        self.feedback_errors = 0

    def start_experiment(self, arm_run: ArmRun, metadata: dict) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = f"{self.prefix}:{arm_run.arm}:r{arm_run.repeat}:{arm_run.key[:8]}:{stamp}"
        project = self.client.create_project(
            name,
            reference_dataset_id=self.dataset.langsmith_id,
            metadata={**metadata, "tool": "upgrade-report", "run_key": arm_run.key},
        )
        self._projects[arm_run.key] = project
        arm_run.experiment = name
        arm_run.experiment_url = getattr(project, "url", None)

    @contextlib.contextmanager
    def trace(self, arm_run: ArmRun, example: Example, metadata: dict) -> Iterator[_Trace]:
        import langsmith as ls

        run_id = str(uuid.uuid4())
        project_url = arm_run.experiment_url
        url = f"{project_url}/r/{run_id}?poll=true" if project_url else None
        ref_id = example.id if self.dataset.source == "langsmith" else None
        with ls.tracing_context(enabled=True, client=self.client, project_name=arm_run.experiment):
            with ls.trace(
                "upgrade-report.target",
                "chain",
                inputs=example.inputs,
                project_name=arm_run.experiment,
                client=self.client,
                run_id=run_id,
                reference_example_id=ref_id,
                metadata=metadata,
            ) as rt:
                def end(outputs: Any) -> None:
                    rt.end(outputs=outputs if isinstance(outputs, dict) else {"output": outputs})

                yield _Trace(run_id, url, end)

    def log_scores(self, arm_run: ArmRun, run_id: str | None, scores: dict) -> None:
        """Mirror scores to LangSmith as feedback. Best effort: the report never depends on it."""
        if not run_id or self.feedback_errors >= 3:
            return
        for metric, entry in scores.items():
            if metric.startswith("_") or entry.get("score") is None:
                continue
            try:
                self.client.create_feedback(run_id, key=metric, score=entry["score"], comment=entry.get("comment"),
                                            stop_after_attempt=3)
            except Exception:
                self.feedback_errors += 1
                return

    def token_usage(self, arm_run: ArmRun, run_ids: list[str]) -> dict[str, tuple[int, int]]:
        """Token counts LangSmith aggregated onto root runs (best effort; ingestion lags)."""
        if not run_ids or not arm_run.experiment:
            return {}
        self.flush()
        out: dict[str, tuple[int, int]] = {}
        for attempt in range(3):
            try:
                for run in self.client.list_runs(project_name=arm_run.experiment, is_root=True):
                    if run.prompt_tokens or run.completion_tokens:
                        out[str(run.id)] = (int(run.prompt_tokens or 0), int(run.completion_tokens or 0))
            except Exception:
                return out
            if len(out) >= len(run_ids):
                break
            time.sleep(2 * (attempt + 1))
        return out

    def flush(self) -> None:
        with contextlib.suppress(Exception):
            self.client.flush()


def make_sink(enabled: bool | None, dataset: Dataset, prefix: str):
    has_key = bool(os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY"))
    if enabled is None:
        enabled = dataset.source == "langsmith" and has_key
    if not enabled:
        return NullSink()
    return LangSmithSink(dataset, prefix)
