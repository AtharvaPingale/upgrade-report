"""Records produced by running and scoring arms."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .usage import Usage


@dataclass
class RunRecord:
    """One target call: one example, one arm, one repeat."""

    example_id: str
    output: dict[str, Any] | None
    error: str | None = None
    latency_s: float | None = None
    usage: Usage | None = None
    run_id: str | None = None
    trace_url: str | None = None
    attempts: int = 1
    # Set when the output breaks the target contract (e.g. no string "answer").
    contract_error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.output is not None

    @property
    def answer(self) -> str:
        if not self.output:
            return ""
        answer = self.output.get("answer")
        return answer if isinstance(answer, str) else ""

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "RunRecord":
        return cls(**data)


@dataclass
class ArmRun:
    """All records for one (arm, repeat): one LangSmith experiment."""

    arm: str
    repeat: int
    key: str
    key_parts: dict[str, Any]
    records: dict[str, RunRecord] = field(default_factory=dict)
    cached_ids: set[str] = field(default_factory=set)
    experiment: str | None = None
    experiment_url: str | None = None

    @property
    def label(self) -> str:
        return f"{self.arm}#r{self.repeat}"

    def executed_records(self) -> list[RunRecord]:
        return [r for eid, r in self.records.items() if eid not in self.cached_ids]


# example_id -> metric -> {"score": float | None, "kind": "binary" | "continuous", "comment": str | None}
ExampleScores = dict[str, dict[str, Any]]
