"""Token usage capture for target calls.

A target can report usage in three ways, checked in this order:

1. a `usage` key in its return value, either `{"input_tokens": .., "output_tokens": ..}`
   or `{"<model>": {"input_tokens": .., "output_tokens": ..}, ...}`;
2. calls to `upgrade_report.record_usage(...)` from the code the target runs
   (same thread as the target call);
3. token counts on the LangSmith trace, when tracing is on.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

Usage = dict[str, dict[str, int]]  # model -> {"input_tokens": n, "output_tokens": n}

_collector: ContextVar["UsageCollector | None"] = ContextVar("upgrade_report_usage", default=None)


class UsageCollector:
    def __init__(self, default_model: str):
        self.default_model = default_model
        self.usage: Usage = {}

    def add(self, input_tokens: int, output_tokens: int, model: str | None = None) -> None:
        slot = self.usage.setdefault(model or self.default_model, {"input_tokens": 0, "output_tokens": 0})
        slot["input_tokens"] += int(input_tokens or 0)
        slot["output_tokens"] += int(output_tokens or 0)


def record_usage(input_tokens: int = 0, output_tokens: int = 0, model: str | None = None) -> None:
    """Report tokens spent by an LLM call made inside the target function.

    `model` defaults to the arm's model. A no-op outside an upgrade-report run.
    """
    collector = _collector.get()
    if collector is not None:
        collector.add(input_tokens, output_tokens, model)


def activate(collector: UsageCollector):
    return _collector.set(collector)


def deactivate(token) -> None:
    _collector.reset(token)


def _tokens(entry: dict) -> tuple[int, int] | None:
    inp = entry.get("input_tokens", entry.get("prompt_tokens"))
    out = entry.get("output_tokens", entry.get("completion_tokens"))
    if inp is None and out is None:
        return None
    return int(inp or 0), int(out or 0)


def normalize_usage(raw: Any, default_model: str) -> Usage | None:
    if not isinstance(raw, dict) or not raw:
        return None
    flat = _tokens(raw)
    if flat is not None:
        return {default_model: {"input_tokens": flat[0], "output_tokens": flat[1]}}
    out: Usage = {}
    for model, entry in raw.items():
        if isinstance(entry, dict) and (t := _tokens(entry)) is not None:
            out[str(model)] = {"input_tokens": t[0], "output_tokens": t[1]}
    return out or None


def totals(usage: Usage | None) -> tuple[int, int]:
    if not usage:
        return 0, 0
    return (sum(u["input_tokens"] for u in usage.values()), sum(u["output_tokens"] for u in usage.values()))
