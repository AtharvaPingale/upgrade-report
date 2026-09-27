"""Example evaluators. Existing LangSmith evaluators work unchanged.

Supported signatures (LangSmith style), matched by argument name:
    def my_eval(inputs, outputs, reference_outputs) -> bool | float | dict
    def my_eval(run, example) -> dict                  # older style
Add a `judge` argument to get the pinned judge model for LLM-as-judge:
    def my_judge(inputs, outputs, judge) -> float:
        reply = judge.complete("...prompt...")

Return a bool for pass/fail metrics (McNemar, flip lists) and a number for
continuous ones. A dict {"key": ..., "score": ...} names the metric explicitly.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "is", "are", "to", "of", "and", "or", "in", "on", "for", "your", "you", "it", "can", "be"}


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def reference_overlap(outputs: dict, reference_outputs: dict) -> float:
    """Share of the reference answer's key terms that appear in the answer."""
    expected = _terms(reference_outputs.get("answer", ""))
    if not expected:
        return 1.0
    return len(expected & _terms(outputs.get("answer", ""))) / len(expected)


def non_empty_answer(outputs: dict) -> bool:
    return bool(outputs.get("answer", "").strip())
