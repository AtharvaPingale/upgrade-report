"""Shared judge registry.

Reference evaluators as `registry:<name>` in config. Other packages can add
judges through the `upgrade_report.judges` entry-point group; each entry point
must resolve to an evaluator callable, and its name is the registry name.

Evaluators follow LangSmith's argument names (`inputs`, `outputs`,
`reference_outputs`, `run`, `example`) and may also take `judge`, the pinned
judge for this run.
"""

from __future__ import annotations

import json
import re
from importlib.metadata import entry_points
from typing import Any, Callable

from ..config import DEFAULT_REFUSAL_PATTERNS
from ..errors import ConfigError
from ..judge import Judge, parse_json_reply

REGISTRY: dict[str, Callable[..., Any]] = {}

DESCRIPTIONS: dict[str, str] = {}


def register(name: str, description: str = "") -> Callable:
    def deco(fn: Callable) -> Callable:
        REGISTRY[name] = fn
        if description:
            DESCRIPTIONS[name] = description
        return fn

    return deco


def question_text(inputs: dict) -> str:
    for key in ("question", "input", "query", "prompt", "message", "text"):
        if isinstance(inputs.get(key), str):
            return inputs[key]
    return json.dumps(inputs, ensure_ascii=False, default=str)


def reference_text(reference_outputs: dict | None) -> str | None:
    if not reference_outputs:
        return None
    for key in ("answer", "output", "expected", "reference"):
        if isinstance(reference_outputs.get(key), str):
            return reference_outputs[key]
    return json.dumps(reference_outputs, ensure_ascii=False, default=str)


def _contexts(outputs: dict) -> list[str]:
    raw = outputs.get("contexts", outputs.get("context"))
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    return [c if isinstance(c, str) else json.dumps(c, default=str) for c in raw]


_GRADER_SYSTEM = (
    "You grade answers produced by an AI system. Judge only what is asked. "
    "Reply with a single JSON object and nothing else."
)


@register("correctness", "Whether the answer agrees with the reference answer.")
def correctness(inputs: dict, outputs: dict, reference_outputs: dict | None, judge: Judge) -> bool:
    reference = reference_text(reference_outputs)
    if reference is None:
        raise ValueError("correctness needs a reference answer in the example outputs")
    prompt = (
        f"Question:\n{question_text(inputs)}\n\nReference answer:\n{reference}\n\n"
        f"Answer to grade:\n{outputs.get('answer', '')}\n\n"
        "Is the answer to grade correct, meaning it agrees with the reference on every point the "
        "question asks about? Extra correct detail is fine; contradictions or omissions of the key "
        'facts are not. Reply as {"correct": true|false, "reason": "<one sentence>"}.'
    )
    reply = parse_json_reply(judge.complete(prompt, system=_GRADER_SYSTEM))
    return bool(reply.get("correct"))


@register("faithfulness", "How much of the answer is supported by the retrieved sources (0 to 1).")
def faithfulness(inputs: dict, outputs: dict, judge: Judge) -> float:
    contexts = _contexts(outputs)
    if not contexts:
        raise ValueError("faithfulness needs the retrieved passages in outputs['contexts']")
    sources = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    prompt = (
        f"Question:\n{question_text(inputs)}\n\nSources:\n{sources}\n\n"
        f"Answer to grade:\n{outputs.get('answer', '')}\n\n"
        "List the factual claims in the answer and decide for each whether the sources support it. "
        'Reply as {"supported": <int>, "total": <int>, "reason": "<one sentence>"}. '
        "An answer with no factual claims (for example a refusal) has total 0."
    )
    reply = parse_json_reply(judge.complete(prompt, system=_GRADER_SYSTEM))
    total = int(reply.get("total") or 0)
    supported = int(reply.get("supported") or 0)
    return 1.0 if total == 0 else max(0.0, min(1.0, supported / total))


@register("answer_relevance", "How directly the answer addresses the question (0 to 1).")
def answer_relevance(inputs: dict, outputs: dict, judge: Judge) -> float:
    prompt = (
        f"Question:\n{question_text(inputs)}\n\nAnswer to grade:\n{outputs.get('answer', '')}\n\n"
        "Rate how directly and completely the answer addresses the question on a 1-5 scale "
        "(1 = unrelated, 3 = partially, 5 = fully and directly). Correctness is graded elsewhere. "
        'Reply as {"score": <1-5>, "reason": "<one sentence>"}.'
    )
    reply = parse_json_reply(judge.complete(prompt, system=_GRADER_SYSTEM))
    score = float(reply.get("score"))
    if not 1 <= score <= 5:
        raise ValueError(f"relevance score {score} outside 1-5")
    return (score - 1) / 4


@register("exact_match", "Whether the answer matches the reference exactly (ignoring case and whitespace).")
def exact_match(outputs: dict, reference_outputs: dict | None) -> bool:
    reference = reference_text(reference_outputs)
    if reference is None:
        raise ValueError("exact_match needs a reference answer")
    norm = lambda s: " ".join(str(s).lower().split())  # noqa: E731
    return norm(outputs.get("answer", "")) == norm(reference)


def _is_refusal(answer: str) -> bool:
    return any(re.search(p, answer, re.IGNORECASE) for p in DEFAULT_REFUSAL_PATTERNS)


@register("refusal_correctness", "Whether the system refused exactly when it should have.")
def refusal_correctness(outputs: dict, reference_outputs: dict | None) -> bool:
    should_refuse = bool((reference_outputs or {}).get("should_refuse", False))
    return _is_refusal(outputs.get("answer", "")) == should_refuse


def _recall_at(k: int) -> Callable[..., float]:
    def recall(outputs: dict, reference_outputs: dict | None) -> float:
        expected = set(map(str, (reference_outputs or {}).get("doc_ids") or []))
        if not expected:
            raise ValueError(f"recall@{k} needs reference doc_ids")
        retrieved = [str(d) for d in (outputs.get("doc_ids") or [])[:k]]
        return len(expected.intersection(retrieved)) / len(expected)

    recall.__name__ = f"recall@{k}"
    return recall


_RECALL = re.compile(r"^recall@(\d+)$")


def get(name: str) -> Callable[..., Any]:
    if name in REGISTRY:
        return REGISTRY[name]
    if m := _RECALL.match(name):
        return _recall_at(int(m.group(1)))
    for ep in entry_points(group="upgrade_report.judges"):
        if ep.name == name:
            return ep.load()
    known = sorted([*REGISTRY, "recall@<k>"])
    raise ConfigError(f"unknown registry judge {name!r}; known: {', '.join(known)}")


def description(name: str) -> str | None:
    if name in DESCRIPTIONS:
        return DESCRIPTIONS[name]
    if m := _RECALL.match(name):
        return f"Share of the relevant documents found in the top {m.group(1)} retrieved."
    return None
