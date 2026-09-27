"""Fake judge client: grades the registry judge prompts with string rules, no network."""

from __future__ import annotations

import json
import re
import threading

from upgrade_report.judge import JudgeResponse

STATE = {"calls": 0, "answer_model": None}
_lock = threading.Lock()


def reset() -> None:
    with _lock:
        STATE["calls"] = 0
        STATE["answer_model"] = None


def _section(prompt: str, start: str, end: str | None) -> str:
    pattern = re.escape(start) + (r"(.*?)" + re.escape(end) if end else r"(.*)")
    m = re.search(pattern, prompt, re.DOTALL)
    return m.group(1).strip() if m else ""


def _grade(prompt: str) -> dict:
    if '"correct"' in prompt:
        ref = _section(prompt, "Reference answer:\n", "\n\nAnswer to grade:")
        ans = _section(prompt, "Answer to grade:\n", "\n\nIs the answer")
        return {"correct": ref.lower().rstrip(".") in ans.lower(), "reason": "string match"}
    if '"supported"' in prompt:
        sources = _section(prompt, "Sources:\n", "\n\nAnswer to grade:")
        ans = _section(prompt, "Answer to grade:\n", "\n\nList the factual")
        claims = [s for s in re.split(r"(?<=\.)\s+", ans) if s.strip()]
        if ans.startswith("I'm sorry"):
            return {"supported": 0, "total": 0}
        supported = sum(1 for c in claims if c.rstrip(".").lower().replace("the answer is ", "") in sources.lower())
        return {"supported": supported, "total": len(claims)}
    if '"winner"' in prompt:
        a = _section(prompt, "Answer A:\n", "\n\nAnswer B:")
        b = _section(prompt, "Answer B:\n", "\n\nWhich answer")
        good_a, good_b = "unclear" not in a, "unclear" not in b
        if good_a == good_b:
            winner = "A" if len(a) < len(b) else "B" if len(b) < len(a) else "tie"
        else:
            winner = "A" if good_a else "B"
        return {"winner": winner}
    if "1-5 scale" in prompt:
        ans = _section(prompt, "Answer to grade:\n", "\n\nRate how")
        return {"score": 2 if "unclear" in ans else 5}
    raise ValueError("fake judge does not recognise this prompt")


def complete(*, model: str, prompt: str, system: str | None, temperature: float | None, max_tokens: int):
    with _lock:
        STATE["calls"] += 1
    text = "```json\n" + json.dumps(_grade(prompt)) + "\n```"
    return JudgeResponse(text=text, model=STATE["answer_model"] or model, input_tokens=len(prompt) // 4,
                         output_tokens=20)
