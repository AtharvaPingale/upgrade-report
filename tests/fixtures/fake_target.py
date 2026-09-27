"""Fake target for tests: deterministic per-example behavior plus injected per-call noise.

Behavior is selected by substrings of the model name:
  regress10  a fixed 10% of examples always get a wrong answer
  verbose    answers are much longer
  refuser    a fixed 20% of examples get a refusal
  pricey     reports twice the output tokens
  broken     raises on a fixed 30% of examples
"""

from __future__ import annotations

import hashlib
import random
import threading

STATE = {"calls": 0}
_lock = threading.Lock()
_rng = random.Random(0)


def reset(seed: int = 0) -> None:
    with _lock:
        STATE["calls"] = 0
        _rng.seed(seed)


def stable(*parts: object) -> float:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _draw() -> tuple[float, float]:
    with _lock:
        return _rng.random(), _rng.gauss(0.0, 1.0)


def regressed_example(qid: str) -> bool:
    return stable(qid, "regress") < 0.10


def run(inputs: dict, model: str, prompt: str) -> dict:
    with _lock:
        STATE["calls"] += 1
    qid, truth = inputs["qid"], inputs["truth"]
    difficulty = stable(qid, "difficulty")
    u, z = _draw()
    if "broken" in model and stable(qid, "broken") < 0.3:
        raise RuntimeError("simulated provider error")

    p_correct = 0.97 - 0.3 * difficulty
    quality = 0.85 - 0.3 * difficulty + 0.08 * z
    if "regress10" in model and regressed_example(qid):
        p_correct, quality = 0.0, quality - 0.4
    correct = u < p_correct
    answer = f"The answer is {truth}." if correct else "The answer is unclear from the sources."
    if "verbose" in model:
        answer += " " + "Here is some additional background that may be useful. " * 12
    if "refuser" in model and stable(qid, "refuse") < 0.2:
        answer = "I'm sorry, but I can't help with that."
    out_tokens = max(1, len(answer) // 4) * (2 if "pricey" in model else 1)
    return {
        "answer": answer,
        "quality": max(0.0, min(1.0, quality)),
        "contexts": [f"Reference note: the answer is {truth}."],
        "doc_ids": [f"doc-{qid}", "doc-other"],
        "citations": ["[1]"] if correct else [],
        "usage": {"input_tokens": 1200, "output_tokens": out_tokens},
    }
