"""Behavior diffs that aggregate metrics miss, plus the optional pairwise judge."""

from __future__ import annotations

import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .cache import Cache, stable_hash
from .config import BehaviorConfig, BehaviorFlags
from .dataset import Example
from .errors import JudgeDriftError
from .judge import Judge, parse_json_reply
from .judges import question_text, reference_text
from .loader import load_object
from .records import ArmRun, RunRecord
from .stats import paired_bootstrap_ci


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 characters per token). Used for both arms alike."""
    return math.ceil(len(text) / 4) if text else 0


def make_refusal_detector(cfg: BehaviorConfig, base_dir: Path) -> Callable[[str], bool]:
    if cfg.refusal_classifier:
        fn = load_object(cfg.refusal_classifier, base_dir)
        return lambda answer: bool(fn(answer))
    patterns = [re.compile(p, re.IGNORECASE) for p in cfg.refusal_patterns]
    return lambda answer: any(p.search(answer) for p in patterns)


def make_format_check(cfg: BehaviorConfig, base_dir: Path) -> Callable[[RunRecord], bool]:
    validator = load_object(cfg.format_validator, base_dir) if cfg.format_validator else None

    def ok(rec: RunRecord) -> bool:
        if rec.contract_error:
            return False
        if validator is None:
            return True
        try:
            return bool(validator(rec.output))
        except Exception:
            return False

    return ok


def _tool_name(call: Any) -> str:
    if isinstance(call, dict):
        return str(call.get("name") or call.get("tool") or call.get("type") or "unknown")
    return str(getattr(call, "name", call))


def arm_behavior(arm_runs: list[ArmRun], example_ids: list[str], is_refusal: Callable[[str], bool],
                 format_ok: Callable[[RunRecord], bool]) -> dict[str, Any]:
    records = [r for ar in arm_runs for r in ar.records.values()]
    ok = [r for r in records if r.ok]
    lengths = []
    for eid in example_ids:
        per = [estimate_tokens(ar.records[eid].answer) for ar in arm_runs
               if eid in ar.records and ar.records[eid].ok]
        if per:
            lengths.append(float(np.mean(per)))
    with_citations = [len(r.output.get("citations") or []) for r in ok if "citations" in r.output]
    with_tools = [r.output.get("tool_calls") or [] for r in ok if "tool_calls" in r.output]
    tool_counter: Counter[str] = Counter(_tool_name(c) for calls in with_tools for c in calls)
    total_calls = sum(tool_counter.values())
    return {
        "records": len(records),
        "length_median": float(np.median(lengths)) if lengths else None,
        "length_p90": float(np.percentile(lengths, 90)) if lengths else None,
        "refusal_rate": (sum(is_refusal(r.answer) for r in ok) / len(ok)) if ok else None,
        "citations_per_answer": float(np.mean(with_citations)) if with_citations else None,
        "format_failure_rate": (sum(not format_ok(r) for r in ok) / len(ok)) if ok else None,
        "error_rate": (sum(not r.ok for r in records) / len(records)) if records else None,
        "tool_calls_per_answer": float(np.mean([len(c) for c in with_tools])) if with_tools else None,
        "tool_call_distribution": {k: v / total_calls for k, v in tool_counter.most_common(10)} if total_calls else {},
    }


def _pct(new: float | None, old: float | None) -> float | None:
    if new is None or old is None or old == 0:
        return None
    return (new - old) / old * 100


def _pts(new: float | None, old: float | None) -> float | None:
    if new is None or old is None:
        return None
    return (new - old) * 100


def _diff(new: float | None, old: float | None) -> float | None:
    return None if new is None or old is None else new - old


def compare_behavior(base: dict, cand: dict, flags: BehaviorFlags) -> dict[str, Any]:
    changes = {
        "length_median_pct": _pct(cand["length_median"], base["length_median"]),
        "length_p90_pct": _pct(cand["length_p90"], base["length_p90"]),
        "refusal_rate_pts": _pts(cand["refusal_rate"], base["refusal_rate"]),
        "format_failure_pts": _pts(cand["format_failure_rate"], base["format_failure_rate"]),
        "error_rate_pts": _pts(cand["error_rate"], base["error_rate"]),
        "citations_per_answer_delta": _diff(cand["citations_per_answer"], base["citations_per_answer"]),
        "tool_calls_per_answer_delta": _diff(cand["tool_calls_per_answer"], base["tool_calls_per_answer"]),
    }
    raised = []
    checks = [
        ("length", changes["length_median_pct"], flags.length_change_pct, "answer length", "%",
         lambda v: f"Answers are {abs(v):.0f}% {'longer' if v > 0 else 'shorter'} than today."),
        ("refusal_rate", changes["refusal_rate_pts"], flags.refusal_rate_change_pts, "refusal rate", " pts",
         lambda v: f"The proposed configuration declines to answer {'more' if v > 0 else 'less'} often "
                   f"({abs(v):.1f} percentage points)."),
        ("format_failures", changes["format_failure_pts"], flags.format_failure_change_pts, "format failure rate",
         " pts", lambda v: f"{'More' if v > 0 else 'Fewer'} answers break the expected format "
                           f"({abs(v):.1f} percentage points)."),
        ("errors", changes["error_rate_pts"], flags.error_rate_change_pts, "target error rate", " pts",
         lambda v: f"{'More' if v > 0 else 'Fewer'} requests fail outright ({abs(v):.1f} percentage points)."),
    ]
    for name, value, limit, label, unit, plain in checks:
        if value is not None and abs(value) > limit:
            raised.append({"name": name, "message": f"{label} {value:+.0f}{unit} (flag: {limit:g}{unit})",
                           "plain": plain(value)})
    return {"baseline": base, "candidate": cand, "changes": changes, "flags": raised}


_PAIRWISE_SYSTEM = (
    "You compare two answers to the same question. Judge which one better serves the person asking: "
    "correct, grounded, complete, and clear. Length alone is not a virtue. Reply with a single JSON object."
)


def _pairwise_prompt(example: Example, first: str, second: str) -> str:
    ref = reference_text(example.outputs)
    ref_block = f"Reference answer (may be partial):\n{ref}\n\n" if ref else ""
    return (
        f"Question:\n{question_text(example.inputs)}\n\n{ref_block}"
        f"Answer A:\n{first}\n\nAnswer B:\n{second}\n\n"
        'Which answer is better? Reply as {"winner": "A" | "B" | "tie", "reason": "<one sentence>"}.'
    )


def pairwise_preference(examples: list[Example], base_runs: list[ArmRun], cand_runs: list[ArmRun],
                        judge: Judge, cache: Cache, *, sample_size: int, seed: int, resamples: int,
                        concurrency: int) -> dict[str, Any]:
    def first_ok(runs: list[ArmRun], eid: str) -> RunRecord | None:
        for ar in runs:
            rec = ar.records.get(eid)
            if rec is not None and rec.ok:
                return rec
        return None

    eligible = [e for e in examples if first_ok(base_runs, e.id) and first_ok(cand_runs, e.id)]
    if not eligible:
        return {"n": 0, "error": "no example has answers from both arms"}
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(eligible), size=min(sample_size, len(eligible)), replace=False))
    sample = [eligible[i] for i in chosen]
    base_key = base_runs[0].key
    cand_key = cand_runs[0].key

    def judge_one(example: Example) -> float | None:
        base_ans = first_ok(base_runs, example.id).answer
        cand_ans = first_ok(cand_runs, example.id).answer
        # Randomized but reproducible order, so position bias cancels out on average.
        candidate_first = int(stable_hash([seed, example.id], 8), 16) % 2 == 0
        key = stable_hash([base_key, cand_key, judge.key, example.id, candidate_first, base_ans, cand_ans])
        cached = cache.get("pairwise", key)
        if cached is not None:
            return cached["score"]
        first, second = (cand_ans, base_ans) if candidate_first else (base_ans, cand_ans)
        try:
            reply = parse_json_reply(judge.complete(_pairwise_prompt(example, first, second), system=_PAIRWISE_SYSTEM))
        except JudgeDriftError:
            raise
        except Exception:
            return None
        winner = str(reply.get("winner", "")).strip().upper()
        if winner == "TIE":
            score = 0.5
        elif winner in ("A", "B"):
            score = 1.0 if (winner == "A") == candidate_first else 0.0
        else:
            return None
        cache.set("pairwise", key, {"score": score})
        return score

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        scores = list(pool.map(judge_one, sample))
    valid = np.array([s for s in scores if s is not None])
    if len(valid) == 0:
        return {"n": 0, "errors": len(scores), "error": "judge produced no usable comparisons"}
    ci = paired_bootstrap_ci(valid, resamples, seed=seed)
    return {
        "n": int(len(valid)),
        "candidate_wins": int(np.sum(valid == 1.0)),
        "baseline_wins": int(np.sum(valid == 0.0)),
        "ties": int(np.sum(valid == 0.5)),
        "preference_rate": float(valid.mean()),
        "ci": [ci.low, ci.high],
        "errors": int(sum(s is None for s in scores)),
    }
