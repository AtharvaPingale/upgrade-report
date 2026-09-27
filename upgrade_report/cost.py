"""Cost and latency from traced token counts and timings, projected onto real traffic."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import BehaviorFlags, Price
from .records import ArmRun, RunRecord
from .usage import totals


def record_cost(rec: RunRecord, pricing: dict[str, Price]) -> tuple[float | None, set[str]]:
    """Cost of one call; None if it used a model with no price."""
    if not rec.usage:
        return None, set()
    missing = {m for m in rec.usage if m not in pricing}
    if missing:
        return None, missing
    return sum(pricing[m].cost(u["input_tokens"], u["output_tokens"]) for m, u in rec.usage.items()), set()


def arm_cost(arm_runs: list[ArmRun], pricing: dict[str, Price], monthly_queries: int) -> dict[str, Any]:
    ok = [r for ar in arm_runs for r in ar.records.values() if r.ok]
    with_usage = [r for r in ok if r.usage]
    costs, missing = [], set()
    for r in with_usage:
        c, miss = record_cost(r, pricing)
        missing |= miss
        if c is not None:
            costs.append(c)
    per_query = float(np.mean(costs)) if costs and not missing else None
    tokens = [totals(r.usage) for r in with_usage]
    return {
        "per_query_usd": per_query,
        "monthly_usd": per_query * monthly_queries if per_query is not None else None,
        "input_tokens_per_query": float(np.mean([t[0] for t in tokens])) if tokens else None,
        "output_tokens_per_query": float(np.mean([t[1] for t in tokens])) if tokens else None,
        "calls_with_usage": len(with_usage),
        "calls": len(ok),
        "missing_pricing": sorted(missing),
    }


def arm_latency(arm_runs: list[ArmRun]) -> dict[str, Any]:
    lat = [r.latency_s for ar in arm_runs for r in ar.records.values() if r.ok and r.latency_s is not None]
    if not lat:
        return {"p50_s": None, "p95_s": None, "n": 0}
    return {"p50_s": float(np.percentile(lat, 50)), "p95_s": float(np.percentile(lat, 95)), "n": len(lat)}


def _pct(new: float | None, old: float | None) -> float | None:
    if new is None or old is None or old == 0:
        return None
    return (new - old) / old * 100


def compare_cost(base: dict, cand: dict, monthly_queries: int, flags: BehaviorFlags) -> dict[str, Any]:
    change = _pct(cand["per_query_usd"], base["per_query_usd"])
    monthly_delta = (cand["monthly_usd"] - base["monthly_usd"]
                     if cand["monthly_usd"] is not None and base["monthly_usd"] is not None else None)
    raised = []
    if change is not None and change > flags.cost_increase_pct:
        raised.append({"name": "cost", "message": f"cost per query {change:+.0f}% (flag: {flags.cost_increase_pct:g}%)",
                       "plain": f"Each question costs {change:.0f}% more than today."})
    return {"baseline": base, "candidate": cand, "change_pct": change, "monthly_delta_usd": monthly_delta,
            "monthly_queries": monthly_queries, "flags": raised}


def compare_latency(base: dict, cand: dict, flags: BehaviorFlags) -> dict[str, Any]:
    p95 = _pct(cand["p95_s"], base["p95_s"])
    raised = []
    if flags.latency_p95_increase_pct is not None and p95 is not None and p95 > flags.latency_p95_increase_pct:
        limit = flags.latency_p95_increase_pct
        raised.append({"name": "latency", "message": f"p95 latency {p95:+.0f}% (flag: {limit:g}%)",
                       "plain": f"The slowest responses take {p95:.0f}% longer than today."})
    return {"baseline": base, "candidate": cand, "p50_change_pct": _pct(cand["p50_s"], base["p50_s"]),
            "p95_change_pct": p95, "flags": raised}


def eval_run_cost(arm_runs: list[ArmRun], pricing: dict[str, Price], judge_model: str,
                  judge_usage: dict) -> dict[str, Any]:
    """What this invocation spent: target calls actually executed plus judge calls."""
    executed = [r for ar in arm_runs for r in ar.executed_records()]
    target_usd, unpriced = 0.0, set()
    for r in executed:
        c, miss = record_cost(r, pricing)
        unpriced |= miss
        target_usd += c or 0.0
    judge_usd = None
    if judge_model in pricing:
        judge_usd = pricing[judge_model].cost(judge_usage.get("input_tokens", 0), judge_usage.get("output_tokens", 0))
    elif judge_usage.get("calls"):
        unpriced.add(judge_model)
    return {
        "usd": target_usd + (judge_usd or 0.0),
        "target_usd": target_usd,
        "judge_usd": judge_usd,
        "target_calls": len(executed),
        "judge_calls": judge_usage.get("calls", 0),
        "unpriced_models": sorted(unpriced),
    }
