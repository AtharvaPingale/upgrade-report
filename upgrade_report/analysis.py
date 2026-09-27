"""Per-example aggregation and paired comparison of a candidate arm with the baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import DecisionConfig
from .dataset import Example
from .records import ArmRun, ExampleScores
from .stats import McNemarResult, mcnemar, minimum_detectable_effect, noise_floor, paired_bootstrap_ci

MIN_FLAGGABLE_SLICE = 5


def detect_kinds(score_sets: list[ExampleScores], overrides: dict[str, str]) -> dict[str, str]:
    """A metric is pass/fail when every score it produced came from a bool."""
    seen: dict[str, set[str]] = {}
    for scores in score_sets:
        for per_example in scores.values():
            for metric, entry in per_example.items():
                if not metric.startswith("_"):
                    seen.setdefault(metric, set()).add(entry.get("kind", "continuous"))
    kinds = {m: ("binary" if k == {"binary"} else "continuous") for m, k in seen.items()}
    kinds.update(overrides)
    return kinds


def score_matrix(arm_runs: list[ArmRun], scores: dict[str, ExampleScores], example_ids: list[str],
                 metric: str) -> np.ndarray:
    """(repeats x examples) matrix of scores, NaN where missing."""
    mat = np.full((len(arm_runs), len(example_ids)), np.nan)
    for r, ar in enumerate(arm_runs):
        per_run = scores.get(ar.key, {})
        for i, eid in enumerate(example_ids):
            entry = per_run.get(eid, {}).get(metric)
            if entry is not None and entry.get("score") is not None:
                mat[r, i] = float(entry["score"])
    return mat


def per_example_mean(mat: np.ndarray) -> np.ndarray:
    """Average across repeats; for pass/fail metrics this is the per-example pass rate."""
    counts = np.sum(~np.isnan(mat), axis=0)
    sums = np.nansum(mat, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def _finite(x: float | None) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


@dataclass
class MetricResult:
    name: str
    role: str
    kind: str
    n: int
    baseline: float | None
    candidate: float | None
    delta: float | None
    ci: tuple[float, float] | None
    noise_floor: float | None
    mde: float | None
    status: str  # fail | within_noise | improved | regressed | pass | missing
    blocking: bool = False
    within_noise: bool = False
    flips: McNemarResult | None = None
    reason: str | None = None
    plain_reason: str | None = None

    def to_json(self) -> dict:
        return {
            "name": self.name, "role": self.role, "kind": self.kind, "n": self.n,
            "baseline": self.baseline, "candidate": self.candidate, "delta": self.delta,
            "ci": list(self.ci) if self.ci else None, "noise_floor": self.noise_floor, "mde": self.mde,
            "status": self.status, "blocking": self.blocking, "within_noise": self.within_noise,
            "reason": self.reason,
        }


def compare_metric(name: str, role: str, kind: str, base: np.ndarray, cand: np.ndarray,
                   decision: DecisionConfig, mask: np.ndarray | None = None) -> MetricResult:
    base_ex, cand_ex = per_example_mean(base), per_example_mean(cand)
    paired = ~np.isnan(base_ex) & ~np.isnan(cand_ex)
    if mask is not None:
        paired &= mask
    n = int(paired.sum())
    if n == 0:
        return MetricResult(name, role, kind, 0, None, None, None, None, None, None, "missing",
                            reason=f"{name}: no example has scores in both arms")
    diffs = cand_ex[paired] - base_ex[paired]
    ci = paired_bootstrap_ci(diffs, decision.bootstrap_resamples, seed=decision.seed)
    nf = noise_floor(base[:, paired], decision.bootstrap_resamples, seed=decision.seed).value
    mde = minimum_detectable_effect(diffs)
    delta = ci.mean
    within = nf is not None and abs(delta) <= nf
    flips = mcnemar(base_ex[paired] >= 0.5, cand_ex[paired] >= 0.5) if kind == "binary" else None

    blocking, reason, plain = False, None, None
    label = name.replace("_", " ")
    if role == "primary":
        # Non-inferiority: block when the candidate is worse beyond the noise floor
        # and the CI cannot rule out a drop larger than the margin.
        margin = decision.primary_margin
        if delta < -(nf or 0.0) and ci.low < margin:
            blocking = True
            reason = (f"{name} (primary): CI lower bound {ci.low:+.3f} is below the margin {margin:+.3f} "
                      f"(Δ {delta:+.3f}{_nf_text(nf)})")
            plain = (f"The main measure, {label}, is lower and we cannot rule out a drop larger than the "
                     "agreed tolerance.")
    elif role == "guardrail":
        tol = (nf or 0.0) if decision.guardrail_tolerance == "noise" else float(decision.guardrail_tolerance)
        if ci.high < -tol and not within:
            blocking = True
            what = "noise floor" if decision.guardrail_tolerance == "noise" else "tolerance"
            reason = (f"{name} (guardrail): CI [{ci.low:+.3f}, {ci.high:+.3f}] is entirely below "
                      f"−{tol:.3f} ({what})")
            plain = f"{label.capitalize()} is clearly worse than today."

    if blocking:
        status = "fail"
    elif within:
        status = "within_noise"
    elif ci.low > 0:
        status = "improved"
    elif ci.high < 0:
        status = "regressed"
    else:
        status = "pass"
    return MetricResult(name, role, kind, n, float(base_ex[paired].mean()), float(cand_ex[paired].mean()),
                        float(delta), (ci.low, ci.high), _finite(nf), _finite(mde), status, blocking, within,
                        flips, reason, plain)


def _nf_text(nf: float | None) -> str:
    return f", noise floor {nf:.3f}" if nf is not None else ""


@dataclass
class SliceResult:
    tag: str
    n: int
    directional: bool
    metrics: list[MetricResult]
    regressions: list[str] = field(default_factory=list)
    plain_regressions: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"tag": self.tag, "n": self.n, "directional": self.directional,
                "metrics": [m.to_json() for m in self.metrics], "regressions": self.regressions}


def _slice_regression(m: MetricResult, decision: DecisionConfig) -> bool:
    if m.n < MIN_FLAGGABLE_SLICE or m.ci is None or m.delta is None:
        return False
    nf = m.noise_floor or 0.0
    if m.role == "primary":
        return m.delta < decision.primary_margin and m.ci[1] < -nf
    tol = nf if decision.guardrail_tolerance == "noise" else float(decision.guardrail_tolerance)
    return m.ci[1] < -tol and not m.within_noise


def compare_slices(examples: list[Example], slice_by: str, metrics: list[tuple[str, str, str]],
                   base: dict[str, np.ndarray], cand: dict[str, np.ndarray],
                   decision: DecisionConfig) -> list[SliceResult]:
    tags: dict[str, list[int]] = {}
    for i, ex in enumerate(examples):
        for tag in ex.tags(slice_by):
            tags.setdefault(tag, []).append(i)
    out = []
    for tag in sorted(tags):
        mask = np.zeros(len(examples), dtype=bool)
        mask[tags[tag]] = True
        results = [compare_metric(name, role, kind, base[name], cand[name], decision, mask)
                   for name, role, kind in metrics if name in base and name in cand]
        n = len(tags[tag])
        directional = n < decision.min_slice_size
        sr = SliceResult(tag, n, directional, results)
        for m in results:
            if _slice_regression(m, decision):
                label = f"slice '{tag}' (n={n}{', directional only' if directional else ''})"
                sr.regressions.append(
                    f"{label}: {m.name} Δ {m.delta:+.3f}, CI [{m.ci[0]:+.3f}, {m.ci[1]:+.3f}]"
                )
                sr.plain_regressions.append(
                    f"{m.name.replace('_', ' ').capitalize()} drops for questions about '{tag}' ({n} test questions)."
                )
        out.append(sr)
    return out


def _representative(runs: list[ArmRun], mat: np.ndarray, i: int, binary: bool, eid: str):
    """The repeat whose score best matches the arm's per-example result (the majority
    outcome for pass/fail metrics, the mean otherwise), so the shown answer is typical."""
    col = mat[:, i]
    candidates = [r for r, ar in enumerate(runs) if not np.isnan(col[r]) and eid in ar.records]
    if not candidates:
        return None
    mean = float(np.nanmean(col))
    target = (1.0 if mean >= 0.5 else 0.0) if binary else mean
    best = min(candidates, key=lambda r: abs(col[r] - target))
    return runs[best].records[eid]


def notable_examples(examples: list[Example], results: list[MetricResult], base: dict[str, np.ndarray],
                     cand: dict[str, np.ndarray], base_runs: list[ArmRun], cand_runs: list[ArmRun],
                     max_per_direction: int, slice_by: str) -> list[dict[str, Any]]:
    """Flipped pass/fail cases (pass→fail first) and the largest continuous changes."""
    out: list[dict[str, Any]] = []
    for m in results:
        if m.role == "tracked" or m.n == 0:
            continue
        b, c = per_example_mean(base[m.name]), per_example_mean(cand[m.name])
        paired = ~np.isnan(b) & ~np.isnan(c)
        idx = np.flatnonzero(paired)
        if m.kind == "binary":
            regressed = [i for i in idx if b[i] >= 0.5 > c[i]]
            improved = [i for i in idx if c[i] >= 0.5 > b[i]]
        else:
            threshold = m.noise_floor or 0.0
            d = c - b
            regressed = sorted((i for i in idx if d[i] < -threshold and d[i] < 0), key=lambda i: d[i])
            improved = sorted((i for i in idx if d[i] > threshold and d[i] > 0), key=lambda i: -d[i])
        for direction, chosen in (("regressed", regressed), ("improved", improved)):
            for i in chosen[:max_per_direction]:
                ex = examples[i]
                binary = m.kind == "binary"
                brec = _representative(base_runs, base[m.name], i, binary, ex.id)
                crec = _representative(cand_runs, cand[m.name], i, binary, ex.id)
                out.append({
                    "example_id": ex.id, "metric": m.name, "role": m.role, "kind": m.kind,
                    "direction": direction, "baseline_score": float(b[i]), "candidate_score": float(c[i]),
                    "inputs": ex.inputs, "reference": ex.outputs, "tags": ex.tags(slice_by),
                    "baseline_answer": brec.answer if brec else None,
                    "candidate_answer": crec.answer if crec else None,
                    "baseline_trace": brec.trace_url if brec else None,
                    "candidate_trace": crec.trace_url if crec else None,
                    "total": len(chosen),
                })
    return out
