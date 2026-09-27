"""Decision rules: Don't ship, Ship with caveats, Ship (evaluated in that order)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .analysis import MetricResult, SliceResult
from .config import DecisionConfig

SHIP = "ship"
CAVEATS = "ship_with_caveats"
DONT_SHIP = "dont_ship"

LABELS = {SHIP: "Ship", CAVEATS: "Ship with caveats", DONT_SHIP: "Don't ship"}
EXIT_CODES = {SHIP: 0, CAVEATS: 10, DONT_SHIP: 20}
_RANK = {SHIP: 0, CAVEATS: 1, DONT_SHIP: 2}


@dataclass
class Verdict:
    verdict: str
    reasons: list[str]  # what drove the verdict
    client_reasons: list[str]  # the same, in plain language for the client report
    warnings: list[str] = field(default_factory=list)  # caveat-level findings behind a Don't ship

    @property
    def label(self) -> str:
        return LABELS[self.verdict]


class _Reasons:
    def __init__(self) -> None:
        self.technical: list[str] = []
        self.plain: list[str] = []

    def add(self, technical: str, plain: str) -> None:
        self.technical.append(technical)
        self.plain.append(plain)

    def __bool__(self) -> bool:
        return bool(self.technical)


def decide(metrics: list[MetricResult], slices: list[SliceResult], behavior: dict, cost: dict, latency: dict,
           pairwise: dict | None, decision: DecisionConfig) -> Verdict:
    margin = decision.primary_margin
    primary = next(m for m in metrics if m.role == "primary")
    primary_label = primary.name.replace("_", " ")

    blocking = _Reasons()
    for m in metrics:
        if m.blocking and m.reason:
            blocking.add(m.reason, m.plain_reason or m.reason)
    if primary.status == "missing":
        blocking.add(f"{primary.name} (primary): no example has scores in both arms",
                     "The main quality measure could not be computed.")

    caveats = _Reasons()
    for flag in [*behavior.get("flags", []), *cost.get("flags", []), *latency.get("flags", [])]:
        caveats.add(flag["message"], flag.get("plain", flag["message"]))
    for s in slices:
        for technical, plain in zip(s.regressions, s.plain_regressions, strict=True):
            caveats.add(technical, plain)
    if primary.mde is not None and primary.mde > abs(margin):
        caveats.add(
            f"MDE {primary.mde:.3f} for {primary.name} is larger than the margin {abs(margin):.3f}: "
            "this dataset is too small to rule out a regression of the size you care about",
            f"The test set is too small to rule out a drop in {primary_label} of the size that matters; "
            "more test questions would make the conclusion firm.",
        )
    elif not primary.blocking and primary.ci is not None and primary.ci[0] < margin:
        caveats.add(
            f"non-inferiority not established for {primary.name}: CI lower bound {primary.ci[0]:+.3f} is below "
            f"the margin {margin:+.3f}, although the difference is within noise",
            f"We cannot yet rule out a meaningful drop in {primary_label}.",
        )
    if primary.noise_floor is not None and primary.noise_floor > abs(margin):
        caveats.add(
            f"run-to-run noise on {primary.name} ({primary.noise_floor:.3f}) is larger than the margin "
            f"{abs(margin):.3f}: add repeats or examples",
            f"Scores for {primary_label} vary too much between identical runs to judge small differences.",
        )
    for m in metrics:
        if m.role == "guardrail" and m.status == "missing":
            caveats.add(f"guardrail {m.name} has no paired scores",
                        f"{m.name.replace('_', ' ').capitalize()} could not be measured.")
    if pairwise and pairwise.get("n") and pairwise["ci"][1] < 0.5:
        caveats.add(
            f"pairwise judge prefers the baseline: candidate preferred {pairwise['preference_rate']:.0%} "
            f"(95% CI {pairwise['ci'][0]:.0%}–{pairwise['ci'][1]:.0%})",
            "In side-by-side comparisons, the grader preferred today's answers more often.",
        )

    if blocking:
        return Verdict(DONT_SHIP, blocking.technical, blocking.plain, caveats.technical)
    if caveats:
        return Verdict(CAVEATS, caveats.technical, caveats.plain)
    return Verdict(SHIP, [f"{primary.name} is non-inferior and no guardrail, behavior, cost or slice flag fired"],
                   [f"No meaningful drop in {primary_label}, and no other measure raised a concern."])


def recommend(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    """Best candidate arm: best verdict, then fewer reasons against it, then larger
    primary delta, then cheaper."""

    def sort_key(c: dict) -> tuple:
        primary = next((m for m in c["metrics"] if m["role"] == "primary"), {})
        delta = primary.get("delta")
        cost = (c.get("cost") or {}).get("candidate", {}).get("per_query_usd")
        concerns = 0 if c["verdict"] == SHIP else len(c["verdict_reasons"]) + len(c.get("warnings", []))
        return (_RANK[c["verdict"]], concerns, -(delta if delta is not None else float("-inf")),
                cost if cost is not None else float("inf"))

    return sorted(comparisons, key=sort_key)[0]
