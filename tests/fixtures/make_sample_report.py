"""Regenerate tests/fixtures/report_sample.json, the input of the render snapshot tests.

    python tests/fixtures/make_sample_report.py

Volatile fields (timestamps, latencies, run keys) are pinned so snapshots only
change when the report structure or the templates change.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import fake_judge  # noqa: E402
import fake_target  # noqa: E402
from conftest import base_config, write_config  # noqa: E402

from upgrade_report.config import load_config  # noqa: E402
from upgrade_report.pipeline import RunOptions, run  # noqa: E402

TRACE = "https://smith.langchain.com/o/org-id/projects/p/project-id/r/{}?poll=true"


def main() -> None:
    fake_target.reset(0)
    fake_judge.reset()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg = base_config(
            tmp, n=80, use_judge=True, pairwise=True, candidate="fake-verbose", concurrency=1,
            adapted_candidate={"model": "fake-good-b", "prompt": "prompts/answer@v2"},
            report={"formats": ["json"], "max_examples": 3},
        )
        report = run(load_config(write_config(tmp, cfg)), RunOptions(log=lambda m: None)).report

    report["generated_at"] = "2026-01-15T12:00:00+00:00"
    for c in report["comparisons"]:
        for side in ("baseline", "candidate"):
            c["latency"][side].update({"p50_s": 1.2 if side == "baseline" else 0.9,
                                       "p95_s": 3.4 if side == "baseline" else 2.6})
        c["latency"]["p50_change_pct"], c["latency"]["p95_change_pct"] = -25.0, -23.5
        for ex in c["examples"]:
            ex["baseline_trace"] = TRACE.format(f"b-{ex['example_id']}")
            ex["candidate_trace"] = TRACE.format(f"c-{ex['example_id']}")
    best = next(c for c in report["comparisons"] if c["arm"] == report["recommended_arm"])
    report["latency"], report["examples"] = best["latency"], best["examples"]
    for arm in report["arms"].values():
        for r in arm["runs"]:
            r["experiment"] = f"test-project:r{r['repeat']}"
            r["experiment_url"] = "https://smith.langchain.com/o/org-id/projects/p/exp-id"
    (HERE / "report_sample.json").write_text(json.dumps(report, indent=2) + "\n")
    print(HERE / "report_sample.json")


if __name__ == "__main__":
    main()
