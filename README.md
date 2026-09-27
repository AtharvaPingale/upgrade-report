# upgrade-report

[![CI](https://github.com/atharvapingale/upgrade-report/actions/workflows/ci.yml/badge.svg)](https://github.com/atharvapingale/upgrade-report/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)

Decide whether to move an LLM application to a new model or prompt version.
`upgrade-report` runs your existing eval suite against a **baseline** (what runs
today) and a **candidate**, compares them example by example with paired
statistics, and writes a report with a verdict:

| Verdict | Meaning | Exit code |
|---|---|---|
| **Ship** | Not meaningfully worse on the primary metric, no guardrail regression, nothing flagged | 0 |
| **Ship with caveats** | No blocking regression, but a behavior, cost, slice or dataset-size flag needs a human look | 10 |
| **Don't ship** | The primary metric may have dropped by more than the margin, or a guardrail clearly regressed | 20 |
| error | Tool or config error | 2 |

It answers: *is the candidate at least as good as what we run today, where does
it differ, and what will it cost?*

## Quick start

```bash
pip install "upgrade-report @ git+https://github.com/atharvapingale/upgrade-report"   # [anthropic] extra adds the built-in Claude judge
upgrade-report init                   # config.yaml, pricing.yaml, target stub, sample data, CI workflows
upgrade-report run --dry-run          # validate config, estimate what the run will cost
upgrade-report run                    # writes ./reports/<timestamp>/report.{json,md}
```

The scaffold runs as-is on a sample dataset, so you can see a report before
wiring anything up. Then implement one function:

```python
# upgrade_eval/target.py
def run(inputs: dict, model: str, prompt: str) -> dict:
    """Return at least {"answer": str}. Optionally "doc_ids", "contexts",
    "citations", "tool_calls", "usage" for richer metrics and cost."""
```

and point `dataset` at your LangSmith dataset. See
[docs/getting-started.md](docs/getting-started.md).

## Commands

```bash
upgrade-report init [DIR]                      # scaffold a project
upgrade-report run -c config.yaml              # full run
upgrade-report run --dry-run                   # validate config, estimate eval cost
upgrade-report run --baseline-ref origin/main  # use main's cached baseline (PRs that change code)
upgrade-report render report.json -f html      # client HTML (or -f md)
upgrade-report aa -c config.yaml               # A/A calibration: baseline vs itself, should be Ship
upgrade-report scheduled -c config.yaml        # evaluate newly released models (notify only)
```

## What's in a report

`report.json` is the source of truth; `report.md` is the PR comment and
`report_client.html` the client-facing version.

- Verdict with the reasons that drove it
- Metric table: baseline, candidate, delta, 95% paired-bootstrap CI, noise floor, status
- Per-tag slices (small slices marked *directional only*)
- Flipped cases, pass→fail first, with both answers and LangSmith trace links
- Cost per query and per month at your traffic, latency p50/p95
- Behavior diffs: answer length, refusals, citations, format failures, errors,
  tool calls, and an optional pairwise judge
- Caveats: judge, dataset version, repeats, minimum detectable effect, and what
  the eval run itself cost

## How it decides

Both arms answer the same examples, so every comparison uses per-example
differences. Repeats of the baseline give a free A/A test that sets a **noise
floor**; differences inside it never count as regressions. The primary metric is
tested for **non-inferiority** against a margin (default −0.02): the bar is "not
meaningfully worse", not "strictly better". Details and the evidence that the
statistics are calibrated: [docs/methodology.md](docs/methodology.md).

## Documentation

- [Getting started](docs/getting-started.md): first report in 30 minutes
- [Configuration reference](docs/configuration.md)
- [Methodology](docs/methodology.md): statistics, decision rules, calibration results
- [Reading a report](docs/reading-reports.md): what the verdict does and does not cover
- [CI and scheduled runs](docs/ci.md): GitHub Action, PR comments, caching, model watch
- [Design plan](PLAN.md): the original goals, requirements and milestones

## Development

```bash
uv venv && uv pip install -e '.[dev]'
pytest                      # 92 tests, ~20 s, no network
pytest -m "not slow"        # skip the calibration simulations
ruff check .
UPDATE_SNAPSHOTS=1 pytest tests/test_render.py   # after an intended template change
python tests/fixtures/make_sample_report.py      # regenerate the snapshot input
```

Layout: `upgrade_report/` (package), `templates/project/` (the `init`
scaffold), `.github/workflows/upgrade-report.yml` (reusable workflow),
`tests/` (stats simulations, fake target and judge fixtures, snapshots), `docs/`.

## License

[MIT](LICENSE)
