# Model Upgrade Impact Report

Working name: `upgrade-report`

## 1. Overview

Teams regularly need to decide whether to move an LLM application to a new model or prompt version. Today that decision is usually made by eyeballing a handful of outputs or comparing two averages. Neither tells you whether a difference is real, where it comes from, or what it will cost.

`upgrade-report` runs a project's existing eval suite against a baseline and a candidate configuration, compares them with paired statistics, and produces a report with a verdict: **Ship**, **Ship with caveats**, or **Don't ship**.

The question it answers:

> Is the candidate at least as good as what we run today, where does it differ, and what will it cost?

## 2. Goals and non-goals

### Goals

- Give any team a one-command, repeatable way to evaluate a model or prompt change.
- Separate real differences from run-to-run noise.
- Surface behavior changes that aggregate metrics miss (verbosity, refusals, format).
- Project cost and latency impact at real traffic volume.
- Produce two outputs: an internal report for PRs, and a clean version for clients.
- Plug into existing LangSmith datasets, evaluators, and CI without rewriting them.

### Non-goals

- Building or managing eval datasets. The tool consumes existing datasets.
- Replacing production monitoring or online evaluators.
- Automatically deploying the winning configuration.
- Evaluating the judge model itself. Judges are treated as fixed, pinned instruments.

## 3. Users

| User | Need |
|---|---|
| Engineer on a project | "Can I merge this model or prompt change?" |
| Tech lead | "Is this upgrade worth the cost and risk across our projects?" |
| Client stakeholder | "Why are you changing the model, and what do we gain?" |
| Non-AI engineers | Run it from a template without understanding the statistics |

## 4. Success criteria

- A new project can produce its first report within 30 minutes of setup.
- A/A runs (baseline vs itself) produce a **Ship** verdict at least 95% of the time. If they don't, the tool is too noisy.
- Synthetic tests with a known injected regression are caught at the expected rate for the dataset size.
- A PR check completes in under 20 minutes on a 300-example dataset with 3 repeats.
- At least two client projects use it for a real upgrade decision within the first quarter.

## 5. How it works

```
config.yaml
    │
    ▼
┌──────────────┐   ┌──────────────┐   ┌─────────────┐   ┌─────────────┐   ┌──────────┐
│ Run arms     │──▶│ Score        │──▶│ Stats +     │──▶│ Verdict     │──▶│ Render   │
│ (baseline,   │   │ (evaluators, │   │ noise floor │   │ engine      │   │ md/html/ │
│  candidate)  │   │  pinned      │   │ + diffs     │   │             │   │ json     │
└──────────────┘   │  judge)      │   └─────────────┘   └─────────────┘   └──────────┘
        │          └──────────────┘
        ▼
  LangSmith experiments (one per arm per repeat), cached by content hash
```

### Pipeline stages

1. **Load config** and validate it with Pydantic. Resolve the dataset version, prompt versions, and model identifiers.
2. **Check cache.** Each arm run is keyed by `(dataset_version, model, prompt_hash, target_version, repeat_index)`. Cached arms are reused. In practice the baseline is almost always cached, so only the candidate runs.
3. **Run arms.** Execute the project's target function on every example, N repeats per arm. Baseline and candidate requests are interleaved so a provider slowdown affects both equally.
4. **Score.** Apply the project's evaluators to each output. The judge model is pinned and must be identical across arms.
5. **Aggregate per example.** Average continuous scores across repeats. For pass/fail metrics, use the per-example pass rate, then the majority pass/fail for McNemar.
6. **Compute statistics.** Noise floor, paired confidence intervals, McNemar, per-tag slices, and minimum detectable effect.
7. **Compute behavior diffs.** Length, refusal rate, citation count, format failures, tool-call patterns, and an optional pairwise judge.
8. **Compute cost and latency.** Take token counts and timings from traces, and project them onto monthly traffic.
9. **Decide the verdict** using the configured decision rules.
10. **Render** `report.json`, which is the source of truth, then `report.md` and optionally `report_client.html`.

## 6. Configuration

```yaml
project: client-x-support-rag
dataset: client-x-golden-v4            # LangSmith dataset name or ID
target: app.eval_entry:run             # module:function, see section 7

baseline:
  model: model-a-2025-06
  prompt: prompts/answer@v12
candidate:
  model: model-b-2026-03
  prompt: prompts/answer@v12
adapted_candidate:                     # optional third arm
  model: model-b-2026-03
  prompt: prompts/answer@v12-b

repeats: 3
concurrency: 8

judge:
  model: judge-pinned-2025-10
  temperature: 0

metrics:
  primary: faithfulness
  guardrails: [refusal_correctness, format_compliance, recall@5]
  tracked: [answer_relevance, citation_accuracy]

decision:
  primary_margin: -0.02                # non-inferiority margin
  guardrail_tolerance: noise           # "noise" or an absolute number
  min_slice_size: 20

behavior_flags:
  length_change_pct: 25
  refusal_rate_change_pts: 3
  cost_increase_pct: 15

traffic:
  monthly_queries: 120000

pricing_file: pricing.yaml

report:
  formats: [md, json]                  # add html for client version
  redact_inputs: false                 # true for the client version
```

### `pricing.yaml`

This file is maintained manually and checked in, because provider pricing changes and must be versioned alongside reports.

```yaml
model-a-2025-06: { input_per_mtok: 3.00, output_per_mtok: 15.00 }
model-b-2026-03: { input_per_mtok: 2.50, output_per_mtok: 10.00 }
```

## 7. Integration contract

Each project exposes one target function. This is the only code a project writes.

```python
def run(inputs: dict, model: str, prompt: str) -> dict:
    """Return at least {"answer": str}. Optionally include
    "doc_ids", "citations", "tool_calls" for richer metrics."""
```

Evaluators are the project's existing LangSmith evaluators, or they can be imported from the shared judge registry. The tool never reaches into application internals beyond this function.

## 8. Statistical methodology

### 8.1 Paired design

Both arms answer the same examples, so every comparison uses per-example differences. This is far more sensitive than comparing two independent means, because question difficulty cancels out.

### 8.2 Noise floor

Repeats of the baseline arm act as a free A/A test. For each metric:

1. Compute the paired difference between baseline repeat pairs (1 vs 2, 1 vs 3, 2 vs 3).
2. Bootstrap the mean of those differences.
3. The noise floor is the 95th percentile of the absolute bootstrapped mean differences.

Any candidate difference smaller than the noise floor is reported as "within noise" and never triggers a regression.

### 8.3 Continuous metrics

Use a paired bootstrap 95% confidence interval on the mean per-example difference (candidate minus baseline), with 10,000 resamples and a fixed seed for reproducibility.

### 8.4 Pass/fail metrics

Use McNemar's test on the discordant pairs: examples that passed in one arm and failed in the other. When there are fewer than 25 discordant pairs, fall back to the exact binomial version. Report both flip counts, since "12 improved, 9 regressed" is more informative than a p-value.

### 8.5 Per-tag slices

Compute the same statistics per tag. Slices with fewer than `min_slice_size` examples are shown but marked **directional only** and are excluded from the verdict. Because many slices are tested, a note in the report warns that some will look significant by chance.

### 8.6 Minimum detectable effect

Report the smallest difference the dataset could reliably detect, given its size and the observed variance. If the MDE is larger than the decision margin, the report says so explicitly: "this dataset is too small to rule out a regression of the size you care about." This prevents false confidence from small eval sets.

## 9. Decision rules

Evaluated in order:

| Verdict | Condition |
|---|---|
| **Don't ship** | The primary metric's CI lower bound falls below `primary_margin`, OR any guardrail's CI sits entirely below the negative noise floor (or the absolute tolerance) |
| **Ship with caveats** | No blocking regression, but any of: a behavior flag triggered, cost increase above threshold, a directional-only slice regression, or MDE larger than the margin |
| **Ship** | None of the above |

The rules are non-inferiority, not superiority. The bar for an upgrade is usually "not meaningfully worse, and cheaper or faster," not "strictly better."

When an adapted-prompt arm is present, the verdict is computed for both candidate arms, and the report recommends the better one.

## 10. Behavior diffs

Computed for every run, independent of the metrics:

- Answer length: median and p90, in tokens.
- Refusal rate, detected by a lightweight classifier or pattern list per project.
- Citations per answer.
- Format or schema failure rate.
- Tool-call counts and distribution (for agentic targets).
- Optional **pairwise judge**: on a random sample of 50 to 100 examples, the pinned judge compares baseline and candidate answers in randomized order and reports the preference rate with a CI.

## 11. Cost and latency

- Per-query cost is computed from input and output token counts in the traces, priced with `pricing.yaml`.
- The monthly projection is per-query cost × `traffic.monthly_queries`.
- Latency is reported as p50 and p95 per arm. Interleaved execution keeps these comparable.
- The eval run's own cost is shown in the caveats section, so teams see what the report cost.

## 12. Report specification

### 12.1 `report.json` (source of truth)

```json
{
  "project": "client-x-support-rag",
  "generated_at": "...",
  "config_hash": "...",
  "verdict": "ship_with_caveats",
  "verdict_reasons": ["answer length +38% (flag: 25%)"],
  "metrics": [
    {
      "name": "faithfulness", "role": "primary",
      "baseline": 0.91, "candidate": 0.92,
      "delta": 0.01, "ci": [-0.005, 0.024],
      "noise_floor": 0.008, "mde": 0.015, "status": "pass"
    }
  ],
  "flips": { "faithfulness": { "improved": 12, "regressed": 9, "p_value": 0.66 } },
  "slices": [],
  "behavior": {},
  "cost": {},
  "latency": {},
  "examples": [],
  "caveats": []
}
```

### 12.2 `report.md` (internal, PR comment)

1. **Verdict banner** with a one-line reason.
2. **Metric table**: baseline, candidate, delta, CI, noise floor, and status for each metric.
3. **Per-tag breakdown**: wins and losses, with small slices marked.
4. **Flipped cases**: pass→fail first, each with the question, both answers, and LangSmith trace links.
5. **Cost and latency**: per query, monthly projection, p50/p95.
6. **Behavior diffs** and the pairwise preference rate.
7. **Caveats**: judge model, dataset version, repeats, MDE warnings, and the eval run's cost.

### 12.3 `report_client.html` (external)

- Verdict, metric table, cost impact, and 3 to 5 curated example comparisons.
- No trace links, no internal IDs, inputs optionally redacted.
- Plain-language explanations of each metric and of what "within noise" means.

## 13. CLI

```bash
upgrade-report init                       # scaffold config.yaml + target stub
upgrade-report run --config config.yaml   # full run, writes ./reports/<timestamp>/
upgrade-report run --dry-run              # validate config, estimate eval cost
upgrade-report render report.json --format html
upgrade-report aa --config config.yaml    # explicit A/A calibration run
```

Exit codes for CI:

| Code | Meaning |
|---|---|
| 0 | Ship |
| 10 | Ship with caveats |
| 20 | Don't ship |
| 2 | Tool or config error |

## 14. CI integration

A reusable GitHub Actions workflow:

- **Triggers** on PRs that touch the model config, the `prompts/` directory, or the retrieval code (path filters).
- **Runs** `upgrade-report run`, with the baseline served from cache.
- **Posts** `report.md` as a PR comment, updating the same comment on re-runs.
- **Blocks** the merge on exit code 20. Exit code 10 is non-blocking but requires a reviewer acknowledgement label.

## 15. Tech stack

| Area | Choice |
|---|---|
| Language | Python 3.11+ |
| CLI | Typer |
| Config | Pydantic + YAML |
| Eval execution and storage | LangSmith SDK (`evaluate`, experiments, datasets) |
| Stats | numpy, scipy |
| Rendering | Jinja2 templates for Markdown and HTML |
| Cache | Local SQLite in CI cache, keyed by content hash |
| Packaging | Internal PyPI package plus a template repo |

## 16. Repository layout

```
upgrade-report/
├── upgrade_report/
│   ├── cli.py
│   ├── config.py          # Pydantic models
│   ├── runner.py          # arm execution, interleaving, repeats
│   ├── cache.py
│   ├── scoring.py         # evaluator orchestration
│   ├── stats/
│   │   ├── bootstrap.py
│   │   ├── mcnemar.py
│   │   ├── noise.py
│   │   └── mde.py
│   ├── behavior.py
│   ├── cost.py
│   ├── verdict.py
│   └── render/
│       ├── templates/
│       └── render.py
├── templates/project/     # scaffold used by `init`
├── .github/workflows/upgrade-report.yml
├── tests/
│   ├── stats/             # simulation-based tests
│   ├── fixtures/          # fake target + fake judge
│   └── snapshots/         # rendered report snapshots
└── docs/
```

## 17. Testing the tool itself

A tool that makes ship decisions needs its own evidence that it's right.

- **Stats simulation tests.** Generate synthetic per-example scores with a known true effect (0, small, large) and check that CI coverage is near 95% and detection rates match expectations.
- **A/A calibration test.** Run baseline vs baseline with a deterministic fake target plus injected noise. The verdict must be **Ship** at least 95% of the time.
- **Injected regression test.** A fake candidate that fails a fixed 10% of examples must be caught.
- **Fixture integration test.** Run end to end with a fake target and a fake judge, and require no network calls.
- **Snapshot tests** for rendered Markdown and HTML.

## 18. Milestones

| Milestone | Scope | Estimate |
|---|---|---|
| **M0: Scaffolding** | Repo, config models, target contract, fake fixtures | 2 to 3 days |
| **M1: Runner + scoring** | Arm execution, repeats, interleaving, LangSmith experiments, evaluator wiring | Week 1 |
| **M2: Stats + verdict** | Noise floor, bootstrap, McNemar, slices, MDE, decision rules, simulation tests | Week 2 |
| **M3: Reports + cost** | `report.json`, Markdown renderer, cost projection, latency | Weeks 2 to 3 |
| **M4: CI + cache** | GitHub Action, PR comments, exit codes, SQLite cache, `--dry-run` cost estimate | Week 3 |
| **M5: Behavior + arms** | Behavior diffs, pairwise judge, adapted-prompt arm | Week 4 |
| **M6: Client + scheduled** | Client HTML, redaction, scheduled mode that runs on new model releases | Week 5+ |

**MVP = M0 through M4.** That's enough to make a real, defensible upgrade decision on one project.

### Pilot plan

1. Run M4 on one internal project with a known upgrade in flight.
2. Compare the tool's verdict against the team's manual judgment, and investigate any disagreement.
3. Onboard a second project using only `init` and the docs, and time the setup.
4. Use the client HTML version (M6) on a real client upgrade conversation.

## 19. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Judge model changes silently | Judge pinned in config, checked at run start, and recorded in the report |
| Datasets too small to detect the regressions that matter | MDE reported, plus a caveat whenever MDE is larger than the margin |
| Prompts tuned to the baseline make candidates look worse | Optional adapted-prompt arm |
| Eval runs get expensive | Baseline caching, `--dry-run` cost estimates, configurable repeats |
| Provider rate limits skew latency | Interleaved execution, concurrency limit, retries with backoff |
| Client data leaking into reports | Redaction option, and client HTML strips trace links and IDs |
| Teams treat "Ship" as a guarantee | Caveats section always rendered, and the docs explain what the verdict does and doesn't cover |

## 20. Open questions

- Should the pricing table live in each project or in a shared internal repo?
- How should multi-step agents be handled: trajectory-level metrics only, or per-step scoring too?
- Should Ship-with-caveats block merges for some projects (for example, regulated clients)?
- Should the scheduled mode open PRs automatically when a new model passes, or only notify?
- Do we support non-LangSmith backends (for example, Langfuse) through an adapter, or stay LangSmith-only?