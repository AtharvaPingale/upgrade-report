# CI and scheduled runs

## PR check

`.github/workflows/upgrade-report.yml` is a reusable workflow. A project calls
it from a workflow with path filters (scaffolded by `init` as
`.github/workflows/model-upgrade.yml`):

```yaml
on:
  pull_request:
    types: [opened, synchronize, reopened, labeled]
    paths: [config.yaml, pricing.yaml, "prompts/**", "app/retrieval/**"]
  push:
    branches: [main]
    paths: [config.yaml, pricing.yaml, "prompts/**", "app/retrieval/**"]

jobs:
  upgrade-report:
    uses: atharvapingale/upgrade-report/.github/workflows/upgrade-report.yml@v1
    permissions:
      contents: read
      pull-requests: write
    with:
      config: config.yaml
    secrets: inherit
```

What it does:

1. Restores the SQLite eval cache (a PR restores its own cache, else main's).
2. Runs `upgrade-report run`. On PRs it passes `--baseline-ref origin/<base>`,
   so the baseline is main's cached run and the comparison includes code and
   local prompt changes. Usually only the candidate runs.
3. Saves the cache and uploads the report directory as an artifact.
4. Posts `report.md` as a PR comment, updating the same comment on re-runs
   (matched by its `<!-- upgrade-report:<project> -->` first line).
5. Gates the merge:

| Exit | Verdict | `report` check | `acknowledge` check |
|---|---|---|---|
| 0 | Ship | pass | skipped |
| 10 | Ship with caveats | pass | fails until a reviewer adds the `upgrade-report-acknowledged` label |
| 20 | Don't ship | fail | skipped |
| 2 | Tool or config error | fail | skipped |

Make both checks required in branch protection. Adding the label re-runs the
workflow (the caller listens for `labeled`); everything is cached by then, so
the re-run is quick.

Pushes to main keep the baseline cache warm, keyed by main's code.

### Inputs and secrets

| Input | Default |
|---|---|
| `config` | `config.yaml` |
| `working-directory` | `.` |
| `python-version` | `3.12` |
| `install` | `pip install -e . "upgrade-report[anthropic] @ git+https://github.com/atharvapingale/upgrade-report@v1"` |
| `ack-label` | `upgrade-report-acknowledged` |
| `cache-dir` | `.upgrade-report` (must contain `cache.path`) |

Secrets passed through: `LANGSMITH_API_KEY`, `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`. The job needs `pull-requests: write` to comment.

### Runtime

Tool overhead is small: statistics, rendering and cache access for three arms
× 300 examples × 3 repeats take under two seconds. Run time is dominated by
provider latency: with the baseline cached, a 300-example, 3-repeat PR check
makes 900 target calls plus judge calls, at `concurrency` in flight. Raise
`concurrency` up to your rate limits to stay under 20 minutes.

### When LangSmith is down

Experiment creation failures do not block the check: the run continues
untraced, the report says so, and it has no trace links. Feedback uploads are
best effort.

## Other CI systems

The CLI works anywhere: use the exit codes, publish `reports/*/report.md`, and
persist `.upgrade-report/` between runs. When `GITHUB_OUTPUT` is set the CLI
also writes `report_dir`, `verdict` and `exit_code` to it.

## Scheduled mode (new model releases)

`upgrade-report scheduled` lists the models a provider offers, and runs a full
comparison for every model it has not seen before, with that model as the
candidate. It only notifies: results go to stdout, to the GitHub step summary,
and to `reports/<model>/`. It never opens PRs or changes config.

```yaml
schedule:
  model_source: anthropic          # or module:function returning a list of model ids
  model_filter: "^claude-(sonnet|haiku)"
  state_file: .upgrade-report/seen_models.json
```

The first run only records the models that exist today (pass
`--include-existing` to evaluate them too). Models that error are retried on
the next run. Add prices for new models to `pricing.yaml`, or their cost is
reported as unknown. `init` scaffolds `.github/workflows/model-watch.yml`, which
runs this every weekday and keeps the state file in the Actions cache.
