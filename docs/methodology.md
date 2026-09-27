# Methodology

How `upgrade-report` turns raw scores into a verdict, and the evidence that it
is calibrated. Code references are to `upgrade_report/`.

## Pipeline

1. **Load config** and validate it (`config.py`).
2. **Check the cache.** Each arm run is keyed by `(dataset_version, model,
   prompt_hash, target_version, repeat_index)` (`cache.py`). Records are stored
   per example, so an interrupted run resumes; failed calls are never cached.
3. **Run arms** (`runner.py`). N repeats per arm. Baseline and candidate calls
   for the same example are submitted side by side, alternating which arm goes
   first, so a provider slowdown hits both arms at once. Each (arm, repeat) is
   one LangSmith experiment when tracing is on (`tracing.py`).
4. **Score** every output with the project's evaluators and the pinned judge
   (`scoring.py`). Scores are cached under `(run_key, scorer_key)`; the scorer
   key covers the judge configuration and the evaluators' source.
5. **Aggregate per example.** Continuous scores are averaged across repeats.
   Pass/fail metrics use the per-example pass rate; its majority (≥ 0.5) feeds
   McNemar and the flip lists.
6. **Statistics** (`stats/`, `analysis.py`), **behavior diffs**
   (`behavior.py`), **cost and latency** (`cost.py`).
7. **Verdict** (`verdict.py`) and **render** (`render/`).

## Versions and cache keys

- `dataset_version` is a content hash of the loaded examples (ids, inputs,
  reference outputs, metadata). Editing one example changes it.
- `prompt_hash` covers the prompt reference and, if it names a local file, the
  file's content.
- `target_version` hashes the files under `target_paths` as git blob SHAs, only
  counting tracked files inside a git repository. Because the same digest can
  be computed from a git ref, `--baseline-ref origin/main` finds the baseline
  that main's CI run cached, even when the PR changes code or local prompts.
  Without a matching cache entry the baseline runs with the PR's code and the
  report says so, since that comparison then excludes the code change.

## Statistics

### Paired design

Every comparison uses per-example differences `candidate − baseline`, so
question difficulty cancels out.

### Confidence intervals (`stats/bootstrap.py`)

Percentile bootstrap on the mean per-example difference: 10,000 resamples, fixed
seed, resampling in memory-bounded chunks. Pass/fail metrics use the same
interval on per-example pass rates.

### Noise floor (`stats/noise.py`)

The baseline's repeats are a free A/A test. For each pair of repeats (1 vs 2,
1 vs 3, 2 vs 3) the per-example differences are bootstrapped, and the 95th
percentile of the absolute bootstrapped means (pooled across pairs) is the noise
of comparing *one repeat with another*.

**Refinement over the plan:** the candidate's delta compares *means of R
repeats*, whose run-to-run noise is smaller by √R. The reported noise floor is
therefore divided by √R. Without this, with 3 repeats the floor is about 1.7×
too large and a regression the size of the margin can be classed "within noise"
on a well-powered dataset: for example, a −0.03 drop with a CI of
[−0.043, −0.017] would ship without a caveat. The unscaled value is kept in the
code as `NoiseFloor.single_repeat`.

Differences no larger than the noise floor are **within noise** and never count
as regressions. With a single repeat there is no noise floor, and the CI alone
decides.

### Pass/fail metrics (`stats/mcnemar.py`)

McNemar's test on discordant pairs, exact binomial below 25 discordant pairs,
chi-square with continuity correction from 25. Both flip counts are reported.

### Slices

The same statistics per tag. Slices under `min_slice_size` are marked
*directional only* and can never block. A slice is flagged (a caveat, never a
block) when, with at least 5 examples:

- primary metric: the slice's delta is below the margin **and** its CI sits
  entirely below minus the slice's noise floor;
- guardrail: the slice's CI sits entirely below minus its tolerance.

Requiring the whole CI to clear the noise floor keeps false flags rare despite
the number of slices; the report still warns that some slice differences will
look significant by chance.

### Minimum detectable effect (`stats/mde.py`)

`MDE = (z₀.₉₇₅ + z₀.₈) · sd(differences) / √n`: the smallest true difference
the dataset detects with 80% power at 5% two-sided significance. When it
exceeds the margin the report says the dataset is too small to rule out a
regression of the size you care about.

## Decision rules (`verdict.py`)

Evaluated in order.

**Don't ship** if any of:

- **Primary:** the delta is below minus the noise floor **and** the CI lower
  bound is below `primary_margin`. The first condition applies the plan's rule
  that within-noise differences never trigger a regression; it also means an
  observed improvement never blocks, which keeps the verdict monotonic in the
  delta.
- **Guardrail:** the CI sits entirely below minus the noise floor (or the
  absolute `guardrail_tolerance`), and the delta is not within noise.
- The primary metric has no paired scores at all.

**Ship with caveats** if nothing blocks but any of:

- a behavior flag (length, refusals, format failures, target errors, latency);
- cost per query above `cost_increase_pct`;
- a slice regression (directional-only or not);
- primary MDE larger than the margin;
- otherwise, the primary CI crosses the margin while the delta is within noise
  (non-inferiority not established);
- the primary noise floor is larger than the margin;
- a guardrail has no scores;
- the pairwise judge's CI for candidate preference lies entirely below 50%.

**Ship** otherwise.

Tracked metrics never affect the verdict. With an adapted-prompt arm, both
candidate arms get a verdict and the report recommends the better one: best
verdict, then fewer reasons against it, then larger primary delta, then lower
cost. The exit code follows the recommended arm.

## Behavior diffs (`behavior.py`)

- Answer length: median and p90 of per-example mean length, estimated as
  characters ÷ 4 (the same estimator for both arms, so the relative change is
  meaningful).
- Refusal rate: configured regexes or classifier.
- Citations per answer (`outputs["citations"]`), tool calls per answer and mix
  (`outputs["tool_calls"]`).
- Format failure rate: outputs without a string `answer`, plus the optional
  `format_validator`.
- Target error rate. Failed calls are excluded from metrics, so this flag is
  what keeps a candidate that crashes on hard inputs from looking good.
- Pairwise judge: a seeded sample of examples; the judge sees both answers in a
  reproducible random order; the preference rate (win = 1, tie = 0.5) gets a
  bootstrap CI.

## Cost and latency (`cost.py`)

Per-call cost uses the call's token usage per model and `pricing.yaml`; the
per-query cost is the mean over successful calls, and the monthly projection
multiplies it by `traffic.monthly_queries`. Latency is wall-clock per successful
call (retries excluded), reported as p50 and p95. The eval run's own cost
covers target calls actually executed (not cache hits) plus judge calls.

## The judge

The judge is a pinned instrument: the model must be an exact version, the
configuration is part of the score cache key and recorded in the report, and
each response's reported model id is checked against the pin
(`JudgeDriftError`, exit code 2). The built-in Anthropic adapter deliberately
does not enable server-side model fallbacks, because a fallback would swap the
judge. Refusals and truncated replies raise instead of producing a score.

## Evidence

Run by the test suite (`tests/stats`, `tests/test_calibration.py`):

| Property | Result |
|---|---|
| Bootstrap CI coverage, true effect 0 / 0.05 / 0.3 (n = 200) | 92–97.5% (target 95%) |
| McNemar p-values | match `scipy.stats.binomtest` / chi-square; type I error ≤ 6% at α = 0.05 |
| Detection at a true effect equal to the MDE | 72–88% (target 80%) |
| A/A differences within the noise floor | ≥ 95% |
| **A/A verdict = Ship** (300 examples, 3 repeats, 5 slices incl. one of 8) | **97.2%** over 500 simulated runs (target ≥ 95%); end-to-end A/A through the CLI also checked |
| Primary drop equal to the margin (−0.02) | Don't ship 90% |
| Primary drop 1.5× / 2× the margin | Don't ship 100% |
| Fake candidate failing a fixed 10% of examples | Don't ship in every seed, flagged examples overlap the injected ones |

In the A/A simulation the most common false caveat was the 8-example
directional slice (1.2% of runs), followed by a primary false block (0.6%).
