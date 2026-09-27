# Configuration reference

`config.yaml` is validated on load; unknown keys are errors, so typos fail fast
(exit code 2). Paths are relative to the config file.

## Project, dataset, target

| Key | Default | Description |
|---|---|---|
| `project` | required | Name used in reports, experiment names and the PR comment marker |
| `dataset` | required | LangSmith dataset name or ID, or a `.json`/`.jsonl` file of `{"id", "inputs", "outputs", "metadata"}` rows |
| `dataset_version` | latest | LangSmith `as_of` tag or timestamp |
| `dataset_splits` | all | LangSmith splits to load |
| `target` | required | `module:function` implementing `run(inputs, model, prompt) -> dict` |
| `target_paths` | target's top-level package | Code whose content is part of the cache key |
| `slice_by` | `tags` | Example metadata key holding slice tags (string or list) |

## Arms

```yaml
baseline:          { model: ..., prompt: ... }   # required: what runs today
candidate:         { model: ..., prompt: ... }   # required
adapted_candidate: { model: ..., prompt: ... }   # optional: candidate with a prompt tuned for it
```

`prompt` is passed to your target verbatim. If it names a local file (the part
before `@`, with or without an extension such as `.txt`, `.md`, `.j2`, `.yaml`),
the file's content is hashed into the cache key. Otherwise the reference string
itself is the version (for example a hub prompt `answer:v12`).

## Execution

| Key | Default | Description |
|---|---|---|
| `evaluators` | required | `module:function` or `registry:<name>`; see below |
| `repeats` | 3 | Runs per arm per example; 2 or more gives a noise floor |
| `concurrency` | 8 | Parallel target and judge calls |
| `retries` | 2 | Retries per failed target call, with exponential backoff |

Registry judges: `correctness` (LLM, pass/fail, needs a reference answer),
`faithfulness` (LLM, 0–1, needs `outputs["contexts"]`), `answer_relevance` (LLM,
0–1), `exact_match`, `refusal_correctness` (needs `should_refuse` in the
reference outputs), `recall@<k>` (needs `doc_ids` in outputs and reference).
Other packages can register judges under the `upgrade_report.judges` entry-point
group.

## Judge

| Key | Default | Description |
|---|---|---|
| `judge.model` | required | Exact model version. Aliases containing `latest` are rejected |
| `judge.temperature` | 0 | `null` for models that reject sampling parameters |
| `judge.client` | none | `anthropic`, or `module:function` for your own client |
| `judge.max_tokens` | 4096 | Truncated judge replies are errors, not scores |
| `judge.verify_response_model` | true | Fail if the provider reports a different model than pinned |

The judge configuration and the evaluators' source code form the score cache
key: changing either re-scores cached outputs without re-running the target.

## Metrics and decision

| Key | Default | Description |
|---|---|---|
| `metrics.primary` | required | Tested for non-inferiority |
| `metrics.guardrails` | `[]` | Block when their CI sits entirely below the tolerance |
| `metrics.tracked` | `[]` | Reported only |
| `metrics.kinds` | auto | `binary` or `continuous` per metric; auto = binary if the evaluator returned bools |
| `metrics.descriptions` | registry text | Plain-language text for the client report |
| `decision.primary_margin` | −0.02 | Non-inferiority margin (must be ≤ 0) |
| `decision.guardrail_tolerance` | `noise` | `noise` (the metric's noise floor) or an absolute number |
| `decision.min_slice_size` | 20 | Smaller slices are directional only |
| `decision.bootstrap_resamples` | 10000 | Bootstrap resamples |
| `decision.seed` | fixed | Seed for resampling and pairwise sampling |

## Behavior, cost, traffic

| Key | Default | Caveat when |
|---|---|---|
| `behavior_flags.length_change_pct` | 25 | Median answer length changes by more than this (either way) |
| `behavior_flags.refusal_rate_change_pts` | 3 | Refusal rate moves by more than this many points |
| `behavior_flags.cost_increase_pct` | 15 | Cost per query rises by more than this |
| `behavior_flags.format_failure_change_pts` | 2 | Format failure rate moves by more than this |
| `behavior_flags.error_rate_change_pts` | 1 | Target error rate moves by more than this |
| `behavior_flags.latency_p95_increase_pct` | off | p95 latency rises by more than this |

| Key | Default | Description |
|---|---|---|
| `behavior.refusal_patterns` | built-in list | Regexes that mark an answer as a refusal |
| `behavior.refusal_classifier` | none | `module:function(answer) -> bool`, replaces the patterns |
| `behavior.format_validator` | none | `module:function(output) -> bool`; outputs without a string `answer` always fail |
| `behavior.pairwise.enabled` | false | Pinned judge compares baseline and candidate answers in random order |
| `behavior.pairwise.sample_size` | 75 | Examples sampled for the pairwise judge (10–500) |
| `traffic.monthly_queries` | required | Volume for the monthly cost projection |
| `pricing_file` | `pricing.yaml` | `model: {input_per_mtok, output_per_mtok}` in USD |
| `estimate.*` | 1500/300 tokens | Fallback per-call token estimates for `--dry-run` before anything is cached |

## Report, cache, LangSmith, schedule

| Key | Default | Description |
|---|---|---|
| `report.formats` | `[md, json]` | Add `html` for the client version; JSON is always written |
| `report.redact_inputs` | false | Replace questions with `[redacted]` in the client HTML |
| `report.client_title` | project name | Title of the client HTML |
| `report.client_examples` | auto | Example IDs to show the client (3–5); auto picks top gains and drops |
| `report.max_examples` | 10 | Flipped cases per metric and direction in the report |
| `cache.enabled` | true | SQLite cache of runs and scores |
| `cache.path` | `.upgrade-report/cache.sqlite` | Keep it in the CI cache |
| `langsmith.enabled` | auto | On when the dataset is in LangSmith and `LANGSMITH_API_KEY` is set |
| `langsmith.experiment_prefix` | project | Experiment name prefix |
| `schedule.model_source` | — | `anthropic` or `module:function` returning model IDs |
| `schedule.model_filter` | none | Regex; only matching models are evaluated |
| `schedule.state_file` | `.upgrade-report/seen_models.json` | Models already seen |

## pricing.yaml

```yaml
model-a-2025-06: { input_per_mtok: 3.00, output_per_mtok: 15.00 }
model-b-2026-03: { input_per_mtok: 2.50, output_per_mtok: 10.00 }
judge-pinned-2025-10: { input_per_mtok: 1.00, output_per_mtok: 5.00 }
```

Models without a price are listed in the caveats and their cost is reported as
unknown rather than zero.
