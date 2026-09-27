# Reading a report

## The verdict

- **Ship**: the candidate is not meaningfully worse on the primary metric (its
  CI does not reach past the margin), no guardrail clearly regressed, and no
  behavior, cost, slice or dataset-size flag fired.
- **Ship with caveats**: nothing blocks, but something needs a person to decide.
  The reasons are listed at the top. In CI this needs a reviewer's
  acknowledgement label.
- **Don't ship**: the primary metric may have dropped by more than the margin,
  or a guardrail clearly regressed.

The rules are *non-inferiority*, not superiority. An upgrade usually has to be
"not meaningfully worse, and cheaper or faster", not "strictly better".

## What the verdict does not cover

- It describes **this dataset, these metrics and this judge**. Behavior on
  traffic the dataset does not represent is not measured.
- A judge is a model too. It is pinned and checked, but its blind spots are the
  report's blind spots.
- **Ship is not a guarantee.** Keep production monitoring and online
  evaluators; this tool does not replace them.
- Failed target calls are excluded from metric scores. The target error rate is
  reported and flagged, so check it whenever the candidate had errors.
- Tracked metrics never affect the verdict, however large their change.

## The metric table

| Column | Meaning |
|---|---|
| Baseline / Candidate | Mean per-example score (pass rate for pass/fail metrics), over examples scored in both arms |
| Δ | Candidate minus baseline |
| 95% CI | Paired bootstrap interval for Δ |
| Noise floor | How much this metric moves between identical runs, at the resolution of an R-repeat average |
| Status | `fail` (blocks), `within noise`, `improved`, `worse, not blocking`, `pass` (no meaningful drop) |

Under the table: the margin, the primary metric's minimum detectable effect
(MDE), and pass/fail flips with McNemar p-values. If the MDE is larger than
the margin, the dataset cannot rule out a regression of the size you care
about. Add examples or repeats before relying on a Ship.

## Slices

One row per tag for the primary metric. *Directional only* means fewer than
`min_slice_size` examples: shown, never blocking. Many slices are compared, so
about one in twenty will look significant by chance. Treat an isolated slice
difference as a lead to investigate.

## Flipped cases

Pass→fail first, then fail→pass, per pass/fail metric. For continuous metrics,
the largest drops and gains beyond the noise floor. Each case shows the
question, the reference, a representative answer from each arm (the repeat
that matches the majority outcome), and LangSmith trace links.

## Cost, latency, behavior

Cost per query is the mean over successful calls, priced from `pricing.yaml`;
the monthly figure multiplies by `traffic.monthly_queries`. Latency is p50/p95
per arm; calls are interleaved, so the arms are comparable. Behavior rows show
what averages hide: longer answers, more refusals, format breaks, errors, tool
use. The pairwise judge line gives the share of sampled comparisons where the
judge preferred the candidate (ties count half), with a 95% CI.

## Caveats

Always rendered: the judge and its settings, the dataset version, repeats, the
MDE, any failures, and what the eval run itself cost.

## The client version

`report_client.html` (or `upgrade-report render report.json -f html`) keeps the
verdict, metric table, cost impact and 3–5 example comparisons, in plain
language. It has no trace links, example IDs, run keys or config hashes.
`--redact-inputs` (or `report.redact_inputs: true`) hides the questions. Pick
the examples with `report.client_examples`; otherwise it shows the top gains
and drops on the primary metric.
