<!-- upgrade-report:test-project -->
## ⚠️ Ship with caveats
**MDE 0.106 for correctness is larger than the margin 0.050: this dataset is too small to rule out a regression of the size you care about** (+1 more)

`test-project` · candidate **`fake-good-b`** · `prompts/answer@v2` vs baseline **`fake-good-a`** · `prompts/answer@v1` · 80 examples × 3 repeats

<details><summary>All 2 reasons</summary>

- MDE 0.106 for correctness is larger than the margin 0.050: this dataset is too small to rule out a regression of the size you care about
- run-to-run noise on correctness (0.087) is larger than the margin 0.050: add repeats or examples

</details>

### Candidate arms

| Arm | Model | Prompt | Verdict | Δ correctness | Cost / query |
|---|---|---|---|--:|--:|
| candidate | `fake-verbose` | `prompts/answer@v1` | ⚠️ Ship with caveats | +0.025 | $0.0047 |
| **adapted_candidate** (recommended) | `fake-good-b` | `prompts/answer@v2` | ⚠️ Ship with caveats | −0.013 | $0.0031 |

The rest of this report covers the recommended arm, **adapted_candidate**.

### Metrics

| Metric | Role | Baseline | Candidate | Δ | 95% CI | Noise floor | Status |
|---|---|--:|--:|--:|---|--:|---|
| correctness | primary · pass/fail | 0.842 | 0.829 | −0.013 | [−0.083, +0.058] | 0.087 | ≈ within noise |
| format_compliance | guardrail · pass/fail | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | 0.000 | ≈ within noise |
| quality | tracked | 0.725 | 0.719 | −0.006 | [−0.017, +0.006] | 0.016 | ≈ within noise |
| faithfulness | tracked | 0.842 | 0.829 | −0.013 | [−0.083, +0.058] | 0.087 | ≈ within noise |

Non-inferiority margin on **correctness**: −0.050 · minimum detectable effect: 0.106 · guardrail tolerance: noise.

Pass/fail flips (per-example majority across repeats):
- `correctness`: 6 improved, 7 regressed (McNemar p = 1.00, exact)
- `format_compliance`: 0 improved, 0 regressed

### Per-tag breakdown

| Tag | n | Metric | Baseline | Candidate | Δ | 95% CI | Status |
|---|--:|---|--:|--:|--:|---|---|
| account | 20 | correctness | 0.867 | 0.817 | −0.050 | [−0.167, +0.067] | ≈ within noise |
| billing | 20 | correctness | 0.883 | 0.850 | −0.033 | [−0.167, +0.083] | ≈ within noise |
| legal · _directional only_ | 6 | correctness | 0.778 | 0.778 | +0.000 | [−0.167, +0.167] | ≈ within noise |
| returns | 20 | correctness | 0.800 | 0.867 | +0.067 | [−0.100, +0.233] | ≈ within noise |
| shipping | 20 | correctness | 0.817 | 0.783 | −0.033 | [−0.200, +0.133] | ≈ within noise |

Slices under 20 examples are _directional only_ and never block. Many slices and metrics are compared. At 95% confidence roughly 1 comparison in 20 looks significant by chance, so treat an isolated slice difference as a lead to investigate, not a finding.

### Flipped cases

<details open><summary><b>correctness</b>: pass → fail (7, showing 3)</summary>

**Q:** What is the value for item 6? · _account_
_Reference:_ The answer is value-6.

Baseline (0.67):
> The answer is value-6.

Candidate (0.33):
> The answer is unclear from the sources.

[baseline trace](https://smith.langchain.com/o/org-id/projects/p/project-id/r/b-q0-0006?poll=true) · [candidate trace](https://smith.langchain.com/o/org-id/projects/p/project-id/r/c-q0-0006?poll=true) · `q0-0006`

---

**Q:** What is the value for item 12? · _billing_
_Reference:_ The answer is value-12.

Baseline (1.00):
> The answer is value-12.

Candidate (0.33):
> The answer is unclear from the sources.

[baseline trace](https://smith.langchain.com/o/org-id/projects/p/project-id/r/b-q0-0012?poll=true) · [candidate trace](https://smith.langchain.com/o/org-id/projects/p/project-id/r/c-q0-0012?poll=true) · `q0-0012`

---

**Q:** What is the value for item 13? · _shipping_
_Reference:_ The answer is value-13.

Baseline (1.00):
> The answer is value-13.

Candidate (0.00):
> The answer is unclear from the sources.

[baseline trace](https://smith.langchain.com/o/org-id/projects/p/project-id/r/b-q0-0013?poll=true) · [candidate trace](https://smith.langchain.com/o/org-id/projects/p/project-id/r/c-q0-0013?poll=true) · `q0-0013`

</details>

<details><summary><b>correctness</b>: fail → pass (6, showing 3)</summary>

**Q:** What is the value for item 7? · _returns_
_Reference:_ The answer is value-7.

Baseline (0.33):
> The answer is unclear from the sources.

Candidate (1.00):
> The answer is value-7.

[baseline trace](https://smith.langchain.com/o/org-id/projects/p/project-id/r/b-q0-0007?poll=true) · [candidate trace](https://smith.langchain.com/o/org-id/projects/p/project-id/r/c-q0-0007?poll=true) · `q0-0007`

---

**Q:** What is the value for item 9? · _shipping_
_Reference:_ The answer is value-9.

Baseline (0.33):
> The answer is unclear from the sources.

Candidate (1.00):
> The answer is value-9.

[baseline trace](https://smith.langchain.com/o/org-id/projects/p/project-id/r/b-q0-0009?poll=true) · [candidate trace](https://smith.langchain.com/o/org-id/projects/p/project-id/r/c-q0-0009?poll=true) · `q0-0009`

---

**Q:** What is the value for item 39? · _returns_
_Reference:_ The answer is value-39.

Baseline (0.33):
> The answer is unclear from the sources.

Candidate (1.00):
> The answer is value-39.

[baseline trace](https://smith.langchain.com/o/org-id/projects/p/project-id/r/b-q0-0039?poll=true) · [candidate trace](https://smith.langchain.com/o/org-id/projects/p/project-id/r/c-q0-0039?poll=true) · `q0-0039`

</details>

### Cost and latency

| | Baseline | Candidate | Change |
|---|--:|--:|--:|
| Cost per query | $0.0037 | $0.0031 | −17% |
| Monthly (120,000 queries) | $442 | $367 | −$75.32 |
| Tokens per query (in / out) | 1200 / 6 | 1200 / 6 | |
| Latency p50 | 1.20 s | 0.90 s | −25% |
| Latency p95 | 3.40 s | 2.60 s | −24% |

### Behavior

| | Baseline | Candidate | Change |
|---|--:|--:|--:|
| Answer length, median (est. tokens) | 6 | 6 | +0% |
| Answer length, p90 (est. tokens) | 7 | 7 | +2% |
| Refusal rate | 0.0% | 0.0% | +0.0 pts |
| Format failure rate | 0.0% | 0.0% | +0.0 pts |
| Target error rate | 0.0% | 0.0% | +0.0 pts |
| Citations per answer | 0.84 | 0.83 | −0.01 |

**Pairwise judge:** candidate preferred in 57% of 20 sampled comparisons (95% CI 45%–70%; 5 wins, 2 losses, 13 ties; order randomized).

### Caveats

- Judge: fake-judge-2025-10 (temperature 0.0), pinned and identical across arms.
- Dataset: dataset.jsonl, content version 65177bd35a7a14ba, 80 examples.
- Repeats: 3 per arm.
- Minimum detectable effect on correctness (candidate): 0.094 vs margin 0.050.
- Minimum detectable effect on correctness (adapted_candidate): 0.106 vs margin 0.050.
- This eval run cost $3.12 (720 target calls, 1480 judge calls).
- What the verdict covers: this dataset, these metrics and this judge. It is not a guarantee about production traffic.

<details><summary>Run details</summary>

| Arm | Model | Prompt | Prompt hash | Code version | Repeats (cached / run / errors) |
|---|---|---|---|---|---|
| baseline | `fake-good-a` | `prompts/answer@v1` | `7e76538bbb6f9aec` | `157d0c0ec53475bc` | [r0](https://smith.langchain.com/o/org-id/projects/p/exp-id) 0/80/0 · [r1](https://smith.langchain.com/o/org-id/projects/p/exp-id) 0/80/0 · [r2](https://smith.langchain.com/o/org-id/projects/p/exp-id) 0/80/0 |
| candidate | `fake-verbose` | `prompts/answer@v1` | `7e76538bbb6f9aec` | `157d0c0ec53475bc` | [r0](https://smith.langchain.com/o/org-id/projects/p/exp-id) 0/80/0 · [r1](https://smith.langchain.com/o/org-id/projects/p/exp-id) 0/80/0 · [r2](https://smith.langchain.com/o/org-id/projects/p/exp-id) 0/80/0 |
| adapted_candidate | `fake-good-b` | `prompts/answer@v2` | `a9344673f453c8c5` | `157d0c0ec53475bc` | [r0](https://smith.langchain.com/o/org-id/projects/p/exp-id) 0/80/0 · [r1](https://smith.langchain.com/o/org-id/projects/p/exp-id) 0/80/0 · [r2](https://smith.langchain.com/o/org-id/projects/p/exp-id) 0/80/0 |

Config `fb88903d973ebffe` · dataset `dataset.jsonl` @ `65177bd35a7a14ba` · judge `fake-judge-2025-10` · upgrade-report 0.1.0 · 2026-01-15T12:00:00+00:00

</details>
