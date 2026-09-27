# Getting started

Goal: your project's first report in about 30 minutes.

## 1. Install and scaffold (2 minutes)

```bash
pip install "upgrade-report @ git+https://github.com/atharvapingale/upgrade-report"
# for the built-in Claude judge (judge.client: anthropic), install "upgrade-report[anthropic] @ ..." instead
cd your-project
upgrade-report init
upgrade-report run          # runs the stub on the sample dataset
```

`init` creates:

| File | Purpose |
|---|---|
| `config.yaml` | Arms, dataset, metrics, decision rules |
| `pricing.yaml` | Per-model token prices; checked in so reports stay reproducible |
| `prompts/answer.txt` | Example local prompt; its content is part of the cache key |
| `upgrade_eval/target.py` | The one function you write |
| `upgrade_eval/evaluators.py` | Example evaluators |
| `upgrade_eval/dataset.jsonl` | Eight sample examples, so `run` works immediately |
| `.github/workflows/model-upgrade.yml` | PR check (see [ci.md](ci.md)) |
| `.github/workflows/model-watch.yml` | Scheduled new-model runs |

## 2. Implement the target (10 minutes)

`upgrade_eval/target.py:run(inputs, model, prompt)` is called once per example,
per arm, per repeat. Make it a thin wrapper around your real entry point so the
eval exercises production code:

```python
from app.pipeline import answer_question

def run(inputs: dict, model: str, prompt: str) -> dict:
    result = answer_question(inputs["question"], model=model, prompt_ref=prompt)
    return {
        "answer": result.text,
        "doc_ids": result.retrieved_ids,                 # enables recall@k
        "contexts": [d.text for d in result.documents],  # enables registry:faithfulness
        "citations": result.citations,
        "usage": {"input_tokens": result.usage.input_tokens,
                  "output_tokens": result.usage.output_tokens},
    }
```

Cost needs token counts. Return `usage` as above, or call
`upgrade_report.record_usage(input_tokens, output_tokens, model=None)` from your
LLM wrapper, or leave it to LangSmith trace token counts when tracing is on.
If the target calls several models (say, an embedding model), return
`usage` keyed by model: `{"text-embed-3": {...}, "model-b-2026-03": {...}}`.

Async targets (`async def run(...)`) work too.

Set `target_paths` in `config.yaml` to every directory whose code affects
answers (for example `[upgrade_eval/, app/]`). Changing files there
invalidates cached runs.

## 3. Point at your dataset and evaluators (10 minutes)

```yaml
dataset: client-x-golden-v4        # LangSmith dataset name or ID; needs LANGSMITH_API_KEY
dataset_version: prod              # optional: a tag or timestamp to pin
evaluators:
  - app.evals:faithfulness         # your existing LangSmith evaluators, unchanged
  - registry:refusal_correctness   # or shared judges
  - registry:recall@5
```

Per-tag slices come from example metadata (`slice_by: tags` by default).

Evaluators that return a bool are pass/fail metrics (they get McNemar and flip
lists); numbers are continuous. Override with `metrics.kinds`.

LLM-as-judge evaluators get the pinned judge by declaring a `judge` argument.
Configure the client:

```yaml
judge:
  model: claude-sonnet-5           # an exact version; aliases like "-latest" are rejected
  temperature: null                # null for models that reject sampling parameters
  client: anthropic                # built-in adapter, or module:function
```

A custom client is any callable
`complete(*, model, prompt, system, temperature, max_tokens) -> str | dict`.
Return `{"text": ..., "model": <model id the provider reports>, "input_tokens":
..., "output_tokens": ...}` so the tool can verify the judge did not change and
price the eval run.

## 4. Choose metrics and arms (5 minutes)

```yaml
baseline:  { model: model-a-2025-06, prompt: prompts/answer@v12 }
candidate: { model: model-b-2026-03, prompt: prompts/answer@v12 }
metrics:
  primary: faithfulness                                         # non-inferiority tested
  guardrails: [refusal_correctness, format_compliance, recall@5] # must not clearly regress
  tracked: [answer_relevance]                                   # reported, never decisive
```

Add prices for every model (including the judge) to `pricing.yaml`.

## 5. Dry run, then run (3 minutes plus run time)

```bash
upgrade-report run --dry-run    # validates everything, shows calls to make and estimated cost
upgrade-report run
upgrade-report aa               # once per dataset: confirm baseline vs itself is Ship
```

If `aa` does not return Ship, the tool is too noisy for this dataset and
config: add repeats or examples before trusting a verdict. Then wire up CI with
[ci.md](ci.md).
