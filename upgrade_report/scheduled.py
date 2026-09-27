"""Scheduled mode: evaluate newly released models as candidates.

Each scheduled run lists the models a provider offers, compares every model it
has not seen before against the baseline, records it as seen, and returns a
summary. It only notifies (reports, step summary, exit code); opening PRs for
passing models is left to the workflow.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import Config
from .errors import ConfigError
from .loader import load_object
from .pipeline import RunOptions, run
from .verdict import LABELS


@dataclass
class ScheduledResult:
    new_models: list[str]
    results: list[dict]
    first_run: bool
    summary_md: str


def anthropic_models() -> list[str]:
    try:
        import anthropic
    except ImportError as exc:
        raise ConfigError("schedule.model_source 'anthropic' needs the anthropic package") from exc
    return [m.id for m in anthropic.Anthropic().models.list()]


def _source(cfg: Config) -> Callable[[], list[str]]:
    assert cfg.schedule is not None
    if cfg.schedule.model_source == "anthropic":
        return anthropic_models
    fn = load_object(cfg.schedule.model_source, cfg.base_dir)
    if not callable(fn):
        raise ConfigError(f"schedule.model_source {cfg.schedule.model_source!r} is not callable")
    return fn


def run_scheduled(cfg: Config, *, out_dir: Path | None, include_existing: bool = False,
                  log: Callable[[str], None] = print) -> ScheduledResult:
    if cfg.schedule is None:
        raise ConfigError("scheduled mode needs a `schedule` section in the config")
    state_path = cfg.resolve(cfg.schedule.state_file)
    first_run = not state_path.exists()
    seen: set[str] = set(json.loads(state_path.read_text())["seen"]) if not first_run else set()
    available = [str(m) for m in _source(cfg)()]
    pattern = re.compile(cfg.schedule.model_filter) if cfg.schedule.model_filter else None
    eligible = [m for m in available if (pattern is None or pattern.search(m)) and m != cfg.baseline.model]

    if first_run and not include_existing:
        # Record what exists today; only models released after this point get evaluated.
        new: list[str] = []
        log(f"First scheduled run: recorded {len(eligible)} existing models; future releases will be evaluated.")
    else:
        new = [m for m in eligible if m not in seen]

    results = []
    for model in new:
        log(f"Evaluating new model {model}")
        candidate_cfg = cfg.model_copy(deep=True, update={
            "candidate": cfg.candidate.model_copy(update={"model": model}),
            "adapted_candidate": None,
        })
        candidate_cfg._base_dir = cfg.base_dir
        candidate_cfg._source_path = cfg.source_path
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model)
        try:
            result = run(candidate_cfg, RunOptions(out_dir=(out_dir or cfg.base_dir / "reports") / safe, log=log))
            results.append({"model": model, "verdict": result.report["verdict"],
                            "reasons": result.report["verdict_reasons"], "report_dir": str(result.report_dir)})
            seen.add(model)
        except Exception as exc:  # one broken model must not stop the others; it is retried next run
            results.append({"model": model, "verdict": "error", "reasons": [f"{type(exc).__name__}: {exc}"],
                            "report_dir": None})

    seen.update(eligible if first_run and not include_existing else [])
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"seen": sorted(seen)}, indent=2) + "\n")
    return ScheduledResult(new, results, first_run, _summary(cfg, new, results, first_run))


def _summary(cfg: Config, new: list[str], results: list[dict], first_run: bool) -> str:
    lines = [f"## upgrade-report scheduled run: {cfg.project}", ""]
    if not new:
        lines.append("No new models to evaluate." if not first_run else
                     "First run: recorded the current model list. New releases will be evaluated from now on.")
        return "\n".join(lines) + "\n"
    lines += [f"Baseline `{cfg.baseline.model}` · {len(new)} new model(s)", "",
              "| Model | Verdict | Reason | Report |", "|---|---|---|---|"]
    for r in results:
        label = LABELS.get(r["verdict"], r["verdict"])
        reason = (r["reasons"][0] if r["reasons"] else "").replace("|", "\\|")
        lines.append(f"| `{r['model']}` | {label} | {reason} | {r['report_dir'] or '—'} |")
    return "\n".join(lines) + "\n"
