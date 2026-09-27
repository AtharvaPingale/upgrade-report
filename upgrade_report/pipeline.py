"""End-to-end run: load → cache check → run arms → score → stats → verdict → render."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .analysis import compare_metric, compare_slices, detect_kinds, notable_examples, per_example_mean, score_matrix
from .behavior import arm_behavior, compare_behavior, make_format_check, make_refusal_detector, pairwise_preference
from .cache import Cache, run_key
from .config import CANDIDATE_ARMS, ArmConfig, Config, load_pricing
from .cost import arm_cost, arm_latency, compare_cost, compare_latency, eval_run_cost
from .dataset import Dataset, load_dataset
from .errors import ConfigError
from .judge import Judge
from .loader import load_object, module_source_paths
from .records import ArmRun
from .render import write_outputs
from .runner import Runner, interleave
from .scoring import Scorer, load_evaluators, scorer_key
from .tracing import NullSink, make_sink
from .verdict import EXIT_CODES, LABELS, decide, recommend
from .versioning import code_version, prompt_hash

Log = Callable[[str], None]

MULTIPLE_COMPARISONS_NOTE = (
    "Many slices and metrics are compared. At 95% confidence roughly 1 comparison in 20 looks "
    "significant by chance, so treat an isolated slice difference as a lead to investigate, not a finding."
)


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


@dataclass
class RunOptions:
    out_dir: Path | None = None
    dry_run: bool = False
    baseline_ref: str | None = None
    use_cache: bool = True
    formats: list[str] | None = None
    aa: bool = False
    log: Log = _stderr


@dataclass
class RunResult:
    exit_code: int
    report: dict | None = None
    report_dir: Path | None = None
    estimate: dict | None = None
    paths: dict[str, Path] = field(default_factory=dict)


@dataclass
class _ArmPlan:
    name: str
    config: ArmConfig
    prompt_hash: str
    target_version: str
    runs: list[ArmRun]


def _plan_arm(name: str, arm: ArmConfig, dataset: Dataset, ph: str, tv: str, repeats: int, offset: int,
              cache: Cache) -> _ArmPlan:
    runs = []
    ids = {e.id for e in dataset.examples}
    for r in range(repeats):
        key, parts = run_key(dataset.version, arm.model, ph, tv, r + offset)
        ar = ArmRun(arm=name, repeat=r + offset, key=key, key_parts=parts)
        cached = {eid: rec for eid, rec in cache.load_records(key).items() if eid in ids}
        ar.records.update(cached)
        ar.cached_ids = set(cached)
        runs.append(ar)
    return _ArmPlan(name, arm, ph, tv, runs)


def _fully_cached(plan: _ArmPlan, dataset: Dataset) -> bool:
    return all(len(ar.cached_ids) == len(dataset.examples) for ar in plan.runs)


def _pending(plan: _ArmPlan, dataset: Dataset) -> list[tuple[ArmRun, ArmConfig, list]]:
    out = []
    for ar in plan.runs:
        todo = [e for e in dataset.examples if e.id not in ar.cached_ids]
        if todo:
            out.append((ar, plan.config, todo))
    return out


def _estimate(plans: dict[str, _ArmPlan], pending: dict[str, list], cache: Cache, cfg: Config,
              pricing: dict, judge_evaluators: int, score_key: str) -> dict[str, Any]:
    rows = []
    total_usd: float | None = 0.0
    new_outputs = 0
    unscored_cached = 0
    for name, plan in plans.items():
        calls = sum(len(todo) for _, _, todo in pending.get(name, []))
        new_outputs += calls
        for ar in plan.runs:
            scored = cache.load_scores(ar.key, score_key)
            unscored_cached += sum(1 for eid in ar.cached_ids if eid not in scored)
        avg = cache.average_usage(plan.config.model)
        source = "cache" if avg else "config estimate"
        tin, tout = avg or (cfg.estimate.input_tokens_per_query, cfg.estimate.output_tokens_per_query)
        price = pricing.get(plan.config.model)
        usd = calls * price.cost(tin, tout) if price else None
        if usd is None and calls:
            total_usd = None
        elif total_usd is not None:
            total_usd += usd or 0.0
        cached = sum(len(ar.cached_ids) for ar in plan.runs)
        rows.append({"arm": name, "model": plan.config.model, "calls_to_run": calls, "cached_calls": cached,
                     "tokens_per_call": [round(tin), round(tout)], "token_source": source, "usd": usd})
    judge_calls = (new_outputs + unscored_cached) * judge_evaluators
    if cfg.behavior.pairwise.enabled:
        judge_calls += cfg.behavior.pairwise.sample_size * (len(plans) - 1)
    judge_price = pricing.get(cfg.judge.model)
    judge_usd = (judge_calls * judge_price.cost(cfg.estimate.judge_input_tokens_per_call,
                                                cfg.estimate.judge_output_tokens_per_call)
                 if judge_price else None)
    if judge_calls and judge_usd is None:
        total_usd = None
    elif total_usd is not None:
        total_usd += judge_usd or 0.0
    return {"arms": rows, "judge_calls": judge_calls, "judge_usd": judge_usd, "total_usd": total_usd,
            "target_calls": new_outputs}


def _write_github_outputs(values: dict[str, str]) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a") as fh:
        for k, v in values.items():
            fh.write(f"{k}={v}\n")


def run(cfg: Config, opts: RunOptions | None = None) -> RunResult:
    opts = opts or RunOptions()
    log = opts.log
    base = cfg.base_dir
    repeats = cfg.repeats
    caveats: list[str] = []

    log(f"Loading dataset {cfg.dataset}")
    dataset = load_dataset(cfg.dataset, base, cfg.dataset_version, cfg.dataset_splits)
    log(f"  {len(dataset.examples)} examples, version {dataset.version}")

    target = load_object(cfg.target, base)
    if not callable(target):
        raise ConfigError(f"target {cfg.target!r} is not callable")
    paths = [cfg.resolve(p) for p in cfg.target_paths] if cfg.target_paths else module_source_paths(cfg.target, base)
    current_tv = code_version(paths, base)

    judge = Judge.from_config(cfg.judge, base)
    evaluators = load_evaluators(cfg.evaluators, base)
    pricing_path = cfg.resolve(cfg.pricing_file)
    pricing = load_pricing(pricing_path)
    if not pricing:
        caveats.append(f"no pricing file at {cfg.pricing_file}; cost is not reported")
    is_refusal = make_refusal_detector(cfg.behavior, base)
    format_ok = make_format_check(cfg.behavior, base)

    cache_path = cfg.resolve(cfg.cache.path) if cfg.cache.enabled else None
    cache = Cache(cache_path, read=opts.use_cache)

    arms = {"baseline": cfg.baseline, "candidate": cfg.baseline} if opts.aa else cfg.arms()
    plans: dict[str, _ArmPlan] = {}
    for name, arm in arms.items():
        ph, tv = prompt_hash(arm.prompt, base), current_tv
        if name == "baseline" and opts.baseline_ref:
            ref_plan = _plan_arm(name, arm, dataset, prompt_hash(arm.prompt, base, opts.baseline_ref),
                                 code_version(paths, base, opts.baseline_ref), repeats, 0, cache)
            if _fully_cached(ref_plan, dataset):
                plans[name] = ref_plan
                log(f"  baseline served from the cache entry for {opts.baseline_ref}")
                continue
            caveats.append(
                f"the baseline for {opts.baseline_ref} was not in the cache, so it ran with this checkout's code: "
                "code changes in this branch are not part of the comparison"
            )
        offset = repeats if (opts.aa and name == "candidate") else 0
        plans[name] = _plan_arm(name, arm, dataset, ph, tv, repeats, offset, cache)

    base_keys = [ar.key for ar in plans["baseline"].runs]
    for name in list(plans):
        if name != "baseline" and [ar.key for ar in plans[name].runs] == base_keys:
            plans[name].runs = plans["baseline"].runs
            caveats.append(f"{name} is identical to the baseline (same model, prompt and code); "
                           "the comparison is trivially within noise. Use `upgrade-report aa` for calibration.")

    unique_runs: dict[str, ArmRun] = {}
    for plan in plans.values():
        for ar in plan.runs:
            unique_runs.setdefault(ar.key, ar)
    pending: dict[str, list] = {}
    seen: set[str] = set()
    for name, plan in plans.items():
        items = [p for p in _pending(plan, dataset) if p[0].key not in seen]
        seen.update(p[0].key for p in items)
        if items:
            pending[name] = items
    if opts.dry_run:
        estimate = _estimate(plans, pending, cache, cfg, pricing, sum(e.uses_judge for e in evaluators),
                             scorer_key(evaluators, judge))
        estimate["dataset"] = {"name": dataset.name, "examples": len(dataset.examples), "version": dataset.version}
        cache.close()
        return RunResult(exit_code=0, estimate=estimate)

    sink = make_sink(cfg.langsmith.enabled, dataset, cfg.langsmith.experiment_prefix or cfg.project)
    try:
        for name, items in pending.items():
            for ar, arm, _ in items:
                sink.start_experiment(ar, {"arm": name, "model": arm.model, "prompt": arm.prompt,
                                           "repeat": ar.repeat, "dataset_version": dataset.version})
    except Exception as exc:  # a LangSmith outage should not block the decision
        log(f"  LangSmith unavailable, running without tracing: {exc}")
        caveats.append(f"LangSmith was unavailable ({type(exc).__name__}); runs were not traced and the "
                       "report has no trace links")
        sink = NullSink()
        for items in pending.values():
            for ar, _, _ in items:
                ar.experiment = ar.experiment_url = None
    tasks = interleave(pending)
    cached_calls = sum(len(ar.cached_ids) for ar in unique_runs.values())
    log(f"Running {len(tasks)} target calls ({cached_calls} served from cache)")
    Runner(target, cache, sink, concurrency=cfg.concurrency, retries=cfg.retries, log=log).run(tasks)
    if not isinstance(sink, NullSink):
        _backfill_usage(unique_runs.values(), sink, cache, plans)

    scorer = Scorer(evaluators, judge, cache, sink, concurrency=cfg.concurrency, log=log)
    scores = scorer.score(list(unique_runs.values()), dataset)

    kinds = detect_kinds(list(scores.values()), cfg.metrics.kinds)
    roles = cfg.metrics.roles()
    if cfg.metrics.primary not in kinds:
        produced = sorted(kinds) or ["<none>"]
        raise ConfigError(f"primary metric {cfg.metrics.primary!r} was not produced by any evaluator "
                          f"(produced: {', '.join(produced)})")
    for m in roles:
        if m not in kinds:
            caveats.append(f"metric {m!r} was not produced by any evaluator")
    metric_names = [m for m in roles if m in kinds]
    ids = [e.id for e in dataset.examples]
    base_runs = plans["baseline"].runs
    base_mats = {m: score_matrix(base_runs, scores, ids, m) for m in metric_names}
    base_behavior = arm_behavior(base_runs, ids, is_refusal, format_ok)
    base_cost = arm_cost(base_runs, pricing, cfg.traffic.monthly_queries)
    base_latency = arm_latency(base_runs)

    comparisons = []
    for name in (n for n in CANDIDATE_ARMS if n in plans):
        cand_runs = plans[name].runs
        cand_mats = {m: score_matrix(cand_runs, scores, ids, m) for m in metric_names}
        metrics = [compare_metric(m, roles[m], kinds[m], base_mats[m], cand_mats[m], cfg.decision)
                   for m in metric_names]
        slice_metrics = [(m, roles[m], kinds[m]) for m in metric_names if roles[m] != "tracked"]
        slices = compare_slices(dataset.examples, cfg.slice_by, slice_metrics, base_mats, cand_mats, cfg.decision)
        behavior = compare_behavior(base_behavior, arm_behavior(cand_runs, ids, is_refusal, format_ok),
                                    cfg.behavior_flags)
        cost = compare_cost(base_cost, arm_cost(cand_runs, pricing, cfg.traffic.monthly_queries),
                            cfg.traffic.monthly_queries, cfg.behavior_flags)
        latency = compare_latency(base_latency, arm_latency(cand_runs), cfg.behavior_flags)
        pairwise = None
        if cfg.behavior.pairwise.enabled:
            if judge.available:
                log(f"Pairwise judge on {cfg.behavior.pairwise.sample_size} examples ({name})")
                pairwise = pairwise_preference(dataset.examples, base_runs, cand_runs, judge, cache,
                                               sample_size=cfg.behavior.pairwise.sample_size,
                                               seed=cfg.decision.seed, resamples=cfg.decision.bootstrap_resamples,
                                               concurrency=cfg.concurrency)
            else:
                caveats.append("pairwise judge is enabled but `judge.client` is not configured; skipped")
        verdict = decide(metrics, slices, behavior, cost, latency, pairwise, cfg.decision)
        comparisons.append({
            "arm": name,
            "model": plans[name].config.model,
            "prompt": plans[name].config.prompt,
            "verdict": verdict.verdict,
            "verdict_label": verdict.label,
            "verdict_reasons": verdict.reasons,
            "client_reasons": verdict.client_reasons,
            "warnings": verdict.warnings,
            "metrics": [m.to_json() for m in metrics],
            "flips": {m.name: m.flips.to_json() for m in metrics if m.flips is not None},
            "slices": [s.to_json() for s in slices],
            "behavior": behavior,
            "cost": cost,
            "latency": latency,
            "pairwise": pairwise,
            "examples": notable_examples(dataset.examples, metrics, base_mats, cand_mats, base_runs, cand_runs,
                                         cfg.report.max_examples, cfg.slice_by),
            "curated_examples": _curated_examples(cfg, dataset, base_mats, cand_mats, base_runs, cand_runs),
        })

    best = recommend(comparisons)
    eval_cost = eval_run_cost(list(unique_runs.values()), pricing, judge.model, judge.usage())
    caveats += _general_caveats(cfg, dataset, plans, scores, eval_cost, comparisons, opts)
    cache.close()
    sink.flush()
    if getattr(sink, "feedback_errors", 0):
        caveats.append("Some scores could not be mirrored to LangSmith as feedback; the report is unaffected.")

    report = _build_report(cfg, dataset, plans, comparisons, best, judge, eval_cost, caveats, opts)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = opts.out_dir or (base / "reports")
    report_dir = out_root / (f"aa-{stamp}" if opts.aa else stamp)
    formats = opts.formats or cfg.report.formats
    paths = write_outputs(report, report_dir, formats, redact_inputs=cfg.report.redact_inputs)
    exit_code = EXIT_CODES[best["verdict"]]
    _write_github_outputs({"report_dir": str(report_dir.resolve()), "verdict": best["verdict"],
                           "exit_code": str(exit_code)})
    return RunResult(exit_code=exit_code, report=report, report_dir=report_dir, paths=paths)


def _curated_examples(cfg: Config, dataset: Dataset, base_mats: dict, cand_mats: dict, base_runs: list[ArmRun],
                      cand_runs: list[ArmRun]) -> list[dict]:
    """Examples hand-picked for the client report via `report.client_examples`."""
    if not cfg.report.client_examples:
        return []
    primary = cfg.metrics.primary
    b, c = per_example_mean(base_mats[primary]), per_example_mean(cand_mats[primary])
    index = {e.id: i for i, e in enumerate(dataset.examples)}
    out = []
    for eid in cfg.report.client_examples:
        if eid not in index:
            raise ConfigError(f"report.client_examples: example {eid!r} is not in the dataset")
        i = index[eid]
        brec = next((ar.records[eid] for ar in base_runs if eid in ar.records and ar.records[eid].ok), None)
        crec = next((ar.records[eid] for ar in cand_runs if eid in ar.records and ar.records[eid].ok), None)
        bs, cs = _finite(b[i]), _finite(c[i])
        direction = "unchanged"
        if bs is not None and cs is not None and cs != bs:
            direction = "improved" if cs > bs else "regressed"
        out.append({"example_id": eid, "metric": primary, "direction": direction, "baseline_score": bs,
                    "candidate_score": cs, "inputs": dataset.examples[i].inputs,
                    "reference": dataset.examples[i].outputs, "tags": dataset.examples[i].tags(cfg.slice_by),
                    "baseline_answer": brec.answer if brec else None,
                    "candidate_answer": crec.answer if crec else None})
    return out


def _finite(x) -> float | None:
    return None if x is None or x != x else float(x)


def _backfill_usage(runs, sink, cache: Cache, plans: dict[str, _ArmPlan]) -> None:
    """Take token counts from LangSmith traces for calls that reported none."""
    models = {ar.key: plan.config.model for plan in plans.values() for ar in plan.runs}
    for ar in runs:
        missing = [r for r in ar.executed_records() if r.ok and not r.usage and r.run_id]
        if not missing:
            continue
        found = sink.token_usage(ar, [r.run_id for r in missing])
        for rec in missing:
            if rec.run_id in found:
                tin, tout = found[rec.run_id]
                rec.usage = {models[ar.key]: {"input_tokens": tin, "output_tokens": tout}}
                cache.save_record(ar.key, ar.key_parts, rec)


def _general_caveats(cfg: Config, dataset: Dataset, plans: dict[str, _ArmPlan], scores: dict, eval_cost: dict,
                     comparisons: list[dict], opts: RunOptions) -> list[str]:
    out = [
        f"Judge: {cfg.judge.model} (temperature {cfg.judge.temperature}), pinned and identical across arms.",
        f"Dataset: {dataset.name}, content version {dataset.version}"
        + (f", as of {dataset.as_of}" if dataset.as_of else "") + f", {len(dataset.examples)} examples.",
        f"Repeats: {cfg.repeats} per arm"
        + ("; with a single repeat there is no noise floor and differences are judged by the CI alone."
           if cfg.repeats < 2 else "."),
    ]
    for c in comparisons:
        primary = next(m for m in c["metrics"] if m["role"] == "primary")
        if primary["mde"] is not None:
            out.append(f"Minimum detectable effect on {primary['name']} ({c['arm']}): {primary['mde']:.3f} "
                       f"vs margin {abs(cfg.decision.primary_margin):.3f}.")
    for name, plan in plans.items():
        errors = sum(1 for ar in plan.runs for r in ar.records.values() if not r.ok)
        total = sum(len(ar.records) for ar in plan.runs)
        if errors:
            out.append(f"{name}: {errors} of {total} target calls failed and were left out of the metrics.")
    evaluator_errors: dict[str, list[str]] = {}
    for per_run in scores.values():
        for per_example in per_run.values():
            for ev, msg in per_example.get("_errors", {}).items():
                evaluator_errors.setdefault(ev, []).append(msg)
    for ev, msgs in sorted(evaluator_errors.items()):
        out.append(f"Evaluator {ev} failed on {len(msgs)} outputs (first error: {msgs[0][:160]}).")
    for c in comparisons:
        missing = sorted(set(c["cost"]["baseline"]["missing_pricing"]) | set(c["cost"]["candidate"]["missing_pricing"]))
        if missing:
            out.append(f"No price in {cfg.pricing_file} for: {', '.join(missing)}; cost for {c['arm']} is incomplete.")
        if c["cost"]["candidate"]["calls_with_usage"] == 0 and c["cost"]["candidate"]["calls"]:
            out.append("The target reported no token usage, so cost could not be computed "
                       "(return `usage` or call `upgrade_report.record_usage`).")
    usd = eval_cost["usd"]
    out.append(
        f"This eval run cost ${usd:,.2f} ({eval_cost['target_calls']} target calls, "
        f"{eval_cost['judge_calls']} judge calls"
        + (f"; unpriced: {', '.join(eval_cost['unpriced_models'])}" if eval_cost["unpriced_models"] else "")
        + ")."
    )
    if opts.aa:
        out.append("A/A run: the candidate is the baseline configuration with independent repeats.")
    return out


def _build_report(cfg: Config, dataset: Dataset, plans: dict[str, _ArmPlan], comparisons: list[dict],
                  best: dict, judge: Judge, eval_cost: dict, caveats: list[str], opts: RunOptions) -> dict:
    arms = {}
    for name, plan in plans.items():
        arms[name] = {
            "model": plan.config.model,
            "prompt": plan.config.prompt,
            "prompt_hash": plan.prompt_hash,
            "target_version": plan.target_version,
            "runs": [
                {"repeat": ar.repeat, "run_key": ar.key, "cached": len(ar.cached_ids),
                 "executed": len(ar.executed_records()), "errors": sum(1 for r in ar.records.values() if not r.ok),
                 "experiment": ar.experiment, "experiment_url": ar.experiment_url}
                for ar in plan.runs
            ],
        }
    mirrored = {k: best[k] for k in ("verdict", "verdict_reasons", "metrics", "flips", "slices", "behavior", "cost",
                                     "latency", "pairwise", "examples")}
    return {
        "schema_version": 1,
        "tool_version": __version__,
        "project": cfg.project,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_hash": cfg.canonical_hash(),
        "mode": "aa" if opts.aa else "compare",
        "baseline_ref": opts.baseline_ref,
        "dataset": {"name": dataset.name, "source": dataset.source, "version": dataset.version,
                    "as_of": dataset.as_of, "examples": len(dataset.examples)},
        "judge": {**cfg.judge.describe(), **judge.usage()},
        "repeats": cfg.repeats,
        "decision": cfg.decision.model_dump(mode="json"),
        "behavior_flags": cfg.behavior_flags.model_dump(mode="json"),
        "metric_descriptions": _metric_descriptions(cfg),
        "arms": arms,
        "recommended_arm": best["arm"],
        "verdict_label": LABELS[best["verdict"]],
        **mirrored,
        "warnings": best["warnings"],
        "comparisons": comparisons,
        "eval_cost": eval_cost,
        "caveats": caveats,
        "notes": {"multiple_comparisons": MULTIPLE_COMPARISONS_NOTE},
        "client": {"title": cfg.report.client_title, "examples": cfg.report.client_examples},
    }


def _metric_descriptions(cfg: Config) -> dict[str, str]:
    from . import judges

    out = {}
    for ref in cfg.evaluators:
        if ref.startswith("registry:"):
            name = ref.split(":", 1)[1]
            if desc := judges.description(name):
                out[name] = desc
    out.update(cfg.metrics.descriptions)
    return out
