"""Command line interface.

Exit codes: 0 Ship, 10 Ship with caveats, 20 Don't ship, 2 tool or config error.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import sys
import traceback
from importlib import resources
from pathlib import Path
from typing import Annotated, Optional

import typer

from .config import load_config
from .errors import UpgradeReportError
from .verdict import LABELS

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Compare a baseline and a candidate model/prompt configuration and get a ship verdict.",
)

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Path to config.yaml.")]
OutOpt = Annotated[Optional[Path], typer.Option("--out", "-o", help="Reports directory (default: ./reports).")]
NoCacheOpt = Annotated[bool, typer.Option("--no-cache", help="Ignore cached runs and scores (still writes them).")]
FormatOpt = Annotated[Optional[list[str]], typer.Option("--format", "-f", help="Output formats: md, json, html.")]


def _err(msg: str) -> None:
    typer.echo(f"error: {msg}", err=True)


def guarded(fn):
    """Map tool/config errors (and anything unexpected) to exit code 2."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except typer.Exit:
            raise
        except UpgradeReportError as exc:
            _err(str(exc))
            raise typer.Exit(2) from None
        except KeyboardInterrupt:
            _err("interrupted; finished calls are cached and will be reused next run")
            raise typer.Exit(2) from None
        except Exception:
            _err("unexpected failure")
            traceback.print_exc()
            raise typer.Exit(2) from None

    return wrapper


def _check_formats(formats: list[str] | None) -> list[str] | None:
    if formats is None:
        return None
    bad = [f for f in formats if f not in ("md", "json", "html")]
    if bad:
        raise UpgradeReportError(f"unknown format(s): {', '.join(bad)} (use md, json, html)")
    return formats


def _print_estimate(est: dict) -> None:
    ds = est["dataset"]
    typer.echo(f"Config OK. Dataset {ds['name']}: {ds['examples']} examples, version {ds['version']}.")
    typer.echo("")
    typer.echo(f"{'arm':<20}{'model':<28}{'to run':>8}{'cached':>8}  {'tokens/call in/out':<26}{'est. cost':>10}")
    for row in est["arms"]:
        tokens = f"{row['tokens_per_call'][0]}/{row['tokens_per_call'][1]} ({row['token_source']})"
        usd = f"${row['usd']:.2f}" if row["usd"] is not None else "no price"
        typer.echo(f"{row['arm']:<20}{row['model']:<28}{row['calls_to_run']:>8}{row['cached_calls']:>8}  "
                   f"{tokens:<26}{usd:>10}")
    judge_usd = f"${est['judge_usd']:.2f}" if est["judge_usd"] is not None else "no price"
    typer.echo(f"{'judge':<20}{'':<28}{est['judge_calls']:>8}{'':>8}  {'':<26}{judge_usd:>10}")
    total = f"${est['total_usd']:.2f}" if est["total_usd"] is not None else "unknown (missing prices)"
    typer.echo(f"\nEstimated cost of this run: {total}")


def _print_result(result) -> None:
    report = result.report
    typer.echo(f"{LABELS[report['verdict']]}: {report['verdict_reasons'][0]}")
    for extra in report["verdict_reasons"][1:]:
        typer.echo(f"  - {extra}")
    if len(report["comparisons"]) > 1:
        typer.echo(f"Recommended arm: {report['recommended_arm']}")
    for kind, path in result.paths.items():
        typer.echo(f"{kind:>4}: {path}")


def _run(config: Path, *, out: Path | None, dry_run: bool, baseline_ref: str | None, no_cache: bool,
         formats: list[str] | None, aa: bool) -> None:
    from .pipeline import RunOptions, run

    cfg = load_config(config)
    result = run(cfg, RunOptions(out_dir=out, dry_run=dry_run, baseline_ref=baseline_ref, use_cache=not no_cache,
                                 formats=_check_formats(formats), aa=aa))
    if dry_run:
        _print_estimate(result.estimate)
        raise typer.Exit(0)
    _print_result(result)
    if aa:
        if result.report["verdict"] == "ship":
            typer.echo("A/A calibration passed: baseline vs itself is Ship.")
        else:
            typer.echo("A/A calibration FAILED: baseline vs itself should be Ship. The tool is too noisy for this "
                       "dataset and config; add repeats or examples, or review the reasons above.")
    raise typer.Exit(result.exit_code)


@app.command()
@guarded
def run(
    config: ConfigOpt = Path("config.yaml"),
    out: OutOpt = None,
    dry_run: Annotated[bool, typer.Option(
        "--dry-run", help="Validate config and estimate eval cost; call nothing.")] = False,
    baseline_ref: Annotated[Optional[str], typer.Option(
        "--baseline-ref",
        help="Git ref whose cached baseline to use (e.g. origin/main), so code changes are compared.")] = None,
    no_cache: NoCacheOpt = False,
    format: FormatOpt = None,
) -> None:
    """Run baseline and candidate, compare them, and write the report."""
    _run(config, out=out, dry_run=dry_run, baseline_ref=baseline_ref, no_cache=no_cache, formats=format, aa=False)


@app.command()
@guarded
def aa(config: ConfigOpt = Path("config.yaml"), out: OutOpt = None, no_cache: NoCacheOpt = False,
       format: FormatOpt = None) -> None:
    """A/A calibration: compare the baseline with independent repeats of itself. Should be Ship."""
    _run(config, out=out, dry_run=False, baseline_ref=None, no_cache=no_cache, formats=format, aa=True)


@app.command()
@guarded
def render(
    report: Annotated[Path, typer.Argument(help="Path to report.json.")],
    format: Annotated[str, typer.Option("--format", "-f", help="md or html (client version).")] = "md",
    out: Annotated[Optional[Path], typer.Option("--out", "-o", help="Output file, or '-' for stdout.")] = None,
    redact_inputs: Annotated[bool, typer.Option("--redact-inputs", help="Hide questions in the client HTML.")] = False,
) -> None:
    """Re-render a report.json as Markdown or client HTML."""
    from .render import render_client_html, render_markdown

    if not report.exists():
        raise UpgradeReportError(f"report not found: {report}")
    try:
        data = json.loads(report.read_text())
    except json.JSONDecodeError as exc:
        raise UpgradeReportError(f"cannot parse {report}: {exc}") from exc
    if format == "md":
        text, default = render_markdown(data), report.with_name("report.md")
    elif format == "html":
        text, default = render_client_html(data, redact_inputs), report.with_name("report_client.html")
    elif format == "json":
        text, default = json.dumps(data, indent=2) + "\n", report
    else:
        raise UpgradeReportError(f"unknown format {format!r} (use md, html or json)")
    if out is not None and str(out) == "-":
        sys.stdout.write(text)
        return
    target = out or default
    target.write_text(text)
    typer.echo(str(target))


def _template_dir() -> Path:
    packaged = resources.files("upgrade_report") / "_project_template"
    if packaged.is_dir():
        return Path(str(packaged))
    return Path(__file__).resolve().parents[1] / "templates" / "project"


@app.command()
@guarded
def init(
    directory: Annotated[Path, typer.Argument(help="Project directory to scaffold into.")] = Path("."),
    force: Annotated[bool, typer.Option("--force", help="Overwrite existing files.")] = False,
) -> None:
    """Scaffold config.yaml, pricing.yaml, a target stub, example evaluators and CI workflows."""
    src = _template_dir()
    if not src.is_dir():
        raise UpgradeReportError(f"project template not found at {src}")
    files = [p for p in src.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    clashes = [p.relative_to(src) for p in files if (directory / p.relative_to(src)).exists()]
    if clashes and not force:
        raise UpgradeReportError("would overwrite: " + ", ".join(map(str, clashes)) + " (use --force)")
    for p in files:
        dest = directory / p.relative_to(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        typer.echo(f"created {dest}")
    typer.echo(
        "\nNext steps:\n"
        "  1. Implement upgrade_eval/target.py:run() so it calls your application.\n"
        "  2. Point `dataset` in config.yaml at your LangSmith dataset (or keep the sample file).\n"
        "  3. Set baseline/candidate models and prompts, metrics, and pricing.yaml.\n"
        "  4. upgrade-report run --dry-run   # validate and estimate cost\n"
        "  5. upgrade-report run\n"
    )


@app.command()
@guarded
def scheduled(
    config: ConfigOpt = Path("config.yaml"),
    out: OutOpt = None,
    include_existing: Annotated[bool, typer.Option(
        "--include-existing",
        help="On the first run, evaluate every listed model instead of only recording them.")] = False,
) -> None:
    """Evaluate models released since the last scheduled run as candidates (notify only)."""
    from .scheduled import run_scheduled

    cfg = load_config(config)
    result = run_scheduled(cfg, out_dir=out, include_existing=include_existing,
                           log=lambda m: typer.echo(m, err=True))
    typer.echo(result.summary_md)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write(result.summary_md)
    if any(r["verdict"] == "error" for r in result.results):
        raise typer.Exit(2)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
