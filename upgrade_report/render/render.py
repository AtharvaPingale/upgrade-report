"""Render report.json into report.md (internal, PR comment) and report_client.html (external)."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape

from ..judges import question_text, reference_text

# GitHub rejects comments over 65,536 characters.
MAX_COMMENT_CHARS = 60_000
MINUS = "−"

STATUS_ICONS = {
    "fail": "❌",
    "regressed": "⚠️",
    "within_noise": "≈",
    "improved": "✅",
    "pass": "✅",
    "missing": "❔",
}
STATUS_TEXT = {
    "fail": "fail",
    "regressed": "worse, not blocking",
    "within_noise": "within noise",
    "improved": "improved",
    "pass": "pass",
    "missing": "no data",
}
CLIENT_STATUS_TEXT = {
    "fail": "Worse",
    "regressed": "Slightly worse, within tolerance",
    "within_noise": "No real change",
    "improved": "Better",
    "pass": "No meaningful drop",
    "missing": "Not measured",
}


def client_status(status: str, role: str) -> str:
    if status == "regressed" and role == "tracked":
        return "Worse (informational)"
    if status == "pass" and role == "tracked":
        return "No clear change"
    return CLIENT_STATUS_TEXT.get(status, status)
VERDICT_ICONS = {"ship": "✅", "ship_with_caveats": "⚠️", "dont_ship": "❌"}


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def _minus(s: str) -> str:
    return s.replace("-", MINUS)


def f_num(x: Any, digits: int = 3) -> str:
    return _minus(f"{x:.{digits}f}") if _is_num(x) else "—"


def f_signed(x: Any, digits: int = 3) -> str:
    return _minus(f"{x:+.{digits}f}") if _is_num(x) else "—"


def f_ci(ci: Any, digits: int = 3) -> str:
    if not ci or not all(_is_num(v) for v in ci):
        return "—"
    return f"[{f_signed(ci[0], digits)}, {f_signed(ci[1], digits)}]"


def f_rate(x: Any, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%" if _is_num(x) else "—"


def f_change_pct(x: Any) -> str:
    return _minus(f"{x:+.0f}%") if _is_num(x) else "—"


def f_pts(x: Any) -> str:
    return _minus(f"{x:+.1f} pts") if _is_num(x) else "—"


def f_usd(x: Any) -> str:
    if not _is_num(x):
        return "—"
    if abs(x) >= 100:
        return _minus(f"${x:,.0f}")
    if abs(x) >= 1:
        return _minus(f"${x:,.2f}")
    return _minus(f"${x:.4f}")


def f_usd_delta(x: Any) -> str:
    if not _is_num(x):
        return "—"
    sign = "+" if x >= 0 else MINUS
    return sign + f_usd(abs(x))


def f_secs(x: Any) -> str:
    return f"{x:.2f} s" if _is_num(x) else "—"


def f_cell(text: Any, limit: int = 160) -> str:
    """Safe inside a Markdown table cell."""
    s = "" if text is None else str(text)
    s = " ".join(s.split())
    if len(s) > limit:
        s = s[: limit - 1] + "…"
    return s.replace("|", "\\|")


def f_block(text: Any, limit: int = 800) -> str:
    """Quoted multi-line block for Markdown."""
    s = "" if text is None else str(text).strip()
    if len(s) > limit:
        s = s[: limit - 1] + "…"
    return "\n".join("> " + line if line else ">" for line in s.splitlines()) or "> _(empty)_"


def f_truncate(text: Any, limit: int = 1200) -> str:
    s = "" if text is None else str(text).strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"


def f_metric_name(name: str) -> str:
    return name.replace("_", " ").capitalize() if "@" not in name else name


def _environment() -> Environment:
    env = Environment(
        loader=PackageLoader("upgrade_report", "render/templates"),
        autoescape=select_autoescape(["html", "html.j2"]),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.filters.update({
        "num": f_num, "signed": f_signed, "ci": f_ci, "rate": f_rate, "change_pct": f_change_pct, "pts": f_pts,
        "usd": f_usd, "usd_delta": f_usd_delta, "secs": f_secs, "cell": f_cell, "block": f_block,
        "truncate_text": f_truncate, "question": lambda inputs: question_text(inputs or {}),
        "reference": lambda outputs: reference_text(outputs), "metric_name": f_metric_name,
        "status_icon": lambda s: STATUS_ICONS.get(s, ""), "status_text": lambda s: STATUS_TEXT.get(s, s),
        "client_status": client_status,
    })
    env.globals.update({"VERDICT_ICONS": VERDICT_ICONS})
    return env


def _comparison(report: dict) -> dict:
    best = report["recommended_arm"]
    return next(c for c in report["comparisons"] if c["arm"] == best)


def render_markdown(report: dict) -> str:
    env = _environment()
    template = env.get_template("report.md.j2")
    limit = None
    while True:
        text = template.render(r=report, c=_comparison(report), example_limit=limit)
        if len(text) <= MAX_COMMENT_CHARS:
            return text
        limit = (limit or 20) // 2
        if limit == 0:
            return text[: MAX_COMMENT_CHARS - 200] + "\n\n_Report truncated to fit a PR comment; see report.json._\n"


def _client_examples(report: dict, comparison: dict, redact: bool) -> list[dict]:
    curated_ids = (report.get("client") or {}).get("examples") or []
    pool = comparison.get("curated_examples", []) + comparison.get("examples", [])
    primary = next(m["name"] for m in comparison["metrics"] if m["role"] == "primary")
    chosen: list[dict] = []
    if curated_ids:
        by_id: dict[str, dict] = {}
        for ex in pool:
            by_id.setdefault(ex["example_id"], ex)
        chosen = [by_id[i] for i in curated_ids if i in by_id][:5]
    else:
        on_primary = [e for e in pool if e["metric"] == primary] or pool
        improved = [e for e in on_primary if e["direction"] == "improved"][:2]
        regressed = [e for e in on_primary if e["direction"] == "regressed"][:2]
        chosen = improved + regressed
        if len(chosen) < 3:
            used = {e["example_id"] for e in chosen}
            chosen += [e for e in pool if e["example_id"] not in used][: 3 - len(chosen)]
    out = []
    for ex in chosen:
        ex = copy.deepcopy(ex)
        ex["question"] = "[redacted]" if redact else question_text(ex.get("inputs") or {})
        for k in ("example_id", "baseline_trace", "candidate_trace", "inputs"):
            ex.pop(k, None)
        out.append(ex)
    return out


def render_client_html(report: dict, redact_inputs: bool = False) -> str:
    env = _environment()
    comparison = _comparison(report)
    return env.get_template("report_client.html.j2").render(
        r=report, c=comparison, examples=_client_examples(report, comparison, redact_inputs),
    )


def write_outputs(report: dict, out_dir: Path, formats: list[str], redact_inputs: bool = False) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"json": out_dir / "report.json"}
    paths["json"].write_text(json.dumps(report, indent=2, default=str) + "\n")
    if "md" in formats:
        paths["md"] = out_dir / "report.md"
        paths["md"].write_text(render_markdown(report))
    if "html" in formats:
        paths["html"] = out_dir / "report_client.html"
        paths["html"].write_text(render_client_html(report, redact_inputs))
    return paths
