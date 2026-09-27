"""Snapshot tests for the rendered reports.

Regenerate after an intended template change with:
    UPDATE_SNAPSHOTS=1 pytest tests/test_render.py
"""

import copy
import json
import os
from pathlib import Path

import pytest

from upgrade_report.render import render_client_html, render_markdown
from upgrade_report.render.render import MAX_COMMENT_CHARS

FIXTURE = Path(__file__).parent / "fixtures" / "report_sample.json"
SNAPSHOTS = Path(__file__).parent / "snapshots"


@pytest.fixture
def report():
    return json.loads(FIXTURE.read_text())


def _check(name: str, text: str) -> None:
    path = SNAPSHOTS / name
    if os.environ.get("UPDATE_SNAPSHOTS") or not path.exists():
        path.write_text(text)
    assert text == path.read_text(), f"{name} changed; rerun with UPDATE_SNAPSHOTS=1 if intended"


def test_markdown_snapshot(report):
    _check("report.md", render_markdown(report))


def test_client_html_snapshot(report):
    _check("report_client.html", render_client_html(report))


def test_client_html_has_no_internal_ids(report):
    html = render_client_html(report)
    assert "smith.langchain.com" not in html
    assert report["config_hash"] not in html
    for ex in report["examples"]:
        assert ex["example_id"] not in html
    for arm in report["arms"].values():
        assert arm["prompt_hash"] not in html
        assert all(r["run_key"] not in html for r in arm["runs"])


def test_client_html_redacts_inputs(report):
    html = render_client_html(report, redact_inputs=True)
    assert "[redacted]" in html
    assert "What is the value for item" not in html


def test_markdown_links_traces_and_marks_the_comment(report):
    md = render_markdown(report)
    assert md.startswith("<!-- upgrade-report:test-project -->")
    assert "[baseline trace](https://smith.langchain.com" in md
    assert "pass → fail" in md or "largest drops" in md
    assert md.index("### Flipped cases") < md.index("### Cost and latency") < md.index("### Caveats")


def test_markdown_fits_in_a_pr_comment(report):
    big = copy.deepcopy(report)
    comparison = next(c for c in big["comparisons"] if c["arm"] == big["recommended_arm"])
    template = comparison["examples"][0]
    comparison["examples"] = [dict(template, example_id=f"x{i}", candidate_answer="word " * 400,
                                   baseline_answer="word " * 400, total=500) for i in range(500)]
    assert len(render_markdown(big)) <= MAX_COMMENT_CHARS
