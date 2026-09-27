"""CLI behavior and CI exit codes: 0 Ship, 10 caveats, 20 Don't ship, 2 error."""

import json

import yaml
from conftest import base_config, write_config
from typer.testing import CliRunner

from upgrade_report.cli import app

runner = CliRunner()


def invoke(*args):
    return runner.invoke(app, [str(a) for a in args])


def test_exit_code_ship(tmp_path, no_network):
    cfg = write_config(tmp_path, base_config(tmp_path, n=300, resamples=1000))
    result = invoke("run", "-c", cfg)
    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("Ship:")


def test_exit_code_caveats(tmp_path, no_network):
    cfg = write_config(tmp_path, base_config(tmp_path, candidate="fake-verbose"))
    assert invoke("run", "-c", cfg).exit_code == 10


def test_exit_code_dont_ship(tmp_path, no_network):
    cfg = write_config(tmp_path, base_config(
        tmp_path, n=300, candidate="fake-regress10", resamples=1000,
        evaluators=["fake_evaluators:correct"], metrics={"primary": "correct"},
        decision={"primary_margin": -0.02}))
    assert invoke("run", "-c", cfg).exit_code == 20


def test_config_errors_exit_2(tmp_path, no_network):
    assert invoke("run", "-c", tmp_path / "missing.yaml").exit_code == 2
    cfg = base_config(tmp_path)
    cfg["decision"]["primary_margin"] = 0.05
    result = invoke("run", "-c", write_config(tmp_path, cfg))
    assert result.exit_code == 2 and "primary_margin" in result.output
    cfg = base_config(tmp_path)
    cfg["metrics"]["primary"] = "not_a_metric"
    result = invoke("run", "-c", write_config(tmp_path, cfg))
    assert result.exit_code == 2 and "not produced by any evaluator" in result.output


def test_dry_run_calls_nothing(tmp_path, no_network):
    import fake_target

    cfg = write_config(tmp_path, base_config(tmp_path))
    result = invoke("run", "-c", cfg, "--dry-run")
    assert result.exit_code == 0, result.output
    assert "Estimated cost of this run" in result.output
    assert fake_target.STATE["calls"] == 0


def test_github_outputs(tmp_path, no_network, monkeypatch):
    out = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    invoke("run", "-c", write_config(tmp_path, base_config(tmp_path, candidate="fake-verbose")))
    values = dict(line.split("=", 1) for line in out.read_text().splitlines())
    assert values["verdict"] == "ship_with_caveats" and values["exit_code"] == "10"


def test_render_command(tmp_path, no_network):
    invoke("run", "-c", write_config(tmp_path, base_config(tmp_path, report={"formats": ["json"]})))
    report = next((tmp_path / "reports").glob("*/report.json"))
    result = invoke("render", report, "--format", "html")
    assert result.exit_code == 0
    assert report.with_name("report_client.html").exists()
    md = invoke("render", report, "--format", "md", "--out", "-")
    assert md.stdout.startswith("<!-- upgrade-report:")
    assert invoke("render", tmp_path / "nope.json").exit_code == 2


def test_aa_command(tmp_path, no_network):
    cfg = write_config(tmp_path, base_config(tmp_path, n=300, resamples=1000))
    result = invoke("aa", "-c", cfg)
    assert result.exit_code == 0, result.output
    assert "A/A calibration passed" in result.output


def test_init_scaffolds_a_runnable_project(tmp_path, no_network):
    result = invoke("init", tmp_path)
    assert result.exit_code == 0, result.output
    for rel in ("config.yaml", "pricing.yaml", "upgrade_eval/target.py", "upgrade_eval/evaluators.py",
                "upgrade_eval/dataset.jsonl", ".github/workflows/model-upgrade.yml"):
        assert (tmp_path / rel).exists(), rel
    assert invoke("init", tmp_path).exit_code == 2  # refuses to overwrite
    assert invoke("run", "-c", tmp_path / "config.yaml", "--dry-run").exit_code == 0
    result = invoke("run", "-c", tmp_path / "config.yaml")
    assert result.exit_code in (0, 10), result.output
    report = json.loads(next((tmp_path / "reports").glob("*/report.json")).read_text())
    assert report["project"] == yaml.safe_load((tmp_path / "config.yaml").read_text())["project"]
