"""Content-hash versioning (incl. --baseline-ref) and scheduled mode."""

import json
import shutil
import subprocess
from pathlib import Path

import fake_target
import pytest
from conftest import FIXTURES, base_config, write_config

from upgrade_report.config import load_config
from upgrade_report.pipeline import RunOptions, run
from upgrade_report.scheduled import run_scheduled
from upgrade_report.versioning import code_version, git_blob_sha, prompt_hash

QUIET = RunOptions(log=lambda m: None)
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def init_repo(repo: Path) -> None:
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "test")


@needs_git
def test_blob_sha_matches_git(tmp_path):
    f = tmp_path / "x.py"
    f.write_bytes(b"print('hi')\n")
    init_repo(tmp_path)
    assert git_blob_sha(f.read_bytes()) == git(tmp_path, "hash-object", str(f)).strip()


@needs_git
def test_worktree_and_git_ref_hash_the_same_content(tmp_path):
    init_repo(tmp_path)
    pkg = tmp_path / "app"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "retrieval.py").write_text("K = 5\n")
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "answer.txt").write_text("Be brief.\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "init")
    v_head = code_version([pkg], tmp_path)
    p_head = prompt_hash("prompts/answer@v1", tmp_path)
    assert v_head == code_version([pkg], tmp_path, ref="main")
    assert p_head == prompt_hash("prompts/answer@v1", tmp_path, ref="main")

    (pkg / "retrieval.py").write_text("K = 10\n")
    (pkg / "scratch.pyc").write_bytes(b"\0")  # untracked build artefacts never count
    (tmp_path / "prompts" / "answer.txt").write_text("Be thorough.\n")
    assert code_version([pkg], tmp_path) != v_head
    assert code_version([pkg], tmp_path, ref="main") == v_head
    assert prompt_hash("prompts/answer@v1", tmp_path) != p_head
    assert prompt_hash("prompts/answer@v1", tmp_path, ref="main") == p_head


def test_prompt_hash_without_a_local_file_uses_the_reference():
    here = Path(".")
    assert prompt_hash("hub/answer:v12", here) != prompt_hash("hub/answer:v13", here)


@needs_git
def test_baseline_ref_reuses_mains_cached_baseline(tmp_path, no_network):
    init_repo(tmp_path)
    (tmp_path / "app").mkdir()
    shutil.copy(FIXTURES / "fake_target.py", tmp_path / "app" / "__init__.py")
    cfg = base_config(tmp_path, target="app:run", candidate="fake-good-b", n=40)
    write_config(tmp_path, cfg)
    (tmp_path / ".gitignore").write_text(".upgrade-report/\nreports/\n__pycache__/\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "main")
    import sys

    sys.modules.pop("app", None)
    run(load_config(tmp_path / "config.yaml"), QUIET)  # main's CI run caches the baseline

    git(tmp_path, "checkout", "-qb", "feature")
    target = tmp_path / "app" / "__init__.py"
    target.write_text(target.read_text() + "\n# retrieval tweak\n")
    git(tmp_path, "commit", "-qam", "change code")
    import app

    before = app.STATE["calls"]
    result = run(load_config(tmp_path / "config.yaml"), RunOptions(baseline_ref="main", log=lambda m: None))
    assert app.STATE["calls"] - before == 40 * 3  # only the candidate ran
    arms = result.report["arms"]
    assert arms["baseline"]["target_version"] != arms["candidate"]["target_version"]
    assert all(r["executed"] == 0 for r in arms["baseline"]["runs"])

    fallback = run(load_config(tmp_path / "config.yaml"), RunOptions(baseline_ref="HEAD", log=lambda m: None))
    assert any("was not in the cache" in c for c in fallback.report["caveats"])
    sys.modules.pop("app", None)


MODELS = {"available": ["fake-good-a", "fake-good-b"]}


def list_models():
    return MODELS["available"]


def test_scheduled_mode_evaluates_only_new_releases(tmp_path, no_network):
    MODELS["available"] = ["fake-good-a", "fake-good-b"]
    cfg_dict = base_config(tmp_path, schedule={"model_source": "test_versioning_and_scheduled:list_models",
                                               "model_filter": "^fake-"})
    cfg = load_config(write_config(tmp_path, cfg_dict))
    first = run_scheduled(cfg, out_dir=tmp_path / "reports", log=lambda m: None)
    assert first.first_run and first.new_models == [] and fake_target.STATE["calls"] == 0

    MODELS["available"].append("fake-verbose")
    second = run_scheduled(cfg, out_dir=tmp_path / "reports", log=lambda m: None)
    assert second.new_models == ["fake-verbose"]
    assert second.results[0]["verdict"] == "ship_with_caveats"
    assert "fake-verbose" in second.summary_md
    state = json.loads((tmp_path / ".upgrade-report" / "seen_models.json").read_text())
    assert "fake-verbose" in state["seen"]

    third = run_scheduled(cfg, out_dir=tmp_path / "reports", log=lambda m: None)
    assert third.new_models == [] and "No new models" in third.summary_md
