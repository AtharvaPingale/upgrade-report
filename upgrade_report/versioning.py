"""Content hashes for cache keys: target code version and prompt version.

Hashes are built from git blob SHAs so that the same files produce the same
digest whether they are read from the working tree or from a git ref. That is
what lets a PR run look up the baseline that main's CI run cached
(`--baseline-ref origin/main`).
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from .errors import ConfigError

_SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", "node_modules", ".upgrade-report", ".pytest_cache"}
_SKIP_SUFFIXES = {".pyc", ".pyo"}
_PROMPT_SUFFIXES = ["", ".txt", ".md", ".j2", ".jinja", ".jinja2", ".prompt", ".yaml", ".yml", ".json"]


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _git(base_dir: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(base_dir), *args], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.decode()


def repo_root(base_dir: Path) -> Path | None:
    out = _git(base_dir, "rev-parse", "--show-toplevel")
    return Path(out.strip()) if out else None


def _skip(rel: str) -> bool:
    parts = rel.split("/")
    return any(p in _SKIP_DIRS for p in parts[:-1]) or Path(rel).suffix in _SKIP_SUFFIXES


def _digest(entries: dict[str, str]) -> str:
    h = hashlib.sha256()
    for rel in sorted(entries):
        h.update(f"{rel}\0{entries[rel]}\n".encode())
    return h.hexdigest()[:16]


def _worktree_entries(paths: list[Path], root: Path | None) -> dict[str, str]:
    """Blob SHAs of files under `paths`. In a git repo only tracked files count."""
    entries: dict[str, str] = {}
    if root is not None:
        rels = [str(p.resolve().relative_to(root)) for p in paths if _within(p, root)]
        outside = [p for p in paths if not _within(p, root)]
        if rels:
            listed = _git(root, "ls-files", "-z", "--", *rels) or ""
            for rel in filter(None, listed.split("\0")):
                f = root / rel
                if f.is_file() and not _skip(rel):
                    entries[rel] = git_blob_sha(f.read_bytes())
        paths = outside
    for p in paths:
        p = p.resolve()
        files = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file())
        for f in files:
            rel = f.relative_to(p.parent).as_posix()
            if not _skip(rel):
                entries[rel] = git_blob_sha(f.read_bytes())
    return entries


def _ref_entries(paths: list[Path], root: Path, ref: str) -> dict[str, str]:
    rels = [str(p.resolve().relative_to(root)) for p in paths if _within(p, root)]
    if len(rels) != len(paths):
        raise ConfigError("--baseline-ref needs every target path to be inside the git repository")
    listed = _git(root, "ls-tree", "-r", "-z", ref, "--", *rels)
    if listed is None:
        raise ConfigError(f"git ref {ref!r} not found (fetch it first, e.g. `git fetch origin main`)")
    entries = {}
    for line in filter(None, listed.split("\0")):
        meta, _, rel = line.partition("\t")
        _mode, kind, sha = meta.split()
        if kind == "blob" and not _skip(rel):
            entries[rel] = sha
    return entries


def _within(p: Path, root: Path) -> bool:
    try:
        p.resolve().relative_to(root)
        return True
    except ValueError:
        return False


def code_version(paths: list[Path], base_dir: Path, ref: str | None = None) -> str:
    root = repo_root(base_dir)
    if ref is None:
        return _digest(_worktree_entries(paths, root))
    if root is None:
        raise ConfigError("--baseline-ref needs the project to be a git repository")
    return _digest(_ref_entries(paths, root, ref))


def _prompt_file(prompt_ref: str, base_dir: Path) -> Path | None:
    stem = prompt_ref.split("@", 1)[0]
    if not stem:
        return None
    for suffix in _PROMPT_SUFFIXES:
        candidate = base_dir / f"{stem}{suffix}"
        try:
            if candidate.exists():
                return candidate
        except OSError:
            return None
    return None


def prompt_hash(prompt_ref: str, base_dir: Path, ref: str | None = None) -> str:
    """Hash of the prompt reference plus, if it names a local file or folder, its content."""
    local = _prompt_file(prompt_ref, base_dir)
    content = ""
    if local is not None:
        content = code_version([local], base_dir, ref)
    return hashlib.sha256(f"{prompt_ref}\0{content}".encode()).hexdigest()[:16]
