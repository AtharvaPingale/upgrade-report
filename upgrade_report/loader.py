"""Import `module:attribute` references relative to a project directory."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

from .errors import ConfigError


def ensure_on_path(base_dir: Path) -> None:
    path = str(base_dir.resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def split_ref(ref: str) -> tuple[str, str]:
    if ":" not in ref:
        raise ConfigError(f"expected 'module:attribute', got {ref!r}")
    module, _, attr = ref.partition(":")
    if not module or not attr:
        raise ConfigError(f"expected 'module:attribute', got {ref!r}")
    return module, attr


def load_object(ref: str, base_dir: Path) -> Any:
    module_name, attr = split_ref(ref)
    ensure_on_path(base_dir)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ConfigError(f"cannot import {module_name!r} (from {ref!r}): {exc}") from exc
    obj: Any = module
    for part in attr.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise ConfigError(f"{module_name!r} has no attribute {attr!r}") from exc
    return obj


def module_source_paths(ref: str, base_dir: Path) -> list[Path]:
    """Default code paths behind a target: its top-level package, or its file."""
    module_name, _ = split_ref(ref)
    top = module_name.split(".")[0]
    ensure_on_path(base_dir)
    spec = importlib.util.find_spec(top)
    if spec is None or spec.origin is None:
        raise ConfigError(f"cannot locate module {top!r} for target {ref!r}")
    origin = Path(spec.origin)
    if spec.submodule_search_locations:
        return [Path(p) for p in spec.submodule_search_locations]
    return [origin]
