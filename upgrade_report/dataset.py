"""Load eval examples from a local JSON/JSONL file or a LangSmith dataset."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import DatasetError


@dataclass
class Example:
    id: str
    inputs: dict[str, Any]
    outputs: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def tags(self, key: str) -> list[str]:
        value = self.metadata.get(key)
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [str(v) for v in value]
        return [str(value)]


@dataclass
class Dataset:
    name: str
    source: str  # "file" or "langsmith"
    examples: list[Example]
    version: str  # content hash of the examples; part of every cache key
    as_of: str | None = None
    langsmith_id: str | None = None

    def by_id(self) -> dict[str, Example]:
        return {e.id: e for e in self.examples}


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def content_version(examples: list[Example]) -> str:
    h = hashlib.sha256()
    for e in sorted(examples, key=lambda e: e.id):
        h.update(_canonical([e.id, e.inputs, e.outputs, e.metadata]).encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


def is_local_dataset(ref: str, base_dir: Path) -> bool:
    p = Path(ref)
    if not p.is_absolute():
        p = base_dir / p
    return p.suffix in {".json", ".jsonl"} or p.exists()


def _load_file(path: Path) -> list[Example]:
    if not path.exists():
        raise DatasetError(f"dataset file not found: {path}")
    text = path.read_text()
    try:
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            rows = json.loads(text)
            if isinstance(rows, dict):
                rows = rows.get("examples", [])
    except json.JSONDecodeError as exc:
        raise DatasetError(f"cannot parse {path}: {exc}") from exc
    examples = []
    seen: set[str] = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("inputs"), dict):
            raise DatasetError(f"{path}: row {i} must be an object with an 'inputs' object")
        ex_id = str(row.get("id") or hashlib.sha256(_canonical(row["inputs"]).encode()).hexdigest()[:12])
        if ex_id in seen:
            raise DatasetError(f"{path}: duplicate example id {ex_id!r}")
        seen.add(ex_id)
        examples.append(
            Example(id=ex_id, inputs=row["inputs"], outputs=row.get("outputs"), metadata=row.get("metadata") or {})
        )
    if not examples:
        raise DatasetError(f"{path}: dataset is empty")
    return examples


def _looks_like_uuid(ref: str) -> bool:
    import uuid

    try:
        uuid.UUID(ref)
        return True
    except ValueError:
        return False


def _load_langsmith(ref: str, as_of: str | None, splits: list[str] | None) -> Dataset:
    try:
        from langsmith import Client
    except ImportError as exc:  # pragma: no cover - langsmith is a hard dependency
        raise DatasetError("langsmith is not installed") from exc
    if not (os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")):
        raise DatasetError(
            f"dataset {ref!r} is not a local file and LANGSMITH_API_KEY is not set; "
            "point `dataset` at a .json/.jsonl file or export a LangSmith key"
        )
    client = Client()
    try:
        ds = client.read_dataset(dataset_id=ref) if _looks_like_uuid(ref) else client.read_dataset(dataset_name=ref)
        rows = list(client.list_examples(dataset_id=ds.id, as_of=as_of, splits=splits))
    except Exception as exc:  # the SDK raises several unrelated types
        raise DatasetError(f"cannot load LangSmith dataset {ref!r}: {exc}") from exc
    examples = [
        Example(id=str(r.id), inputs=dict(r.inputs or {}), outputs=dict(r.outputs) if r.outputs else None,
                metadata=dict(r.metadata or {}))
        for r in rows
    ]
    if not examples:
        raise DatasetError(f"LangSmith dataset {ref!r} has no examples (as_of={as_of}, splits={splits})")
    resolved_as_of = as_of or (ds.modified_at.isoformat() if getattr(ds, "modified_at", None) else None)
    return Dataset(name=ds.name, source="langsmith", examples=examples, version=content_version(examples),
                   as_of=resolved_as_of, langsmith_id=str(ds.id))


def load_dataset(ref: str, base_dir: Path, as_of: str | None = None, splits: list[str] | None = None) -> Dataset:
    if is_local_dataset(ref, base_dir):
        path = Path(ref) if Path(ref).is_absolute() else base_dir / ref
        examples = _load_file(path)
        return Dataset(name=ref, source="file", examples=examples, version=content_version(examples))
    return _load_langsmith(ref, as_of, splits)
