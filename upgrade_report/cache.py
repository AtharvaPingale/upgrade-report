"""SQLite cache for arm runs and scores, keyed by content hash.

Arm runs are keyed by (dataset_version, model, prompt_hash, target_version,
repeat_index). Records are stored per example, so an interrupted run resumes
where it stopped; failed calls are not cached and are retried next time.
Scores are keyed separately by (run_key, scorer_key), where the scorer key
covers the pinned judge and the evaluator code, so changing an evaluator
re-scores cached outputs without re-running the target.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .records import RunRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS arm_runs (
    run_key TEXT PRIMARY KEY,
    key_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS run_records (
    run_key TEXT NOT NULL,
    example_id TEXT NOT NULL,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_key, example_id)
);
CREATE TABLE IF NOT EXISTS scores (
    run_key TEXT NOT NULL,
    scorer_key TEXT NOT NULL,
    example_id TEXT NOT NULL,
    scores_json TEXT NOT NULL,
    PRIMARY KEY (run_key, scorer_key, example_id)
);
CREATE TABLE IF NOT EXISTS kv (
    namespace TEXT NOT NULL,
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    PRIMARY KEY (namespace, key)
);
"""


def stable_hash(obj: Any, length: int = 24) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:length]


def run_key(dataset_version: str, model: str, prompt_hash: str, target_version: str,
            repeat_index: int) -> tuple[str, dict]:
    parts = {
        "dataset_version": dataset_version,
        "model": model,
        "prompt_hash": prompt_hash,
        "target_version": target_version,
        "repeat_index": repeat_index,
    }
    return stable_hash(parts), parts


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Cache:
    """Thread-safe. `path=None` gives a throwaway in-memory cache."""

    def __init__(self, path: Path | None, read: bool = True):
        self.path = path
        self.read = read
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path) if path else ":memory:", check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            if path is not None:
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.commit()
            self._conn.close()

    # -- arm runs ---------------------------------------------------------

    def load_records(self, key: str) -> dict[str, RunRecord]:
        if not self.read:
            return {}
        with self._lock:
            rows = self._conn.execute(
                "SELECT example_id, record_json FROM run_records WHERE run_key = ?", (key,)
            ).fetchall()
        return {eid: RunRecord.from_json(json.loads(blob)) for eid, blob in rows}

    def save_record(self, key: str, key_parts: dict, record: RunRecord) -> None:
        if not record.ok:
            return
        now = _now()
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO arm_runs (run_key, key_json, created_at) VALUES (?, ?, ?)",
                (key, json.dumps(key_parts, sort_keys=True), now),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO run_records (run_key, example_id, record_json, created_at) VALUES (?, ?, ?, ?)",
                (key, record.example_id, json.dumps(record.to_json(), default=str), now),
            )
            self._conn.commit()

    # -- scores -----------------------------------------------------------

    def load_scores(self, key: str, scorer_key: str) -> dict[str, dict]:
        if not self.read:
            return {}
        with self._lock:
            rows = self._conn.execute(
                "SELECT example_id, scores_json FROM scores WHERE run_key = ? AND scorer_key = ?", (key, scorer_key)
            ).fetchall()
        return {eid: json.loads(blob) for eid, blob in rows}

    def save_scores(self, key: str, scorer_key: str, example_id: str, scores: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO scores (run_key, scorer_key, example_id, scores_json) VALUES (?, ?, ?, ?)",
                (key, scorer_key, example_id, json.dumps(scores, default=str)),
            )
            self._conn.commit()

    # -- generic ----------------------------------------------------------

    def get(self, namespace: str, key: str) -> Any | None:
        if not self.read:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT value_json FROM kv WHERE namespace = ? AND key = ?", (namespace, key)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, namespace: str, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (namespace, key, value_json) VALUES (?, ?, ?)",
                (namespace, key, json.dumps(value, default=str)),
            )
            self._conn.commit()

    def average_usage(self, model: str) -> tuple[float, float] | None:
        """Mean tokens per call seen for `model` across all cached records (for --dry-run)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT r.record_json FROM run_records r JOIN arm_runs a ON a.run_key = r.run_key "
                "WHERE json_extract(a.key_json, '$.model') = ?",
                (model,),
            ).fetchall()
        ins, outs = [], []
        for (blob,) in rows:
            usage = json.loads(blob).get("usage") or {}
            if usage:
                ins.append(sum(u["input_tokens"] for u in usage.values()))
                outs.append(sum(u["output_tokens"] for u in usage.values()))
        if not ins:
            return None
        return sum(ins) / len(ins), sum(outs) / len(outs)
