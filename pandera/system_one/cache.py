"""Answer caching.

Dataframe workloads re-run constantly -- a new day appended to last week's, a
notebook cell run six times, a backfill overlapping a prior run. Without a
cache every re-run pays full freight; with one, incremental parsing is the
normal case.

The key covers the provider, the resolved model version, the state, *and* the
questions, so rewording a column's ``description`` correctly invalidates its
cached answers. That is the payoff for putting the question in the schema
rather than in a prompt string somewhere else.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from pandera.system_one.primitives import Decision, Question


def cache_key(
    provider_id: str,
    model_version: str,
    state: Any,
    questions: Mapping[str, Question],
) -> str:
    """Stable key for one (provider, model, state, question set)."""
    payload = json.dumps(
        [
            provider_id,
            model_version,
            state,
            {
                name: _question_fingerprint(q)
                for name, q in sorted(questions.items())
            },
        ],
        sort_keys=True,
        default=repr,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _question_fingerprint(question: Question) -> Any:
    """Everything about a question that could change its answer."""
    return {
        "kind": question.kind,
        "instructions": question.instructions,
        "criteria": getattr(question, "criteria", None),
    }


@runtime_checkable
class AnswerCache(Protocol):
    """Somewhere to keep answers between runs."""

    def get(self, key: str) -> Mapping[str, Decision] | None: ...

    def set(self, key: str, value: Mapping[str, Decision]) -> None: ...


class MemoryCache:
    """Keeps answers for the lifetime of the process."""

    def __init__(self) -> None:
        self._store: dict[str, Mapping[str, Decision]] = {}

    def get(self, key: str) -> Mapping[str, Decision] | None:
        return self._store.get(key)

    def set(self, key: str, value: Mapping[str, Decision]) -> None:
        self._store[key] = value

    def __len__(self) -> int:
        return len(self._store)


class SQLiteCache:
    """Keeps answers in a SQLite file, so they survive the process."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS answers "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def get(self, key: str) -> Mapping[str, Decision] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM answers WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return {
            name: Decision(
                value=payload["value"],
                confidence=payload.get("confidence"),
                probabilities=payload.get("probabilities", {}),
            )
            for name, payload in json.loads(row[0]).items()
        }

    def set(self, key: str, value: Mapping[str, Decision]) -> None:
        payload = json.dumps(
            {
                name: {
                    "value": decision.value,
                    "confidence": decision.confidence,
                    "probabilities": decision.probabilities,
                }
                for name, decision in value.items()
            },
            default=repr,
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO answers (key, value) VALUES (?, ?)",
                (key, payload),
            )


def build_cache(spec: Any) -> AnswerCache | None:
    """Turn a cache specification into a cache.

    ``"memory"``, a ``sqlite://`` URL, or any object with ``get``/``set``.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        if spec == "memory":
            return MemoryCache()
        if spec.startswith("sqlite:///"):
            return SQLiteCache(spec[len("sqlite:///") :])
        raise ValueError(
            f"unknown cache {spec!r}. Use 'memory', 'sqlite:///path.db', or "
            "an object with get/set methods."
        )
    return spec
