from __future__ import annotations

import sys
import types
from datetime import date, time

if "psycopg_pool" not in sys.modules:
    psycopg_pool = types.ModuleType("psycopg_pool")
    psycopg_pool.ConnectionPool = object
    sys.modules["psycopg_pool"] = psycopg_pool

from app.adapters.postgres_vector_store import PostgresVectorStore
from app.config import Settings
from app.schemas import QueryFilters


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeTransaction:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeConnection:
    def __init__(self) -> None:
        self.commands: list[tuple[str, object]] = []

    def execute(self, sql, params=None):
        self.commands.append((sql, params))
        return _FakeResult([])

    def transaction(self):
        return _FakeTransaction()


class _FakePoolConnection:
    def __init__(self, connection: _FakeConnection) -> None:
        self._connection = connection

    def __enter__(self):
        return self._connection

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakePool:
    def __init__(self, connection: _FakeConnection) -> None:
        self._connection = connection

    def connection(self):
        return _FakePoolConnection(self._connection)


def test_search_chunks_retries_filtered_queries_with_exact_scan(monkeypatch) -> None:
    store = PostgresVectorStore(Settings())
    fake_connection = _FakeConnection()
    monkeypatch.setattr(store, "_ensure_pool", lambda: _FakePool(fake_connection))

    calls = []
    row = {
        "chunk_id": "issue-1",
        "document_id": "doc-1",
        "document_type": "issue",
        "channel": "이슈데이터_10000건",
        "user_name": "Sujin",
        "message_date": date(2026, 3, 20),
        "message_time": time(9, 0),
        "access_scopes": ["public"],
        "chunk_text": "[이슈] 테스트 이슈",
        "metadata": {"assignee": "Sujin", "chunk_kind": "overview"},
        "vector_score": 0.42,
    }

    def fake_execute_vector_search(connection, sql, params):
        calls.append((sql, params))
        return [] if len(calls) == 1 else [row]

    monkeypatch.setattr(store, "_execute_vector_search", fake_execute_vector_search)

    chunks = store._search_chunks_sync(
        [0.1] * store.settings.embedding_dimensions,
        QueryFilters(document_types=["issue"], assignees=["Sujin"], access_scopes=["public"]),
        5,
    )

    assert [chunk.chunk_id for chunk in chunks] == ["issue-1"]
    assert len(calls) == 2
    assert any("SET LOCAL enable_indexscan = off;" in sql for sql, _ in fake_connection.commands)
