from __future__ import annotations

import sys
import types

import pytest

if "psycopg_pool" not in sys.modules:
    psycopg_pool = types.ModuleType("psycopg_pool")
    psycopg_pool.ConnectionPool = object
    sys.modules["psycopg_pool"] = psycopg_pool

from app.config import Settings
from app.services.query_analyzer import QueryAnalyzer
from app.services.retrieval import RetrievalService


class FakePostgres:
    def __init__(self) -> None:
        self.channel_calls = 0
        self.user_calls = 0

    async def list_channels(self, limit: int = 200) -> list[str]:
        self.channel_calls += 1
        return [f"general-{self.channel_calls}"]

    async def list_users(self, limit: int = 200) -> list[str]:
        self.user_calls += 1
        return [f"민수-{self.user_calls}"]


class FakeNeo4j:
    pass


class FakeEmbeddingProvider:
    pass


class FakeCodexProxy:
    pass


@pytest.mark.asyncio
async def test_metadata_cache_is_reused_until_invalidated() -> None:
    postgres = FakePostgres()
    service = RetrievalService(
        settings=Settings(),
        postgres=postgres,
        neo4j=FakeNeo4j(),
        embedding_provider=FakeEmbeddingProvider(),
        codex_proxy=FakeCodexProxy(),
        query_analyzer=QueryAnalyzer(),
    )

    first_channels, first_users = await service._get_metadata_lists()
    second_channels, second_users = await service._get_metadata_lists()

    assert first_channels == second_channels
    assert first_users == second_users
    assert postgres.channel_calls == 1
    assert postgres.user_calls == 1

    service.invalidate_metadata_cache()
    third_channels, third_users = await service._get_metadata_lists()

    assert third_channels != first_channels
    assert third_users != first_users
    assert postgres.channel_calls == 2
    assert postgres.user_calls == 2
