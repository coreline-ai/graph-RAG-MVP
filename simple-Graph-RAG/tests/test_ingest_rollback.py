"""Tests for ingest create-path rollback when Neo4j fails."""
from __future__ import annotations

import sys
import types
from datetime import date, time

import pytest

if "psycopg_pool" not in sys.modules:
    psycopg_pool = types.ModuleType("psycopg_pool")
    psycopg_pool.ConnectionPool = object
    sys.modules["psycopg_pool"] = psycopg_pool

from app.config import Settings
from app.schemas import ChunkRecord, DocumentMetadata, ParsedMessage
from app.services.ingest import IngestService


class TrackingPostgres:
    """Tracks upsert/delete calls for assertion."""

    def __init__(self) -> None:
        self.upserted_documents: list[str] = []
        self.upserted_chunk_count = 0
        self.deleted_document_ids: list[str] = []

    async def upsert_document(self, document: DocumentMetadata) -> None:
        self.upserted_documents.append(document.document_id)

    async def upsert_chunks(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        self.upserted_chunk_count += len(chunks)

    async def delete_document(self, document_id: str) -> bool:
        self.deleted_document_ids.append(document_id)
        return True


class FailingNeo4j:
    """Always raises on upsert_graph."""

    def __init__(self, error: Exception | None = None) -> None:
        self._error = error or RuntimeError("Neo4j connection lost")

    async def upsert_graph(self, document, graph_rows) -> None:
        raise self._error


class FakeEmbeddingProvider:
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]


class FakeChunkingService:
    def parse_log_content(self, content: str) -> list[ParsedMessage]:
        return [
            ParsedMessage(
                line_number=1,
                date=date(2024, 1, 15),
                time=time(9, 0, 0),
                channel="general",
                user_name="민수",
                content=content,
                original=f"[2024-01-15, 09:00:00, general, {content}, 민수]",
                sentences=[content],
            )
        ]

    def build_chunks(self, messages, *, document_id=None, default_access_scopes=None):
        return [
            ChunkRecord(
                chunk_id=f"{document_id}_chunk_0000",
                document_id=document_id or "doc-1",
                channel="general",
                user_name="민수",
                message_date=date(2024, 1, 15),
                message_time=time(9, 0, 0),
                access_scopes=default_access_scopes or ["public"],
                chunk_text="general 민수: test",
                token_count=4,
                seq=0,
            )
        ]


class FakeGraphBuilder:
    def build_graph_rows(self, chunks):
        return [{"chunk_id": c.chunk_id} for c in chunks]


class CacheTracker:
    def __init__(self) -> None:
        self.invalidations = 0

    def invalidate_metadata_cache(self) -> None:
        self.invalidations += 1


@pytest.mark.asyncio
async def test_neo4j_failure_during_ingest_triggers_postgres_rollback() -> None:
    """When Neo4j upsert_graph fails, Postgres document should be deleted (rollback)."""
    postgres = TrackingPostgres()
    cache = CacheTracker()

    service = IngestService(
        settings=Settings(),
        postgres=postgres,
        neo4j=FailingNeo4j(),
        embedding_provider=FakeEmbeddingProvider(),
        chunking=FakeChunkingService(),
        graph_builder=FakeGraphBuilder(),
        retrieval=cache,
    )

    with pytest.raises(RuntimeError, match="Neo4j connection lost"):
        await service.ingest_document(filename="test.txt", content="테스트 내용")

    # Postgres document was upserted then rolled back
    assert len(postgres.upserted_documents) == 1
    assert len(postgres.deleted_document_ids) == 1
    assert postgres.upserted_documents[0] == postgres.deleted_document_ids[0]
    # Cache should NOT be invalidated on failure
    assert cache.invalidations == 0


@pytest.mark.asyncio
async def test_neo4j_failure_preserves_original_exception() -> None:
    """The original Neo4j exception should propagate, not be swallowed."""
    service = IngestService(
        settings=Settings(),
        postgres=TrackingPostgres(),
        neo4j=FailingNeo4j(ValueError("custom neo4j error")),
        embedding_provider=FakeEmbeddingProvider(),
        chunking=FakeChunkingService(),
        graph_builder=FakeGraphBuilder(),
    )

    with pytest.raises(ValueError, match="custom neo4j error"):
        await service.ingest_document(filename="test.txt", content="테스트")
