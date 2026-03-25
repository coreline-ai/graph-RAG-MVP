from __future__ import annotations

import sys
import types
from datetime import date, datetime, time, timezone

import pytest

if "psycopg_pool" not in sys.modules:
    psycopg_pool = types.ModuleType("psycopg_pool")
    psycopg_pool.ConnectionPool = object
    sys.modules["psycopg_pool"] = psycopg_pool

from app.config import Settings
from app.schemas import ChunkRecord, DocumentMetadata, ParsedMessage
from app.services.ingest import IngestService


class FakePostgres:
    def __init__(self) -> None:
        self.documents: dict[str, DocumentMetadata] = {}
        self.upserted_chunks: list[ChunkRecord] = []

    async def upsert_document(self, document: DocumentMetadata) -> None:
        self.documents[document.document_id] = document

    async def upsert_chunks(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        self.upserted_chunks = chunks

    async def get_document(self, document_id: str) -> DocumentMetadata | None:
        return self.documents.get(document_id)

    async def delete_document(self, document_id: str) -> bool:
        return self.documents.pop(document_id, None) is not None


class FakeNeo4j:
    def __init__(self) -> None:
        self.deleted_ids: list[str] = []
        self.raise_on_delete: Exception | None = None

    async def upsert_graph(self, document: DocumentMetadata, graph_rows: list[dict[str, object]]) -> None:
        return None

    async def delete_document(self, document_id: str) -> bool:
        self.deleted_ids.append(document_id)
        if self.raise_on_delete is not None:
            raise self.raise_on_delete
        return True


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
                original=content,
                sentences=[content],
            )
        ]

    def build_chunks(
        self,
        messages: list[ParsedMessage],
        *,
        document_id: str | None = None,
        default_access_scopes: list[str] | None = None,
    ) -> list[ChunkRecord]:
        return [
            ChunkRecord(
                chunk_id=f"{document_id}_chunk_0000",
                document_id=document_id or "doc-1",
                channel="general",
                user_name="민수",
                message_date=date(2024, 1, 15),
                message_time=time(9, 0, 0),
                access_scopes=default_access_scopes or ["public"],
                chunk_text="general 민수: 작업 시작합니다.",
                token_count=4,
                seq=0,
                metadata={},
                original_lines=["[2024-01-15, 09:00:00, general, 작업 시작합니다., 민수]"],
            )
        ]


class FakeGraphBuilder:
    def build_graph_rows(self, chunks: list[ChunkRecord]) -> list[dict[str, object]]:
        return [{"chunk_id": chunk.chunk_id} for chunk in chunks]


class CacheAwareRetrieval:
    def __init__(self) -> None:
        self.invalidations = 0

    def invalidate_metadata_cache(self) -> None:
        self.invalidations += 1


@pytest.mark.asyncio
async def test_ingest_document_invalidates_retrieval_cache_on_success() -> None:
    retrieval = CacheAwareRetrieval()
    service = IngestService(
        settings=Settings(),
        postgres=FakePostgres(),
        neo4j=FakeNeo4j(),
        embedding_provider=FakeEmbeddingProvider(),
        chunking=FakeChunkingService(),
        graph_builder=FakeGraphBuilder(),
        retrieval=retrieval,
    )

    document = await service.ingest_document(
        filename="chat.txt",
        content="작업 시작합니다.",
    )

    assert document.filename == "chat.txt"
    assert retrieval.invalidations == 1


@pytest.mark.asyncio
async def test_delete_document_surfaces_neo4j_errors() -> None:
    postgres = FakePostgres()
    document = DocumentMetadata(
        document_id="doc-1",
        filename="chat.txt",
        source="manual",
        access_scopes=["public"],
        total_messages=1,
        total_chunks=1,
        created_at=datetime.now(timezone.utc),
    )
    postgres.documents[document.document_id] = document

    neo4j = FakeNeo4j()
    neo4j.raise_on_delete = RuntimeError("neo4j delete failed")
    retrieval = CacheAwareRetrieval()
    service = IngestService(
        settings=Settings(),
        postgres=postgres,
        neo4j=neo4j,
        embedding_provider=FakeEmbeddingProvider(),
        chunking=FakeChunkingService(),
        graph_builder=FakeGraphBuilder(),
        retrieval=retrieval,
    )

    with pytest.raises(RuntimeError, match="neo4j delete failed"):
        await service.delete_document(document.document_id)

    assert document.document_id in postgres.documents
    assert retrieval.invalidations == 0


@pytest.mark.asyncio
async def test_delete_document_invalidates_cache_after_success() -> None:
    postgres = FakePostgres()
    document = DocumentMetadata(
        document_id="doc-1",
        filename="chat.txt",
        source="manual",
        access_scopes=["public"],
        total_messages=1,
        total_chunks=1,
        created_at=datetime.now(timezone.utc),
    )
    postgres.documents[document.document_id] = document
    retrieval = CacheAwareRetrieval()
    service = IngestService(
        settings=Settings(),
        postgres=postgres,
        neo4j=FakeNeo4j(),
        embedding_provider=FakeEmbeddingProvider(),
        chunking=FakeChunkingService(),
        graph_builder=FakeGraphBuilder(),
        retrieval=retrieval,
    )

    deleted = await service.delete_document(document.document_id)

    assert deleted is True
    assert document.document_id not in postgres.documents
    assert retrieval.invalidations == 1
