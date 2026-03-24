from __future__ import annotations

from uuid import uuid4

from app.adapters.embedding_provider import BgeM3EmbeddingProvider
from app.adapters.neo4j_store import Neo4jStore
from app.adapters.postgres_vector_store import PostgresVectorStore
from app.config import Settings
from app.schemas import DocumentMetadata
from app.services.chunking import ChunkingService
from app.services.graph_builder import GraphBuilder


class IngestService:
    def __init__(
        self,
        *,
        settings: Settings,
        postgres: PostgresVectorStore,
        neo4j: Neo4jStore,
        embedding_provider: BgeM3EmbeddingProvider,
        chunking: ChunkingService,
        graph_builder: GraphBuilder,
    ) -> None:
        self.settings = settings
        self.postgres = postgres
        self.neo4j = neo4j
        self.embedding_provider = embedding_provider
        self.chunking = chunking
        self.graph_builder = graph_builder

    async def ingest_document(
        self,
        *,
        filename: str,
        content: str,
        default_access_scopes: list[str] | None = None,
        source: str = "manual",
    ) -> DocumentMetadata:
        document_id = str(uuid4())
        messages = self.chunking.parse_log_content(content)
        chunks = self.chunking.build_chunks(
            messages,
            document_id=document_id,
            default_access_scopes=default_access_scopes or self.settings.parsed_default_access_scopes,
        )
        embeddings = await self.embedding_provider.embed_texts([chunk.chunk_text for chunk in chunks])
        document = DocumentMetadata(
            document_id=document_id,
            filename=filename,
            source=source,
            access_scopes=default_access_scopes or self.settings.parsed_default_access_scopes,
            total_messages=len(messages),
            total_chunks=len(chunks),
        )
        graph_rows = self.graph_builder.build_graph_rows(chunks)

        await self.postgres.upsert_document(document)
        await self.postgres.upsert_chunks(chunks, embeddings)
        try:
            await self.neo4j.upsert_graph(document, graph_rows)
        except Exception:
            await self.postgres.delete_document(document.document_id)
            raise
        return document

    async def list_documents(self) -> list[DocumentMetadata]:
        return await self.postgres.list_documents()

    async def get_document(self, document_id: str) -> DocumentMetadata | None:
        return await self.postgres.get_document(document_id)

    async def delete_document(self, document_id: str) -> bool:
        try:
            await self.neo4j.delete_document(document_id)
        except Exception:
            pass
        return await self.postgres.delete_document(document_id)
