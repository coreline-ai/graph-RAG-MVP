from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from app.adapters.embedding_cache_store import EmbeddingCacheStore
from app.adapters.embedding_provider import BgeM3EmbeddingProvider
from app.adapters.neo4j_store import Neo4jStore
from app.adapters.postgres_vector_store import PostgresVectorStore
from app.config import Settings
from app.schemas import DocumentMetadata
from app.services.chunking import ChunkingService
from app.services.graph_builder import GraphBuilder
from app.services.issue_chunking import IssueChunkingService
from app.services.workbook_parser import PayloadTooLargeError, WorkbookParser

if TYPE_CHECKING:
    from app.services.retrieval import RetrievalService


class IngestService:
    def __init__(
        self,
        *,
        settings: Settings,
        postgres: PostgresVectorStore,
        neo4j: Neo4jStore,
        embedding_provider: BgeM3EmbeddingProvider,
        embedding_cache: EmbeddingCacheStore | None = None,
        chunking: ChunkingService,
        issue_chunking: IssueChunkingService | None = None,
        workbook_parser: WorkbookParser | None = None,
        graph_builder: GraphBuilder,
        retrieval: "RetrievalService | None" = None,
    ) -> None:
        self.settings = settings
        self.postgres = postgres
        self.neo4j = neo4j
        self.embedding_provider = embedding_provider
        self.embedding_cache = embedding_cache
        self.chunking = chunking
        self.issue_chunking = issue_chunking
        self.workbook_parser = workbook_parser
        self.graph_builder = graph_builder
        self.retrieval = retrieval

    async def ingest_document(
        self,
        *,
        filename: str,
        content: str | None = None,
        file_bytes: bytes | None = None,
        default_access_scopes: list[str] | None = None,
        source: str = "manual",
        document_type: str = "auto",
        replace_filename: bool = False,
        byte_limit: int | None = None,
        row_limit: int | None = None,
    ) -> DocumentMetadata:
        access_scopes = default_access_scopes or self.settings.parsed_default_access_scopes
        resolved_document_type = self._resolve_document_type(filename, document_type)
        if replace_filename:
            await self._replace_existing_documents(filename, resolved_document_type)

        if resolved_document_type == "issue":
            if file_bytes is None:
                raise ValueError("Issue ingestion requires workbook bytes.")
            if byte_limit is not None and len(file_bytes) > byte_limit:
                raise PayloadTooLargeError(
                    f"Workbook is larger than {byte_limit} bytes. Use CLI for large XLSX files."
                )
            return await self._ingest_issue_document(
                filename=filename,
                file_bytes=file_bytes,
                access_scopes=access_scopes,
                source=source,
                row_limit=row_limit,
            )

        chat_content = content
        if chat_content is None and file_bytes is not None:
            try:
                chat_content = file_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("Uploaded file must be UTF-8 encoded text.") from exc
        if chat_content is None:
            raise ValueError("Chat ingestion requires content.")
        return await self._ingest_chat_document(
            filename=filename,
            content=chat_content,
            access_scopes=access_scopes,
            source=source,
        )

    async def _ingest_chat_document(
        self,
        *,
        filename: str,
        content: str,
        access_scopes: list[str],
        source: str,
    ) -> DocumentMetadata:
        document_id = str(uuid4())
        messages = self.chunking.parse_log_content(content)
        chunks = self.chunking.build_chunks(
            messages,
            document_id=document_id,
            default_access_scopes=access_scopes,
        )
        embeddings = await self._embed_texts([chunk.chunk_text for chunk in chunks])
        document = DocumentMetadata(
            document_id=document_id,
            filename=filename,
            source=source,
            document_type="chat",
            access_scopes=access_scopes,
            total_messages=len(messages),
            total_chunks=len(chunks),
            ingest_summary={},
        )
        await self._persist_document(document, chunks, embeddings)
        return document

    async def _ingest_issue_document(
        self,
        *,
        filename: str,
        file_bytes: bytes,
        access_scopes: list[str],
        source: str,
        row_limit: int | None,
    ) -> DocumentMetadata:
        if self.workbook_parser is None or self.issue_chunking is None:
            raise RuntimeError("Issue ingestion dependencies are not configured.")
        document_id = str(uuid4())
        parsed = self.workbook_parser.parse_issue_workbook(file_bytes, row_limit=row_limit)
        chunks, chunk_summary = self.issue_chunking.build_chunks(
            parsed.rows,
            document_id=document_id,
            default_access_scopes=access_scopes,
        )
        embeddings = await self._embed_texts([chunk.chunk_text for chunk in chunks])
        ingest_summary = {
            "total_rows": parsed.total_rows,
            "ingested_rows": len(parsed.rows),
            "skipped_rows": parsed.skipped_rows,
            "overview_chunks": chunk_summary["overview_chunks"],
            "analysis_chunks": chunk_summary["analysis_chunks"],
            "warnings_count": len(parsed.warnings),
            "warnings": parsed.warnings,
        }
        document = DocumentMetadata(
            document_id=document_id,
            filename=filename,
            source=source,
            document_type="issue",
            access_scopes=access_scopes,
            total_messages=parsed.total_rows,
            total_chunks=len(chunks),
            ingest_summary=ingest_summary,
        )
        await self._persist_document(document, chunks, embeddings)
        return document

    async def _persist_document(
        self,
        document: DocumentMetadata,
        chunks,
        embeddings: list[list[float]],
    ) -> None:
        graph_rows = self.graph_builder.build_graph_rows(chunks)
        await self.postgres.upsert_document(document)
        await self.postgres.upsert_chunks(chunks, embeddings)
        try:
            await self.neo4j.upsert_graph(document, graph_rows)
        except Exception:
            await self.postgres.delete_document(document.document_id)
            raise
        self._invalidate_retrieval_cache()

    async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.embedding_cache is None:
            return await self.embedding_provider.embed_texts(texts)

        cached = await self.embedding_cache.get_cached_embeddings(texts, self.settings.embedding_model)
        missing_texts = [text for text in dict.fromkeys(texts) if text not in cached]
        if missing_texts:
            missing_embeddings = await self.embedding_provider.embed_texts(missing_texts)
            await self.embedding_cache.upsert_embeddings(
                missing_texts,
                missing_embeddings,
                self.settings.embedding_model,
            )
            cached.update(zip(missing_texts, missing_embeddings, strict=True))
        return [cached[text] for text in texts]

    async def _replace_existing_documents(self, filename: str, document_type: str) -> None:
        existing = await self.postgres.list_documents_by_filename(filename, document_type)
        for document in existing:
            await self.delete_document(document.document_id)

    @staticmethod
    def _resolve_document_type(filename: str, requested: str) -> str:
        if requested != "auto":
            return requested
        if filename.lower().endswith(".xlsx"):
            return "issue"
        return "chat"

    async def list_documents(self) -> list[DocumentMetadata]:
        return await self.postgres.list_documents()

    async def get_document(self, document_id: str) -> DocumentMetadata | None:
        return await self.postgres.get_document(document_id)

    async def delete_document(self, document_id: str) -> bool:
        document = await self.postgres.get_document(document_id)
        if document is None:
            return False

        await self.neo4j.delete_document(document_id)
        deleted_in_postgres = await self.postgres.delete_document(document_id)
        if not deleted_in_postgres:
            raise RuntimeError(f"Document {document_id} was not deleted from Postgres.")

        self._invalidate_retrieval_cache()
        return True

    def _invalidate_retrieval_cache(self) -> None:
        if self.retrieval is not None:
            self.retrieval.invalidate_metadata_cache()
