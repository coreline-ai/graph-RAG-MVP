from __future__ import annotations

import asyncio
from collections.abc import Iterable
from datetime import date
from typing import Any

import psycopg
from psycopg_pool import ConnectionPool
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.config import Settings
from app.schemas import ChunkRecord, DocumentMetadata, QueryFilters, RetrievedChunk

_ALLOWED_DISTINCT_FIELDS = frozenset({"channel", "user_name"})


class PostgresVectorStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pool: ConnectionPool | None = None

    def _ensure_pool(self) -> ConnectionPool:
        if self._pool is None:
            self._pool = ConnectionPool(
                self.settings.postgres_dsn,
                min_size=2,
                max_size=10,
                kwargs={"row_factory": dict_row, "autocommit": True},
                configure=self._configure_connection,
            )
        return self._pool

    @staticmethod
    def _configure_connection(connection: psycopg.Connection) -> None:
        register_vector(connection)

    def _bootstrap_sync(self) -> None:
        table = self.settings.pgvector_table
        with psycopg.connect(
            self.settings.postgres_dsn,
            row_factory=dict_row,
            autocommit=True,
        ) as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            register_vector(connection)
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    source TEXT NOT NULL,
                    access_scopes TEXT[] NOT NULL,
                    total_messages INTEGER NOT NULL DEFAULT 0,
                    total_chunks INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
                    channel TEXT NOT NULL,
                    user_name TEXT NOT NULL,
                    message_date DATE NOT NULL,
                    message_time TIME NOT NULL,
                    access_scopes TEXT[] NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding vector({self.settings.embedding_dimensions}) NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            connection.execute(
                f"CREATE INDEX IF NOT EXISTS {table}_channel_idx ON {table} (channel);"
            )
            connection.execute(
                f"CREATE INDEX IF NOT EXISTS {table}_message_date_idx ON {table} (message_date);"
            )
            connection.execute(
                f"CREATE INDEX IF NOT EXISTS {table}_access_scopes_idx ON {table} USING GIN (access_scopes);"
            )
            connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {table}_embedding_idx
                ON {table}
                USING hnsw (embedding vector_cosine_ops);
                """
            )

    async def bootstrap(self) -> None:
        await asyncio.to_thread(self._bootstrap_sync)
        await asyncio.to_thread(self._ensure_pool)

    def _healthcheck_sync(self) -> str:
        try:
            pool = self._ensure_pool()
            with pool.connection() as connection:
                connection.execute("SELECT 1;").fetchone()
        except Exception as exc:  # pragma: no cover - depends on runtime infra
            return f"error:{exc}"
        return "ok"

    async def healthcheck(self) -> str:
        return await asyncio.to_thread(self._healthcheck_sync)

    def _upsert_document_sync(self, document: DocumentMetadata) -> None:
        pool = self._ensure_pool()
        with pool.connection() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id,
                    filename,
                    source,
                    access_scopes,
                    total_messages,
                    total_chunks,
                    created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (document_id) DO UPDATE SET
                    filename = EXCLUDED.filename,
                    source = EXCLUDED.source,
                    access_scopes = EXCLUDED.access_scopes,
                    total_messages = EXCLUDED.total_messages,
                    total_chunks = EXCLUDED.total_chunks,
                    created_at = EXCLUDED.created_at;
                """,
                (
                    document.document_id,
                    document.filename,
                    document.source,
                    document.access_scopes,
                    document.total_messages,
                    document.total_chunks,
                    document.created_at,
                ),
            )

    async def upsert_document(self, document: DocumentMetadata) -> None:
        await asyncio.to_thread(self._upsert_document_sync, document)

    def _list_documents_sync(self) -> list[DocumentMetadata]:
        pool = self._ensure_pool()
        with pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    document_id,
                    filename,
                    source,
                    access_scopes,
                    total_messages,
                    total_chunks,
                    created_at
                FROM documents
                ORDER BY created_at DESC;
                """
            ).fetchall()
        return [DocumentMetadata(**row) for row in rows]

    async def list_documents(self) -> list[DocumentMetadata]:
        return await asyncio.to_thread(self._list_documents_sync)

    def _get_document_sync(self, document_id: str) -> DocumentMetadata | None:
        pool = self._ensure_pool()
        with pool.connection() as connection:
            row = connection.execute(
                """
                SELECT
                    document_id,
                    filename,
                    source,
                    access_scopes,
                    total_messages,
                    total_chunks,
                    created_at
                FROM documents
                WHERE document_id = %s;
                """,
                (document_id,),
            ).fetchone()
        return DocumentMetadata(**row) if row else None

    async def get_document(self, document_id: str) -> DocumentMetadata | None:
        return await asyncio.to_thread(self._get_document_sync, document_id)

    def _delete_document_sync(self, document_id: str) -> bool:
        table = self.settings.pgvector_table
        pool = self._ensure_pool()
        with pool.connection() as connection:
            connection.execute(f"DELETE FROM {table} WHERE document_id = %s;", (document_id,))
            result = connection.execute(
                "DELETE FROM documents WHERE document_id = %s;",
                (document_id,),
            )
        return result.rowcount > 0

    async def delete_document(self, document_id: str) -> bool:
        return await asyncio.to_thread(self._delete_document_sync, document_id)

    def _upsert_chunks_sync(
        self,
        chunks: list[ChunkRecord],
        embeddings: list[list[float]],
    ) -> None:
        if not chunks:
            return
        table = self.settings.pgvector_table
        sql = f"""
            INSERT INTO {table} (
                chunk_id,
                document_id,
                channel,
                user_name,
                message_date,
                message_time,
                access_scopes,
                chunk_text,
                embedding,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET
                document_id = EXCLUDED.document_id,
                channel = EXCLUDED.channel,
                user_name = EXCLUDED.user_name,
                message_date = EXCLUDED.message_date,
                message_time = EXCLUDED.message_time,
                access_scopes = EXCLUDED.access_scopes,
                chunk_text = EXCLUDED.chunk_text,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata;
        """
        params_seq = [
            (
                chunk.chunk_id,
                chunk.document_id,
                chunk.channel,
                chunk.user_name,
                chunk.message_date,
                chunk.message_time,
                chunk.access_scopes,
                chunk.chunk_text,
                embedding,
                Jsonb(chunk.metadata),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        pool = self._ensure_pool()
        with pool.connection() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(sql, params_seq)

    async def upsert_chunks(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        await asyncio.to_thread(self._upsert_chunks_sync, chunks, embeddings)

    def _row_to_chunk(self, row: dict[str, Any]) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            channel=row["channel"],
            user_name=row["user_name"],
            message_date=row["message_date"],
            message_time=row["message_time"],
            access_scopes=row.get("access_scopes") or [],
            chunk_text=row["chunk_text"],
            metadata=row.get("metadata") or {},
            vector_score=float(row.get("vector_score") or 0.0),
        )

    def _build_filter_sql(self, filters: QueryFilters) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if filters.date_from:
            clauses.append("message_date >= %s")
            params.append(filters.date_from)
        if filters.date_to:
            clauses.append("message_date <= %s")
            params.append(filters.date_to)
        if filters.channel:
            clauses.append("channel = %s")
            params.append(filters.channel)
        if filters.user_names:
            clauses.append("user_name = ANY(%s)")
            params.append(filters.user_names)
        if filters.access_scopes:
            clauses.append("access_scopes && %s")
            params.append(filters.access_scopes)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return where, params

    def _search_chunks_sync(
        self,
        query_embedding: list[float],
        filters: QueryFilters,
        top_k: int,
    ) -> list[RetrievedChunk]:
        table = self.settings.pgvector_table
        where_sql, params = self._build_filter_sql(filters)
        query_vector = Vector(query_embedding)
        sql = f"""
            SELECT
                chunk_id,
                document_id,
                channel,
                user_name,
                message_date,
                message_time,
                access_scopes,
                chunk_text,
                metadata,
                1 - (embedding <=> %s) AS vector_score
            FROM {table}
            {where_sql}
            ORDER BY embedding <=> %s
            LIMIT %s;
        """
        query_params = [query_vector, *params, query_vector, top_k]
        pool = self._ensure_pool()
        with pool.connection() as connection:
            rows = connection.execute(sql, query_params).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    async def search_chunks(
        self,
        query_embedding: list[float],
        filters: QueryFilters,
        top_k: int,
    ) -> list[RetrievedChunk]:
        return await asyncio.to_thread(self._search_chunks_sync, query_embedding, filters, top_k)

    def _get_chunks_by_ids_sync(self, chunk_ids: Iterable[str]) -> list[RetrievedChunk]:
        ids = list(dict.fromkeys(chunk_ids))
        if not ids:
            return []
        table = self.settings.pgvector_table
        pool = self._ensure_pool()
        with pool.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    chunk_id,
                    document_id,
                    channel,
                    user_name,
                    message_date,
                    message_time,
                    access_scopes,
                    chunk_text,
                    metadata,
                    0.0 AS vector_score
                FROM {table}
                WHERE chunk_id = ANY(%s);
                """,
                (ids,),
            ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    async def get_chunks_by_ids(self, chunk_ids: Iterable[str]) -> list[RetrievedChunk]:
        return await asyncio.to_thread(self._get_chunks_by_ids_sync, chunk_ids)

    def _list_distinct_sync(self, field_name: str, limit: int) -> list[str]:
        if field_name not in _ALLOWED_DISTINCT_FIELDS:
            raise ValueError(f"Invalid field_name: {field_name!r}")
        table = self.settings.pgvector_table
        pool = self._ensure_pool()
        with pool.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT DISTINCT {field_name}
                FROM {table}
                WHERE {field_name} IS NOT NULL
                ORDER BY {field_name}
                LIMIT %s;
                """,
                (limit,),
            ).fetchall()
        return [row[field_name] for row in rows]

    async def list_channels(self, limit: int = 200) -> list[str]:
        return await asyncio.to_thread(self._list_distinct_sync, "channel", limit)

    async def list_users(self, limit: int = 200) -> list[str]:
        return await asyncio.to_thread(self._list_distinct_sync, "user_name", limit)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)
