from __future__ import annotations

import asyncio
from collections.abc import Iterable
from datetime import date
from typing import Any

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from app.config import Settings
from app.schemas import ChunkRecord, DocumentMetadata, QueryFilters, RetrievedChunk

_ALLOWED_DISTINCT_FIELDS = frozenset({"channel", "user_name", "document_type"})
_ALLOWED_METADATA_FIELDS = frozenset({"assignee", "status"})


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
                    document_type TEXT NOT NULL DEFAULT 'chat',
                    access_scopes TEXT[] NOT NULL,
                    total_messages INTEGER NOT NULL DEFAULT 0,
                    total_chunks INTEGER NOT NULL DEFAULT 0,
                    ingest_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
                    document_type TEXT NOT NULL DEFAULT 'chat',
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
            connection.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS document_type TEXT NOT NULL DEFAULT 'chat';")
            connection.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS ingest_summary JSONB NOT NULL DEFAULT '{}'::jsonb;")
            connection.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS document_type TEXT NOT NULL DEFAULT 'chat';")
            connection.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb;")
            connection.execute(f"CREATE INDEX IF NOT EXISTS {table}_channel_idx ON {table} (channel);")
            connection.execute(f"CREATE INDEX IF NOT EXISTS {table}_user_name_idx ON {table} (user_name);")
            connection.execute(f"CREATE INDEX IF NOT EXISTS {table}_document_type_idx ON {table} (document_type);")
            connection.execute(f"CREATE INDEX IF NOT EXISTS {table}_message_date_idx ON {table} (message_date);")
            connection.execute(
                f"CREATE INDEX IF NOT EXISTS {table}_access_scopes_idx ON {table} USING GIN (access_scopes);"
            )
            connection.execute(
                f"CREATE INDEX IF NOT EXISTS {table}_metadata_gin_idx ON {table} USING GIN (metadata);"
            )
            connection.execute(
                f"CREATE INDEX IF NOT EXISTS {table}_assignee_idx ON {table} ((metadata->>'assignee'));"
            )
            connection.execute(
                f"CREATE INDEX IF NOT EXISTS {table}_status_idx ON {table} ((metadata->>'status'));"
            )
            connection.execute(
                f"CREATE INDEX IF NOT EXISTS {table}_chunk_kind_idx ON {table} ((metadata->>'chunk_kind'));"
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
        except Exception as exc:  # pragma: no cover
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
                    document_type,
                    access_scopes,
                    total_messages,
                    total_chunks,
                    ingest_summary,
                    created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (document_id) DO UPDATE SET
                    filename = EXCLUDED.filename,
                    source = EXCLUDED.source,
                    document_type = EXCLUDED.document_type,
                    access_scopes = EXCLUDED.access_scopes,
                    total_messages = EXCLUDED.total_messages,
                    total_chunks = EXCLUDED.total_chunks,
                    ingest_summary = EXCLUDED.ingest_summary,
                    created_at = EXCLUDED.created_at;
                """,
                (
                    document.document_id,
                    document.filename,
                    document.source,
                    document.document_type,
                    document.access_scopes,
                    document.total_messages,
                    document.total_chunks,
                    Jsonb(document.ingest_summary),
                    document.created_at,
                ),
            )

    async def upsert_document(self, document: DocumentMetadata) -> None:
        await asyncio.to_thread(self._upsert_document_sync, document)

    def _row_to_document(self, row: dict[str, Any]) -> DocumentMetadata:
        return DocumentMetadata(
            document_id=row["document_id"],
            filename=row["filename"],
            source=row["source"],
            document_type=row.get("document_type") or "chat",
            access_scopes=row.get("access_scopes") or [],
            total_messages=row.get("total_messages") or 0,
            total_chunks=row.get("total_chunks") or 0,
            ingest_summary=row.get("ingest_summary") or {},
            created_at=row["created_at"],
        )

    def _list_documents_sync(self) -> list[DocumentMetadata]:
        pool = self._ensure_pool()
        with pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    document_id,
                    filename,
                    source,
                    document_type,
                    access_scopes,
                    total_messages,
                    total_chunks,
                    ingest_summary,
                    created_at
                FROM documents
                ORDER BY created_at DESC;
                """
            ).fetchall()
        return [self._row_to_document(row) for row in rows]

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
                    document_type,
                    access_scopes,
                    total_messages,
                    total_chunks,
                    ingest_summary,
                    created_at
                FROM documents
                WHERE document_id = %s;
                """,
                (document_id,),
            ).fetchone()
        return self._row_to_document(row) if row else None

    async def get_document(self, document_id: str) -> DocumentMetadata | None:
        return await asyncio.to_thread(self._get_document_sync, document_id)

    def _list_documents_by_filename_sync(
        self,
        filename: str,
        document_type: str,
    ) -> list[DocumentMetadata]:
        pool = self._ensure_pool()
        with pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    document_id,
                    filename,
                    source,
                    document_type,
                    access_scopes,
                    total_messages,
                    total_chunks,
                    ingest_summary,
                    created_at
                FROM documents
                WHERE filename = %s
                  AND document_type = %s
                ORDER BY created_at DESC;
                """,
                (filename, document_type),
            ).fetchall()
        return [self._row_to_document(row) for row in rows]

    async def list_documents_by_filename(
        self,
        filename: str,
        document_type: str,
    ) -> list[DocumentMetadata]:
        return await asyncio.to_thread(self._list_documents_by_filename_sync, filename, document_type)

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
                document_type,
                channel,
                user_name,
                message_date,
                message_time,
                access_scopes,
                chunk_text,
                embedding,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET
                document_id = EXCLUDED.document_id,
                document_type = EXCLUDED.document_type,
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
                chunk.document_type,
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
        metadata = row.get("metadata") or {}
        return RetrievedChunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            document_type=row.get("document_type") or "chat",
            channel=row["channel"],
            user_name=row["user_name"],
            message_date=row["message_date"],
            message_time=row["message_time"],
            access_scopes=row.get("access_scopes") or [],
            chunk_text=row["chunk_text"],
            metadata=metadata,
            vector_score=float(row.get("vector_score") or 0.0),
        )

    def _build_filter_clauses(self, filters: QueryFilters) -> tuple[list[str], list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if filters.date_from:
            clauses.append("message_date >= %s")
            params.append(filters.date_from)
        if filters.date_to:
            clauses.append("message_date <= %s")
            params.append(filters.date_to)
        if filters.all_channels:
            clauses.append("channel = ANY(%s)")
            params.append(filters.all_channels)
        if filters.user_names:
            clauses.append("user_name = ANY(%s)")
            params.append(filters.user_names)
        if filters.assignees:
            clauses.append("COALESCE(metadata->>'assignee', '') = ANY(%s)")
            params.append(filters.assignees)
        if filters.statuses:
            clauses.append("COALESCE(metadata->>'status', '') = ANY(%s)")
            params.append(filters.statuses)
        if filters.all_document_types:
            clauses.append("document_type = ANY(%s)")
            params.append(filters.all_document_types)
        if filters.access_scopes:
            clauses.append("access_scopes && %s")
            params.append(filters.access_scopes)
        return clauses, params

    def _build_filter_sql(self, filters: QueryFilters) -> tuple[str, list[Any]]:
        clauses, params = self._build_filter_clauses(filters)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return where, params

    @staticmethod
    def _should_retry_exact_filtered_search(filters: QueryFilters) -> bool:
        return bool(
            filters.all_channels
            or filters.user_names
            or filters.assignees
            or filters.statuses
            or filters.all_document_types
        )

    @staticmethod
    def _countable_issue_clause() -> str:
        return "(document_type <> 'issue' OR COALESCE(metadata->>'chunk_kind', '') = 'overview')"

    def _execute_vector_search(
        self,
        connection: psycopg.Connection,
        sql: str,
        query_params: list[Any],
    ) -> list[dict[str, Any]]:
        return connection.execute(sql, query_params).fetchall()

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
                document_type,
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
            rows = self._execute_vector_search(connection, sql, query_params)
            if not rows and self._should_retry_exact_filtered_search(filters):
                with connection.transaction():
                    connection.execute("SET LOCAL enable_indexscan = off;")
                    connection.execute("SET LOCAL enable_bitmapscan = off;")
                    connection.execute("SET LOCAL enable_indexonlyscan = off;")
                    rows = self._execute_vector_search(connection, sql, query_params)
        return [self._row_to_chunk(row) for row in rows]

    async def search_chunks(
        self,
        query_embedding: list[float],
        filters: QueryFilters,
        top_k: int,
    ) -> list[RetrievedChunk]:
        return await asyncio.to_thread(self._search_chunks_sync, query_embedding, filters, top_k)

    def _search_document_type_candidates_sync(
        self,
        query_embedding: list[float],
        filters: QueryFilters,
        top_k: int,
        *,
        document_type: str,
    ) -> list[RetrievedChunk]:
        scoped_filters = filters.model_copy(deep=True)
        scoped_filters.document_types = [document_type]
        if document_type == "chat":
            scoped_filters.assignees = []
            scoped_filters.statuses = []
        return self._search_chunks_sync(query_embedding, scoped_filters, top_k)

    async def search_issue_candidates(
        self,
        query_embedding: list[float],
        filters: QueryFilters,
        top_k: int,
    ) -> list[RetrievedChunk]:
        return await asyncio.to_thread(
            self._search_document_type_candidates_sync,
            query_embedding,
            filters,
            top_k,
            document_type="issue",
        )

    async def search_chat_candidates(
        self,
        query_embedding: list[float],
        filters: QueryFilters,
        top_k: int,
    ) -> list[RetrievedChunk]:
        return await asyncio.to_thread(
            self._search_document_type_candidates_sync,
            query_embedding,
            filters,
            top_k,
            document_type="chat",
        )

    def _summarize_filtered_results_sync(
        self,
        filters: QueryFilters,
        *,
        limit: int,
    ) -> dict[str, Any]:
        table = self.settings.pgvector_table
        clauses, params = self._build_filter_clauses(filters)
        clauses.append(self._countable_issue_clause())
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        pool = self._ensure_pool()
        with pool.connection() as connection:
            count_row = connection.execute(
                f"""
                SELECT COUNT(*) AS matched_count
                FROM {table}
                {where_sql};
                """,
                params,
            ).fetchone()
            sample_rows = connection.execute(
                f"""
                SELECT
                    chunk_id,
                    document_id,
                    document_type,
                    channel,
                    user_name,
                    message_date,
                    message_time,
                    access_scopes,
                    chunk_text,
                    metadata,
                    0.0 AS vector_score
                FROM {table}
                {where_sql}
                ORDER BY message_date DESC, message_time DESC, chunk_id DESC
                LIMIT %s;
                """,
                (*params, limit),
            ).fetchall()
        return {
            "matched_count": int(count_row["matched_count"]) if count_row else 0,
            "count_basis": "matching_records",
            "sample_chunks": [self._row_to_chunk(row) for row in sample_rows],
        }

    async def summarize_filtered_results(
        self,
        filters: QueryFilters,
        *,
        limit: int,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self._summarize_filtered_results_sync, filters, limit=limit)

    def _keyword_match_clause(
        self,
        term_groups: list[tuple[str, ...]],
        *,
        params: list[Any],
    ) -> str:
        clauses: list[str] = []
        for group in term_groups:
            group_clauses: list[str] = []
            for term in group:
                pattern = f"%{term.lower()}%"
                group_clauses.append("LOWER(chunk_text) LIKE %s")
                params.append(pattern)
                group_clauses.append("LOWER(COALESCE(metadata->>'issue_title', '')) LIKE %s")
                params.append(pattern)
            clauses.append("(" + " OR ".join(group_clauses) + ")")
        return "(" + " OR ".join(clauses) + ")" if clauses else "FALSE"

    def _summarize_special_keyword_results_sync(
        self,
        filters: QueryFilters,
        *,
        exact_groups: list[tuple[str, ...]],
        alias_groups: list[tuple[str, ...]],
        limit: int,
    ) -> dict[str, Any]:
        table = self.settings.pgvector_table

        def summarize_for_groups(term_groups: list[tuple[str, ...]], *, basis: str) -> dict[str, Any]:
            clauses, params = self._build_filter_clauses(filters)
            clauses.append(self._countable_issue_clause())
            clauses.append(self._keyword_match_clause(term_groups, params=params))
            where_sql = f"WHERE {' AND '.join(clauses)}"
            pool = self._ensure_pool()
            with pool.connection() as connection:
                count_row = connection.execute(
                    f"""
                    SELECT COUNT(*) AS matched_count
                    FROM {table}
                    {where_sql};
                    """,
                    params,
                ).fetchone()
                sample_rows = connection.execute(
                    f"""
                    SELECT
                        chunk_id,
                        document_id,
                        document_type,
                        channel,
                        user_name,
                        message_date,
                        message_time,
                        access_scopes,
                        chunk_text,
                        metadata,
                        0.0 AS vector_score
                    FROM {table}
                    {where_sql}
                    ORDER BY message_date DESC, message_time DESC, chunk_id DESC
                    LIMIT %s;
                    """,
                    (*params, limit),
                ).fetchall()
            return {
                "matched_count": int(count_row["matched_count"]) if count_row else 0,
                "count_basis": basis,
                "sample_chunks": [self._row_to_chunk(row) for row in sample_rows],
            }

        if exact_groups:
            exact_summary = summarize_for_groups(exact_groups, basis="special_keyword_exact")
            if exact_summary["matched_count"] > 0:
                return exact_summary
        if alias_groups:
            return summarize_for_groups(alias_groups, basis="special_keyword_alias")
        return {"matched_count": 0, "count_basis": "special_keyword_none", "sample_chunks": []}

    async def summarize_special_keyword_results(
        self,
        filters: QueryFilters,
        *,
        exact_groups: list[tuple[str, ...]],
        alias_groups: list[tuple[str, ...]],
        limit: int,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._summarize_special_keyword_results_sync,
            filters,
            exact_groups=exact_groups,
            alias_groups=alias_groups,
            limit=limit,
        )

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
                    document_type,
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

    def _list_distinct_sync(
        self,
        field_name: str,
        *,
        limit: int,
        document_type: str | None = None,
    ) -> list[str]:
        if field_name not in _ALLOWED_DISTINCT_FIELDS:
            raise ValueError(f"Invalid field_name: {field_name!r}")
        table = self.settings.pgvector_table
        clauses = [f"{field_name} IS NOT NULL"]
        params: list[Any] = []
        if document_type and document_type != "all":
            clauses.append("document_type = %s")
            params.append(document_type)
        where_sql = " AND ".join(clauses)
        pool = self._ensure_pool()
        with pool.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT DISTINCT {field_name}
                FROM {table}
                WHERE {where_sql}
                ORDER BY {field_name}
                LIMIT %s;
                """,
                (*params, limit),
            ).fetchall()
        return [row[field_name] for row in rows if row[field_name]]

    def _list_metadata_values_sync(
        self,
        field_name: str,
        *,
        limit: int,
        document_type: str | None = None,
    ) -> list[str]:
        if field_name not in _ALLOWED_METADATA_FIELDS:
            raise ValueError(f"Invalid metadata field: {field_name!r}")
        table = self.settings.pgvector_table
        clauses = [f"COALESCE(metadata->>'{field_name}', '') <> ''"]
        params: list[Any] = []
        if document_type and document_type != "all":
            clauses.append("document_type = %s")
            params.append(document_type)
        where_sql = " AND ".join(clauses)
        pool = self._ensure_pool()
        with pool.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT DISTINCT metadata->>'{field_name}' AS value
                FROM {table}
                WHERE {where_sql}
                ORDER BY value
                LIMIT %s;
                """,
                (*params, limit),
            ).fetchall()
        return [row["value"] for row in rows if row["value"]]

    async def list_channels(self, limit: int = 200, document_type: str | None = None) -> list[str]:
        return await asyncio.to_thread(
            self._list_distinct_sync,
            "channel",
            limit=limit,
            document_type=document_type,
        )

    async def list_users(self, limit: int = 200, document_type: str | None = None) -> list[str]:
        return await asyncio.to_thread(
            self._list_distinct_sync,
            "user_name",
            limit=limit,
            document_type=document_type,
        )

    async def list_document_types(self, limit: int = 20) -> list[str]:
        return await asyncio.to_thread(self._list_distinct_sync, "document_type", limit=limit)

    async def list_assignees(self, limit: int = 200, document_type: str = "issue") -> list[str]:
        return await asyncio.to_thread(
            self._list_metadata_values_sync,
            "assignee",
            limit=limit,
            document_type=document_type,
        )

    async def list_statuses(self, limit: int = 50, document_type: str = "issue") -> list[str]:
        return await asyncio.to_thread(
            self._list_metadata_values_sync,
            "status",
            limit=limit,
            document_type=document_type,
        )

    def _get_latest_event_date_sync(self, document_type: str | None = None) -> date | None:
        table = self.settings.pgvector_table
        clauses = []
        params: list[Any] = []
        if document_type and document_type != "all":
            clauses.append("document_type = %s")
            params.append(document_type)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        pool = self._ensure_pool()
        with pool.connection() as connection:
            row = connection.execute(
                f"""
                SELECT MAX(message_date) AS latest_event_date
                FROM {table}
                {where_sql};
                """,
                params,
            ).fetchone()
        return row["latest_event_date"] if row else None

    async def get_latest_event_date(self, document_type: str | None = None) -> date | None:
        return await asyncio.to_thread(self._get_latest_event_date_sync, document_type)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)
