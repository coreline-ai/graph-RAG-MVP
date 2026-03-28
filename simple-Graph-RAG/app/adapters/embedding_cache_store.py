from __future__ import annotations

import asyncio
import hashlib

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import Settings


class EmbeddingCacheStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pool: ConnectionPool | None = None

    def _ensure_pool(self) -> ConnectionPool:
        if self._pool is None:
            self._pool = ConnectionPool(
                self.settings.postgres_dsn,
                min_size=1,
                max_size=4,
                kwargs={"row_factory": dict_row, "autocommit": True},
                configure=self._configure_connection,
            )
        return self._pool

    @staticmethod
    def _configure_connection(connection: psycopg.Connection) -> None:
        register_vector(connection)

    @staticmethod
    def build_cache_key(model_name: str, text: str) -> str:
        raw = f"{model_name}:{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _bootstrap_sync(self) -> None:
        with psycopg.connect(self.settings.postgres_dsn, row_factory=dict_row, autocommit=True) as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            register_vector(connection)
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    cache_key TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    embedding vector({self.settings.embedding_dimensions}) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )

    async def bootstrap(self) -> None:
        await asyncio.to_thread(self._bootstrap_sync)
        await asyncio.to_thread(self._ensure_pool)

    def _get_cached_embeddings_sync(self, texts: list[str], model_name: str) -> dict[str, list[float]]:
        keys = {text: self.build_cache_key(model_name, text) for text in dict.fromkeys(texts)}
        if not keys:
            return {}
        pool = self._ensure_pool()
        with pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT cache_key, embedding
                FROM embedding_cache
                WHERE cache_key = ANY(%s);
                """,
                (list(keys.values()),),
            ).fetchall()
        embeddings_by_key = {row["cache_key"]: list(row["embedding"]) for row in rows}
        return {
            text: embeddings_by_key[key]
            for text, key in keys.items()
            if key in embeddings_by_key
        }

    async def get_cached_embeddings(self, texts: list[str], model_name: str) -> dict[str, list[float]]:
        return await asyncio.to_thread(self._get_cached_embeddings_sync, texts, model_name)

    def _upsert_embeddings_sync(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        model_name: str,
    ) -> None:
        if not texts:
            return
        params = [
            (self.build_cache_key(model_name, text), model_name, embedding)
            for text, embedding in zip(texts, embeddings, strict=True)
        ]
        pool = self._ensure_pool()
        with pool.connection() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO embedding_cache (cache_key, model_name, embedding)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (cache_key) DO UPDATE SET
                        model_name = EXCLUDED.model_name,
                        embedding = EXCLUDED.embedding;
                    """,
                    params,
                )

    async def upsert_embeddings(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        model_name: str,
    ) -> None:
        await asyncio.to_thread(self._upsert_embeddings_sync, texts, embeddings, model_name)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)
