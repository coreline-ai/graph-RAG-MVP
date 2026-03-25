from __future__ import annotations

from functools import lru_cache
from typing import Literal
from urllib.parse import quote_plus

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Simple-Graph-RAG"
    app_env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    top_k: int = 10
    log_level: str = "INFO"

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "graph_rag"
    postgres_user: str = "graph_rag"
    postgres_password: str = "graph_rag"
    postgres_sslmode: str = "disable"
    pgvector_table: str = "chunk_embeddings"

    neo4j_uri: str = "bolt://127.0.0.1:8768"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "graph-rag-password"
    neo4j_database: str = "neo4j"

    chunker_backend: str = "kss"
    chunk_max_tokens: int = 256
    chunk_overlap_turns: int = 1
    chunk_merge_time_gap_seconds: int = 300

    embedding_provider: str = "local_transformer"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimensions: int = 1024
    embedding_batch_size: int = 16
    embedding_device: str = "cpu"

    codex_proxy_base_url: str = "http://127.0.0.1:8800"
    codex_proxy_api_style: Literal["auto", "legacy", "openai_responses"] = "auto"
    codex_model: str = "gpt-5.3-codex"
    codex_timeout_seconds: int = 45

    graph_neighbor_hops: int = 1
    graph_next_window: int = 2
    default_access_scopes: str = Field(default="public")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def postgres_dsn(self) -> str:
        password = quote_plus(self.postgres_password)
        return (
            f"postgresql://{self.postgres_user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            f"?sslmode={self.postgres_sslmode}"
        )

    @property
    def parsed_default_access_scopes(self) -> list[str]:
        return parse_access_scopes(self.default_access_scopes)


def parse_access_scopes(raw: str | list[str] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [item.strip() for item in raw if item and item.strip()]
    return [item.strip() for item in raw.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
