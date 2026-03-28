from __future__ import annotations

import json
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
    kss_min_length: int = 80
    excel_row_max_chars: int = 600
    api_issue_upload_max_bytes: int = 5 * 1024 * 1024
    api_issue_upload_max_rows: int = 300
    use_kiwi_keywords: bool = True

    embedding_provider: str = "local_transformer"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimensions: int = 1024
    embedding_batch_size: int = 16
    embedding_device: str = "cpu"

    codex_proxy_base_url: str = "http://127.0.0.1:8800"
    codex_proxy_api_style: Literal["auto", "legacy", "openai_responses"] = "auto"
    codex_model: str = "gpt-5.3-codex"
    codex_timeout_seconds: int = 45
    request_user_access_map: str = ""

    graph_neighbor_hops: int = 1
    graph_next_window: int = 2
    graph_entity_seed_enabled: bool = True
    graph_entity_seed_limit: int = 20
    graph_multihop_enabled: bool = True
    graph_entity_expansion_limit: int = 20
    graph_author_expansion_limit: int = 10
    community_detection_enabled: bool = False
    community_min_size: int = 3
    default_access_scopes: str = Field(default="public")

    model_config = SettingsConfigDict(
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

    @property
    def parsed_request_user_access_map(self) -> dict[str, list[str]]:
        return parse_request_user_access_map(self.request_user_access_map)


def parse_access_scopes(raw: str | list[str] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [item.strip() for item in raw if item and item.strip()]
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_request_user_access_map(raw: str | dict[str, str | list[str]] | None) -> dict[str, list[str]]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {
            user.strip(): parse_access_scopes(scopes)
            for user, scopes in raw.items()
            if user and user.strip()
        }

    text = raw.strip()
    if not text:
        return {}

    if text.startswith("{"):
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("request_user_access_map JSON must be an object")
        return parse_request_user_access_map(parsed)

    mapping: dict[str, list[str]] = {}
    for entry in text.split(";"):
        item = entry.strip()
        if not item:
            continue
        user, separator, scopes = item.partition("=")
        user = user.strip()
        if not separator or not user:
            raise ValueError(
                "request_user_access_map entries must use 'user=scope1,scope2' format"
            )
        mapping[user] = parse_access_scopes(scopes)
    return mapping


def resolve_access_scopes_for_user(
    *,
    settings: Settings,
    request_user: str | None,
) -> list[str]:
    normalized_user = (request_user or "").strip()
    if normalized_user:
        mapped = settings.parsed_request_user_access_map.get(normalized_user)
        if mapped:
            return mapped
    return settings.parsed_default_access_scopes


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(_env_file=".env", _env_file_encoding="utf-8")
