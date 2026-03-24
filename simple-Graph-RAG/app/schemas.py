from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class DocumentCreateRequest(BaseModel):
    filename: str
    content: str
    default_access_scopes: list[str] = Field(default_factory=lambda: ["public"])
    source: str = "manual"


class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    source: str = "manual"
    access_scopes: list[str] = Field(default_factory=list)
    total_messages: int = 0
    total_chunks: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentListResponse(BaseModel):
    documents: list[DocumentMetadata]


class DocumentDeleteResponse(BaseModel):
    document_id: str
    deleted: bool


class UploadFileResponse(BaseModel):
    document: DocumentMetadata


class ParsedMessage(BaseModel):
    line_number: int
    date: date
    time: time
    channel: str
    user_name: str
    content: str
    original: str
    sentences: list[str] = Field(default_factory=list)


class ChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    channel: str
    user_name: str
    message_date: date
    message_time: time
    access_scopes: list[str] = Field(default_factory=list)
    chunk_text: str
    token_count: int
    seq: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    original_lines: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 10


class QueryFilters(BaseModel):
    date_from: date | None = None
    date_to: date | None = None
    channel: str | None = None
    user_name: str | None = None
    access_scopes: list[str] = Field(default_factory=list)


class QueryAnalysis(BaseModel):
    original_question: str
    clean_question: str
    intent: Literal["search", "summary", "timeline", "aggregate", "relationship"] = "search"
    filters: QueryFilters = Field(default_factory=QueryFilters)
    entities: list[str] = Field(default_factory=list)


class GraphExpansion(BaseModel):
    chunk_id: str
    graph_neighbors: list[str] = Field(default_factory=list)
    expanded_chunk_ids: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunk_id: str
    document_id: str
    channel: str
    user_name: str
    message_date: date
    message_time: time
    access_scopes: list[str] = Field(default_factory=list)
    chunk_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    vector_score: float = 0.0
    graph_score: float = 0.0
    metadata_score: float = 0.0
    recency_score: float = 0.0
    final_score: float = 0.0
    graph_neighbors: list[str] = Field(default_factory=list)


class QuerySource(BaseModel):
    chunk_id: str
    score: float
    content: str
    graph_neighbors: list[str] = Field(default_factory=list)
    channel: str | None = None
    user_name: str | None = None
    message_date: date | None = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieval_strategy: str
    answer_mode: Literal["llm", "fallback_sources_only"]
    sources: list[QuerySource] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    neo4j: str
    postgres: str
    embedding: str
    codex_proxy: str


class CodexGenerateRequest(BaseModel):
    model: str
    system_prompt: str
    user_prompt: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CodexGenerateResponse(BaseModel):
    text: str
    model: str | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

