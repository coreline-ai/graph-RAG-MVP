from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


DocumentType = Literal["chat", "issue"]


class DocumentCreateRequest(BaseModel):
    filename: str
    content: str
    default_access_scopes: list[str] = Field(default_factory=lambda: ["public"])
    source: str = "manual"
    document_type: DocumentType = "chat"


class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    source: str = "manual"
    document_type: DocumentType = "chat"
    access_scopes: list[str] = Field(default_factory=list)
    total_messages: int = 0
    total_chunks: int = 0
    ingest_summary: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentListResponse(BaseModel):
    documents: list[DocumentMetadata]


class DocumentDeleteResponse(BaseModel):
    document_id: str
    deleted: bool


class UploadFileResponse(BaseModel):
    document: DocumentMetadata
    ingest_summary: dict[str, Any] = Field(default_factory=dict)


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
    document_type: DocumentType = "chat"
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
    top_k: int = Field(default=10, ge=1, le=50)
    debug: bool = False
    filters: "QueryRequestFilters | None" = None


class QueryRequestFilters(BaseModel):
    date_from: date | None = None
    date_to: date | None = None
    document_types: list[DocumentType] = Field(default_factory=list)
    channels: list[str] = Field(default_factory=list)
    user_names: list[str] = Field(default_factory=list)
    assignees: list[str] = Field(default_factory=list)
    statuses: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize(self) -> "QueryRequestFilters":
        self.document_types = list(dict.fromkeys(self.document_types))
        self.channels = _dedupe_strings(self.channels)
        self.user_names = _dedupe_strings(self.user_names)
        self.assignees = _dedupe_strings(self.assignees)
        self.statuses = _dedupe_strings(self.statuses)
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must be on or before date_to")
        return self


QueryRequest.model_rebuild()


class QueryFilters(BaseModel):
    date_from: date | None = None
    date_to: date | None = None
    channel: str | None = None
    document_types: list[DocumentType] = Field(default_factory=list)
    channels: list[str] = Field(default_factory=list)
    user_names: list[str] = Field(default_factory=list)
    assignees: list[str] = Field(default_factory=list)
    statuses: list[str] = Field(default_factory=list)
    access_scopes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize(self) -> "QueryFilters":
        self.document_types = list(dict.fromkeys(self.document_types))
        normalized_channels = _dedupe_strings([
            *(self.channels or []),
            self.channel or "",
        ])
        self.channels = normalized_channels
        self.channel = normalized_channels[0] if normalized_channels else None
        self.user_names = _dedupe_strings(self.user_names)
        self.assignees = _dedupe_strings(self.assignees)
        self.statuses = _dedupe_strings(self.statuses)
        return self

    @property
    def all_channels(self) -> list[str]:
        return list(self.channels)

    @property
    def all_document_types(self) -> list[DocumentType]:
        return list(self.document_types)


class QueryAnalysis(BaseModel):
    original_question: str
    clean_question: str
    search_text: str | None = None
    intent: Literal["search", "summary", "timeline", "aggregate", "relationship", "list"] = "search"
    detected_document_type: DocumentType | None = None
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
    document_type: DocumentType = "chat"
    channel: str
    user_name: str
    message_date: date
    message_time: time
    access_scopes: list[str] = Field(default_factory=list)
    chunk_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    vector_score: float = 0.0
    graph_score: float = 0.0
    entity_score: float = 0.0
    entity_overlap_score: float = 0.0
    metadata_score: float = 0.0
    recency_score: float = 0.0
    final_score: float = 0.0
    graph_neighbors: list[str] = Field(default_factory=list)
    retrieval_source: Literal["vector", "graph_entity", "graph_expanded"] = "vector"


class GraphTriple(BaseModel):
    source: str
    relationship: str
    target: str
    target_type: str


class SubgraphContext(BaseModel):
    triples: list[GraphTriple] = Field(default_factory=list)
    entity_summary: dict[str, int] = Field(default_factory=dict)


class QuerySource(BaseModel):
    chunk_id: str
    score: float
    content: str
    document_type: DocumentType = "chat"
    source_badge: str = "chat"
    graph_neighbors: list[str] = Field(default_factory=list)
    channel: str | None = None
    user_name: str | None = None
    message_date: date | None = None
    issue_title: str | None = None
    assignee: str | None = None
    status: str | None = None
    flow_name: str | None = None


class PipelineTiming(BaseModel):
    step: str
    duration_ms: float
    start_offset_ms: float = 0.0


class ScoreBreakdown(BaseModel):
    chunk_id: str
    vector_score: float = 0.0
    graph_score: float = 0.0
    entity_score: float = 0.0
    metadata_score: float = 0.0
    recency_score: float = 0.0
    final_score: float = 0.0
    retrieval_source: str = "vector"


class IntentWeights(BaseModel):
    intent: str
    vector: float
    graph: float
    entity: float
    metadata: float
    recency: float


class SubgraphNode(BaseModel):
    id: str
    label: str
    type: str


class SubgraphEdge(BaseModel):
    source: str
    target: str
    relationship: str


class CooccurrenceEdge(BaseModel):
    entity_a: str
    entity_b: str
    shared_chunk_count: int


class CommunityCluster(BaseModel):
    community_id: str
    summary: str
    entities: list[str] = Field(default_factory=list)


class DebugData(BaseModel):
    timing: list[PipelineTiming] = Field(default_factory=list)
    total_time_ms: float = 0.0
    bottleneck_step: str = ""
    score_breakdowns: list[ScoreBreakdown] = Field(default_factory=list)
    intent_weights: IntentWeights | None = None
    subgraph_nodes: list[SubgraphNode] = Field(default_factory=list)
    subgraph_edges: list[SubgraphEdge] = Field(default_factory=list)
    query_entities: list[str] = Field(default_factory=list)
    seed_chunk_ids: list[str] = Field(default_factory=list)
    graph_seeded_chunk_ids: list[str] = Field(default_factory=list)
    expanded_chunk_ids: list[str] = Field(default_factory=list)
    entity_mention_counts: dict[str, int] = Field(default_factory=dict)
    cooccurrence_edges: list[CooccurrenceEdge] = Field(default_factory=list)
    community_clusters: list[CommunityCluster] = Field(default_factory=list)
    vector_only_ids: list[str] = Field(default_factory=list)
    graph_entity_ids: list[str] = Field(default_factory=list)
    multihop_ids: list[str] = Field(default_factory=list)
    detected_intent: str = "search"
    clean_question: str = ""
    route: str = "standard"
    strategy: str = "standard_query"
    count_kind: Literal["overall", "subtype", "none"] = "none"
    chat_match_count: int = 0
    issue_match_count: int = 0


class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieval_strategy: str
    answer_mode: Literal["llm", "fallback_sources_only"]
    sources: list[QuerySource] = Field(default_factory=list)
    debug: DebugData | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ready: bool = True
    neo4j: str
    postgres: str
    embedding: str
    codex_proxy: str
    startup_errors: dict[str, str] = Field(default_factory=dict)


class IssueRow(BaseModel):
    sheet_name: str
    row_index: int
    title: str
    registered_date: date
    start_date: date | None = None
    due_date: date | None = None
    completed_date: date | None = None
    check_text: str = ""
    work_text: str = ""
    instruction_text: str = ""
    assignee: str = "unassigned"
    status: str = ""
    status_raw: str = ""
    analysis: str = ""


class BehaviorFlowChunk(BaseModel):
    flow_name: str
    labels: list[str] = Field(default_factory=list)
    text: str


class WorkbookParseResult(BaseModel):
    rows: list[IssueRow] = Field(default_factory=list)
    total_rows: int = 0
    skipped_rows: int = 0
    warnings: list[str] = Field(default_factory=list)


class MetadataFacetsResponse(BaseModel):
    document_types: list[str] = Field(default_factory=list)
    channels: list[str] = Field(default_factory=list)
    users: list[str] = Field(default_factory=list)
    assignees: list[str] = Field(default_factory=list)
    statuses: list[str] = Field(default_factory=list)
    latest_event_date: date | None = None


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


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return deduped
