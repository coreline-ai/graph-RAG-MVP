from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SearchMessagesRequest(BaseModel):
    query: str
    date_from: str | None = None
    date_to: str | None = None
    rooms: list[str] = Field(default_factory=list)
    users: list[str] = Field(default_factory=list)
    top_k: int = 10

    model_config = ConfigDict(str_strip_whitespace=True)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, value: int) -> int:
        if value < 1 or value > 50:
            raise ValueError("top_k must be between 1 and 50")
        return value

    @model_validator(mode="after")
    def validate_dates(self) -> "SearchMessagesRequest":
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must not be greater than date_to")
        return self


class AppliedFilters(BaseModel):
    date_from: str | None = None
    date_to: str | None = None
    rooms: list[str] = Field(default_factory=list)
    users: list[str] = Field(default_factory=list)


class MessageContextEntry(BaseModel):
    message_id: str
    occurred_at: str
    date: str
    time: str
    room_name: str
    user_name: str
    content: str


class SearchScores(BaseModel):
    vector: float | None = None
    fulltext: float | None = None
    rrf: float


class SearchResultContext(BaseModel):
    previous_in_room: list[MessageContextEntry] = Field(default_factory=list)
    next_in_room: list[MessageContextEntry] = Field(default_factory=list)
    recent_by_user: list[MessageContextEntry] = Field(default_factory=list)
    same_day_same_room_samples: list[MessageContextEntry] = Field(default_factory=list)


class SearchMessageResult(BaseModel):
    message_id: str
    occurred_at: str
    date: str
    time: str
    room_name: str
    user_name: str
    content: str
    scores: SearchScores
    context: SearchResultContext


class SearchMessagesResponse(BaseModel):
    query: str
    applied_filters: AppliedFilters
    total_hits: int
    results: list[SearchMessageResult]


class MessageDetailResponse(BaseModel):
    message_id: str
    occurred_at: str
    date: str
    time: str
    room_name: str
    user_name: str
    content: str
    context: SearchResultContext


class CountByKey(BaseModel):
    key: str
    count: int


class KeywordSample(BaseModel):
    keyword: str
    message_id: str
    occurred_at: str
    room_name: str
    user_name: str
    content: str


class InsightsOverviewResponse(BaseModel):
    date_from: str | None = None
    date_to: str | None = None
    rooms: list[str] = Field(default_factory=list)
    users: list[str] = Field(default_factory=list)
    messages_by_date: list[CountByKey]
    top_rooms: list[CountByKey]
    top_users: list[CountByKey]
    keyword_samples: list[KeywordSample]


class HealthResponse(BaseModel):
    status: str
    neo4j_connected: bool
    total_messages: int
    last_ingestion_timestamp: str | None = None


class IngestionReport(BaseModel):
    total: int
    success: int
    parse_failed: int
    embedding_failed: int
    users_created: int
    rooms_created: int
    dates_created: int
    prev_links_created: int
    failures: list[dict[str, Any]] = Field(default_factory=list)
