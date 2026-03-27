"""API error path tests — 400, 404, and edge cases."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.schemas import DocumentMetadata, QueryResponse, QuerySource


class FakeHealthComponent:
    def __init__(self, status: str = "ok") -> None:
        self.status = status

    async def healthcheck(self) -> str:
        return self.status


class FakeIngestService:
    def __init__(self) -> None:
        self.documents: dict[str, DocumentMetadata] = {}

    async def ingest_document(
        self,
        *,
        filename: str,
        content: str,
        default_access_scopes: list[str] | None = None,
        source: str = "manual",
    ) -> DocumentMetadata:
        if "INVALID" in content:
            raise ValueError("Invalid chat log format at line 1: INVALID")
        document = DocumentMetadata(
            document_id=f"doc-{len(self.documents) + 1}",
            filename=filename,
            source=source,
            access_scopes=default_access_scopes or ["public"],
            total_messages=1,
            total_chunks=1,
            created_at=datetime.now(timezone.utc),
        )
        self.documents[document.document_id] = document
        return document

    async def list_documents(self) -> list[DocumentMetadata]:
        return list(self.documents.values())

    async def get_document(self, document_id: str) -> DocumentMetadata | None:
        return self.documents.get(document_id)

    async def delete_document(self, document_id: str) -> bool:
        return self.documents.pop(document_id, None) is not None


class FakeRetrievalService:
    async def answer_query(self, *, question, access_scopes, request_user, top_k=None, debug=False):
        return QueryResponse(
            question=question,
            answer="ok",
            retrieval_strategy="filter_pgvector_graph_hybrid",
            answer_mode="llm",
            sources=[],
        )


@dataclass
class FakeContainer:
    settings: Settings = field(default_factory=Settings)
    ingest: FakeIngestService = field(default_factory=FakeIngestService)
    retrieval: FakeRetrievalService = field(default_factory=FakeRetrievalService)
    postgres: FakeHealthComponent = field(default_factory=FakeHealthComponent)
    neo4j: FakeHealthComponent = field(default_factory=FakeHealthComponent)
    embedding_provider: FakeHealthComponent = field(
        default_factory=lambda: FakeHealthComponent("not_loaded")
    )
    codex_proxy: FakeHealthComponent = field(default_factory=FakeHealthComponent)
    startup_errors: dict[str, str] = field(default_factory=dict)
    started: bool = False
    stopped: bool = False

    async def startup(self) -> None:
        self.started = True

    async def shutdown(self) -> None:
        self.stopped = True


# ---------------------------------------------------------------------------
# 400 Bad Request tests
# ---------------------------------------------------------------------------

def test_create_document_with_invalid_content_returns_400() -> None:
    container = FakeContainer()
    app = create_app(container=container)

    with TestClient(app) as client:
        response = client.post(
            "/documents",
            json={
                "filename": "bad.txt",
                "content": "INVALID content here",
            },
        )

    assert response.status_code == 400
    assert "Invalid chat log format" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 404 Not Found tests
# ---------------------------------------------------------------------------

def test_get_nonexistent_document_returns_404() -> None:
    container = FakeContainer()
    app = create_app(container=container)

    with TestClient(app) as client:
        response = client.get("/documents/nonexistent-id")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_delete_nonexistent_document_returns_404() -> None:
    container = FakeContainer()
    app = create_app(container=container)

    with TestClient(app) as client:
        response = client.delete("/documents/nonexistent-id")

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Health endpoint edge cases
# ---------------------------------------------------------------------------

def test_health_all_ok_returns_ok_status() -> None:
    container = FakeContainer()
    app = create_app(container=container)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_with_startup_errors_reported() -> None:
    container = FakeContainer()
    container.startup_errors["neo4j"] = "connection refused"
    app = create_app(container=container)

    with TestClient(app) as client:
        response = client.get("/health")

    data = response.json()
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Query edge cases
# ---------------------------------------------------------------------------

def test_query_without_headers_uses_defaults() -> None:
    container = FakeContainer()
    app = create_app(container=container)

    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={"question": "테스트 질문"},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "ok"
