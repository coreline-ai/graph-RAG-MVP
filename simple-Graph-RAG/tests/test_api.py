from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.schemas import DocumentMetadata, MetadataFacetsResponse, QueryRequestFilters, QueryResponse, QuerySource


class FakeHealthComponent:
    def __init__(self, status: str) -> None:
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
        content: str | None = None,
        file_bytes: bytes | None = None,
        default_access_scopes: list[str] | None = None,
        source: str = "manual",
        document_type: str = "chat",
        replace_filename: bool = False,
        byte_limit: int | None = None,
        row_limit: int | None = None,
    ) -> DocumentMetadata:
        raw_content = content or (file_bytes.decode("utf-8") if file_bytes else "")
        if "INVALID" in raw_content:
            raise ValueError("Invalid chat log format at line 1: INVALID")
        document = DocumentMetadata(
            document_id=f"doc-{len(self.documents) + 1}",
            filename=filename,
            source=source,
            document_type=document_type,
            access_scopes=default_access_scopes or ["public"],
            total_messages=max(raw_content.count("\n") + 1, 1),
            total_chunks=1,
            ingest_summary={},
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
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def answer_query(
        self,
        *,
        question: str,
        access_scopes: list[str],
        request_user: str | None,
        top_k: int | None = None,
        debug: bool = False,
        request_filters: QueryRequestFilters | None = None,
    ) -> QueryResponse:
        self.calls.append(
            {
                "question": question,
                "access_scopes": access_scopes,
                "request_user": request_user,
                "top_k": top_k,
                "request_filters": request_filters.model_dump(mode="json") if request_filters else None,
            }
        )
        return QueryResponse(
            question=question,
            answer="stubbed answer",
            retrieval_strategy="filter_pgvector_graph_hybrid",
            answer_mode="fallback_sources_only",
            sources=[
                QuerySource(
                    chunk_id="chunk-1",
                    score=0.91,
                    content="stubbed content",
                    graph_neighbors=["민수", "general"],
                )
            ],
        )

    async def get_facets(self, document_type: str = "all") -> MetadataFacetsResponse:
        return MetadataFacetsResponse(
            document_types=["chat", "issue"],
            channels=["general", "dev"],
            users=["민수", "지현"],
            assignees=["Sujin"],
            statuses=["완료", "진행"],
        )


@dataclass
class FakeContainer:
    settings: Settings = field(default_factory=Settings)
    ingest: FakeIngestService = field(default_factory=FakeIngestService)
    retrieval: FakeRetrievalService = field(default_factory=FakeRetrievalService)
    postgres: FakeHealthComponent = field(default_factory=lambda: FakeHealthComponent("ok"))
    neo4j: FakeHealthComponent = field(default_factory=lambda: FakeHealthComponent("ok"))
    embedding_provider: FakeHealthComponent = field(
        default_factory=lambda: FakeHealthComponent("not_loaded")
    )
    codex_proxy: FakeHealthComponent = field(default_factory=lambda: FakeHealthComponent("ok"))
    startup_errors: dict[str, str] = field(default_factory=dict)
    ready: bool = True
    started: bool = False
    stopped: bool = False

    async def startup(self) -> None:
        self.started = True

    async def shutdown(self) -> None:
        self.stopped = True


def test_health_endpoint_reports_degraded_when_component_errors() -> None:
    container = FakeContainer(codex_proxy=FakeHealthComponent("error:proxy down"))
    app = create_app(container=container)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["ready"] is True
    assert response.json()["codex_proxy"] == "error:proxy down"
    assert container.started is True
    assert container.stopped is True


def test_document_lifecycle_endpoints() -> None:
    container = FakeContainer()
    app = create_app(container=container)

    with TestClient(app) as client:
        create_response = client.post(
            "/documents",
            json={
                "filename": "chat_logs_100.txt",
                "content": "[2024-01-15, 09:00:00, general, 작업 시작합니다., 민수]",
                "default_access_scopes": ["public", "team-a"],
                "source": "manual-test",
            },
        )
        document_id = create_response.json()["document"]["document_id"]
        list_response = client.get("/documents")
        get_response = client.get(f"/documents/{document_id}")
        delete_response = client.delete(f"/documents/{document_id}")

    assert create_response.status_code == 201
    assert list_response.status_code == 200
    assert len(list_response.json()["documents"]) == 1
    assert get_response.status_code == 200
    assert get_response.json()["access_scopes"] == ["public", "team-a"]
    assert delete_response.status_code == 200
    assert delete_response.json() == {"document_id": document_id, "deleted": True}


def test_query_endpoint_forwards_scope_and_user_headers() -> None:
    container = FakeContainer(
        settings=Settings(
            default_access_scopes="public,team-default",
            request_user_access_map="alice=public,team-a",
        )
    )
    app = create_app(container=container)

    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={
                "question": "general 채널 요약",
                "top_k": 3,
                "filters": {
                    "channels": ["general", "dev"],
                    "user_names": ["민수"],
                    "date_from": "2024-01-01",
                    "date_to": "2024-01-31",
                },
            },
            headers={
                "X-Access-Scopes": "public,malicious-scope",
                "X-Request-User": "alice",
            },
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "stubbed answer"
    assert container.retrieval.calls == [
        {
            "question": "general 채널 요약",
            "access_scopes": ["public", "team-a"],
            "request_user": "alice",
            "top_k": 3,
            "request_filters": {
                "document_types": [],
                "channels": ["general", "dev"],
                "user_names": ["민수"],
                "assignees": [],
                "statuses": [],
                "date_from": "2024-01-01",
                "date_to": "2024-01-31",
            },
        }
    ]


def test_metadata_facets_endpoint_returns_issue_filters() -> None:
    container = FakeContainer()
    app = create_app(container=container)

    with TestClient(app) as client:
        response = client.get("/metadata/facets?document_type=issue")

    assert response.status_code == 200
    data = response.json()
    assert data["document_types"] == ["chat", "issue"]
    assert data["assignees"] == ["Sujin"]
    assert data["statuses"] == ["완료", "진행"]
