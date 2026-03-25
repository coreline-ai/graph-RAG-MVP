from fastapi.testclient import TestClient

import app.main as main_module
from app.settings import Settings


class UnavailableNeo4jClient:
    def __init__(self, *args, **kwargs):
        pass

    def close(self) -> None:
        pass

    def verify_connectivity(self, raise_on_error: bool = True) -> bool:
        return False

    def fetch_total_messages(self) -> int:
        raise AssertionError("healthcheck should not fetch counts when Neo4j is unavailable")

    def fetch_last_ingestion_timestamp(self) -> str | None:
        raise AssertionError("healthcheck should not fetch timestamps when Neo4j is unavailable")


def test_app_starts_with_degraded_health_when_neo4j_is_unavailable(monkeypatch):
    monkeypatch.setattr(main_module, "Neo4jClient", UnavailableNeo4jClient)

    app = main_module.create_app(settings=Settings(), enable_runtime=True)

    with TestClient(app) as client:
        health_response = client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json() == {
            "status": "degraded",
            "neo4j_connected": False,
            "total_messages": 0,
            "last_ingestion_timestamp": None,
        }

        search_response = client.post("/api/v1/search/messages", json={"query": "배포"})
        assert search_response.status_code == 503
        assert search_response.json() == {"detail": "neo4j is not connected"}
