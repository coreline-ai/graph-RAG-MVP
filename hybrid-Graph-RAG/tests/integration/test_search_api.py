import os

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


pytestmark = pytest.mark.integration


def test_search_api_roundtrip():
    if os.getenv("RUN_INTEGRATION") != "1":
        pytest.skip("integration tests require RUN_INTEGRATION=1")

    app = create_app(enable_runtime=True)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/search/messages",
            json={"query": "배포", "date_from": "2024-01-01", "date_to": "2024-01-31", "top_k": 5},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["query"] == "배포"
        assert "results" in payload
