import os

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


pytestmark = pytest.mark.integration


def test_insights_api_roundtrip():
    if os.getenv("RUN_INTEGRATION") != "1":
        pytest.skip("integration tests require RUN_INTEGRATION=1")

    app = create_app(enable_runtime=True)
    with TestClient(app) as client:
        response = client.get("/api/v1/insights/overview?date_from=2024-01-01&date_to=2024-01-31")
        assert response.status_code == 200
        payload = response.json()
        assert "top_rooms" in payload
        assert "keyword_samples" in payload
