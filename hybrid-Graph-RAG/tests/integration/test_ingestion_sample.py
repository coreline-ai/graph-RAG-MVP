import os

import pytest

from app.repositories.ingest_repo import IngestRepository
from app.repositories.neo4j_client import Neo4jClient
from app.repositories.schema import ensure_schema
from app.services.embedder import BgeM3Embedder
from app.services.ingestion import IngestionService
from app.settings import get_settings


pytestmark = pytest.mark.integration


def _require_integration() -> None:
    if os.getenv("RUN_INTEGRATION") != "1":
        pytest.skip("integration tests require RUN_INTEGRATION=1")


def test_ingestion_sample_file():
    _require_integration()
    settings = get_settings()
    client = Neo4jClient(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    try:
        ensure_schema(client)
        service = IngestionService(
            settings=settings,
            embedder=BgeM3Embedder(settings),
            ingest_repo=IngestRepository(client),
        )
        report = service.ingest_files([settings.data_dir / "chat_logs_100.txt"], rebuild_prev_links=True)
        assert report.total == 100
        assert report.parse_failed == 0
        assert report.success == 100
    finally:
        client.close()
