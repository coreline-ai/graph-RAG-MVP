import logging

from fastapi import APIRouter, Request

from app.models.api import HealthResponse


router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse)
def healthcheck(request: Request) -> HealthResponse:
    neo4j_client = getattr(request.app.state, "neo4j_client", None)
    connected = False
    total_messages = 0
    last_ingestion_timestamp = None

    if neo4j_client is not None:
        connected = neo4j_client.verify_connectivity(raise_on_error=False)
        if connected:
            try:
                total_messages = neo4j_client.fetch_total_messages()
                last_ingestion_timestamp = neo4j_client.fetch_last_ingestion_timestamp()
            except Exception as exc:  # noqa: BLE001
                logger.warning("healthcheck failed while reading Neo4j metrics: %s", exc)
                connected = False
                total_messages = 0
                last_ingestion_timestamp = None

    status = "ok" if connected else "degraded"
    return HealthResponse(
        status=status,
        neo4j_connected=connected,
        total_messages=total_messages,
        last_ingestion_timestamp=last_ingestion_timestamp,
    )
