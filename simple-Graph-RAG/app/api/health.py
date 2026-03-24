from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends

from app.container import ServiceContainer, get_container
from app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def healthcheck(
    container: ServiceContainer = Depends(get_container),
) -> HealthResponse:
    postgres_status, neo4j_status, embedding_status, codex_status = await asyncio.gather(
        container.postgres.healthcheck(),
        container.neo4j.healthcheck(),
        container.embedding_provider.healthcheck(),
        container.codex_proxy.healthcheck(),
    )

    startup_errors = getattr(container, "startup_errors", {})
    if startup_errors.get("postgres"):
        postgres_status = f"error:{startup_errors['postgres']}"
    if startup_errors.get("neo4j"):
        neo4j_status = f"error:{startup_errors['neo4j']}"

    degraded = any(
        status.startswith("error")
        for status in (postgres_status, neo4j_status, embedding_status, codex_status)
    )
    return HealthResponse(
        status="degraded" if degraded else "ok",
        neo4j=neo4j_status,
        postgres=postgres_status,
        embedding=embedding_status,
        codex_proxy=codex_status,
    )
