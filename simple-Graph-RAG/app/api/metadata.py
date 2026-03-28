from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.container import ServiceContainer, get_container
from app.schemas import MetadataFacetsResponse

router = APIRouter(prefix="/metadata", tags=["metadata"])


@router.get("/facets", response_model=MetadataFacetsResponse)
async def get_metadata_facets(
    document_type: str = Query(default="all"),
    container: ServiceContainer = Depends(get_container),
) -> MetadataFacetsResponse:
    return await container.retrieval.get_facets(document_type=document_type)
