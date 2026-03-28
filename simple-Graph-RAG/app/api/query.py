from __future__ import annotations

from fastapi import APIRouter, Depends, Header

from app.config import resolve_access_scopes_for_user
from app.container import ServiceContainer, get_container
from app.schemas import QueryRequest, QueryResponse

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    payload: QueryRequest,
    request_user: str | None = Header(default=None, alias="X-Request-User"),
    container: ServiceContainer = Depends(get_container),
) -> QueryResponse:
    access_scopes = resolve_access_scopes_for_user(
        settings=container.settings,
        request_user=request_user,
    )
    return await container.retrieval.answer_query(
        question=payload.question,
        access_scopes=access_scopes,
        request_user=request_user,
        top_k=payload.top_k,
        debug=payload.debug,
        request_filters=payload.filters,
    )
