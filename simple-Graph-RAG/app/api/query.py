from __future__ import annotations

from fastapi import APIRouter, Depends, Header

from app.config import parse_access_scopes
from app.container import ServiceContainer, get_container
from app.schemas import QueryRequest, QueryResponse

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    payload: QueryRequest,
    access_scopes_header: str | None = Header(default=None, alias="X-Access-Scopes"),
    request_user: str | None = Header(default=None, alias="X-Request-User"),
    container: ServiceContainer = Depends(get_container),
) -> QueryResponse:
    access_scopes = parse_access_scopes(access_scopes_header)
    return await container.retrieval.answer_query(
        question=payload.question,
        access_scopes=access_scopes,
        request_user=request_user,
        top_k=payload.top_k,
    )
