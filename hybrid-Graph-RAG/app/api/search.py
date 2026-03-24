from fastapi import APIRouter, Request

from app.models.api import SearchMessagesRequest, SearchMessagesResponse


router = APIRouter(prefix="/api/v1/search", tags=["search"])


def _require_search_service(request: Request):
    service = getattr(request.app.state, "search_service", None)
    if service is None:
        raise RuntimeError("search service is not available")
    return service


@router.post("/messages", response_model=SearchMessagesResponse)
def search_messages(payload: SearchMessagesRequest, request: Request) -> SearchMessagesResponse:
    service = _require_search_service(request)
    return service.search(payload)
