from fastapi import APIRouter, Request

from app.models.api import MessageDetailResponse


router = APIRouter(prefix="/api/v1/messages", tags=["messages"])


def _require_search_service(request: Request):
    service = getattr(request.app.state, "search_service", None)
    if service is None:
        raise RuntimeError("search service is not available")
    return service


@router.get("/{message_id}", response_model=MessageDetailResponse)
def get_message_detail(message_id: str, request: Request) -> MessageDetailResponse:
    service = _require_search_service(request)
    return service.get_message_detail(message_id)
