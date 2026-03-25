from fastapi import APIRouter, Request

from app.api.runtime import require_search_service
from app.models.api import MessageDetailResponse


router = APIRouter(prefix="/api/v1/messages", tags=["messages"])


@router.get("/{message_id}", response_model=MessageDetailResponse)
def get_message_detail(message_id: str, request: Request) -> MessageDetailResponse:
    service = require_search_service(request)
    return service.get_message_detail(message_id)
