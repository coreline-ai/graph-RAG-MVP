from fastapi import APIRouter, Request

from app.api.runtime import require_search_service
from app.models.api import SearchMessagesRequest, SearchMessagesResponse


router = APIRouter(prefix="/api/v1/search", tags=["search"])


@router.post("/messages", response_model=SearchMessagesResponse)
def search_messages(payload: SearchMessagesRequest, request: Request) -> SearchMessagesResponse:
    service = require_search_service(request)
    return service.search(payload)
