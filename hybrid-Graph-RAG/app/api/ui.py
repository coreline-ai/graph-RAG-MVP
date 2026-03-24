from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from app.models.api import InsightsOverviewResponse, MessageDetailResponse, SearchMessagesRequest


router = APIRouter(tags=["ui"])


def _templates(request: Request) -> Jinja2Templates:
    settings = request.app.state.settings
    return Jinja2Templates(directory=str(settings.templates_dir))


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _require_search_service(request: Request):
    service = getattr(request.app.state, "search_service", None)
    if service is None:
        raise RuntimeError("search service is not available")
    return service


def _require_insights_service(request: Request):
    service = getattr(request.app.state, "insights_service", None)
    if service is None:
        raise RuntimeError("insights service is not available")
    return service


@router.get("/")
def search_page(
    request: Request,
    q: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    rooms: str | None = None,
    users: str | None = None,
    message_id: str | None = None,
):
    templates = _templates(request)
    search_response = None
    detail_response = None
    room_list = _split_csv(rooms)
    user_list = _split_csv(users)

    if q:
        payload = SearchMessagesRequest(
            query=q,
            date_from=date_from,
            date_to=date_to,
            rooms=room_list,
            users=user_list,
        )
        search_response = _require_search_service(request).search(payload)

    if message_id:
        detail_response = _require_search_service(request).get_message_detail(message_id)

    return templates.TemplateResponse(
        request=request,
        name="search.html",
        context={
            "request": request,
            "query": q or "",
            "date_from": date_from or "",
            "date_to": date_to or "",
            "rooms": rooms or "",
            "users": users or "",
            "search_response": search_response,
            "detail_response": detail_response,
        },
    )


@router.get("/messages/{message_id}")
def message_detail_page(request: Request, message_id: str):
    templates = _templates(request)
    detail: MessageDetailResponse = _require_search_service(request).get_message_detail(message_id)
    return templates.TemplateResponse(
        request=request,
        name="message_detail.html",
        context={"request": request, "detail": detail},
    )


@router.get("/insights")
def insights_page(
    request: Request,
    date_from: str | None = None,
    date_to: str | None = None,
    rooms: str | None = None,
    users: str | None = None,
):
    templates = _templates(request)
    response: InsightsOverviewResponse = _require_insights_service(request).overview(
        date_from=date_from,
        date_to=date_to,
        rooms=_split_csv(rooms),
        users=_split_csv(users),
    )
    return templates.TemplateResponse(
        request=request,
        name="insights.html",
        context={
            "request": request,
            "date_from": date_from or "",
            "date_to": date_to or "",
            "rooms": rooms or "",
            "users": users or "",
            "insights": response,
        },
    )
