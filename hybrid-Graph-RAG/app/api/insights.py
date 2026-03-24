from fastapi import APIRouter, Request

from app.models.api import InsightsOverviewResponse


router = APIRouter(prefix="/api/v1/insights", tags=["insights"])


def _require_insights_service(request: Request):
    service = getattr(request.app.state, "insights_service", None)
    if service is None:
        raise RuntimeError("insights service is not available")
    return service


@router.get("/overview", response_model=InsightsOverviewResponse)
def get_insights_overview(
    request: Request,
    date_from: str | None = None,
    date_to: str | None = None,
    rooms: list[str] | None = None,
    users: list[str] | None = None,
) -> InsightsOverviewResponse:
    service = _require_insights_service(request)
    return service.overview(
        date_from=date_from,
        date_to=date_to,
        rooms=rooms or [],
        users=users or [],
    )
