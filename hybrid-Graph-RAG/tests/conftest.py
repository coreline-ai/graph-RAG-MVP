from fastapi.testclient import TestClient
import pytest

from app.main import create_app
from app.models.api import (
    CountByKey,
    InsightsOverviewResponse,
    MessageContextEntry,
    MessageDetailResponse,
    SearchMessageResult,
    SearchMessagesResponse,
    SearchResultContext,
    SearchScores,
)
from app.settings import Settings


class FakeSearchService:
    def search(self, payload):
        return SearchMessagesResponse(
            query=payload.query,
            applied_filters={
                "date_from": payload.date_from,
                "date_to": payload.date_to,
                "rooms": payload.rooms,
                "users": payload.users,
            },
            total_hits=1,
            results=[
                SearchMessageResult(
                    message_id="msg-1",
                    occurred_at="2024-01-05T07:59:12",
                    date="2024-01-05",
                    time="07:59:12",
                    room_name="프로젝트C",
                    user_name="박소율",
                    content="서버 배포 380차 완료했습니다",
                    scores=SearchScores(vector=0.9, fulltext=7.5, rrf=0.03),
                    context=SearchResultContext(
                        previous_in_room=[],
                        next_in_room=[],
                        recent_by_user=[],
                        same_day_same_room_samples=[],
                    ),
                )
            ],
        )

    def get_message_detail(self, message_id: str):
        return MessageDetailResponse(
            message_id=message_id,
            occurred_at="2024-01-05T07:59:12",
            date="2024-01-05",
            time="07:59:12",
            room_name="프로젝트C",
            user_name="박소율",
            content="서버 배포 380차 완료했습니다",
            context=SearchResultContext(
                previous_in_room=[
                    MessageContextEntry(
                        message_id="msg-0",
                        occurred_at="2024-01-05T07:40:00",
                        date="2024-01-05",
                        time="07:40:00",
                        room_name="프로젝트C",
                        user_name="김다은",
                        content="이전 배포 상태 확인했습니다",
                    )
                ],
                next_in_room=[],
                recent_by_user=[],
                same_day_same_room_samples=[],
            ),
        )


class FakeInsightsService:
    def overview(self, date_from=None, date_to=None, rooms=None, users=None):
        return InsightsOverviewResponse(
            date_from=date_from,
            date_to=date_to,
            rooms=rooms or [],
            users=users or [],
            messages_by_date=[CountByKey(key="2024-01-05", count=3)],
            top_rooms=[CountByKey(key="프로젝트C", count=3)],
            top_users=[CountByKey(key="박소율", count=1)],
            keyword_samples=[
                {
                    "keyword": "배포",
                    "message_id": "msg-1",
                    "occurred_at": "2024-01-05T07:59:12",
                    "room_name": "프로젝트C",
                    "user_name": "박소율",
                    "content": "서버 배포 380차 완료했습니다",
                }
            ],
        )


@pytest.fixture
def test_app():
    settings = Settings()
    app = create_app(settings=settings, enable_runtime=False)
    app.state.search_service = FakeSearchService()
    app.state.insights_service = FakeInsightsService()
    return app


@pytest.fixture
def client(test_app):
    return TestClient(test_app)
