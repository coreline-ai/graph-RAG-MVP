from app.models.api import CountByKey, InsightsOverviewResponse, KeywordSample
from app.repositories.insights_repo import InsightsRepository


DEFAULT_KEYWORDS = ["배포", "롤백", "장애", "PR #", "이슈 #", "마이그레이션", "API"]


class InsightsService:
    def __init__(self, insights_repo: InsightsRepository):
        self.insights_repo = insights_repo

    def overview(
        self,
        date_from: str | None,
        date_to: str | None,
        rooms: list[str],
        users: list[str],
    ) -> InsightsOverviewResponse:
        return InsightsOverviewResponse(
            date_from=date_from,
            date_to=date_to,
            rooms=rooms,
            users=users,
            messages_by_date=[
                CountByKey(**row)
                for row in self.insights_repo.messages_by_date(date_from, date_to, rooms, users)
            ],
            top_rooms=[
                CountByKey(**row)
                for row in self.insights_repo.top_rooms(date_from, date_to, rooms, users)
            ],
            top_users=[
                CountByKey(**row)
                for row in self.insights_repo.top_users(date_from, date_to, rooms, users)
            ],
            keyword_samples=[
                KeywordSample(**row)
                for row in self.insights_repo.keyword_samples(
                    date_from, date_to, rooms, users, DEFAULT_KEYWORDS
                )
            ],
        )
