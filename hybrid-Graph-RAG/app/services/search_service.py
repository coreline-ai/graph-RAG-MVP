from fastapi import HTTPException

from app.models.api import (
    AppliedFilters,
    MessageContextEntry,
    MessageDetailResponse,
    SearchMessageResult,
    SearchMessagesRequest,
    SearchMessagesResponse,
    SearchResultContext,
    SearchScores,
)
from app.repositories.search_repo import SearchRepository
from app.services.embedder import BgeM3Embedder
from app.services.ranking import rrf_fuse


class SearchService:
    def __init__(self, search_repo: SearchRepository, embedder: BgeM3Embedder):
        self.search_repo = search_repo
        self.embedder = embedder

    def search(self, payload: SearchMessagesRequest) -> SearchMessagesResponse:
        query_vector = self.embedder.embed([payload.query])[0]
        candidate_limit = max(payload.top_k * 5, 20)
        vector_candidates = self.search_repo.fetch_vector_candidates(
            query_vector=query_vector,
            date_from=payload.date_from,
            date_to=payload.date_to,
            rooms=payload.rooms,
            users=payload.users,
            limit=candidate_limit,
        )
        fulltext_hits = self.search_repo.search_fulltext(
            query=payload.query,
            date_from=payload.date_from,
            date_to=payload.date_to,
            rooms=payload.rooms,
            users=payload.users,
            limit=candidate_limit,
        )
        fused_hits = rrf_fuse(
            vector_hits=vector_candidates,
            fulltext_hits=fulltext_hits,
            top_k=payload.top_k,
        )

        results = []
        for hit in fused_hits:
            context = self.search_repo.fetch_context(hit["message_id"])
            results.append(
                SearchMessageResult(
                    message_id=hit["message_id"],
                    occurred_at=hit["occurred_at"],
                    date=hit["date"],
                    time=hit["time"],
                    room_name=hit["room_name"],
                    user_name=hit["user_name"],
                    content=hit["content"],
                    scores=SearchScores(**hit["scores"]),
                    context=self._build_context(context),
                )
            )

        return SearchMessagesResponse(
            query=payload.query,
            applied_filters=AppliedFilters(
                date_from=payload.date_from,
                date_to=payload.date_to,
                rooms=payload.rooms,
                users=payload.users,
            ),
            total_hits=len(results),
            results=results,
        )

    def get_message_detail(self, message_id: str) -> MessageDetailResponse:
        row = self.search_repo.fetch_message_by_id(message_id)
        if row is None:
            raise HTTPException(status_code=404, detail="message not found")
        context = self.search_repo.fetch_context(message_id)
        return MessageDetailResponse(
            message_id=row["message_id"],
            occurred_at=row["occurred_at"],
            date=row["date"],
            time=row["time"],
            room_name=row["room_name"],
            user_name=row["user_name"],
            content=row["content"],
            context=self._build_context(context),
        )

    def _build_context(self, context: dict[str, list[dict]]) -> SearchResultContext:
        return SearchResultContext(
            previous_in_room=[MessageContextEntry(**row) for row in context["previous_in_room"]],
            next_in_room=[MessageContextEntry(**row) for row in context["next_in_room"]],
            recent_by_user=[MessageContextEntry(**row) for row in context["recent_by_user"]],
            same_day_same_room_samples=[
                MessageContextEntry(**row) for row in context["same_day_same_room_samples"]
            ],
        )
