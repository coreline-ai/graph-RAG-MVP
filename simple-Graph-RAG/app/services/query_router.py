from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.schemas import QueryAnalysis
from app.services.query_terms import looks_like_count_query, looks_like_related_chat_query


QueryRoute = Literal["standard", "count", "mixed_issue_chat"]


@dataclass(frozen=True)
class QueryRouteDecision:
    route: QueryRoute
    strategy: str


class QueryRouter:
    def route(self, *, question: str, analysis: QueryAnalysis) -> QueryRouteDecision:
        if looks_like_count_query(question):
            return QueryRouteDecision(route="count", strategy="count_query")

        is_issue_context = (
            analysis.detected_document_type == "issue"
            or "issue" in analysis.filters.all_document_types
            or "이슈" in question
        )
        if looks_like_related_chat_query(analysis) and is_issue_context:
            return QueryRouteDecision(route="mixed_issue_chat", strategy="mixed_issue_chat")

        return QueryRouteDecision(route="standard", strategy="standard_query")
