from __future__ import annotations

from app.schemas import QueryAnalysis, QueryFilters
from app.services.query_router import QueryRouter


def _analysis(
    question: str,
    *,
    intent: str = "search",
    detected_document_type: str | None = None,
) -> QueryAnalysis:
    return QueryAnalysis(
        original_question=question,
        clean_question=question,
        search_text=question,
        intent=intent,
        detected_document_type=detected_document_type,
        filters=QueryFilters(document_types=[detected_document_type] if detected_document_type else []),
        entities=[],
    )


def test_query_router_routes_count_questions_to_count_strategy() -> None:
    router = QueryRouter()

    decision = router.route(
        question="완료 이슈는 몇 건이야",
        analysis=_analysis("완료 이슈는 몇 건이야", intent="aggregate", detected_document_type="issue"),
    )

    assert decision.route == "count"
    assert decision.strategy == "count_query"


def test_query_router_routes_related_issue_chat_questions_to_mixed_strategy() -> None:
    router = QueryRouter()

    decision = router.route(
        question="최근 2주 GPU 메모리 이슈와 관련 대화 요약",
        analysis=_analysis(
            "최근 2주 GPU 메모리 이슈와 관련 대화 요약",
            intent="summary",
            detected_document_type="issue",
        ),
    )

    assert decision.route == "mixed_issue_chat"
    assert decision.strategy == "mixed_issue_chat"


def test_query_router_routes_plain_chat_questions_to_standard_strategy() -> None:
    router = QueryRouter()

    decision = router.route(
        question="백엔드개발 최근 대화",
        analysis=_analysis("백엔드개발 최근 대화", intent="summary", detected_document_type="chat"),
    )

    assert decision.route == "standard"
    assert decision.strategy == "standard_query"
