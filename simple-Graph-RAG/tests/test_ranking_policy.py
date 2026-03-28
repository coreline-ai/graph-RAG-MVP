from __future__ import annotations

from datetime import date

from app.schemas import QueryAnalysis, QueryFilters
from app.services.ranking_policy import MixedRankingPolicy, RankingContext, StandardRankingPolicy
from tests.retrieval_test_support import make_chunk


def _analysis(
    question: str,
    *,
    intent: str = "summary",
    document_type: str | None = None,
    assignees: list[str] | None = None,
    statuses: list[str] | None = None,
) -> QueryAnalysis:
    return QueryAnalysis(
        original_question=question,
        clean_question=question,
        search_text=question,
        intent=intent,
        detected_document_type=document_type,
        filters=QueryFilters(
            document_types=[document_type] if document_type else [],
            assignees=assignees or [],
            statuses=statuses or [],
        ),
        entities=[],
    )


def test_standard_ranking_policy_prefers_issue_overview_for_issue_summary() -> None:
    policy = StandardRankingPolicy()
    analysis = _analysis("Sujin의 완료된 이슈 요약", document_type="issue", assignees=["Sujin"], statuses=["완료"])
    overview = make_chunk(
        "issue-overview",
        document_type="issue",
        vector_score=0.6,
        user_name="Sujin",
        message_date=date(2026, 3, 20),
        chunk_text="[이슈] GPU 메모리 부족 발생",
        metadata={"chunk_kind": "overview", "assignee": "Sujin", "status": "완료", "entities": ["GPU"]},
    )
    flow = make_chunk(
        "issue-flow",
        document_type="issue",
        vector_score=0.6,
        user_name="Sujin",
        message_date=date(2026, 3, 20),
        chunk_text="[이슈] GPU 메모리 부족 발생\n[수정 및 결과] 배치 제한",
        metadata={"chunk_kind": "analysis_flow", "assignee": "Sujin", "status": "완료", "entities": ["GPU"]},
    )

    ranked = policy.rank(
        seed_chunks=[overview, flow],
        expanded_chunks=[],
        analysis=analysis,
        context=RankingContext(latest_event_date=date(2026, 3, 20)),
    )

    assert ranked[0].chunk_id == "issue-overview"


def test_mixed_chat_ranking_policy_prefers_chat_lane_metadata_bonus_for_chat_chunks() -> None:
    chat_policy = MixedRankingPolicy(lane="chat")
    analysis = _analysis("최근 2주 GPU 메모리 이슈와 관련 대화 요약", intent="summary")
    relevant_chat = make_chunk(
        "chat-relevant",
        document_type="chat",
        vector_score=0.55,
        channel="프로젝트C",
        user_name="박소율",
        message_date=date(2026, 3, 19),
        chunk_text="프로젝트C 박소율: GPU 메모리 부족 대응 위해 배치 크기 제한 적용",
        metadata={"entities": ["GPU", "메모리 부족"], "original_lines": ["프로젝트C 박소율: GPU 메모리 부족 대응 위해 배치 크기 제한 적용"]},
    )
    generic_chat = make_chunk(
        "chat-generic",
        document_type="chat",
        vector_score=0.55,
        channel="개발팀",
        user_name="김민수",
        message_date=date(2026, 3, 19),
        chunk_text="개발팀 김민수: 회의록 공유 부탁드립니다",
        metadata={"entities": ["회의록"], "original_lines": ["개발팀 김민수: 회의록 공유 부탁드립니다"]},
    )

    ranked = chat_policy.rank(
        seed_chunks=[generic_chat, relevant_chat],
        expanded_chunks=[],
        analysis=analysis,
        context=RankingContext(latest_event_date=date(2026, 3, 20), lane="chat"),
    )

    assert ranked[0].chunk_id == "chat-relevant"
