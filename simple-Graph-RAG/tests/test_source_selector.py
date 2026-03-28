from __future__ import annotations

from app.schemas import QueryAnalysis, QueryFilters
from app.services.source_selector import MixedSourceSelector, StandardSourceSelector
from tests.retrieval_test_support import make_chunk


def _analysis(question: str, *, intent: str = "summary", document_type: str | None = None) -> QueryAnalysis:
    return QueryAnalysis(
        original_question=question,
        clean_question=question,
        search_text=question,
        intent=intent,
        detected_document_type=document_type,
        filters=QueryFilters(document_types=[document_type] if document_type else []),
        entities=[],
    )


def test_standard_source_selector_prefers_aggregate_overview_samples_for_generic_issue_summary() -> None:
    selector = StandardSourceSelector()
    analysis = _analysis("이슈 요약", document_type="issue")
    ranked = [
        make_chunk(
            "flow-1",
            document_type="issue",
            chunk_text="[이슈] GPU 메모리 부족 발생\n[수정 및 결과] 배치 제한",
            metadata={"chunk_kind": "analysis_flow", "issue_title": "GPU 메모리 부족 발생"},
        )
    ]
    aggregate_context = {
        "sample_chunks": [
            make_chunk(
                "overview-1",
                document_type="issue",
                chunk_text="[이슈] GPU 메모리 부족 발생",
                metadata={"chunk_kind": "overview", "issue_title": "GPU 메모리 부족 발생"},
            )
        ],
    }

    selected = selector.select(
        ranked_chunks=ranked,
        analysis=analysis,
        top_k=3,
        aggregate_context=aggregate_context,
    )

    assert [chunk.chunk_id for chunk in selected] == ["overview-1"]


def test_mixed_source_selector_blends_issue_and_chat_sources_without_duplicate_issue_titles() -> None:
    selector = MixedSourceSelector()
    analysis = _analysis("최근 2주 이슈와 관련 대화 요약", document_type="issue")
    issue_chunks = [
        make_chunk(
            "issue-overview-1",
            document_type="issue",
            chunk_text="[이슈] 최근 이슈 A",
            metadata={"chunk_kind": "overview", "issue_title": "최근 이슈 A"},
        ),
        make_chunk(
            "issue-flow-1",
            document_type="issue",
            chunk_text="[이슈] 최근 이슈 A\n[수정 및 결과] 대응 완료",
            metadata={"chunk_kind": "analysis_flow", "issue_title": "최근 이슈 A"},
        ),
        make_chunk(
            "issue-overview-2",
            document_type="issue",
            chunk_text="[이슈] 최근 이슈 B",
            metadata={"chunk_kind": "overview", "issue_title": "최근 이슈 B"},
        ),
    ]
    chat_chunks = [
        make_chunk(
            "chat-1",
            document_type="chat",
            chunk_text="개발팀 박소율: 최근 이슈 A 관련 대화",
            metadata={"original_lines": ["개발팀 박소율: 최근 이슈 A 관련 대화"]},
        )
    ]

    selected = selector.select(
        issue_chunks=issue_chunks,
        chat_chunks=chat_chunks,
        analysis=analysis,
        top_k=4,
    )

    assert [chunk.document_type for chunk in selected[:2]] == ["issue", "chat"]
    assert len([chunk for chunk in selected if chunk.document_type == "issue"]) == 2
