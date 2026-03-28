from __future__ import annotations

import pytest

from tests.retrieval_test_support import EchoCodexProxy, FailingCodexProxy, build_service, sample_corpus


@pytest.mark.asyncio
async def test_mixed_strategy_surfaces_issue_and_chat_sources_when_overlap_exists() -> None:
    service = build_service(chunks=sample_corpus(), codex_proxy=EchoCodexProxy())

    response = await service.answer_query(
        question="최근 2주 GPU 메모리 이슈와 관련 대화 요약",
        access_scopes=["public"],
        request_user="alice",
        top_k=6,
        debug=True,
    )

    assert response.debug is not None
    assert response.debug.route == "mixed_issue_chat"
    assert response.debug.strategy == "mixed_issue_chat"
    assert any(source.document_type == "issue" for source in response.sources)
    assert any(source.document_type == "chat" for source in response.sources)
    assert "matching_chat_sources:" in response.answer


@pytest.mark.asyncio
async def test_mixed_strategy_does_not_attach_unrelated_chat_and_reports_no_chat_coverage() -> None:
    corpus = [chunk for chunk in sample_corpus() if chunk.chunk_id != "chat-gpu"]
    service = build_service(chunks=corpus, codex_proxy=FailingCodexProxy())

    response = await service.answer_query(
        question="최근 2주 GPU 메모리 이슈와 관련 대화 요약",
        access_scopes=["public"],
        request_user="alice",
        top_k=6,
        debug=True,
    )

    assert response.debug is not None
    assert response.debug.route == "mixed_issue_chat"
    assert all(source.document_type == "issue" for source in response.sources)
    assert "채팅 근거" in response.answer
    assert "이슈 근거만 제공합니다" in response.answer
