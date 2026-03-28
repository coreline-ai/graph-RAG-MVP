from __future__ import annotations

import pytest

from tests.retrieval_test_support import EchoCodexProxy, build_service, sample_corpus


@pytest.mark.asyncio
async def test_count_strategy_uses_overview_samples_and_exposes_count_debug_fields() -> None:
    service = build_service(chunks=sample_corpus(), codex_proxy=EchoCodexProxy())

    response = await service.answer_query(
        question="완료 이슈는 몇 건이야",
        access_scopes=["public"],
        request_user="alice",
        top_k=5,
        debug=True,
    )

    assert response.debug is not None
    assert response.debug.route == "count"
    assert response.debug.strategy == "count_query"
    assert response.debug.count_kind == "overall"
    assert response.sources
    assert all(source.document_type == "issue" for source in response.sources)
    assert all(source.source_badge == "issue-overview" for source in response.sources)
    assert "Aggregate Summary:" in response.answer
    assert "- count_basis: matching_records" in response.answer


@pytest.mark.asyncio
async def test_count_strategy_uses_subtype_aggregate_for_special_keywords() -> None:
    service = build_service(chunks=sample_corpus(), codex_proxy=EchoCodexProxy())

    response = await service.answer_query(
        question="최근 한 달 동안 timeout이나 gateway 성격의 장애 이슈가 몇 건 있었고 주요 조치가 뭐였는지 알려줘",
        access_scopes=["public"],
        request_user="alice",
        top_k=5,
        debug=True,
    )

    assert response.debug is not None
    assert response.debug.route == "count"
    assert response.debug.count_kind == "subtype"
    assert "Aggregate Summary:" in response.answer
    assert "- count_basis: special_exact_records" in response.answer or "- count_basis: special_alias_records" in response.answer
    assert "API gateway timeout 증가" in response.answer
    assert all(source.source_badge == "issue-overview" for source in response.sources)
