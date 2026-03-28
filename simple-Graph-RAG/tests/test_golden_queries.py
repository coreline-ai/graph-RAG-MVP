from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.retrieval_test_support import EchoCodexProxy, build_service, sample_corpus


def _load_cases() -> list[dict]:
    path = Path(__file__).with_name("golden_queries.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _load_cases(), ids=lambda case: case["id"])
async def test_golden_queries(case: dict) -> None:
    service = build_service(chunks=sample_corpus(), codex_proxy=EchoCodexProxy())

    response = await service.answer_query(
        question=case["question"],
        access_scopes=["public"],
        request_user="alice",
        top_k=case.get("max_source_count", 5),
        debug=True,
    )

    assert response.debug is not None
    assert response.debug.route == case["expected_route"]
    assert response.debug.count_kind == case["expected_count_kind"]
    assert case["min_source_count"] <= len(response.sources) <= case["max_source_count"]

    source_types = {source.document_type for source in response.sources}
    for required_source_type in case.get("required_source_types", []):
        assert required_source_type in source_types
    for forbidden_source_type in case.get("forbidden_source_types", []):
        assert forbidden_source_type not in source_types

    if case.get("expected_route") == "mixed_issue_chat" and not case.get("allow_no_chat", True):
        assert "chat" in source_types

    haystack_parts = [response.answer]
    haystack_parts.extend(source.content for source in response.sources)
    haystack_parts.extend(source.issue_title or "" for source in response.sources)
    haystack = "\n".join(haystack_parts)

    for term in case.get("required_terms", []):
        assert term in haystack
    for term in case.get("forbidden_terms", []):
        assert term not in haystack
