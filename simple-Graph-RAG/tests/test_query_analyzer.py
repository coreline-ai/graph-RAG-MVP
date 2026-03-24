from __future__ import annotations

from datetime import date

from app.services.query_analyzer import QueryAnalyzer


def test_query_analyzer_extracts_date_channel_user_and_intent() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "2024년 1월 15일 general에서 민수 관련 논의 요약해줘",
        access_scopes=["public", "team-a"],
        channels=["general", "random"],
        users=["민수", "지현"],
    )

    assert result.intent == "summary"
    assert result.filters.channel == "general"
    assert result.filters.user_name == "민수"
    assert result.filters.date_from == date(2024, 1, 15)
    assert result.filters.date_to == date(2024, 1, 15)
    assert result.filters.access_scopes == ["public", "team-a"]


def test_query_analyzer_handles_recent_keyword() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "최근 random 채널에서 어떤 이슈가 있었나?",
        access_scopes=["public"],
        channels=["general", "random"],
        users=[],
    )

    assert result.filters.channel == "random"
    assert result.filters.date_from is not None
    assert result.filters.date_to is not None
    assert result.filters.date_from <= result.filters.date_to
