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
    assert result.filters.user_names == ["민수"]
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


def test_korean_name_requires_db_match() -> None:
    """C3: 3글자 한국어 단어가 users 목록에 없으면 사용자로 인식하지 않는다."""
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "진행된 작업 내용 알려줘",
        access_scopes=["public"],
        channels=[],
        users=["김민수", "박지현"],
    )

    # "진행된"은 users에 없으므로 user_names가 빈 리스트
    assert result.filters.user_names == []


def test_korean_name_matches_when_in_users() -> None:
    """C3: 3글자 한국어 이름이 users 목록에 있으면 정상 매칭."""
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "이순신이 보낸 메시지 검색",
        access_scopes=["public"],
        channels=[],
        users=["이순신", "김유신"],
    )

    assert result.filters.user_names == ["이순신"]


def test_december_month_range() -> None:
    """L3: 12월 날짜 범위가 올바르게 계산되어야 한다."""
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "2024년 12월 배포 기록",
        access_scopes=["public"],
        channels=[],
        users=[],
    )

    assert result.filters.date_from == date(2024, 12, 1)
    assert result.filters.date_to == date(2024, 12, 31)


def test_invalid_date_does_not_crash() -> None:
    """L3: 잘못된 날짜(2월 31일 등)가 입력되어도 에러가 발생하지 않는다."""
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "2024년 2월 31일 회의 내용",
        access_scopes=["public"],
        channels=[],
        users=[],
    )

    # 유효하지 않은 날짜이므로 필터가 설정되지 않아야 함
    assert result.filters.date_from is None
    assert result.filters.date_to is None


def test_multi_user_extraction() -> None:
    """복수 사용자가 쿼리에 언급되면 모두 추출된다."""
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "김민수와 박지현이 서버배포 관계 뭐라고 했어?",
        access_scopes=["public"],
        channels=[],
        users=["김민수", "박지현", "이순신"],
    )

    assert set(result.filters.user_names) == {"김민수", "박지현"}
    assert result.intent == "relationship"


def test_multi_user_korean_name_fallback() -> None:
    """직접 매칭 안 되고 한국어 이름 패턴으로도 복수 사용자 추출."""
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "이순신과 김유신의 대화 내용",
        access_scopes=["public"],
        channels=[],
        users=["이순신", "김유신"],
    )

    assert set(result.filters.user_names) == {"이순신", "김유신"}


def test_intent_detection_keywords() -> None:
    """H3: 다양한 의도가 올바르게 감지되어야 한다."""
    analyzer = QueryAnalyzer()

    timeline = analyzer.analyze("배포 흐름 알려줘", access_scopes=[], channels=[], users=[])
    assert timeline.intent == "timeline"

    relationship = analyzer.analyze("민수와 같이 언급된 사람", access_scopes=[], channels=[], users=[])
    assert relationship.intent == "relationship"

    aggregate = analyzer.analyze("가장 활발한 채널", access_scopes=[], channels=[], users=[])
    assert aggregate.intent == "aggregate"

    search = analyzer.analyze("서버 배포 관련 내용", access_scopes=[], channels=[], users=[])
    assert search.intent == "search"
