from __future__ import annotations

from datetime import date

from app.config import Settings
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


def test_invalid_iso_date_does_not_crash() -> None:
    """잘못된 ISO 날짜도 예외 없이 무시되어야 한다."""
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "2024-13-40 회의 내용",
        access_scopes=["public"],
        channels=[],
        users=[],
    )

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


def test_issue_query_extracts_assignee_status_month_and_document_type() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "김민수님이 담당한 3월 완료 이슈 알려줘",
        access_scopes=["public"],
        channels=["general"],
        users=["김민수"],
        assignees=["김민수", "박지현"],
        statuses=["완료", "진행"],
        reference_today=date(2025, 3, 31),
    )

    assert result.detected_document_type == "issue"
    assert result.filters.document_types == ["issue"]
    assert result.filters.assignees == ["김민수"]
    assert result.filters.statuses == ["완료"]
    assert result.filters.date_from == date(2025, 3, 1)
    assert result.filters.date_to == date(2025, 3, 31)


def test_issue_query_extracts_status_variant_with_normalized_status_list() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "Sujin의 완료된 이슈 요약",
        access_scopes=["public"],
        channels=["백엔드개발"],
        users=["Sujin"],
        assignees=["Sujin", "Donghyun"],
        statuses=["보류", "완료", "진행", "재현중", "검증대기"],
        reference_today=date(2026, 3, 20),
    )

    assert result.detected_document_type == "issue"
    assert result.filters.document_types == ["issue"]
    assert result.filters.assignees == ["Sujin"]
    assert result.filters.statuses == ["완료"]


def test_issue_query_without_summary_keyword_is_treated_as_list() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "Sujin의 이슈",
        access_scopes=["public"],
        channels=["백엔드개발"],
        users=["Sujin"],
        assignees=["Sujin", "Donghyun"],
        statuses=["보류", "완료", "진행", "재현중", "검증대기"],
        reference_today=date(2026, 3, 20),
    )

    assert result.detected_document_type == "issue"
    assert result.filters.document_types == ["issue"]
    assert result.filters.assignees == ["Sujin"]
    assert result.intent == "list"


def test_progress_status_query_is_not_misclassified_as_timeline() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "Hyunwoo 담당 진행중 이슈 알려줘",
        access_scopes=["public"],
        channels=["백엔드개발"],
        users=["Hyunwoo"],
        assignees=["Hyunwoo", "Sujin"],
        statuses=["보류", "완료", "진행", "재현중", "검증대기"],
        reference_today=date(2026, 3, 20),
    )

    assert result.detected_document_type == "issue"
    assert result.filters.statuses == ["진행"]
    assert result.intent == "list"


def test_mixed_issue_chat_query_does_not_force_issue_document_type() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "최근 2주 GPU 메모리 이슈와 관련 대화 요약",
        access_scopes=["public"],
        channels=["백엔드개발", "프로젝트C"],
        users=["박소율"],
        assignees=["Sujin"],
        statuses=["완료", "진행"],
        reference_today=date(2026, 3, 20),
    )

    assert result.detected_document_type is None
    assert result.filters.document_types == []
    assert result.intent == "summary"


def test_recent_relative_date_uses_reference_today() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "최근 2주 GPU 메모리 이슈",
        access_scopes=["public"],
        channels=[],
        users=[],
        assignees=[],
        statuses=[],
        reference_today=date(2025, 3, 20),
    )

    assert result.filters.date_from == date(2025, 3, 6)
    assert result.filters.date_to == date(2025, 3, 20)


def test_document_type_hint_detects_chat_query_from_channel_and_keyword() -> None:
    analyzer = QueryAnalyzer()

    detected = analyzer.detect_document_type_hint(
        "백엔드개발 최근 대화",
        channels=["백엔드개발", "이슈데이터_10000건"],
        assignees=["Sujin"],
        statuses=["완료", "진행"],
    )

    assert detected == "chat"


def test_document_type_hint_detects_chat_content_request() -> None:
    analyzer = QueryAnalyzer()

    detected = analyzer.detect_document_type_hint(
        "서버 배포 관련 내용 알려줘",
        channels=["백엔드개발", "이슈데이터_10000건"],
        assignees=["Sujin"],
        statuses=["완료", "진행"],
    )

    assert detected == "chat"


def test_document_type_hint_keeps_issue_when_cause_keyword_present() -> None:
    analyzer = QueryAnalyzer()

    detected = analyzer.detect_document_type_hint(
        "GPU 메모리 원인 내용 알려줘",
        channels=["백엔드개발", "이슈데이터_10000건"],
        assignees=["Sujin"],
        statuses=["완료", "진행"],
    )

    assert detected == "issue"


def test_entity_extraction_filters_generic_request_tokens() -> None:
    analyzer = QueryAnalyzer(Settings(use_kiwi_keywords=False))

    result = analyzer.analyze(
        "Hyunwoo 담당 진행중 이슈 알려줘",
        access_scopes=["public"],
        channels=["백엔드개발"],
        users=["Hyunwoo"],
        assignees=["Hyunwoo"],
        statuses=["진행"],
        reference_today=date(2026, 3, 20),
    )

    assert "알려줘" not in result.entities
    assert result.entities == []


def test_entity_extraction_filters_particle_attached_generic_terms() -> None:
    analyzer = QueryAnalyzer(Settings(use_kiwi_keywords=False))

    result = analyzer.analyze(
        "최근 2주 이슈와 관련 대화 요약",
        access_scopes=["public"],
        channels=["개발팀"],
        users=["박소율"],
        assignees=["Sujin"],
        statuses=["완료"],
        reference_today=date(2026, 3, 20),
    )

    assert "이슈와" not in result.entities
    assert result.entities == []


def test_entity_extraction_keeps_special_aliases_for_oom_query() -> None:
    analyzer = QueryAnalyzer(Settings(use_kiwi_keywords=False))

    result = analyzer.analyze(
        "OOM 발견 후 어떤 수정이 이뤄졌나",
        access_scopes=["public"],
        channels=[],
        users=[],
        assignees=[],
        statuses=[],
        reference_today=date(2026, 3, 20),
    )

    assert "메모리 부족" in result.entities


def test_entity_extraction_filters_request_stems_when_kiwi_enabled() -> None:
    analyzer = QueryAnalyzer(Settings(use_kiwi_keywords=True))

    result = analyzer.analyze(
        "Hyunwoo 담당 진행중 이슈 알려줘",
        access_scopes=["public"],
        channels=["백엔드개발"],
        users=["Hyunwoo"],
        assignees=["Hyunwoo"],
        statuses=["진행"],
        reference_today=date(2026, 3, 20),
    )

    assert "알리" not in result.entities
