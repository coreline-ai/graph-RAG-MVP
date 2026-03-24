import pytest

from app.models.errors import ParseErrorCode, ParseLineError
from app.services.parser import parse_line


def test_parse_line_success():
    parsed = parse_line(
        raw_text="[2024-01-05, 07:59:12, 프로젝트C, 서버 배포 380차 완료했습니다, 박소율]",
        source_file="data/chat_logs_100.txt",
        line_no=1,
    )
    assert parsed.date == "2024-01-05"
    assert parsed.time == "07:59:12"
    assert parsed.room_name == "프로젝트C"
    assert parsed.user_name == "박소율"
    assert parsed.content == "서버 배포 380차 완료했습니다"


def test_parse_line_preserves_internal_commas():
    parsed = parse_line(
        raw_text="[2024-01-01, 06:53:58, 신규사업TF, 장애 대응 107차 완료, 모니터링 중, 장다은]",
        source_file="data/chat_logs_100.txt",
        line_no=4,
    )
    assert parsed.content == "장애 대응 107차 완료, 모니터링 중"


def test_parse_line_rejects_missing_brackets():
    with pytest.raises(ParseLineError) as exc_info:
        parse_line(
            raw_text="2024-01-01, 01:00:18, QA팀, 슬랙 채널 확인 부탁드립니다, 강시우",
            source_file="data/chat_logs_100.txt",
            line_no=2,
        )
    assert exc_info.value.failure.error_code == ParseErrorCode.BRACKET_MISMATCH


def test_parse_line_rejects_insufficient_fields():
    with pytest.raises(ParseLineError) as exc_info:
        parse_line(
            raw_text="[2024-01-01, 데이터분석팀, 좋은 의견이네요 반영하겠습니다, 황서준]",
            source_file="data/chat_logs_100.txt",
            line_no=3,
        )
    assert exc_info.value.failure.error_code == ParseErrorCode.INSUFFICIENT_FIELDS
