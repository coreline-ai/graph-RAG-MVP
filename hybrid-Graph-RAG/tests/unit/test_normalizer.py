from pathlib import Path

from app.services.normalizer import build_message_id, normalize_parsed_line
from app.services.parser import parse_line


def test_build_message_id_is_deterministic():
    value = build_message_id("data/chat_logs_100.txt", 1, "raw")
    assert value == build_message_id("data/chat_logs_100.txt", 1, "raw")


def test_build_message_id_normalizes_relative_and_absolute_paths():
    relative_path = "data/chat_logs_100.txt"
    absolute_path = str(Path(relative_path).resolve())
    assert build_message_id(relative_path, 1, "raw") == build_message_id(absolute_path, 1, "raw")


def test_normalize_parsed_line_builds_occurred_at():
    parsed = parse_line(
        raw_text="[2024-01-05, 07:59:12, 프로젝트C, 서버 배포 380차 완료했습니다, 박소율]",
        source_file="data/chat_logs_100.txt",
        line_no=1,
    )
    record = normalize_parsed_line(parsed)
    assert record.occurred_at == "2024-01-05T07:59:12"
    assert record.embedding_status == "pending"
    assert record.embedding is None
