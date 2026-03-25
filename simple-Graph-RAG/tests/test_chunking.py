from __future__ import annotations

import pytest

from app.config import Settings
from app.services.chunking import ChunkingService


def test_parse_log_content_and_merge_adjacent_turns() -> None:
    service = ChunkingService(Settings())
    content = "\n".join(
        [
            "[2024-01-15, 09:00:00, general, 작업 시작합니다. API부터 보겠습니다., 민수]",
            "[2024-01-15, 09:02:00, general, 이어서 DB 스키마도 정리할게요., 민수]",
            "[2024-01-15, 09:09:00, general, 그럼 저는 검색 파트를 맡을게요., 지현]",
        ]
    )

    messages = service.parse_log_content(content)
    chunks = service.build_chunks(messages, document_id="doc-1", default_access_scopes=["public"])

    assert len(messages) == 3
    assert len(chunks) == 2
    assert chunks[0].chunk_id == "doc-1_chunk_0000"
    assert chunks[0].metadata["line_numbers"] == [1, 2]
    assert "작업 시작합니다." in chunks[0].metadata["sentences"][0]
    assert chunks[1].user_name == "지현"


def test_parse_log_content_rejects_invalid_line() -> None:
    service = ChunkingService(Settings())

    with pytest.raises(ValueError, match="line 2"):
        service.parse_log_content(
            "\n".join(
                [
                    "[2024-01-15, 09:00:00, general, 정상 라인입니다., 민수]",
                    "this is not a valid log line",
                ]
            )
        )


def test_parse_content_with_commas_in_message() -> None:
    """C2: content에 콤마가 포함되어도 올바르게 파싱되어야 한다."""
    service = ChunkingService(Settings())
    content = "[2024-01-15, 09:00:00, general, API 호출: GET /api/v1/users, 결과 200, 민수]"

    messages = service.parse_log_content(content)

    assert len(messages) == 1
    assert messages[0].user_name == "민수"
    assert "API 호출" in messages[0].content
    assert "결과 200" in messages[0].content


def test_long_message_does_not_merge_with_short() -> None:
    """C4: 긴 메시지 뒤에 짧은 메시지가 와도 긴 메시지끼리는 병합하지 않는다."""
    service = ChunkingService(Settings())
    long_content = "서버 배포 관련하여 매우 긴 설명을 합니다. " * 20  # > 80 chars
    content = "\n".join(
        [
            f"[2024-01-15, 09:00:00, general, {long_content}, 민수]",
            "[2024-01-15, 09:00:30, general, ok, 민수]",
            f"[2024-01-15, 09:01:00, general, {long_content}, 민수]",
        ]
    )

    messages = service.parse_log_content(content)
    chunks = service.build_chunks(messages, document_id="doc-2", default_access_scopes=["public"])

    # 긴+짧은 → 병합, 긴 → 별도 청크 (최소 2개)
    assert len(chunks) >= 2


def test_chunk_max_tokens_splits_oversized_group() -> None:
    """L1: chunk_max_tokens 초과 시 그룹이 분할되어야 한다."""
    settings = Settings(chunk_max_tokens=20, chunk_merge_time_gap_seconds=600)
    service = ChunkingService(settings)
    # 짧은 메시지 10개 — 병합 조건 충족하지만 토큰 초과 시 분할
    lines = [
        f"[2024-01-15, 09:0{i}:00, dev, 메시지 내용 번호 {i}입니다, 민수]"
        for i in range(10)
    ]
    content = "\n".join(lines)

    messages = service.parse_log_content(content)
    chunks = service.build_chunks(messages, document_id="doc-3", default_access_scopes=["public"])

    # 10개의 짧은 메시지가 max_tokens=20으로 인해 여러 청크로 분할
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count > 0


def test_estimate_tokens_korean_and_english() -> None:
    """L2: 한국어/영어 혼합 텍스트에서 토큰 추정이 합리적이어야 한다."""
    assert ChunkingService._estimate_tokens("hello world") >= 2
    assert ChunkingService._estimate_tokens("서버 배포 완료") >= 2
    assert ChunkingService._estimate_tokens("API server 배포 완료했습니다") >= 4
    assert ChunkingService._estimate_tokens("") >= 1  # 최소 1
