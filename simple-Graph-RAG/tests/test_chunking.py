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
