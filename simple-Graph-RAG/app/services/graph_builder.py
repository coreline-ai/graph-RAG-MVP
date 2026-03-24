from __future__ import annotations

import re

from app.schemas import ChunkRecord


class GraphBuilder:
    STOPWORDS = {
        "합니다",
        "했습니다",
        "확인",
        "부탁드립니다",
        "관련",
        "대화",
        "이슈",
        "기록",
        "오늘",
        "내일",
        "이번",
        "관련된",
        "정리",
    }

    TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_.-]+|#[0-9]+|[가-힣]{2,}")

    def build_graph_rows(self, chunks: list[ChunkRecord]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for chunk in chunks:
            entities = self.extract_entities(chunk.chunk_text)
            chunk.metadata["entities"] = entities
            rows.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "text": chunk.chunk_text,
                    "channel": chunk.channel,
                    "user_name": chunk.user_name,
                    "date": chunk.message_date.isoformat(),
                    "date_int": int(chunk.message_date.strftime("%Y%m%d")),
                    "time": chunk.message_time.isoformat(),
                    "seq": chunk.seq,
                    "token_count": chunk.token_count,
                    "access_scopes": chunk.access_scopes,
                    "entities": entities,
                }
            )
        return rows

    def extract_entities(self, text: str) -> list[str]:
        seen: set[str] = set()
        entities: list[str] = []
        for token in self.TOKEN_PATTERN.findall(text):
            normalized = token.strip()
            if len(normalized) < 2 or normalized in self.STOPWORDS:
                continue
            if normalized not in seen:
                entities.append(normalized)
                seen.add(normalized)
            if len(entities) >= 8:
                break
        return entities

