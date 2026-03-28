from __future__ import annotations

import re
from typing import Any

from app.schemas import ChunkRecord

# Korean particle suffixes to strip from tokens
_KOREAN_PARTICLES = re.compile(r"(은|는|이|가|을|를|의|에|로|와|과|으로|에서|까지|부터|마다|도|만)$")

# CamelCase pattern for system/tool names
_CAMELCASE_RE = re.compile(r"[A-Z][a-z]+(?:[A-Z][a-z]+)+")


class GraphBuilder:
    STOPWORDS = {
        # Common verbs / endings
        "합니다", "했습니다", "됩니다", "됐습니다", "입니다", "있습니다",
        "없습니다", "드립니다", "주세요", "하겠습니다", "바랍니다",
        "부탁드립니다", "감사합니다",
        # Meta / filler words
        "확인", "관련", "대화", "이슈", "기록", "정리", "관련된",
        "오늘", "내일", "이번", "최근", "현재", "전체",
        "그리고", "하지만", "그래서", "그런데", "왜냐하면",
        "여기", "거기", "어디", "무엇", "어떤", "어떻게",
        # Chat noise
        "네네", "아아", "ㅎㅎ", "ㅋㅋ", "감사", "수고",
    }

    TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_.-]+|#[0-9]+|[가-힣]{2,}")

    MAX_ENTITIES = 16

    def build_graph_rows(self, chunks: list[ChunkRecord]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for chunk in chunks:
            typed_entities = self.extract_typed_entities(chunk.chunk_text)
            entity_names = [e["name"] for e in typed_entities]
            chunk.metadata["entities"] = entity_names
            chunk.metadata["typed_entities"] = typed_entities
            rows.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "document_type": chunk.document_type,
                    "text": chunk.chunk_text,
                    "channel": chunk.channel,
                    "user_name": chunk.user_name,
                    "date": chunk.message_date.isoformat(),
                    "date_int": int(chunk.message_date.strftime("%Y%m%d")),
                    "time": chunk.message_time.isoformat(),
                    "seq": chunk.seq,
                    "token_count": chunk.token_count,
                    "access_scopes": chunk.access_scopes,
                    "title": chunk.metadata.get("title") or chunk.metadata.get("issue_title"),
                    "issue_title": chunk.metadata.get("issue_title"),
                    "status": chunk.metadata.get("status"),
                    "status_raw": chunk.metadata.get("status_raw"),
                    "assignee": chunk.metadata.get("assignee"),
                    "chunk_kind": chunk.metadata.get("chunk_kind"),
                    "flow_name": chunk.metadata.get("flow_name"),
                    "created_at_iso": chunk.metadata.get("created_at_iso"),
                    "created_at_int": chunk.metadata.get("created_at_int"),
                    "start_at_int": chunk.metadata.get("start_at_int"),
                    "due_at_int": chunk.metadata.get("due_at_int"),
                    "completed_at_int": chunk.metadata.get("completed_at_int"),
                    "entities": typed_entities,
                }
            )
        return rows

    def extract_entities(self, text: str) -> list[str]:
        """Return flat entity name list (backward-compatible)."""
        return [e["name"] for e in self.extract_typed_entities(text)]

    def extract_typed_entities(self, text: str) -> list[dict[str, str]]:
        seen: set[str] = set()
        entities: list[dict[str, str]] = []
        for token in self.TOKEN_PATTERN.findall(text):
            normalized = self._normalize(token)
            if len(normalized) < 2 or normalized in self.STOPWORDS:
                continue
            if normalized not in seen:
                entities.append({"name": normalized, "type": self._classify(token)})
                seen.add(normalized)
            if len(entities) >= self.MAX_ENTITIES:
                break
        return entities

    @staticmethod
    def _normalize(token: str) -> str:
        """Normalize a token: strip Korean particles, lowercase English."""
        # Check if token is Korean
        if any("\uac00" <= ch <= "\ud7a3" for ch in token):
            return _KOREAN_PARTICLES.sub("", token)
        # English: lowercase
        return token.lower()

    @staticmethod
    def _classify(token: str) -> str:
        """Classify entity type by pattern."""
        if token.startswith("#") and token[1:].isdigit():
            return "issue"
        if _CAMELCASE_RE.match(token):
            return "system"
        if token.isascii():
            return "system"
        # Korean 3-char: possible person name
        korean_chars = [ch for ch in token if "\uac00" <= ch <= "\ud7a3"]
        if len(korean_chars) == 3 and len(token) <= 4:
            return "person"
        return "topic"
