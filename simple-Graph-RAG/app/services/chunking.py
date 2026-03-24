from __future__ import annotations

import re
from datetime import datetime, timedelta
from uuid import uuid4

from app.config import Settings
from app.schemas import ChunkRecord, ParsedMessage


class ChunkingService:
    LOG_PATTERN = re.compile(
        r"^\[(?P<date>\d{4}-\d{2}-\d{2}), (?P<time>\d{2}:\d{2}:\d{2}), "
        r"(?P<channel>.*?), (?P<content>.*), (?P<user>.*?)\]$"
    )

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def parse_log_content(self, content: str) -> list[ParsedMessage]:
        messages: list[ParsedMessage] = []
        for line_number, raw_line in enumerate(content.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            match = self.LOG_PATTERN.match(line)
            if not match:
                raise ValueError(f"Invalid chat log format at line {line_number}: {raw_line}")
            parsed_date = datetime.strptime(match.group("date"), "%Y-%m-%d").date()
            parsed_time = datetime.strptime(match.group("time"), "%H:%M:%S").time()
            body = match.group("content").strip()
            messages.append(
                ParsedMessage(
                    line_number=line_number,
                    date=parsed_date,
                    time=parsed_time,
                    channel=match.group("channel").strip(),
                    user_name=match.group("user").strip(),
                    content=body,
                    original=line,
                    sentences=self.split_sentences(body),
                )
            )
        return messages

    def split_sentences(self, text: str) -> list[str]:
        normalized = text.strip()
        if not normalized:
            return []
        if self.settings.chunker_backend.lower() == "kss":
            try:
                import kss

                sentences = kss.split_sentences(normalized)
                cleaned = [sentence.strip() for sentence in sentences if sentence.strip()]
                if cleaned:
                    return cleaned
            except Exception:
                pass
        fallback = re.split(r"(?<=[.!?])\s+|(?<=[다요죠니다])\s+", normalized)
        return [part.strip() for part in fallback if part.strip()] or [normalized]

    def build_chunks(
        self,
        messages: list[ParsedMessage],
        *,
        document_id: str | None = None,
        default_access_scopes: list[str] | None = None,
    ) -> list[ChunkRecord]:
        if not messages:
            return []
        document_id = document_id or str(uuid4())
        access_scopes = default_access_scopes or self.settings.parsed_default_access_scopes

        chunk_groups: list[list[ParsedMessage]] = []
        buffer: list[ParsedMessage] = []
        for message in messages:
            if not buffer:
                buffer = [message]
                continue
            if self._should_merge(buffer[-1], message):
                buffer.append(message)
            else:
                chunk_groups.append(buffer)
                buffer = [message]
        if buffer:
            chunk_groups.append(buffer)

        chunks: list[ChunkRecord] = []
        for seq, group in enumerate(chunk_groups):
            first = group[0]
            chunk_text = "\n".join(
                f"{message.channel} {message.user_name}: {message.content}" for message in group
            )
            token_count = max(len(chunk_text.split()), max(1, len(chunk_text) // 4))
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{document_id}_chunk_{seq:04d}",
                    document_id=document_id,
                    channel=first.channel,
                    user_name=first.user_name,
                    message_date=first.date,
                    message_time=first.time,
                    access_scopes=access_scopes,
                    chunk_text=chunk_text,
                    token_count=token_count,
                    seq=seq,
                    metadata={
                        "line_numbers": [item.line_number for item in group],
                        "original_lines": [item.original for item in group],
                        "sentences": [sentence for item in group for sentence in item.sentences],
                    },
                    original_lines=[item.original for item in group],
                )
            )
        return chunks

    def _should_merge(self, previous: ParsedMessage, current: ParsedMessage) -> bool:
        if previous.channel != current.channel or previous.user_name != current.user_name:
            return False
        previous_dt = datetime.combine(previous.date, previous.time)
        current_dt = datetime.combine(current.date, current.time)
        if current_dt - previous_dt > timedelta(seconds=self.settings.chunk_merge_time_gap_seconds):
            return False
        short_turn = len(current.content) <= 80 or len(previous.content) <= 80
        return short_turn or len(previous.sentences) <= 2

