import hashlib
from pathlib import Path

from app.models.records import ChatMessageRecord, ParsedLine


def canonicalize_source_file(source_file: str) -> str:
    return str(Path(source_file).expanduser().resolve(strict=False))


def build_message_id(source_file: str, line_no: int, raw_text: str) -> str:
    normalized_source_file = canonicalize_source_file(source_file)
    payload = f"{normalized_source_file}:{line_no}:{raw_text}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def normalize_parsed_line(parsed: ParsedLine) -> ChatMessageRecord:
    normalized_source_file = canonicalize_source_file(parsed.source_file)
    return ChatMessageRecord(
        message_id=build_message_id(normalized_source_file, parsed.line_no, parsed.raw_text),
        source_file=normalized_source_file,
        line_no=parsed.line_no,
        raw_text=parsed.raw_text,
        date=parsed.date,
        time=parsed.time,
        occurred_at=f"{parsed.date}T{parsed.time}",
        room_name=parsed.room_name,
        user_name=parsed.user_name,
        content=parsed.content,
        embedding_status="pending",
        embedding=None,
    )
