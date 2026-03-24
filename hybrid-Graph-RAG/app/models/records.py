from dataclasses import dataclass


@dataclass(slots=True)
class ParsedLine:
    source_file: str
    line_no: int
    raw_text: str
    date: str
    time: str
    room_name: str
    user_name: str
    content: str


@dataclass(slots=True)
class ChatMessageRecord:
    message_id: str
    source_file: str
    line_no: int
    raw_text: str
    date: str
    time: str
    occurred_at: str
    room_name: str
    user_name: str
    content: str
    embedding_status: str
    embedding: list[float] | None = None
