import re

from app.models.errors import ParseErrorCode, ParseFailure, ParseLineError
from app.models.records import ParsedLine


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")


def parse_line(raw_text: str, source_file: str, line_no: int) -> ParsedLine:
    text = raw_text.strip()
    if not text.startswith("[") or not text.endswith("]"):
        raise ParseLineError(
            ParseFailure(
                source_file=source_file,
                line_no=line_no,
                raw_text=raw_text,
                error_code=ParseErrorCode.BRACKET_MISMATCH,
                message="line must start with '[' and end with ']'",
            )
        )

    body = text[1:-1]
    parts = body.split(", ")
    if len(parts) < 5:
        raise ParseLineError(
            ParseFailure(
                source_file=source_file,
                line_no=line_no,
                raw_text=raw_text,
                error_code=ParseErrorCode.INSUFFICIENT_FIELDS,
                message="line must contain at least 5 fields",
            )
        )

    date = parts[0]
    time = parts[1]
    room_name = parts[2]
    user_name = parts[-1]
    content = ", ".join(parts[3:-1])

    if not DATE_RE.match(date):
        raise ParseLineError(
            ParseFailure(
                source_file=source_file,
                line_no=line_no,
                raw_text=raw_text,
                error_code=ParseErrorCode.INVALID_DATE,
                message="date must match YYYY-MM-DD",
            )
        )
    if not TIME_RE.match(time):
        raise ParseLineError(
            ParseFailure(
                source_file=source_file,
                line_no=line_no,
                raw_text=raw_text,
                error_code=ParseErrorCode.INVALID_TIME,
                message="time must match HH:MM:SS",
            )
        )
    if not room_name.strip() or not user_name.strip() or not content.strip():
        raise ParseLineError(
            ParseFailure(
                source_file=source_file,
                line_no=line_no,
                raw_text=raw_text,
                error_code=ParseErrorCode.EMPTY_FIELD,
                message="room, content, and user must not be blank",
            )
        )

    return ParsedLine(
        source_file=source_file,
        line_no=line_no,
        raw_text=raw_text.rstrip("\n"),
        date=date,
        time=time,
        room_name=room_name,
        user_name=user_name,
        content=content,
    )
