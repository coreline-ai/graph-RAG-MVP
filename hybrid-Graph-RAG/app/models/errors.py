from dataclasses import dataclass
from enum import StrEnum


class ParseErrorCode(StrEnum):
    BRACKET_MISMATCH = "BRACKET_MISMATCH"
    INSUFFICIENT_FIELDS = "INSUFFICIENT_FIELDS"
    INVALID_DATE = "INVALID_DATE"
    INVALID_TIME = "INVALID_TIME"
    EMPTY_FIELD = "EMPTY_FIELD"


@dataclass(slots=True)
class ParseFailure:
    source_file: str
    line_no: int
    raw_text: str
    error_code: ParseErrorCode
    message: str


class ParseLineError(ValueError):
    def __init__(self, failure: ParseFailure):
        super().__init__(failure.message)
        self.failure = failure
