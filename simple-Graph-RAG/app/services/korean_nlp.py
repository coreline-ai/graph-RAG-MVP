from __future__ import annotations

import re
from functools import lru_cache

from app.config import Settings

_STOPWORD_POS = {
    "JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC",
    "EP", "EF", "EC", "ETN", "ETM",
    "SF", "SP", "SS", "SE", "SO",
}

_STATUS_MAP = {
    "완료": "완료",
    "완료된": "완료",
    "끝난": "완료",
    "진행": "진행",
    "진행중": "진행",
    "진행 중": "진행",
    "대기": "대기",
    "대기중": "대기",
    "미완료": "미완료",
}


def normalize_status(raw: str) -> str:
    cleaned = " ".join(raw.split()).strip()
    return _STATUS_MAP.get(cleaned, cleaned)


def build_name_pattern(name: str, *, allow_assignee_suffix: bool = False) -> re.Pattern[str]:
    suffix = r"(?:\s*담당)?" if allow_assignee_suffix else ""
    return re.compile(
        rf"(?<![가-힣A-Za-z0-9_]){re.escape(name)}(?:님|씨)?(?:이|가|은|는|을|를|의|와|과)?{suffix}(?=\s|$|[,.?!])"
    )


@lru_cache(maxsize=1)
def _load_kiwi():
    try:
        from kiwipiepy import Kiwi  # type: ignore
    except Exception:
        return False
    return Kiwi()


def extract_keywords(text: str, *, settings: Settings) -> str:
    if not settings.use_kiwi_keywords:
        return text
    kiwi = _load_kiwi()
    if not kiwi:
        return text
    try:
        tokens = kiwi.tokenize(text)
    except Exception:
        return text
    keywords = [token.form for token in tokens if token.tag not in _STOPWORD_POS]
    return " ".join(keywords).strip() or text
