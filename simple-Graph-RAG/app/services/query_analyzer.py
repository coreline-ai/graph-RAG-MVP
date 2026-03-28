from __future__ import annotations

import re
from datetime import date, timedelta

from app.config import Settings, get_settings
from app.schemas import QueryAnalysis, QueryFilters
from app.services.korean_nlp import build_name_pattern, extract_keywords, normalize_status

_STATUS_KEYWORDS = ("완료", "완료된", "끝난", "진행", "진행중", "진행 중", "대기", "대기중", "미완료")
_ISSUE_KEYWORDS = (
    "이슈", "담당", "등록일", "문제 원인", "업무지시", "진행", "완료",
    "해결", "재현", "검증", "원인", "수정", "후속", "오류", "에러", "장애",
)
_CHAT_KEYWORDS = ("채널", "대화", "메시지", "채팅", "공지")
_CHAT_CONTENT_TARGET_KEYWORDS = ("내용", "이야기", "얘기", "언급", "말")
_CHAT_CONTENT_REQUEST_KEYWORDS = ("알려줘", "말해줘", "무슨", "어떤", "있어", "있나")
_TIMELINE_KEYWORDS = ("흐름", "타임라인", "진행 상황", "진행상황", "진행 과정", "진행과정")
_SPECIAL_ENTITY_ALIASES = {
    "oom": ("메모리 부족", "gpu 메모리 부족"),
    "504": ("gateway", "timeout"),
    "gateway": ("gateway", "timeout"),
    "timeout": ("timeout", "타임아웃"),
    "타임아웃": ("timeout", "타임아웃"),
}
_ENTITY_BLACKLIST = {
    "관련", "관련된", "관련해", "관련해서", "대화", "요약", "정리", "내용", "기록",
    "흐름", "무슨", "일이", "최근", "완료", "진행", "진행중", "이슈", "이슈와",
    "이슈를", "이슈가", "이슈는", "담당", "알려줘", "말해줘", "알려", "말", "보여줘",
    "주세요", "어떤", "무엇", "있어", "있나", "설명해줘",
    "알리", "보이", "말하", "설명하", "정리하", "요약하",
    "발견", "수정", "해결", "검증", "후속", "이뤄지",
}
_ENTITY_PARTICLE_SUFFIXES = ("와", "과", "의", "이", "가", "을", "를", "은", "는", "도", "만")


class QueryAnalyzer:
    EXACT_DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")
    KOREAN_DATE_PATTERN = re.compile(r"(\d{4})년\s*(\d{1,2})월(?:\s*(\d{1,2})일)?")
    MONTH_PATTERN = re.compile(r"(?:(\d{4})년\s*)?(\d{1,2})월")
    RECENT_PATTERN = re.compile(r"최근(?:에)?(?:\s*(\d+)\s*(일|주|개월|달))?")
    KOREAN_NAME_PATTERN = re.compile(r"([가-힣]{3})")
    KOREAN_NAME_BLACKLIST = frozenset({
        "오늘은", "이번엔", "관련된", "진행된", "어떤거", "지금은", "최근에",
        "어제는", "내일은", "그리고", "그래서", "하지만", "그런데", "때문에",
        "무엇을", "어떻게", "이것은", "저것은", "에서는", "에서의", "으로의",
        "했던거", "한것은", "된것은", "있었던", "없었던", "했는데", "인것은",
        "위해서", "대해서", "통해서", "따르면", "가지고", "만들어", "시작된",
    })

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def analyze(
        self,
        question: str,
        *,
        access_scopes: list[str],
        channels: list[str],
        users: list[str],
        assignees: list[str] | None = None,
        statuses: list[str] | None = None,
        reference_today: date | None = None,
    ) -> QueryAnalysis:
        clean_question = question
        filters = QueryFilters(access_scopes=access_scopes)
        reference_day = reference_today or date.today()
        status_candidates = self._status_candidates(statuses or [])
        document_type_hint = self.detect_document_type_hint(
            question,
            channels=channels,
            assignees=assignees or [],
            statuses=status_candidates,
        )

        clean_question = self._extract_date_filters(clean_question, filters, reference_day)
        clean_question = self._extract_channel_filters(clean_question, filters, channels)
        clean_question = self._extract_assignee_filters(clean_question, filters, assignees or [])
        clean_question = self._extract_user_filters(clean_question, filters, users)
        clean_question = self._extract_status_filters(clean_question, filters, status_candidates)

        detected_document_type = self._detect_document_type(question, filters) or document_type_hint
        if detected_document_type and not filters.document_types:
            filters.document_types = [detected_document_type]

        cleaned = " ".join(clean_question.split()).strip() or question
        intent = self._refine_intent(
            self._detect_intent(question),
            question=question,
            cleaned_question=cleaned,
            detected_document_type=detected_document_type,
            filters=filters,
        )
        search_text = extract_keywords(cleaned, settings=self.settings)
        entities = self._extract_entities(search_text or cleaned, original_question=question)
        return QueryAnalysis(
            original_question=question,
            clean_question=cleaned,
            search_text=search_text,
            intent=intent,
            detected_document_type=detected_document_type,
            filters=filters,
            entities=entities,
        )

    def detect_document_type_hint(
        self,
        question: str,
        *,
        channels: list[str],
        assignees: list[str],
        statuses: list[str],
    ) -> str | None:
        if self._looks_like_mixed_issue_chat_query(question):
            return None
        if any(keyword in question for keyword in _ISSUE_KEYWORDS):
            return "issue"

        for assignee in sorted(assignees, key=len, reverse=True):
            if not assignee:
                continue
            pattern = build_name_pattern(assignee, allow_assignee_suffix=True)
            if pattern.search(question):
                return "issue"

        for status in sorted(self._status_candidates(statuses), key=len, reverse=True):
            if status and status in question:
                return "issue"

        if self._looks_like_chat_content_request(question):
            return "chat"

        if any(keyword in question for keyword in _CHAT_KEYWORDS):
            return "chat"

        for channel in sorted(channels, key=len, reverse=True):
            if channel and channel in question:
                return "chat"
        return None

    def _extract_date_filters(self, question: str, filters: QueryFilters, reference_day: date) -> str:
        clean_question = question
        exact_date = self.EXACT_DATE_PATTERN.search(clean_question)
        if exact_date:
            try:
                parsed_date = date.fromisoformat(exact_date.group(1))
            except ValueError:
                pass
            else:
                filters.date_from = parsed_date
                filters.date_to = parsed_date
                return clean_question.replace(exact_date.group(1), " ")

        korean_date = self.KOREAN_DATE_PATTERN.search(clean_question)
        if korean_date:
            year = int(korean_date.group(1))
            month = int(korean_date.group(2))
            day = korean_date.group(3)
            try:
                if day:
                    parsed_date = date(year, month, int(day))
                    filters.date_from = parsed_date
                    filters.date_to = parsed_date
                else:
                    filters.date_from, filters.date_to = self._month_range(year, month)
                clean_question = clean_question.replace(korean_date.group(0), " ")
            except ValueError:
                pass
            return clean_question

        month = self.MONTH_PATTERN.search(clean_question)
        if month:
            year = int(month.group(1)) if month.group(1) else reference_day.year
            try:
                filters.date_from, filters.date_to = self._month_range(year, int(month.group(2)))
                clean_question = clean_question.replace(month.group(0), " ")
            except ValueError:
                pass
            return clean_question

        if "이번 주" in clean_question or "이번주" in clean_question:
            start = reference_day - timedelta(days=reference_day.weekday())
            filters.date_from = start
            filters.date_to = reference_day
            return clean_question.replace("이번 주", " ").replace("이번주", " ")

        recent = self.RECENT_PATTERN.search(clean_question)
        if recent:
            amount = int(recent.group(1) or 7)
            unit = recent.group(2) or "일"
            days = amount
            if unit == "주":
                days = amount * 7
            elif unit in ("개월", "달"):
                days = amount * 30
            filters.date_from = reference_day - timedelta(days=days)
            filters.date_to = reference_day
            return clean_question.replace(recent.group(0), " ")

        if "오늘" in clean_question:
            filters.date_from = reference_day
            filters.date_to = reference_day
            return clean_question.replace("오늘", " ")
        if "어제" in clean_question:
            target = reference_day - timedelta(days=1)
            filters.date_from = target
            filters.date_to = target
            return clean_question.replace("어제", " ")
        return clean_question

    def _extract_channel_filters(
        self,
        question: str,
        filters: QueryFilters,
        channels: list[str],
    ) -> str:
        clean_question = question
        matched: list[str] = []
        for channel in sorted(channels, key=len, reverse=True):
            if channel and channel in clean_question:
                matched.append(channel)
                clean_question = clean_question.replace(channel, " ")
        if matched:
            filters.channels = list(dict.fromkeys(matched))
            filters.channel = filters.channels[0]
        return clean_question

    def _extract_user_filters(
        self,
        question: str,
        filters: QueryFilters,
        users: list[str],
    ) -> str:
        clean_question = question
        for user in sorted(users, key=len, reverse=True):
            if not user:
                continue
            pattern = build_name_pattern(user)
            if pattern.search(clean_question):
                if user not in filters.user_names:
                    filters.user_names.append(user)
                clean_question = pattern.sub(" ", clean_question)

        if not filters.user_names and users:
            for name_match in self.KOREAN_NAME_PATTERN.finditer(question):
                candidate = name_match.group(1)
                if candidate in self.KOREAN_NAME_BLACKLIST:
                    continue
                if candidate in users and candidate not in filters.user_names:
                    filters.user_names.append(candidate)
                    clean_question = clean_question.replace(candidate, " ")
        return clean_question

    def _extract_assignee_filters(
        self,
        question: str,
        filters: QueryFilters,
        assignees: list[str],
    ) -> str:
        clean_question = question
        for assignee in sorted(assignees, key=len, reverse=True):
            if not assignee:
                continue
            pattern = build_name_pattern(assignee, allow_assignee_suffix=True)
            if pattern.search(clean_question):
                if assignee not in filters.assignees:
                    filters.assignees.append(assignee)
                clean_question = pattern.sub(" ", clean_question)
        return clean_question

    def _extract_status_filters(
        self,
        question: str,
        filters: QueryFilters,
        statuses: list[str],
    ) -> str:
        clean_question = question
        for raw_status in sorted(statuses, key=len, reverse=True):
            if raw_status and raw_status in clean_question:
                normalized = normalize_status(raw_status)
                if normalized not in filters.statuses:
                    filters.statuses.append(normalized)
                clean_question = clean_question.replace(raw_status, " ")
        return clean_question

    @staticmethod
    def _status_candidates(statuses: list[str]) -> list[str]:
        return list(dict.fromkeys([*statuses, *_STATUS_KEYWORDS]))

    @staticmethod
    def _month_range(year: int, month: int) -> tuple[date, date]:
        start = date(year, month, 1)
        next_month = date(year + (month // 12), (month % 12) + 1, 1)
        return start, next_month - timedelta(days=1)

    def _detect_document_type(self, question: str, filters: QueryFilters) -> str | None:
        if self._looks_like_mixed_issue_chat_query(question):
            return None
        if filters.assignees or filters.statuses:
            return "issue"
        if any(keyword in question for keyword in _ISSUE_KEYWORDS):
            return "issue"
        if self._looks_like_chat_content_request(question):
            return "chat"
        if any(keyword in question for keyword in _CHAT_KEYWORDS):
            return "chat"
        return None

    @staticmethod
    def _looks_like_chat_content_request(question: str) -> bool:
        return (
            any(keyword in question for keyword in _CHAT_CONTENT_TARGET_KEYWORDS)
            and any(keyword in question for keyword in _CHAT_CONTENT_REQUEST_KEYWORDS)
        )

    @staticmethod
    def _looks_like_related_chat_query(question: str) -> bool:
        return "관련 대화" in question or ("관련" in question and "대화" in question)

    @classmethod
    def _looks_like_mixed_issue_chat_query(cls, question: str) -> bool:
        return "이슈" in question and cls._looks_like_related_chat_query(question)

    def _detect_intent(self, question: str):
        if any(keyword in question for keyword in ("같이 언급", "관계")):
            return "relationship"
        if any(keyword in question for keyword in _TIMELINE_KEYWORDS):
            return "timeline"
        if any(keyword in question for keyword in ("목록", "리스트", "누구", "이름", "멤버", "담당자 전체")):
            return "list"
        if any(keyword in question for keyword in ("가장", "통계", "몇 명", "활발", "모두", "전체", "몇 개", "몇 건", "건수")):
            return "aggregate"
        if any(keyword in question for keyword in ("요약", "무슨 일이", "정리", "어떤 일", "무엇을 했", "뭘 했")):
            return "summary"
        return "search"

    def _refine_intent(
        self,
        base_intent: str,
        *,
        question: str,
        cleaned_question: str,
        detected_document_type: str | None,
        filters: QueryFilters,
    ) -> str:
        if base_intent != "search":
            return base_intent
        if detected_document_type != "issue":
            return base_intent
        if self._looks_like_issue_list_query(question, cleaned_question, filters):
            return "list"
        return base_intent

    @staticmethod
    def _looks_like_issue_list_query(question: str, cleaned_question: str, filters: QueryFilters) -> bool:
        if "이슈" not in question:
            return False
        if any(keyword in question for keyword in ("요약", "원인", "해결", "수정", "재현", "검증", "후속", "흐름", "관계")):
            return False
        normalized = " ".join(cleaned_question.split())
        if normalized in {"이슈", "이슈들"}:
            return True
        return any(
            phrase in question
            for phrase in ("이슈 목록", "이슈 리스트", "어떤 이슈", "무슨 이슈", "이슈 알려", "이슈 보여")
        ) and bool(filters.assignees or filters.statuses or filters.date_from)

    @classmethod
    def _normalize_entity_token(cls, token: str) -> str:
        normalized = token.strip().strip(".,?![](){}<>\"'").lower()
        if not normalized:
            return ""
        if normalized in _ENTITY_BLACKLIST:
            return ""
        for suffix in _ENTITY_PARTICLE_SUFFIXES:
            if normalized.endswith(suffix) and normalized[:-1] in _ENTITY_BLACKLIST:
                return ""
        return normalized

    @classmethod
    def _extract_entities(cls, text: str, *, original_question: str | None = None) -> list[str]:
        tokens = re.findall(r"[A-Za-z0-9#._-]+|[가-힣]{2,}", text)
        entities: list[str] = []
        for token in tokens:
            normalized = cls._normalize_entity_token(token)
            if not normalized:
                continue
            if normalized not in entities:
                entities.append(normalized)
            if len(entities) >= 8:
                break
        alias_source = (original_question or text).lower()
        for needle, aliases in _SPECIAL_ENTITY_ALIASES.items():
            if needle in alias_source:
                for alias in aliases:
                    if alias not in entities:
                        entities.append(alias)
        return entities
