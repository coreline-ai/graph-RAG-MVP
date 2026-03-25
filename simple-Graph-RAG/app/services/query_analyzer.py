from __future__ import annotations

import re
from datetime import date, timedelta

from app.schemas import QueryAnalysis, QueryFilters


class QueryAnalyzer:
    EXACT_DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")
    KOREAN_DATE_PATTERN = re.compile(r"(\d{4})년\s*(\d{1,2})월(?:\s*(\d{1,2})일)?")
    KOREAN_NAME_PATTERN = re.compile(r"([가-힣]{3})")
    KOREAN_NAME_BLACKLIST = frozenset({
        "오늘은", "이번엔", "관련된", "진행된", "어떤거", "지금은", "최근에",
        "어제는", "내일은", "그리고", "그래서", "하지만", "그런데", "때문에",
        "무엇을", "어떻게", "이것은", "저것은", "에서는", "에서의", "으로의",
        "했던거", "한것은", "된것은", "있었던", "없었던", "했는데", "인것은",
        "위해서", "대해서", "통해서", "따르면", "가지고", "만들어", "시작된",
    })

    def analyze(
        self,
        question: str,
        *,
        access_scopes: list[str],
        channels: list[str],
        users: list[str],
    ) -> QueryAnalysis:
        clean_question = question
        filters = QueryFilters(access_scopes=access_scopes)

        exact_date = self.EXACT_DATE_PATTERN.search(question)
        if exact_date:
            parsed_date = date.fromisoformat(exact_date.group(1))
            filters.date_from = parsed_date
            filters.date_to = parsed_date
            clean_question = clean_question.replace(exact_date.group(1), " ")
        else:
            korean_date = self.KOREAN_DATE_PATTERN.search(question)
            if korean_date:
                year = int(korean_date.group(1))
                month = int(korean_date.group(2))
                day = korean_date.group(3)
                try:
                    if day:
                        parsed_date = date(year, month, int(day))
                        filters.date_from = parsed_date
                        filters.date_to = parsed_date
                    elif 1 <= month <= 12:
                        filters.date_from = date(year, month, 1)
                        next_month = date(year + (month // 12), (month % 12) + 1, 1)
                        filters.date_to = next_month - timedelta(days=1)
                    clean_question = clean_question.replace(korean_date.group(0), " ")
                except ValueError:
                    pass
            elif "오늘" in question:
                filters.date_from = date.today()
                filters.date_to = date.today()
                clean_question = clean_question.replace("오늘", " ")
            elif "어제" in question:
                target = date.today() - timedelta(days=1)
                filters.date_from = target
                filters.date_to = target
                clean_question = clean_question.replace("어제", " ")
            elif "최근" in question:
                filters.date_from = date.today() - timedelta(days=7)
                filters.date_to = date.today()
                clean_question = clean_question.replace("최근", " ")

        for channel in sorted(channels, key=len, reverse=True):
            if channel and channel in question:
                filters.channel = channel
                clean_question = clean_question.replace(channel, " ")
                break

        for user in sorted(users, key=len, reverse=True):
            if user and user in question:
                filters.user_name = user
                clean_question = clean_question.replace(user, " ")
                break

        if not filters.user_name and users:
            for name_match in self.KOREAN_NAME_PATTERN.finditer(question):
                candidate = name_match.group(1)
                if candidate in self.KOREAN_NAME_BLACKLIST:
                    continue
                if candidate in users:
                    filters.user_name = candidate
                    clean_question = clean_question.replace(candidate, " ")
                    break

        intent = self._detect_intent(question)
        entities = self._extract_entities(clean_question)
        cleaned = " ".join(clean_question.split()).strip()
        return QueryAnalysis(
            original_question=question,
            clean_question=cleaned or question,
            intent=intent,
            filters=filters,
            entities=entities,
        )

    def _detect_intent(self, question: str):
        if any(keyword in question for keyword in ("같이 언급", "관계")):
            return "relationship"
        if any(keyword in question for keyword in ("흐름", "진행", "타임라인")):
            return "timeline"
        if any(keyword in question for keyword in ("가장", "통계", "몇 명", "활발", "모두", "목록", "전체", "리스트", "몇 개")):
            return "aggregate"
        if any(keyword in question for keyword in ("요약", "무슨 일이", "정리")):
            return "summary"
        return "search"

    def _extract_entities(self, question: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z0-9#._-]+|[가-힣]{2,}", question)
        blacklist = {"관련", "대화", "요약", "기록", "흐름", "무슨", "일이", "최근"}
        entities: list[str] = []
        for token in tokens:
            if token in blacklist:
                continue
            if token not in entities:
                entities.append(token)
            if len(entities) >= 6:
                break
        return entities

