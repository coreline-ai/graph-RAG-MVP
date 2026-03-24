from __future__ import annotations

import re
from datetime import date, timedelta

from app.schemas import QueryAnalysis, QueryFilters


class QueryAnalyzer:
    EXACT_DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")
    KOREAN_DATE_PATTERN = re.compile(r"(\d{4})년\s*(\d{1,2})월(?:\s*(\d{1,2})일)?")
    KOREAN_NAME_PATTERN = re.compile(r"([가-힣]{3})")

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
                if day:
                    parsed_date = date(year, month, int(day))
                    filters.date_from = parsed_date
                    filters.date_to = parsed_date
                else:
                    filters.date_from = date(year, month, 1)
                    next_month = date(year + (month // 12), (month % 12) + 1, 1)
                    filters.date_to = next_month - timedelta(days=1)
                clean_question = clean_question.replace(korean_date.group(0), " ")
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

        if not filters.user_name:
            name_match = self.KOREAN_NAME_PATTERN.search(question)
            if name_match and name_match.group(1) not in {"오늘은", "이번엔", "관련된"}:
                filters.user_name = name_match.group(1)
                clean_question = clean_question.replace(name_match.group(1), " ")

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
        if any(keyword in question for keyword in ("가장", "통계", "몇 명", "활발")):
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

