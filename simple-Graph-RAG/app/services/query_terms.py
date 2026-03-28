from __future__ import annotations

from app.schemas import QueryAnalysis, RetrievedChunk

GENERIC_ISSUE_SUMMARY_QUERIES = frozenset({"이슈 요약", "이슈 정리", "요약", "정리"})
SPECIAL_LEXICAL_ALIASES = {
    "oom": ("oom", "out of memory", "메모리 부족", "gpu 메모리 부족"),
    "504": ("504", "504 에러", "504 오류", "gateway timeout", "gateway", "timeout", "타임아웃"),
    "timeout": ("timeout", "타임아웃", "응답 지연", "시간 초과"),
    "gateway": ("gateway", "게이트웨이", "gateway timeout", "504"),
}
SPECIAL_EXACT_TERMS = {
    "oom": ("oom", "out of memory"),
    "504": ("504", "504 에러", "504 오류"),
    "timeout": ("timeout", "타임아웃", "시간 초과"),
    "gateway": ("gateway", "게이트웨이", "gateway timeout"),
}
GENERIC_QUERY_TERMS = frozenset({
    "관련", "내용", "알려줘", "알려", "주세요", "정리", "요약",
    "무엇", "뭐", "어떤", "사항", "이야기", "논의", "문의",
})


def query_match_terms(analysis: QueryAnalysis) -> list[str]:
    candidates = analysis.entities or analysis.clean_question.split()
    terms: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        token = raw.strip().lower()
        if not token or token in GENERIC_QUERY_TERMS:
            continue
        if len(token) < 2 and not token.isdigit():
            continue
        if token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms


def query_phrase_candidates(analysis: QueryAnalysis) -> list[str]:
    phrases: list[str] = []
    for text in (analysis.search_text or "", analysis.clean_question):
        tokens = [
            token
            for token in text.lower().split()
            if token and token not in GENERIC_QUERY_TERMS and not token.isdigit()
        ]
        phrase = " ".join(tokens).strip()
        if len(tokens) < 2 or not phrase or phrase in phrases:
            continue
        phrases.append(phrase)
    return phrases


def chunk_search_text(chunk: RetrievedChunk) -> str:
    entity_text = " ".join(str(entity) for entity in chunk.metadata.get("entities", []))
    issue_title = str(chunk.metadata.get("issue_title") or "")
    return " ".join((chunk.chunk_text, issue_title, entity_text)).lower()


def looks_like_flow_query(analysis: QueryAnalysis) -> bool:
    return any(keyword in analysis.original_question for keyword in ("원인", "해결", "재현", "검증", "후속", "실패", "수정"))


def looks_like_count_query(question: str) -> bool:
    return any(keyword in question for keyword in ("몇 건", "몇건", "건수", "카운트", "총 몇"))


def looks_like_related_chat_query(analysis: QueryAnalysis) -> bool:
    question = analysis.original_question
    return "관련 대화" in question or ("관련" in question and "대화" in question)


def looks_like_generic_issue_summary(analysis: QueryAnalysis) -> bool:
    normalized = " ".join(analysis.clean_question.split())
    return normalized in GENERIC_ISSUE_SUMMARY_QUERIES


def looks_like_mixed_issue_chat_summary(analysis: QueryAnalysis) -> bool:
    return (
        analysis.intent == "summary"
        and looks_like_related_chat_query(analysis)
        and "이슈" in analysis.original_question
        and not analysis.entities
    )


def strict_lexical_groups(analysis: QueryAnalysis) -> list[tuple[str, ...]]:
    question = f"{analysis.original_question} {analysis.clean_question}".lower()
    groups: list[tuple[str, ...]] = []
    seen: set[str] = set()
    for keyword, aliases in SPECIAL_LEXICAL_ALIASES.items():
        if keyword in question and keyword not in seen:
            seen.add(keyword)
            groups.append(tuple(alias.lower() for alias in aliases))
    return groups


def exact_special_groups(analysis: QueryAnalysis) -> list[tuple[str, ...]]:
    question = analysis.original_question.lower()
    groups: list[tuple[str, ...]] = []
    seen: set[str] = set()
    for keyword, terms in SPECIAL_EXACT_TERMS.items():
        if keyword in question and keyword not in seen:
            seen.add(keyword)
            groups.append(tuple(term.lower() for term in terms))
    return groups


def chunk_matches_alias_group(chunk: RetrievedChunk, alias_group: tuple[str, ...]) -> bool:
    chunk_text = chunk_search_text(chunk)
    return any(alias in chunk_text for alias in alias_group)


def count_kind_for_analysis(question: str, analysis: QueryAnalysis) -> str:
    if not looks_like_count_query(question):
        return "none"
    if exact_special_groups(analysis) or strict_lexical_groups(analysis):
        return "subtype"
    return "overall"
