"""Shared entity stopwords for graph_builder and query_analyzer.

Both modules must filter the same words so that entity extraction on stored
chunks and on incoming queries stays consistent.  Adding a word to only one
side breaks graph-based entity matching.
"""
from __future__ import annotations

# ── Structural chunk-tag words ──────────────────────────────────────
# These appear as field labels in issue chunk text (e.g. "[등록일] ...")
# and should never become entity nodes or query entity terms.
STRUCTURAL_TAG_WORDS: frozenset[str] = frozenset({
    "등록일", "확인내용", "작업내용", "업무지시", "완료예정", "완료일",
    "담당자", "진행", "기본", "문제점", "분석", "내용",
    "판단", "근거", "범위", "조치", "영향",
    "이슈", "대화", "기록", "정리", "관련", "관련된",
})

# ── Generic verbs / endings ─────────────────────────────────────────
GENERIC_VERB_WORDS: frozenset[str] = frozenset({
    "합니다", "했습니다", "됩니다", "됐습니다", "입니다", "있습니다",
    "없습니다", "드립니다", "주세요", "하겠습니다", "바랍니다",
    "부탁드립니다", "감사합니다",
})

# ── Filler / meta words ────────────────────────────────────────────
FILLER_WORDS: frozenset[str] = frozenset({
    "오늘", "내일", "이번", "최근", "현재", "전체",
    "그리고", "하지만", "그래서", "그런데", "왜냐하면",
    "여기", "거기", "어디", "무엇", "어떤", "어떻게",
    "네네", "아아", "ㅎㅎ", "ㅋㅋ", "감사", "수고",
})

# ── High-frequency generic verbs ──────────────────────────────────
# 2-char Korean words appearing in >20% of chunks that act as hub nodes
# in the graph, diluting entity search precision.
# NOTE: "확인" is intentionally excluded — it carries domain meaning
# (e.g. "로그 확인") and removing it zeroes out graph reach for
# queries like "확인 결과".
HIGH_FREQ_GENERIC_WORDS: frozenset[str] = frozenset({
    "설정",   # 40.3% — too generic ("설정 변경", "설정값" etc.)
    "로그",   # 29.2% — appears in almost every analysis section
    "결과",   # 27.5% — "확인 결과", "테스트 결과" etc.
    "패치",   # 20.7% — "패치 반영", "패치 적용" etc.
})

# ── Query-side blacklist extras ─────────────────────────────────────
# Words that only appear in user questions (not in chunk text) but still
# should not be treated as entity search terms.
QUERY_ONLY_BLACKLIST: frozenset[str] = frozenset({
    "관련해", "관련해서", "요약", "흐름", "무슨", "일이",
    "완료", "진행중", "이슈와", "이슈를", "이슈가", "이슈는",
    "담당", "알려줘", "말해줘", "알려", "말", "보여줘",
    "어떤", "무엇", "있어", "있나", "설명해줘",
    "알리", "보이", "말하", "설명하", "정리하", "요약하",
    "발견", "수정", "해결", "검증", "후속", "이뤄지",
})

# ── Combined sets for each consumer ────────────────────────────────

# graph_builder uses this for chunk entity extraction
GRAPH_ENTITY_STOPWORDS: frozenset[str] = (
    STRUCTURAL_TAG_WORDS | GENERIC_VERB_WORDS | FILLER_WORDS
    | HIGH_FREQ_GENERIC_WORDS
)

# query_analyzer uses this for query entity extraction
QUERY_ENTITY_BLACKLIST: frozenset[str] = (
    STRUCTURAL_TAG_WORDS | FILLER_WORDS | QUERY_ONLY_BLACKLIST
    | HIGH_FREQ_GENERIC_WORDS
)
