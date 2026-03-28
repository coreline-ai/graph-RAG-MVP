from __future__ import annotations

import re
from typing import Iterable

from app.config import Settings
from app.schemas import BehaviorFlowChunk

_LABEL_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("discovery", ("확인", "발견", "로그", "보니", "파악", "조사", "분석", "살펴", "검토", "나타나", "드러나")),
    ("attempt", ("시도", "추가", "적용", "변경", "설정", "조정", "도입", "구현", "작성")),
    ("failure", ("실패", "해결되지", "동일", "재발", "지속", "안 됨", "불가", "재현", "여전히", "못")),
    ("fix", ("수정", "제거", "교체", "반영", "개선", "패치", "업데이트", "롤백", "복구")),
    ("result", ("정상", "해결", "복구", "개선", "안정", "완료", "효과")),
    ("verification", ("재현 테스트", "검증", "테스트 완료", "확인 완료", "모니터링", "관찰", "추적")),
    ("next_action", ("예정", "추후", "향후", "계획", "해야", "필요합니다", "권장")),
]

_FLOW_NAME_MAP: dict[frozenset[str], str] = {
    frozenset({"discovery"}): "원인 발견",
    frozenset({"attempt", "failure"}): "시도 및 실패",
    frozenset({"attempt", "fix"}): "시도 및 수정",
    frozenset({"fix", "result"}): "수정 및 결과",
    frozenset({"result", "verification"}): "결과 검증",
    frozenset({"verification"}): "검증",
    frozenset({"next_action"}): "후속 조치",
    frozenset({"analysis_misc"}): "분석 메모",
}

_SECTION_HEADING_RE = re.compile(r"^(원인 요약|확인 근거|기술 판단|영향 범위|추가 조치)\s*:")
_SECTION_SPLIT_RE = re.compile(r"\n+|(?=(?:원인 요약|확인 근거|기술 판단|영향 범위|추가 조치)\s*:)")
_SECTION_LABEL_MAP = {
    "원인 요약": "discovery",
    "확인 근거": "discovery",
    "기술 판단": "discovery",
    "영향 범위": "result",
    "추가 조치": "next_action",
}


class BehaviorLabeler:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def split_and_label(self, text: str) -> list[BehaviorFlowChunk]:
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        labeled = [(sentence, self._label_sentence(sentence)) for sentence in sentences]
        return self._merge_adjacent(labeled)

    def _split_sentences(self, text: str) -> list[str]:
        normalized = text.strip()
        if not normalized:
            return []
        section_sentences = self._split_structured_sections(normalized)
        if section_sentences:
            return section_sentences
        try:
            import kss

            if len(normalized) >= self.settings.kss_min_length:
                try:
                    sentences = kss.split_sentences(normalized, backend="pecab")
                except TypeError:
                    sentences = kss.split_sentences(normalized)
            else:
                sentences = [normalized]
            cleaned = [sentence.strip() for sentence in sentences if sentence.strip()]
            if cleaned:
                return cleaned
        except Exception:
            pass
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+|(?<=[다요죠니다])\s+", normalized)
            if sentence.strip()
        ] or [normalized]

    def _split_structured_sections(self, text: str) -> list[str]:
        sections = [part.strip() for part in _SECTION_SPLIT_RE.split(text) if part.strip()]
        if len(sections) <= 1:
            return []

        sentences: list[str] = []
        for section in sections:
            if _SECTION_HEADING_RE.match(section):
                sentences.append(section)
                continue
            sentences.extend(self._split_plain_text(section))
        return [sentence for sentence in sentences if sentence]

    def _split_plain_text(self, text: str) -> list[str]:
        normalized = text.strip()
        if not normalized:
            return []
        try:
            import kss

            if len(normalized) >= self.settings.kss_min_length:
                try:
                    sentences = kss.split_sentences(normalized, backend="pecab")
                except TypeError:
                    sentences = kss.split_sentences(normalized)
            else:
                sentences = [normalized]
            cleaned = [sentence.strip() for sentence in sentences if sentence.strip()]
            if cleaned:
                return cleaned
        except Exception:
            pass
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+|(?<=[다요죠니다])\s+", normalized)
            if sentence.strip()
        ] or [normalized]

    def _label_sentence(self, sentence: str) -> list[str]:
        heading_match = _SECTION_HEADING_RE.match(sentence)
        if heading_match:
            mapped = _SECTION_LABEL_MAP.get(heading_match.group(1))
            if mapped:
                return [mapped]
        labels = [label for label, keywords in _LABEL_PATTERNS if any(keyword in sentence for keyword in keywords)]
        return labels or ["analysis_misc"]

    def _merge_adjacent(
        self,
        sentences: Iterable[tuple[str, list[str]]],
    ) -> list[BehaviorFlowChunk]:
        merged: list[BehaviorFlowChunk] = []
        buffer_sentences: list[str] = []
        buffer_labels: set[str] = set()

        def flush() -> None:
            if not buffer_sentences:
                return
            labels = sorted(buffer_labels) or ["analysis_misc"]
            flow_name = _FLOW_NAME_MAP.get(frozenset(labels), "분석 메모")
            merged.append(
                BehaviorFlowChunk(
                    flow_name=flow_name,
                    labels=labels,
                    text=" ".join(buffer_sentences).strip(),
                )
            )

        for sentence, labels in sentences:
            next_label_set = set(labels)
            if not buffer_sentences:
                buffer_sentences = [sentence]
                buffer_labels = next_label_set
                continue
            candidate = buffer_labels | next_label_set
            if frozenset(candidate) in _FLOW_NAME_MAP:
                buffer_sentences.append(sentence)
                buffer_labels = candidate
                continue
            flush()
            buffer_sentences = [sentence]
            buffer_labels = next_label_set

        flush()
        return merged
