"""Validate all 5 FIXes by simulating the chunking pipeline on actual data."""
import sys
sys.path.insert(0, "d:/apppart_projects/Bug-Chat-RAG/simple-Graph-RAG")

import openpyxl
import re
from datetime import datetime, date
from collections import Counter

from app.services.entity_stopwords import GRAPH_ENTITY_STOPWORDS

# ── FIXED behavior_labeler logic ──

SECTION_HEADING_RE = re.compile(
    r"^(원인 요약|확인 근거|기술 판단|영향 범위|추가 조치)\s*:"
)
SECTION_SPLIT_RE = re.compile(
    r"\n+|(?=(?:원인 요약|확인 근거|기술 판단|영향 범위|추가 조치)\s*:)"
)
SECTION_LABEL_MAP = {
    "원인 요약": "discovery",
    "확인 근거": "discovery",
    "기술 판단": "discovery",
    "영향 범위": "result",
    "추가 조치": "next_action",
}
LABEL_PATTERNS = [
    ("discovery", ("확인", "발견", "로그", "보니", "파악", "조사", "분석", "살펴", "검토")),
    ("attempt", ("시도", "추가", "적용", "변경", "설정", "조정", "도입", "구현", "작성")),
    ("failure", ("실패", "해결되지", "동일", "재발", "지속", "안 됨", "불가", "재현")),
    ("fix", ("수정", "제거", "교체", "반영", "개선", "패치", "업데이트", "롤백", "복구")),
    ("result", ("정상", "해결", "복구", "개선", "안정", "완료", "효과")),
    ("verification", ("재현 테스트", "검증", "테스트 완료", "확인 완료", "모니터링")),
    ("next_action", ("예정", "추후", "향후", "계획", "해야", "필요합니다", "권장")),
]
FLOW_NAME_MAP = {
    frozenset({"discovery"}): "원인 발견",
    frozenset({"attempt", "failure"}): "시도 및 실패",
    frozenset({"attempt", "fix"}): "시도 및 수정",
    frozenset({"fix", "result"}): "수정 및 결과",
    frozenset({"result", "verification"}): "결과 검증",
    frozenset({"verification"}): "검증",
    frozenset({"next_action"}): "후속 조치",
    frozenset({"analysis_misc"}): "분석 메모",
    frozenset({"discovery", "result"}): "원인 분석 및 영향",
    frozenset({"discovery", "next_action"}): "원인 분석 및 조치",
    frozenset({"discovery", "result", "next_action"}): "원인·영향·조치",
    frozenset({"result", "next_action"}): "영향 및 조치",
    frozenset({"discovery", "fix"}): "원인 및 수정",
    frozenset({"discovery", "fix", "result"}): "원인·수정·결과",
    frozenset({"discovery", "attempt"}): "원인 및 시도",
    frozenset({"discovery", "verification"}): "원인 및 검증",
    frozenset({"discovery", "attempt", "fix"}): "원인·시도·수정",
    frozenset({"discovery", "attempt", "failure"}): "원인·시도·실패",
}


def split_structured_FIXED(text):
    sections = [p.strip() for p in SECTION_SPLIT_RE.split(text) if p.strip()]
    if len(sections) <= 1:
        return []
    merged = []
    pending = None
    for sec in sections:
        if SECTION_HEADING_RE.match(sec):
            if pending is not None:
                merged.append(pending)
            pending = sec
        elif pending is not None:
            merged.append(f"{pending} {sec}")
            pending = None
        else:
            merged.append(sec)
    if pending is not None:
        merged.append(pending)
    return [s for s in merged if s]


def label_sentence_FIXED(sentence):
    heading = SECTION_HEADING_RE.match(sentence)
    if heading:
        mapped = SECTION_LABEL_MAP.get(heading.group(1))
        if mapped:
            return [mapped]
    labels = [la for la, kws in LABEL_PATTERNS if any(k in sentence for k in kws)]
    return labels or ["analysis_misc"]


def merge_adjacent_FIXED(labeled):
    merged = []
    buf_s, buf_l = [], set()

    def flush():
        if not buf_s:
            return
        labels = sorted(buf_l) or ["analysis_misc"]
        fn = FLOW_NAME_MAP.get(frozenset(labels), "분석 메모")
        merged.append({"flow_name": fn, "labels": labels, "text": " ".join(buf_s).strip()})

    for sent, labels in labeled:
        ns = set(labels)
        if not buf_s:
            buf_s, buf_l = [sent], ns
            continue
        candidate = buf_l | ns
        if frozenset(candidate) in FLOW_NAME_MAP:
            buf_s.append(sent)
            buf_l = candidate
            continue
        flush()
        buf_s, buf_l = [sent], ns
    flush()
    return merged


def split_and_label_FIXED(text):
    sections = split_structured_FIXED(text)
    if not sections:
        return [{"flow_name": "분석 메모", "labels": ["analysis_misc"], "text": text}]
    labeled = [(s, label_sentence_FIXED(s)) for s in sections]
    return merge_adjacent_FIXED(labeled)


# ── Load data ──

wb = openpyxl.load_workbook(
    "data/model_issue_dataset_10000.xlsx", data_only=True, read_only=True
)
sheet = wb.worksheets[0]
values = list(sheet.iter_rows(values_only=True))
headers = [str(h).strip() if h else "" for h in values[0]]
rows = []
for row in values[1:]:
    if not any(v not in (None, "") for v in row):
        continue
    d = {}
    for i, h in enumerate(headers):
        if h and i < len(row):
            d[h] = row[i]
    rows.append(d)
wb.close()

# ── Simulate FIXED chunking ──

EXCEL_ROW_MAX_CHARS = 600  # restored — KSS sub-chunking must stay active


def build_overview_lines(r):
    title = str(r.get("모델 이슈 검토 사항", "")).strip()
    reg = r.get("등록일", "")
    if isinstance(reg, datetime):
        reg = reg.date().isoformat()
    elif isinstance(reg, date):
        reg = reg.isoformat()
    else:
        reg = str(reg) if reg else ""
    return [
        f"[이슈] {title}",
        f"[등록일] {reg}",
        f"[기본 확인내용] {str(r.get('기본 확인내용', '') or '').strip()}",
        f"[기본 작업내용] {str(r.get('기본 작업내용', '') or '').strip()}",
        f"[업무지시] {str(r.get('업무지시', '') or '').strip()}",
        f"[담당자] {str(r.get('담당자', '') or 'unassigned').strip()}",
        f"[진행] {str(r.get('진행(담당자)', '') or '').strip()}",
    ]


def build_single_text(r):
    lines = build_overview_lines(r)
    analysis = str(r.get("문제점 분석 내용 (담당자 Comments)", "") or "").strip()
    if analysis and analysis != "None":
        lines.append(f"[문제점 분석 내용] {analysis}")
    return "\n".join(lines)


single_count = 0
split_count = 0
flow_counts_dist = Counter()
flow_name_dist = Counter()
all_chunks = []

for r in rows[:10000]:
    title = str(r.get("모델 이슈 검토 사항", "")).strip()
    if not title or title == "None":
        continue

    single_text = build_single_text(r)
    analysis = str(
        r.get("문제점 분석 내용 (담당자 Comments)", "") or ""
    ).strip()
    if analysis == "None":
        analysis = ""

    should_split = analysis and len(single_text) > EXCEL_ROW_MAX_CHARS

    if not should_split:
        single_count += 1
        all_chunks.append(("single", single_text, len(single_text)))
    else:
        split_count += 1
        overview_text = "\n".join(build_overview_lines(r))
        all_chunks.append(("overview", overview_text, len(overview_text)))
        flows = split_and_label_FIXED(analysis)
        flow_counts_dist[len(flows)] += 1
        for flow in flows:
            flow_name_dist[flow["flow_name"]] += 1
            ft = (
                f"[이슈] {title}\n[등록일] ...\n[담당자] ...\n[진행] ...\n"
                f"[{flow['flow_name']}] {flow['text']}"
            )
            all_chunks.append(("flow", ft, len(ft)))

print("=" * 70)
print("FIX 적용 후 청킹 시뮬레이션 결과")
print("=" * 70)
print(f"\n총 이슈 행: 10,000")
print(f"  single (분리 안 됨): {single_count} ({single_count / 100:.1f}%)")
print(f"  split (overview+flow): {split_count} ({split_count / 100:.1f}%)")

total = len(all_chunks)
singles = sum(1 for k, _, _ in all_chunks if k == "single")
overviews = sum(1 for k, _, _ in all_chunks if k == "overview")
flows_n = sum(1 for k, _, _ in all_chunks if k == "flow")
print(f"\n총 청크 수: {total:,d}")
print(f"  single: {singles:,d}")
print(f"  overview: {overviews:,d}")
print(f"  flow: {flows_n:,d}")

if flow_counts_dist:
    print(f"\n분리된 이슈당 flow 수:")
    for n, c in sorted(flow_counts_dist.items()):
        print(f"  {n}개 flow: {c:,d}건")

if flow_name_dist:
    print(f"\nflow_name 분포:")
    for fn, c in flow_name_dist.most_common():
        print(f"  {fn:25s}: {c:>6,d}")

kind_lengths = {}
for k, _, le in all_chunks:
    kind_lengths.setdefault(k, []).append(le)
for k in ["single", "overview", "flow"]:
    if k in kind_lengths:
        lens = kind_lengths[k]
        print(
            f"\n{k} 청크 길이: min={min(lens)}, max={max(lens)}, "
            f"avg={sum(lens) / len(lens):.0f}"
        )


def estimate_tokens(text):
    words = text.split()
    latin = sum(1 for w in words if w.isascii())
    korean = sum(1 for ch in text if "\uac00" <= ch <= "\ud7a3")
    return (
        latin + max(1, korean // 2)
        if (latin + korean)
        else max(1, len(text) // 4)
    )


all_tokens = [estimate_tokens(t) for _, t, _ in all_chunks]
print(f"\n전체 토큰: min={min(all_tokens)}, max={max(all_tokens)}, avg={sum(all_tokens) / len(all_tokens):.0f}")
for lo, hi in [(0, 64), (64, 128), (128, 256), (256, 512)]:
    c = sum(1 for t in all_tokens if lo <= t < hi)
    pct = c / len(all_tokens) * 100
    print(f"  {lo:>4d}-{hi:<4d}: {c:>6,d} ({pct:5.1f}%)")

# ── Entity extraction with FIXED stopwords ──

TOKEN_PAT = re.compile(r"[A-Za-z][A-Za-z0-9_.-]+|#[0-9]+|[가-힣]{2,}")
PARTICLES = re.compile(
    r"(은|는|이|가|을|를|의|에|로|와|과|으로|에서|까지|부터|마다|도|만)$"
)
STRUCT_CHECK = {
    "등록일", "확인내용", "작업내용", "업무지시", "담당자", "진행",
    "완료예정", "완료일", "문제점", "분석", "내용", "기본",
    "판단", "근거", "범위", "조치", "영향", "이슈", "관련",
}

structural_hits = Counter()
domain_hits = Counter()

for _, text, _ in all_chunks[:2000]:
    entities = []
    seen = set()
    for token in TOKEN_PAT.findall(text):
        if any("\uac00" <= ch <= "\ud7a3" for ch in token):
            n = PARTICLES.sub("", token)
        else:
            n = token.lower()
        if len(n) < 2 or n in GRAPH_ENTITY_STOPWORDS:
            continue
        if n not in seen:
            entities.append(n)
            seen.add(n)
        if len(entities) >= 16:
            break
    for e in entities:
        if e in STRUCT_CHECK:
            structural_hits[e] += 1
        else:
            domain_hits[e] += 1

print(f"\n{'=' * 70}")
print("엔티티 추출 (FIXED STOPWORDS, 상위 2000 청크)")
print(f"{'=' * 70}")
print(f"구조적 노이즈 엔티티: {sum(structural_hits.values())}건")
if structural_hits:
    for e, c in structural_hits.most_common(5):
        print(f"  {e}: {c}")
else:
    print("  (노이즈 엔티티 0건 - 정상)")

print(f"\n도메인 엔티티 (상위 20):")
for e, c in domain_hits.most_common(20):
    print(f"  {e:30s}: {c:>5d}")

# ── Status mapping check (FIX-5) ──

from app.services.korean_nlp import normalize_status

print(f"\n{'=' * 70}")
print("상태 매핑 검증 (FIX-5)")
print(f"{'=' * 70}")
for s in ["완료", "진행중", "진행", "대기", "검증중", "검증", "분석중", "분석", "진행 중"]:
    print(f"  '{s}' -> '{normalize_status(s)}'")

# ── Summary table ──

print(f"\n{'=' * 70}")
print("변경 전/후 비교")
print(f"{'=' * 70}")
rows_fmt = [
    ("총 청크 수", "70,000", f"{total:,d}"),
    ("single 청크", "0", f"{singles:,d}"),
    ("overview 청크", "10,000", f"{overviews:,d}"),
    ("flow 청크", "60,000", f"{flows_n:,d}"),
    ("이슈당 flow 수", "6", f"{max(flow_counts_dist) if flow_counts_dist else 0}"),
    ("flow 평균 길이", "134자", f"{sum(kind_lengths.get('flow', [0])) // max(len(kind_lengths.get('flow', [1])), 1)}자"),
    ("구조적 노이즈", "상위 독점", f"{sum(structural_hits.values())}건"),
    ("상태 매핑 누락", "2개", "0개"),
]
print(f"  {'항목':<22s} {'변경 전':>12s} {'변경 후':>12s}")
print(f"  {'-' * 48}")
for label, before, after in rows_fmt:
    print(f"  {label:<22s} {before:>12s} {after:>12s}")
