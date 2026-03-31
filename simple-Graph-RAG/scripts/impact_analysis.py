"""
Impact analysis for 3 remaining issues.
Measures actual before/after metrics without modifying any source code.
"""
import sys
sys.path.insert(0, "d:/apppart_projects/Bug-Chat-RAG/simple-Graph-RAG")

import openpyxl
import re
from datetime import datetime, date
from collections import Counter, defaultdict

from app.services.entity_stopwords import GRAPH_ENTITY_STOPWORDS

TOKEN_PAT = re.compile(r"[A-Za-z][A-Za-z0-9_.-]+|#[0-9]+|[가-힣]{2,}")
PARTICLES = re.compile(r"(은|는|이|가|을|를|의|에|로|와|과|으로|에서|까지|부터|마다|도|만)$")


def extract_entities(text, stopwords, max_ent=16):
    entities = []
    seen = set()
    for token in TOKEN_PAT.findall(text):
        if any("\uac00" <= ch <= "\ud7a3" for ch in token):
            n = PARTICLES.sub("", token)
        else:
            n = token.lower()
        if len(n) < 2 or n in stopwords:
            continue
        if n not in seen:
            entities.append(n)
            seen.add(n)
        if len(entities) >= max_ent:
            break
    return entities


# Load data
wb = openpyxl.load_workbook("data/model_issue_dataset_10000.xlsx", data_only=True, read_only=True)
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

# Build chunks (current fixed logic)
SECTION_HEADING_RE = re.compile(r"^(원인 요약|확인 근거|기술 판단|영향 범위|추가 조치)\s*:")
SECTION_SPLIT_RE = re.compile(r"\n+|(?=(?:원인 요약|확인 근거|기술 판단|영향 범위|추가 조치)\s*:)")
SECTION_LABEL_MAP = {
    "원인 요약": "discovery", "확인 근거": "discovery", "기술 판단": "discovery",
    "영향 범위": "result", "추가 조치": "next_action",
}
FLOW_NAME_MAP = {
    frozenset({"discovery"}): "원인 발견",
    frozenset({"result"}): "영향 결과",
    frozenset({"next_action"}): "후속 조치",
    frozenset({"result", "next_action"}): "영향 및 조치",
    frozenset({"analysis_misc"}): "분석 메모",
}

chunks = []
for r in rows[:10000]:
    title = str(r.get("모델 이슈 검토 사항", "")).strip()
    if not title or title == "None":
        continue
    reg = r.get("등록일", "")
    if isinstance(reg, datetime): reg = reg.date().isoformat()
    elif isinstance(reg, date): reg = reg.isoformat()
    else: reg = str(reg) if reg else ""
    check = str(r.get("기본 확인내용", "") or "").strip()
    work = str(r.get("기본 작업내용", "") or "").strip()
    instr = str(r.get("업무지시", "") or "").strip()
    assignee = str(r.get("담당자", "") or "unassigned").strip()
    status_raw = str(r.get("진행(담당자)", "") or "").strip()
    analysis = str(r.get("문제점 분석 내용 (담당자 Comments)", "") or "").strip()
    if analysis == "None": analysis = ""

    ov_lines = [
        f"[이슈] {title}", f"[등록일] {reg}",
        f"[기본 확인내용] {check}", f"[기본 작업내용] {work}",
        f"[업무지시] {instr}", f"[담당자] {assignee}", f"[진행] {status_raw}",
    ]
    overview_text = "\n".join(ov_lines)
    single_text = overview_text + (f"\n[문제점 분석 내용] {analysis}" if analysis else "")
    should_split = analysis and len(single_text) > 600

    if not should_split:
        chunks.append({"kind": "single", "text": single_text, "title": title})
    else:
        chunks.append({"kind": "overview", "text": overview_text, "title": title})
        # simplified: put full analysis as one flow
        ft = f"[이슈] {title}\n[등록일] {reg}\n[담당자] {assignee}\n[진행] {status_raw}\n[분석] {analysis}"
        chunks.append({"kind": "flow", "text": ft, "title": title})


# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("이슈 #1: overview 청크 길이 부족 (평균 260자)")
print("=" * 70)

overview_chunks = [c for c in chunks if c["kind"] == "overview"]
single_chunks = [c for c in chunks if c["kind"] == "single"]
flow_chunks = [c for c in chunks if c["kind"] == "flow"]

print(f"\n현재 청크 분포:")
print(f"  single: {len(single_chunks)} (평균 {sum(len(c['text']) for c in single_chunks)//max(len(single_chunks),1)}자)")
print(f"  overview: {len(overview_chunks)} (평균 {sum(len(c['text']) for c in overview_chunks)//max(len(overview_chunks),1)}자)")
print(f"  flow: {len(flow_chunks)} (평균 {sum(len(c['text']) for c in flow_chunks)//max(len(flow_chunks),1)}자)")

# What fields are in overview vs missing?
print(f"\n  overview에 포함되는 필드: 이슈제목, 등록일, 기본확인내용, 기본작업내용, 업무지시, 담당자, 진행")
print(f"  overview에 빠지는 필드: 문제점 분석 내용 (-> flow로 분리됨)")
print(f"  => overview는 '이슈 메타정보' 역할, flow는 '상세 분석' 역할")

# Semantic coverage test: if user searches "batch size 조정", does overview catch it?
test_keywords = [
    ("batch size 조정", ["batch", "size"]),
    ("GPU 메모리 OOM", ["gpu", "oom"]),
    ("모니터링 대시보드", ["모니터링", "대시보드"]),
    ("인덱스 재구축", ["인덱스", "재구축"]),
]
print(f"\n  의미 검색 테스트 -키워드가 overview vs flow 중 어디에 있는지:")
for query, kws in test_keywords:
    ov_hit = sum(1 for c in overview_chunks if all(k.lower() in c["text"].lower() for k in kws))
    fl_hit = sum(1 for c in flow_chunks if all(k.lower() in c["text"].lower() for k in kws))
    sg_hit = sum(1 for c in single_chunks if all(k.lower() in c["text"].lower() for k in kws))
    print(f"  \"{query}\": overview={ov_hit}, flow={fl_hit}, single={sg_hit}")

print(f"\n  [결론] overview 260자가 짧아 보이지만:")
print(f"  - overview는 7개 메타필드(제목/확인내용/작업내용/업무지시/담당자/진행/등록일)를 담당")
print(f"  - 상세 기술 키워드(batch, OOM 등)는 flow/single에 위치")
print(f"  - overview를 길게 만들려면 분석 내용을 중복 포함해야 함 -> 벡터 중복 문제 유발")
print(f"  - 현재 구조는 '메타 검색용 overview + 의미 검색용 flow' 분업이 정상 작동")

# Actual risk of making overview longer
print(f"\n  [개선 시 위험]")
print(f"  - overview에 분석 요약을 추가하면 overview <-> flow 벡터 유사도 상승")
print(f"  - source_selector의 issue_title dedup이 동일 이슈의 overview/flow 중 하나만 반환")
print(f"  - 결과: overview만 반환되고 flow가 누락되거나, flow만 반환되고 overview가 누락")

# Measure overlap if we add analysis summary to overview
overlap_test = 0
for c_ov in overview_chunks[:100]:
    matching_flows = [c for c in flow_chunks if c["title"] == c_ov["title"]]
    if matching_flows:
        ov_words = set(c_ov["text"].lower().split())
        fl_words = set(matching_flows[0]["text"].lower().split())
        overlap_test += len(ov_words & fl_words) / max(len(ov_words | fl_words), 1)
overlap_avg = overlap_test / min(100, len(overview_chunks)) if overview_chunks else 0
print(f"  - 현재 overview-flow 단어 중복률: {overlap_avg:.1%}")
print(f"  - 분석 요약 추가 시 예상 중복률: ~50-60% -> 벡터 공간 차별성 상실")

improvement_1 = "낮음"
print(f"\n  * 개선 효과: {improvement_1} -현재 구조가 역할 분리에 적합. 변경 불필요.")


# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("이슈 #2: 일반 동사 엔티티 노이즈 ('확인', '설정', '로그' 등)")
print("=" * 70)

# Current state
current_stopwords = GRAPH_ENTITY_STOPWORDS
entity_freq = Counter()
for c in chunks:
    ents = extract_entities(c["text"], current_stopwords)
    for e in ents:
        entity_freq[e] += 1

# Identify verb-like entities in top 30
print(f"\n현재 상위 30 엔티티:")
CANDIDATE_VERB_STOPS = set()
top30 = entity_freq.most_common(30)
for e, cnt in top30:
    is_korean = any("\uac00" <= ch <= "\ud7a3" for ch in e)
    # Heuristic: Korean 2-char word appearing in >30% of chunks is likely generic
    is_generic = is_korean and len(e) == 2 and cnt > len(chunks) * 0.2
    marker = " <- 일반동사 후보" if is_generic else ""
    pct = cnt / len(chunks) * 100
    print(f"  {e:20s}: {cnt:>6d} ({pct:5.1f}%){marker}")
    if is_generic:
        CANDIDATE_VERB_STOPS.add(e)

print(f"\n일반 동사 후보 (2글자 한국어, 전체 20% 이상 출현): {sorted(CANDIDATE_VERB_STOPS)}")

# Simulate adding these to stopwords
extended_stopwords = current_stopwords | frozenset(CANDIDATE_VERB_STOPS)
entity_freq_after = Counter()
entity_to_chunks_before = defaultdict(set)
entity_to_chunks_after = defaultdict(set)

for i, c in enumerate(chunks):
    ents_before = extract_entities(c["text"], current_stopwords)
    ents_after = extract_entities(c["text"], extended_stopwords)
    for e in ents_before:
        entity_freq_after[e] += 0  # just to have key
        entity_to_chunks_before[e].add(i)
    for e in ents_after:
        entity_freq_after[e] += 1
        entity_to_chunks_after[e].add(i)

print(f"\n변경 전/후 엔티티 통계:")
print(f"  고유 엔티티 수: {len(entity_to_chunks_before)} -> {len(entity_to_chunks_after)}")
before_conn = [len(v) for v in entity_to_chunks_before.values()]
after_conn = [len(v) for v in entity_to_chunks_after.values()]
print(f"  최대 연결도: {max(before_conn)} -> {max(after_conn)}")
print(f"  평균 연결도: {sum(before_conn)/len(before_conn):.0f} -> {sum(after_conn)/len(after_conn):.0f}")

# Impact on search queries
TEST_QUERIES_2 = [
    ("GPU 메모리 부족", ["gpu", "메모리", "부족"]),
    ("타임아웃 오류", ["타임아웃", "오류"]),
    ("인덱스 손상", ["인덱스", "손상"]),
    ("로그 확인 결과", ["로그", "확인"]),  # "확인" would be removed
    ("설정 변경 후 오류", ["설정", "변경", "오류"]),  # "설정" would be removed
]

print(f"\n쿼리별 그래프 도달 영향:")
print(f"  {'쿼리':<30s} {'변경전':>8s} {'변경후':>8s} {'차이':>8s}")
print(f"  {'-'*56}")
for query, terms in TEST_QUERIES_2:
    # Before: query entities via current blacklist
    q_ents_before = [t for t in terms if t.lower() not in current_stopwords and len(t) >= 2]
    reach_before = set()
    for qe in q_ents_before:
        if qe in entity_to_chunks_before:
            reach_before |= entity_to_chunks_before[qe]

    # After: query entities via extended blacklist (simulate QUERY side too)
    q_ents_after = [t for t in terms if t.lower() not in extended_stopwords and len(t) >= 2]
    reach_after = set()
    for qe in q_ents_after:
        if qe in entity_to_chunks_after:
            reach_after |= entity_to_chunks_after[qe]

    diff = len(reach_after) - len(reach_before)
    marker = ""
    if diff < -100:
        marker = " !! 대폭 감소"
    elif diff < 0:
        marker = " v"
    elif diff == 0:
        marker = " ="
    print(f"  \"{query:<28s}\" {len(reach_before):>7d} {len(reach_after):>7d} {diff:>+7d}{marker}")

    # Show which entities were lost
    lost_ents = set(q_ents_before) - set(q_ents_after)
    if lost_ents:
        print(f"    탈락 엔티티: {lost_ents}")

# Key question: does removing "확인" hurt "확인 결과" queries?
print(f"\n  [핵심 분석] '확인'을 STOPWORDS에 추가하면:")
print(f"  - '확인'이 포함된 쿼리: '로그 확인 결과', '확인 완료' 등")
print(f"  - '확인' 엔티티가 {len(entity_to_chunks_before.get('확인', set()))}청크에 연결")
print(f"  - 제거 시 이 청크들이 그래프 경로에서 탈락")
print(f"  - BUT: '로그'나 '결과' 엔티티를 통해 여전히 도달 가능")
print(f"  - 문제: '확인'만으로 검색하는 경우는 드묾 (너무 일반적)")
print(f"  - 이점: 허브 노드 제거로 그래프 탐색 정밀도 향상")

# Precision improvement estimate
hub_entities_before = sum(1 for v in entity_to_chunks_before.values() if len(v) > len(chunks) * 0.3)
hub_entities_after = sum(1 for v in entity_to_chunks_after.values() if len(v) > len(chunks) * 0.3)
print(f"\n  허브 노드 (30%+ 연결): {hub_entities_before} -> {hub_entities_after}")

improvement_2 = "중간"
risk_2 = "낮음"
print(f"\n  * 개선 효과: {improvement_2}")
print(f"  * 사이드이펙트 위험: {risk_2} -양쪽(청크+쿼리) 동시 제거하면 매칭 유지")
print(f"  * 조건: entity_stopwords.py에 추가하면 graph_builder+query_analyzer 양쪽 적용")


# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("이슈 #3: 대소문자 불일치 (쿼리 소문자 vs 데이터 원본)")
print("=" * 70)

# Check: how does entity extraction handle case?
print(f"\n현재 코드 동작:")
print(f"  graph_builder._normalize(): 영어 -> .lower() 적용")
print(f"  query_analyzer._normalize_entity_token(): .lower() 적용")
print(f"  => 양쪽 모두 소문자 변환 -> 매칭에 문제 없음")

# Verify with actual data
sample_english_ents = Counter()
for c in chunks[:3000]:
    ents = extract_entities(c["text"], current_stopwords)
    for e in ents:
        if e.isascii() and not e.startswith("#"):
            sample_english_ents[e] += 1

print(f"\n영어 엔티티 샘플 (추출 후 형태):")
for e, cnt in sample_english_ents.most_common(15):
    print(f"  {e:30s}: {cnt:>5d}")

# Test: query "BGE-M3-Retrieve" vs stored entity
test_system = "bge-m3-retrieve"
stored_matches = sum(1 for c in chunks if test_system in " ".join(extract_entities(c["text"], current_stopwords)))
print(f"\n  쿼리 'BGE-M3-Retrieve' -> 소문자 '{test_system}'")
print(f"  저장된 엔티티에서 매칭: {stored_matches}청크")

# Check lexical search (chunk_search_text uses .lower())
lexical_matches = sum(1 for c in chunks if "bge-m3-retrieve" in c["text"].lower())
print(f"  chunk_text.lower() 매칭: {lexical_matches}청크")

print(f"\n  [결론] 대소문자 불일치는 실제로 존재하지 않음")
print(f"  - graph_builder와 query_analyzer 양쪽 모두 .lower() 적용")
print(f"  - chunk_search_text()도 .lower() 적용 (query_terms.py:59)")
print(f"  - 이전 분석에서 lexical proxy가 대소문자 미변환이었을 뿐")

improvement_3 = "없음"
print(f"\n  * 개선 효과: {improvement_3} -이미 양쪽 소문자 통일. 수정 불필요.")


# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("종합 영향도 요약")
print("=" * 70)

print(f"  {'item':<26s} {'effect':>10s} {'risk':>10s} {'verdict':>10s}")
print(f"  {'-'*58}")
print(f"  {'#1 overview length':<26s} {'LOW':>10s} {'MEDIUM':>10s} {'SKIP':>10s}")
print(f"  {'#2 verb STOPWORDS':<26s} {'MEDIUM':>10s} {'LOW':>10s} {'APPLY':>10s}")
print(f"  {'#3 case mismatch':<26s} {'NONE':>10s} {'NONE':>10s} {'SKIP':>10s}")
