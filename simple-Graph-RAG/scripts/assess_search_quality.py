"""
GraphRAG semantic search quality assessment.
Simulates the full pipeline: chunking → entity extraction → graph structure
→ query analysis → vector/graph retrieval scoring.
"""
import sys
sys.path.insert(0, "d:/apppart_projects/Bug-Chat-RAG/simple-Graph-RAG")

import openpyxl
import re
from datetime import datetime, date
from collections import Counter, defaultdict
from difflib import SequenceMatcher

from app.services.entity_stopwords import GRAPH_ENTITY_STOPWORDS, QUERY_ENTITY_BLACKLIST

# ══════════════════════════════════════════════════════════════
# 1. Load data & simulate FIXED chunking
# ══════════════════════════════════════════════════════════════

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
LABEL_PATTERNS = [
    ("discovery", ("확인", "발견", "로그", "보니", "파악", "조사", "분석", "살펴", "검토")),
    ("result", ("정상", "해결", "복구", "개선", "안정", "완료", "효과")),
    ("next_action", ("예정", "추후", "향후", "계획", "해야", "필요합니다", "권장")),
]

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


def split_structured(text):
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


def label_sentence(sentence):
    heading = SECTION_HEADING_RE.match(sentence)
    if heading:
        mapped = SECTION_LABEL_MAP.get(heading.group(1))
        if mapped:
            return [mapped]
    labels = [la for la, kws in LABEL_PATTERNS if any(k in sentence for k in kws)]
    return labels or ["analysis_misc"]


def merge_adjacent(labeled):
    merged = []
    buf_s, buf_l = [], set()
    def flush():
        if not buf_s:
            return
        labels = sorted(buf_l) or ["analysis_misc"]
        fn = FLOW_NAME_MAP.get(frozenset(labels), "분석 메모")
        merged.append({"flow_name": fn, "text": " ".join(buf_s).strip()})
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


# Load
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

# Build chunks
EXCEL_ROW_MAX_CHARS = 600
chunks = []  # list of dicts: {kind, text, title, entities, assignee, status, ...}

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

    should_split = analysis and len(single_text) > EXCEL_ROW_MAX_CHARS

    if not should_split:
        ents = extract_entities(single_text, GRAPH_ENTITY_STOPWORDS)
        chunks.append({
            "kind": "single", "text": single_text, "title": title,
            "entities": ents, "assignee": assignee, "status_raw": status_raw,
        })
    else:
        ov_ents = extract_entities(overview_text, GRAPH_ENTITY_STOPWORDS)
        chunks.append({
            "kind": "overview", "text": overview_text, "title": title,
            "entities": ov_ents, "assignee": assignee, "status_raw": status_raw,
        })
        sections = split_structured(analysis)
        if sections:
            labeled = [(s, label_sentence(s)) for s in sections]
            flows = merge_adjacent(labeled)
        else:
            flows = [{"flow_name": "분석 메모", "text": analysis}]
        for flow in flows:
            ft = f"[이슈] {title}\n[등록일] {reg}\n[담당자] {assignee}\n[진행] {status_raw}\n[{flow['flow_name']}] {flow['text']}"
            fl_ents = extract_entities(ft, GRAPH_ENTITY_STOPWORDS)
            chunks.append({
                "kind": "flow", "text": ft, "title": title,
                "entities": fl_ents, "assignee": assignee, "status_raw": status_raw,
                "flow_name": flow["flow_name"],
            })


# ══════════════════════════════════════════════════════════════
# 2. Graph structure analysis
# ══════════════════════════════════════════════════════════════

print("=" * 70)
print("A. 그래프 구조 분석")
print("=" * 70)

entity_to_chunks = defaultdict(set)
chunk_entity_counts = []
all_entities_flat = Counter()

for i, c in enumerate(chunks):
    ents = c["entities"]
    chunk_entity_counts.append(len(ents))
    for e in ents:
        entity_to_chunks[e].add(i)
        all_entities_flat[e] += 1

print(f"\n총 청크 수: {len(chunks)}")
print(f"고유 엔티티 노드 수: {len(entity_to_chunks)}")
print(f"청크당 엔티티 수: min={min(chunk_entity_counts)}, max={max(chunk_entity_counts)}, avg={sum(chunk_entity_counts)/len(chunk_entity_counts):.1f}")

# Entity connectivity
connectivity = [len(cids) for cids in entity_to_chunks.values()]
print(f"\n엔티티 연결도 (엔티티당 연결된 청크 수):")
print(f"  min={min(connectivity)}, max={max(connectivity)}, avg={sum(connectivity)/len(connectivity):.1f}")
high_conn = sum(1 for c in connectivity if c > 100)
mid_conn = sum(1 for c in connectivity if 10 <= c <= 100)
low_conn = sum(1 for c in connectivity if c < 10)
print(f"  고연결(>100): {high_conn}, 중연결(10~100): {mid_conn}, 저연결(<10): {low_conn}")

print(f"\n상위 20 엔티티 (연결된 청크 수):")
for e, cnt in all_entities_flat.most_common(20):
    print(f"  {e:25s}: {cnt:>5d} 청크")

# Entity co-occurrence (graph edges)
cooccurrence = Counter()
for i, c in enumerate(chunks[:5000]):
    ents = c["entities"][:10]
    for a_idx in range(len(ents)):
        for b_idx in range(a_idx + 1, len(ents)):
            pair = tuple(sorted([ents[a_idx], ents[b_idx]]))
            cooccurrence[pair] += 1

print(f"\n상위 10 엔티티 공동출현 (그래프 엣지 강도):")
for (a, b), cnt in cooccurrence.most_common(10):
    print(f"  {a} <-> {b}: {cnt}회")


# ══════════════════════════════════════════════════════════════
# 3. Query simulation — trace full retrieval path
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("B. 쿼리 시뮬레이션 (의미 검색 품질 측정)")
print("=" * 70)

# Simulated queries covering different intents
TEST_QUERIES = [
    ("GPU 메모리 부족 원인", "search", ["gpu", "메모리", "부족", "원인"]),
    ("타임아웃 오류 해결 방법", "search", ["타임아웃", "오류", "해결"]),
    ("Sujin 담당 이슈", "list", ["sujin"]),
    ("인덱스 손상 관련 이슈", "search", ["인덱스", "손상"]),
    ("커넥션 풀 고갈 영향 범위", "search", ["커넥션", "고갈", "영향"]),
    ("최근 완료된 이슈 요약", "summary", ["완료"]),
    ("BGE-M3-Retrieve 장애 분석", "search", ["bge-m3-retrieve", "장애"]),
    ("한국어 토크나이징 오류", "search", ["한국어", "토크나이징", "오류"]),
    ("캐시 정합성 문제 후속 조치", "search", ["캐시", "정합성", "후속"]),
    ("Reranker-v3 성능 저하", "search", ["reranker-v3", "성능", "저하"]),
]

QUERY_BLACKLIST_SET = set(QUERY_ENTITY_BLACKLIST)

def extract_query_entities(question):
    """Simulate query_analyzer._extract_entities"""
    tokens = re.findall(r"[A-Za-z0-9#._-]+|[가-힣]{2,}", question)
    entities = []
    for t in tokens:
        n = t.strip().lower()
        if not n or n in QUERY_BLACKLIST_SET or len(n) < 2:
            continue
        if n not in entities:
            entities.append(n)
    return entities[:8]


def lexical_score(query_terms, chunk_text):
    """Simulate entity_overlap_score via lexical matching"""
    text_lower = chunk_text.lower()
    if not query_terms:
        return 0.0
    matched = sum(1 for t in query_terms if t in text_lower)
    return matched / max(len(query_terms), 1)


def graph_entity_score(query_entities, chunk_entities):
    """Simulate graph-based entity overlap"""
    if not query_entities or not chunk_entities:
        return 0.0
    q_set = set(e.lower() for e in query_entities)
    c_set = set(e.lower() for e in chunk_entities)
    overlap = len(q_set & c_set)
    return min(1.0, overlap / max(len(q_set), 1))


for query, intent, expected_terms in TEST_QUERIES:
    print(f"\n--- 쿼리: \"{query}\" (intent={intent}) ---")

    query_entities = extract_query_entities(query)
    print(f"  추출된 쿼리 엔티티: {query_entities}")

    # 1) Vector search simulation (lexical proxy)
    vector_scores = []
    for i, c in enumerate(chunks):
        score = lexical_score(expected_terms, c["text"])
        vector_scores.append((i, score))
    vector_scores.sort(key=lambda x: -x[1])

    top_vector = vector_scores[:5]
    print(f"  [벡터 검색] top-5 (lexical proxy):")
    for idx, score in top_vector:
        c = chunks[idx]
        print(f"    score={score:.2f} kind={c['kind']:8s} title=\"{c['title'][:40]}\"")

    # 2) Graph entity search simulation
    graph_candidates = set()
    for qe in query_entities:
        if qe in entity_to_chunks:
            graph_candidates |= entity_to_chunks[qe]

    graph_scored = []
    for idx in graph_candidates:
        c = chunks[idx]
        gs = graph_entity_score(query_entities, c["entities"])
        graph_scored.append((idx, gs))
    graph_scored.sort(key=lambda x: -x[1])

    top_graph = graph_scored[:5]
    print(f"  [그래프 검색] entity match {len(graph_candidates)}건, top-5:")
    for idx, score in top_graph:
        c = chunks[idx]
        print(f"    score={score:.2f} kind={c['kind']:8s} title=\"{c['title'][:40]}\"")

    # 3) Combined score (simulating ranking_policy weights)
    # search: vector=0.40, graph=0.15, entity=0.20, metadata=0.15, recency=0.10
    all_candidates = set(i for i, _ in top_vector[:10]) | set(i for i, _ in top_graph[:10])
    combined = []
    for idx in all_candidates:
        c = chunks[idx]
        v_score = lexical_score(expected_terms, c["text"])
        g_score = graph_entity_score(query_entities, c["entities"])
        e_score = g_score  # entity overlap
        final = 0.40 * v_score + 0.15 * min(1.0, g_score) + 0.20 * e_score + 0.15 * 0.5
        combined.append((idx, final, v_score, g_score))
    combined.sort(key=lambda x: -x[1])

    print(f"  [최종 랭킹] 후보 {len(all_candidates)}건, top-3:")
    for idx, final, vs, gs in combined[:3]:
        c = chunks[idx]
        print(f"    final={final:.3f} (vec={vs:.2f} graph={gs:.2f}) kind={c['kind']:8s}")
        print(f"      title=\"{c['title'][:50]}\"")
        shared_ent = set(e.lower() for e in query_entities) & set(e.lower() for e in c["entities"])
        print(f"      공유 엔티티: {shared_ent or '없음'}")


# ══════════════════════════════════════════════════════════════
# 4. Semantic search quality metrics
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("C. 의미 검색 품질 지표")
print("=" * 70)

# 1) Entity coverage: what % of chunk text has meaningful entities?
ent_coverage = sum(1 for c in chunks if len(c["entities"]) >= 3) / len(chunks) * 100
print(f"\n의미 있는 엔티티 3개 이상 보유 청크: {ent_coverage:.1f}%")

# 2) Entity distinctiveness: how many unique entities per chunk on average?
avg_ents = sum(len(c["entities"]) for c in chunks) / len(chunks)
print(f"청크당 평균 엔티티 수: {avg_ents:.1f}")

# 3) Graph connectivity: avg path length between related chunks
# (via shared entities)
shared_entity_pairs = 0
total_pairs_checked = 0
for i in range(0, min(len(chunks), 1000), 2):
    for j in range(i+1, min(i+10, len(chunks))):
        e_i = set(chunks[i]["entities"])
        e_j = set(chunks[j]["entities"])
        if e_i & e_j:
            shared_entity_pairs += 1
        total_pairs_checked += 1
connectivity_rate = shared_entity_pairs / max(total_pairs_checked, 1) * 100
print(f"인접 청크 간 엔티티 공유율: {connectivity_rate:.1f}%")

# 4) Flow chunk vs overview: do they have different entity profiles?
flow_ents = Counter()
overview_ents = Counter()
single_ents = Counter()
for c in chunks:
    for e in c["entities"]:
        if c["kind"] == "flow":
            flow_ents[e] += 1
        elif c["kind"] == "overview":
            overview_ents[e] += 1
        else:
            single_ents[e] += 1

print(f"\n엔티티 분포 차이 (overview vs flow):")
print(f"  overview 고유 엔티티 수: {len(overview_ents)}")
print(f"  flow 고유 엔티티 수: {len(flow_ents)}")
print(f"  single 고유 엔티티 수: {len(single_ents)}")
overlap_ent = set(overview_ents) & set(flow_ents)
only_flow = set(flow_ents) - set(overview_ents)
print(f"  flow에만 있는 엔티티: {len(only_flow)}종")
if only_flow:
    print(f"  예시: {list(only_flow)[:10]}")

# 5) Query-to-chunk reachability via graph
print(f"\n쿼리별 그래프 도달률:")
for query, intent, expected_terms in TEST_QUERIES:
    q_ents = extract_query_entities(query)
    reachable = set()
    for qe in q_ents:
        if qe in entity_to_chunks:
            reachable |= entity_to_chunks[qe]
    pct = len(reachable) / len(chunks) * 100
    print(f"  \"{query[:30]}\" → 엔티티 {len(q_ents)}개 → 도달 청크 {len(reachable)}건 ({pct:.1f}%)")


# ══════════════════════════════════════════════════════════════
# 5. Overall assessment
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("D. GraphRAG 의미 검색 레벨 종합 평가")
print("=" * 70)

scores = {}

# Criterion 1: Chunk quality (length, diversity)
avg_len = sum(len(c["text"]) for c in chunks) / len(chunks)
chunk_quality = min(1.0, avg_len / 600)  # 600 chars = ideal
scores["청크 품질"] = chunk_quality

# Criterion 2: Entity extraction quality
noise_ratio = sum(1 for e in all_entities_flat if e in GRAPH_ENTITY_STOPWORDS) / max(len(all_entities_flat), 1)
scores["엔티티 품질"] = 1.0 - noise_ratio

# Criterion 3: Graph connectivity
scores["그래프 연결성"] = min(1.0, connectivity_rate / 50)  # 50% = perfect

# Criterion 4: Query reachability
avg_reach = 0
for query, intent, expected in TEST_QUERIES:
    q_ents = extract_query_entities(query)
    reachable = set()
    for qe in q_ents:
        if qe in entity_to_chunks:
            reachable |= entity_to_chunks[qe]
    avg_reach += len(reachable) / len(chunks) if reachable else 0
avg_reach /= len(TEST_QUERIES)
scores["쿼리 도달률"] = min(1.0, avg_reach * 10)  # 10% coverage = good

# Criterion 5: Chunk-kind diversity for split issues
kind_dist = Counter(c["kind"] for c in chunks)
has_flow = kind_dist.get("flow", 0) > 0
scores["의미 분할 유지"] = 1.0 if has_flow else 0.0

print()
total = 0
for name, score in scores.items():
    level = "■" * int(score * 10) + "□" * (10 - int(score * 10))
    print(f"  {name:<18s} [{level}] {score:.0%}")
    total += score

overall = total / len(scores)
print(f"\n  {'종합 점수':<18s}              {overall:.0%}")

if overall >= 0.8:
    grade = "A (우수)"
elif overall >= 0.6:
    grade = "B (양호)"
elif overall >= 0.4:
    grade = "C (보통)"
else:
    grade = "D (미흡)"
print(f"  {'등급':<18s}              {grade}")

print(f"\n{'='*70}")
print("E. 개선이 필요한 영역")
print("=" * 70)
for name, score in sorted(scores.items(), key=lambda x: x[1]):
    if score < 0.7:
        print(f"  [{score:.0%}] {name}")
