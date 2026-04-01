# CLAUDE.md - Simple-Graph-RAG Project Rules

## Architecture Overview

- **Stack**: FastAPI + PostgreSQL(pgvector) + Neo4j + BGE-M3 + KSS
- **Data**: 10,000 issue records → ~29,762 chunks (overview + analysis flow)
- **Search**: Hybrid vector(pgvector cosine) + graph(Neo4j entity) + metadata filter

## Critical Rules (Past Mistakes — Do NOT Repeat)

### 1. NEVER use `search_text` for query embedding

- **File**: `retrieval.py:365`, `strategies/mixed_issue_chat.py:51`
- **Rule**: Always use `analysis.clean_question` for BGE-M3 embedding
- **Why**: `search_text` is kiwi-tokenized keywords. kiwi splits compound terms:
  - `"타임아웃"` → `"타임" + "아웃"` (meaning destroyed)
  - `"BGE-M3-Retrieve"` → `"BGE" + "M" + "3" + "Retrieve"` (system name destroyed)
- **Result**: Vector search returns irrelevant results because the query vector occupies a different embedding space than the stored chunk vectors
- `search_text` is still used for entity matching / lexical scoring — that's fine

### 2. NEVER create HNSW index BEFORE bulk insert

- **Rule**: Insert all data first, then CREATE INDEX
- **Why**: HNSW index rebuilds on every INSERT. 29,762 rows with index = 13+ hours stuck. Without index = 74 seconds + 22 seconds for index creation = 97 seconds total
- **Pattern**:
  ```sql
  -- 1. Insert all chunks (no HNSW index)
  INSERT INTO chunk_embeddings ...
  -- 2. Create HNSW index AFTER data
  CREATE INDEX ce_hnsw_idx ON chunk_embeddings USING hnsw (embedding vector_cosine_ops);
  ```

### 3. NEVER raise `excel_row_max_chars` to eliminate splitting

- **File**: `config.py:38` — currently `600`
- **Rule**: Keep at 600. Do NOT raise to 800+ to avoid splitting
- **Why**: The analysis text (문제점 분석 내용) contains 5 semantic sections (원인/근거/판단/영향/조치). KSS-based splitting via BehaviorLabeler creates separate flow chunks for fine-grained semantic search. If everything becomes a single chunk, searching for "원인 분석" returns the entire 600-char block instead of the precise cause section
- **Correct split**: overview (metadata) + flow 1 (원인 발견) + flow 2 (영향 및 조치) = 3 chunks per issue

### 4. Entity stopwords MUST be synchronized between graph_builder and query_analyzer

- **File**: `app/services/entity_stopwords.py` (shared module)
- **Rule**: Add/remove stopwords in `entity_stopwords.py` ONLY — never directly in `graph_builder.py` or `query_analyzer.py`
- **Why**: If a word is filtered from chunk entities but not from query entities (or vice versa), graph-based entity matching breaks completely — one side has the node, the other doesn't
- **Currently filtered**: structural tags (등록일, 담당자, 진행...) + generic verbs (설정, 로그, 결과, 패치)
- **Intentionally NOT filtered**: `"확인"` — removing it zeros out graph reach for "확인 결과" queries

### 5. Status values MUST be mapped in TWO places

- **Files**: `korean_nlp.py:_STATUS_MAP` AND `query_analyzer.py:_STATUS_KEYWORDS`
- **Rule**: When adding a new status value, update BOTH
- **Why**: `_STATUS_MAP` normalizes the value, `_STATUS_KEYWORDS` tells the query analyzer to recognize it as a status. If only one is updated, the status filter silently fails
- **Current statuses**: 완료, 진행중, 검증중, 분석중, 대기

### 6. BehaviorLabeler merge groups must NOT cross semantic boundaries

- **File**: `behavior_labeler.py:_FLOW_NAME_MAP`
- **Rule**: Do NOT add `frozenset({"discovery", "result", "next_action"})` — this merges ALL 5 sections into 1 chunk, destroying semantic search granularity
- **Correct groups**:
  - Cause group: `discovery` (원인 요약 + 확인 근거 + 기술 판단)
  - Impact group: `result` + `next_action` (영향 범위 + 추가 조치)
- These two groups must stay as **separate chunks**

### 7. CPU-only embedding takes ~15 hours for 29k chunks

- **Config**: `embedding_device: str = "cpu"` in config.py
- On Mac M4: change to `"mps"` → ~12-15 minutes
- On NVIDIA GPU: change to `"cuda"` → ~5-6 minutes
- `embedding_cache` (PostgreSQL table) caches results — second run is seconds
- Always check device before running full ingest

## Current System Status (2026-04-01)

### Data
- **PostgreSQL**: 29,762 chunks (overview 10,000 + analysis_flow 19,762), 임베딩 100% 완료
- **Neo4j**: 40,777 노드 (Chunk 29,762 + Issue 10,000 + Entity 552 + Date 444 + User 12 + Status 5 + Document 1 + Channel 1), 714,228 엣지
- **임베딩 캐시**: 29,762건 (재인제스트 시 즉시)

### Multi-LLM Routing
| 모델 | 라우팅 | 포트 |
|------|--------|------|
| Qwen3.5 9B (Local) | Jan API → `openai_chat` (/chat/completions) | 1337 |
| GPT-5.4 | Claude CLI Proxy → `legacy` (/generate) | 8800 |
| Claude Sonnet 4.6 | Claude CLI Proxy → `legacy` (/generate) | 8800 |
| Claude Haiku 4.5 | Claude CLI Proxy → `legacy` (/generate) | 8800 |

- `_LOCAL_MODELS` set in `codex_proxy.py` → Jan API, 나머지 → Claude Proxy
- `.env` 기준: `CODEX_PROXY_BASE_URL=http://0.0.0.0:1337`, `CLAUDE_PROXY_BASE_URL=http://127.0.0.1:8800`
- Qwen3 thinking 토큰으로 인해 `max_tokens=4096`, `CODEX_TIMEOUT_SECONDS=300` 필요

### Infrastructure
| 서비스 | 이미지 | 포트 | 메모리 |
|--------|--------|------|--------|
| PostgreSQL | pgvector/pgvector:pg16 | 5432 | default |
| Neo4j | neo4j:5.22 | 8747(HTTP)/8768(bolt) | heap 512MB, pagecache 128MB, tx 256MB |

- Neo4j 메모리: 기존 128MB → 512MB로 증설 (29K 청크 그래프 백필 시 OOM 방지)
- 그래프 백필: 500건/배치, 60배치 전량 적재 완료

### Verified Test Results (25 queries)
- **PASS**: 22/25 (88%)
- **TIMEOUT**: 3/25 — count/aggregate 쿼리, 테스트 클라이언트 60초 제한 (서버 300초 정상)
- **FAIL**: 0/25
- **평균 응답**: 19.4초 (Claude Sonnet 기준)
- **그래프 기여**: 12/22건에서 Graph-Only 청크 발견 (벡터 검색 보완)

### Debug Panels (UI)
5개 디버그 패널 정상 동작 (디버그 토글 ⚙️ 활성화 필요):
1. **Pipeline Waterfall** — 단계별 시간 차트 (bottleneck 빨간색)
2. **5-Factor Ranking** — 레이더 차트 (가중치 점선 + 소스별 실제 점수 자동 오버레이)
3. **Source Comparison** — Vector / Graph Entity / Multihop 비교 + Graph-Only 발견 수
4. **Graph Traversal** — D3 force-directed 그래프 (노드/엣지 시각화)
5. **Entity Map** — 엔티티 유형별 빈도 테이블

## Project Structure (Key Files)

```
app/
  config.py                    — Settings (excel_row_max_chars=600, embedding_device=mps, etc.)
  container.py                 — DI container (ServiceContainer)
  schemas.py                   — Pydantic models (363 lines)
  services/
    entity_stopwords.py        — SHARED stopwords (graph + query synchronized)
    behavior_labeler.py        — KSS sentence split + flow merge (_FLOW_NAME_MAP)
    issue_chunking.py          — Issue row → overview + analysis flow chunks
    retrieval.py:365           — Query embedding (MUST use clean_question)
    query_analyzer.py          — Query parse + entity extract + status filter
    korean_nlp.py              — Status normalization map
    ranking_policy.py          — Score weights (vector/graph/entity/metadata/recency)
    graph_builder.py           — Entity extraction (uses GRAPH_ENTITY_STOPWORDS)
    query_router.py            — Route: standard / count / mixed_issue_chat
    source_selector.py         — Dedup + top-k selection
    strategies/
      count_query.py           — Count/aggregate query strategy
      mixed_issue_chat.py      — Dual-lane issue+chat strategy
  adapters/
    codex_proxy.py             — Multi-LLM routing (Local→Jan, Others→Claude Proxy)
    embedding_provider.py      — BGE-M3 encode (normalize_embeddings=True, device=mps)
    postgres_vector_store.py   — pgvector HNSW cosine search (810 lines)
    neo4j_store.py             — Graph upsert/expand/entity seed (423 lines)
    embedding_cache_store.py   — SHA256 key cache for embeddings
  static/
    index.html                 — SPA UI (2,140 lines, debug panels, model selector)
scripts/
  claude_proxy.py              — Stream-JSON process pool for Claude CLI
  ingest_file.py               — CLI ingest (--skip-graph for PG-only)
  generate_unique_dataset.py   — 10,000 unique issue generator (seed=42)
  backfill_graph.py            — Rebuild Neo4j graph from PG chunks
data/
  model_issue_dataset_10000.xlsx — 10,000 unique issues, 5-section analysis
.env                           — EMBEDDING_DEVICE=mps, CODEX_PROXY_BASE_URL, CLAUDE_PROXY_BASE_URL
```

## Docker

```bash
# Start infrastructure (PostgreSQL + Neo4j)
docker run -d --name simple-graph-rag-neo4j \
  -e NEO4J_AUTH=neo4j/graph-rag-password \
  -e 'NEO4J_PLUGINS=["graph-data-science"]' \
  -e NEO4J_server_memory_heap_initial__size=512m \
  -e NEO4J_server_memory_heap_max__size=512m \
  -e NEO4J_server_memory_pagecache_size=128m \
  -e NEO4J_dbms_memory_transaction_total_max=256m \
  -p 8747:7474 -p 8768:7687 \
  -v simple-graph-rag_neo4j_data:/data \
  neo4j:5.22
```

## Run Commands (Mac)

```bash
# 1. LLM Proxy (Claude CLI)
python3 scripts/claude_proxy.py --port 8800 --model sonnet --workers 2

# 2. App Server
EMBEDDING_DEVICE=mps uvicorn app.main:app --host 0.0.0.0 --port 8000

# 3. Ingest (PG + Neo4j graph backfill)
EMBEDDING_DEVICE=mps python scripts/ingest_file.py data/model_issue_dataset_10000.xlsx --replace-filename --skip-graph
# Then backfill graph separately (500 rows/batch) due to Neo4j memory

# 4. Jan AI (Local LLM) — started from Jan app, port 1337
```

## RAG 품질 평가 (GAS/AAR)

### 평가 구조
- **Layer 1 (Retrieval)**: Recall@k, MRR, nDCG — "검색이 잘 됐냐" → ✅ 구현 완료 (5-factor ranking)
- **Layer 2 (GAS)**: Grounded Answer Score — "검색된 근거를 기반으로 맞게 답했냐" → ✅ 프롬프트 기반 동작 확인
- **Layer 3 (AAR)**: Abstain Accuracy Rate — "모르면 멈췄냐" → ✅ 프롬프트 기반 동작 확인

### 설계 결정 (2026-04-01 검증 완료)

| 방안 | 결정 | 근거 |
|------|------|------|
| 벡터 유사도 threshold | **❌ 미적용** | 무관한 질문도 cosine 0.36~0.44로 통과, 정상 질문(0.43~0.47)과 겹쳐 hard threshold가 정상 검색 파괴 위험. LLM이 이미 AAR 수행 중 |
| 프로그래밍적 근거 부족 판단 | **❌ 미적용** | P999/김영희/2030년 테스트에서 LLM이 "정보가 없다"고 정직하게 답변. 프롬프트 한 줄이 충분히 동작 |
| Golden Q&A 평가 셋 | **✅ 적용** | 기존 25건 테스트에 기대 키워드 추가, 자동 GAS/AAR 회귀 방지 |

### AAR 동작 확인 결과
```
Q: "프로젝트 P999 상태?"     → "P999 정보가 포함되어 있지 않습니다" ✅
Q: "김영희가 작성한 보고서?"  → "김영희라는 사용자는 확인되지 않습니다" ✅
Q: "2030년 이슈 알려줘"      → "2030년 이슈가 존재하지 않습니다" ✅
```
→ 시스템 프롬프트 `"근거가 부족하면 단정하지 말고 제한사항을 짧게 밝혀라"`가 AAR 역할 수행

### 평가 실행
```bash
EMBEDDING_DEVICE=mps python scripts/evaluate_golden_qa.py          # 전체 실행
EMBEDDING_DEVICE=mps python scripts/evaluate_golden_qa.py --tag aar # AAR만
```

## Known Issues

- **Count/aggregate 쿼리 느림**: count 라우트에서 DB 집계 후 LLM 요약까지 60초+ 소요 가능
- **Qwen3 로컬 모델 응답 시간**: thinking 토큰 포함 1~3분 소요 (Claude Sonnet ~15초 대비)
- **Neo4j 그래프 백필**: 인제스트 시 `--skip-graph`로 PG 먼저 적재 후 별도 배치 스크립트로 그래프 적재 필요 (한번에 29K 그래프 쓰기 시 메모리 부족 가능)
- **담당자 필터 시 엔티티 미추출**: QueryAnalyzer가 이름을 assignee 필터로 추출하면 clean_question에서 제거되어 graph entity seed에 0 반환 (정상 동작 — 메타데이터 필터로 정확히 검색됨)
