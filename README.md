# Graph-RAG MVP

> **한국어 조직 데이터를 위한 Graph-Augmented Retrieval 시스템**
>
> 채팅 로그, 이슈 워크북 등 한국어 조직 데이터를 그래프 기반으로 검색·분석하는 두 가지 RAG 시스템의 모노레포

---

## Overview

이 레포지토리는 동일한 한국어 채팅 로그 데이터를 서로 다른 아키텍처로 처리하는 **두 개의 독립 프로젝트**를 포함합니다.

| | [Simple Graph-RAG](./simple-Graph-RAG/) | [Hybrid Graph-RAG](./hybrid-Graph-RAG/) |
|---|---|---|
| **핵심 전략** | 4-Factor 가중 랭킹 (Vector + Graph + Metadata + Recency) | RRF 퓨전 (Vector + Fulltext) + Graph Context Expansion |
| **데이터 타입** | 채팅 로그 + 이슈 워크북 (`.txt` + `.xlsx`) | 채팅 로그 (`.txt`) |
| **Vector Store** | PostgreSQL + pgvector (HNSW) | Neo4j Vector Index |
| **Graph DB** | Neo4j (Entity Relationship Graph) | Neo4j (Conversation Flow Graph) |
| **LLM 통합** | Claude CLI Stream-JSON Proxy → 답변 생성 | 없음 (검색 + 인사이트 특화) |
| **청킹** | KSS 한국어 문장 분리 + 시간 기반 병합 (256 토큰) | 메시지 단위 (청킹 없음) |
| **UI** | SPA 채팅 인터페이스 | Jinja2 SSR 검색/인사이트 페이지 |

---

## Architecture Comparison

### Simple Graph-RAG — 지식 추출 + 답변 생성

```
┌─────────────────────────────────────────────────────────────────────┐
│  Ingestion                                                          │
│                                                                     │
│  .txt / .xlsx ─→ KSS 문장분리 ─→ 시간기반 병합 ─→ BGE-M3 임베딩   │
│                                      │                              │
│                          ┌───────────┴───────────┐                  │
│                          ▼                       ▼                  │
│                   PostgreSQL/pgvector        Neo4j                  │
│                   (chunk + vector)      (Entity Graph)              │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  Retrieval                                                          │
│                                                                     │
│  Query ─→ QueryAnalyzer ─→ Intent/Filter/Entity 추출               │
│                │                                                    │
│       ┌───────┴───────┐                                             │
│       ▼               ▼                                             │
│  Vector Search   Graph Expansion                                    │
│  (pgvector)     (Neo4j BFS)                                        │
│       │               │                                             │
│       └───────┬───────┘                                             │
│               ▼                                                     │
│  4-Factor Ranking (0.45 vector + 0.25 graph + 0.20 meta + 0.10 recency) │
│               ▼                                                     │
│  Claude LLM ─→ 답변 생성 (source attribution)                      │
└─────────────────────────────────────────────────────────────────────┘
```

**그래프 스키마**: `Document`, `Chunk`, `User`, `Channel`, `Date`, `Entity`, `Issue`, `Status`, `Community`
**엣지 타입**: `PART_OF`, `SENT_BY`, `IN_CHANNEL`, `ON_DATE`, `MENTIONS`, `CO_OCCURS`, `NEXT`

### Hybrid Graph-RAG — 운영 검색 + 인사이트

```
┌─────────────────────────────────────────────────────────────────────┐
│  Ingestion                                                          │
│                                                                     │
│  .txt ─→ Line Parser ─→ Normalizer ─→ BGE-M3 임베딩 ─→ Neo4j     │
│          [date, time, room, content, user]                          │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  Search                                                             │
│                                                                     │
│  Query ─→ Embed ─→ Metadata Filter 적용                            │
│                        │                                            │
│             ┌──────────┴──────────┐                                 │
│             ▼                     ▼                                  │
│       Vector Search         Fulltext Search                         │
│       (Neo4j Vector)        (Neo4j Fulltext)                        │
│        top 100               top 100                                │
│             │                     │                                  │
│             └──────────┬──────────┘                                  │
│                        ▼                                             │
│              RRF Fusion (k=60)                                      │
│                        ▼                                             │
│  Graph Context: prev(2) + next(2) + recent_by_user(3) + same_day(5)│
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  Insights                                                           │
│                                                                     │
│  Messages by Date │ Top Rooms │ Top Users │ Keyword Samples         │
│  (배포, 롤백, 장애, PR #, 이슈 #, 마이그레이션, API)               │
└─────────────────────────────────────────────────────────────────────┘
```

**그래프 스키마**: `Message`, `User`, `Room`, `Date`
**엣지 타입**: `SENT`, `IN_ROOM`, `ON_DATE`, `PREV_IN_ROOM`

---

## Tech Stack

| Layer | Simple Graph-RAG | Hybrid Graph-RAG |
|-------|-------------------|-------------------|
| **Language** | Python 3.11 | Python 3.12 |
| **Framework** | FastAPI 0.115 + Uvicorn | FastAPI 0.115 + Uvicorn |
| **Vector DB** | PostgreSQL 16 + pgvector (HNSW) | Neo4j Vector Index (cosine) |
| **Graph DB** | Neo4j 5.22 | Neo4j 5.18 |
| **Embedding** | BAAI/bge-m3 (1024-dim) | BAAI/bge-m3 (1024-dim) |
| **Korean NLP** | KSS 6.0 (문장분리) + Kiwipiepy (형태소) | — |
| **LLM** | Claude via Stream-JSON Proxy | — |
| **Frontend** | Vanilla HTML/CSS/JS SPA | Jinja2 SSR Templates |
| **Infra** | Docker Compose (PostgreSQL + Neo4j) | Docker Compose (Neo4j + App) |

---

## Quick Start

### Simple Graph-RAG

```bash
cd simple-Graph-RAG

# 1. 환경 설정
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env

# 2. 인프라 시작
docker compose up -d   # PostgreSQL + Neo4j

# 3. LLM 프록시 (별도 터미널)
python3 scripts/claude_proxy.py --port 8800 --model sonnet --workers 2

# 4. 서버 시작
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 5. 데이터 적재
python3 scripts/ingest_file.py data/sample_chat.txt --source demo

# 6. http://localhost:8000 접속
```

### Hybrid Graph-RAG

```bash
cd hybrid-Graph-RAG

# 1. 환경 설정
pip install uv
uv sync
cp .env.example .env

# 2. 인프라 시작
docker compose up -d   # Neo4j

# 3. 스키마 생성 + 데이터 적재
python scripts/bootstrap_schema.py
python scripts/ingest_chat_logs.py --sample

# 4. 서버 시작
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 5. http://localhost:8000 접속
```

---

## Data Format

두 프로젝트 모두 동일한 채팅 로그 포맷을 사용합니다:

```
[2024-01-05, 07:59:12, 프로젝트C, 서버 배포 380차 완료했습니다, 박소율]
[2024-01-05, 08:03:45, 프로젝트A, PR #142 리뷰 부탁드립니다, 김민수]
```

| 필드 | 형식 | 설명 |
|------|------|------|
| date | `YYYY-MM-DD` | 메시지 날짜 |
| time | `HH:MM:SS` | 메시지 시간 |
| room | 문자열 | 채널/방 이름 |
| content | 문자열 (내부 쉼표 허용) | 메시지 본문 |
| user | 문자열 | 발신자 이름 |

**데이터 규모**: 10,000건 채팅 로그 / 30개 채널 / 179명 사용자 / 2024-01-01 ~ 2026-03-23

Simple Graph-RAG은 추가로 **이슈 워크북** (`.xlsx`)도 지원합니다:
- 이슈 ID, 제목, 상태, 담당자, 5-섹션 분석 텍스트 (원인요약/확인근거/기술판단/영향범위/추가조치)
- 10,000건 합성 이슈 데이터셋 포함

---

## API Endpoints

### Simple Graph-RAG (`:8000`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | 시스템 상태 (neo4j, postgres, embedding, llm proxy) |
| `POST` | `/query` | RAG 질의 → LLM 답변 생성 + 소스 인용 |
| `GET` | `/documents` | 문서 목록 조회 |
| `POST` | `/documents` | 문서 등록 (텍스트 직접 입력) |
| `POST` | `/documents/upload-file` | 파일 업로드 (`.txt` / `.xlsx`) |
| `DELETE` | `/documents/{id}` | 문서 삭제 (cascade) |
| `GET` | `/metadata` | 메타데이터 패싯 (채널, 사용자, 날짜, 상태, 담당자) |

### Hybrid Graph-RAG (`:8000`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | 시스템 상태 (neo4j, 메시지 수, 마지막 적재) |
| `POST` | `/api/v1/search/messages` | 하이브리드 검색 (vector + fulltext + RRF) |
| `GET` | `/api/v1/messages/{id}` | 메시지 상세 + 전후 컨텍스트 |
| `GET` | `/api/v1/insights/overview` | 인사이트 대시보드 (날짜별, 채널별, 사용자별, 키워드) |

---

## Retrieval Strategy Deep Dive

### Simple Graph-RAG: 4-Factor Weighted Ranking

```
Final Score = 0.45 × Vector + 0.25 × Graph + 0.20 × Metadata + 0.10 × Recency
```

| Factor | Weight | Source | Description |
|--------|--------|--------|-------------|
| **Vector** | 0.45 | pgvector cosine | 질의-청크 의미 유사도 |
| **Graph** | 0.25 | Neo4j BFS | 엔티티 공유, 관계 경로 중첩 |
| **Metadata** | 0.20 | 필터 매칭 | 날짜/채널/사용자 필터 + Intent 보너스 |
| **Recency** | 0.10 | `1/(1+days/30)` | 최신 데이터 가중치 |

**Intent 분류**: search, summary, timeline, aggregate, list, chat → 각 Intent별 가중치 동적 조정

**Query Router**: count 쿼리 / flow 쿼리 / issue+chat 혼합 쿼리를 자동 감지하여 전략 분기

### Hybrid Graph-RAG: Reciprocal Rank Fusion

```
RRF Score = 1/(k + rank_vector) + 1/(k + rank_fulltext),  k = 60
```

- Vector Search: 질의 임베딩 → Neo4j Vector Index → cosine 유사도 top 100
- Fulltext Search: 질의 텍스트 → Neo4j Fulltext Index → BM25 top 100
- 두 결과를 RRF로 합산 → 최종 top-k 선택
- Graph Expansion으로 전후 메시지 + 동일 사용자 최근 메시지 + 같은 날 같은 방 메시지 컨텍스트 추가

---

## Performance Benchmarks

### Simple Graph-RAG

| Metric | Value |
|--------|-------|
| RAG 쿼리 응답 (stream-json pool) | ~10s |
| RAG 쿼리 응답 (subprocess) | 15–19s |
| 10K 데이터 임베딩 (CPU) | ~128s |
| 10K 데이터 임베딩 (MPS/M4) | ~12–15min |
| 10K 데이터 PostgreSQL 적재 | 16.7s |
| 10K 데이터 Neo4j 적재 | 10.7s |

### Hybrid Graph-RAG

| Metric | Target |
|--------|--------|
| 검색 응답 p95 | < 2s |
| Top-5 Recall | 측정 중 |
| 파싱 성공률 | 100% (0 malformed lines) |

---

## Project Structure

```
graph-RAG-MVP/
├── README.md                          ← 현재 파일
├── simple-Graph-RAG/
│   ├── app/
│   │   ├── main.py                    # FastAPI 엔트리포인트
│   │   ├── config.py                  # Pydantic Settings
│   │   ├── container.py               # DI 컨테이너
│   │   ├── schemas.py                 # API 스키마
│   │   ├── adapters/                  # 외부 연동
│   │   │   ├── postgres_vector_store.py
│   │   │   ├── neo4j_store.py
│   │   │   ├── embedding_provider.py
│   │   │   ├── embedding_cache_store.py
│   │   │   └── codex_proxy.py
│   │   ├── api/                       # REST 엔드포인트
│   │   │   ├── health.py
│   │   │   ├── query.py
│   │   │   ├── documents.py
│   │   │   └── metadata.py
│   │   └── static/                    # SPA 프론트엔드
│   ├── scripts/                       # CLI 도구
│   │   ├── ingest_file.py             # 데이터 적재
│   │   ├── claude_proxy.py            # LLM 프록시 서버
│   │   ├── backfill_graph.py          # 그래프 재구축
│   │   ├── detect_communities.py      # 커뮤니티 탐지
│   │   ├── generate_unique_dataset.py # 합성 데이터 생성
│   │   └── assess_search_quality.py   # 검색 품질 평가
│   ├── tests/                         # 26개 테스트 파일
│   ├── data/                          # 채팅 로그 + 이슈 데이터
│   ├── docs/                          # PRD + TRD
│   ├── docker-compose.yml             # PostgreSQL + Neo4j
│   └── pyproject.toml
│
└── hybrid-Graph-RAG/
    ├── app/
    │   ├── main.py                    # FastAPI 엔트리포인트
    │   ├── settings.py                # Pydantic Settings
    │   ├── services/                  # 비즈니스 로직
    │   │   ├── search_service.py      # 하이브리드 검색
    │   │   ├── ingestion.py           # 적재 파이프라인
    │   │   ├── embedder.py            # BGE-M3 임베더
    │   │   ├── parser.py              # 로그 파서
    │   │   ├── normalizer.py          # 정규화
    │   │   ├── ranking.py             # RRF 퓨전
    │   │   └── insights_service.py    # 인사이트 집계
    │   ├── repositories/              # Neo4j 쿼리
    │   │   ├── neo4j_client.py
    │   │   ├── schema.py
    │   │   ├── search_repo.py
    │   │   ├── ingest_repo.py
    │   │   └── insights_repo.py
    │   ├── models/                    # 데이터 모델
    │   │   ├── api.py
    │   │   ├── records.py
    │   │   └── errors.py
    │   ├── api/                       # REST 엔드포인트
    │   │   ├── health.py
    │   │   ├── search.py
    │   │   ├── messages.py
    │   │   ├── insights.py
    │   │   └── ui.py
    │   └── templates/                 # Jinja2 SSR
    ├── scripts/                       # CLI 도구
    │   ├── bootstrap_schema.py        # 스키마 초기화
    │   └── ingest_chat_logs.py        # 데이터 적재
    ├── tests/                         # Unit + Smoke + Integration
    ├── data/                          # 채팅 로그
    ├── docs/                          # PRD + TRD + Init Guide
    ├── Dockerfile
    ├── docker-compose.yml             # Neo4j + App
    └── pyproject.toml
```

---

## Key Design Decisions

| Decision | Simple Graph-RAG | Hybrid Graph-RAG |
|----------|-------------------|-------------------|
| **청킹 전략** | KSS 문장분리 + 시간 기반 병합 (256 토큰) | 메시지 = 원자 단위 (청킹 없음) |
| **검색 방식** | 4-factor 가중 합산 | RRF 퓨전 (vector + fulltext) |
| **그래프 활용** | 엔티티 관계 + 공출현 → 랭킹에 반영 | 대화 흐름 (PREV_IN_ROOM) → 컨텍스트 확장 |
| **LLM** | Claude로 최종 답변 생성 | LLM 없이 검색 결과 직접 반환 |
| **이슈 데이터** | 지원 (BehaviorLabeler로 의미 단위 분리) | 미지원 |
| **인사이트** | 미지원 | 날짜별/채널별/사용자별 집계 + 키워드 샘플 |

---

## Environment Variables

<details>
<summary><strong>Simple Graph-RAG</strong></summary>

```env
# App
APP_NAME=simple-graph-rag
APP_ENV=development           # development / production

# PostgreSQL + pgvector
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=graph_rag
POSTGRES_USER=graph_rag
POSTGRES_PASSWORD=graph_rag

# Neo4j
NEO4J_URI=bolt://localhost:8768
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=graph-rag-password

# Embedding
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIMENSIONS=1024
EMBEDDING_DEVICE=cpu          # cpu / mps / cuda

# Chunking
CHUNK_MAX_TOKENS=256
CHUNK_MERGE_TIME_GAP_SECONDS=300
CHUNKER_BACKEND=kss

# LLM Proxy
CODEX_PROXY_BASE_URL=http://127.0.0.1:8800
CODEX_TIMEOUT_SECONDS=45
```

</details>

<details>
<summary><strong>Hybrid Graph-RAG</strong></summary>

```env
# App
APP_HOST=0.0.0.0
APP_PORT=8000

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# Embedding
EMBEDDING_MODEL_NAME=BAAI/bge-m3
EMBEDDING_DEVICE=cpu          # cpu / gpu
EMBEDDING_BATCH_SIZE=128
EMBEDDING_MAX_LENGTH=512
```

</details>

---

## Testing

```bash
# Simple Graph-RAG
cd simple-Graph-RAG
.venv/bin/python -m pytest tests/ -v

# Hybrid Graph-RAG
cd hybrid-Graph-RAG
uv run pytest tests/ -v

# Integration tests (requires running infrastructure)
RUN_INTEGRATION=1 uv run pytest tests/integration/ -v
```

---

## License

Private repository — internal use only.
