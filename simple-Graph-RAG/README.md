<div align="center">

# Simple-Graph-RAG

**Korean Chat Log Analysis with Graph-Augmented Retrieval**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.22-4581C3?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![PostgreSQL](https://img.shields.io/badge/pgvector-PG16-4169E1?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](#license)

</div>

---

## Overview

한국어 조직 채팅 로그를 **지식 그래프 + 벡터 검색**으로 분석하는 GraphRAG 시스템입니다.

PostgreSQL/pgvector의 시맨틱 검색과 Neo4j의 관계 그래프를 결합하여, 단순 키워드 매칭을 넘어 **맥락을 이해하는 검색**을 제공합니다.

### Key Features

- **Hybrid Retrieval** — 벡터 유사도 + 그래프 관계 + 메타데이터 필터링 4-factor 랭킹
- **Korean-Optimized** — KSS 문장 분리, bge-m3 임베딩, 한국어 날짜/이름 파싱
- **Intent-Aware** — 질문 의도(검색/요약/타임라인/집계/관계) 자동 감지 및 가중치 조정
- **Stream-JSON Process Pool** — Claude CLI 프로세스 재사용으로 LLM 응답 지연 최소화
- **Production-Ready UI** — 한국어 챗봇 GUI, 필터 사이드바, 다크모드, 접근성 지원

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Client (Browser)                     │
│                   app/static/index.html                  │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Server (:8000)                 │
│                       app/main.py                        │
├─────────┬────────────┬──────────────┬───────────────────┤
│ /health │ /documents │   /query     │   / (static)      │
└─────────┴─────┬──────┴──────┬───────┴───────────────────┘
                │             │
        ┌───────▼───────┐  ┌──▼──────────────────────┐
        │ IngestService │  │   RetrievalService       │
        │  ┌──────────┐ │  │  ┌────────────────────┐  │
        │  │ Chunking │ │  │  │  QueryAnalyzer     │  │
        │  │ GraphBld │ │  │  │  4-Factor Ranking  │  │
        │  └──────────┘ │  │  └────────────────────┘  │
        └───┬───────┬───┘  └──┬──────┬──────┬─────────┘
            │       │         │      │      │
   ┌────────▼──┐ ┌──▼─────┐  │   ┌──▼──┐ ┌─▼──────────┐
   │ PostgreSQL│ │ Neo4j  │  │   │Neo4j│ │Claude Proxy│
   │ pgvector  │ │ Graph  │  │   │     │ │  (:8800)   │
   │  (:5432)  │ │(:8768) │  │   └─────┘ └─────┬──────┘
   └───────────┘ └────────┘  │                  │
                             │           ┌──────▼──────┐
                    ┌────────▼────────┐  │ Claude CLI  │
                    │   bge-m3        │  │ Stream-JSON │
                    │ (1024-dim, CPU) │  │ Worker Pool │
                    └─────────────────┘  └─────────────┘
```

### Retrieval Pipeline

```
Question ──► QueryAnalyzer ──► Embedding ──► pgvector Search ──► Neo4j Expand
                │                                                     │
                │ intent, filters                                     │ graph neighbors
                ▼                                                     ▼
           4-Factor Ranking ◄──────────────────────────────── Merge & Deduplicate
                │
                │  vector(0.45) + graph(0.25) + metadata(0.20) + recency(0.10)
                ▼
           Top-K Sources ──► LLM Generation ──► Response
```

---

## Tech Stack

| Layer | Technology | Role |
|:------|:-----------|:-----|
| API | FastAPI + Uvicorn | 비동기 웹 서버 |
| Vector DB | PostgreSQL 16 + pgvector | HNSW 벡터 검색 + 메타데이터 필터 |
| Graph DB | Neo4j 5.22 | 엔티티 관계 그래프, 인접 확장 |
| Embedding | BAAI/bge-m3 | 1024차원, 다국어 임베딩 |
| LLM | Claude CLI (Stream-JSON) | 프로세스 풀 기반 응답 생성 |
| NLP | KSS (Korean Sentence Splitter) | 한국어 문장 분리 |
| Frontend | Vanilla HTML/CSS/JS | SPA 챗봇 UI |
| Infra | Docker Compose | PostgreSQL + Neo4j 오케스트레이션 |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- [Claude CLI](https://docs.anthropic.com/en/docs/claude-code) (LLM 생성용)

### 1. Clone & Setup

```bash
git clone <repository-url>
cd simple-Graph-RAG

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Infrastructure

```bash
# PostgreSQL (pgvector) + Neo4j 시작
docker compose up -d

# 상태 확인
docker compose ps
```

### 3. Environment

```bash
cp .env.example .env
# .env 파일에서 필요 시 설정 수정
```

### 4. LLM Proxy

기본값은 현재 저장소의 `claude_proxy.py`를 사용합니다.

```bash
# 별도 터미널에서 실행
python3 scripts/claude_proxy.py --port 8800 --model sonnet --workers 2
```

`multi_model_tui` 프록시를 쓰려면 OpenAI 호환 엔드포인트에 연결하면 됩니다.

```bash
# multi_model_tui 기본 포트 예시
CODEX_PROXY_BASE_URL=http://127.0.0.1:4317
CODEX_PROXY_API_STYLE=openai_responses
CODEX_MODEL=gpt-5
```

### 5. Run Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Ingest Data

```bash
# 샘플 데이터
python3 scripts/ingest_file.py data/sample_chat.txt --source demo

# 전체 데이터 (10,000건)
python3 scripts/ingest_file.py data/chat_logs.txt --source production
```

### 7. Open UI

브라우저에서 **http://localhost:8000** 접속

---

## Project Structure

```
simple-Graph-RAG/
├── app/
│   ├── main.py                  # FastAPI 엔트리포인트
│   ├── config.py                # Pydantic Settings
│   ├── container.py             # DI 컨테이너 (서비스 조합)
│   ├── schemas.py               # Pydantic 데이터 모델
│   ├── adapters/
│   │   ├── postgres_vector_store.py  # pgvector 커넥션 풀 + CRUD
│   │   ├── neo4j_store.py            # Neo4j 그래프 드라이버
│   │   ├── embedding_provider.py     # bge-m3 임베딩 모델
│   │   └── codex_proxy.py            # LLM proxy HTTP 클라이언트 (legacy / OpenAI-compatible)
│   ├── api/
│   │   ├── health.py            # GET /health
│   │   ├── query.py             # POST /query
│   │   └── documents.py         # /documents CRUD
│   ├── services/
│   │   ├── chunking.py          # 로그 파싱 + 청크 빌드
│   │   ├── graph_builder.py     # 엔티티 추출 → 그래프 노드/엣지
│   │   ├── ingest.py            # 수집 파이프라인 오케스트레이션
│   │   ├── retrieval.py         # 하이브리드 검색 + LLM 생성
│   │   └── query_analyzer.py    # 의도/필터/엔티티 추출
│   └── static/
│       └── index.html           # 챗봇 GUI (SPA)
│
├── scripts/
│   ├── claude_proxy.py          # Claude CLI 프록시 (Stream-JSON 풀)
│   ├── ingest_file.py           # CLI 배치 수집
│   ├── query_documents.py       # CLI 쿼리 테스트
│   ├── backfill_graph.py        # 그래프 재구축
│   ├── bootstrap_postgres.sql   # 테이블/인덱스 DDL
│   └── bootstrap_neo4j.cypher   # 제약조건 Cypher
│
├── data/
│   ├── sample_chat.txt          # 29건 샘플
│   ├── chat_logs_100.txt        # 100건 테스트
│   └── chat_logs.txt            # 10,000건 전체
│
├── tests/
│   ├── test_api.py
│   ├── test_chunking.py
│   ├── test_query_analyzer.py
│   └── test_codex_proxy.py
│
├── docs/
│   ├── PRD.md                   # 제품 요구사항
│   └── TRD.md                   # 기술 요구사항
│
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

---

## API Reference

### Health Check

```
GET /health
```
```json
{
  "status": "ok",
  "neo4j": "ok",
  "postgres": "ok",
  "embedding": "ok",
  "codex_proxy": "ok"
}
```

### Query

```
POST /query
```
```json
{
  "question": "개발팀에서 최근 무슨 대화가 있었어?",
  "access_scopes": ["general"],
  "top_k": 10
}
```

**Response:**
```json
{
  "question": "개발팀에서 최근 무슨 대화가 있었어?",
  "answer": "## 개발팀 채널 대화 요약 ...",
  "answer_mode": "llm",
  "retrieval_strategy": "filter_pgvector_graph_hybrid",
  "sources": [
    {
      "chunk_id": "...",
      "score": 0.8234,
      "content": "회의록 공유 부탁드립니다",
      "channel": "개발팀",
      "user_name": "김민수",
      "message_date": "2025-03-20",
      "graph_neighbors": ["Entity:배포", "User:박지현"]
    }
  ]
}
```

### Documents

```
POST   /documents              # 텍스트 직접 생성
POST   /documents/upload-file  # 파일 업로드
GET    /documents              # 목록 조회
GET    /documents/{id}         # 상세 조회
DELETE /documents/{id}         # 삭제 (cascade)
```

---

## Graph Schema

```cypher
(:Document {document_id})
(:Chunk {chunk_id, text, token_count})
(:User {name})
(:Channel {name})
(:Date {date})
(:Entity {name})

(Chunk)-[:PART_OF]->(Document)
(Chunk)-[:SENT_BY]->(User)
(Chunk)-[:IN_CHANNEL]->(Channel)
(Chunk)-[:ON_DATE]->(Date)
(Chunk)-[:MENTIONS]->(Entity)
(Chunk)-[:NEXT]->(Chunk)          // 시간순 인접
```

---

## Data Format

입력 로그 포맷 (한 줄 = 한 메시지):

```
[2025-03-20, 14:30:22, 개발팀, 회의록 공유 부탁드립니다, 김민수]
```

| Field | Description |
|:------|:------------|
| `YYYY-MM-DD` | 메시지 날짜 |
| `HH:MM:SS` | 메시지 시간 |
| `channel` | 채널명 |
| `content` | 메시지 본문 |
| `username` | 발신자 |

---

## Configuration

주요 환경 변수 (`.env`):

| Variable | Default | Description |
|:---------|:--------|:------------|
| `TOP_K` | `10` | 검색 결과 수 |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | 임베딩 모델 |
| `EMBEDDING_DIMENSIONS` | `1024` | 벡터 차원 |
| `CHUNK_MAX_TOKENS` | `256` | 청크 최대 토큰 |
| `CHUNK_MERGE_TIME_GAP_SECONDS` | `300` | 병합 시간 간격 (초) |
| `CODEX_PROXY_BASE_URL` | `http://127.0.0.1:8800` | Claude 프록시 URL |
| `GRAPH_NEXT_WINDOW` | `2` | 그래프 확장 윈도우 |
| `DEFAULT_ACCESS_SCOPES` | `public` | 기본 접근 범위 |

---

## Testing

```bash
# 전체 테스트
pytest

# 커버리지
pytest --cov=app --cov-report=term-missing

# 특정 모듈
pytest tests/test_chunking.py -v
```

---

## Performance

Stream-JSON 프로세스 풀 적용 후 벤치마크:

| Metric | Before (subprocess) | After (stream-json pool) |
|:-------|:-------------------:|:------------------------:|
| 짧은 프롬프트 | 13–15s | **2–3s** |
| RAG 쿼리 (긴 프롬프트) | 15–19s | **~10s** |
| Node.js 부트 오버헤드 | 매 호출 ~5s | 첫 호출만 |
| 10K 데이터 임베딩 | 128.8s | 128.8s |
| 10K 데이터 PG 적재 | 16.7s | 16.7s |
| 10K 데이터 Neo4j 적재 | 10.7s | 10.7s |

---

## License

MIT
