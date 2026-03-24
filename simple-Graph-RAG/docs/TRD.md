# TRD — Simple-Graph-RAG 기술 참조 문서

**문서 버전**: 2.0
**작성일**: 2026-03-24
**상태**: 목표 아키텍처 정의 완료

---

## 1. 시스템 아키텍처

### 1.1 전체 구조

```text
┌──────────────────────────────────────────────────────────────┐
│                     클라이언트 / 관리자 UI                    │
│                문서 업로드, 질의, 검색 결과 확인               │
└──────────────────────────┬───────────────────────────────────┘
                           │ HTTP / JSON
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    FastAPI 애플리케이션 레이어                │
│  /documents   /query   /health   /jobs                       │
└───────────────┬───────────────────────────────┬──────────────┘
                │                               │
                ▼                               ▼
┌──────────────────────────────┐   ┌───────────────────────────┐
│      적재/인덱싱 파이프라인     │   │      질의/검색 파이프라인         │
│ parse -> KSS chunk -> embed  │   │ analyze -> filter(date/acl/channel) │
│ -> extract entities          │   │ -> pgvector seed search             │
│ -> write graph + pgvector    │   │ -> expand graph -> rerank           │
└───────────────┬───────┬──────┘   └──────────────┬──────┬─────┘
                │       │                          │      │
                ▼       ▼                          ▼      ▼
┌──────────────────────┐  ┌──────────────────────┐  ┌────────────────────────────┐
│        Neo4j         │  │ PostgreSQL+pgvector  │  │ Codex CLI OAuth Proxy 계층 │
│ Property Graph       │  │ Chunk embeddings     │  │ ChatGPT 로그인 세션 기반     │
│ Fulltext/Graph index │  │ Metadata + ANN search│  │ Codex 모델 호출 어댑터       │
└─────────────┬────────┘  └─────────────┬────────┘  └────────────────────────────┘
              │                         │
              └────────────┬────────────┘
                           ▼
                ┌──────────────────────────┐
                │ 한국어 임베딩 모델 계층     │
                │ 기본 모델: bge-m3         │
                │ pluggable embedding adapter│
                └──────────────────────────┘
```

### 1.2 설계 원칙

- 관계 지식의 단일 원천은 Neo4j 그래프다.
- 벡터 검색 기본값은 PostgreSQL + pgvector다.
- 한국어 청킹 기본 구현은 `KSS`이며, 화자 전환과 턴 경계를 함께 고려한다.
- 기본 임베딩 모델은 `bge-m3`이며 차원 수는 1024로 고정한다.
- LLM 호출은 애플리케이션이 직접 API 키를 보관하지 않고 Codex CLI 인증 세션을 우회 사용한다.
- 기존 `chroma_db` 디렉터리 기반 ChromaDB는 사용하지 않는다.
- 검색 기본 순서는 `날짜/권한/채널 필터 -> pgvector 시드 검색 -> 그래프 확장`이다.
- 검색 결과는 항상 청크 근거와 그래프 근거를 함께 남긴다.

### 1.3 레이어 구조

```text
┌─────────────────────────────────────────┐
│ 프레젠테이션 레이어                       │
│ app/api/*.py, app/static/*              │
├─────────────────────────────────────────┤
│ 스키마 레이어                            │
│ app/schemas.py                          │
├─────────────────────────────────────────┤
│ 도메인 서비스 레이어                      │
│ chunking, graph_builder, retrieval      │
│ query_analyzer, llm_adapter             │
├─────────────────────────────────────────┤
│ 인프라 어댑터 레이어                      │
│ neo4j, postgres_vector_store,           │
│ embedding_provider, codex_proxy         │
├─────────────────────────────────────────┤
│ 외부 런타임                              │
│ Neo4j, PostgreSQL, Embedding Model,     │
│ Codex CLI                               │
└─────────────────────────────────────────┘
```

---

## 2. 기술 스택 상세

### 2.1 핵심 의존성

| 항목 | 역할 |
|------|------|
| FastAPI | API 서버 |
| Pydantic / pydantic-settings | 요청 검증 및 설정 관리 |
| Neo4j + neo4j Python Driver | 그래프 저장 및 질의 |
| PostgreSQL + pgvector | 기본 벡터 저장소 및 벡터 유사도 검색 |
| psycopg / asyncpg | PostgreSQL 연결 |
| Neo4j Fulltext Index | 엔티티/키워드 검색 |
| `kss` | 한국어 문장 분절 및 청킹 보조 |
| `bge-m3` | 기본 한국어 임베딩 모델 |
| 한국어 임베딩 모델 어댑터 | 청크 임베딩 생성 |
| Codex CLI (`@openai/codex`) | ChatGPT 계정 인증 기반 LLM 런타임 |
| Codex Proxy Adapter | 앱과 Codex CLI 사이의 로컬 브리지 |

### 2.2 선택 기술

| 항목 | 목적 |
|------|------|
| APOC | 그래프 유틸리티 및 배치 처리 |
| Graph Data Science | 향후 커뮤니티 탐지/유사도 확장 |
| Redis/큐 | 대량 적재 및 생성 요청 직렬화 |

### 2.3 외부 런타임

| 서비스 | 용도 | 비고 |
|--------|------|------|
| Neo4j 5.x | Property graph + fulltext index | 필수 |
| PostgreSQL 15+ + pgvector | chunk vector store | 필수 |
| `bge-m3` 임베딩 런타임 | 청크 임베딩 생성 | 필수 |
| Codex CLI | 답변 생성 | 필수 |

---

## 3. 데이터 모델

### 3.1 핵심 노드

| 노드 | 주요 속성 | 설명 |
|------|-----------|------|
| `Document` | `document_id`, `filename`, `created_at` | 업로드 단위 |
| `Chunk` | `chunk_id`, `text`, `channel`, `date`, `time`, `seq`, `token_count`, `access_scope` | 그래프와 연결되는 검색 기본 단위 |
| `User` | `name`, `normalized_name` | 화자 |
| `Channel` | `name` | 채널(원본 채팅방) |
| `Date` | `date`, `date_int` | 시간 필터 |
| `Entity` | `name`, `type` | 이슈, 시스템, 기능, 사람 외 개체 |
| `Topic` | `name` | 요약/주제 단위 |

### 3.2 핵심 관계

| 관계 | 방향 | 설명 |
|------|------|------|
| `PART_OF` | `Chunk -> Document` | 소속 문서 |
| `SENT_BY` | `Chunk -> User` | 화자 연결 |
| `IN_CHANNEL` | `Chunk -> Channel` | 채널 연결 |
| `ON_DATE` | `Chunk -> Date` | 날짜 연결 |
| `MENTIONS` | `Chunk -> Entity` | 엔티티 언급 |
| `HAS_TOPIC` | `Chunk -> Topic` | 주제 분류 |
| `NEXT` | `Chunk -> Chunk` | 시간순 이웃 |
| `RELATED_TO` | `Entity -> Entity` | 공기/규칙 기반 관계 |

### 3.3 Neo4j 저장 예시

```cypher
(:Chunk {
  chunk_id: "doc_001_chunk_042",
  text: "백엔드개발 김민수: 서버 배포 3차 완료했습니다",
  channel: "백엔드개발",
  seq: 42,
  date: "2024-02-21",
  time: "14:30:00",
  access_scope: "team:backend"
})-[:SENT_BY]->(:User {name: "김민수"})
```

### 3.4 인덱스 전략

```cypher
CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE;

CREATE FULLTEXT INDEX entity_name_idx IF NOT EXISTS
FOR (e:Entity) ON EACH [e.name];
```

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunk_embeddings (
  chunk_id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL,
  channel TEXT NOT NULL,
  user_name TEXT,
  message_date DATE NOT NULL,
  message_time TIME NOT NULL,
  access_scope TEXT,
  chunk_text TEXT NOT NULL,
  embedding vector(1024) NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS chunk_embeddings_channel_idx
  ON chunk_embeddings (channel);

CREATE INDEX IF NOT EXISTS chunk_embeddings_message_date_idx
  ON chunk_embeddings (message_date);

CREATE INDEX IF NOT EXISTS chunk_embeddings_access_scope_idx
  ON chunk_embeddings (access_scope);

CREATE INDEX IF NOT EXISTS chunk_embeddings_embedding_idx
  ON chunk_embeddings
  USING hnsw (embedding vector_cosine_ops);
```

### 3.5 저장 전략

- 그래프 관계와 엔티티 탐색은 Neo4j가 담당한다.
- 청크 임베딩과 벡터 검색은 PostgreSQL + pgvector가 담당한다.
- 두 저장소는 `chunk_id`와 `document_id`를 공통 키로 동기화한다.
- 검색 시 `date`, `access_scope`, `channel` 필터를 먼저 적용한다.
- 기존 `chroma_db` 또는 ChromaDB 컬렉션은 운영 구성에서 제외한다.

---

## 4. 핵심 알고리즘

### 4.1 한국어 청킹 파이프라인

```text
입력 로그 라인
  -> 정규식 파싱
  -> 사용자/채팅방/시간 메타데이터 추출
  -> 본문에 KSS 문장 분절 적용
  -> 화자 전환 기준 분절
  -> 짧은 인접 턴 병합
  -> 최대 토큰 제한 적용
  -> overlap 턴 연결
  -> chunk_text 생성
  -> bge-m3 embedding 생성
  -> PostgreSQL/pgvector 저장
  -> Neo4j 그래프 저장
```

### 4.2 청킹 규칙

| 규칙 | 목적 |
|------|------|
| KSS 기반 문장 분절 후 턴 재조합 | 한국어 문맥 보존 |
| 동일 화자의 짧은 연속 발화 병합 | 의미 단위 보존 |
| 동일 이슈 키워드의 인접 턴 병합 | 관계 정보 유지 |
| 질문-응답 쌍 분리 방지 | 검색 정밀도 향상 |
| 시간 간격이 큰 경우 강제 분리 | 대화 세션 혼합 방지 |

### 4.3 임베딩 전략

- 임베딩 대상은 원문 전체가 아니라 `정규화된 chunk_text`다.
- 기본 포맷은 `채널 + 사용자 + 내용 + 필요 시 엔티티 요약`이다.
- 기본 임베딩 모델은 `bge-m3`이며 1024차원 코사인 유사도를 사용한다.
- 모델 구현은 인터페이스로 감싸고, 차원 수와 전처리 규칙은 설정값으로 외부화한다.

```python
{
    "embedding_text": "백엔드개발 김민수: 서버 배포 3차 완료했습니다 [엔티티: 서버 배포]",
    "metadata": {
        "channel": "백엔드개발",
        "user": "김민수",
        "date": "2024-02-21",
        "time": "14:30:00",
        "access_scope": "team:backend"
    }
}
```

### 4.4 GraphRAG 검색 라우팅

```text
질의 입력
  -> 날짜/권한/채널/사용자/엔티티 추출
  -> 의도 분류
  -> 메타데이터 선필터 적용
  -> PostgreSQL/pgvector 시드 청크 벡터 검색
  -> 관련 엔티티/이웃 노드 그래프 확장
  -> 하이브리드 점수 계산
  -> 상위 컨텍스트 선택
  -> LLM 답변 생성
```

### 4.5 하이브리드 점수 예시

```text
final_score =
  0.45 * vector_score +
  0.25 * graph_neighbor_score +
  0.20 * metadata_match_score +
  0.10 * recency_score
```

### 4.6 관계형 질의 처리

- 자유 형식 Cypher 생성을 기본값으로 두지 않는다.
- 질의 의도별 템플릿을 먼저 적용한다.
- 예:
  - "누가 누구와 같이 언급됐나?" -> 엔티티/사용자 공기 질의
  - "어떤 흐름으로 진행됐나?" -> `NEXT` 기반 타임라인 질의
  - "가장 활발한 채널은?" -> 집계 Cypher

---

## 5. LLM 계층 설계

### 5.1 Codex CLI OAuth Proxy 개요

본 시스템은 LLM 호출을 위해 OpenAI API 키를 앱 설정에 저장하지 않는다.
대신 개발자/운영자가 로컬에서 **Codex CLI에 ChatGPT 계정으로 로그인한 세션**을
사용하고, 애플리케이션은 이 세션을 감싼 **로컬 Proxy Adapter**에 요청한다.

즉, `Codex CLI OAuth Proxy`는 OpenAI 공식 제품명이 아니라
**이 프로젝트 내부에서 Codex CLI 인증 세션을 서비스형 인터페이스로 감싸는 방식**을
가리키는 설계 용어다.

### 5.2 호출 흐름

```text
FastAPI /query
  -> Retrieval context 구성
  -> CodexProxyLLM.generate(...)
  -> local proxy process / service
  -> authenticated Codex CLI session
  -> Codex model response
  -> answer + citations 반환
```

### 5.3 설계 요구사항

| 항목 | 요구사항 |
|------|----------|
| 인증 방식 | Codex CLI ChatGPT 로그인 세션 |
| 앱 비밀값 | OpenAI API 키 저장 금지 |
| 프록시 노출 | `127.0.0.1` 기본 |
| 실패 처리 | 세션 만료/로그인 필요 상태 감지 |
| 모델 선택 | Codex CLI 설정 또는 프록시 설정으로 외부화 |
| 감사 추적 | 프롬프트, 컨텍스트, 응답 메타정보 로깅 가능 |

### 5.4 프롬프트 구조

```text
System:
- 한국어로 답변
- 제공된 근거 밖 추론 최소화
- 관계와 시간 흐름을 분리해서 설명

User Question:
{question}

Graph Evidence:
{graph_context}

Chunk Evidence:
{chunk_context}
```

### 5.5 폴백 전략

- Codex Proxy 장애 시 답변 대신 검색 근거만 반환
- 세션 만료 시 `reauth_required` 에러 코드 반환
- 응답 길이 초과 시 요약형 재시도

---

## 6. API 상세 명세

### 6.1 POST /query

**요청**
```json
{
    "question": "2024년 3월 개발팀의 배포 흐름을 요약해줘",
    "top_k": 10
}
```

**응답**
```json
{
    "question": "2024년 3월 개발팀의 배포 흐름을 요약해줘",
    "answer": "2024년 3월 개발팀에서는 ...",
    "retrieval_strategy": "graph_hybrid",
    "sources": [
        {
            "chunk_id": "doc_001_chunk_042",
            "score": 0.91,
            "content": "...",
            "graph_neighbors": ["개발팀", "서버 배포", "박서준"]
        }
    ]
}
```

**내부 처리 흐름**
```text
1. QueryRequest 검증
2. analyze_query(question)
3. apply_date_acl_channel_filters()
4. retrieve_seed_chunks_from_pgvector()
5. expand_graph_neighbors()
6. rerank_context()
7. codex_proxy.generate()
8. QueryResponse 반환
```

### 6.2 POST /documents

**요청**
```json
{
    "filename": "chat_logs.txt",
    "content": "..."
}
```

**내부 처리 흐름**
```text
1. parse_log_lines()
2. split_sentences_with_kss()
3. build_chunks()
4. generate_embeddings_bge_m3()
5. write_pgvector_rows()
6. extract_entities_relations()
7. write_neo4j_graph()
8. update ingestion job state
```

### 6.3 GET /health

**응답**
```json
{
    "status": "ok",
    "neo4j": "ok",
    "postgres": "ok",
    "embedding": "ok",
    "codex_proxy": "ok"
}
```

---

## 7. 설정 명세

### 7.1 환경 변수

| 변수명 | 설명 | 예시 |
|--------|------|------|
| `NEO4J_URI` | Neo4j 접속 URI | `bolt://localhost:7687` |
| `NEO4J_USERNAME` | Neo4j 계정 | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j 비밀번호 | `password` |
| `POSTGRES_HOST` | PostgreSQL 호스트 | `localhost` |
| `POSTGRES_PORT` | PostgreSQL 포트 | `5432` |
| `POSTGRES_DB` | PostgreSQL 데이터베이스명 | `graph_rag` |
| `POSTGRES_USER` | PostgreSQL 사용자 | `postgres` |
| `POSTGRES_PASSWORD` | PostgreSQL 비밀번호 | `password` |
| `PGVECTOR_TABLE` | 벡터 저장 테이블명 | `chunk_embeddings` |
| `CHUNKER_BACKEND` | 한국어 청킹 백엔드 | `kss` |
| `EMBEDDING_PROVIDER` | 임베딩 어댑터 종류 | `local_transformer` |
| `EMBEDDING_MODEL` | 기본 한국어 임베딩 모델명 | `bge-m3` |
| `EMBEDDING_DIMENSIONS` | 벡터 차원 수 | `1024` |
| `CHUNK_MAX_TOKENS` | 청크 최대 토큰 수 | `256` |
| `CHUNK_OVERLAP_TURNS` | 겹침 턴 수 | `1` |
| `CODEX_PROXY_BASE_URL` | 로컬 프록시 URL | `http://127.0.0.1:8800` |
| `CODEX_MODEL` | Codex 모델명 | `gpt-5.3-codex` |
| `TOP_K` | 기본 검색 결과 수 | `10` |

### 7.2 프록시 설정 원칙

- `CODEX_PROXY_BASE_URL`은 기본적으로 로컬 루프백만 허용한다.
- 프록시는 Codex CLI 바이너리 존재 여부와 로그인 상태를 시작 시 검증한다.
- 애플리케이션은 프록시에게만 요청하고 OpenAI에 직접 통신하지 않는다.
- PostgreSQL에는 `pgvector` extension이 미리 설치되어 있어야 한다.

---

## 8. 보안 및 운영 고려사항

### 8.1 보안

| 항목 | 대응 |
|------|------|
| API 키 무보관 | Codex CLI 로그인 세션 사용 |
| 프록시 오남용 | 로컬 바인딩, 토큰/소켓 보호 |
| 과도한 프롬프트 유출 | 민감정보 마스킹 |
| Cypher 인젝션 | 자유 질의 대신 템플릿 기반 실행 |
| ChromaDB 잔재 구성 | 사용 금지, pgvector로 통일 |
| 권한 필터 누락 | ACL/채널 범위 선필터 강제 |

### 8.2 운영

| 항목 | 대응 |
|------|------|
| 적재 대기열 | 비동기 잡 큐 도입 가능 |
| 재색인 비용 | 문서 단위 부분 재색인 |
| 모델 전환 | 어댑터 인터페이스 유지 |
| 세션 만료 | 헬스체크 및 운영 가이드 |

---

## 9. 테스트 전략

### 9.1 단위 테스트

| 대상 | 내용 |
|------|------|
| `chunking` | 한국어 로그 파싱, KSS 분절, 턴 병합, overlap |
| `postgres_vector_store` | pgvector 저장/검색, metadata filter |
| `query_analyzer` | 날짜/채널/권한/사용자/엔티티 감지 |
| `graph_builder` | 노드/관계 생성 규칙 |
| `retrieval` | 날짜/권한/채널 필터, 시드 검색, 확장, 재정렬 |
| `codex_proxy` | 성공/실패/세션 만료 처리 |

### 9.2 통합 테스트

| 대상 | 내용 |
|------|------|
| PostgreSQL pgvector 적재 | 업로드 후 벡터 row 및 인덱스 생성 검증 |
| Neo4j 적재 | 업로드 후 그래프/인덱스 생성 검증 |
| GraphRAG 검색 | `필터 -> pgvector -> graph expand` 라우팅 검증 |
| LLM 어댑터 | 프록시 mock 기반 응답 검증 |

### 9.3 E2E 시나리오

| 시나리오 | 기대 결과 |
|----------|-----------|
| 문서 업로드 -> 질의 -> 답변 | 근거 포함 답변 반환 |
| 세션 만료 상태 질의 | `reauth_required` 반환 |
| 관계형 질문 | 그래프 이웃을 포함한 답변 생성 |

---

## 10. 목표 디렉토리 구조

```text
simple-Graph-RAG/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── schemas.py
│   ├── api/
│   │   ├── documents.py
│   │   ├── query.py
│   │   └── health.py
│   ├── services/
│   │   ├── chunking.py
│   │   ├── query_analyzer.py
│   │   ├── graph_builder.py
│   │   ├── retrieval.py
│   │   └── ingest.py
│   ├── adapters/
│   │   ├── neo4j_store.py
│   │   ├── postgres_vector_store.py
│   │   ├── embedding_provider.py
│   │   └── codex_proxy.py
│   └── static/
│       └── index.html
├── data/
├── docs/
│   ├── PRD.md
│   └── TRD.md
├── tests/
└── scripts/
```

---

## 11. 실행 및 배포 기준

### 11.1 로컬 준비

```bash
# 1. PostgreSQL + pgvector 실행
# 2. Neo4j 실행
# 3. Codex CLI 설치
npm i -g @openai/codex

# 4. Codex CLI 로그인
codex

# 5. 애플리케이션 환경 변수 설정
# 6. FastAPI 서버 실행
```

### 11.2 운영 전 체크리스트

- PostgreSQL + pgvector 접속 가능 여부
- Neo4j 접속 가능 여부
- pgvector 테이블/인덱스 생성 여부
- KSS 청킹 라이브러리 준비 여부
- `bge-m3` 모델 준비 여부
- Codex CLI 로그인 상태
- Codex Proxy 헬스체크 성공 여부
