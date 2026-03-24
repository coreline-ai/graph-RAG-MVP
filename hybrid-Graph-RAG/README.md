# Hybrid GraphRAG MVP

한국어 채팅 로그를 대상으로 날짜/방/사용자 필터, 의미 검색, 그래프 맥락 확장, 구조화 인사이트를 제공하는 FastAPI + Neo4j 기반 MVP다.

## 요구사항

- Python 3.12
- `uv`
- Docker
- `docker-compose` 또는 Compose 플러그인

## 빠른 시작

1. 환경 파일 준비

```bash
cp .env.example .env
```

기본값만으로 실행할 때는 이 단계는 생략 가능하다.

2. 의존성 설치

```bash
uv sync
```

3. Neo4j 실행

```bash
docker-compose up -d neo4j
```

Compose 플러그인을 쓰는 환경이면 아래도 가능하다.

```bash
docker compose up -d neo4j
```

4. 스키마 초기화

```bash
uv run python scripts/bootstrap_schema.py
```

5. 샘플 데이터 적재

```bash
uv run python scripts/ingest_chat_logs.py --input data/chat_logs_100.txt
```

6. 앱 실행

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

7. 접속

- 앱: http://localhost:8000
- Neo4j Browser: http://localhost:7474

## 전체 데이터 적재

```bash
uv run python scripts/ingest_chat_logs.py --input data/chat_logs.txt --rebuild-prev-links
```

## 테스트

빠른 단위 테스트:

```bash
uv run pytest tests/unit tests/smoke
```

Neo4j와 임베딩 모델이 모두 준비된 통합 테스트:

```bash
RUN_INTEGRATION=1 uv run pytest tests/integration -m integration
```

## 주요 경로

- 검색 API: `POST /api/v1/search/messages`
- 메시지 상세 API: `GET /api/v1/messages/{message_id}`
- 인사이트 API: `GET /api/v1/insights/overview`
- 상태 확인: `GET /health`
- 검색 UI: `/`
- 인사이트 UI: `/insights`

## 환경 변수

- `APP_HOST`: 기본 `0.0.0.0`
- `APP_PORT`: 기본 `8000`
- `NEO4J_URI`: 기본 `bolt://localhost:7687`
- `NEO4J_USERNAME`: 기본 `neo4j`
- `NEO4J_PASSWORD`: 기본 `password`
- `NEO4J_DATABASE`: 기본 `neo4j`
- `EMBEDDING_MODEL_NAME`: 기본 `BAAI/bge-m3`
- `EMBEDDING_DEVICE`: 기본 `cpu`
- `EMBEDDING_BATCH_SIZE`: 기본 `128`
- `EMBEDDING_MAX_LENGTH`: 기본 `512`
- `DATA_DIR`: 기본 `./data`
- `LOG_DIR`: 기본 `./logs`
