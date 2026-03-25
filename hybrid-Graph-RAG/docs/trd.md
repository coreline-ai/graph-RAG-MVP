# Hybrid GraphRAG TRD

기준 데이터: `data/chat_logs*.txt` 분석 반영

- 전체 로그: 10,000건
- 샘플 파일: 100건
- 날짜 범위: 2024-01-01 ~ 2026-03-23
- 채팅방 수: 30개
- 사용자 수: 179명
- 메시지 본문 평균 길이: 약 17자
- 메시지 본문 최대 길이: 26자
- 본문 내부 `, ` 포함 메시지: 569건
- malformed line: 0건

## 1. 문서 개요

### 1.1 목적

본 문서는 한국어 채팅 로그 기반 Hybrid GraphRAG MVP의 기술 요구사항과 구현 기준을 정의한다. 구현자는 본 문서를 기준으로 별도 설계 결정을 추가하지 않고 파서, 적재기, 검색 API, UI를 구현할 수 있어야 한다.

### 1.2 범위

- 입력: 현재 정규화된 TXT 채팅 로그
- 처리: 파싱, 정규화, 임베딩, Neo4j 적재, 검색, 인사이트 집계
- 출력: REST API, 내부용 SSR UI

## 2. 기술 기준과 외부 의존성

### 2.1 런타임 및 프레임워크

- Python 3.12
- FastAPI
- Jinja2 기반 SSR UI
- Neo4j 5.18.1 이상
- Neo4j Python Driver

### 2.2 Retrieval 및 모델 선택

- 그래프/질의 계층: `neo4j` Python Driver + 직접 작성한 Cypher 질의
  - 메타데이터 필터, 벡터 검색, full-text 검색, 그래프 컨텍스트 확장을 애플리케이션 코드에서 조합
  - 현재 MVP는 Microsoft GraphRAG나 `neo4j-graphrag`를 직접 연동하지 않음
- 임베딩 모델: `BAAI/bge-m3`
  - dense vector 1024 차원 사용
  - 현재 채팅 로그는 짧은 문장 중심이므로 dense retrieval만 MVP에 포함
- KSS
  - 현재 MVP에는 사용하지 않음
  - 향후 긴 문서 ingestion 시 sentence split 용도로만 검토

### 2.3 기술 선택 근거

- `BAAI/bge-m3` 모델 카드: https://huggingface.co/BAAI/bge-m3
- Neo4j Vector Index 문서: https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/
- Neo4j Full-text Index 문서: https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/full-text-indexes/

## 3. 시스템 아키텍처

전체 파이프라인은 아래 한 경로로 고정한다.

`TXT Parser -> Normalizer -> Embedder -> Neo4j Loader -> Search API -> Internal UI`

### 3.1 구성요소

1. TXT Parser
   - raw line을 구조화 레코드로 변환
2. Normalizer
   - 날짜/시각 정규화, message_id 생성, 검증 오류 분류
3. Embedder
   - 메시지 content에 대해 dense embedding 생성
4. Neo4j Loader
   - 노드/관계 upsert 및 인덱스 초기화
5. Search API
   - metadata filter + vector/fulltext + graph expansion 제공
6. Internal UI
   - 검색 화면, 인사이트 화면 제공

### 3.2 Hybrid 전략 해석

본 MVP에서의 `Hybrid`는 다음 두 층을 의미한다.

- 운영 검색층: 메타데이터 필터 + dense vector retrieval + fulltext retrieval + graph expansion
- 인사이트 확장층: 집계 카드와 대표 메시지 샘플 중심의 구조화 인사이트

대규모 엔터티 추출, 커뮤니티 탐지, LLM 요약 그래프는 현재 구현 범위에 포함하지 않고 후속 확장으로 둔다.

## 4. 입력 데이터 스펙

### 4.1 Raw line grammar

각 메시지는 아래 문법을 따른다.

`[` `date` `, ` `time` `, ` `room` `, ` `content` `, ` `user` `]`

예시:

```text
[2024-01-05, 07:59:12, 프로젝트C, 서버 배포 380차 완료했습니다, 박소율]
[2024-01-01, 06:53:58, 신규사업TF, 장애 대응 107차 완료, 모니터링 중, 장다은]
```

두 번째 예시처럼 `content` 내부에도 `, `가 포함될 수 있다.

### 4.2 파서 규칙

아래 규칙을 고정 스펙으로 사용한다.

```python
parts = body.split(", ")
date = parts[0]
time = parts[1]
room = parts[2]
user = parts[-1]
content = ", ".join(parts[3:-1])
```

파서는 반드시 다음 검사를 수행한다.

- `[` 와 `]` 존재 여부
- 분리 후 최소 파트 수 5 이상 여부
- `date`가 `YYYY-MM-DD` 형식인지 여부
- `time`이 `HH:MM:SS` 형식인지 여부
- `room`, `content`, `user`가 비어 있지 않은지 여부

### 4.3 실패 처리 규칙

- bracket 누락: `ParseError.BRACKET_MISMATCH`
- 파트 수 부족: `ParseError.INSUFFICIENT_FIELDS`
- 날짜 형식 오류: `ParseError.INVALID_DATE`
- 시간 형식 오류: `ParseError.INVALID_TIME`
- 공란 필드: `ParseError.EMPTY_FIELD`

실패 레코드는 즉시 중단하지 않고 아래 정책을 따른다.

- 에러 로그 파일에 `source_file`, `line_no`, `raw_text`, `error_code` 기록
- 적재 통계에 실패 건수 반영
- 실패율이 1% 이상이면 배치 실패로 간주

## 5. 정규화 데이터 모델

### 5.1 애플리케이션 레코드 타입

```python
class ChatMessageRecord(TypedDict):
    message_id: str
    source_file: str
    line_no: int
    raw_text: str
    date: str
    time: str
    occurred_at: str
    room_name: str
    user_name: str
    content: str
    embedding_status: str
```

### 5.2 필드 규칙

- `message_id`: `sha1(source_file + ":" + line_no + ":" + raw_text)`로 생성
- `occurred_at`: `date + "T" + time`을 UTC naive ISO 8601 문자열로 저장
- `embedding_status`: `pending`, `completed`, `failed` 중 하나
- `source_file`: 절대 경로를 canonicalize해서 저장

## 6. Neo4j 그래프 스키마

### 6.1 노드

```text
(:Message {
  message_id,
  occurred_at,
  date,
  time,
  content,
  source_file,
  line_no,
  embedding,
  embedding_status
})

(:User {name})
(:Room {name})
(:Date {date})
```

### 6.2 관계

```text
(:User)-[:SENT]->(:Message)
(:Message)-[:IN_ROOM]->(:Room)
(:Message)-[:ON_DATE]->(:Date)
(:Message)-[:PREV_IN_ROOM]->(:Message)
```

### 6.3 관계 생성 규칙

- `SENT`: 메시지 작성자와 메시지 연결
- `IN_ROOM`: 메시지와 채팅방 연결
- `ON_DATE`: 메시지와 날짜 연결
- `PREV_IN_ROOM`: 같은 방에서 `occurred_at` 기준 직전 메시지와 연결

`PREV_IN_ROOM` 생성 알고리즘:

1. 방별로 메시지를 `occurred_at`, `line_no` 순으로 정렬
2. 인접한 두 메시지 `(prev, curr)` 사이에 `(curr)-[:PREV_IN_ROOM]->(prev)` 생성

## 7. 인덱스와 제약조건

### 7.1 제약조건

```cypher
CREATE CONSTRAINT message_id_unique IF NOT EXISTS
FOR (m:Message) REQUIRE m.message_id IS UNIQUE;

CREATE CONSTRAINT user_name_unique IF NOT EXISTS
FOR (u:User) REQUIRE u.name IS UNIQUE;

CREATE CONSTRAINT room_name_unique IF NOT EXISTS
FOR (r:Room) REQUIRE r.name IS UNIQUE;

CREATE CONSTRAINT date_unique IF NOT EXISTS
FOR (d:Date) REQUIRE d.date IS UNIQUE;
```

### 7.2 인덱스

```cypher
CREATE VECTOR INDEX message_embedding_index IF NOT EXISTS
FOR (m:Message) ON (m.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
  }
};

CREATE FULLTEXT INDEX message_content_fulltext IF NOT EXISTS
FOR (m:Message) ON EACH [m.content];

CREATE INDEX message_date_idx IF NOT EXISTS
FOR (m:Message) ON (m.date);

CREATE INDEX message_occurred_at_idx IF NOT EXISTS
FOR (m:Message) ON (m.occurred_at);
```

## 8. 임베딩 전략

### 8.1 임베딩 대상

- 기본 대상은 `Message.content`
- `room_name`, `user_name`, `date`는 메타데이터 필터로만 사용

### 8.2 임베딩 생성 규칙

- 배치 크기: 128
- 실패 시 최대 3회 재시도
- 임베딩 생성 실패 시 `embedding_status = failed`로 기록
- 검색 대상에서는 `embedding_status = completed`만 사용

### 8.3 비포함 항목

- sparse embedding
- ColBERT retrieval
- 문장 분리 기반 chunking

## 9. 적재 파이프라인

### 9.1 배치 순서

1. 입력 파일 스캔
2. 라인별 파싱 및 정규화
3. 임베딩 생성
4. Neo4j 제약조건 및 인덱스 확인
5. 노드/관계 upsert
6. 방별 `PREV_IN_ROOM` 관계 생성
7. 적재 결과 리포트 출력

### 9.2 업서트 원칙

- 동일 `message_id`는 중복 삽입하지 않는다.
- 사용자/방/날짜는 `MERGE`로 upsert 한다.
- 메시지 내용 변경은 허용하지 않는다. 동일 `message_id`의 기존 레코드와 충돌하면 에러로 기록한다.

### 9.3 적재 결과 리포트

배치 종료 후 아래 메트릭을 반드시 출력한다.

- 총 입력 건수
- 정상 적재 건수
- 파싱 실패 건수
- 임베딩 실패 건수
- 생성된 사용자 수
- 생성된 채팅방 수
- 생성된 날짜 수
- 생성된 `PREV_IN_ROOM` 관계 수

## 10. 검색 아키텍처

### 10.1 검색 전략

검색 전략은 아래 순서로 고정한다.

`metadata filter -> vector retrieval + fulltext retrieval -> RRF fusion -> graph expansion`

### 10.2 metadata filter

입력 조건:

- `date_from`
- `date_to`
- `rooms[]`
- `users[]`

필터 규칙:

- 날짜는 inclusive range
- `rooms[]`, `users[]`는 다중 OR
- 조건이 없는 필드는 필터에서 제외
- 모든 필터는 retrieval 이전에 후보군 축소에 사용

### 10.3 vector retrieval

- 쿼리 문자열을 `BAAI/bge-m3` dense vector로 임베딩
- 필터된 후보군에 대해 cosine similarity 검색
- vector top-k 기본값: 20

### 10.4 fulltext retrieval

- Neo4j fulltext index 사용
- 필터된 후보군에 대한 keyword relevance 확보
- fulltext top-k 기본값: 20

### 10.5 RRF fusion

최종 랭킹은 Reciprocal Rank Fusion으로 계산한다.

```text
score = 1 / (k + rank_vector) + 1 / (k + rank_fulltext)
```

- RRF 상수 `k = 60`
- 양쪽 결과 중 하나에만 존재하는 문서도 포함 가능
- 최종 결과 기본값: top 10

### 10.6 graph expansion

최종 검색 결과 각 메시지에 대해 다음 맥락만 붙인다.

- 같은 방의 이전 메시지 2건
- 같은 방의 다음 메시지 2건
- 같은 사용자의 최근 메시지 3건
- 같은 날짜, 같은 방의 대표 메시지 최대 5건

엔터티 추출 기반 그래프 확장은 MVP에서 제외한다.

## 11. API 명세

### 11.1 공통 규칙

- API prefix: `/api/v1`
- 응답 형식: JSON
- 시간 문자열은 ISO 8601 문자열 사용

### 11.2 `POST /api/v1/search/messages`

요청 바디:

```json
{
  "query": "배포 관련 내용",
  "date_from": "2024-01-01",
  "date_to": "2024-01-31",
  "rooms": ["프로젝트C", "개발팀"],
  "users": ["박소율"],
  "top_k": 10
}
```

응답 바디:

```json
{
  "query": "배포 관련 내용",
  "applied_filters": {
    "date_from": "2024-01-01",
    "date_to": "2024-01-31",
    "rooms": ["프로젝트C", "개발팀"],
    "users": ["박소율"]
  },
  "total_hits": 10,
  "results": [
    {
      "message_id": "sha1...",
      "occurred_at": "2024-01-05T07:59:12",
      "date": "2024-01-05",
      "time": "07:59:12",
      "room_name": "프로젝트C",
      "user_name": "박소율",
      "content": "서버 배포 380차 완료했습니다",
      "scores": {
        "vector": 0.82,
        "fulltext": 7.91,
        "rrf": 0.031
      },
      "context": {
        "previous_in_room": [],
        "next_in_room": [],
        "recent_by_user": [],
        "same_day_same_room_samples": []
      }
    }
  ]
}
```

입력 검증 규칙:

- `query`는 빈 문자열 불가
- `top_k` 기본값 10, 최대 50
- `date_from > date_to` 금지

### 11.3 `GET /api/v1/messages/{message_id}`

기능:

- 단일 메시지 상세 조회
- 인접 메시지와 메타데이터 포함

### 11.4 `GET /api/v1/insights/overview`

쿼리 파라미터:

- `date_from`
- `date_to`
- `rooms[]` optional
- `users[]` optional

응답 포함 항목:

- 기간별 메시지 수
- 상위 채팅방
- 상위 사용자
- 운영 키워드 샘플 메시지

### 11.5 `GET /health`

반환 항목:

- API 상태
- Neo4j 연결 상태
- 마지막 적재 시각
- 메시지 총 수

## 12. UI 명세

### 12.1 검색 화면

구성 요소:

- 검색 입력창
- 날짜 범위 필터
- 채팅방 multi-select
- 사용자 multi-select
- 검색 결과 리스트
- 상세 패널

결과 리스트 각 행 표시:

- 메시지 내용
- 날짜/시간
- 채팅방
- 사용자
- 랭킹 점수 요약

상세 패널 표시:

- 원문 메시지
- 메타데이터
- 같은 방 전후 메시지
- 같은 사용자 최근 메시지

### 12.2 인사이트 화면

구성 요소:

- 기간 필터
- 상위 채팅방 카드
- 상위 사용자 카드
- 기간별 메시지 수 차트
- 운영 키워드 대표 메시지 표

### 12.3 UI 제약

- 단일 FastAPI 앱에서 SSR 제공
- 클라이언트 상태 관리는 최소화
- 복잡한 SPA 프레임워크는 도입하지 않음

## 13. 저장소 구조 제안

```text
app/
  api/
  services/
  repositories/
  models/
  templates/
  static/
scripts/
  ingest_chat_logs.py
tests/
  unit/
  integration/
```

## 14. 테스트 전략

### 14.1 단위 테스트

파서 테스트:

- 정상 라인
- 본문 내부 쉼표 포함 라인
- 빈 라인
- bracket 누락
- 필드 부족
- 잘못된 날짜/시간

정규화 테스트:

- `message_id` 생성 일관성
- `occurred_at` 생성 규칙
- 공란 필드 검증

### 14.2 통합 테스트

`data/chat_logs_100.txt` 기준:

- 메시지 100건 적재
- 고유 사용자 수 검증
- 고유 채팅방 수 검증
- 고유 날짜 수 검증
- `SENT`, `IN_ROOM`, `ON_DATE` 관계 수 검증
- 방별 `PREV_IN_ROOM` 관계 생성 검증

### 14.3 검색 테스트

- 날짜 필터만 적용
- 채팅방 필터만 적용
- 사용자 필터만 적용
- 날짜 + 방 + 사용자 조합 필터
- 키워드 검색
- 의미 검색
- 그래프 컨텍스트 확장 검증

### 14.4 인사이트 테스트

- 기간별 방 카운트
- 사용자 활동 랭킹
- 특정 키워드 대표 메시지 샘플 추출

### 14.5 성능 테스트

- 10,000 메시지에서 필터 포함 검색 p95 2초 이하
- 10,000 메시지에서 인사이트 집계 p95 3초 이하

### 14.6 UI 스모크 테스트

- 검색 입력
- 필터 적용
- 결과 렌더링
- 상세 패널 열기
- 인사이트 카드 렌더링

## 15. 운영 및 관측성

### 15.1 로그

- 파싱 실패 로그
- 임베딩 실패 로그
- 검색 요청 로그
- 적재 배치 결과 로그

### 15.2 메트릭

- 총 메시지 수
- 적재 성공/실패 수
- API 응답시간
- 검색 질의 건수
- 인사이트 질의 건수

## 16. 보안 및 제한사항

- 현재 데이터에는 권한 정보가 없으므로 ACL은 설계 메모만 남긴다.
- 내부용 서비스로 가정하며 외부 노출을 고려한 인증 체계는 MVP 범위에서 제외한다.
- 개인정보 마스킹은 현재 범위에 포함하지 않는다.

## 17. 구현 우선순위

1. 파서 및 정규화
2. Neo4j 스키마 및 적재기
3. 임베딩 파이프라인
4. 검색 API
5. 인사이트 API
6. 검색 UI
7. 인사이트 UI
8. 테스트 및 성능 검증

## 18. 후속 확장

- 문서/티켓 ingestion 추가
- KSS 기반 긴 문서 sentence chunking
- sparse/ColBERT retrieval 추가
- LLM 기반 설명형 요약
- ACL 및 감사 로그
- Slack/Kakao/이슈 트래커 멀티소스 커넥터

## 19. 참고 자료

- 로컬 기획 기준: `docs/init_guide.md`
- 로컬 입력 데이터: `data/chat_logs.txt`, `data/chat_logs_100.txt`
- Neo4j GraphRAG for Python: https://neo4j.com/docs/neo4j-graphrag-python/current/index.html
- BAAI/bge-m3 모델 카드: https://huggingface.co/BAAI/bge-m3
- KSS 저장소: https://github.com/hyunwoongko/kss
