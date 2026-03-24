신규 구축에는 오히려 그 혼합형이 더 현실적입니다.

제가 추천하는 건 **“2층 구조 혼합형”**입니다.

1층은 Lightweight GraphRAG로 갑니다.
여기에는 Message / User / Room / Document / Ticket / Date 같은 업무 원본 객체 그래프만 넣습니다. 그리고 검색은 날짜·권한·채널 필터 → 벡터 검색 → 그래프 확장으로 처리합니다. Neo4j의 공식 neo4j-graphrag는 현재 유지보수되는 1st-party 패키지이고, 벡터 검색과 VectorCypherRetriever처럼 벡터 검색 뒤 Cypher로 그래프 확장하는 패턴을 공식적으로 지원합니다.

2층은 Microsoft GraphRAG 스타일 요약 그래프를 일부만 얹습니다.
여기서는 모든 데이터에 대해 대형 KG를 만들지 말고, 중요 문서 묶음이나 특정 프로젝트 범위에만 엔터티·관계 추출, 커뮤니티 탐지, 요약을 추가합니다. Neo4j 쪽의 ms-graphrag-neo4j는 Microsoft GraphRAG 접근을 Neo4j에 옮긴 구현이지만, 저장소 자체가 experimental이고 큰 그래프에서 최적화가 부족하다고 명시하고 있으므로, 메인 엔진이 아니라 선택적 상위 계층으로 쓰는 게 맞습니다.

즉 구조는 이렇게 보면 됩니다.

A. 운영 검색층 = Lightweight GraphRAG

원본 메시지/문서/채널/사용자 중심 그래프
KSS로 한국어 문장 분리
BGE-M3로 임베딩
날짜/권한/채널 필터 후 벡터 검색
찾은 노드 기준으로 1~2 hop 그래프 확장

B. 요약/인사이트층 = Microsoft GraphRAG식 보조 그래프

특정 범위 문서만 엔터티/관계 추출
커뮤니티 탐지
커뮤니티/엔터티 요약 생성
“이 프로젝트에서 핵심 주제 흐름이 뭐야?” 같은 질문에만 사용

이 혼합형의 장점은 분명합니다.
일상 질의는 가볍고 빠르게,
복잡한 전역 요약은 더 깊게 처리할 수 있습니다.
반대로 처음부터 Microsoft GraphRAG식 전체 파이프라인으로 가면 인덱싱 비용과 운영 복잡도가 커집니다. Microsoft GraphRAG 공식 저장소도 인덱싱 비용이 크고 작은 데이터로 먼저 시작하라고 안내합니다.

실전 추천은 이렇게 잡으면 됩니다.

추천 혼합안

기본 저장소: Neo4j
기본 검색: neo4j-graphrag
한국어 청킹: KSS
임베딩: BGE-M3
확장 요약: ms-graphrag-neo4j를 프로젝트 단위 배치 작업으로만 사용

한 줄로 정리하면:

운영용 검색은 Lightweight GraphRAG로 만들고, Microsoft GraphRAG → Neo4j 버전은 “전역 요약/커뮤니티 분석” 전용 보조 계층으로 섞는 방식이 가장 안전합니다.