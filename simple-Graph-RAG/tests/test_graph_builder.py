"""Phase 2 tests: enhanced entity extraction with normalization, typing, particle stripping."""
from __future__ import annotations

from app.services.graph_builder import GraphBuilder


builder = GraphBuilder()


class TestNormalization:
    def test_korean_particle_stripping(self) -> None:
        assert builder._normalize("서버를") == "서버"
        assert builder._normalize("배포는") == "배포"
        assert builder._normalize("API에서") == "API"  # mixed token: Korean particle stripped

    def test_english_lowercase(self) -> None:
        assert builder._normalize("Redis") == "redis"
        assert builder._normalize("API") == "api"

    def test_korean_no_particle(self) -> None:
        assert builder._normalize("모니터링") == "모니터링"


class TestClassification:
    def test_issue_number(self) -> None:
        assert builder._classify("#456") == "issue"
        assert builder._classify("#12") == "issue"

    def test_camelcase_system(self) -> None:
        assert builder._classify("RedisCache") == "system"

    def test_ascii_system(self) -> None:
        assert builder._classify("deploy") == "system"
        assert builder._classify("API") == "system"

    def test_korean_person_name(self) -> None:
        assert builder._classify("김민수") == "person"

    def test_korean_topic(self) -> None:
        assert builder._classify("모니터링") == "topic"
        assert builder._classify("서버배포") == "topic"


class TestExtractTypedEntities:
    def test_basic_extraction(self) -> None:
        text = "API 서버 배포 완료합니다. #456 Redis 캐시 추가했습니다."
        entities = builder.extract_typed_entities(text)
        names = [e["name"] for e in entities]
        assert "api" in names
        assert "redis" in names
        assert "#456" in names

    def test_max_16_entities(self) -> None:
        tokens = " ".join(f"엔티티{i}" for i in range(30))
        entities = builder.extract_typed_entities(tokens)
        assert len(entities) <= 16

    def test_stopwords_filtered(self) -> None:
        text = "확인 부탁드립니다 오늘 정리 합니다"
        entities = builder.extract_typed_entities(text)
        names = [e["name"] for e in entities]
        assert "확인" not in names
        assert "부탁드립니다" not in names

    def test_particle_stripped_in_entities(self) -> None:
        text = "서버를 배포는 모니터링에 문제가"
        entities = builder.extract_typed_entities(text)
        names = [e["name"] for e in entities]
        assert "서버" in names
        assert "배포" in names
        assert "모니터링" in names
        # particles should be stripped
        assert "서버를" not in names
        assert "배포는" not in names

    def test_backward_compatible_extract_entities(self) -> None:
        text = "API 서버 배포"
        flat = builder.extract_entities(text)
        typed = builder.extract_typed_entities(text)
        assert flat == [e["name"] for e in typed]

    def test_deduplication(self) -> None:
        text = "API 서버 API 서버 API"
        entities = builder.extract_typed_entities(text)
        names = [e["name"] for e in entities]
        assert names.count("api") == 1
        assert names.count("서버") == 1
