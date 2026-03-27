"""Tests for the full answer_query() pipeline and scoring functions."""
from __future__ import annotations

import sys
import types
from datetime import date, time, timedelta

import pytest

if "psycopg_pool" not in sys.modules:
    psycopg_pool = types.ModuleType("psycopg_pool")
    psycopg_pool.ConnectionPool = object
    sys.modules["psycopg_pool"] = psycopg_pool

from app.config import Settings
from app.schemas import (
    GraphExpansion,
    QueryAnalysis,
    QueryFilters,
    QueryResponse,
    RetrievedChunk,
)
from app.services.query_analyzer import QueryAnalyzer
from app.services.retrieval import RetrievalService


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def _make_chunk(
    chunk_id: str,
    *,
    vector_score: float = 0.8,
    channel: str = "general",
    user_name: str = "민수",
    message_date: date | None = None,
    entities: list[str] | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        channel=channel,
        user_name=user_name,
        message_date=message_date or date(2025, 3, 20),
        message_time=time(14, 30),
        access_scopes=["public"],
        chunk_text=f"general {user_name}: 테스트 청크 {chunk_id}",
        metadata={"entities": entities or [], "original_lines": [f"[테스트 라인 {chunk_id}]"]},
        vector_score=vector_score,
    )


class FakePostgres:
    def __init__(self, seed_chunks: list[RetrievedChunk] | None = None) -> None:
        self._seed_chunks = seed_chunks or []
        self._chunks_by_id: dict[str, RetrievedChunk] = {}

    def set_chunks_by_id(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks_by_id = {c.chunk_id: c for c in chunks}

    async def list_channels(self, limit: int = 200) -> list[str]:
        return ["general", "dev"]

    async def list_users(self, limit: int = 200) -> list[str]:
        return ["민수", "지현"]

    async def search_chunks(self, query_embedding, filters, top_k) -> list[RetrievedChunk]:
        return self._seed_chunks[:top_k]

    async def get_chunks_by_ids(self, chunk_ids) -> list[RetrievedChunk]:
        ids = set(chunk_ids)
        return [c for cid, c in self._chunks_by_id.items() if cid in ids]


class FakeNeo4j:
    def __init__(
        self,
        entity_chunk_ids: list[str] | None = None,
        expansions: dict[str, GraphExpansion] | None = None,
    ) -> None:
        self._entity_chunk_ids = entity_chunk_ids or []
        self._expansions = expansions or {}

    async def find_chunks_by_entities(self, entity_names, *, limit=20) -> list[str]:
        return self._entity_chunk_ids[:limit]

    async def expand_from_seed_chunks(self, chunk_ids, *, next_window=2) -> dict[str, GraphExpansion]:
        return {k: v for k, v in self._expansions.items() if k in chunk_ids}

    async def expand_via_entity_cooccurrence(self, chunk_ids, *, limit=20) -> list[dict]:
        return []

    async def expand_via_same_author(self, chunk_ids, *, limit=10) -> list[dict]:
        return []

    async def extract_subgraph(self, chunk_ids) -> list[dict]:
        return []

    async def find_communities_for_entities(self, entity_names) -> list[dict]:
        return []

    async def get_entity_cooccurrence_network(self, entity_names, *, limit=50) -> list[dict]:
        return []

    async def get_entity_mention_counts(self, entity_names) -> dict[str, int]:
        return {}


class FakeEmbeddingProvider:
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 10 for _ in texts]


class FakeCodexProxy:
    def __init__(self, response_text: str = "LLM 응답입니다.") -> None:
        self._response_text = response_text
        self.calls: list[dict] = []

    async def generate(self, *, system_prompt, user_prompt, metadata=None):
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})

        class _Response:
            text = self._response_text
        return _Response()


class FakeCodexProxyFailing:
    async def generate(self, *, system_prompt, user_prompt, metadata=None):
        from app.adapters.codex_proxy import CodexProxyError
        raise CodexProxyError("timeout", "llm_unavailable")


def _build_service(
    *,
    seed_chunks: list[RetrievedChunk] | None = None,
    extra_chunks: list[RetrievedChunk] | None = None,
    entity_chunk_ids: list[str] | None = None,
    expansions: dict[str, GraphExpansion] | None = None,
    codex_proxy=None,
    settings: Settings | None = None,
) -> RetrievalService:
    postgres = FakePostgres(seed_chunks)
    if extra_chunks:
        postgres.set_chunks_by_id(extra_chunks)
    return RetrievalService(
        settings=settings or Settings(),
        postgres=postgres,
        neo4j=FakeNeo4j(entity_chunk_ids, expansions),
        embedding_provider=FakeEmbeddingProvider(),
        codex_proxy=codex_proxy or FakeCodexProxy(),
        query_analyzer=QueryAnalyzer(),
    )


# ---------------------------------------------------------------------------
# answer_query() full pipeline tests
# ---------------------------------------------------------------------------

class TestAnswerQueryPipeline:
    @pytest.mark.asyncio
    async def test_happy_path_returns_llm_answer(self) -> None:
        """Seed chunks found → LLM generates answer → answer_mode='llm'."""
        seed = _make_chunk("c1", vector_score=0.9, entities=["API"])
        service = _build_service(seed_chunks=[seed])

        response = await service.answer_query(
            question="API 관련 내용 알려줘",
            access_scopes=["public"],
            request_user="alice",
        )

        assert isinstance(response, QueryResponse)
        assert response.answer_mode == "llm"
        assert response.answer == "LLM 응답입니다."
        assert len(response.sources) >= 1
        assert response.sources[0].chunk_id == "c1"

    @pytest.mark.asyncio
    async def test_no_chunks_returns_fallback(self) -> None:
        """No seed chunks and no graph chunks → fallback message."""
        service = _build_service(seed_chunks=[])

        response = await service.answer_query(
            question="존재하지 않는 내용",
            access_scopes=["public"],
            request_user=None,
        )

        assert response.answer_mode == "fallback_sources_only"
        assert "찾지 못했습니다" in response.answer
        assert response.sources == []

    @pytest.mark.asyncio
    async def test_llm_failure_returns_fallback_with_sources(self) -> None:
        """LLM fails → fallback answer built from sources, answer_mode='fallback_sources_only'."""
        seed = _make_chunk("c1", vector_score=0.85)
        service = _build_service(
            seed_chunks=[seed],
            codex_proxy=FakeCodexProxyFailing(),
        )

        response = await service.answer_query(
            question="서버 배포 관련",
            access_scopes=["public"],
            request_user=None,
        )

        assert response.answer_mode == "fallback_sources_only"
        assert len(response.sources) >= 1
        assert response.retrieval_strategy == "filter_pgvector_graph_hybrid"

    @pytest.mark.asyncio
    async def test_graph_entity_seed_brings_additional_chunks(self) -> None:
        """Graph entity seeding adds chunks not found by vector search."""
        seed = _make_chunk("c1", vector_score=0.9, entities=["API"])
        graph_extra = _make_chunk("c2", vector_score=0.0, entities=["API", "서버"])

        service = _build_service(
            seed_chunks=[seed],
            extra_chunks=[graph_extra],
            entity_chunk_ids=["c2"],
        )

        response = await service.answer_query(
            question="API 서버 관련 내용",
            access_scopes=["public"],
            request_user=None,
        )

        source_ids = [s.chunk_id for s in response.sources]
        assert "c1" in source_ids
        assert "c2" in source_ids

    @pytest.mark.asyncio
    async def test_debug_mode_includes_debug_data(self) -> None:
        """debug=True → response.debug is populated."""
        seed = _make_chunk("c1", vector_score=0.85, entities=["API"])
        service = _build_service(seed_chunks=[seed])

        response = await service.answer_query(
            question="API 관련",
            access_scopes=["public"],
            request_user=None,
            debug=True,
        )

        assert response.debug is not None
        assert len(response.debug.timing) > 0
        assert response.debug.total_time_ms > 0
        assert len(response.debug.score_breakdowns) >= 1

    @pytest.mark.asyncio
    async def test_debug_false_excludes_debug_data(self) -> None:
        """debug=False → response.debug is None."""
        seed = _make_chunk("c1", vector_score=0.85)
        service = _build_service(seed_chunks=[seed])

        response = await service.answer_query(
            question="테스트",
            access_scopes=["public"],
            request_user=None,
            debug=False,
        )

        assert response.debug is None

    @pytest.mark.asyncio
    async def test_graph_expansion_adds_neighbors(self) -> None:
        """Graph expansion attaches neighbor info to seed chunks."""
        seed = _make_chunk("c1", vector_score=0.9, entities=["API"])
        expansion = GraphExpansion(
            chunk_id="c1",
            graph_neighbors=["민수", "general", "API"],
            expanded_chunk_ids=["c3"],
        )
        expanded = _make_chunk("c3", vector_score=0.0)

        service = _build_service(
            seed_chunks=[seed],
            extra_chunks=[expanded],
            expansions={"c1": expansion},
        )

        response = await service.answer_query(
            question="API 관련",
            access_scopes=["public"],
            request_user=None,
        )

        c1_source = next(s for s in response.sources if s.chunk_id == "c1")
        assert len(c1_source.graph_neighbors) > 0


# ---------------------------------------------------------------------------
# Scoring function unit tests
# ---------------------------------------------------------------------------

class TestMetadataScore:
    def _service(self) -> RetrievalService:
        return _build_service(seed_chunks=[])

    def test_full_match_channel_user_date(self) -> None:
        service = self._service()
        chunk = _make_chunk("c1", channel="general", user_name="민수", message_date=date(2025, 3, 20))
        analysis = QueryAnalysis(
            original_question="test",
            clean_question="test",
            filters=QueryFilters(
                channel="general",
                user_names=["민수"],
                date_from=date(2025, 3, 20),
                date_to=date(2025, 3, 20),
            ),
        )
        score = service._metadata_score(chunk, analysis)
        assert score == pytest.approx(1.0)

    def test_no_match_returns_zero(self) -> None:
        service = self._service()
        chunk = _make_chunk("c1", channel="dev", user_name="지현")
        analysis = QueryAnalysis(
            original_question="test",
            clean_question="test",
            filters=QueryFilters(channel="general", user_names=["민수"]),
        )
        score = service._metadata_score(chunk, analysis)
        assert score == 0.0

    def test_partial_match(self) -> None:
        service = self._service()
        chunk = _make_chunk("c1", channel="general", user_name="지현")
        analysis = QueryAnalysis(
            original_question="test",
            clean_question="test",
            filters=QueryFilters(channel="general", user_names=["민수"]),
        )
        score = service._metadata_score(chunk, analysis)
        assert score == pytest.approx(0.4)  # channel match only


class TestRecencyScore:
    def _service(self) -> RetrievalService:
        return _build_service(seed_chunks=[])

    def test_today_returns_one(self) -> None:
        score = self._service()._recency_score(date.today())
        assert score == pytest.approx(1.0)

    def test_30_days_ago_returns_half(self) -> None:
        score = self._service()._recency_score(date.today() - timedelta(days=30))
        assert score == pytest.approx(0.5)

    def test_older_is_lower(self) -> None:
        service = self._service()
        recent = service._recency_score(date.today() - timedelta(days=7))
        old = service._recency_score(date.today() - timedelta(days=90))
        assert recent > old


class TestCombinedScore:
    def test_weight_application(self) -> None:
        chunk = _make_chunk("c1")
        chunk.vector_score = 1.0
        chunk.graph_score = 0.0
        chunk.entity_score = 0.0
        chunk.metadata_score = 0.0
        chunk.recency_score = 0.0

        weights = (0.40, 0.15, 0.20, 0.15, 0.10)  # search default
        score = RetrievalService._combined_score(chunk, weights)
        assert score == pytest.approx(0.40)

    def test_all_ones_equals_weight_sum(self) -> None:
        chunk = _make_chunk("c1")
        chunk.vector_score = 1.0
        chunk.graph_score = 1.0
        chunk.entity_score = 1.0
        chunk.metadata_score = 1.0
        chunk.recency_score = 1.0

        weights = (0.40, 0.15, 0.20, 0.15, 0.10)
        score = RetrievalService._combined_score(chunk, weights)
        assert score == pytest.approx(1.0)


class TestGetWeights:
    def test_all_intents_sum_to_one(self) -> None:
        for intent in ("search", "summary", "timeline", "aggregate", "relationship"):
            analysis = QueryAnalysis(
                original_question="test", clean_question="test", intent=intent,
            )
            weights = RetrievalService._get_weights(analysis)
            assert sum(weights) == pytest.approx(1.0), f"{intent} weights don't sum to 1.0"

    def test_relationship_boosts_graph(self) -> None:
        search_weights = RetrievalService._get_weights(
            QueryAnalysis(original_question="t", clean_question="t", intent="search")
        )
        rel_weights = RetrievalService._get_weights(
            QueryAnalysis(original_question="t", clean_question="t", intent="relationship")
        )
        # relationship intent should have higher graph weight than search
        assert rel_weights[1] > search_weights[1]
