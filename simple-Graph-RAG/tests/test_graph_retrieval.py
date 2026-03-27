"""Phase 1 tests: entity-seeded graph retrieval and entity overlap scoring."""
from __future__ import annotations

import sys
import types
from datetime import date, time

import pytest

if "psycopg_pool" not in sys.modules:
    psycopg_pool = types.ModuleType("psycopg_pool")
    psycopg_pool.ConnectionPool = object
    sys.modules["psycopg_pool"] = psycopg_pool

from app.schemas import QueryAnalysis, QueryFilters, RetrievedChunk
from app.services.retrieval import RetrievalService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(
    chunk_id: str,
    *,
    vector_score: float = 0.0,
    entities: list[str] | None = None,
    retrieval_source: str = "vector",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        channel="개발팀",
        user_name="김민수",
        message_date=date(2025, 3, 20),
        message_time=time(14, 30),
        access_scopes=["public"],
        chunk_text="테스트 청크",
        metadata={"entities": entities or []},
        vector_score=vector_score,
        retrieval_source=retrieval_source,
    )


def _make_analysis(
    entities: list[str] | None = None,
    intent: str = "search",
) -> QueryAnalysis:
    return QueryAnalysis(
        original_question="test",
        clean_question="test",
        intent=intent,
        filters=QueryFilters(access_scopes=["public"]),
        entities=entities or [],
    )


# ---------------------------------------------------------------------------
# _entity_overlap_score unit tests
# ---------------------------------------------------------------------------

class TestEntityOverlapScore:
    def test_no_query_entities_returns_zero(self) -> None:
        chunk = _make_chunk("c1", entities=["API", "서버"])
        analysis = _make_analysis(entities=[])
        assert RetrievalService._entity_overlap_score(chunk, analysis) == 0.0

    def test_no_chunk_entities_returns_zero(self) -> None:
        chunk = _make_chunk("c1", entities=[])
        analysis = _make_analysis(entities=["API"])
        assert RetrievalService._entity_overlap_score(chunk, analysis) == 0.0

    def test_full_overlap(self) -> None:
        chunk = _make_chunk("c1", entities=["API", "서버", "배포"])
        analysis = _make_analysis(entities=["API", "서버"])
        score = RetrievalService._entity_overlap_score(chunk, analysis)
        assert score == 1.0

    def test_partial_overlap(self) -> None:
        chunk = _make_chunk("c1", entities=["API", "모니터링"])
        analysis = _make_analysis(entities=["API", "서버", "배포"])
        score = RetrievalService._entity_overlap_score(chunk, analysis)
        assert abs(score - 1.0 / 3.0) < 1e-6

    def test_no_overlap(self) -> None:
        chunk = _make_chunk("c1", entities=["Redis", "캐시"])
        analysis = _make_analysis(entities=["API", "서버"])
        assert RetrievalService._entity_overlap_score(chunk, analysis) == 0.0


# ---------------------------------------------------------------------------
# _rank_chunks with graph_seeded_chunks
# ---------------------------------------------------------------------------

class TestRankWithGraphSeeded:
    def test_graph_seeded_chunks_included_in_ranking(self) -> None:
        """Graph-seeded chunks should appear in ranked results."""
        seed = _make_chunk("c1", vector_score=0.8, entities=["API"])
        graph_seed = _make_chunk("c2", vector_score=0.0, entities=["API", "서버"])
        graph_seed.retrieval_source = "graph_entity"
        analysis = _make_analysis(entities=["API", "서버"])

        from app.config import Settings
        from app.services.query_analyzer import QueryAnalyzer

        service = RetrievalService(
            settings=Settings(),
            postgres=object(),
            neo4j=object(),
            embedding_provider=object(),
            codex_proxy=object(),
            query_analyzer=QueryAnalyzer(),
        )

        ranked = service._rank_chunks(
            seed_chunks=[seed],
            expanded_chunks=[],
            expansions={},
            analysis=analysis,
            graph_seeded_chunks=[graph_seed],
        )

        chunk_ids = [c.chunk_id for c in ranked]
        assert "c1" in chunk_ids
        assert "c2" in chunk_ids

    def test_graph_seeded_not_duplicated_with_seed(self) -> None:
        """If a chunk appears in both seed and graph-seeded, it should not be duplicated."""
        seed = _make_chunk("c1", vector_score=0.8, entities=["API"])
        graph_seed = _make_chunk("c1", vector_score=0.0, entities=["API"])
        graph_seed.retrieval_source = "graph_entity"
        analysis = _make_analysis(entities=["API"])

        from app.config import Settings
        from app.services.query_analyzer import QueryAnalyzer

        service = RetrievalService(
            settings=Settings(),
            postgres=object(),
            neo4j=object(),
            embedding_provider=object(),
            codex_proxy=object(),
            query_analyzer=QueryAnalyzer(),
        )

        ranked = service._rank_chunks(
            seed_chunks=[seed],
            expanded_chunks=[],
            expansions={},
            analysis=analysis,
            graph_seeded_chunks=[graph_seed],
        )

        assert len(ranked) == 1
        assert ranked[0].chunk_id == "c1"

    def test_empty_graph_seeded_preserves_original_behavior(self) -> None:
        """With no graph-seeded chunks, behavior should be identical to before."""
        seed = _make_chunk("c1", vector_score=0.8)
        analysis = _make_analysis(entities=[])

        from app.config import Settings
        from app.services.query_analyzer import QueryAnalyzer

        service = RetrievalService(
            settings=Settings(),
            postgres=object(),
            neo4j=object(),
            embedding_provider=object(),
            codex_proxy=object(),
            query_analyzer=QueryAnalyzer(),
        )

        ranked = service._rank_chunks(
            seed_chunks=[seed],
            expanded_chunks=[],
            expansions={},
            analysis=analysis,
            graph_seeded_chunks=[],
        )

        assert len(ranked) == 1
        assert ranked[0].chunk_id == "c1"
        assert ranked[0].entity_overlap_score == 0.0


# ---------------------------------------------------------------------------
# find_chunks_by_entities (FakeNeo4j)
# ---------------------------------------------------------------------------

class FakeNeo4jWithEntities:
    def __init__(self, chunk_map: dict[str, list[str]]) -> None:
        """chunk_map: entity_name -> list of chunk_ids that mention it."""
        self._chunk_map = chunk_map

    async def find_chunks_by_entities(
        self, entity_names: list[str], *, limit: int = 20
    ) -> list[str]:
        from collections import Counter
        hits: Counter[str] = Counter()
        for name in entity_names:
            for cid in self._chunk_map.get(name, []):
                hits[cid] += 1
        return [cid for cid, _ in hits.most_common(limit)]


@pytest.mark.asyncio
async def test_fake_neo4j_find_chunks_by_entities() -> None:
    neo4j = FakeNeo4jWithEntities({
        "API": ["c1", "c2", "c3"],
        "서버": ["c2", "c4"],
        "배포": ["c3"],
    })

    result = await neo4j.find_chunks_by_entities(["API", "서버"])
    assert result[0] == "c2"  # c2 has 2 hits (API + 서버)
    assert set(result) == {"c1", "c2", "c3", "c4"}


@pytest.mark.asyncio
async def test_fake_neo4j_empty_entities() -> None:
    neo4j = FakeNeo4jWithEntities({"API": ["c1"]})
    result = await neo4j.find_chunks_by_entities([])
    assert result == []
