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
    QueryRequestFilters,
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
    document_type: str = "chat",
    vector_score: float = 0.8,
    channel: str = "general",
    user_name: str = "민수",
    message_date: date | None = None,
    entities: list[str] | None = None,
    chunk_text: str | None = None,
    metadata: dict | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        document_type=document_type,
        channel=channel,
        user_name=user_name,
        message_date=message_date or date(2025, 3, 20),
        message_time=time(14, 30),
        access_scopes=["public"],
        chunk_text=chunk_text or f"general {user_name}: 테스트 청크 {chunk_id}",
        metadata=metadata or {"entities": entities or [], "original_lines": [f"[테스트 라인 {chunk_id}]"]},
        vector_score=vector_score,
    )


class FakePostgres:
    def __init__(self, seed_chunks: list[RetrievedChunk] | None = None) -> None:
        self._seed_chunks = seed_chunks or []
        self._chunks_by_id: dict[str, RetrievedChunk] = {
            chunk.chunk_id: chunk for chunk in self._seed_chunks
        }
        self.last_search_filters: QueryFilters | None = None
        self.latest_dates = {
            None: date(2026, 3, 20),
            "chat": date(2024, 3, 20),
            "issue": date(2026, 3, 20),
        }

    def set_chunks_by_id(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks_by_id.update({c.chunk_id: c for c in chunks})

    def _all_chunks(self) -> list[RetrievedChunk]:
        return list(self._chunks_by_id.values())

    async def list_channels(self, limit: int = 200, document_type: str | None = None) -> list[str]:
        if document_type == "chat":
            return ["general", "dev"]
        if document_type == "issue":
            return ["이슈데이터_10000건"]
        return ["general", "dev", "이슈데이터_10000건"]

    async def list_users(self, limit: int = 200, document_type: str | None = None) -> list[str]:
        if document_type == "chat":
            return ["민수", "지현"]
        if document_type == "issue":
            return ["Sujin", "Donghyun"]
        return ["민수", "지현", "Sujin", "Donghyun"]

    async def list_assignees(self, document_type: str = "issue", limit: int = 200) -> list[str]:
        return ["Sujin", "Donghyun"]

    async def list_statuses(self, document_type: str = "issue", limit: int = 200) -> list[str]:
        return ["보류", "완료", "진행", "재현중", "검증대기"]

    async def list_document_types(self, limit: int = 200) -> list[str]:
        return ["chat", "issue"]

    async def get_latest_event_date(self, document_type: str | None = None) -> date | None:
        return self.latest_dates.get(document_type, self.latest_dates[None])

    async def search_chunks(self, query_embedding, filters, top_k) -> list[RetrievedChunk]:
        self.last_search_filters = filters
        def matches(chunk: RetrievedChunk) -> bool:
            if filters.all_document_types and chunk.document_type not in filters.all_document_types:
                return False
            if filters.all_channels and chunk.channel not in filters.all_channels:
                return False
            if filters.user_names and chunk.user_name not in filters.user_names:
                return False
            if filters.assignees:
                assignee = chunk.metadata.get("assignee") or chunk.user_name
                if assignee not in filters.assignees:
                    return False
            if filters.statuses and chunk.metadata.get("status") not in filters.statuses:
                return False
            if filters.access_scopes and not (set(chunk.access_scopes) & set(filters.access_scopes)):
                return False
            if filters.date_from and chunk.message_date < filters.date_from:
                return False
            if filters.date_to and chunk.message_date > filters.date_to:
                return False
            return True

        matched = [chunk for chunk in self._all_chunks() if matches(chunk)]
        matched.sort(key=lambda chunk: chunk.vector_score, reverse=True)
        return matched[:top_k]

    async def get_chunks_by_ids(self, chunk_ids) -> list[RetrievedChunk]:
        ids = set(chunk_ids)
        return [c for cid, c in self._chunks_by_id.items() if cid in ids]

    async def summarize_filtered_results(self, filters, *, limit: int) -> dict:
        def matches(chunk: RetrievedChunk) -> bool:
            if filters.all_document_types and chunk.document_type not in filters.all_document_types:
                return False
            if filters.assignees:
                assignee = chunk.metadata.get("assignee") or chunk.user_name
                if assignee not in filters.assignees:
                    return False
            if filters.statuses and chunk.metadata.get("status") not in filters.statuses:
                return False
            if filters.date_from and chunk.message_date < filters.date_from:
                return False
            if filters.date_to and chunk.message_date > filters.date_to:
                return False
            if chunk.document_type == "issue" and chunk.metadata.get("chunk_kind") != "overview":
                return False
            return True

        matched = [chunk for chunk in self._seed_chunks if matches(chunk)]
        return {
            "matched_count": len(matched),
            "count_basis": "matching_records",
            "sample_chunks": matched[:limit],
        }


class FakeNeo4j:
    def __init__(
        self,
        entity_chunk_ids: list[str] | None = None,
        expansions: dict[str, GraphExpansion] | None = None,
    ) -> None:
        self._entity_chunk_ids = entity_chunk_ids or []
        self._expansions = expansions or {}
        self.entity_cooccurrence_calls: list[dict] = []
        self.same_author_calls: list[dict] = []

    async def find_chunks_by_entities(self, entity_names, *, limit=20) -> list[str]:
        return self._entity_chunk_ids[:limit]

    async def expand_from_seed_chunks(self, chunk_ids, *, next_window=2) -> dict[str, GraphExpansion]:
        return {k: v for k, v in self._expansions.items() if k in chunk_ids}

    async def expand_via_entity_cooccurrence(self, chunk_ids, *, entity_names, limit=20) -> list[dict]:
        self.entity_cooccurrence_calls.append({
            "chunk_ids": list(chunk_ids),
            "entity_names": list(entity_names),
            "limit": limit,
        })
        return []

    async def expand_via_same_author(self, chunk_ids, *, limit=10) -> list[dict]:
        self.same_author_calls.append({
            "chunk_ids": list(chunk_ids),
            "limit": limit,
        })
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
    async def test_exact_keyword_matches_outrank_irrelevant_recent_issue_chunks(self) -> None:
        issue_seed = _make_chunk(
            "issue-1",
            document_type="issue",
            vector_score=0.46,
            channel="이슈데이터_10000건",
            user_name="Jiho",
            message_date=date(2025, 10, 7),
            chunk_text="[이슈] 임베딩 버전 불일치로 유사도 분포 이상",
            metadata={"entities": ["임베딩", "버전"], "chunk_kind": "overview"},
        )
        chat_graph = _make_chunk(
            "chat-1",
            vector_score=0.0,
            channel="프로젝트C",
            user_name="박소율",
            message_date=date(2024, 1, 5),
            chunk_text="프로젝트C 박소율: 서버 배포 380차 완료했습니다",
            metadata={"entities": [], "original_lines": ["프로젝트C 박소율: 서버 배포 380차 완료했습니다"]},
        )

        service = _build_service(
            seed_chunks=[issue_seed],
            extra_chunks=[chat_graph],
            entity_chunk_ids=["chat-1"],
        )

        response = await service.answer_query(
            question="서버 배포 관련 내용 알려줘",
            access_scopes=["public"],
            request_user=None,
        )

        assert response.sources[0].chunk_id == "chat-1"
        assert response.sources[0].document_type == "chat"

    @pytest.mark.asyncio
    async def test_oom_alias_grounding_prefers_gpu_memory_issue(self) -> None:
        irrelevant_issue = _make_chunk(
            "issue-1",
            document_type="issue",
            vector_score=0.55,
            channel="이슈데이터_10000건",
            user_name="Jiho",
            chunk_text="[이슈] 임베딩 버전 불일치로 유사도 분포 이상",
            metadata={"entities": ["임베딩", "유사도"], "chunk_kind": "overview"},
        )
        gpu_issue = _make_chunk(
            "issue-2",
            document_type="issue",
            vector_score=0.42,
            channel="이슈데이터_10000건",
            user_name="Eunji",
            chunk_text="[이슈] 장문 요청 처리 중 GPU 메모리 부족 발생",
            metadata={"entities": ["GPU", "메모리 부족"], "chunk_kind": "overview"},
        )
        service = _build_service(seed_chunks=[irrelevant_issue, gpu_issue])

        response = await service.answer_query(
            question="OOM 발견 후 어떤 수정이 이뤄졌나",
            access_scopes=["public"],
            request_user=None,
            top_k=5,
        )

        assert response.answer_mode == "llm"
        assert response.sources[0].chunk_id == "issue-2"

    @pytest.mark.asyncio
    async def test_strict_error_grounding_filters_irrelevant_504_sources(self) -> None:
        irrelevant_issue = _make_chunk(
            "issue-1",
            document_type="issue",
            vector_score=0.7,
            channel="이슈데이터_10000건",
            user_name="Jiho",
            chunk_text="[이슈] 임베딩 버전 불일치로 유사도 분포 이상",
            metadata={"entities": ["임베딩", "유사도"], "chunk_kind": "overview"},
        )
        service = _build_service(seed_chunks=[irrelevant_issue])

        response = await service.answer_query(
            question="504 에러 관련 내용 알려줘",
            access_scopes=["public"],
            request_user=None,
            top_k=5,
        )

        assert response.answer_mode == "fallback_sources_only"
        assert response.sources == []

    @pytest.mark.asyncio
    async def test_exact_504_match_is_preferred_over_timeout_fallback(self) -> None:
        exact_issue = _make_chunk(
            "issue-504",
            document_type="issue",
            vector_score=0.41,
            channel="이슈데이터_10000건",
            user_name="Hyunwoo",
            chunk_text="[이슈] API Gateway 504 오류 발생",
            metadata={"entities": ["504", "gateway"], "chunk_kind": "overview"},
        )
        timeout_issue = _make_chunk(
            "issue-timeout",
            document_type="issue",
            vector_score=0.72,
            channel="이슈데이터_10000건",
            user_name="Yuna",
            chunk_text="[이슈] AnswerGen-v5 응답 지연 및 timeout 증가",
            metadata={"entities": ["timeout", "응답 지연"], "chunk_kind": "overview"},
        )
        service = _build_service(seed_chunks=[timeout_issue, exact_issue])

        response = await service.answer_query(
            question="504 에러 관련 내용 알려줘",
            access_scopes=["public"],
            request_user=None,
            top_k=5,
        )

        assert response.answer_mode == "llm"
        assert response.sources[0].chunk_id == "issue-504"

    @pytest.mark.asyncio
    async def test_related_chat_queries_keep_at_least_one_chat_source(self) -> None:
        issue_seeds = [
            _make_chunk(
                f"issue-{index}",
                document_type="issue",
                vector_score=0.7 - (index * 0.01),
                channel="이슈데이터_10000건",
                user_name="Eunji",
                message_date=date(2026, 3, 20),
                chunk_text="[이슈] 장문 요청 처리 중 GPU 메모리 부족 발생",
                metadata={"entities": ["GPU", "메모리 부족"], "chunk_kind": "overview"},
            )
            for index in range(5)
        ]
        chat_extra = _make_chunk(
            "chat-1",
            document_type="chat",
            vector_score=0.1,
            channel="프로젝트C",
            user_name="박소율",
            message_date=date(2026, 3, 19),
            chunk_text="프로젝트C 박소율: GPU 메모리 부족 대응 위해 배치 크기 제한 적용",
            metadata={"entities": ["GPU", "메모리 부족"], "original_lines": ["프로젝트C 박소율: GPU 메모리 부족 대응 위해 배치 크기 제한 적용"]},
        )
        service = _build_service(
            seed_chunks=issue_seeds,
            extra_chunks=[chat_extra],
        )

        response = await service.answer_query(
            question="최근 2주 GPU 메모리 이슈와 관련 대화 요약",
            access_scopes=["public"],
            request_user=None,
            top_k=5,
        )

        assert response.answer_mode == "llm"
        assert any(source.document_type == "chat" for source in response.sources)

    @pytest.mark.asyncio
    async def test_related_chat_queries_do_not_attach_out_of_range_chat_sources(self) -> None:
        issue_seeds = [
            _make_chunk(
                f"issue-{index}",
                document_type="issue",
                vector_score=0.7 - (index * 0.01),
                channel="이슈데이터_10000건",
                user_name="Eunji",
                message_date=date(2026, 3, 20),
                chunk_text="[이슈] 장문 요청 처리 중 GPU 메모리 부족 발생",
                metadata={"entities": ["GPU", "메모리 부족"], "chunk_kind": "overview"},
            )
            for index in range(5)
        ]
        old_chat = _make_chunk(
            "chat-1",
            document_type="chat",
            vector_score=0.3,
            channel="개발팀",
            user_name="박소율",
            message_date=date(2024, 1, 5),
            chunk_text="개발팀 박소율: GPU 메모리 부족 대응 위해 배치 크기 제한 적용",
            metadata={"entities": ["GPU", "메모리 부족"], "original_lines": ["개발팀 박소율: GPU 메모리 부족 대응 위해 배치 크기 제한 적용"]},
        )
        codex = FakeCodexProxy()
        service = _build_service(
            seed_chunks=issue_seeds,
            extra_chunks=[old_chat],
            codex_proxy=codex,
        )

        response = await service.answer_query(
            question="최근 2주 GPU 메모리 이슈와 관련 대화 요약",
            access_scopes=["public"],
            request_user=None,
            top_k=5,
        )

        assert response.answer_mode == "llm"
        assert all(source.document_type == "issue" for source in response.sources)
        prompt = codex.calls[0]["user_prompt"]
        assert "Chat Coverage:" in prompt
        assert "matching_chat_sources: 0" in prompt

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
    async def test_issue_prompt_includes_structured_metadata_for_llm(self) -> None:
        issue_chunk = _make_chunk(
            "issue-1",
            document_type="issue",
            vector_score=0.9,
            channel="이슈데이터_10000건",
            user_name="Sujin",
            message_date=date(2026, 3, 20),
            chunk_text="[이슈] 메타데이터 필터 적용 시 관련 문서 누락",
            metadata={
                "entities": ["메타데이터", "필터"],
                "chunk_kind": "overview",
                "issue_title": "메타데이터 필터 적용 시 관련 문서 누락",
                "assignee": "Sujin",
                "status": "완료",
            },
        )
        codex = FakeCodexProxy()
        service = _build_service(seed_chunks=[issue_chunk], codex_proxy=codex)

        response = await service.answer_query(
            question="Sujin의 완료된 이슈 요약",
            access_scopes=["public"],
            request_user=None,
        )

        assert response.answer_mode == "llm"
        prompt = codex.calls[0]["user_prompt"]
        assert "Issue Title: 메타데이터 필터 적용 시 관련 문서 누락" in prompt
        assert "Assignee: Sujin" in prompt
        assert "Status: 완료" in prompt

    @pytest.mark.asyncio
    async def test_aggregate_queries_use_database_summary_not_top_k_guess(self) -> None:
        aggregate_chunks = [
            _make_chunk(
                f"issue-{index}",
                document_type="issue",
                vector_score=0.0,
                channel="이슈데이터_10000건",
                user_name="Sujin",
                message_date=date(2026, 3, 20) - timedelta(days=index),
                chunk_text=f"[이슈] 완료 이슈 {index}",
                metadata={
                    "entities": ["완료", "이슈"],
                    "chunk_kind": "overview",
                    "issue_title": f"완료 이슈 {index}",
                    "assignee": "Sujin",
                    "status": "완료",
                },
            )
            for index in range(7)
        ]
        codex = FakeCodexProxy()
        service = _build_service(seed_chunks=aggregate_chunks, codex_proxy=codex)

        response = await service.answer_query(
            question="완료 이슈는 몇 건이야",
            access_scopes=["public"],
            request_user=None,
            top_k=5,
        )

        assert response.answer_mode == "llm"
        assert service.postgres.last_search_filters is None
        prompt = codex.calls[0]["user_prompt"]
        assert "Aggregate Summary:" in prompt
        assert "total_matches: 7" in prompt

    @pytest.mark.asyncio
    async def test_summary_without_query_entities_skips_entity_multihop_expansion(self) -> None:
        issue_chunk = _make_chunk(
            "issue-1",
            document_type="issue",
            vector_score=0.9,
            channel="이슈데이터_10000건",
            user_name="Sujin",
            message_date=date(2026, 3, 20),
            chunk_text="[이슈] 메타데이터 필터 적용 시 관련 문서 누락",
            metadata={
                "entities": ["메타데이터", "필터"],
                "chunk_kind": "overview",
                "issue_title": "메타데이터 필터 적용 시 관련 문서 누락",
                "assignee": "Sujin",
                "status": "완료",
            },
        )
        postgres = FakePostgres([issue_chunk])
        neo4j = FakeNeo4j()
        service = RetrievalService(
            settings=Settings(),
            postgres=postgres,
            neo4j=neo4j,
            embedding_provider=FakeEmbeddingProvider(),
            codex_proxy=FakeCodexProxy(),
            query_analyzer=QueryAnalyzer(),
        )

        response = await service.answer_query(
            question="Sujin의 완료된 이슈 요약",
            access_scopes=["public"],
            request_user=None,
            top_k=5,
        )

        assert response.answer_mode == "llm"
        assert neo4j.entity_cooccurrence_calls == []

    def test_multihop_seed_selection_is_capped(self) -> None:
        selected = RetrievalService._select_multihop_seed_ids(
            [f"chunk-{index}" for index in range(8)]
        )

        assert selected == [f"chunk-{index}" for index in range(5)]

    @pytest.mark.asyncio
    async def test_issue_list_queries_include_aggregate_summary_for_total_count(self) -> None:
        issue_chunks = [
            _make_chunk(
                f"issue-{index}",
                document_type="issue",
                vector_score=0.42,
                channel="이슈데이터_10000건",
                user_name="Sujin",
                message_date=date(2026, 3, 20) - timedelta(days=index),
                chunk_text=f"[이슈] Sujin 이슈 {index}",
                metadata={
                    "entities": ["이슈"],
                    "chunk_kind": "overview",
                    "issue_title": f"Sujin 이슈 {index}",
                    "assignee": "Sujin",
                    "status": "완료" if index % 2 == 0 else "진행",
                },
            )
            for index in range(7)
        ]
        codex = FakeCodexProxy()
        service = _build_service(seed_chunks=issue_chunks, codex_proxy=codex)

        response = await service.answer_query(
            question="Sujin의 이슈",
            access_scopes=["public"],
            request_user=None,
            top_k=5,
        )

        assert response.answer_mode == "llm"
        assert service.postgres.last_search_filters is not None
        assert response.debug is None
        prompt = codex.calls[0]["user_prompt"]
        assert "Aggregate Summary:" in prompt
        assert "total_matches: 7" in prompt

    @pytest.mark.asyncio
    async def test_generic_issue_summary_queries_include_aggregate_summary(self) -> None:
        issue_chunks = [
            _make_chunk(
                f"issue-{index}",
                document_type="issue",
                vector_score=0.48,
                channel="이슈데이터_10000건",
                user_name="Sujin",
                message_date=date(2026, 3, 20) - timedelta(days=index),
                chunk_text=f"[이슈] 완료 이슈 {index}",
                metadata={
                    "entities": ["완료", "이슈"],
                    "chunk_kind": "overview",
                    "issue_title": f"완료 이슈 {index}",
                    "assignee": "Sujin",
                    "status": "완료",
                },
            )
            for index in range(7)
        ]
        codex = FakeCodexProxy()
        service = _build_service(seed_chunks=issue_chunks, codex_proxy=codex)

        response = await service.answer_query(
            question="Sujin의 완료된 이슈 요약",
            access_scopes=["public"],
            request_user=None,
            top_k=5,
        )

        assert response.answer_mode == "llm"
        prompt = codex.calls[0]["user_prompt"]
        assert "Aggregate Summary:" in prompt
        assert "total_matches: 7" in prompt

    @pytest.mark.asyncio
    async def test_generic_issue_summary_prefers_overview_examples_over_flow_chunks(self) -> None:
        issue_chunks = [
            _make_chunk(
                f"overview-{index}",
                document_type="issue",
                vector_score=0.2,
                channel="이슈데이터_10000건",
                user_name="Sujin",
                message_date=date(2026, 3, 20) - timedelta(days=index),
                chunk_text=f"[이슈] 완료 이슈 {index}",
                metadata={
                    "entities": ["완료", "이슈"],
                    "chunk_kind": "overview",
                    "issue_title": f"완료 이슈 {index}",
                    "assignee": "Sujin",
                    "status": "완료",
                },
            )
            for index in range(3)
        ] + [
            _make_chunk(
                f"flow-{index}",
                document_type="issue",
                vector_score=0.9 - (index * 0.01),
                channel="이슈데이터_10000건",
                user_name="Sujin",
                message_date=date(2026, 3, 20) - timedelta(days=index),
                chunk_text=f"[이슈] 완료 이슈 {index}\n[수정 및 결과] 상세 흐름 {index}",
                metadata={
                    "entities": ["완료", "이슈"],
                    "chunk_kind": "analysis_flow",
                    "issue_title": f"완료 이슈 {index}",
                    "assignee": "Sujin",
                    "status": "완료",
                    "flow_name": "수정 및 결과",
                },
            )
            for index in range(3)
        ]
        service = _build_service(seed_chunks=issue_chunks, codex_proxy=FakeCodexProxy())

        response = await service.answer_query(
            question="Sujin의 완료된 이슈 요약",
            access_scopes=["public"],
            request_user=None,
            top_k=5,
        )

        assert response.answer_mode == "llm"
        assert response.sources
        assert all(source.source_badge == "issue-overview" for source in response.sources)
        assert len({source.issue_title for source in response.sources}) == len(response.sources)

    @pytest.mark.asyncio
    async def test_mixed_issue_chat_summary_includes_issue_side_aggregate_summary(self) -> None:
        issue_chunks = [
            _make_chunk(
                f"issue-{index}",
                document_type="issue",
                vector_score=0.5 - (index * 0.01),
                channel="이슈데이터_10000건",
                user_name="Sujin",
                message_date=date(2026, 3, 20) - timedelta(days=index),
                chunk_text=f"[이슈] 최근 이슈 {index}",
                metadata={
                    "entities": ["이슈"],
                    "chunk_kind": "overview",
                    "issue_title": f"최근 이슈 {index}",
                    "assignee": "Sujin",
                    "status": "완료",
                },
            )
            for index in range(3)
        ]
        chat_chunk = _make_chunk(
            "chat-1",
            document_type="chat",
            vector_score=0.2,
            channel="개발팀",
            user_name="박소율",
            message_date=date(2026, 3, 19),
            chunk_text="개발팀 박소율: 관련 대화 요약",
            metadata={"entities": ["이슈"], "original_lines": ["개발팀 박소율: 관련 대화 요약"]},
        )
        codex = FakeCodexProxy()
        service = _build_service(
            seed_chunks=[*issue_chunks, chat_chunk],
            codex_proxy=codex,
        )

        response = await service.answer_query(
            question="최근 2주 이슈와 관련 대화 요약",
            access_scopes=["public"],
            request_user=None,
            top_k=5,
        )

        assert response.answer_mode == "llm"
        prompt = codex.calls[0]["user_prompt"]
        assert "Aggregate Summary:" in prompt
        assert "total_matches: 3" in prompt
        assert "document_types=issue" in prompt
        assert response.sources
        assert all(
            source.source_badge in {"issue-overview", "chat"}
            for source in response.sources
        )
        assert any(source.source_badge == "issue-overview" for source in response.sources)

    @pytest.mark.asyncio
    async def test_content_specific_mixed_issue_chat_summary_does_not_attach_generic_issue_count(self) -> None:
        issue_chunks = [
            _make_chunk(
                f"issue-{index}",
                document_type="issue",
                vector_score=0.5 - (index * 0.01),
                channel="이슈데이터_10000건",
                user_name="Sujin",
                message_date=date(2026, 3, 20) - timedelta(days=index),
                chunk_text=f"[이슈] GPU 메모리 관련 이슈 {index}",
                metadata={
                    "entities": ["GPU", "메모리 부족"],
                    "chunk_kind": "overview",
                    "issue_title": f"GPU 메모리 관련 이슈 {index}",
                    "assignee": "Sujin",
                    "status": "완료",
                },
            )
            for index in range(3)
        ]
        codex = FakeCodexProxy()
        service = _build_service(seed_chunks=issue_chunks, codex_proxy=codex)

        response = await service.answer_query(
            question="최근 2주 GPU 메모리 이슈와 관련 대화 요약",
            access_scopes=["public"],
            request_user=None,
            top_k=5,
        )

        assert response.answer_mode == "llm"
        prompt = codex.calls[0]["user_prompt"]
        assert "Aggregate Summary:" not in prompt

    @pytest.mark.asyncio
    async def test_specific_issue_summary_does_not_attach_generic_aggregate_summary(self) -> None:
        issue_chunk = _make_chunk(
            "issue-1",
            document_type="issue",
            vector_score=0.9,
            channel="이슈데이터_10000건",
            user_name="Sujin",
            message_date=date(2026, 3, 20),
            chunk_text="[이슈] 메타데이터 필터 적용 시 관련 문서 누락",
            metadata={
                "entities": ["메타데이터", "필터"],
                "chunk_kind": "overview",
                "issue_title": "메타데이터 필터 적용 시 관련 문서 누락",
                "assignee": "Sujin",
                "status": "완료",
            },
        )
        codex = FakeCodexProxy()
        service = _build_service(seed_chunks=[issue_chunk], codex_proxy=codex)

        response = await service.answer_query(
            question="메타데이터 필터 적용 시 관련 문서 누락 요약",
            access_scopes=["public"],
            request_user=None,
            top_k=5,
        )

        assert response.answer_mode == "llm"
        prompt = codex.calls[0]["user_prompt"]
        assert "Aggregate Summary:" not in prompt

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

    @pytest.mark.asyncio
    async def test_request_filters_override_text_suffix_hacks(self) -> None:
        """구조화 필터는 질문 문자열이 아니라 실제 검색 필터로 반영되어야 한다."""
        seed = _make_chunk(
            "c1",
            vector_score=0.9,
            channel="general",
            entities=["배포"],
            message_date=date(2024, 1, 5),
        )
        service = _build_service(seed_chunks=[seed])

        response = await service.answer_query(
            question="배포 질문",
            access_scopes=["public"],
            request_user=None,
            request_filters=QueryRequestFilters(
                channels=["general", "dev"],
                user_names=["민수"],
                date_from=date(2024, 1, 1),
                date_to=date(2024, 1, 31),
            ),
        )

        assert response.answer_mode == "llm"
        assert service.postgres.last_search_filters is not None
        assert service.postgres.last_search_filters.all_channels == ["general", "dev"]
        assert service.postgres.last_search_filters.user_names == ["민수"]
        assert service.postgres.last_search_filters.date_from == date(2024, 1, 1)
        assert service.postgres.last_search_filters.date_to == date(2024, 1, 31)

    @pytest.mark.asyncio
    async def test_chat_recent_query_uses_chat_latest_event_date(self) -> None:
        seed = _make_chunk("c1", vector_score=0.9, channel="general", message_date=date(2024, 3, 20))
        service = _build_service(seed_chunks=[seed])

        response = await service.answer_query(
            question="general 최근 대화",
            access_scopes=["public"],
            request_user=None,
        )

        assert response.answer_mode == "llm"
        assert service.postgres.last_search_filters is not None
        assert service.postgres.last_search_filters.document_types == ["chat"]
        assert service.postgres.last_search_filters.date_from == date(2024, 3, 13)
        assert service.postgres.last_search_filters.date_to == date(2024, 3, 20)

    @pytest.mark.asyncio
    async def test_expanded_chunks_are_filtered_by_explicit_channel(self) -> None:
        seed = _make_chunk("c1", vector_score=0.9, channel="general", message_date=date(2024, 3, 20))
        expanded_other_channel = _make_chunk("c2", vector_score=0.0, channel="random", message_date=date(2024, 3, 20))
        expansion = GraphExpansion(
            chunk_id="c1",
            graph_neighbors=["민수", "general"],
            expanded_chunk_ids=["c2"],
        )
        service = _build_service(
            seed_chunks=[seed],
            extra_chunks=[expanded_other_channel],
            expansions={"c1": expansion},
        )

        response = await service.answer_query(
            question="general 최근 대화",
            access_scopes=["public"],
            request_user=None,
        )

        assert response.answer_mode == "llm"
        assert [source.chunk_id for source in response.sources] == ["c1"]

    @pytest.mark.asyncio
    async def test_chat_facets_do_not_leak_issue_only_metadata(self) -> None:
        service = _build_service(seed_chunks=[])

        facets = await service.get_facets(document_type="chat")

        assert facets.document_types == ["chat"]
        assert facets.channels == ["general", "dev"]
        assert facets.users == ["민수", "지현"]
        assert facets.assignees == []
        assert facets.statuses == []
        assert facets.latest_event_date == date(2024, 3, 20)

    @pytest.mark.asyncio
    async def test_generic_chat_query_prefers_full_topic_match_over_partial_match(self) -> None:
        partial_match = _make_chunk(
            "chat-1",
            document_type="chat",
            vector_score=0.96,
            channel="점심약속",
            user_name="민수",
            message_date=date(2026, 3, 20),
            chunk_text="점심약속 민수: 서버 점검은 끝났습니다.",
            metadata={"entities": ["서버"], "original_lines": ["점심약속 민수: 서버 점검은 끝났습니다."]},
        )
        exact_match = _make_chunk(
            "chat-2",
            document_type="chat",
            vector_score=0.82,
            channel="프로젝트C",
            user_name="박소율",
            message_date=date(2026, 3, 20),
            chunk_text="프로젝트C 박소율: 서버 배포 380차 완료했습니다.",
            metadata={"entities": ["서버", "배포"], "original_lines": ["프로젝트C 박소율: 서버 배포 380차 완료했습니다."]},
        )
        service = _build_service(seed_chunks=[partial_match, exact_match])

        response = await service.answer_query(
            question="서버 배포 관련 내용 알려줘",
            access_scopes=["public"],
            request_user=None,
            top_k=5,
        )

        assert response.answer_mode == "llm"
        assert response.sources[0].chunk_id == "chat-2"
        assert response.sources[0].channel == "프로젝트C"

    @pytest.mark.asyncio
    async def test_mixed_related_chat_summary_surfaces_chat_sources_in_preview(self) -> None:
        issue_chunks = [
            _make_chunk(
                f"issue-{index}",
                document_type="issue",
                vector_score=0.72 - (index * 0.01),
                channel="이슈데이터_10000건",
                user_name="Sujin",
                message_date=date(2026, 3, 20) - timedelta(days=index),
                chunk_text=f"[이슈] 최근 이슈 {index}",
                metadata={
                    "entities": ["이슈"],
                    "chunk_kind": "overview",
                    "issue_title": f"최근 이슈 {index}",
                    "assignee": "Sujin",
                    "status": "완료",
                },
            )
            for index in range(4)
        ]
        chat_chunks = [
            _make_chunk(
                f"chat-{index}",
                document_type="chat",
                vector_score=0.24 - (index * 0.01),
                channel="개발팀",
                user_name="박소율",
                message_date=date(2026, 3, 19) - timedelta(days=index),
                chunk_text=f"개발팀 박소율: 최근 이슈 관련 대화 {index}",
                metadata={"entities": ["이슈"], "original_lines": [f"개발팀 박소율: 최근 이슈 관련 대화 {index}"]},
            )
            for index in range(2)
        ]
        service = _build_service(seed_chunks=[*issue_chunks, *chat_chunks])

        response = await service.answer_query(
            question="최근 2주 이슈와 관련 대화 요약",
            access_scopes=["public"],
            request_user=None,
            top_k=5,
        )

        assert response.answer_mode == "llm"
        preview = response.sources[:4]
        assert any(source.document_type == "chat" for source in preview)
        assert any(source.document_type == "issue" for source in preview)


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
