from __future__ import annotations

import logging
import time
from datetime import date

_log = logging.getLogger("retrieval.perf")
_log.setLevel(logging.INFO)
if not _log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    _log.addHandler(_h)

from app.adapters.codex_proxy import CodexProxyClient, CodexProxyError
from app.adapters.embedding_provider import BgeM3EmbeddingProvider
from app.adapters.neo4j_store import Neo4jStore
from app.adapters.postgres_vector_store import PostgresVectorStore
from app.config import Settings
from app.schemas import QueryAnalysis, QueryResponse, QuerySource, RetrievedChunk
from app.services.query_analyzer import QueryAnalyzer

_METADATA_CACHE_TTL = 300  # 5 minutes


class RetrievalService:
    def __init__(
        self,
        *,
        settings: Settings,
        postgres: PostgresVectorStore,
        neo4j: Neo4jStore,
        embedding_provider: BgeM3EmbeddingProvider,
        codex_proxy: CodexProxyClient,
        query_analyzer: QueryAnalyzer,
    ) -> None:
        self.settings = settings
        self.postgres = postgres
        self.neo4j = neo4j
        self.embedding_provider = embedding_provider
        self.codex_proxy = codex_proxy
        self.query_analyzer = query_analyzer
        self._channels_cache: list[str] = []
        self._users_cache: list[str] = []
        self._cache_ts: float = 0.0

    async def _get_metadata_lists(self) -> tuple[list[str], list[str]]:
        now = time.monotonic()
        if now - self._cache_ts > _METADATA_CACHE_TTL or not self._channels_cache:
            self._channels_cache = await self.postgres.list_channels()
            self._users_cache = await self.postgres.list_users()
            self._cache_ts = now
        return self._channels_cache, self._users_cache

    async def answer_query(
        self,
        *,
        question: str,
        access_scopes: list[str],
        request_user: str | None,
        top_k: int | None = None,
    ) -> QueryResponse:
        t_start = time.perf_counter()
        top_k = top_k or self.settings.top_k

        t0 = time.perf_counter()
        channels, users = await self._get_metadata_lists()
        analysis = self.query_analyzer.analyze(
            question,
            access_scopes=access_scopes or self.settings.parsed_default_access_scopes,
            channels=channels,
            users=users,
        )
        _log.info("[PERF] query_analysis       : %.3fs", time.perf_counter() - t0)

        t0 = time.perf_counter()
        query_embedding = (await self.embedding_provider.embed_texts([analysis.clean_question]))[0]
        _log.info("[PERF] embedding            : %.3fs", time.perf_counter() - t0)

        t0 = time.perf_counter()
        seed_chunks = await self.postgres.search_chunks(query_embedding, analysis.filters, top_k)
        _log.info("[PERF] pgvector_search      : %.3fs (%d chunks)", time.perf_counter() - t0, len(seed_chunks))
        if not seed_chunks:
            return QueryResponse(
                question=question,
                answer="조건에 맞는 근거를 찾지 못했습니다.",
                retrieval_strategy="filter_pgvector_graph_hybrid",
                answer_mode="fallback_sources_only",
                sources=[],
            )

        t0 = time.perf_counter()
        try:
            expansions = await self.neo4j.expand_from_seed_chunks(
                [chunk.chunk_id for chunk in seed_chunks],
                next_window=self.settings.graph_next_window,
            )
        except Exception:
            expansions = {}
        _log.info("[PERF] neo4j_graph_expand   : %.3fs", time.perf_counter() - t0)

        t0 = time.perf_counter()
        expanded_ids = {
            chunk_id
            for expansion in expansions.values()
            for chunk_id in expansion.expanded_chunk_ids
            if chunk_id not in {seed.chunk_id for seed in seed_chunks}
        }
        expanded_chunks = await self.postgres.get_chunks_by_ids(expanded_ids)
        _log.info("[PERF] pg_expand_fetch      : %.3fs (%d extra)", time.perf_counter() - t0, len(expanded_chunks))

        t0 = time.perf_counter()
        ranked_chunks = self._rank_chunks(seed_chunks, expanded_chunks, expansions, analysis)
        sources = [self._to_source(chunk) for chunk in ranked_chunks[:top_k]]
        _log.info("[PERF] ranking              : %.3fs", time.perf_counter() - t0)

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            question, sources, request_user,
            analysis=analysis, channels=channels, users=users,
        )

        t0 = time.perf_counter()
        try:
            llm_response = await self.codex_proxy.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                metadata={
                    "request_user": request_user,
                    "retrieval_strategy": "filter_pgvector_graph_hybrid",
                    "top_k": top_k,
                },
            )
            _log.info("[PERF] llm_generate         : %.3fs", time.perf_counter() - t0)
            _log.info("[PERF] ════ TOTAL           : %.3fs ════", time.perf_counter() - t_start)
            return QueryResponse(
                question=question,
                answer=llm_response.text,
                retrieval_strategy="filter_pgvector_graph_hybrid",
                answer_mode="llm",
                sources=sources,
            )
        except CodexProxyError:
            _log.info("[PERF] llm_generate (FAIL)  : %.3fs", time.perf_counter() - t0)
            _log.info("[PERF] ════ TOTAL           : %.3fs ════", time.perf_counter() - t_start)
            return QueryResponse(
                question=question,
                answer=self._build_fallback_answer(analysis, sources, top_k),
                retrieval_strategy="filter_pgvector_graph_hybrid",
                answer_mode="fallback_sources_only",
                sources=sources,
            )

    def _rank_chunks(
        self,
        seed_chunks: list[RetrievedChunk],
        expanded_chunks: list[RetrievedChunk],
        expansions: dict[str, object],
        analysis: QueryAnalysis,
    ) -> list[RetrievedChunk]:
        weights = self._get_weights(analysis)
        has_date_filter = analysis.filters.date_from is not None

        ranked: dict[str, RetrievedChunk] = {chunk.chunk_id: chunk for chunk in seed_chunks}
        for chunk in seed_chunks:
            expansion = expansions.get(chunk.chunk_id)
            if expansion:
                chunk.graph_neighbors = expansion.graph_neighbors
                chunk.graph_score = min(1.0, len(expansion.graph_neighbors) / 4)
            chunk.metadata_score = self._metadata_score(chunk, analysis)
            chunk.recency_score = 0.0 if has_date_filter else self._recency_score(chunk.message_date)
            chunk.final_score = self._combined_score(chunk, weights)

        for chunk in expanded_chunks:
            chunk.graph_score = 0.6
            chunk.metadata_score = self._metadata_score(chunk, analysis)
            chunk.recency_score = 0.0 if has_date_filter else self._recency_score(chunk.message_date)
            chunk.final_score = self._combined_score(chunk, weights)
            ranked.setdefault(chunk.chunk_id, chunk)

        return sorted(ranked.values(), key=lambda item: item.final_score, reverse=True)

    def _metadata_score(self, chunk: RetrievedChunk, analysis: QueryAnalysis) -> float:
        score = 0.0
        filters = analysis.filters
        if filters.channel and chunk.channel == filters.channel:
            score += 0.4
        if filters.user_name and chunk.user_name == filters.user_name:
            score += 0.3
        if filters.date_from and filters.date_to and filters.date_from <= chunk.message_date <= filters.date_to:
            score += 0.3
        return min(score, 1.0)

    def _recency_score(self, chunk_date: date) -> float:
        days = max((date.today() - chunk_date).days, 0)
        return 1.0 / (1.0 + (days / 30.0))

    @staticmethod
    def _get_weights(analysis: QueryAnalysis) -> tuple[float, float, float, float]:
        intent = analysis.intent
        if intent == "timeline":
            return (0.35, 0.20, 0.25, 0.20)
        if intent == "relationship":
            return (0.30, 0.40, 0.20, 0.10)
        if intent == "aggregate":
            return (0.35, 0.30, 0.25, 0.10)
        if intent == "summary":
            return (0.40, 0.25, 0.25, 0.10)
        return (0.45, 0.25, 0.20, 0.10)

    @staticmethod
    def _combined_score(chunk: RetrievedChunk, weights: tuple[float, float, float, float]) -> float:
        w_vector, w_graph, w_metadata, w_recency = weights
        return (
            w_vector * chunk.vector_score
            + w_graph * chunk.graph_score
            + w_metadata * chunk.metadata_score
            + w_recency * chunk.recency_score
        )

    def _to_source(self, chunk: RetrievedChunk) -> QuerySource:
        content = "\n".join(chunk.metadata.get("original_lines", [])[:2]) or chunk.chunk_text
        return QuerySource(
            chunk_id=chunk.chunk_id,
            score=round(chunk.final_score, 4),
            content=content,
            graph_neighbors=chunk.graph_neighbors,
            channel=chunk.channel,
            user_name=chunk.user_name,
            message_date=chunk.message_date,
        )

    def _build_system_prompt(self) -> str:
        return (
            "당신은 한국어 조직 채팅 로그를 분석하는 GraphRAG 어시스턴트다.\n"
            "제공된 근거 밖의 추론은 최소화하고, 가능하면 날짜/채널/사용자 맥락을 유지해 답하라.\n"
            "근거가 부족하면 단정하지 말고 제한사항을 짧게 밝혀라."
        )

    def _build_user_prompt(
        self,
        question: str,
        sources: list[QuerySource],
        request_user: str | None,
        analysis: QueryAnalysis | None = None,
        channels: list[str] | None = None,
        users: list[str] | None = None,
    ) -> str:
        rendered_sources = "\n\n".join(
            (
                f"[Source {index + 1}] {source.message_date} "
                f"{source.channel} {source.user_name}\n"
                f"{source.content}\n"
                f"Graph neighbors: {', '.join(source.graph_neighbors)}"
            )
            for index, source in enumerate(sources)
        )
        requester = request_user or "anonymous"

        metadata_section = ""
        if analysis and analysis.intent in ("aggregate", "relationship"):
            parts = []
            if channels:
                parts.append(f"전체 채널 목록 ({len(channels)}개): {', '.join(channels)}")
            if users:
                parts.append(f"전체 사용자 목록 ({len(users)}명): {', '.join(users)}")
            if parts:
                metadata_section = "\n\nDatabase Metadata:\n" + "\n".join(parts)

        return (
            f"Request User: {requester}\n"
            f"Question: {question}\n\n"
            f"Evidence:\n{rendered_sources}"
            f"{metadata_section}"
        )

    def _build_fallback_answer(
        self, analysis: QueryAnalysis, sources: list[QuerySource], top_k: int = 10,
    ) -> str:
        if not sources:
            return "조건에 맞는 근거를 찾지 못했습니다."
        header_parts = []
        if analysis.filters.channel:
            header_parts.append(f"채널={analysis.filters.channel}")
        if analysis.filters.user_name:
            header_parts.append(f"사용자={analysis.filters.user_name}")
        if analysis.filters.date_from:
            if analysis.filters.date_to and analysis.filters.date_to != analysis.filters.date_from:
                header_parts.append(
                    f"기간={analysis.filters.date_from.isoformat()}~{analysis.filters.date_to.isoformat()}"
                )
            else:
                header_parts.append(f"날짜={analysis.filters.date_from.isoformat()}")
        header = ", ".join(header_parts) if header_parts else "특정 필터 없음"
        max_fallback = max(3, min(top_k, 10))
        evidence_lines = [
            f"- {source.message_date} {source.channel} {source.user_name}: {source.content.splitlines()[0]}"
            for source in sources[:max_fallback]
        ]
        return "LLM 응답을 생성하지 못해 검색 근거만 반환합니다.\n" + header + "\n" + "\n".join(
            evidence_lines
        )
