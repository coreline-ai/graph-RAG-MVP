from __future__ import annotations

from datetime import date

from app.adapters.codex_proxy import CodexProxyClient, CodexProxyError
from app.adapters.embedding_provider import BgeM3EmbeddingProvider
from app.adapters.neo4j_store import Neo4jStore
from app.adapters.postgres_vector_store import PostgresVectorStore
from app.config import Settings
from app.schemas import QueryAnalysis, QueryResponse, QuerySource, RetrievedChunk
from app.services.query_analyzer import QueryAnalyzer


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

    async def answer_query(
        self,
        *,
        question: str,
        access_scopes: list[str],
        request_user: str | None,
        top_k: int | None = None,
    ) -> QueryResponse:
        top_k = top_k or self.settings.top_k
        channels = await self.postgres.list_channels()
        users = await self.postgres.list_users()
        analysis = self.query_analyzer.analyze(
            question,
            access_scopes=access_scopes or self.settings.parsed_default_access_scopes,
            channels=channels,
            users=users,
        )

        query_embedding = (await self.embedding_provider.embed_texts([analysis.clean_question]))[0]
        seed_chunks = await self.postgres.search_chunks(query_embedding, analysis.filters, top_k)
        if not seed_chunks:
            return QueryResponse(
                question=question,
                answer="조건에 맞는 근거를 찾지 못했습니다.",
                retrieval_strategy="filter_pgvector_graph_hybrid",
                answer_mode="fallback_sources_only",
                sources=[],
            )

        try:
            expansions = await self.neo4j.expand_from_seed_chunks(
                [chunk.chunk_id for chunk in seed_chunks],
                next_window=self.settings.graph_next_window,
            )
        except Exception:
            expansions = {}
        expanded_ids = {
            chunk_id
            for expansion in expansions.values()
            for chunk_id in expansion.expanded_chunk_ids
            if chunk_id not in {seed.chunk_id for seed in seed_chunks}
        }
        expanded_chunks = await self.postgres.get_chunks_by_ids(expanded_ids)

        ranked_chunks = self._rank_chunks(seed_chunks, expanded_chunks, expansions, analysis)
        sources = [self._to_source(chunk) for chunk in ranked_chunks[:top_k]]
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(question, sources, request_user)

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
            return QueryResponse(
                question=question,
                answer=llm_response.text,
                retrieval_strategy="filter_pgvector_graph_hybrid",
                answer_mode="llm",
                sources=sources,
            )
        except CodexProxyError:
            return QueryResponse(
                question=question,
                answer=self._build_fallback_answer(analysis, sources),
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
        ranked: dict[str, RetrievedChunk] = {chunk.chunk_id: chunk for chunk in seed_chunks}
        for chunk in seed_chunks:
            expansion = expansions.get(chunk.chunk_id)
            if expansion:
                chunk.graph_neighbors = expansion.graph_neighbors
                chunk.graph_score = min(1.0, len(expansion.graph_neighbors) / 4)
            chunk.metadata_score = self._metadata_score(chunk, analysis)
            chunk.recency_score = self._recency_score(chunk.message_date)
            chunk.final_score = self._combined_score(chunk)

        for chunk in expanded_chunks:
            chunk.graph_score = 0.6
            chunk.metadata_score = self._metadata_score(chunk, analysis)
            chunk.recency_score = self._recency_score(chunk.message_date)
            chunk.final_score = self._combined_score(chunk)
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

    def _combined_score(self, chunk: RetrievedChunk) -> float:
        return (
            0.45 * chunk.vector_score
            + 0.25 * chunk.graph_score
            + 0.20 * chunk.metadata_score
            + 0.10 * chunk.recency_score
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
        return (
            f"Request User: {requester}\n"
            f"Question: {question}\n\n"
            f"Evidence:\n{rendered_sources}"
        )

    def _build_fallback_answer(self, analysis: QueryAnalysis, sources: list[QuerySource]) -> str:
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
        evidence_lines = [
            f"- {source.message_date} {source.channel} {source.user_name}: {source.content.splitlines()[0]}"
            for source in sources[:3]
        ]
        return "LLM 응답을 생성하지 못해 검색 근거만 반환합니다.\n" + header + "\n" + "\n".join(
            evidence_lines
        )
