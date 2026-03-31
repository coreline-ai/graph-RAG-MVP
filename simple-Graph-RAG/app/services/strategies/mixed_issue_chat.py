from __future__ import annotations

from typing import TYPE_CHECKING

from app.adapters.codex_proxy import CodexProxyError
from app.schemas import QueryResponse, RetrievedChunk
from app.services.ranking_policy import MixedRankingPolicy, RankingContext
from app.services.query_terms import (
    chunk_matches_alias_group,
    chunk_search_text,
    exact_special_groups,
    looks_like_mixed_issue_chat_summary,
    query_match_terms,
    strict_lexical_groups,
)
from app.services.source_selector import MixedSourceSelector

if TYPE_CHECKING:
    from app.services.retrieval import PipelineTimer, RetrievalService
    from app.schemas import QueryAnalysis, QueryFilters


class MixedIssueChatStrategy:
    def __init__(self, service: "RetrievalService") -> None:
        self.service = service
        self.issue_policy = MixedRankingPolicy(lane="issue")
        self.chat_policy = MixedRankingPolicy(lane="chat")
        self.selector = MixedSourceSelector()

    async def execute(
        self,
        *,
        question: str,
        analysis: "QueryAnalysis",
        top_k: int,
        request_user: str | None,
        debug: bool,
        timer: "PipelineTimer",
        metadata_context: dict[str, object],
    ) -> QueryResponse:
        aggregate_context = None
        if looks_like_mixed_issue_chat_summary(analysis):
            timer.start_step("aggregate_summary")
            aggregate_context = await self.service._build_aggregate_context(analysis, top_k)
            timer.end_step(
                "aggregate_summary",
                f"({aggregate_context['matched_count']} matches)" if aggregate_context else "",
            )

        timer.start_step("embedding")
        query_embedding = (await self.service.embedding_provider.embed_texts([analysis.clean_question]))[0]
        timer.end_step("embedding")

        issue_filters = self._issue_filters(analysis)
        chat_filters = self._chat_filters(analysis)

        timer.start_step("issue_lane_search")
        issue_candidates = await self._search_issue_candidates(
            query_embedding=query_embedding,
            filters=issue_filters,
            top_k=max(top_k * 3, top_k),
        )
        timer.end_step("issue_lane_search", f"({len(issue_candidates)} chunks)")

        timer.start_step("chat_lane_search")
        chat_candidates = await self._search_chat_candidates(
            query_embedding=query_embedding,
            filters=chat_filters,
            top_k=max(top_k * 4, top_k),
        )
        timer.end_step("chat_lane_search", f"({len(chat_candidates)} chunks)")

        timer.start_step("ranking")
        issue_ranked = self.issue_policy.rank(
            seed_chunks=issue_candidates,
            expanded_chunks=[],
            analysis=analysis,
            context=RankingContext(
                latest_event_date=metadata_context["latest_event_date"],
                lane="issue",
            ),
        )
        issue_ranked = self.issue_policy.apply_special_keyword_grounding(issue_ranked, analysis)
        chat_ranked = self.chat_policy.rank(
            seed_chunks=chat_candidates,
            expanded_chunks=[],
            analysis=analysis,
            context=RankingContext(
                latest_event_date=metadata_context["latest_event_date"],
                lane="chat",
            ),
        )
        chat_ranked = self._filter_relevant_chat(issue_ranked, chat_ranked, analysis)
        selected_chunks = self.selector.select(
            issue_chunks=issue_ranked,
            chat_chunks=chat_ranked,
            analysis=analysis,
            top_k=top_k,
            aggregate_context=aggregate_context,
        )
        timer.end_step("ranking")

        if not selected_chunks:
            debug_data = None
            if debug:
                debug_data = await self.service._build_debug_data(
                    timer=timer,
                    analysis=analysis,
                    ranked_chunks=[],
                    top_k=top_k,
                    subgraph_rows=[],
                    community_rows=[],
                    seed_chunks=[],
                    graph_seeded_chunks=[],
                    multihop_ids=[],
                    expanded_ids=[],
                    route="mixed_issue_chat",
                    strategy="mixed_issue_chat",
                    count_kind="none",
                    chat_match_count=0,
                    issue_match_count=0,
                )
            return QueryResponse(
                question=question,
                answer="조건에 맞는 근거를 찾지 못했습니다.",
                retrieval_strategy="filter_pgvector_graph_hybrid",
                answer_mode="fallback_sources_only",
                sources=[],
                debug=debug_data,
            )

        sources = [self.service._to_source(chunk) for chunk in selected_chunks]

        timer.start_step("subgraph_extract")
        try:
            top_chunk_ids = [s.chunk_id for s in sources[:5]]
            subgraph_rows = await self.service.neo4j.extract_subgraph(top_chunk_ids)
            subgraph_context = self.service._build_subgraph_context(subgraph_rows)
        except Exception:
            subgraph_rows = []
            subgraph_context = ""
        timer.end_step("subgraph_extract")

        system_prompt = self.service._build_system_prompt()
        user_prompt = self.service._build_user_prompt(
            question,
            sources,
            request_user,
            analysis=analysis,
            channels=metadata_context["channels"],
            users=metadata_context["users"],
            assignees=metadata_context["assignees"],
            statuses=metadata_context["statuses"],
            subgraph_context=subgraph_context,
            aggregate_context=aggregate_context,
        )

        timer.start_step("llm_generate")
        try:
            llm_response = await self.service.codex_proxy.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                metadata={
                    "request_user": request_user,
                    "retrieval_strategy": "filter_pgvector_graph_hybrid",
                    "top_k": top_k,
                    "query_route": "mixed_issue_chat",
                },
            )
            timer.end_step("llm_generate")
            debug_data = None
            if debug:
                debug_data = await self.service._build_debug_data(
                    timer=timer,
                    analysis=analysis,
                    ranked_chunks=[*issue_ranked, *chat_ranked],
                    top_k=top_k,
                    subgraph_rows=subgraph_rows,
                    community_rows=[],
                    seed_chunks=[*issue_ranked, *chat_ranked],
                    graph_seeded_chunks=[],
                    multihop_ids=[],
                    expanded_ids=[],
                    route="mixed_issue_chat",
                    strategy="mixed_issue_chat",
                    count_kind="none",
                    chat_match_count=len([source for source in sources if source.document_type == "chat"]),
                    issue_match_count=len([source for source in sources if source.document_type == "issue"]),
                )
            return QueryResponse(
                question=question,
                answer=llm_response.text,
                retrieval_strategy="filter_pgvector_graph_hybrid",
                answer_mode="llm",
                sources=sources,
                debug=debug_data,
            )
        except CodexProxyError:
            timer.end_step("llm_generate")
            debug_data = None
            if debug:
                debug_data = await self.service._build_debug_data(
                    timer=timer,
                    analysis=analysis,
                    ranked_chunks=[*issue_ranked, *chat_ranked],
                    top_k=top_k,
                    subgraph_rows=subgraph_rows,
                    community_rows=[],
                    seed_chunks=[*issue_ranked, *chat_ranked],
                    graph_seeded_chunks=[],
                    multihop_ids=[],
                    expanded_ids=[],
                    route="mixed_issue_chat",
                    strategy="mixed_issue_chat",
                    count_kind="none",
                    chat_match_count=len([source for source in sources if source.document_type == "chat"]),
                    issue_match_count=len([source for source in sources if source.document_type == "issue"]),
                )
            fallback_answer = (
                self.service._build_aggregate_fallback_answer(aggregate_context, sources)
                if aggregate_context is not None
                else self.service._build_fallback_answer(analysis, sources, top_k)
            )
            return QueryResponse(
                question=question,
                answer=fallback_answer,
                retrieval_strategy="filter_pgvector_graph_hybrid",
                answer_mode="fallback_sources_only",
                sources=sources,
                debug=debug_data,
            )

    def _issue_filters(self, analysis: "QueryAnalysis") -> "QueryFilters":
        filters = analysis.filters.model_copy(deep=True)
        filters.document_types = ["issue"]
        return filters

    def _chat_filters(self, analysis: "QueryAnalysis") -> "QueryFilters":
        filters = analysis.filters.model_copy(deep=True)
        filters.document_types = ["chat"]
        filters.assignees = []
        filters.statuses = []
        return filters

    async def _search_issue_candidates(
        self,
        *,
        query_embedding: list[float],
        filters: "QueryFilters",
        top_k: int,
    ) -> list[RetrievedChunk]:
        specialized = await self.service._call_optional(
            self.service.postgres,
            "search_issue_candidates",
            None,
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
        )
        if specialized is not None:
            return specialized
        return await self.service.postgres.search_chunks(query_embedding, filters, top_k)

    async def _search_chat_candidates(
        self,
        *,
        query_embedding: list[float],
        filters: "QueryFilters",
        top_k: int,
    ) -> list[RetrievedChunk]:
        specialized = await self.service._call_optional(
            self.service.postgres,
            "search_chat_candidates",
            None,
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
        )
        if specialized is not None:
            return specialized
        return await self.service.postgres.search_chunks(query_embedding, filters, top_k)

    def _filter_relevant_chat(
        self,
        issue_ranked: list[RetrievedChunk],
        chat_ranked: list[RetrievedChunk],
        analysis: "QueryAnalysis",
    ) -> list[RetrievedChunk]:
        if not chat_ranked:
            return []
        overlap_terms = set(query_match_terms(analysis))
        alias_groups = exact_special_groups(analysis) or strict_lexical_groups(analysis)
        for chunk in issue_ranked[:8]:
            issue_title = str(chunk.metadata.get("issue_title") or "").strip().lower()
            if issue_title:
                overlap_terms.add(issue_title)
            overlap_terms.update(
                str(entity).lower()
                for entity in chunk.metadata.get("entities", [])
                if str(entity).strip()
            )

        filtered: list[RetrievedChunk] = []
        for chunk in chat_ranked:
            text = chunk_search_text(chunk)
            has_term_overlap = any(term and term in text for term in overlap_terms)
            has_alias_overlap = any(chunk_matches_alias_group(chunk, group) for group in alias_groups)
            if has_term_overlap or has_alias_overlap:
                filtered.append(chunk)
        return filtered
