from __future__ import annotations

from typing import TYPE_CHECKING

from app.adapters.codex_proxy import CodexProxyError
from app.schemas import QueryResponse
from app.services.query_terms import count_kind_for_analysis, exact_special_groups, strict_lexical_groups
from app.services.source_selector import CountSourceSelector

if TYPE_CHECKING:
    from app.services.retrieval import PipelineTimer, RetrievalService
    from app.schemas import QueryAnalysis


class CountQueryStrategy:
    def __init__(self, service: "RetrievalService") -> None:
        self.service = service
        self.selector = CountSourceSelector()

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
        count_kind = count_kind_for_analysis(question, analysis)

        timer.start_step("aggregate_summary")
        aggregate_context = await self._build_count_context(analysis, top_k, count_kind)
        timer.end_step(
            "aggregate_summary",
            f"({aggregate_context['matched_count']} matches)" if aggregate_context else "",
        )

        if not aggregate_context or aggregate_context.get("matched_count", 0) == 0:
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
                    route="count",
                    strategy="count_query",
                    count_kind=count_kind,
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

        ranked_chunks = list(aggregate_context["sample_chunks"])
        selected_chunks = self.selector.select(sample_chunks=ranked_chunks, top_k=top_k)
        sources = [self.service._to_source(chunk) for chunk in selected_chunks]

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
                    "query_route": "count",
                    "count_kind": count_kind,
                },
            )
            timer.end_step("llm_generate")
            debug_data = None
            if debug:
                debug_data = await self.service._build_debug_data(
                    timer=timer,
                    analysis=analysis,
                    ranked_chunks=ranked_chunks,
                    top_k=top_k,
                    subgraph_rows=[],
                    community_rows=[],
                    seed_chunks=ranked_chunks,
                    graph_seeded_chunks=[],
                    multihop_ids=[],
                    expanded_ids=[],
                    route="count",
                    strategy="count_query",
                    count_kind=count_kind,
                    chat_match_count=0,
                    issue_match_count=len(sources),
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
                    ranked_chunks=ranked_chunks,
                    top_k=top_k,
                    subgraph_rows=[],
                    community_rows=[],
                    seed_chunks=ranked_chunks,
                    graph_seeded_chunks=[],
                    multihop_ids=[],
                    expanded_ids=[],
                    route="count",
                    strategy="count_query",
                    count_kind=count_kind,
                    chat_match_count=0,
                    issue_match_count=len(sources),
                )
            return QueryResponse(
                question=question,
                answer=self.service._build_aggregate_fallback_answer(aggregate_context, sources),
                retrieval_strategy="filter_pgvector_graph_hybrid",
                answer_mode="fallback_sources_only",
                sources=sources,
                debug=debug_data,
            )

    async def _build_count_context(
        self,
        analysis: "QueryAnalysis",
        top_k: int,
        count_kind: str,
    ) -> dict[str, object] | None:
        aggregate_filters = self.service._aggregate_filters_for_analysis(analysis)
        if count_kind == "subtype":
            summary = await self.service._call_optional(
                self.service.postgres,
                "summarize_special_keyword_results",
                None,
                filters=aggregate_filters,
                exact_groups=exact_special_groups(analysis),
                alias_groups=strict_lexical_groups(analysis),
                limit=min(max(top_k * 3, top_k), 30),
            )
        else:
            summary = await self.service._call_optional(
                self.service.postgres,
                "summarize_filtered_results",
                None,
                filters=aggregate_filters,
                limit=min(max(top_k * 3, top_k), 30),
            )
        if not summary:
            return None
        summary["applied_filters"] = self.service._render_applied_filters(aggregate_filters)
        summary["count_kind"] = count_kind
        return summary
