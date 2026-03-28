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
from app.schemas import (
    CooccurrenceEdge,
    CommunityCluster,
    DebugData,
    IntentWeights,
    MetadataFacetsResponse,
    PipelineTiming,
    QueryAnalysis,
    QueryFilters,
    QueryResponse,
    QueryRequestFilters,
    QuerySource,
    RetrievedChunk,
    ScoreBreakdown,
    SubgraphEdge,
    SubgraphNode,
)
from app.services.query_router import QueryRouter
from app.services.query_analyzer import QueryAnalyzer
from app.services.query_terms import (
    GENERIC_ISSUE_SUMMARY_QUERIES,
    SPECIAL_EXACT_TERMS,
    SPECIAL_LEXICAL_ALIASES,
    chunk_matches_alias_group,
    chunk_search_text,
    exact_special_groups,
    looks_like_count_query,
    looks_like_flow_query,
    looks_like_generic_issue_summary,
    looks_like_mixed_issue_chat_summary,
    looks_like_related_chat_query,
    query_match_terms,
    query_phrase_candidates,
    strict_lexical_groups,
)
from app.services.ranking_policy import RankingContext, StandardRankingPolicy
from app.services.source_selector import StandardSourceSelector, aggregate_sample_chunks, dedupe_source_candidates
from app.services.strategies.count_query import CountQueryStrategy
from app.services.strategies.mixed_issue_chat import MixedIssueChatStrategy

_METADATA_CACHE_TTL = 300  # 5 minutes
_MULTIHOP_SEED_LIMIT = 5


class PipelineTimer:
    """Collects per-step timings. Always logs; collects data only when enabled."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.steps: list[dict[str, float | str]] = []
        self._start = time.perf_counter()
        self._step_start: float = 0.0

    def start_step(self, name: str) -> None:
        self._step_start = time.perf_counter()

    def end_step(self, name: str, extra: str = "") -> None:
        elapsed = time.perf_counter() - self._step_start
        msg = f"[PERF] {name:<22s}: {elapsed:.3f}s"
        if extra:
            msg += f" {extra}"
        _log.info(msg)
        if self.enabled:
            self.steps.append({
                "step": name,
                "duration_ms": round(elapsed * 1000, 1),
                "start_offset_ms": round((self._step_start - self._start) * 1000, 1),
            })

    def total_ms(self) -> float:
        return round((time.perf_counter() - self._start) * 1000, 1)

    def bottleneck(self) -> str:
        if not self.steps:
            return ""
        return max(self.steps, key=lambda s: s["duration_ms"])["step"]


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
        self.query_router = QueryRouter()
        self._standard_ranking_policy = StandardRankingPolicy()
        self._standard_source_selector = StandardSourceSelector()
        self._count_strategy = CountQueryStrategy(self)
        self._mixed_strategy = MixedIssueChatStrategy(self)
        self._channels_cache: list[str] = []
        self._users_cache: list[str] = []
        self._assignees_cache: list[str] = []
        self._statuses_cache: list[str] = []
        self._document_types_cache: list[str] = []
        self._latest_event_date: date | None = None
        self._latest_event_dates_by_type: dict[str, date | None] = {}
        self._cache_ts: float = 0.0

    def invalidate_metadata_cache(self) -> None:
        self._channels_cache = []
        self._users_cache = []
        self._assignees_cache = []
        self._statuses_cache = []
        self._document_types_cache = []
        self._latest_event_date = None
        self._latest_event_dates_by_type = {}
        self._cache_ts = 0.0

    async def _get_metadata_lists(self) -> tuple[list[str], list[str]]:
        context = await self._get_metadata_context()
        return context["channels"], context["users"]

    async def _get_metadata_context(self) -> dict[str, object]:
        now = time.monotonic()
        if (
            now - self._cache_ts > _METADATA_CACHE_TTL
            or not self._channels_cache
            or not self._users_cache
            or not self._document_types_cache
        ):
            self._channels_cache = await self.postgres.list_channels()
            self._users_cache = await self.postgres.list_users()
            self._assignees_cache = await self._call_optional(self.postgres, "list_assignees", [])
            self._statuses_cache = await self._call_optional(self.postgres, "list_statuses", [])
            self._document_types_cache = await self._call_optional(
                self.postgres,
                "list_document_types",
                ["chat", "issue"],
            )
            self._latest_event_date = await self._call_optional(
                self.postgres,
                "get_latest_event_date",
                date.today(),
            )
            self._latest_event_dates_by_type = {
                "chat": await self._call_optional(
                    self.postgres,
                    "get_latest_event_date",
                    None,
                    document_type="chat",
                ),
                "issue": await self._call_optional(
                    self.postgres,
                    "get_latest_event_date",
                    None,
                    document_type="issue",
                ),
            }
            self._cache_ts = now
        return {
            "channels": self._channels_cache,
            "users": self._users_cache,
            "assignees": self._assignees_cache,
            "statuses": self._statuses_cache,
            "document_types": self._document_types_cache,
            "latest_event_date": self._latest_event_date,
            "latest_event_dates_by_type": self._latest_event_dates_by_type,
        }

    async def get_facets(self, document_type: str = "all") -> MetadataFacetsResponse:
        normalized_document_type = document_type if document_type in {"all", "chat", "issue"} else "all"
        available_document_types = await self._call_optional(
            self.postgres,
            "list_document_types",
            ["chat", "issue"],
        )
        if normalized_document_type == "all":
            document_types = available_document_types
            channels = await self.postgres.list_channels(document_type="all")
            users = await self.postgres.list_users(document_type="all")
            assignees = await self._call_optional(
                self.postgres,
                "list_assignees",
                [],
                document_type="issue",
            )
            statuses = await self._call_optional(
                self.postgres,
                "list_statuses",
                [],
                document_type="issue",
            )
            latest_event_date = await self._call_optional(
                self.postgres,
                "get_latest_event_date",
                None,
                document_type=None,
            )
        else:
            document_types = [
                dtype
                for dtype in available_document_types
                if dtype == normalized_document_type
            ]
            channels = await self.postgres.list_channels(document_type=normalized_document_type)
            users = await self.postgres.list_users(document_type=normalized_document_type)
            latest_event_date = await self._call_optional(
                self.postgres,
                "get_latest_event_date",
                None,
                document_type=normalized_document_type,
            )
            if normalized_document_type == "issue":
                assignees = await self._call_optional(
                    self.postgres,
                    "list_assignees",
                    [],
                    document_type="issue",
                )
                statuses = await self._call_optional(
                    self.postgres,
                    "list_statuses",
                    [],
                    document_type="issue",
                )
            else:
                assignees = []
                statuses = []

        return MetadataFacetsResponse(
            document_types=document_types,
            channels=channels,
            users=users,
            assignees=assignees,
            statuses=statuses,
            latest_event_date=latest_event_date,
        )

    @staticmethod
    async def _call_optional(component, method_name: str, default, **kwargs):
        method = getattr(component, method_name, None)
        if method is None:
            return default
        return await method(**kwargs)

    async def answer_query(
        self,
        *,
        question: str,
        access_scopes: list[str],
        request_user: str | None,
        top_k: int | None = None,
        debug: bool = False,
        request_filters: QueryRequestFilters | None = None,
    ) -> QueryResponse:
        timer = PipelineTimer(enabled=debug)
        top_k = self.settings.top_k if top_k is None else top_k

        timer.start_step("query_analysis")
        metadata_context = await self._get_metadata_context()
        channels = metadata_context["channels"]
        users = metadata_context["users"]
        assignees = metadata_context["assignees"]
        statuses = metadata_context["statuses"]
        latest_event_date = metadata_context["latest_event_date"]
        latest_event_dates_by_type = metadata_context["latest_event_dates_by_type"]
        reference_today = latest_event_date
        requested_document_type = None
        if request_filters and len(request_filters.document_types) == 1:
            requested_document_type = request_filters.document_types[0]
        else:
            requested_document_type = self.query_analyzer.detect_document_type_hint(
                question,
                channels=channels,
                assignees=assignees,
                statuses=statuses,
            )
        if requested_document_type:
            reference_today = latest_event_dates_by_type.get(requested_document_type) or latest_event_date
        analysis = self.query_analyzer.analyze(
            question,
            access_scopes=access_scopes or self.settings.parsed_default_access_scopes,
            channels=channels,
            users=users,
            assignees=assignees,
            statuses=statuses,
            reference_today=reference_today,
        )
        analysis.filters = self._merge_request_filters(analysis.filters, request_filters)
        timer.end_step("query_analysis")

        route_decision = self.query_router.route(question=question, analysis=analysis)
        if route_decision.route == "count":
            return await self._count_strategy.execute(
                question=question,
                analysis=analysis,
                top_k=top_k,
                request_user=request_user,
                debug=debug,
                timer=timer,
                metadata_context=metadata_context,
            )
        if route_decision.route == "mixed_issue_chat":
            return await self._mixed_strategy.execute(
                question=question,
                analysis=analysis,
                top_k=top_k,
                request_user=request_user,
                debug=debug,
                timer=timer,
                metadata_context=metadata_context,
            )
        return await self._execute_standard_query(
            question=question,
            analysis=analysis,
            top_k=top_k,
            request_user=request_user,
            debug=debug,
            timer=timer,
            metadata_context=metadata_context,
            latest_event_date=latest_event_date,
        )

    async def _execute_standard_query(
        self,
        *,
        question: str,
        analysis: QueryAnalysis,
        top_k: int,
        request_user: str | None,
        debug: bool,
        timer: PipelineTimer,
        metadata_context: dict[str, object],
        latest_event_date: date | None,
    ) -> QueryResponse:
        aggregate_context = None
        if self._should_attach_aggregate_context(analysis):
            timer.start_step("aggregate_summary")
            aggregate_context = await self._build_aggregate_context(analysis, top_k)
            timer.end_step(
                "aggregate_summary",
                f"({aggregate_context['matched_count']} matches)" if aggregate_context else "",
            )

        timer.start_step("embedding")
        query_embedding = (await self.embedding_provider.embed_texts([analysis.search_text or analysis.clean_question]))[0]
        timer.end_step("embedding")

        timer.start_step("pgvector_search")
        seed_chunks = await self.postgres.search_chunks(query_embedding, analysis.filters, top_k)
        timer.end_step("pgvector_search", f"({len(seed_chunks)} chunks)")

        graph_seeded_chunks: list[RetrievedChunk] = []
        if self.settings.graph_entity_seed_enabled and analysis.entities:
            timer.start_step("graph_entity_seed")
            try:
                graph_seed_ids = await self.neo4j.find_chunks_by_entities(
                    analysis.entities,
                    limit=self.settings.graph_entity_seed_limit,
                )
            except Exception:
                _log.warning("graph_entity_seed failed, falling back to vector only", exc_info=True)
                graph_seed_ids = []
            seed_chunk_ids = {c.chunk_id for c in seed_chunks}
            new_graph_ids = {cid for cid in graph_seed_ids if cid not in seed_chunk_ids}
            if new_graph_ids:
                graph_seeded_chunks = [
                    chunk
                    for chunk in await self.postgres.get_chunks_by_ids(new_graph_ids)
                    if self._chunk_matches_filters(chunk, analysis.filters)
                ]
                for chunk in graph_seeded_chunks:
                    chunk.retrieval_source = "graph_entity"
            timer.end_step("graph_entity_seed", f"({len(graph_seeded_chunks)} chunks)")

        if not seed_chunks and not graph_seeded_chunks:
            debug_data = None
            if debug:
                debug_data = await self._build_debug_data(
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
                    route="standard",
                    strategy="standard_query",
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

        timer.start_step("neo4j_graph_expand")
        all_seed_ids = [c.chunk_id for c in seed_chunks] + [c.chunk_id for c in graph_seeded_chunks]
        all_seed_id_set = set(all_seed_ids)
        try:
            expansions = await self.neo4j.expand_from_seed_chunks(
                all_seed_ids,
                next_window=self.settings.graph_next_window,
            )
        except Exception:
            _log.warning("neo4j_graph_expand failed, skipping expansion", exc_info=True)
            expansions = {}
        timer.end_step("neo4j_graph_expand")

        multihop_ids: set[str] = set()
        multihop_seed_ids = self._select_multihop_seed_ids(all_seed_ids)
        if self.settings.graph_multihop_enabled and analysis.intent in ("relationship", "summary", "timeline"):
            timer.start_step("multihop_expand")
            try:
                if analysis.intent in ("relationship", "summary") and analysis.entities:
                    cooccurrence_rows = await self.neo4j.expand_via_entity_cooccurrence(
                        multihop_seed_ids,
                        entity_names=analysis.entities,
                        limit=self.settings.graph_entity_expansion_limit,
                    )
                    multihop_ids.update(row["neighbor_id"] for row in cooccurrence_rows)
                if analysis.intent in ("relationship", "timeline") and multihop_seed_ids:
                    author_rows = await self.neo4j.expand_via_same_author(
                        multihop_seed_ids,
                        limit=self.settings.graph_author_expansion_limit,
                    )
                    multihop_ids.update(row["neighbor_id"] for row in author_rows)
            except Exception:
                _log.warning("multihop_expand failed, skipping multi-hop", exc_info=True)
            multihop_ids -= all_seed_id_set
            timer.end_step("multihop_expand", f"({len(multihop_ids)} chunks)")

        timer.start_step("pg_expand_fetch")
        expanded_ids = {
            chunk_id
            for expansion in expansions.values()
            for chunk_id in expansion.expanded_chunk_ids
            if chunk_id not in all_seed_id_set
        }
        expanded_ids |= multihop_ids
        expanded_chunks = [
            chunk
            for chunk in await self.postgres.get_chunks_by_ids(expanded_ids)
            if self._chunk_matches_filters(chunk, analysis.filters)
        ]
        timer.end_step("pg_expand_fetch", f"({len(expanded_chunks)} extra)")

        timer.start_step("ranking")
        ranked_chunks = self._rank_chunks(
            seed_chunks,
            expanded_chunks,
            expansions,
            analysis,
            graph_seeded_chunks,
            latest_event_date=latest_event_date,
        )
        ranked_chunks = self._apply_special_keyword_grounding(ranked_chunks, analysis)
        selected_chunks = self._select_source_chunks(
            ranked_chunks,
            analysis,
            top_k,
            aggregate_context=aggregate_context,
        )
        sources = [self._to_source(chunk) for chunk in selected_chunks]
        timer.end_step("ranking")

        if not sources:
            debug_data = None
            if debug:
                debug_data = await self._build_debug_data(
                    timer=timer,
                    analysis=analysis,
                    ranked_chunks=ranked_chunks,
                    top_k=top_k,
                    subgraph_rows=[],
                    community_rows=[],
                    seed_chunks=seed_chunks,
                    graph_seeded_chunks=graph_seeded_chunks,
                    multihop_ids=list(multihop_ids),
                    expanded_ids=list(expanded_ids),
                    route="standard",
                    strategy="standard_query",
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

        subgraph_context = ""
        subgraph_rows: list[dict] = []
        timer.start_step("subgraph_extract")
        try:
            top_chunk_ids = [s.chunk_id for s in sources[:5]]
            subgraph_rows = await self.neo4j.extract_subgraph(top_chunk_ids)
            subgraph_context = self._build_subgraph_context(subgraph_rows)
        except Exception:
            _log.warning("subgraph_extract failed, skipping subgraph context", exc_info=True)
            subgraph_context = ""
        timer.end_step("subgraph_extract")

        community_context = ""
        community_rows: list[dict] = []
        if self.settings.community_detection_enabled and analysis.entities:
            timer.start_step("community_lookup")
            try:
                community_rows = await self.neo4j.find_communities_for_entities(analysis.entities)
                if community_rows:
                    lines = []
                    for row in community_rows[:3]:
                        matched = ", ".join(row.get("matched_entities", []))
                        summary = row.get("summary", "")
                        lines.append(f"- Community {row['community_id']}: {summary} (관련 엔티티: {matched})")
                    community_context = "\n\nCommunity Context:\n" + "\n".join(lines)
            except Exception:
                _log.warning("community_lookup failed, skipping community context", exc_info=True)
                community_context = ""
            timer.end_step("community_lookup")

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            question,
            sources,
            request_user,
            analysis=analysis,
            channels=metadata_context["channels"],
            users=metadata_context["users"],
            assignees=metadata_context["assignees"],
            statuses=metadata_context["statuses"],
            subgraph_context=subgraph_context + community_context,
            aggregate_context=aggregate_context,
        )

        timer.start_step("llm_generate")
        try:
            llm_response = await self.codex_proxy.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                metadata={
                    "request_user": request_user,
                    "retrieval_strategy": "filter_pgvector_graph_hybrid",
                    "top_k": top_k,
                    "query_route": "standard",
                },
            )
            timer.end_step("llm_generate")
            debug_data = None
            if debug:
                debug_data = await self._build_debug_data(
                    timer=timer,
                    analysis=analysis,
                    ranked_chunks=ranked_chunks,
                    top_k=top_k,
                    subgraph_rows=subgraph_rows,
                    community_rows=community_rows,
                    seed_chunks=seed_chunks,
                    graph_seeded_chunks=graph_seeded_chunks,
                    multihop_ids=list(multihop_ids),
                    expanded_ids=list(expanded_ids),
                    route="standard",
                    strategy="standard_query",
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
                debug_data = await self._build_debug_data(
                    timer=timer,
                    analysis=analysis,
                    ranked_chunks=ranked_chunks,
                    top_k=top_k,
                    subgraph_rows=subgraph_rows,
                    community_rows=community_rows,
                    seed_chunks=seed_chunks,
                    graph_seeded_chunks=graph_seeded_chunks,
                    multihop_ids=list(multihop_ids),
                    expanded_ids=list(expanded_ids),
                    route="standard",
                    strategy="standard_query",
                    count_kind="none",
                    chat_match_count=len([source for source in sources if source.document_type == "chat"]),
                    issue_match_count=len([source for source in sources if source.document_type == "issue"]),
                )
            return QueryResponse(
                question=question,
                answer=(
                    self._build_aggregate_fallback_answer(aggregate_context, sources)
                    if aggregate_context is not None
                    else self._build_fallback_answer(analysis, sources, top_k)
                ),
                retrieval_strategy="filter_pgvector_graph_hybrid",
                answer_mode="fallback_sources_only",
                sources=sources,
                debug=debug_data,
            )

    async def _build_debug_data(
        self,
        timer: PipelineTimer,
        analysis: QueryAnalysis,
        ranked_chunks: list[RetrievedChunk],
        top_k: int,
        subgraph_rows: list[dict],
        community_rows: list[dict],
        seed_chunks: list[RetrievedChunk],
        graph_seeded_chunks: list[RetrievedChunk],
        multihop_ids: list[str],
        expanded_ids: list[str],
        route: str,
        strategy: str,
        count_kind: str,
        chat_match_count: int,
        issue_match_count: int,
    ) -> DebugData:
        # Timing
        timing = [PipelineTiming(**s) for s in timer.steps]
        total_time_ms = timer.total_ms()
        bottleneck_step = timer.bottleneck()

        # Score breakdowns
        score_breakdowns = [
            ScoreBreakdown(
                chunk_id=c.chunk_id,
                vector_score=round(c.vector_score, 4),
                graph_score=round(c.graph_score, 4),
                entity_score=round(c.entity_score, 4),
                metadata_score=round(c.metadata_score, 4),
                recency_score=round(c.recency_score, 4),
                final_score=round(c.final_score, 4),
                retrieval_source=c.retrieval_source,
            )
            for c in ranked_chunks[:top_k]
        ]

        # Intent weights
        weights = self._get_weights(analysis)
        intent_weights = IntentWeights(
            intent=analysis.intent,
            vector=weights[0], graph=weights[1], entity=weights[2],
            metadata=weights[3], recency=weights[4],
        )

        # Subgraph nodes & edges
        subgraph_nodes: list[SubgraphNode] = []
        subgraph_edges: list[SubgraphEdge] = []
        seen_nodes: set[str] = set()
        for row in subgraph_rows:
            src = row.get("source", "")
            target = row.get("target_name", "")
            rel = row.get("relationship", "")
            target_type = row.get("target_type", "")
            if src and src not in seen_nodes:
                seen_nodes.add(src)
                subgraph_nodes.append(SubgraphNode(id=src, label=src[:12], type="Chunk"))
            if target and target not in seen_nodes:
                seen_nodes.add(target)
                subgraph_nodes.append(SubgraphNode(id=target, label=target, type=target_type))
            if src and target:
                subgraph_edges.append(SubgraphEdge(source=src, target=target, relationship=rel))

        # Source comparison IDs
        vector_only_ids = [c.chunk_id for c in seed_chunks]
        graph_entity_ids = [c.chunk_id for c in graph_seeded_chunks]

        # Entity co-occurrence + mention counts (debug-only queries)
        cooccurrence_edges: list[CooccurrenceEdge] = []
        entity_mention_counts: dict[str, int] = {}
        if analysis.entities:
            try:
                cooc_rows = await self.neo4j.get_entity_cooccurrence_network(analysis.entities, limit=50)
                cooccurrence_edges = [
                    CooccurrenceEdge(
                        entity_a=r["entity_a"], entity_b=r["entity_b"],
                        shared_chunk_count=r["shared_chunk_count"],
                    )
                    for r in cooc_rows
                ]
            except Exception:
                pass
            try:
                entity_mention_counts = await self.neo4j.get_entity_mention_counts(analysis.entities)
            except Exception:
                pass

        # Community clusters
        community_clusters = [
            CommunityCluster(
                community_id=str(r.get("community_id", "")),
                summary=r.get("summary", ""),
                entities=r.get("matched_entities", []),
            )
            for r in community_rows
        ]

        return DebugData(
            timing=timing,
            total_time_ms=total_time_ms,
            bottleneck_step=bottleneck_step,
            score_breakdowns=score_breakdowns,
            intent_weights=intent_weights,
            subgraph_nodes=subgraph_nodes,
            subgraph_edges=subgraph_edges,
            query_entities=analysis.entities,
            seed_chunk_ids=vector_only_ids,
            graph_seeded_chunk_ids=graph_entity_ids,
            expanded_chunk_ids=list(expanded_ids),
            entity_mention_counts=entity_mention_counts,
            cooccurrence_edges=cooccurrence_edges,
            community_clusters=community_clusters,
            vector_only_ids=vector_only_ids,
            graph_entity_ids=graph_entity_ids,
            multihop_ids=multihop_ids,
            detected_intent=analysis.intent,
            clean_question=analysis.clean_question,
            route=route,
            strategy=strategy,
            count_kind=count_kind,
            chat_match_count=chat_match_count,
            issue_match_count=issue_match_count,
        )

    def _rank_chunks(
        self,
        seed_chunks: list[RetrievedChunk],
        expanded_chunks: list[RetrievedChunk],
        expansions: dict[str, object],
        analysis: QueryAnalysis,
        graph_seeded_chunks: list[RetrievedChunk] | None = None,
        latest_event_date: date | None = None,
    ) -> list[RetrievedChunk]:
        return self._standard_ranking_policy.rank(
            seed_chunks=seed_chunks,
            expanded_chunks=expanded_chunks,
            analysis=analysis,
            context=RankingContext(
                expansions=expansions,
                latest_event_date=latest_event_date,
                lane="standard",
            ),
            graph_seeded_chunks=graph_seeded_chunks,
        )

    async def _maybe_attach_chat_candidates(
        self,
        ranked_chunks: list[RetrievedChunk],
        analysis: QueryAnalysis,
        *,
        query_embedding: list[float],
        latest_event_date: date | None,
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not self._looks_like_related_chat_query(analysis):
            return ranked_chunks
        if analysis.filters.all_document_types and "chat" not in analysis.filters.all_document_types:
            return ranked_chunks
        if any(chunk.document_type == "chat" for chunk in ranked_chunks):
            return ranked_chunks

        strict_filters = analysis.filters.model_copy(deep=True)
        strict_filters.document_types = ["chat"]
        strict_filters.assignees = []
        strict_filters.statuses = []

        appended_chunks = await self._search_chat_candidates(
            query_embedding=query_embedding,
            analysis=analysis,
            filters=strict_filters,
            latest_event_date=latest_event_date,
            top_k=top_k,
            score_penalty=0.0,
            existing_ids={chunk.chunk_id for chunk in ranked_chunks},
        )
        if appended_chunks:
            return sorted([*ranked_chunks, *appended_chunks], key=lambda item: item.final_score, reverse=True)
        return ranked_chunks

    async def _search_chat_candidates(
        self,
        *,
        query_embedding: list[float],
        analysis: QueryAnalysis,
        filters: QueryFilters,
        latest_event_date: date | None,
        top_k: int,
        score_penalty: float,
        existing_ids: set[str],
    ) -> list[RetrievedChunk]:
        chat_candidates = await self.postgres.search_chunks(query_embedding, filters, max(3, top_k))
        if not chat_candidates:
            return []

        has_date_filter = filters.date_from is not None
        weights = self._get_weights(analysis)
        appended: list[RetrievedChunk] = []
        for chunk in chat_candidates:
            if chunk.chunk_id in existing_ids:
                continue
            if not self._chunk_matches_filters(chunk, filters):
                continue
            chunk.retrieval_source = "chat_vector_fallback"
            chunk.graph_score = 0.0
            chunk.entity_overlap_score = self._entity_overlap_score(chunk, analysis)
            chunk.entity_score = chunk.entity_overlap_score
            chunk.metadata_score = self._metadata_score(chunk, analysis)
            chunk.recency_score = 0.0 if has_date_filter else self._recency_score(
                chunk.message_date,
                latest_event_date,
            )
            chunk.final_score = max(0.0, self._combined_score(chunk, weights) - score_penalty)
            appended.append(chunk)
        return appended

    @staticmethod
    def _chat_coverage_note(
        analysis: QueryAnalysis | None,
        sources: list[QuerySource],
    ) -> str:
        if analysis is None or not RetrievalService._looks_like_related_chat_query(analysis):
            return ""

        matching_chat_sources = [source for source in sources if source.document_type == "chat"]
        if matching_chat_sources:
            return (
                "\n\nChat Coverage:\n"
                f"- matching_chat_sources: {len(matching_chat_sources)}\n"
                "- note: 요청 조건과 기간에 맞는 채팅 근거가 포함되어 있습니다."
            )

        note = "요청 조건에 맞는 채팅 근거가 없어 이슈 근거만 제공합니다."
        if analysis.filters.date_from and analysis.filters.date_to:
            if analysis.filters.date_from == analysis.filters.date_to:
                note = (
                    f"{analysis.filters.date_from.isoformat()} 기준으로 일치하는 채팅 근거가 없어 "
                    "이슈 근거만 제공합니다."
                )
            else:
                note = (
                    f"{analysis.filters.date_from.isoformat()}~{analysis.filters.date_to.isoformat()} 기간에 "
                    "일치하는 채팅 근거가 없어 이슈 근거만 제공합니다."
                )
        return (
            "\n\nChat Coverage:\n"
            "- matching_chat_sources: 0\n"
            f"- note: {note}"
        )

    @staticmethod
    def _should_prefer_aggregate_samples(
        analysis: QueryAnalysis,
        aggregate_context: dict[str, object] | None,
    ) -> bool:
        return StandardSourceSelector().should_prefer_aggregate_samples(analysis, aggregate_context)

    @staticmethod
    def _dedupe_source_candidates(
        chunks: list[RetrievedChunk],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        return dedupe_source_candidates(chunks, top_k=top_k)

    @classmethod
    def _aggregate_sample_chunks(
        cls,
        aggregate_context: dict[str, object] | None,
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        return aggregate_sample_chunks(aggregate_context, top_k=top_k)

    @classmethod
    def _prioritize_related_chat_sources(
        cls,
        selected: list[RetrievedChunk],
        ranked_chunks: list[RetrievedChunk],
        analysis: QueryAnalysis,
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not cls._looks_like_related_chat_query(analysis):
            return cls._dedupe_source_candidates(selected, top_k=top_k)
        if analysis.filters.all_document_types and "chat" not in analysis.filters.all_document_types:
            return cls._dedupe_source_candidates(selected, top_k=top_k)

        chat_pool = cls._dedupe_source_candidates(
            [chunk for chunk in ranked_chunks if chunk.document_type == "chat"],
            top_k=top_k,
        )
        if not chat_pool:
            return cls._dedupe_source_candidates(selected, top_k=top_k)

        selected = cls._dedupe_source_candidates(selected, top_k=top_k)
        if cls._looks_like_mixed_issue_chat_summary(analysis):
            issue_pool = [chunk for chunk in selected if chunk.document_type != "chat"]
            existing_chat = [chunk for chunk in selected if chunk.document_type == "chat"]
            chat_pool = cls._dedupe_source_candidates([*existing_chat, *chat_pool], top_k=top_k)
            blended: list[RetrievedChunk] = []
            while len(blended) < top_k and (issue_pool or chat_pool):
                if issue_pool:
                    blended.append(issue_pool.pop(0))
                if len(blended) >= top_k:
                    break
                if chat_pool:
                    blended.append(chat_pool.pop(0))
            if len(blended) < top_k:
                blended.extend(issue_pool[: top_k - len(blended)])
            if len(blended) < top_k:
                blended.extend(chat_pool[: top_k - len(blended)])
            return cls._dedupe_source_candidates(blended, top_k=top_k)

        if any(chunk.document_type == "chat" for chunk in selected[: min(3, len(selected))]):
            return selected

        best_chat = chat_pool[0]
        reordered = [chunk for chunk in selected if chunk.chunk_id != best_chat.chunk_id]
        if len(reordered) >= top_k:
            reordered = reordered[: top_k - 1]
        insert_at = 1 if reordered else 0
        reordered.insert(min(insert_at, len(reordered)), best_chat)
        return cls._dedupe_source_candidates(reordered, top_k=top_k)

    @staticmethod
    def _query_match_terms(analysis: QueryAnalysis) -> list[str]:
        return query_match_terms(analysis)

    @classmethod
    def _query_phrase_candidates(cls, analysis: QueryAnalysis) -> list[str]:
        return query_phrase_candidates(analysis)

    @staticmethod
    def _chunk_search_text(chunk: RetrievedChunk) -> str:
        return chunk_search_text(chunk)

    @classmethod
    def _lexical_coverage_score(cls, chunk: RetrievedChunk, analysis: QueryAnalysis) -> float:
        return StandardRankingPolicy().lexical_coverage_score(chunk, analysis)

    @classmethod
    def _chat_relevance_bonus(cls, chunk: RetrievedChunk, analysis: QueryAnalysis) -> float:
        return StandardRankingPolicy().chat_relevance_bonus(chunk, analysis)

    @classmethod
    def _strict_lexical_groups(cls, analysis: QueryAnalysis) -> list[tuple[str, ...]]:
        return strict_lexical_groups(analysis)

    @classmethod
    def _exact_special_groups(cls, analysis: QueryAnalysis) -> list[tuple[str, ...]]:
        return exact_special_groups(analysis)

    @classmethod
    def _chunk_matches_alias_group(cls, chunk: RetrievedChunk, alias_group: tuple[str, ...]) -> bool:
        return chunk_matches_alias_group(chunk, alias_group)

    @classmethod
    def _entity_overlap_score(cls, chunk: RetrievedChunk, analysis: QueryAnalysis) -> float:
        return StandardRankingPolicy().entity_overlap_score(chunk, analysis)

    def _metadata_score(self, chunk: RetrievedChunk, analysis: QueryAnalysis) -> float:
        return self._standard_ranking_policy.metadata_score(chunk, analysis)

    @staticmethod
    def _chunk_matches_filters(chunk: RetrievedChunk, filters) -> bool:
        if filters.access_scopes and not (set(chunk.access_scopes) & set(filters.access_scopes)):
            return False
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
        if filters.date_from and chunk.message_date < filters.date_from:
            return False
        if filters.date_to and chunk.message_date > filters.date_to:
            return False
        return True

    def _recency_score(self, chunk_date: date, latest_event_date: date | None = None) -> float:
        return self._standard_ranking_policy.recency_score(chunk_date, latest_event_date)

    @staticmethod
    def _get_weights(analysis: QueryAnalysis) -> tuple[float, float, float, float, float]:
        return StandardRankingPolicy().get_weights(analysis)

    @staticmethod
    def _looks_like_flow_query(analysis: QueryAnalysis) -> bool:
        return looks_like_flow_query(analysis)

    @staticmethod
    def _looks_like_count_query(question: str) -> bool:
        return looks_like_count_query(question)

    @staticmethod
    def _looks_like_related_chat_query(analysis: QueryAnalysis) -> bool:
        return looks_like_related_chat_query(analysis)

    @staticmethod
    def _select_multihop_seed_ids(chunk_ids: list[str]) -> list[str]:
        return list(chunk_ids[:_MULTIHOP_SEED_LIMIT])

    @staticmethod
    def _should_attach_aggregate_context(analysis: QueryAnalysis) -> bool:
        if looks_like_mixed_issue_chat_summary(analysis):
            return True
        if not (analysis.detected_document_type == "issue" or "issue" in analysis.filters.all_document_types):
            return False
        if analysis.intent == "list":
            return True
        if analysis.intent == "summary":
            return looks_like_generic_issue_summary(analysis)
        return False

    @staticmethod
    def _looks_like_generic_issue_summary(analysis: QueryAnalysis) -> bool:
        return looks_like_generic_issue_summary(analysis)

    @staticmethod
    def _looks_like_mixed_issue_chat_summary(analysis: QueryAnalysis) -> bool:
        return looks_like_mixed_issue_chat_summary(analysis)

    @classmethod
    def _apply_special_keyword_grounding(
        cls,
        ranked_chunks: list[RetrievedChunk],
        analysis: QueryAnalysis,
    ) -> list[RetrievedChunk]:
        return StandardRankingPolicy().apply_special_keyword_grounding(ranked_chunks, analysis)

    @classmethod
    def _select_source_chunks(
        cls,
        ranked_chunks: list[RetrievedChunk],
        analysis: QueryAnalysis,
        top_k: int,
        aggregate_context: dict[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        return StandardSourceSelector().select(
            ranked_chunks=ranked_chunks,
            analysis=analysis,
            top_k=top_k,
            aggregate_context=aggregate_context,
        )

    @staticmethod
    def _combined_score(chunk: RetrievedChunk, weights: tuple[float, float, float, float, float]) -> float:
        w_vector, w_graph, w_entity, w_metadata, w_recency = weights
        return (
            w_vector * chunk.vector_score
            + w_graph * chunk.graph_score
            + w_entity * chunk.entity_score
            + w_metadata * chunk.metadata_score
            + w_recency * chunk.recency_score
        )

    def _to_source(self, chunk: RetrievedChunk) -> QuerySource:
        if chunk.document_type == "issue":
            content = chunk.chunk_text
            chunk_kind = chunk.metadata.get("chunk_kind")
            source_badge = "issue-flow" if chunk_kind == "analysis_flow" else "issue-overview"
        else:
            content = "\n".join(chunk.metadata.get("original_lines", [])[:2]) or chunk.chunk_text
            source_badge = "chat"
        return QuerySource(
            chunk_id=chunk.chunk_id,
            score=round(chunk.final_score, 4),
            content=content,
            document_type=chunk.document_type,
            source_badge=source_badge,
            graph_neighbors=chunk.graph_neighbors,
            channel=chunk.channel,
            user_name=chunk.user_name,
            message_date=chunk.message_date,
            issue_title=chunk.metadata.get("issue_title"),
            assignee=chunk.metadata.get("assignee"),
            status=chunk.metadata.get("status"),
            flow_name=chunk.metadata.get("flow_name"),
        )

    def _build_system_prompt(self) -> str:
        return (
            "당신은 한국어 조직 채팅 로그를 분석하는 GraphRAG 어시스턴트다.\n"
            "제공된 근거 밖의 추론은 최소화하고, 가능하면 날짜/채널/사용자 맥락을 유지해 답하라.\n"
            "Issue source에는 status, assignee, issue_title 같은 구조화 메타데이터가 포함될 수 있으며 이는 신뢰 가능한 근거다.\n"
            "Aggregate Summary가 있으면 total_matches를 전체 집계의 기준값으로 사용하고, Source는 예시로만 사용하라.\n"
            "특히 summary/list/aggregate 응답에서 Aggregate Summary가 있으면 total_matches를 문장이나 표 상단에 명시하라.\n"
            "Chat Coverage 섹션에 matching_chat_sources: 0 이 있으면, 요청 기간/조건에 맞는 채팅 근거가 없음을 명시하고 이슈 근거만으로 설명하라.\n"
            "근거가 부족하면 단정하지 말고 제한사항을 짧게 밝혀라."
        )

    @staticmethod
    def _build_subgraph_context(subgraph_rows: list[dict[str, str]]) -> str:
        """Build a structured graph context string from subgraph triples."""
        if not subgraph_rows:
            return ""
        from collections import defaultdict
        entity_chunks: defaultdict[str, set[str]] = defaultdict(set)
        entity_authors: defaultdict[str, set[str]] = defaultdict(set)
        author_chunks: defaultdict[str, list[str]] = defaultdict(list)

        for row in subgraph_rows:
            rel = row.get("relationship", "")
            target = row.get("target_name", "")
            source = row.get("source", "")
            target_type = row.get("target_type", "")
            if rel == "MENTIONS" and target_type == "Entity":
                entity_chunks[target].add(source)
            elif rel == "SENT_BY" and target_type == "User":
                author_chunks[target].append(source)
                for r2 in subgraph_rows:
                    if r2["source"] == source and r2.get("relationship") == "MENTIONS":
                        entity_authors[r2["target_name"]].add(target)

        lines = []
        sorted_entities = sorted(entity_chunks.items(), key=lambda x: len(x[1]), reverse=True)
        for entity, chunks in sorted_entities[:8]:
            authors = entity_authors.get(entity, set())
            author_str = f" ({', '.join(sorted(authors))})" if authors else ""
            lines.append(f"- {entity}: {len(chunks)}개 청크에서 언급{author_str}")

        for author, chunks in sorted(author_chunks.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
            lines.append(f"- {author}: {len(chunks)}개 청크 작성")

        if not lines:
            return ""
        return "\n\nGraph Context:\n" + "\n".join(lines)

    def _build_user_prompt(
        self,
        question: str,
        sources: list[QuerySource],
        request_user: str | None,
        analysis: QueryAnalysis | None = None,
        channels: list[str] | None = None,
        users: list[str] | None = None,
        assignees: list[str] | None = None,
        statuses: list[str] | None = None,
        subgraph_context: str = "",
        aggregate_context: dict[str, object] | None = None,
    ) -> str:
        rendered_sources = "\n\n".join(
            self._render_source_for_prompt(index, source)
            for index, source in enumerate(sources)
        )
        requester = request_user or "anonymous"

        metadata_section = ""
        if analysis and analysis.intent in ("aggregate", "relationship", "list"):
            parts = []
            if channels:
                parts.append(f"전체 채널 목록 ({len(channels)}개): {', '.join(channels)}")
            if users:
                parts.append(f"전체 사용자 목록 ({len(users)}명): {', '.join(users)}")
            if analysis.detected_document_type == "issue" or "issue" in analysis.filters.all_document_types:
                if assignees:
                    parts.append(f"담당자 목록 ({len(assignees)}명): {', '.join(assignees)}")
                if statuses:
                    parts.append(f"상태 목록 ({len(statuses)}개): {', '.join(statuses)}")
            if parts:
                metadata_section = "\n\nDatabase Metadata:\n" + "\n".join(parts)

        aggregate_section = ""
        if aggregate_context is not None:
            aggregate_filters = aggregate_context.get("applied_filters") or []
            aggregate_section = (
                "\n\nAggregate Summary:\n"
                f"- total_matches: {aggregate_context.get('matched_count', 0)}\n"
                f"- count_basis: {aggregate_context.get('count_basis', 'matching_records')}\n"
                f"- applied_filters: {', '.join(aggregate_filters) if aggregate_filters else 'none'}"
            )

        coverage_section = self._chat_coverage_note(analysis, sources)

        return (
            f"Request User: {requester}\n"
            f"Question: {question}\n\n"
            f"Evidence:\n{rendered_sources}"
            f"{subgraph_context}"
            f"{aggregate_section}"
            f"{coverage_section}"
            f"{metadata_section}"
        )

    @staticmethod
    def _render_source_for_prompt(index: int, source: QuerySource) -> str:
        lines = [
            f"[Source {index + 1}]",
            f"Date: {source.message_date}",
            f"Channel: {source.channel}",
            f"User: {source.user_name}",
            f"Document Type: {source.document_type}",
            f"Source Badge: {source.source_badge}",
        ]
        if source.issue_title:
            lines.append(f"Issue Title: {source.issue_title}")
        if source.assignee:
            lines.append(f"Assignee: {source.assignee}")
        if source.status:
            lines.append(f"Status: {source.status}")
        if source.flow_name:
            lines.append(f"Flow Name: {source.flow_name}")
        lines.append(f"Content:\n{source.content}")
        lines.append(f"Graph neighbors: {', '.join(source.graph_neighbors)}")
        return "\n".join(lines)

    def _build_fallback_answer(
        self, analysis: QueryAnalysis, sources: list[QuerySource], top_k: int = 10,
    ) -> str:
        if not sources:
            return "조건에 맞는 근거를 찾지 못했습니다."
        header_parts = []
        if analysis.filters.all_channels:
            header_parts.append(f"채널={', '.join(analysis.filters.all_channels)}")
        if analysis.filters.user_names:
            header_parts.append(f"사용자={', '.join(analysis.filters.user_names)}")
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
        lines = ["LLM 응답을 생성하지 못해 검색 근거만 반환합니다.", header]
        coverage_note = self._chat_coverage_note(analysis, sources)
        if coverage_note and "matching_chat_sources: 0" in coverage_note:
            lines.append("요청 기간/조건에 맞는 채팅 근거가 없어 이슈 근거만 제공합니다.")
        lines.extend(evidence_lines)
        return "\n".join(lines)

    @staticmethod
    def _build_aggregate_fallback_answer(
        aggregate_context: dict[str, object],
        sources: list[QuerySource],
    ) -> str:
        matched_count = int(aggregate_context.get("matched_count", 0))
        lines = [f"조건에 맞는 전체 집계 결과는 총 {matched_count}건입니다."]
        if sources:
            lines.append("아래는 최신 예시 근거입니다.")
            lines.extend(
                f"- {source.message_date} {source.channel} {source.user_name}: {source.content.splitlines()[0]}"
                for source in sources[:5]
            )
        return "\n".join(lines)

    async def _build_aggregate_context(
        self,
        analysis: QueryAnalysis,
        top_k: int,
    ) -> dict[str, object] | None:
        if analysis.intent not in ("aggregate", "list", "summary"):
            return None
        aggregate_filters = self._aggregate_filters_for_analysis(analysis)
        summary = await self._call_optional(
            self.postgres,
            "summarize_filtered_results",
            None,
            filters=aggregate_filters,
            limit=min(max(top_k * 3, top_k), 30),
        )
        if not summary:
            return None
        applied_filters = self._render_applied_filters(aggregate_filters)
        summary["applied_filters"] = applied_filters
        return summary

    @classmethod
    def _aggregate_filters_for_analysis(cls, analysis: QueryAnalysis) -> QueryFilters:
        filters = analysis.filters.model_copy(deep=True)
        if cls._looks_like_mixed_issue_chat_summary(analysis):
            filters.document_types = ["issue"]
            filters.channels = []
            filters.channel = None
            filters.user_names = []
        return filters

    @staticmethod
    def _render_applied_filters(filters: QueryFilters) -> list[str]:
        applied_filters: list[str] = []
        if filters.statuses:
            applied_filters.append(f"statuses={','.join(filters.statuses)}")
        if filters.assignees:
            applied_filters.append(f"assignees={','.join(filters.assignees)}")
        if filters.all_document_types:
            applied_filters.append(f"document_types={','.join(filters.all_document_types)}")
        if filters.date_from:
            applied_filters.append(f"date_from={filters.date_from.isoformat()}")
        if filters.date_to:
            applied_filters.append(f"date_to={filters.date_to.isoformat()}")
        return applied_filters

    @staticmethod
    def _merge_request_filters(
        base_filters,
        request_filters: QueryRequestFilters | None,
    ):
        if request_filters is None:
            return base_filters

        merged = base_filters.model_copy(deep=True)
        if request_filters.document_types:
            merged.document_types = list(request_filters.document_types)
        if request_filters.channels:
            merged.channels = list(request_filters.channels)
            merged.channel = request_filters.channels[0]
        if request_filters.user_names:
            merged.user_names = list(request_filters.user_names)
        if request_filters.assignees:
            merged.assignees = list(request_filters.assignees)
        if request_filters.statuses:
            merged.statuses = list(request_filters.statuses)
        if request_filters.date_from is not None:
            merged.date_from = request_filters.date_from
        if request_filters.date_to is not None:
            merged.date_to = request_filters.date_to
        return merged
