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
    PipelineTiming,
    QueryAnalysis,
    QueryResponse,
    QuerySource,
    RetrievedChunk,
    ScoreBreakdown,
    SubgraphEdge,
    SubgraphNode,
)
from app.services.query_analyzer import QueryAnalyzer

_METADATA_CACHE_TTL = 300  # 5 minutes


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
        self._channels_cache: list[str] = []
        self._users_cache: list[str] = []
        self._cache_ts: float = 0.0

    def invalidate_metadata_cache(self) -> None:
        self._channels_cache = []
        self._users_cache = []
        self._cache_ts = 0.0

    async def _get_metadata_lists(self) -> tuple[list[str], list[str]]:
        now = time.monotonic()
        if (
            now - self._cache_ts > _METADATA_CACHE_TTL
            or not self._channels_cache
            or not self._users_cache
        ):
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
        debug: bool = False,
    ) -> QueryResponse:
        timer = PipelineTimer(enabled=debug)
        top_k = top_k or self.settings.top_k

        timer.start_step("query_analysis")
        channels, users = await self._get_metadata_lists()
        analysis = self.query_analyzer.analyze(
            question,
            access_scopes=access_scopes or self.settings.parsed_default_access_scopes,
            channels=channels,
            users=users,
        )
        timer.end_step("query_analysis")

        timer.start_step("embedding")
        query_embedding = (await self.embedding_provider.embed_texts([analysis.clean_question]))[0]
        timer.end_step("embedding")

        timer.start_step("pgvector_search")
        seed_chunks = await self.postgres.search_chunks(query_embedding, analysis.filters, top_k)
        timer.end_step("pgvector_search", f"({len(seed_chunks)} chunks)")

        # --- Entity-seeded graph retrieval (Phase 1) ---
        graph_seeded_chunks: list[RetrievedChunk] = []
        if self.settings.graph_entity_seed_enabled and analysis.entities:
            timer.start_step("graph_entity_seed")
            try:
                graph_seed_ids = await self.neo4j.find_chunks_by_entities(
                    analysis.entities, limit=self.settings.graph_entity_seed_limit,
                )
            except Exception:
                _log.warning("graph_entity_seed failed, falling back to vector only", exc_info=True)
                graph_seed_ids = []
            seed_chunk_ids = {c.chunk_id for c in seed_chunks}
            new_graph_ids = {cid for cid in graph_seed_ids if cid not in seed_chunk_ids}
            if new_graph_ids:
                graph_seeded_chunks = await self.postgres.get_chunks_by_ids(new_graph_ids)
                for chunk in graph_seeded_chunks:
                    chunk.retrieval_source = "graph_entity"
            timer.end_step("graph_entity_seed", f"({len(graph_seeded_chunks)} chunks)")

        if not seed_chunks and not graph_seeded_chunks:
            return QueryResponse(
                question=question,
                answer="조건에 맞는 근거를 찾지 못했습니다.",
                retrieval_strategy="filter_pgvector_graph_hybrid",
                answer_mode="fallback_sources_only",
                sources=[],
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

        # --- Multi-hop expansion (Phase 3) ---
        multihop_ids: set[str] = set()
        if self.settings.graph_multihop_enabled and analysis.intent in ("relationship", "summary", "timeline"):
            timer.start_step("multihop_expand")
            try:
                if analysis.intent in ("relationship", "summary"):
                    cooccurrence_rows = await self.neo4j.expand_via_entity_cooccurrence(
                        all_seed_ids, limit=self.settings.graph_entity_expansion_limit,
                    )
                    multihop_ids.update(row["neighbor_id"] for row in cooccurrence_rows)
                if analysis.intent in ("relationship", "timeline"):
                    author_rows = await self.neo4j.expand_via_same_author(
                        all_seed_ids, limit=self.settings.graph_author_expansion_limit,
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
        expanded_chunks = await self.postgres.get_chunks_by_ids(expanded_ids)
        timer.end_step("pg_expand_fetch", f"({len(expanded_chunks)} extra)")

        timer.start_step("ranking")
        ranked_chunks = self._rank_chunks(seed_chunks, expanded_chunks, expansions, analysis, graph_seeded_chunks)
        sources = [self._to_source(chunk) for chunk in ranked_chunks[:top_k]]
        timer.end_step("ranking")

        # --- Subgraph extraction (Phase 4) ---
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

        # --- Community context (Phase 5) ---
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
            question, sources, request_user,
            analysis=analysis, channels=channels, users=users,
            subgraph_context=subgraph_context + community_context,
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
                },
            )
            timer.end_step("llm_generate")
            _log.info("[PERF] ════ TOTAL           : %.3fs ════", timer.total_ms() / 1000)

            debug_data = None
            if debug:
                debug_data = await self._build_debug_data(
                    timer, analysis, ranked_chunks, top_k, subgraph_rows,
                    community_rows, seed_chunks, graph_seeded_chunks,
                    list(multihop_ids), list(expanded_ids),
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
            _log.info("[PERF] ════ TOTAL           : %.3fs ════", timer.total_ms() / 1000)

            debug_data = None
            if debug:
                debug_data = await self._build_debug_data(
                    timer, analysis, ranked_chunks, top_k, subgraph_rows,
                    community_rows, seed_chunks, graph_seeded_chunks,
                    list(multihop_ids), list(expanded_ids),
                )

            return QueryResponse(
                question=question,
                answer=self._build_fallback_answer(analysis, sources, top_k),
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
        )

    def _rank_chunks(
        self,
        seed_chunks: list[RetrievedChunk],
        expanded_chunks: list[RetrievedChunk],
        expansions: dict[str, object],
        analysis: QueryAnalysis,
        graph_seeded_chunks: list[RetrievedChunk] | None = None,
    ) -> list[RetrievedChunk]:
        weights = self._get_weights(analysis)
        has_date_filter = analysis.filters.date_from is not None

        ranked: dict[str, RetrievedChunk] = {chunk.chunk_id: chunk for chunk in seed_chunks}
        for chunk in seed_chunks:
            expansion = expansions.get(chunk.chunk_id)
            if expansion:
                chunk.graph_neighbors = expansion.graph_neighbors
                chunk.graph_score = min(1.0, len(expansion.graph_neighbors) / 4)
            chunk.entity_overlap_score = self._entity_overlap_score(chunk, analysis)
            chunk.entity_score = chunk.entity_overlap_score
            chunk.metadata_score = self._metadata_score(chunk, analysis)
            chunk.recency_score = 0.0 if has_date_filter else self._recency_score(chunk.message_date)
            chunk.final_score = self._combined_score(chunk, weights)

        for chunk in (graph_seeded_chunks or []):
            if chunk.chunk_id in ranked:
                continue
            expansion = expansions.get(chunk.chunk_id)
            if expansion:
                chunk.graph_neighbors = expansion.graph_neighbors
            chunk.entity_overlap_score = self._entity_overlap_score(chunk, analysis)
            chunk.entity_score = chunk.entity_overlap_score
            chunk.graph_score = 0.7  # found via graph entity traversal
            chunk.metadata_score = self._metadata_score(chunk, analysis)
            chunk.recency_score = 0.0 if has_date_filter else self._recency_score(chunk.message_date)
            chunk.final_score = self._combined_score(chunk, weights)
            ranked[chunk.chunk_id] = chunk

        for chunk in expanded_chunks:
            if chunk.chunk_id in ranked:
                continue
            chunk.retrieval_source = "graph_expanded"
            chunk.entity_overlap_score = self._entity_overlap_score(chunk, analysis)
            chunk.entity_score = chunk.entity_overlap_score
            chunk.graph_score = 0.6
            chunk.metadata_score = self._metadata_score(chunk, analysis)
            chunk.recency_score = 0.0 if has_date_filter else self._recency_score(chunk.message_date)
            chunk.final_score = self._combined_score(chunk, weights)
            ranked[chunk.chunk_id] = chunk

        return sorted(ranked.values(), key=lambda item: item.final_score, reverse=True)

    @staticmethod
    def _entity_overlap_score(chunk: RetrievedChunk, analysis: QueryAnalysis) -> float:
        if not analysis.entities:
            return 0.0
        chunk_entities = set(chunk.metadata.get("entities", []))
        query_entities = set(analysis.entities)
        if not chunk_entities:
            return 0.0
        overlap = len(chunk_entities & query_entities)
        return min(1.0, overlap / max(len(query_entities), 1))

    def _metadata_score(self, chunk: RetrievedChunk, analysis: QueryAnalysis) -> float:
        score = 0.0
        filters = analysis.filters
        if filters.channel and chunk.channel == filters.channel:
            score += 0.4
        if filters.user_names and chunk.user_name in filters.user_names:
            score += 0.3
        if filters.date_from and filters.date_to and filters.date_from <= chunk.message_date <= filters.date_to:
            score += 0.3
        return min(score, 1.0)

    def _recency_score(self, chunk_date: date) -> float:
        days = max((date.today() - chunk_date).days, 0)
        return 1.0 / (1.0 + (days / 30.0))

    @staticmethod
    def _get_weights(analysis: QueryAnalysis) -> tuple[float, float, float, float, float]:
        """Return (vector, graph, entity, metadata, recency) weights by intent."""
        intent = analysis.intent
        if intent == "timeline":
            return (0.30, 0.15, 0.15, 0.20, 0.20)
        if intent == "relationship":
            return (0.20, 0.30, 0.25, 0.15, 0.10)
        if intent == "aggregate":
            return (0.30, 0.20, 0.20, 0.20, 0.10)
        if intent == "summary":
            return (0.35, 0.20, 0.20, 0.15, 0.10)
        # search (default)
        return (0.40, 0.15, 0.20, 0.15, 0.10)

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
        subgraph_context: str = "",
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
            f"{subgraph_context}"
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
        return "LLM 응답을 생성하지 못해 검색 근거만 반환합니다.\n" + header + "\n" + "\n".join(
            evidence_lines
        )
