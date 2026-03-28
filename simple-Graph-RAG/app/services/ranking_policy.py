from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from app.schemas import GraphExpansion, QueryAnalysis, RetrievedChunk
from app.services.query_terms import (
    chunk_matches_alias_group,
    chunk_search_text,
    exact_special_groups,
    looks_like_flow_query,
    looks_like_related_chat_query,
    query_match_terms,
    query_phrase_candidates,
    strict_lexical_groups,
)


LaneType = Literal["standard", "issue", "chat"]


@dataclass
class RankingContext:
    expansions: dict[str, GraphExpansion] = field(default_factory=dict)
    latest_event_date: date | None = None
    lane: LaneType = "standard"


class StandardRankingPolicy:
    def entity_overlap_score(self, chunk: RetrievedChunk, analysis: QueryAnalysis) -> float:
        query_terms = query_match_terms(analysis)
        alias_groups = strict_lexical_groups(analysis)
        if not query_terms and not alias_groups:
            return 0.0
        chunk_entities = {str(entity).lower() for entity in chunk.metadata.get("entities", [])}
        entity_overlap = len(chunk_entities & set(query_terms)) if chunk_entities else 0
        chunk_text = chunk_search_text(chunk)
        lexical_overlap = sum(1 for term in query_terms if term in chunk_text)
        alias_hits = sum(1 for group in alias_groups if chunk_matches_alias_group(chunk, group))
        overlap = max(entity_overlap, lexical_overlap + alias_hits)
        total_terms = max(len(query_terms) + len(alias_groups), 1)
        return min(1.0, overlap / total_terms)

    def metadata_score(self, chunk: RetrievedChunk, analysis: QueryAnalysis) -> float:
        score = 0.0
        filters = analysis.filters
        chunk_kind = str(chunk.metadata.get("chunk_kind", ""))
        exact_match_score = self.entity_overlap_score(chunk, analysis)
        chat_relevance_bonus = self.chat_relevance_bonus(chunk, analysis)
        if filters.all_channels and chunk.channel in filters.all_channels:
            score += 0.4
        if filters.user_names and chunk.user_name in filters.user_names:
            score += 0.3
        if filters.assignees and (chunk.metadata.get("assignee") in filters.assignees or chunk.user_name in filters.assignees):
            score += 0.2
        if filters.statuses and chunk.metadata.get("status") in filters.statuses:
            score += 0.15
        if filters.all_document_types and chunk.document_type in filters.all_document_types:
            score += 0.15
        if filters.date_from and filters.date_to and filters.date_from <= chunk.message_date <= filters.date_to:
            score += 0.3
        if exact_match_score >= 0.5:
            score += 0.2
        if looks_like_related_chat_query(analysis) and chunk.document_type == "chat":
            score += 0.2
        if chat_relevance_bonus > 0.0:
            score += chat_relevance_bonus
        if chunk.document_type == "issue" and analysis.intent in ("summary", "aggregate", "list") and chunk_kind == "overview":
            score += 0.1
        if chunk.document_type == "issue" and looks_like_flow_query(analysis) and chunk_kind == "analysis_flow":
            score += 0.1
        return max(0.0, min(score, 1.0))

    def recency_score(self, chunk_date: date, latest_event_date: date | None = None) -> float:
        reference_day = latest_event_date or date.today()
        days = max((reference_day - chunk_date).days, 0)
        return 1.0 / (1.0 + (days / 30.0))

    def get_weights(self, analysis: QueryAnalysis) -> tuple[float, float, float, float, float]:
        intent = analysis.intent
        if intent == "timeline":
            return (0.30, 0.15, 0.15, 0.20, 0.20)
        if intent == "relationship":
            return (0.20, 0.30, 0.25, 0.15, 0.10)
        if intent in ("aggregate", "list"):
            return (0.30, 0.20, 0.20, 0.20, 0.10)
        if intent == "summary":
            return (0.35, 0.20, 0.20, 0.15, 0.10)
        return (0.40, 0.15, 0.20, 0.15, 0.10)

    def combined_score(self, chunk: RetrievedChunk, weights: tuple[float, float, float, float, float]) -> float:
        w_vector, w_graph, w_entity, w_metadata, w_recency = weights
        return (
            w_vector * chunk.vector_score
            + w_graph * chunk.graph_score
            + w_entity * chunk.entity_score
            + w_metadata * chunk.metadata_score
            + w_recency * chunk.recency_score
        )

    def chat_relevance_bonus(self, chunk: RetrievedChunk, analysis: QueryAnalysis) -> float:
        if chunk.document_type != "chat":
            return 0.0
        if analysis.detected_document_type != "chat" and not looks_like_related_chat_query(analysis):
            return 0.0

        chunk_text = chunk_search_text(chunk)
        if any(phrase in chunk_text for phrase in query_phrase_candidates(analysis)):
            return 0.30

        lexical_coverage = self.lexical_coverage_score(chunk, analysis)
        query_terms = query_match_terms(analysis)
        if len(query_terms) >= 2:
            if lexical_coverage >= 1.0:
                return 0.22
            if lexical_coverage >= 0.5:
                return 0.08
            return 0.0
        return 0.12 * lexical_coverage

    def lexical_coverage_score(self, chunk: RetrievedChunk, analysis: QueryAnalysis) -> float:
        query_terms = query_match_terms(analysis)
        if not query_terms:
            return 0.0
        chunk_text = chunk_search_text(chunk)
        matched = sum(1 for term in query_terms if term in chunk_text)
        if matched <= 0:
            return 0.0
        return matched / max(len(query_terms), 1)

    def apply_special_keyword_grounding(
        self,
        ranked_chunks: list[RetrievedChunk],
        analysis: QueryAnalysis,
    ) -> list[RetrievedChunk]:
        exact_groups = exact_special_groups(analysis)
        alias_groups = strict_lexical_groups(analysis)
        if not exact_groups and not alias_groups:
            return ranked_chunks
        if exact_groups:
            exact_matches = [
                chunk
                for chunk in ranked_chunks
                if any(chunk_matches_alias_group(chunk, group) for group in exact_groups)
            ]
            if exact_matches:
                return exact_matches
        if not alias_groups:
            return []
        return [
            chunk
            for chunk in ranked_chunks
            if any(chunk_matches_alias_group(chunk, group) for group in alias_groups)
        ]

    def rank(
        self,
        *,
        seed_chunks: list[RetrievedChunk],
        expanded_chunks: list[RetrievedChunk],
        analysis: QueryAnalysis,
        context: RankingContext,
        graph_seeded_chunks: list[RetrievedChunk] | None = None,
    ) -> list[RetrievedChunk]:
        weights = self.get_weights(analysis)
        has_date_filter = analysis.filters.date_from is not None
        ranked: dict[str, RetrievedChunk] = {chunk.chunk_id: chunk for chunk in seed_chunks}

        for chunk in seed_chunks:
            expansion = context.expansions.get(chunk.chunk_id)
            if expansion:
                chunk.graph_neighbors = expansion.graph_neighbors
                chunk.graph_score = min(1.0, len(expansion.graph_neighbors) / 4)
            chunk.entity_overlap_score = self.entity_overlap_score(chunk, analysis)
            chunk.entity_score = chunk.entity_overlap_score
            chunk.metadata_score = self.metadata_score(chunk, analysis)
            chunk.recency_score = 0.0 if has_date_filter else self.recency_score(
                chunk.message_date,
                context.latest_event_date,
            )
            chunk.final_score = self.combined_score(chunk, weights)

        for chunk in (graph_seeded_chunks or []):
            if chunk.chunk_id in ranked:
                continue
            expansion = context.expansions.get(chunk.chunk_id)
            if expansion:
                chunk.graph_neighbors = expansion.graph_neighbors
            chunk.entity_overlap_score = self.entity_overlap_score(chunk, analysis)
            chunk.entity_score = chunk.entity_overlap_score
            chunk.graph_score = max(0.7, min(1.0, 0.7 + (0.3 * chunk.entity_overlap_score)))
            chunk.metadata_score = self.metadata_score(chunk, analysis)
            chunk.recency_score = 0.0 if has_date_filter else self.recency_score(
                chunk.message_date,
                context.latest_event_date,
            )
            chunk.final_score = self.combined_score(chunk, weights)
            ranked[chunk.chunk_id] = chunk

        for chunk in expanded_chunks:
            if chunk.chunk_id in ranked:
                continue
            chunk.retrieval_source = "graph_expanded"
            chunk.entity_overlap_score = self.entity_overlap_score(chunk, analysis)
            chunk.entity_score = chunk.entity_overlap_score
            chunk.graph_score = 0.6
            chunk.metadata_score = self.metadata_score(chunk, analysis)
            chunk.recency_score = 0.0 if has_date_filter else self.recency_score(
                chunk.message_date,
                context.latest_event_date,
            )
            chunk.final_score = self.combined_score(chunk, weights)
            ranked[chunk.chunk_id] = chunk

        return sorted(ranked.values(), key=lambda item: item.final_score, reverse=True)


class MixedRankingPolicy(StandardRankingPolicy):
    def __init__(self, *, lane: LaneType) -> None:
        self.lane = lane

    def metadata_score(self, chunk: RetrievedChunk, analysis: QueryAnalysis) -> float:
        score = 0.0
        filters = analysis.filters
        chunk_kind = str(chunk.metadata.get("chunk_kind", ""))
        exact_match_score = self.entity_overlap_score(chunk, analysis)
        if self.lane == "issue":
            if filters.assignees and (chunk.metadata.get("assignee") in filters.assignees or chunk.user_name in filters.assignees):
                score += 0.2
            if filters.statuses and chunk.metadata.get("status") in filters.statuses:
                score += 0.15
            if filters.date_from and filters.date_to and filters.date_from <= chunk.message_date <= filters.date_to:
                score += 0.3
            if filters.all_document_types and chunk.document_type == "issue":
                score += 0.15
            if chunk_kind == "overview":
                score += 0.18
            if looks_like_flow_query(analysis) and chunk_kind == "analysis_flow":
                score += 0.08
        elif self.lane == "chat":
            if filters.all_channels and chunk.channel in filters.all_channels:
                score += 0.3
            if filters.user_names and chunk.user_name in filters.user_names:
                score += 0.2
            if filters.date_from and filters.date_to and filters.date_from <= chunk.message_date <= filters.date_to:
                score += 0.25
            if filters.all_document_types and chunk.document_type == "chat":
                score += 0.15
            score += self.chat_relevance_bonus(chunk, analysis)
        if exact_match_score >= 0.5:
            score += 0.2
        return max(0.0, min(score, 1.0))

    def get_weights(self, analysis: QueryAnalysis) -> tuple[float, float, float, float, float]:
        if self.lane == "chat":
            return (0.45, 0.0, 0.20, 0.25, 0.10)
        return (0.35, 0.10, 0.20, 0.25, 0.10)
