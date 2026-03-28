from __future__ import annotations

from app.schemas import QueryAnalysis, RetrievedChunk
from app.services.query_terms import (
    looks_like_flow_query,
    looks_like_generic_issue_summary,
    looks_like_mixed_issue_chat_summary,
    looks_like_related_chat_query,
)


def dedupe_source_candidates(
    chunks: list[RetrievedChunk],
    *,
    top_k: int,
    dedupe_issue_titles: bool = True,
) -> list[RetrievedChunk]:
    deduped: list[RetrievedChunk] = []
    seen_chunk_ids: set[str] = set()
    seen_issue_titles: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen_chunk_ids:
            continue
        issue_title = str(chunk.metadata.get("issue_title") or "").strip()
        if dedupe_issue_titles and chunk.document_type == "issue" and issue_title:
            issue_key = issue_title.casefold()
            if issue_key in seen_issue_titles:
                continue
            seen_issue_titles.add(issue_key)
        seen_chunk_ids.add(chunk.chunk_id)
        deduped.append(chunk)
        if len(deduped) >= top_k:
            break
    return deduped


def aggregate_sample_chunks(
    aggregate_context: dict[str, object] | None,
    *,
    top_k: int,
) -> list[RetrievedChunk]:
    if aggregate_context is None:
        return []
    sample_chunks = [
        chunk
        for chunk in aggregate_context.get("sample_chunks", [])
        if isinstance(chunk, RetrievedChunk)
    ]
    return dedupe_source_candidates(sample_chunks, top_k=top_k)


class StandardSourceSelector:
    def should_prefer_aggregate_samples(
        self,
        analysis: QueryAnalysis,
        aggregate_context: dict[str, object] | None,
    ) -> bool:
        if aggregate_context is None:
            return False
        if not (analysis.detected_document_type == "issue" or "issue" in analysis.filters.all_document_types):
            return False
        if analysis.intent in ("aggregate", "list"):
            return True
        return analysis.intent == "summary" and looks_like_generic_issue_summary(analysis)

    def select(
        self,
        *,
        ranked_chunks: list[RetrievedChunk],
        analysis: QueryAnalysis,
        top_k: int,
        aggregate_context: dict[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        if self.should_prefer_aggregate_samples(analysis, aggregate_context):
            return aggregate_sample_chunks(aggregate_context, top_k=top_k)
        return dedupe_source_candidates(
            ranked_chunks,
            top_k=top_k,
            dedupe_issue_titles=not looks_like_flow_query(analysis),
        )


class CountSourceSelector:
    def select(
        self,
        *,
        sample_chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        return dedupe_source_candidates(sample_chunks, top_k=top_k)


class MixedSourceSelector:
    def select(
        self,
        *,
        issue_chunks: list[RetrievedChunk],
        chat_chunks: list[RetrievedChunk],
        analysis: QueryAnalysis,
        top_k: int,
        aggregate_context: dict[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        issue_pool = (
            aggregate_sample_chunks(aggregate_context, top_k=top_k)
            if aggregate_context is not None and looks_like_mixed_issue_chat_summary(analysis)
            else dedupe_source_candidates(issue_chunks, top_k=top_k)
        )
        chat_pool = dedupe_source_candidates(chat_chunks, top_k=top_k)
        if not chat_pool:
            return issue_pool[:top_k]

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

        return dedupe_source_candidates(blended, top_k=top_k)
