from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fastapi import Request

from app.config import Settings

if TYPE_CHECKING:
    from app.adapters.codex_proxy import CodexProxyClient
    from app.adapters.embedding_cache_store import EmbeddingCacheStore
    from app.adapters.embedding_provider import BgeM3EmbeddingProvider
    from app.adapters.neo4j_store import Neo4jStore
    from app.adapters.postgres_vector_store import PostgresVectorStore
    from app.services.behavior_labeler import BehaviorLabeler
    from app.services.chunking import ChunkingService
    from app.services.graph_builder import GraphBuilder
    from app.services.ingest import IngestService
    from app.services.issue_chunking import IssueChunkingService
    from app.services.query_analyzer import QueryAnalyzer
    from app.services.retrieval import RetrievalService
    from app.services.workbook_parser import WorkbookParser


@dataclass
class ServiceContainer:
    settings: Settings
    postgres: Any
    neo4j: Any
    embedding_cache: Any
    embedding_provider: Any
    codex_proxy: Any
    chunking: Any
    behavior_labeler: Any
    issue_chunking: Any
    workbook_parser: Any
    graph_builder: Any
    query_analyzer: Any
    ingest: Any
    retrieval: Any
    startup_errors: dict[str, str] = field(default_factory=dict)
    ready: bool = True

    @classmethod
    def create(cls, settings: Settings) -> "ServiceContainer":
        from app.adapters.codex_proxy import CodexProxyClient
        from app.adapters.embedding_cache_store import EmbeddingCacheStore
        from app.adapters.embedding_provider import BgeM3EmbeddingProvider
        from app.adapters.neo4j_store import Neo4jStore
        from app.adapters.postgres_vector_store import PostgresVectorStore
        from app.services.behavior_labeler import BehaviorLabeler
        from app.services.chunking import ChunkingService
        from app.services.graph_builder import GraphBuilder
        from app.services.ingest import IngestService
        from app.services.issue_chunking import IssueChunkingService
        from app.services.query_analyzer import QueryAnalyzer
        from app.services.retrieval import RetrievalService
        from app.services.workbook_parser import WorkbookParser

        postgres = PostgresVectorStore(settings)
        neo4j = Neo4jStore(settings)
        embedding_cache = EmbeddingCacheStore(settings)
        embedding_provider = BgeM3EmbeddingProvider(settings)
        codex_proxy = CodexProxyClient(settings)
        chunking = ChunkingService(settings)
        behavior_labeler = BehaviorLabeler(settings)
        issue_chunking = IssueChunkingService(settings, behavior_labeler)
        workbook_parser = WorkbookParser(settings)
        graph_builder = GraphBuilder()
        query_analyzer = QueryAnalyzer()
        retrieval = RetrievalService(
            settings=settings,
            postgres=postgres,
            neo4j=neo4j,
            embedding_provider=embedding_provider,
            codex_proxy=codex_proxy,
            query_analyzer=query_analyzer,
        )
        ingest = IngestService(
            settings=settings,
            postgres=postgres,
            neo4j=neo4j,
            embedding_cache=embedding_cache,
            embedding_provider=embedding_provider,
            chunking=chunking,
            issue_chunking=issue_chunking,
            workbook_parser=workbook_parser,
            graph_builder=graph_builder,
            retrieval=retrieval,
        )
        return cls(
            settings=settings,
            postgres=postgres,
            neo4j=neo4j,
            embedding_cache=embedding_cache,
            embedding_provider=embedding_provider,
            codex_proxy=codex_proxy,
            chunking=chunking,
            behavior_labeler=behavior_labeler,
            issue_chunking=issue_chunking,
            workbook_parser=workbook_parser,
            graph_builder=graph_builder,
            query_analyzer=query_analyzer,
            ingest=ingest,
            retrieval=retrieval,
        )

    async def startup(self) -> None:
        self.startup_errors.clear()
        self.ready = True
        critical_failures: list[str] = []
        for name, component in (
            ("postgres", self.postgres),
            ("neo4j", self.neo4j),
            ("embedding_cache", self.embedding_cache),
        ):
            try:
                await component.bootstrap()
            except Exception as exc:  # pragma: no cover - depends on runtime infra
                self.startup_errors[name] = str(exc)
                critical_failures.append(name)
        if critical_failures:
            self.ready = False

    async def shutdown(self) -> None:
        for method_name, component in (
            ("aclose", self.postgres),
            ("aclose", self.embedding_cache),
            ("aclose", self.codex_proxy),
            ("close", self.neo4j),
        ):
            try:
                await getattr(component, method_name)()
            except Exception:  # pragma: no cover - defensive cleanup only
                continue


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container
