from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fastapi import Request

from app.config import Settings

if TYPE_CHECKING:
    from app.adapters.codex_proxy import CodexProxyClient
    from app.adapters.embedding_provider import BgeM3EmbeddingProvider
    from app.adapters.neo4j_store import Neo4jStore
    from app.adapters.postgres_vector_store import PostgresVectorStore
    from app.services.chunking import ChunkingService
    from app.services.graph_builder import GraphBuilder
    from app.services.ingest import IngestService
    from app.services.query_analyzer import QueryAnalyzer
    from app.services.retrieval import RetrievalService


@dataclass
class ServiceContainer:
    settings: Settings
    postgres: Any
    neo4j: Any
    embedding_provider: Any
    codex_proxy: Any
    chunking: Any
    graph_builder: Any
    query_analyzer: Any
    ingest: Any
    retrieval: Any
    startup_errors: dict[str, str] = field(default_factory=dict)

    @classmethod
    def create(cls, settings: Settings) -> "ServiceContainer":
        from app.adapters.codex_proxy import CodexProxyClient
        from app.adapters.embedding_provider import BgeM3EmbeddingProvider
        from app.adapters.neo4j_store import Neo4jStore
        from app.adapters.postgres_vector_store import PostgresVectorStore
        from app.services.chunking import ChunkingService
        from app.services.graph_builder import GraphBuilder
        from app.services.ingest import IngestService
        from app.services.query_analyzer import QueryAnalyzer
        from app.services.retrieval import RetrievalService

        postgres = PostgresVectorStore(settings)
        neo4j = Neo4jStore(settings)
        embedding_provider = BgeM3EmbeddingProvider(settings)
        codex_proxy = CodexProxyClient(settings)
        chunking = ChunkingService(settings)
        graph_builder = GraphBuilder()
        query_analyzer = QueryAnalyzer()
        ingest = IngestService(
            settings=settings,
            postgres=postgres,
            neo4j=neo4j,
            embedding_provider=embedding_provider,
            chunking=chunking,
            graph_builder=graph_builder,
        )
        retrieval = RetrievalService(
            settings=settings,
            postgres=postgres,
            neo4j=neo4j,
            embedding_provider=embedding_provider,
            codex_proxy=codex_proxy,
            query_analyzer=query_analyzer,
        )
        return cls(
            settings=settings,
            postgres=postgres,
            neo4j=neo4j,
            embedding_provider=embedding_provider,
            codex_proxy=codex_proxy,
            chunking=chunking,
            graph_builder=graph_builder,
            query_analyzer=query_analyzer,
            ingest=ingest,
            retrieval=retrieval,
        )

    async def startup(self) -> None:
        self.startup_errors.clear()
        for name, component in (("postgres", self.postgres), ("neo4j", self.neo4j)):
            try:
                await component.bootstrap()
            except Exception as exc:  # pragma: no cover - depends on runtime infra
                self.startup_errors[name] = str(exc)

    async def shutdown(self) -> None:
        for method_name, component in (("aclose", self.codex_proxy), ("close", self.neo4j)):
            try:
                await getattr(component, method_name)()
            except Exception:  # pragma: no cover - defensive cleanup only
                continue


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container
