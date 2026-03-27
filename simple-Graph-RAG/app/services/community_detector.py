"""Community detection service using Neo4j GDS Louvain algorithm.

This is an offline/batch service — not called per-query. Run via
`scripts/detect_communities.py` after data ingestion.
"""
from __future__ import annotations

import logging
from typing import Any

from app.adapters.codex_proxy import CodexProxyClient
from app.adapters.neo4j_store import Neo4jStore
from app.config import Settings

_log = logging.getLogger(__name__)

_GRAPH_NAME = "entity-cooccurrence"


class CommunityDetector:
    def __init__(
        self,
        *,
        settings: Settings,
        neo4j: Neo4jStore,
        codex_proxy: CodexProxyClient | None = None,
    ) -> None:
        self.settings = settings
        self.neo4j = neo4j
        self.codex_proxy = codex_proxy

    async def project_entity_graph(self) -> bool:
        """Create a GDS graph projection of entity co-occurrence via shared chunks."""
        # Drop existing projection if any
        try:
            await self.neo4j._run_query(
                f"CALL gds.graph.drop('{_GRAPH_NAME}', false)",
            )
        except Exception:
            pass

        try:
            await self.neo4j._run_query(
                f"""
                CALL gds.graph.project.cypher(
                    '{_GRAPH_NAME}',
                    'MATCH (e:Entity) RETURN id(e) AS id',
                    'MATCH (e1:Entity)<-[:MENTIONS]-(c:Chunk)-[:MENTIONS]->(e2:Entity)
                     WHERE id(e1) < id(e2)
                     RETURN id(e1) AS source, id(e2) AS target, count(c) AS weight'
                )
                """,
            )
            return True
        except Exception as exc:
            _log.error("Failed to project entity graph: %s", exc)
            return False

    async def run_louvain(self) -> dict[int, list[str]]:
        """Run Louvain community detection. Returns {community_id: [entity_names]}."""
        rows = await self.neo4j._run_query(
            f"""
            CALL gds.louvain.stream('{_GRAPH_NAME}', {{
                relationshipWeightProperty: 'weight'
            }})
            YIELD nodeId, communityId
            WITH communityId, gds.util.asNode(nodeId).name AS entityName
            RETURN communityId, collect(entityName) AS entities
            ORDER BY size(collect(entityName)) DESC
            """,
        )
        communities: dict[int, list[str]] = {}
        min_size = self.settings.community_min_size
        for row in rows:
            entities = row["entities"]
            if len(entities) >= min_size:
                communities[row["communityId"]] = entities
        return communities

    async def generate_community_summaries(
        self,
        communities: dict[int, list[str]],
    ) -> dict[int, dict[str, Any]]:
        """Generate LLM summaries for each community. Returns enriched community data."""
        result: dict[int, dict[str, Any]] = {}
        for cid, entities in communities.items():
            summary = f"엔티티 그룹: {', '.join(entities[:10])}"
            if self.codex_proxy:
                try:
                    response = await self.codex_proxy.generate(
                        system_prompt="다음 엔티티들의 공통 주제를 한 문장으로 요약하라.",
                        user_prompt=f"엔티티 목록: {', '.join(entities)}",
                    )
                    summary = response.text
                except Exception:
                    pass
            result[cid] = {"entities": entities, "summary": summary}
        return result

    async def detect_and_store(self) -> int:
        """Full pipeline: project → detect → summarize → store. Returns community count."""
        projected = await self.project_entity_graph()
        if not projected:
            _log.warning("Graph projection failed — skipping community detection")
            return 0

        communities = await self.run_louvain()
        _log.info("Detected %d communities (min_size=%d)", len(communities), self.settings.community_min_size)

        enriched = await self.generate_community_summaries(communities)
        await self.neo4j.upsert_communities(enriched)

        # Cleanup projection
        try:
            await self.neo4j._run_query(
                f"CALL gds.graph.drop('{_GRAPH_NAME}', false)",
            )
        except Exception:
            pass

        return len(enriched)
