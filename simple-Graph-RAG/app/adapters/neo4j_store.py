from __future__ import annotations

from itertools import islice
from typing import Any

from neo4j import AsyncGraphDatabase

from app.config import Settings
from app.schemas import ChunkRecord, DocumentMetadata, GraphExpansion


class Neo4jStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )

    async def _run_query(self, query: str, **params: Any) -> list[dict[str, Any]]:
        async with self._driver.session(database=self.settings.neo4j_database) as session:
            result = await session.run(query, **params)
            return [record.data() async for record in result]

    @staticmethod
    def _batched(rows: list[dict[str, Any]], size: int = 10) -> list[list[dict[str, Any]]]:
        iterator = iter(rows)
        batches: list[list[dict[str, Any]]] = []
        while batch := list(islice(iterator, size)):
            batches.append(batch)
        return batches

    async def bootstrap(self) -> None:
        statements = [
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT user_name_unique IF NOT EXISTS FOR (u:User) REQUIRE u.name IS UNIQUE",
            "CREATE CONSTRAINT channel_name_unique IF NOT EXISTS FOR (c:Channel) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT date_value_unique IF NOT EXISTS FOR (d:Date) REQUIRE d.date IS UNIQUE",
        ]
        for statement in statements:
            await self._run_query(statement)

    async def healthcheck(self) -> str:
        try:
            await self._run_query("RETURN 1 AS ok")
        except Exception as exc:  # pragma: no cover - depends on runtime infra
            return f"error:{exc}"
        return "ok"

    async def upsert_graph(self, document: DocumentMetadata, graph_rows: list[dict[str, Any]]) -> None:
        await self._run_query(
            """
            MERGE (d:Document {document_id: $document_id})
            SET d.filename = $filename,
                d.source = $source,
                d.access_scopes = $access_scopes,
                d.total_messages = $total_messages,
                d.total_chunks = $total_chunks,
                d.created_at = $created_at
            """,
            document_id=document.document_id,
            filename=document.filename,
            source=document.source,
            access_scopes=document.access_scopes,
            total_messages=document.total_messages,
            total_chunks=document.total_chunks,
            created_at=document.created_at.isoformat(),
        )

        for batch in self._batched(graph_rows, size=10):
            await self._run_query(
                """
                UNWIND $rows AS row
                MERGE (c:Chunk {chunk_id: row.chunk_id})
                SET c.document_id = row.document_id,
                    c.text = row.text,
                    c.channel = row.channel,
                    c.user_name = row.user_name,
                    c.date = row.date,
                    c.time = row.time,
                    c.seq = row.seq,
                    c.token_count = row.token_count,
                    c.access_scopes = row.access_scopes
                WITH c, row
                MATCH (d:Document {document_id: row.document_id})
                MERGE (c)-[:PART_OF]->(d)
                MERGE (u:User {name: row.user_name})
                MERGE (c)-[:SENT_BY]->(u)
                MERGE (ch:Channel {name: row.channel})
                MERGE (c)-[:IN_CHANNEL]->(ch)
                MERGE (dt:Date {date: row.date})
                SET dt.date_int = row.date_int
                MERGE (c)-[:ON_DATE]->(dt)
                FOREACH (entity_name IN row.entities |
                    MERGE (e:Entity {name: entity_name})
                    MERGE (c)-[:MENTIONS]->(e)
                )
                """,
                rows=batch,
            )

        next_edges = [
            {"from_chunk_id": current["chunk_id"], "to_chunk_id": next_row["chunk_id"]}
            for current, next_row in zip(graph_rows, graph_rows[1:], strict=False)
        ]
        if next_edges:
            for batch in self._batched(next_edges, size=20):
                await self._run_query(
                    """
                    UNWIND $edges AS edge
                    MATCH (a:Chunk {chunk_id: edge.from_chunk_id})
                    MATCH (b:Chunk {chunk_id: edge.to_chunk_id})
                    MERGE (a)-[:NEXT]->(b)
                    """,
                    edges=batch,
                )

    async def delete_document(self, document_id: str) -> bool:
        document_rows = await self._run_query(
            """
            MATCH (d:Document {document_id: $document_id})
            OPTIONAL MATCH (d)<-[:PART_OF]-(c:Chunk)
            OPTIONAL MATCH (c)--(n)
            WHERE n:User OR n:Channel OR n:Date OR n:Entity OR n:Topic
            RETURN count(DISTINCT d) AS doc_count, collect(DISTINCT id(n)) AS neighbor_ids
            """,
            document_id=document_id,
        )
        if not document_rows or document_rows[0]["doc_count"] == 0:
            return False

        neighbor_ids = [value for value in document_rows[0]["neighbor_ids"] if value is not None]
        await self._run_query(
            """
            MATCH (d:Document {document_id: $document_id})<-[:PART_OF]-(c:Chunk)
            DETACH DELETE c
            """,
            document_id=document_id,
        )
        await self._run_query(
            """
            MATCH (d:Document {document_id: $document_id})
            DETACH DELETE d
            """,
            document_id=document_id,
        )
        if neighbor_ids:
            await self._run_query(
                """
                UNWIND $neighbor_ids AS neighbor_id
                MATCH (n)
                WHERE id(n) = neighbor_id
                  AND (n:User OR n:Channel OR n:Date OR n:Entity OR n:Topic)
                  AND NOT (n)--()
                DELETE n
                """,
                neighbor_ids=neighbor_ids,
            )
        return True

    async def expand_from_seed_chunks(
        self,
        chunk_ids: list[str],
        *,
        next_window: int,
    ) -> dict[str, GraphExpansion]:
        if not chunk_ids:
            return {}
        safe_window = max(1, int(next_window))
        rows = await self._run_query(
            f"""
            MATCH (seed:Chunk)
            WHERE seed.chunk_id IN $chunk_ids
            OPTIONAL MATCH (seed)-[:SENT_BY]->(u:User)
            OPTIONAL MATCH (seed)-[:IN_CHANNEL]->(ch:Channel)
            OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)
            OPTIONAL MATCH (seed)-[:ON_DATE]->(dt:Date)
            OPTIONAL MATCH (seed)-[:NEXT*1..{safe_window}]->(near:Chunk)
            RETURN
                seed.chunk_id AS chunk_id,
                collect(DISTINCT u.name) AS users,
                collect(DISTINCT ch.name) AS channels,
                collect(DISTINCT e.name) AS entities,
                collect(DISTINCT dt.date) AS dates,
                collect(DISTINCT near.chunk_id) AS expanded_chunk_ids
            """,
            chunk_ids=chunk_ids,
        )
        expansions: dict[str, GraphExpansion] = {}
        for row in rows:
            neighbors = [
                value
                for group in (row["users"], row["channels"], row["entities"], row["dates"])
                for value in group
                if value
            ]
            expanded_ids = [value for value in row["expanded_chunk_ids"] if value]
            expansions[row["chunk_id"]] = GraphExpansion(
                chunk_id=row["chunk_id"],
                graph_neighbors=neighbors,
                expanded_chunk_ids=expanded_ids,
            )
        return expansions

    async def close(self) -> None:
        await self._driver.close()
