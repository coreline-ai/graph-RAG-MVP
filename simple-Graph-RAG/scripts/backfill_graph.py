from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.adapters.neo4j_store import Neo4jStore
from app.config import get_settings
from app.schemas import ChunkRecord, DocumentMetadata
from app.services.graph_builder import GraphBuilder


def _parse_seq(chunk_id: str) -> int:
    try:
        return int(chunk_id.rsplit("_chunk_", maxsplit=1)[1])
    except (IndexError, ValueError):
        return 0


async def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill Neo4j graph from PostgreSQL chunks.")
    parser.add_argument("document_id", help="Existing document_id stored in PostgreSQL.")
    args = parser.parse_args()

    settings = get_settings()
    graph_builder = GraphBuilder()
    neo4j = Neo4jStore(settings)

    with psycopg.connect(settings.postgres_dsn, row_factory=dict_row) as connection:
        document_row = connection.execute(
            """
            SELECT document_id, filename, source, access_scopes, total_messages, total_chunks, created_at
            FROM documents
            WHERE document_id = %s
            """,
            (args.document_id,),
        ).fetchone()
        if not document_row:
            raise SystemExit(f"Document not found: {args.document_id}")

        chunk_rows = connection.execute(
            f"""
            SELECT chunk_id, document_id, channel, user_name, message_date, message_time,
                   access_scopes, chunk_text, metadata
            FROM {settings.pgvector_table}
            WHERE document_id = %s
            ORDER BY chunk_id
            """,
            (args.document_id,),
        ).fetchall()

    document = DocumentMetadata(**document_row)
    chunks: list[ChunkRecord] = []
    for row in chunk_rows:
        metadata = row.get("metadata") or {}
        chunk_text = row["chunk_text"]
        chunks.append(
            ChunkRecord(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                channel=row["channel"],
                user_name=row["user_name"],
                message_date=row["message_date"],
                message_time=row["message_time"],
                access_scopes=row.get("access_scopes") or [],
                chunk_text=chunk_text,
                token_count=max(len(chunk_text.split()), max(1, len(chunk_text) // 4)),
                seq=_parse_seq(row["chunk_id"]),
                metadata=metadata,
                original_lines=metadata.get("original_lines") or [],
            )
        )

    graph_rows = graph_builder.build_graph_rows(chunks)
    await neo4j.bootstrap()
    await neo4j.upsert_graph(document, graph_rows)
    await neo4j.close()

    print(
        json.dumps(
            {
                "document_id": document.document_id,
                "chunks": len(chunks),
                "graph_rows": len(graph_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
