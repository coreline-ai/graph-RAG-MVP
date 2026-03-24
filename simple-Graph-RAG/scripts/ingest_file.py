from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings, parse_access_scopes
from app.container import ServiceContainer
from app.schemas import DocumentMetadata


async def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest a chat log file into PostgreSQL and Neo4j.")
    parser.add_argument("file", help="Path to a UTF-8 chat log text file.")
    parser.add_argument("--source", default="cli-ingest", help="Document source label.")
    parser.add_argument(
        "--access-scopes",
        default="public",
        help="Comma-separated access scopes. Example: public,team-a",
    )
    parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="Store chunks in PostgreSQL only and skip Neo4j graph writes.",
    )
    args = parser.parse_args()

    file_path = Path(args.file).resolve()
    content = file_path.read_text(encoding="utf-8")

    settings = get_settings()
    container = ServiceContainer.create(settings)
    await container.startup()
    try:
        access_scopes = parse_access_scopes(args.access_scopes)
        if args.skip_graph:
            document_id = str(uuid4())
            messages = container.chunking.parse_log_content(content)
            chunks = container.chunking.build_chunks(
                messages,
                document_id=document_id,
                default_access_scopes=access_scopes,
            )
            embeddings = await container.embedding_provider.embed_texts(
                [chunk.chunk_text for chunk in chunks]
            )
            document = DocumentMetadata(
                document_id=document_id,
                filename=file_path.name,
                source=args.source,
                access_scopes=access_scopes,
                total_messages=len(messages),
                total_chunks=len(chunks),
                created_at=datetime.now(timezone.utc),
            )
            await container.postgres.upsert_document(document)
            await container.postgres.upsert_chunks(chunks, embeddings)
        else:
            document = await container.ingest.ingest_document(
                filename=file_path.name,
                content=content,
                default_access_scopes=access_scopes,
                source=args.source,
            )
    finally:
        await container.shutdown()

    print(json.dumps(document.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
