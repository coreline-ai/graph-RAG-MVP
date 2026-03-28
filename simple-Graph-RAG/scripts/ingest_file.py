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
    parser = argparse.ArgumentParser(description="Ingest a chat log or issue workbook into PostgreSQL and Neo4j.")
    parser.add_argument("file", help="Path to a .txt chat log or .xlsx issue workbook.")
    parser.add_argument("--source", default="cli-ingest", help="Document source label.")
    parser.add_argument("--document-type", default="auto", choices=("auto", "chat", "issue"))
    parser.add_argument("--replace-filename", action="store_true")
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
    file_bytes = file_path.read_bytes()

    settings = get_settings()
    container = ServiceContainer.create(settings)
    await container.startup()
    try:
        access_scopes = parse_access_scopes(args.access_scopes)
        resolved_type = args.document_type
        if resolved_type == "auto":
            resolved_type = "issue" if file_path.suffix.lower() == ".xlsx" else "chat"

        if args.skip_graph:
            document = await _ingest_postgres_only(
                container,
                file_path=file_path,
                file_bytes=file_bytes,
                access_scopes=access_scopes,
                source=args.source,
                document_type=resolved_type,
                replace_filename=args.replace_filename,
            )
        else:
            content = None
            if resolved_type == "chat":
                content = file_bytes.decode("utf-8")
            document = await container.ingest.ingest_document(
                filename=file_path.name,
                content=content,
                file_bytes=file_bytes,
                default_access_scopes=access_scopes,
                source=args.source,
                document_type=resolved_type,
                replace_filename=args.replace_filename,
            )
    finally:
        await container.shutdown()

    print(json.dumps(document.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


async def _ingest_postgres_only(
    container: ServiceContainer,
    *,
    file_path: Path,
    file_bytes: bytes,
    access_scopes: list[str],
    source: str,
    document_type: str,
    replace_filename: bool,
) -> DocumentMetadata:
    if replace_filename:
        await container.ingest._replace_existing_documents(file_path.name, document_type)

    document_id = str(uuid4())
    if document_type == "issue":
        parsed = container.workbook_parser.parse_issue_workbook(file_bytes)
        chunks, chunk_summary = container.issue_chunking.build_chunks(
            parsed.rows,
            document_id=document_id,
            default_access_scopes=access_scopes,
        )
        ingest_summary = {
            "total_rows": parsed.total_rows,
            "ingested_rows": len(parsed.rows),
            "skipped_rows": parsed.skipped_rows,
            "overview_chunks": chunk_summary["overview_chunks"],
            "analysis_chunks": chunk_summary["analysis_chunks"],
            "warnings_count": len(parsed.warnings),
            "warnings": parsed.warnings,
        }
        total_messages = parsed.total_rows
    else:
        content = file_bytes.decode("utf-8")
        messages = container.chunking.parse_log_content(content)
        chunks = container.chunking.build_chunks(
            messages,
            document_id=document_id,
            default_access_scopes=access_scopes,
        )
        ingest_summary = {}
        total_messages = len(messages)

    embeddings = await container.ingest._embed_texts([chunk.chunk_text for chunk in chunks])
    document = DocumentMetadata(
        document_id=document_id,
        filename=file_path.name,
        source=source,
        document_type=document_type,
        access_scopes=access_scopes,
        total_messages=total_messages,
        total_chunks=len(chunks),
        ingest_summary=ingest_summary,
        created_at=datetime.now(timezone.utc),
    )
    await container.postgres.upsert_document(document)
    await container.postgres.upsert_chunks(chunks, embeddings)
    if container.retrieval is not None:
        container.retrieval.invalidate_metadata_cache()
    return document


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
