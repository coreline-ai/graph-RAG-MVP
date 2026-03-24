import argparse
import json
from pathlib import Path

from app.repositories.ingest_repo import IngestRepository
from app.repositories.neo4j_client import Neo4jClient
from app.repositories.schema import ensure_schema
from app.services.embedder import BgeM3Embedder
from app.services.ingestion import IngestionService
from app.settings import get_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Korean chat logs into Neo4j.")
    parser.add_argument("--input", action="append", help="Path to input log file", default=[])
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use the bundled sample dataset at data/chat_logs_100.txt",
    )
    parser.add_argument(
        "--rebuild-prev-links",
        action="store_true",
        help="Delete and rebuild PREV_IN_ROOM relationships",
    )
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace, settings) -> list[Path]:
    if args.input:
        return [Path(item) for item in args.input]
    if args.sample:
        return [settings.data_dir / "chat_logs_100.txt"]
    return [settings.data_dir / "chat_logs_100.txt"]


def main() -> None:
    args = parse_args()
    settings = get_settings()
    input_paths = resolve_inputs(args, settings)

    client = Neo4jClient(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    try:
        ensure_schema(client)
        service = IngestionService(
            settings=settings,
            embedder=BgeM3Embedder(settings),
            ingest_repo=IngestRepository(client),
        )
        report = service.ingest_files(
            input_paths=input_paths,
            rebuild_prev_links=args.rebuild_prev_links,
        )
        print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
