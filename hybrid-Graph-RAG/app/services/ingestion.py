import json
from pathlib import Path

from app.models.api import IngestionReport
from app.models.errors import ParseFailure, ParseLineError
from app.models.records import ChatMessageRecord
from app.repositories.ingest_repo import IngestRepository
from app.services.embedder import BgeM3Embedder
from app.services.normalizer import normalize_parsed_line
from app.services.parser import parse_line
from app.settings import Settings


class IngestionService:
    def __init__(self, settings: Settings, embedder: BgeM3Embedder, ingest_repo: IngestRepository):
        self.settings = settings
        self.embedder = embedder
        self.ingest_repo = ingest_repo

    def ingest_files(self, input_paths: list[Path], rebuild_prev_links: bool = False) -> IngestionReport:
        parse_failures: list[ParseFailure] = []
        records: list[ChatMessageRecord] = []
        total = 0

        for input_path in input_paths:
            for line_no, raw_text in enumerate(input_path.read_text(encoding="utf-8").splitlines(), start=1):
                if not raw_text.strip():
                    continue
                total += 1
                try:
                    parsed = parse_line(raw_text=raw_text, source_file=str(input_path), line_no=line_no)
                except ParseLineError as exc:
                    parse_failures.append(exc.failure)
                    continue
                records.append(normalize_parsed_line(parsed))

        embedding_failed = self._attach_embeddings(records)
        self._write_failures(parse_failures)

        self.ingest_repo.upsert_messages(records)
        if rebuild_prev_links:
            self.ingest_repo.clear_prev_in_room_relationships()
        prev_links_created = self.ingest_repo.create_prev_in_room_relationships()
        counts = self.ingest_repo.fetch_counts()

        report = IngestionReport(
            total=total,
            success=len(records),
            parse_failed=len(parse_failures),
            embedding_failed=embedding_failed,
            users_created=counts["users"],
            rooms_created=counts["rooms"],
            dates_created=counts["dates"],
            prev_links_created=prev_links_created or counts["prev_links"],
            failures=[
                {
                    "source_file": failure.source_file,
                    "line_no": failure.line_no,
                    "raw_text": failure.raw_text,
                    "error_code": failure.error_code,
                    "message": failure.message,
                }
                for failure in parse_failures
            ],
        )

        if total > 0 and report.parse_failed / total >= 0.01:
            raise RuntimeError("parse failure rate exceeded 1%")
        return report

    def _attach_embeddings(self, records: list[ChatMessageRecord]) -> int:
        if not records:
            return 0

        failed = 0
        batch_size = self.settings.embedding_batch_size
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            texts = [record.content for record in batch]
            try:
                embeddings = self.embedder.embed(texts)
            except RuntimeError:
                failed += len(batch)
                for record in batch:
                    record.embedding_status = "failed"
                    record.embedding = None
                continue

            for record, embedding in zip(batch, embeddings, strict=True):
                record.embedding_status = "completed"
                record.embedding = embedding
        return failed

    def _write_failures(self, failures: list[ParseFailure]) -> None:
        path = self.settings.log_dir / "ingestion_errors.jsonl"
        if not failures:
            if not path.exists():
                path.touch()
            return
        with path.open("w", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(
                    json.dumps(
                        {
                            "source_file": failure.source_file,
                            "line_no": failure.line_no,
                            "raw_text": failure.raw_text,
                            "error_code": failure.error_code,
                            "message": failure.message,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
