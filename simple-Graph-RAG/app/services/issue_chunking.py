from __future__ import annotations

from datetime import date, time

from app.config import Settings
from app.schemas import ChunkRecord, IssueRow
from app.services.behavior_labeler import BehaviorLabeler
from app.services.chunking import ChunkingService


class IssueChunkingService:
    def __init__(self, settings: Settings, behavior_labeler: BehaviorLabeler) -> None:
        self.settings = settings
        self.behavior_labeler = behavior_labeler

    def build_chunks(
        self,
        rows: list[IssueRow],
        *,
        document_id: str,
        default_access_scopes: list[str],
    ) -> tuple[list[ChunkRecord], dict[str, int]]:
        chunks: list[ChunkRecord] = []
        analysis_count = 0
        overview_count = 0

        for row in rows:
            split_analysis = self._should_split_analysis(row)
            overview_text = self._build_overview_text(row) if split_analysis else self._build_single_row_text(row)
            overview_count += 1
            chunks.append(
                self._build_chunk(
                    chunk_id=f"{document_id}_issue_{row.row_index:05d}_overview",
                    document_id=document_id,
                    row=row,
                    access_scopes=default_access_scopes,
                    chunk_text=overview_text,
                    seq=len(chunks),
                    metadata=self._build_issue_metadata(
                        document_id=document_id,
                        row=row,
                        extra={
                            "chunk_kind": "overview",
                            "row_index": row.row_index,
                            "sheet_name": row.sheet_name,
                            "split_mode": "split" if split_analysis else "single",
                        },
                    ),
                )
            )

            if not split_analysis or not row.analysis:
                continue

            flow_chunks = self.behavior_labeler.split_and_label(row.analysis)
            for flow_index, flow_chunk in enumerate(flow_chunks, start=1):
                analysis_count += 1
                text = "\n".join(
                    [
                        f"[이슈] {row.title}",
                        f"[등록일] {row.registered_date.isoformat()}",
                        f"[담당자] {row.assignee}",
                        f"[진행] {row.status}",
                        f"[{flow_chunk.flow_name}] {flow_chunk.text}",
                    ]
                )
                chunks.append(
                    self._build_chunk(
                        chunk_id=f"{document_id}_issue_{row.row_index:05d}_flow_{flow_index:02d}",
                        document_id=document_id,
                        row=row,
                        access_scopes=default_access_scopes,
                        chunk_text=text,
                        seq=len(chunks),
                        metadata=self._build_issue_metadata(
                            document_id=document_id,
                            row=row,
                            extra={
                                "chunk_kind": "analysis_flow",
                                "flow_name": flow_chunk.flow_name,
                                "labels": ",".join(flow_chunk.labels),
                                "row_index": row.row_index,
                                "sheet_name": row.sheet_name,
                                "split_mode": "split",
                            },
                        ),
                    )
                )

        summary = {
            "overview_chunks": overview_count,
            "analysis_chunks": analysis_count,
        }
        return chunks, summary

    def _should_split_analysis(self, row: IssueRow) -> bool:
        if not row.analysis:
            return False
        return len(self._build_single_row_text(row)) > self.settings.excel_row_max_chars

    def _build_overview_text(self, row: IssueRow) -> str:
        return "\n".join(self._build_overview_lines(row))

    def _build_single_row_text(self, row: IssueRow) -> str:
        lines = self._build_overview_lines(row)
        if row.analysis:
            lines.append(f"[문제점 분석 내용] {row.analysis}")
        return "\n".join(lines)

    def _build_overview_lines(self, row: IssueRow) -> list[str]:
        lines = [
            f"[이슈] {row.title}",
            f"[등록일] {row.registered_date.isoformat()}",
            f"[기본 확인내용] {row.check_text}",
            f"[기본 작업내용] {row.work_text}",
            f"[업무지시] {row.instruction_text}",
            f"[담당자] {row.assignee}",
            f"[진행] {row.status}",
        ]
        if row.status_raw:
            lines.append(f"[진행(담당자)] {row.status_raw}")
        if row.start_date:
            lines.append(f"[업무시작일] {row.start_date.isoformat()}")
        if row.due_date:
            lines.append(f"[완료예정] {row.due_date.isoformat()}")
        if row.completed_date:
            lines.append(f"[완료일] {row.completed_date.isoformat()}")
        return lines

    def _build_issue_metadata(
        self,
        *,
        document_id: str,
        row: IssueRow,
        extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "doc_type": "issue",
            "doc_id": document_id,
            "title": row.title,
            "issue_title": row.title,
            "assignee": row.assignee,
            "status": row.status,
            "status_raw": row.status_raw,
            "created_at_iso": row.registered_date.isoformat(),
            "created_at_int": self._date_to_int(row.registered_date),
            "start_at_iso": row.start_date.isoformat() if row.start_date else None,
            "start_at_int": self._date_to_int(row.start_date),
            "due_at_iso": row.due_date.isoformat() if row.due_date else None,
            "due_at_int": self._date_to_int(row.due_date),
            "completed_at_iso": row.completed_date.isoformat() if row.completed_date else None,
            "completed_at_int": self._date_to_int(row.completed_date),
            "date": row.registered_date.isoformat(),
            "date_int": self._date_to_int(row.registered_date),
        }
        if extra:
            metadata.update(extra)
        return metadata

    @staticmethod
    def _date_to_int(value: date | None) -> int | None:
        if value is None:
            return None
        return int(value.strftime("%Y%m%d"))

    def _build_chunk(
        self,
        *,
        chunk_id: str,
        document_id: str,
        row: IssueRow,
        access_scopes: list[str],
        chunk_text: str,
        seq: int,
        metadata: dict[str, object],
    ) -> ChunkRecord:
        return ChunkRecord(
            chunk_id=chunk_id,
            document_id=document_id,
            document_type="issue",
            channel=row.sheet_name,
            user_name=row.assignee or "unassigned",
            message_date=row.registered_date,
            message_time=time(0, 0, 0),
            access_scopes=access_scopes,
            chunk_text=chunk_text,
            token_count=ChunkingService._estimate_tokens(chunk_text),
            seq=seq,
            metadata=metadata,
            original_lines=chunk_text.splitlines(),
        )
