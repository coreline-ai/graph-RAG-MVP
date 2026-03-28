from __future__ import annotations

from datetime import date
from io import BytesIO
import zipfile

import pytest

from app.adapters.embedding_cache_store import EmbeddingCacheStore
from app.config import Settings
from app.schemas import IssueRow
from app.services.behavior_labeler import BehaviorLabeler
from app.services.issue_chunking import IssueChunkingService
from app.services.workbook_parser import PayloadTooLargeError, WorkbookParser

_HEADERS = [
    "모델 이슈 검토 사항",
    "등록일",
    "기본 확인내용",
    "기본 작업내용",
    "업무지시",
    "담당자",
    "업무시작일",
    "완료예정",
    "진행(담당자)",
    "완료일",
    "문제점 분석 내용 (담당자 Comments)",
    "상태_도우미",
]


def _column_letter(index: int) -> str:
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _make_sheet_xml(headers: list[str], rows: list[list[object]]) -> str:
    sheet_rows: list[str] = []
    all_rows = [headers, *rows]
    for row_index, row in enumerate(all_rows, start=1):
        cells: list[str] = []
        for col_index, value in enumerate(row, start=1):
            ref = f"{_column_letter(col_index)}{row_index}"
            if isinstance(value, (int, float)):
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                text = str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>')
        sheet_rows.append(f"<row r=\"{row_index}\">{''.join(cells)}</row>")
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(sheet_rows)}</sheetData>"
        "</worksheet>"
    )


def _make_multisheet_workbook(
    sheets: list[tuple[str, list[str], list[list[object]], str]],
) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        overrides = "".join(
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            for index, _ in enumerate(sheets, start=1)
        )
        archive.writestr(
            "[Content_Types].xml",
            f"""<?xml version="1.0" encoding="UTF-8"?>
            <Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
              <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
              <Default Extension="xml" ContentType="application/xml"/>
              <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
              {overrides}
            </Types>""",
        )
        archive.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
            <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
              <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
            </Relationships>""",
        )
        archive.writestr(
            "xl/workbook.xml",
            f"""<?xml version="1.0" encoding="UTF-8"?>
            <workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
                      xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
              <sheets>
                {''.join(f'<sheet name="{name}" sheetId="{index}" r:id="rId{index}"/>' for index, (name, _, _, _) in enumerate(sheets, start=1))}
              </sheets>
            </workbook>""",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            f"""<?xml version="1.0" encoding="UTF-8"?>
            <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
              {''.join(f'<Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="{target}"/>' for index, (_, _, _, target) in enumerate(sheets, start=1))}
            </Relationships>""",
        )
        for index, (_, headers, rows, _) in enumerate(sheets, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _make_sheet_xml(headers, rows))
    return buffer.getvalue()


def _make_workbook(
    headers: list[str],
    rows: list[list[object]],
    *,
    sheet_target: str = "worksheets/sheet1.xml",
) -> bytes:
    return _make_multisheet_workbook([("이슈시트", headers, rows, sheet_target)])


def test_invalid_html_bytes_are_rejected() -> None:
    parser = WorkbookParser(Settings())
    with pytest.raises(ValueError, match="Invalid workbook"):
        parser.parse_issue_workbook(b"<!DOCTYPE html>")


def test_missing_required_headers_fail() -> None:
    parser = WorkbookParser(Settings())
    workbook = _make_workbook(["모델 이슈 검토 사항", "등록일"], [["GPU 메모리 부족", "2025-03-01"]])

    with pytest.raises(ValueError, match="Missing required headers"):
        parser.parse_issue_workbook(workbook)


def test_excel_serial_and_string_dates_are_normalized() -> None:
    parser = WorkbookParser(Settings())
    workbook = _make_workbook(
        _HEADERS,
        [
            ["GPU 메모리 부족", 45717, "로그 확인", "batch size 조정", "정책 마련", "Sujin", 45717, 45723, "[진행중] 로그 확인", "", "로그를 확인해 보니 OOM이 발견되었다.", "진행중"],
            ["API 응답 지연", "2025-03-02", "모니터링 확인", "캐시 적용", "추가 검증", "Hyunwoo", "2025-03-02", "2025-03-03", "[완료] 조치 반영", "2025-03-03", "", "완료"],
        ],
    )

    result = parser.parse_issue_workbook(workbook)

    assert result.total_rows == 2
    assert result.skipped_rows == 0
    assert result.rows[0].registered_date == date(2025, 3, 1)
    assert result.rows[0].start_date == date(2025, 3, 1)
    assert result.rows[0].due_date == date(2025, 3, 7)
    assert result.rows[1].registered_date == date(2025, 3, 2)
    assert result.rows[1].completed_date == date(2025, 3, 3)


def test_xml_fallback_handles_targets_with_xl_prefix() -> None:
    parser = WorkbookParser(Settings())
    workbook = _make_workbook(
        _HEADERS,
        [["GPU 메모리 부족", "2025-03-01", "로그 확인", "batch size 조정", "정책 마련", "Sujin", "2025-03-01", "2025-03-07", "[진행중] 영향도 분석", "", "로그를 확인해 보니 OOM이 발견되었다.", "진행중"]],
        sheet_target="xl/worksheets/sheet1.xml",
    )

    result = parser.parse_issue_workbook(workbook)

    assert result.total_rows == 1
    assert result.rows[0].title == "GPU 메모리 부족"


def test_actual_issue_dataset_headers_and_status_fields_are_normalized() -> None:
    parser = WorkbookParser(Settings())
    workbook = _make_workbook(
        _HEADERS,
        [[
            "GPU 메모리 부족",
            "2025-03-01",
            "로그 확인",
            "batch size 조정",
            "정책 마련",
            "Sujin",
            "2025-03-01",
            "2025-03-07",
            "[진행중] 영향도 분석",
            "",
            "로그를 확인해 보니 OOM이 발견되었다.",
            "진행중",
        ]],
    )

    result = parser.parse_issue_workbook(workbook)

    assert result.total_rows == 1
    assert result.rows[0].title == "GPU 메모리 부족"
    assert result.rows[0].status == "진행"
    assert result.rows[0].status_raw == "[진행중] 영향도 분석"
    assert result.rows[0].start_date == date(2025, 3, 1)
    assert result.rows[0].due_date == date(2025, 3, 7)
    assert result.rows[0].analysis == "로그를 확인해 보니 OOM이 발견되었다."


def test_status_helper_is_used_when_progress_column_is_empty() -> None:
    parser = WorkbookParser(Settings())
    workbook = _make_workbook(
        _HEADERS,
        [[
            "임베딩 캐시 갱신 누락",
            "2025-03-01",
            "",
            "",
            "",
            "Jiho",
            "",
            "",
            "",
            "",
            "",
            "검증대기",
        ]],
    )

    result = parser.parse_issue_workbook(workbook)

    assert result.rows[0].status == "검증대기"


def test_analysis_multiline_text_is_preserved() -> None:
    parser = WorkbookParser(Settings())
    workbook = _make_workbook(
        _HEADERS,
        [[
            "장문 요청 처리 중 GPU 메모리 부족 발생",
            "2025-03-01",
            "",
            "",
            "",
            "Sujin",
            "",
            "",
            "[진행중] 영향도 분석",
            "",
            "원인 요약: GPU 메모리 부족\n확인 근거: OOM 로그 반복",
            "진행중",
        ]],
    )

    result = parser.parse_issue_workbook(workbook)

    assert result.rows[0].analysis == "원인 요약: GPU 메모리 부족\n확인 근거: OOM 로그 반복"


def test_non_issue_sheets_are_skipped_with_warning() -> None:
    parser = WorkbookParser(Settings())
    workbook = _make_multisheet_workbook(
        [
            ("이슈시트", _HEADERS, [["GPU 메모리 부족", "2025-03-01", "", "", "", "Sujin", "", "", "[진행중] 로그 확인", "", "로그 확인", "진행중"]], "worksheets/sheet1.xml"),
            ("요약", ["모델 이슈 데이터셋 요약"], [["총 이슈 건수"]], "worksheets/sheet2.xml"),
        ]
    )

    result = parser.parse_issue_workbook(workbook)

    assert result.total_rows == 1
    assert len(result.rows) == 1
    assert any("Skipping non-issue sheet '요약'" == warning for warning in result.warnings)


def test_row_limit_raises_payload_too_large() -> None:
    parser = WorkbookParser(Settings())
    workbook = _make_workbook(
        _HEADERS,
        [
            ["이슈1", "2025-03-01", "", "", "", "", "", "", "", "", "", ""],
            ["이슈2", "2025-03-02", "", "", "", "", "", "", "", "", "", ""],
        ],
    )

    with pytest.raises(PayloadTooLargeError):
        parser.parse_issue_workbook(workbook, row_limit=1)


def test_issue_chunking_keeps_single_chunk_for_short_rows() -> None:
    chunking = IssueChunkingService(Settings(), BehaviorLabeler(Settings()))
    rows = [
        IssueRow(
            sheet_name="이슈시트",
            row_index=2,
            title="GPU 메모리 부족",
            registered_date=date(2025, 3, 1),
            start_date=date(2025, 3, 1),
            due_date=date(2025, 3, 5),
            check_text="로그 확인",
            work_text="batch size 조정",
            instruction_text="정책 마련",
            assignee="Sujin",
            status="진행",
            status_raw="[진행중] 영향도 분석",
            analysis="OOM 발생 원인 확인.",
        )
    ]

    chunks, summary = chunking.build_chunks(rows, document_id="doc-1", default_access_scopes=["public"])

    assert len(chunks) == 1
    assert chunks[0].document_type == "issue"
    assert chunks[0].metadata["chunk_kind"] == "overview"
    assert chunks[0].metadata["doc_type"] == "issue"
    assert chunks[0].metadata["doc_id"] == "doc-1"
    assert chunks[0].metadata["created_at_int"] == 20250301
    assert chunks[0].metadata["start_at_int"] == 20250301
    assert chunks[0].metadata["due_at_int"] == 20250305
    assert chunks[0].metadata["completed_at_int"] is None
    assert "[문제점 분석 내용]" in chunks[0].chunk_text
    assert summary == {"overview_chunks": 1, "analysis_chunks": 0}


def test_issue_chunking_splits_long_analysis_into_overview_and_flows() -> None:
    settings = Settings(excel_row_max_chars=120)
    chunking = IssueChunkingService(settings, BehaviorLabeler(settings))
    rows = [
        IssueRow(
            sheet_name="이슈시트",
            row_index=2,
            title="GPU 메모리 부족",
            registered_date=date(2025, 3, 1),
            start_date=date(2025, 3, 1),
            due_date=date(2025, 3, 5),
            completed_date=date(2025, 3, 6),
            check_text="로그 확인",
            work_text="batch size 조정",
            instruction_text="정책 마련",
            assignee="Sujin",
            status="진행",
            status_raw="[진행중] 영향도 분석",
            analysis=(
                "로그를 확인해 보니 OOM이 발견되었다. "
                "batch size를 조정했으나 동일 증상이 재현되었다. "
                "메모리 제한을 수정하여 정상 복구되었다. "
                "재현 테스트 완료 후 모니터링 중이다."
            ),
        )
    ]

    chunks, summary = chunking.build_chunks(rows, document_id="doc-1", default_access_scopes=["public"])

    assert len(chunks) >= 2
    assert chunks[0].metadata["chunk_kind"] == "overview"
    assert chunks[0].metadata["split_mode"] == "split"
    assert "[문제점 분석 내용]" not in chunks[0].chunk_text
    assert all(chunk.metadata["chunk_kind"] == "analysis_flow" for chunk in chunks[1:])
    assert all(chunk.metadata["completed_at_int"] == 20250306 for chunk in chunks)
    assert summary["overview_chunks"] == 1
    assert summary["analysis_chunks"] == len(chunks) - 1


def test_behavior_labeler_uses_structured_section_boundaries() -> None:
    labeler = BehaviorLabeler(Settings())

    chunks = labeler.split_and_label(
        "원인 요약: 긴 문맥 요청이 집중되며 OOM이 발생했다.\n"
        "확인 근거: worker 로그에 OOM killed 이벤트가 반복됐다.\n"
        "추가 조치: 최대 토큰 상한과 batch size ceiling을 함께 조정해야 한다."
    )

    assert len(chunks) == 2
    assert chunks[0].text.startswith("원인 요약:")
    assert chunks[-1].flow_name == "후속 조치"


def test_embedding_cache_key_depends_on_model_name() -> None:
    key_a = EmbeddingCacheStore.build_cache_key("bge-m3", "서버 배포 완료")
    key_b = EmbeddingCacheStore.build_cache_key("multilingual-e5", "서버 배포 완료")

    assert key_a != key_b
