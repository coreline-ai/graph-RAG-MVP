from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import BytesIO
import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile

from app.config import Settings
from app.schemas import IssueRow, WorkbookParseResult
from app.services.korean_nlp import normalize_status

_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "docrel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
_REQUIRED_HEADERS = (
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
)
_HEADER_ALIASES = {
    "이슈": "모델 이슈 검토 사항",
    "진행": "진행(담당자)",
    "문제 원인 분석 결과": "문제점 분석 내용 (담당자 Comments)",
}
_STATUS_PREFIX_RE = re.compile(r"^\[(?P<label>[^\]]+)\]")


class PayloadTooLargeError(ValueError):
    pass


@dataclass
class _SheetRows:
    name: str
    headers: list[str]
    rows: list[dict[str, object]]


class WorkbookParser:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def validate_xlsx_signature(self, file_bytes: bytes) -> None:
        if not zipfile.is_zipfile(BytesIO(file_bytes)):
            raise ValueError("Invalid workbook: not a valid .xlsx file")

    def parse_issue_workbook(
        self,
        file_bytes: bytes,
        *,
        row_limit: int | None = None,
    ) -> WorkbookParseResult:
        self.validate_xlsx_signature(file_bytes)
        sheet_rows = self._load_rows(file_bytes)
        if not sheet_rows:
            raise ValueError("Workbook does not contain any visible sheets.")

        warnings: list[str] = []
        rows: list[IssueRow] = []
        total_rows = 0
        skipped_rows = 0
        issue_sheet_count = 0

        for sheet in sheet_rows:
            canonical_sheet = self._canonicalize_sheet(sheet)
            matched_headers = set(canonical_sheet.headers).intersection(_REQUIRED_HEADERS)
            if not matched_headers:
                warnings.append(f"Skipping non-issue sheet {sheet.name!r}")
                continue
            self._validate_headers(canonical_sheet.name, canonical_sheet.headers, canonical_sheet.rows)
            issue_sheet_count += 1
            for row_index, raw in enumerate(canonical_sheet.rows, start=2):
                total_rows += 1
                if row_limit is not None and total_rows > row_limit:
                    raise PayloadTooLargeError(
                        f"Workbook has {total_rows} rows. Use CLI for files larger than {row_limit} rows."
                    )
                try:
                    issue_row = self._normalize_row(canonical_sheet.name, row_index, raw)
                except ValueError as exc:
                    skipped_rows += 1
                    warnings.append(f"{canonical_sheet.name}!{row_index}: {exc}")
                    continue
                rows.append(issue_row)

        if issue_sheet_count == 0:
            raise ValueError("Workbook does not contain any issue data sheets.")

        return WorkbookParseResult(
            rows=rows,
            total_rows=total_rows,
            skipped_rows=skipped_rows,
            warnings=warnings,
        )

    def _canonicalize_sheet(self, sheet: _SheetRows) -> _SheetRows:
        canonical_headers = [self._canonical_header(header) for header in sheet.headers]
        canonical_rows: list[dict[str, object]] = []
        for raw in sheet.rows:
            row_map: dict[str, object] = {}
            for header, value in raw.items():
                canonical = self._canonical_header(header)
                row_map[canonical] = value
            canonical_rows.append(row_map)
        return _SheetRows(name=sheet.name, headers=canonical_headers, rows=canonical_rows)

    @staticmethod
    def _canonical_header(header: str) -> str:
        return _HEADER_ALIASES.get(header, header)

    def _validate_headers(
        self,
        sheet_name: str,
        headers: list[str],
        rows: list[dict[str, object]],
    ) -> None:
        if not rows:
            raise ValueError(f"Sheet {sheet_name!r} does not contain any data rows.")
        header_set = set(headers)
        missing = [header for header in _REQUIRED_HEADERS if header not in header_set]
        if missing:
            raise ValueError(f"Missing required headers in sheet {sheet_name!r}: {', '.join(missing)}")

    def _normalize_row(self, sheet_name: str, row_index: int, raw: dict[str, object]) -> IssueRow:
        title = self._normalize_text(raw.get("모델 이슈 검토 사항"))
        if not title:
            raise ValueError("missing issue title")
        registered_date = self._parse_optional_date(raw.get("등록일"))
        if registered_date is None:
            raise ValueError("invalid 등록일")
        assignee = self._normalize_text(raw.get("담당자")) or "unassigned"
        status_raw = self._normalize_text(raw.get("진행(담당자)"))
        helper_status = self._normalize_text(raw.get("상태_도우미"))
        return IssueRow(
            sheet_name=sheet_name,
            row_index=row_index,
            title=title,
            registered_date=registered_date,
            start_date=self._parse_optional_date(raw.get("업무시작일")),
            due_date=self._parse_optional_date(raw.get("완료예정")),
            completed_date=self._parse_optional_date(raw.get("완료일")),
            check_text=self._normalize_text(raw.get("기본 확인내용")),
            work_text=self._normalize_text(raw.get("기본 작업내용")),
            instruction_text=self._normalize_text(raw.get("업무지시")),
            assignee=assignee,
            status=self._normalize_issue_status(status_raw, helper_status),
            status_raw=status_raw,
            analysis=self._normalize_multiline_text(raw.get("문제점 분석 내용 (담당자 Comments)")),
        )

    @staticmethod
    def _normalize_text(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        return " ".join(str(value).split()).strip()

    @staticmethod
    def _normalize_multiline_text(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        lines = [line.strip() for line in str(value).splitlines()]
        cleaned = [" ".join(line.split()) for line in lines if line.strip()]
        return "\n".join(cleaned).strip()

    def _parse_optional_date(self, value: object) -> date | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if isinstance(value, (int, float)):
            return self._excel_serial_to_date(float(value))
        text = self._normalize_text(value)
        if not text:
            return None
        if re.fullmatch(r"\d+(?:\.\d+)?", text):
            return self._excel_serial_to_date(float(text))
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y년 %m월 %d일"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(text).date()
        except ValueError:
            return None

    @staticmethod
    def _normalize_issue_status(progress_text: str, helper_status: str) -> str:
        if progress_text:
            match = _STATUS_PREFIX_RE.match(progress_text)
            if match:
                return normalize_status(match.group("label"))
            normalized = normalize_status(progress_text)
            if normalized:
                return normalized
        return normalize_status(helper_status)

    @staticmethod
    def _excel_serial_to_date(serial: float) -> date:
        return (datetime(1899, 12, 30) + timedelta(days=serial)).date()

    def _load_rows(self, file_bytes: bytes) -> list[_SheetRows]:
        try:
            import openpyxl  # type: ignore
        except Exception:
            return self._load_rows_from_xml(file_bytes)
        try:
            sheets = self._load_rows_with_openpyxl(file_bytes, openpyxl)
        except Exception:
            return self._load_rows_from_xml(file_bytes)
        if sheets:
            return sheets
        return self._load_rows_from_xml(file_bytes)

    def _load_rows_with_openpyxl(self, file_bytes: bytes, openpyxl) -> list[_SheetRows]:
        workbook = openpyxl.load_workbook(BytesIO(file_bytes), data_only=True, read_only=True)
        sheets: list[_SheetRows] = []
        for sheet in workbook.worksheets:
            if sheet.sheet_state != "visible":
                continue
            values = list(sheet.iter_rows(values_only=True))
            if not values:
                continue
            headers = [self._normalize_text(value) for value in values[0]]
            rows: list[dict[str, object]] = []
            for row in values[1:]:
                if not any(value not in (None, "") for value in row):
                    continue
                row_map: dict[str, object] = {}
                for index, header in enumerate(headers):
                    if header:
                        row_map[header] = row[index] if index < len(row) else None
                rows.append(row_map)
            if rows:
                sheets.append(_SheetRows(name=sheet.title, headers=headers, rows=rows))
        return sheets

    def _load_rows_from_xml(self, file_bytes: bytes) -> list[_SheetRows]:
        with zipfile.ZipFile(BytesIO(file_bytes)) as archive:
            shared_strings = self._read_shared_strings(archive)
            sheet_meta = self._read_visible_sheet_paths(archive)
            sheets: list[_SheetRows] = []
            for sheet_name, sheet_path in sheet_meta:
                headers, rows = self._read_sheet_rows(archive, sheet_path, shared_strings)
                if rows:
                    sheets.append(_SheetRows(name=sheet_name, headers=headers, rows=rows))
            return sheets

    def _read_shared_strings(self, archive: zipfile.ZipFile) -> list[str]:
        if "xl/sharedStrings.xml" not in archive.namelist():
            return []
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        values: list[str] = []
        for node in root.findall("main:si", _NS):
            text = "".join(t.text or "" for t in node.findall(".//main:t", _NS))
            values.append(text)
        return values

    def _read_visible_sheet_paths(self, archive: zipfile.ZipFile) -> list[tuple[str, str]]:
        workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        rel_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rel_root.findall("rel:Relationship", _NS)
        }
        sheets: list[tuple[str, str]] = []
        for sheet in workbook_root.findall("main:sheets/main:sheet", _NS):
            if sheet.attrib.get("state") == "hidden":
                continue
            rel_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
            target = rel_map.get(rel_id or "")
            if not target:
                continue
            normalized = self._normalize_sheet_target(target)
            sheets.append((sheet.attrib.get("name", "Sheet"), normalized))
        return sheets

    @staticmethod
    def _normalize_sheet_target(target: str) -> str:
        stripped = target.lstrip("/")
        if stripped.startswith("xl/"):
            return posixpath.normpath(stripped)
        return posixpath.normpath(posixpath.join("xl", stripped))

    def _read_sheet_rows(
        self,
        archive: zipfile.ZipFile,
        sheet_path: str,
        shared_strings: list[str],
    ) -> tuple[list[str], list[dict[str, object]]]:
        root = ET.fromstring(archive.read(sheet_path))
        rows: list[dict[str, object]] = []
        headers: list[str] = []
        for row_index, row in enumerate(root.findall(".//main:sheetData/main:row", _NS), start=1):
            values = self._parse_row_cells(row, shared_strings)
            if not values:
                continue
            max_index = max(values)
            flat_row = [values.get(index) for index in range(1, max_index + 1)]
            if row_index == 1:
                headers = [self._normalize_text(value) for value in flat_row]
                continue
            if not headers:
                continue
            if not any(value not in (None, "") for value in flat_row):
                continue
            row_map: dict[str, object] = {}
            for index, header in enumerate(headers, start=1):
                if header:
                    row_map[header] = values.get(index)
            rows.append(row_map)
        return headers, rows

    def _parse_row_cells(self, row: ET.Element, shared_strings: list[str]) -> dict[int, object]:
        values: dict[int, object] = {}
        for cell in row.findall("main:c", _NS):
            ref = cell.attrib.get("r", "A1")
            index = self._column_to_index(ref)
            cell_type = cell.attrib.get("t")
            value: object
            if cell_type == "s":
                raw = cell.findtext("main:v", default="", namespaces=_NS)
                value = shared_strings[int(raw)] if raw.isdigit() and int(raw) < len(shared_strings) else ""
            elif cell_type == "inlineStr":
                value = "".join(t.text or "" for t in cell.findall(".//main:t", _NS))
            else:
                raw = cell.findtext("main:v", default="", namespaces=_NS)
                if raw == "":
                    value = ""
                elif re.fullmatch(r"-?\d+", raw):
                    value = int(raw)
                elif re.fullmatch(r"-?\d+\.\d+", raw):
                    value = float(raw)
                else:
                    value = raw
            values[index] = value
        return values

    @staticmethod
    def _column_to_index(cell_ref: str) -> int:
        letters = "".join(ch for ch in cell_ref if ch.isalpha())
        total = 0
        for ch in letters:
            total = total * 26 + (ord(ch.upper()) - 64)
        return total or 1
