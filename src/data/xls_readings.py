"""Parse NRICH bulk XLS export (meta + readings sheets)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import xlrd

from nrich_common import normalize_korean_title

META_HEADER_ROW = 2
META_DATA_START = 4
READINGS_HEADER_ROW = 2
READINGS_DATA_START = 4

META_COLUMNS = [
    "row_num",
    "title_ko",
    "title_hanja",
    "formal_name",
    "era",
    "year",
    "type_name",
    "material",
    "size",
    "excavation_addr",
    "location",
    "font",
    "author",
    "carver",
    "scribe",
    "has_pandok",
    "has_haeseok",
]

READINGS_COLUMNS = [
    "row_num",
    "title_ko",
    "title_hanja",
    "reader",
    "pandok_text",
    "interpreter",
    "haeseok_text",
    "notes",
]


@dataclass
class XlsMetaRow:
    row_num: int
    title_ko: str
    title_hanja: str
    era: str
    year: str
    type_name: str
    material: str
    location: str


@dataclass
class XlsReadingRow:
    row_num: int
    title_ko: str
    title_hanja: str
    reader: str
    pandok_text: str
    interpreter: str
    haeseok_text: str
    notes: str


def _cell(row_values: list, index: int) -> str:
    if index >= len(row_values):
        return ""
    value = row_values[index]
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def load_xls_workbook(xls_path: Path) -> xlrd.Book:
    return xlrd.open_workbook(str(xls_path))


def parse_meta_sheet(workbook: xlrd.Book) -> list[XlsMetaRow]:
    sheet = workbook.sheet_by_name("메타정보")
    rows: list[XlsMetaRow] = []
    for row_idx in range(META_DATA_START, sheet.nrows):
        values = sheet.row_values(row_idx)
        title_ko = _cell(values, 1)
        if not title_ko:
            continue
        rows.append(
            XlsMetaRow(
                row_num=int(float(values[0])) if values[0] else row_idx,
                title_ko=title_ko,
                title_hanja=_cell(values, 2),
                era=_cell(values, 4),
                year=_cell(values, 5),
                type_name=_cell(values, 6),
                material=_cell(values, 7),
                location=_cell(values, 10) or _cell(values, 9),
            )
        )
    return rows


def parse_readings_sheet(workbook: xlrd.Book) -> list[XlsReadingRow]:
    sheet = workbook.sheet_by_name("판독문,해석문")
    rows: list[XlsReadingRow] = []
    for row_idx in range(READINGS_DATA_START, sheet.nrows):
        values = sheet.row_values(row_idx)
        title_ko = _cell(values, 1)
        if not title_ko:
            continue
        rows.append(
            XlsReadingRow(
                row_num=int(float(values[0])) if values[0] else row_idx,
                title_ko=title_ko,
                title_hanja=_cell(values, 2),
                reader=_cell(values, 3),
                pandok_text=_cell(values, 4),
                interpreter=_cell(values, 5),
                haeseok_text=_cell(values, 6),
                notes=_cell(values, 7),
            )
        )
    return rows


def index_readings_by_title(rows: list[XlsReadingRow]) -> dict[str, list[XlsReadingRow]]:
    index: dict[str, list[XlsReadingRow]] = {}
    for row in rows:
        key = normalize_korean_title(row.title_ko)
        index.setdefault(key, []).append(row)
    return index


def index_meta_by_title(rows: list[XlsMetaRow]) -> dict[str, XlsMetaRow]:
    return {normalize_korean_title(row.title_ko): row for row in rows}


def default_xls_path(project_root: Path) -> Path:
    return project_root / "data" / "금석문 사용자 검색_2026-07-03.xls"
