"""Parse NRICH bulk XLS export (meta + readings sheets)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

from src.nrich.text_utils import normalize_title_key


META_HEADER_ROW = 2
META_DATA_START = 4
PAN_HEADER_ROW = 2
PAN_DATA_START = 4

META_COLUMNS = [
    "xls_seq",
    "title_ko",
    "title_hanja",
    "formal_name",
    "era",
    "year",
    "object_type",
    "material",
    "dimensions",
    "excavation_addr",
    "location",
    "script_style",
    "author_praise",
    "author_engrave",
    "author_compose",
    "has_reading",
    "has_translation",
]

PAN_COLUMNS = [
    "xls_seq",
    "title_ko",
    "title_hanja",
    "reader",
    "reading_text",
    "interpreter",
    "translation_text",
    "notes",
]


@dataclass
class XlsMetaRow:
    xls_seq: int
    title_ko: str
    title_hanja: str
    era: str
    year: str
    location: str
    title_key: str


@dataclass
class XlsReadingRow:
    xls_seq: int
    title_ko: str
    title_hanja: str
    reader: str
    reading_text: str
    title_key: str


def _cell_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def load_meta_rows(xls_path: Path) -> list[XlsMetaRow]:
    sheet = pd.read_excel(xls_path, sheet_name="메타정보", header=None, engine="xlrd")
    rows: list[XlsMetaRow] = []
    for idx in range(META_DATA_START, len(sheet)):
        raw = [_cell_str(sheet.iat[idx, col]) for col in range(len(META_COLUMNS))]
        if not any(raw):
            continue
        try:
            xls_seq = int(float(raw[0]))
        except ValueError:
            continue
        title_ko = raw[1]
        title_hanja = raw[2]
        rows.append(
            XlsMetaRow(
                xls_seq=xls_seq,
                title_ko=title_ko,
                title_hanja=title_hanja,
                era=raw[4],
                year=raw[5],
                location=raw[10],
                title_key=normalize_title_key(title_ko, title_hanja),
            )
        )
    return rows


def load_reading_rows(xls_path: Path) -> list[XlsReadingRow]:
    sheet = pd.read_excel(xls_path, sheet_name="판독문,해석문", header=None, engine="xlrd")
    rows: list[XlsReadingRow] = []
    for idx in range(PAN_DATA_START, len(sheet)):
        raw = [_cell_str(sheet.iat[idx, col]) for col in range(len(PAN_COLUMNS))]
        if not any(raw):
            continue
        reading_text = raw[4]
        if not reading_text:
            continue
        try:
            xls_seq = int(float(raw[0]))
        except ValueError:
            continue
        title_ko = raw[1]
        title_hanja = raw[2]
        rows.append(
            XlsReadingRow(
                xls_seq=xls_seq,
                title_ko=title_ko,
                title_hanja=title_hanja,
                reader=raw[3],
                reading_text=reading_text,
                title_key=normalize_title_key(title_ko, title_hanja),
            )
        )
    return rows


def iter_readings_with_meta(xls_path: Path) -> Iterator[tuple[XlsReadingRow, XlsMetaRow | None]]:
    meta_by_key = {row.title_key: row for row in load_meta_rows(xls_path)}
    for reading in load_reading_rows(xls_path):
        yield reading, meta_by_key.get(reading.title_key)
