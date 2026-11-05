"""NRICH detail-page client: ksmUsrView → ksmPandokWriter / ajaxPandokClob."""

from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass
from typing import Iterable

import requests

from src.nrich.constants import (
    DEFAULT_MENU_IDX,
    DETAIL_VIEW_PATH,
    NRICH_BASE_URL,
    PANDOK_CLOB_PATH,
    PANDOK_WRITER_PATH,
    REQUEST_HEADERS,
)
from src.nrich.text_utils import normalize_unicode


@dataclass
class DetailReading:
    ksm_idx: str
    pan_idx: str
    reader: str
    reading_type: str
    text_raw: str
    source_url: str


class NrichDetailClient:
    def __init__(
        self,
        base_url: str = NRICH_BASE_URL,
        sleep_seconds: float = 0.2,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.sleep_seconds = sleep_seconds
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(REQUEST_HEADERS)

    def detail_url(self, ksm_idx: str | int) -> str:
        return (
            f"{self.base_url}{DETAIL_VIEW_PATH}"
            f"?menuIdx={DEFAULT_MENU_IDX}&ksm_idx={ksm_idx}"
        )

    def pandok_view_url(self, ksm_idx: str | int, pan_idx: str | int) -> str:
        return (
            f"{self.base_url}/kor/ksmPandokView.do"
            f"?ksm_idx={ksm_idx}&pan_idx1={pan_idx}"
        )

    def _post_json(self, path: str, data: dict) -> dict:
        time.sleep(self.sleep_seconds)
        response = self.session.post(
            f"{self.base_url}{path}",
            data=data,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def fetch_readings(self, ksm_idx: str | int, reading_type: str = "P") -> list[DetailReading]:
        """Return 판독문(P) or 해석문(H) entries for one inscription."""
        payload = {"ksm_idx": str(ksm_idx), "PanDokType": reading_type}
        data = self._post_json(PANDOK_WRITER_PATH, payload)
        readings: list[DetailReading] = []
        for item in data.get("rs", []):
            pan_idx = str(item.get("pan_idx") or "")
            if not pan_idx:
                continue
            reader = str(item.get("sol_pan_psn") or "").strip()
            row_type = str(item.get("sol_pan_typ") or reading_type)
            clob = self.fetch_reading_by_pan_idx(ksm_idx, pan_idx)
            if clob is not None:
                clob.reader = reader or clob.reader
                clob.reading_type = row_type
                readings.append(clob)
                continue
            text = html.unescape(item.get("sol_pan_txt") or "")
            text = normalize_unicode(text.replace("\r\n", "\n"))
            if not text.strip():
                continue
            readings.append(
                DetailReading(
                    ksm_idx=str(ksm_idx),
                    pan_idx=pan_idx,
                    reader=reader,
                    reading_type=row_type,
                    text_raw=text,
                    source_url=self.pandok_view_url(ksm_idx, pan_idx),
                )
            )
        return readings

    def fetch_reading_by_pan_idx(self, ksm_idx: str | int, pan_idx: str | int) -> DetailReading | None:
        data = self._post_json(PANDOK_CLOB_PATH, {"pan_idx": str(pan_idx)})
        rows = data.get("rs") or []
        if not rows:
            return None
        item = rows[0]
        text = html.unescape(item.get("sol_pan_txt") or "")
        text = normalize_unicode(text.replace("\r\n", "\n"))
        if not text.strip():
            return None
        return DetailReading(
            ksm_idx=str(ksm_idx),
            pan_idx=str(pan_idx),
            reader=str(item.get("sol_pan_psn") or "").strip(),
            reading_type=str(item.get("sol_pan_typ") or "P"),
            text_raw=text,
            source_url=self.pandok_view_url(ksm_idx, pan_idx),
        )

    def fetch_pandok_readings(self, ksm_idx: str | int) -> list[DetailReading]:
        """Fetch only 판독문 (P) rows."""
        return [r for r in self.fetch_readings(ksm_idx, reading_type="P") if r.reading_type == "P"]

    def parse_pan_idx_from_detail_html(self, ksm_idx: str | int) -> list[str]:
        time.sleep(self.sleep_seconds)
        response = self.session.get(
            self.detail_url(ksm_idx),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return re.findall(r'name="panDokChkBox"\s+value="(\d+)"', response.text)


def pick_best_reading_match(
    xls_text: str,
    xls_reader: str,
    detail_readings: Iterable[DetailReading],
) -> DetailReading | None:
    """Match XLS row to the closest detail reading by reader name and length."""
    candidates = list(detail_readings)
    if not candidates:
        return None
    reader_norm = (xls_reader or "").strip()
    if reader_norm:
        for item in candidates:
            if item.reader == reader_norm:
                return item
    xls_len = len(xls_text or "")
    return max(candidates, key=lambda item: (len(item.text_raw), item.reader == reader_norm))
