"""NRICH portal HTML client (ksmUsrView / ksmPandokView) and Open API list parser.

Moved from the former top-level ``nrich_common`` module, which
``src/data/collect_nrich_detail.py`` imported but which was never committed.
Function bodies are unchanged; only the names that script needs are kept.

This client scrapes the portal HTML pages and pages the ``B551011/KsmService``
list API by ``totalCount``. It is not interchangeable with
``detail_client.NrichDetailClient`` (AJAX endpoints, no HTML-body extraction) or
``api_client.NrichApiClient`` (different endpoint, dict records, empty-page stop).
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable

import requests
from bs4 import BeautifulSoup

# Same values as nrich_common's own constants; aliased so the moved bodies stay unchanged.
from src.nrich.constants import DETAIL_VIEW_PATH as NRICH_DETAIL_PATH
from src.nrich.constants import NRICH_BASE_URL as NRICH_PORTAL_BASE
from src.nrich.constants import NRICH_MD_DATA_TO_FIELD as MD_DATA_FIELD_BY_TAG
from src.nrich.constants import PANDOK_VIEW_PATH as NRICH_PANDOK_PATH

NRICH_LIST_API = "http://apis.data.go.kr/B551011/KsmService/getKsmList"
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; geumseokmun-research/1.0)",
}


@dataclass
class PandokEntry:
    ksm_idx: str
    pan_idx: str
    doc_type: str
    reader: str
    preview_text: str
    preview_length: int


@dataclass
class ListRecord:
    ksm_idx: str
    title_ko: str
    title_hanja: str
    era: str
    year: str
    location: str
    detail_url: str
    raw_fields: dict[str, str]


class NrichClient:
    def __init__(
        self,
        service_key: str | None = None,
        delay_sec: float = 0.35,
        timeout_sec: float = 60.0,
    ) -> None:
        self.service_key = service_key
        self.delay_sec = delay_sec
        self.timeout_sec = timeout_sec
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

    def _sleep(self) -> None:
        if self.delay_sec > 0:
            time.sleep(self.delay_sec)

    def detail_url(self, ksm_idx: str | int) -> str:
        return (
            f"{NRICH_PORTAL_BASE}{NRICH_DETAIL_PATH}?menuIdx=584&ksm_idx={ksm_idx}"
        )

    def pandok_url(self, ksm_idx: str | int, pan_idx: str | int) -> str:
        return (
            f"{NRICH_PORTAL_BASE}{NRICH_PANDOK_PATH}?ksm_idx={ksm_idx}&pan_idx1={pan_idx}"
        )

    def fetch_text(self, url: str) -> str:
        self._sleep()
        response = self.session.get(url, timeout=self.timeout_sec)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or "utf-8"
        return response.text

    def fetch_list_page(
        self,
        page_no: int = 1,
        num_of_rows: int = 100,
        result_type: str = "xml",
    ) -> str:
        if not self.service_key:
            raise ValueError(
                "NRICH Open API list requires service key. Set NRICH_SERVICE_KEY or pass --service-key."
            )
        params = {
            "serviceKey": self.service_key,
            "pageNo": page_no,
            "numOfRows": num_of_rows,
            "resultType": result_type,
        }
        self._sleep()
        response = self.session.get(NRICH_LIST_API, params=params, timeout=self.timeout_sec)
        response.raise_for_status()
        return response.text

    def iter_list_records(
        self,
        max_pages: int | None = None,
        num_of_rows: int = 100,
    ) -> Iterable[ListRecord]:
        page_no = 1
        while True:
            if max_pages is not None and page_no > max_pages:
                return
            xml_text = self.fetch_list_page(page_no=page_no, num_of_rows=num_of_rows)
            records, total_count = parse_list_xml(xml_text)
            if not records:
                return
            yield from records
            if page_no * num_of_rows >= total_count:
                return
            page_no += 1

    def fetch_detail_meta(self, ksm_idx: str | int) -> dict[str, str]:
        html = self.fetch_text(self.detail_url(ksm_idx))
        soup = BeautifulSoup(html, "lxml")
        title_node = soup.select_one("h3.tit") or soup.select_one("em.f_c_b")
        title = title_node.get_text(strip=True) if title_node else ""
        title_ko = title.split("(")[0] if "(" in title else title
        return {
            "ksm_idx": str(ksm_idx),
            "title": title_ko,
            "detail_url": self.detail_url(ksm_idx),
        }

    def fetch_pandok_list(self, ksm_idx: str | int) -> list[PandokEntry]:
        html = self.fetch_text(self.detail_url(ksm_idx))
        soup = BeautifulSoup(html, "lxml")
        entries: list[PandokEntry] = []
        for row in soup.select("table.tbl02 tbody tr"):
            cells = row.find_all("td")
            if len(cells) < 4:
                continue
            doc_type = cells[1].get_text(strip=True)
            reader = cells[2].get_text(strip=True)
            checkbox = row.select_one('input[name="panDokChkBox"]')
            if checkbox is None:
                continue
            preview = cells[3].get_text("\n", strip=False).strip()
            entries.append(
                PandokEntry(
                    ksm_idx=str(ksm_idx),
                    pan_idx=str(checkbox["value"]),
                    doc_type=doc_type,
                    reader=reader,
                    preview_text=preview,
                    preview_length=len(preview),
                )
            )
        return entries

    def fetch_pandok_text(self, ksm_idx: str | int, pan_idx: str | int) -> str:
        html = self.fetch_text(self.pandok_url(ksm_idx, pan_idx))
        soup = BeautifulSoup(html, "lxml")
        node = soup.select_one("#text_all") or soup.select_one(".contentsWrap")
        if node is None:
            return ""
        return extract_hanja_body(node.get_text("\n"))


def parse_list_xml(xml_text: str) -> tuple[list[ListRecord], int]:
    root = ET.fromstring(xml_text)
    total_count = int(_find_text(root, ".//totalCount") or "0")
    records: list[ListRecord] = []
    for item in root.findall(".//item"):
        fields = {
            tag: (child.text or "").strip()
            for tag, child in ((tag, item.find(tag)) for tag in MD_DATA_FIELD_BY_TAG)
            if child is not None
        }
        ksm_idx = fields.get("md_data2", "")
        detail_url = fields.get("md_data22", "")
        if not ksm_idx:
            continue
        if not detail_url:
            detail_url = f"{NRICH_PORTAL_BASE}{NRICH_DETAIL_PATH}?menuIdx=584&ksm_idx={ksm_idx}"
        records.append(
            ListRecord(
                ksm_idx=ksm_idx,
                title_ko=fields.get("md_data3", ""),
                title_hanja=fields.get("md_data4", ""),
                era=fields.get("md_data6", ""),
                year=fields.get("md_data7", ""),
                location=fields.get("md_data11", ""),
                detail_url=detail_url,
                raw_fields=fields,
            )
        )
    return records, total_count


def _find_text(root: ET.Element, path: str) -> str | None:
    node = root.find(path)
    if node is None or node.text is None:
        return None
    return node.text.strip()


def extract_hanja_body(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    start = 0
    for index, line in enumerate(lines):
        if any("\u4e00" <= ch <= "\u9fff" for ch in line):
            start = index
            break
    return "\n".join(lines[start:])
