"""Optional NRICH Open API list client (requires service key in NRICH_API_KEY)."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Iterator

import requests

from src.nrich.constants import NRICH_MD_DATA_TO_FIELD, REQUEST_HEADERS
from src.nrich.text_utils import normalize_title_key


class NrichApiClient:
    """Fetch paginated list XML from apis.data.go.kr when a key is configured."""

    DEFAULT_ENDPOINT = (
        "https://apis.data.go.kr/15115633/openapi/15115633"
        "/getKsmSearchList"
    )

    def __init__(
        self,
        service_key: str | None = None,
        endpoint: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.service_key = service_key or os.getenv("NRICH_API_KEY", "")
        self.endpoint = endpoint or os.getenv("NRICH_API_ENDPOINT", self.DEFAULT_ENDPOINT)
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.service_key)

    def iter_records(
        self,
        page_size: int = 100,
        max_pages: int | None = None,
    ) -> Iterator[dict[str, str]]:
        if not self.enabled:
            raise RuntimeError("NRICH_API_KEY is not configured")
        page = 1
        while True:
            if max_pages is not None and page > max_pages:
                break
            params = {
                "serviceKey": self.service_key,
                "pageNo": page,
                "numOfRows": page_size,
            }
            response = requests.get(
                self.endpoint,
                params=params,
                headers=REQUEST_HEADERS,
                timeout=self.timeout,
            )
            response.raise_for_status()
            root = ET.fromstring(response.content)
            items = root.findall(".//item")
            if not items:
                break
            for item in items:
                record: dict[str, str] = {}
                for tag, field in NRICH_MD_DATA_TO_FIELD.items():
                    node = item.find(tag)
                    if node is not None and node.text is not None:
                        record[field] = node.text.strip()
                yield record
            page += 1

    def build_title_index(self, max_pages: int | None = None) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for record in self.iter_records(max_pages=max_pages):
            ksm_idx = record.get("KSM_IDX")
            if not ksm_idx:
                continue
            title_key = normalize_title_key(
                record.get("KSM_KOR_NAME", ""),
                record.get("KSM_CHI_NAME", ""),
            )
            mapping.setdefault(title_key, ksm_idx)
        return mapping
