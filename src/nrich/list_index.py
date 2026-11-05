"""Build title → KSM_IDX index from NRICH list pages or Open API cache."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from src.nrich.constants import DEFAULT_MENU_IDX, LIST_VIEW_PATH, NRICH_BASE_URL, REQUEST_HEADERS
from src.nrich.text_utils import normalize_title_key, parse_list_title


class TitleIndexBuilder:
    def __init__(
        self,
        base_url: str = NRICH_BASE_URL,
        sleep_seconds: float = 0.35,
        timeout: float = 30.0,
        max_retries: int = 5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.sleep_seconds = sleep_seconds
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(REQUEST_HEADERS)

    def fetch_page(self, pageindex: int) -> list[tuple[str, str, str]]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.sleep_seconds)
                response = self.session.get(
                    f"{self.base_url}{LIST_VIEW_PATH}",
                    params={"menuIdx": DEFAULT_MENU_IDX, "pageindex": str(pageindex)},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                break
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(min(2 ** attempt, 20))
        else:
            raise last_error  # type: ignore[misc]

        soup = BeautifulSoup(response.text, "lxml")
        rows: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        for anchor in soup.find_all("a", href=True):
            match = re.search(r"ksm_idx=(\d+)", anchor["href"])
            if not match:
                continue
            ksm_idx = match.group(1)
            if ksm_idx in seen:
                continue
            label = anchor.get_text(strip=True)
            if not label:
                continue
            seen.add(ksm_idx)
            title_ko, title_hanja = parse_list_title(label)
            title_key = normalize_title_key(title_ko, title_hanja)
            rows.append((ksm_idx, title_key, label))
        return rows

    def build_index(
        self,
        max_pages: int | None = None,
        stop_after_empty: int = 3,
        start_page: int = 1,
        existing: dict[str, str] | None = None,
        checkpoint_path: Path | None = None,
        checkpoint_every: int = 25,
    ) -> dict[str, str]:
        title_to_ksm = dict(existing or {})
        empty_streak = 0
        page = start_page
        while True:
            if max_pages is not None and page > max_pages:
                break
            batch = self.fetch_page(page)
            if not batch:
                empty_streak += 1
                if empty_streak >= stop_after_empty:
                    break
            else:
                empty_streak = 0
                for ksm_idx, title_key, _label in batch:
                    title_to_ksm.setdefault(title_key, ksm_idx)
            if checkpoint_path and page % checkpoint_every == 0:
                self.save_index(checkpoint_path, title_to_ksm)
            page += 1
        if checkpoint_path:
            self.save_index(checkpoint_path, title_to_ksm)
        return title_to_ksm

    @staticmethod
    def save_index(path: Path, title_to_ksm: dict[str, str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(title_to_ksm, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def load_index(path: Path) -> dict[str, str]:
        return json.loads(path.read_text(encoding="utf-8"))
