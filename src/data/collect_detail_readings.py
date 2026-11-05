"""Collect full 판독문 via Open API index → ksmUsrView → ksmPandokWriter."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.nrich.api_client import NrichApiClient
from src.nrich.detail_client import NrichDetailClient
from src.nrich.list_index import TitleIndexBuilder
DETAIL_COLUMNS = [
    "ksm_idx",
    "pan_idx",
    "reader",
    "reading_type",
    "text_len",
    "detail_url",
    "pandok_view_url",
    "text_raw",
]


def _load_ksm_ids(index_path: Path, max_pages: int | None) -> list[str]:
    if index_path.exists():
        mapping = TitleIndexBuilder.load_index(index_path)
        return sorted(set(mapping.values()), key=lambda value: int(value))

    api = NrichApiClient()
    if api.enabled:
        mapping = api.build_title_index(max_pages=max_pages)
    else:
        mapping = TitleIndexBuilder().build_index(max_pages=max_pages)
    TitleIndexBuilder.save_index(index_path, mapping)
    return sorted(set(mapping.values()), key=lambda value: int(value))


def collect_detail_readings(
    ksm_ids: list[str],
    client: NrichDetailClient,
    limit: int | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    for idx, ksm_idx in enumerate(ksm_ids):
        if limit is not None and idx >= limit:
            break
        try:
            readings = client.fetch_pandok_readings(ksm_idx)
        except Exception as exc:  # noqa: BLE001 - collect pipeline should continue
            rows.append(
                {
                    "ksm_idx": ksm_idx,
                    "pan_idx": "",
                    "reader": "",
                    "reading_type": "P",
                    "text_len": 0,
                    "detail_url": client.detail_url(ksm_idx),
                    "pandok_view_url": "",
                    "text_raw": "",
                    "error": str(exc),
                }
            )
            continue
        if not readings:
            rows.append(
                {
                    "ksm_idx": ksm_idx,
                    "pan_idx": "",
                    "reader": "",
                    "reading_type": "P",
                    "text_len": 0,
                    "detail_url": client.detail_url(ksm_idx),
                    "pandok_view_url": "",
                    "text_raw": "",
                    "error": "no_pandok_readings",
                }
            )
            continue
        for reading in readings:
            rows.append(
                {
                    "ksm_idx": reading.ksm_idx,
                    "pan_idx": reading.pan_idx,
                    "reader": reading.reader,
                    "reading_type": reading.reading_type,
                    "text_len": len(reading.text_raw),
                    "detail_url": client.detail_url(reading.ksm_idx),
                    "pandok_view_url": reading.source_url,
                    "text_raw": reading.text_raw,
                    "error": "",
                }
            )
    return pd.DataFrame(rows, columns=DETAIL_COLUMNS + ["error"])


def validate_sample(sample_path: Path, client: NrichDetailClient) -> dict:
    """Validate pipeline on known KSM_IDX samples."""
    expected = {
        "3": 2,
        "5": 1,
    }
    results: dict[str, dict] = {}
    for ksm_idx, min_count in expected.items():
        readings = client.fetch_pandok_readings(ksm_idx)
        results[ksm_idx] = {
            "expected_min": min_count,
            "actual": len(readings),
            "passed": len(readings) >= min_count,
            "max_len": max((len(item.text_raw) for item in readings), default=0),
        }
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--title-index",
        type=Path,
        default=Path("data/interim/title_to_ksm_idx.json"),
    )
    parser.add_argument(
        "--index-max-pages",
        type=int,
        default=20,
        help="Pages for title index when cache missing (use None for full crawl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/candidate_detail_readings.csv"),
    )
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=Path("reports/detail_collection_validation.json"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Number of KSM_IDX records to collect (None for all).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Collect all KSM_IDX from title index (overrides --limit).",
    )
    args = parser.parse_args()

    client = NrichDetailClient()
    validation = validate_sample(args.validation_report, client)
    print("Validation:", validation)

    ksm_ids = _load_ksm_ids(args.title_index, args.index_max_pages)
    limit = None if args.full else args.limit
    df = collect_detail_readings(ksm_ids, client, limit=limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "records": len(df),
        "unique_ksm_idx": df["ksm_idx"].nunique(),
        "errors": int((df["error"] != "").sum()),
        "output": str(args.output),
        "validation": validation,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
