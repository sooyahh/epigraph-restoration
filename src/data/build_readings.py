"""Build readings.csv from XLS (+ optional detail augmentation)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.nrich.api_client import NrichApiClient
from src.nrich.detail_client import NrichDetailClient, pick_best_reading_match
from src.nrich.list_index import TitleIndexBuilder
from src.nrich.xls_parser import iter_readings_with_meta
from src.preprocess.gsm_v4 import PREPROCESS_RULE_VERSION, preprocess_reading_text

READINGS_COLUMNS = [
    "reading_id",
    "source",
    "source_record_id",
    "inscription_id",
    "artifact_id",
    "title",
    "era",
    "year",
    "material",
    "location",
    "text_raw",
    "text_normalized",
    "text_length",
    "hanja_count",
    "has_real_defect",
    "defect_count",
    "defect_markers_raw",
    "is_training_text",
    "exclude_reason",
    "quality_flag",
    "notes",
]


def _load_title_index(index_path: Path, max_pages: int | None) -> dict[str, str]:
    if index_path.exists():
        return TitleIndexBuilder.load_index(index_path)
    api = NrichApiClient()
    if api.enabled:
        mapping = api.build_title_index(max_pages=max_pages)
    else:
        builder = TitleIndexBuilder()
        mapping = builder.build_index(max_pages=max_pages)
    TitleIndexBuilder.save_index(index_path, mapping)
    return mapping


def _make_reading_id(ksm_idx: str, pan_suffix: str) -> str:
    return f"nrich_ksm_{int(ksm_idx):06d}_{pan_suffix}"


def build_readings_dataframe(
    xls_path: Path,
    title_index: dict[str, str],
    augment_from_detail: bool = False,
    augment_truncated_only: bool = False,
    augment_min_xls_len: int = 5000,
    detail_client: NrichDetailClient | None = None,
) -> tuple[pd.DataFrame, dict]:
    detail_cache: dict[str, list] = {}
    client = detail_client or NrichDetailClient()
    rows: list[dict] = []
    marker_stats: dict[str, int] = {}
    hangul_replaced_total = 0
    augmented_count = 0
    unmapped_titles: set[str] = set()

    for reading, meta in iter_readings_with_meta(xls_path):
        ksm_idx = title_index.get(reading.title_key)
        if not ksm_idx:
            unmapped_titles.add(reading.title_key)
            continue

        text_raw = reading.reading_text
        source_note = f"source=xls | xls_seq={reading.xls_seq} | reader={reading.reader}"

        if augment_from_detail:
            if ksm_idx not in detail_cache:
                detail_cache[ksm_idx] = client.fetch_pandok_readings(ksm_idx)
            match = pick_best_reading_match(
                reading.reading_text,
                reading.reader,
                detail_cache[ksm_idx],
            )
            if match and len(match.text_raw) > len(text_raw):
                text_raw = match.text_raw
                source_note = (
                    f"source=detail_augment | pan_idx={match.pan_idx} | "
                    f"reader={match.reader} | {match.source_url}"
                )
                augmented_count += 1
        elif augment_truncated_only and len(text_raw) >= augment_min_xls_len:
            if ksm_idx not in detail_cache:
                detail_cache[ksm_idx] = client.fetch_pandok_readings(ksm_idx)
            match = pick_best_reading_match(
                reading.reading_text,
                reading.reader,
                detail_cache[ksm_idx],
            )
            if match and len(match.text_raw) > len(text_raw):
                text_raw = match.text_raw
                source_note = (
                    f"source=detail_augment_truncated | pan_idx={match.pan_idx} | "
                    f"reader={match.reader} | {match.source_url}"
                )
                augmented_count += 1

        processed = preprocess_reading_text(
            text_raw,
            source_note=f"{source_note} | detail_url={client.detail_url(ksm_idx)}",
        )
        hangul_replaced_total += processed.hangul_replaced
        for marker in processed.observed_markers:
            marker_stats[marker] = marker_stats.get(marker, 0) + 1

        pan_suffix = f"x{reading.xls_seq}"
        rows.append(
            {
                "reading_id": _make_reading_id(ksm_idx, pan_suffix),
                "source": "nrich",
                "source_record_id": str(ksm_idx),
                "inscription_id": f"insc_nrich_{int(ksm_idx):06d}",
                "artifact_id": "",
                "title": reading.title_ko,
                "era": meta.era if meta else "",
                "year": meta.year if meta else "",
                "material": "",
                "location": meta.location if meta else "",
                "text_raw": processed.text_raw,
                "text_normalized": processed.text_normalized,
                "text_length": processed.text_length,
                "hanja_count": processed.hanja_count,
                "has_real_defect": processed.has_real_defect,
                "defect_count": processed.defect_count,
                "defect_markers_raw": processed.defect_markers_raw,
                "is_training_text": processed.is_training_text,
                "exclude_reason": processed.exclude_reason,
                "quality_flag": processed.quality_flag,
                "notes": processed.notes,
            }
        )

    inventory = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "preprocess_rule": PREPROCESS_RULE_VERSION,
        "normalized_defect_symbol": "□",
        "records_processed": len(rows),
        "augmented_from_detail": augmented_count,
        "unmapped_title_count": len(unmapped_titles),
        "observed_markers": sorted(marker_stats.keys()),
        "normalized_defect_total": int(
            sum(row["defect_count"] for row in rows)
        ),
        "hangul_replaced_total": hangul_replaced_total,
        "marker_catalog": [
            {
                "marker": marker,
                "occurrence_count": count,
                "normalized_to": "□",
            }
            for marker, count in sorted(marker_stats.items())
        ],
    }
    return pd.DataFrame(rows, columns=READINGS_COLUMNS), inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xls",
        type=Path,
        default=Path("data/금석문 사용자 검색_2026-07-03.xls"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed_layer/readings.csv"),
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("data/processed_layer/defect_markers_inventory.json"),
    )
    parser.add_argument(
        "--title-index",
        type=Path,
        default=Path("data/interim/title_to_ksm_idx.json"),
    )
    parser.add_argument(
        "--index-max-pages",
        type=int,
        default=None,
        help="Limit list/API pages while building title index (None = all).",
    )
    parser.add_argument(
        "--augment-detail",
        action="store_true",
        help="Replace XLS rows with longer detail-page text (all inscriptions).",
    )
    parser.add_argument(
        "--augment-truncated-only",
        action="store_true",
        help="Fetch detail pages only for long XLS rows likely truncated.",
    )
    parser.add_argument(
        "--augment-min-xls-len",
        type=int,
        default=5000,
        help="Minimum XLS length before truncated-only augment runs.",
    )
    args = parser.parse_args()

    title_index = _load_title_index(args.title_index, args.index_max_pages)
    df, inventory = build_readings_dataframe(
        args.xls,
        title_index,
        augment_from_detail=args.augment_detail,
        augment_truncated_only=args.augment_truncated_only,
        augment_min_xls_len=args.augment_min_xls_len,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8")
    args.inventory.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(df)} readings -> {args.output}")
    print(f"Training texts: {int(df['is_training_text'].sum())}")
    print(f"Inventory -> {args.inventory}")


if __name__ == "__main__":
    main()
