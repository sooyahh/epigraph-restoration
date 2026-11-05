"""Collect NRICH pandok readings via Open API list -> detail -> pandok view."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.nrich.portal_client import ListRecord, NrichClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_READINGS_DIR = PROJECT_ROOT / "data" / "raw_layer" / "nrich" / "readings"
MANIFEST_PATH = PROJECT_ROOT / "data" / "raw_layer" / "nrich" / "collect_manifest.json"


@dataclass
class CollectStats:
    requested: int = 0
    collected: int = 0
    skipped_non_pandok: int = 0
    errors: int = 0


def raw_json_path(ksm_idx: str, pan_idx: str) -> Path:
    return RAW_READINGS_DIR / f"ksm_{ksm_idx}_pan_{pan_idx}.json"


def write_raw_record(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)


def collect_for_list_record(client: NrichClient, list_record: ListRecord, stats: CollectStats) -> list[Path]:
    written: list[Path] = []
    stats.requested += 1
    try:
        entries = client.fetch_pandok_list(list_record.ksm_idx)
    except Exception as exc:
        stats.errors += 1
        print(f"ERROR list ksm_idx={list_record.ksm_idx}: {exc}")
        return written

    pandok_entries = [entry for entry in entries if entry.doc_type == "판독문"]
    if not pandok_entries:
        stats.skipped_non_pandok += 1
        return written

    for entry in pandok_entries:
        try:
            text_raw = client.fetch_pandok_text(list_record.ksm_idx, entry.pan_idx)
        except Exception as exc:
            stats.errors += 1
            print(
                f"ERROR pandok ksm_idx={list_record.ksm_idx} pan_idx={entry.pan_idx}: {exc}"
            )
            continue
        if not text_raw.strip():
            stats.errors += 1
            continue

        output_path = raw_json_path(list_record.ksm_idx, entry.pan_idx)
        payload = {
            "source": "nrich",
            "ksm_idx": list_record.ksm_idx,
            "pan_idx": entry.pan_idx,
            "reader": entry.reader,
            "title": list_record.title_ko,
            "title_hanja": list_record.title_hanja,
            "era": list_record.era,
            "year": list_record.year,
            "location": list_record.location,
            "text_raw": text_raw,
            "source_url": client.pandok_url(list_record.ksm_idx, entry.pan_idx),
            "detail_url": list_record.detail_url,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "collection_path": "open_api_list->ksmUsrView->ksmPandokView",
        }
        write_raw_record(payload, output_path)
        written.append(output_path)
        stats.collected += 1
    return written


def collect_by_ksm_indices(client: NrichClient, ksm_indices: list[int], stats: CollectStats) -> None:
    for ksm_idx in ksm_indices:
        meta = client.fetch_detail_meta(ksm_idx)
        list_record = ListRecord(
            ksm_idx=str(ksm_idx),
            title_ko=meta["title"],
            title_hanja="",
            era="",
            year="",
            location="",
            detail_url=meta["detail_url"],
            raw_fields={},
        )
        collect_for_list_record(client, list_record, stats)


def collect_via_open_api(
    client: NrichClient,
    max_pages: int | None,
    num_of_rows: int,
    stats: CollectStats,
) -> None:
    for list_record in client.iter_list_records(max_pages=max_pages, num_of_rows=num_of_rows):
        collect_for_list_record(client, list_record, stats)


def write_manifest(
    mode: str,
    stats: CollectStats,
    extra: dict,
    manifest_path: Path = MANIFEST_PATH,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stats": asdict(stats),
        **extra,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-key", default=os.getenv("NRICH_SERVICE_KEY", ""))
    parser.add_argument("--ksm-idx", type=int, action="append", dest="ksm_indices")
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--num-of-rows", type=int, default=100)
    parser.add_argument("--delay-sec", type=float, default=0.35)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Collect a small verification set (ksm 3,5,10) without Open API.",
    )
    args = parser.parse_args()

    client = NrichClient(
        service_key=args.service_key or None,
        delay_sec=args.delay_sec,
    )
    stats = CollectStats()

    if args.verify_only or args.ksm_indices:
        indices = args.ksm_indices or [3, 5, 10]
        collect_by_ksm_indices(client, indices, stats)
        write_manifest("verify_by_ksm_idx", stats, {"ksm_indices": indices})
    else:
        collect_via_open_api(client, args.max_pages, args.num_of_rows, stats)
        write_manifest(
            "open_api_full",
            stats,
            {"max_pages": args.max_pages, "num_of_rows": args.num_of_rows},
        )

    print(
        f"Collected {stats.collected} pandok files "
        f"(requested={stats.requested}, errors={stats.errors})"
    )
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
