"""Spot-check XLS vs NRICH detail length for long training readings."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.nrich.detail_client import NrichDetailClient, pick_best_reading_match
from src.nrich.xls_parser import load_reading_rows


def _reader_from_notes(notes: str) -> str:
    match = re.search(r"reader=([^|]+)", notes or "")
    return match.group(1).strip() if match else ""


def select_spotcheck_targets(readings_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(readings_csv)
    notes = df["notes"].fillna("")
    aug = notes.str.contains("detail_augment", case=False)
    targets = df[
        (df["text_length"] >= 5000)
        & (df["is_training_text"] == True)  # noqa: E712
        & ~aug
    ].copy()
    targets["reader"] = targets["notes"].map(_reader_from_notes)
    return targets


def run_spotcheck(
    readings_csv: Path,
    xls_path: Path,
    report_json: Path,
    report_md: Path,
) -> pd.DataFrame:
    targets = select_spotcheck_targets(readings_csv)
    xls_readings = load_reading_rows(xls_path)
    client = NrichDetailClient()
    rows: list[dict] = []

    for _, row in targets.iterrows():
        ksm_idx = str(row["source_record_id"])
        title = str(row.get("title", ""))
        reader = str(row.get("reader", ""))
        xls_text = str(row.get("text_raw", row.get("text_normalized", "")))
        xls_len = len(xls_text)

        xls_row = None
        for candidate in xls_readings:
            if candidate.title_ko == title and (not reader or candidate.reader == reader):
                xls_row = candidate
                break
        if xls_row is None:
            for candidate in xls_readings:
                if candidate.title_ko == title:
                    xls_row = candidate
                    break

        detail_readings = client.fetch_pandok_readings(ksm_idx)
        detail_match = pick_best_reading_match(
            xls_row.reading_text if xls_row else xls_text,
            reader,
            detail_readings,
        )
        detail_len = len(detail_match.text_raw) if detail_match else 0
        xls_source_len = len(xls_row.reading_text) if xls_row else xls_len
        delta = detail_len - xls_source_len
        truncated = delta > 0

        rows.append(
            {
                "reading_id": row["reading_id"],
                "ksm_idx": ksm_idx,
                "title": title,
                "reader": reader,
                "readings_text_length": int(row["text_length"]),
                "xls_len": xls_source_len,
                "detail_len": detail_len,
                "delta": delta,
                "truncated_in_xls": truncated,
                "detail_pan_idx": detail_match.pan_idx if detail_match else "",
                "detail_url": client.detail_url(ksm_idx),
            }
        )

    df = pd.DataFrame(rows)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "readings_csv": str(readings_csv),
        "xls_path": str(xls_path),
        "sample_count": len(rows),
        "truncated_count": int(df["truncated_in_xls"].sum()) if len(df) else 0,
        "rows": rows,
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Truncation spot-check (long training readings)",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Targets: `is_training_text=True`, `text_length>=5000`, no `detail_augment` note",
        f"- Samples: {payload['sample_count']}",
        f"- Truncated in XLS (detail longer): {payload['truncated_count']}",
        "",
        "## Split / eval policy note",
        "",
        "- Quantitative eval uses `artificial_mask_dataset.csv` split column only.",
        "- `splits.csv` may include defect-only inscriptions; era breakdowns should filter",
        "  to inscriptions present in the mask dataset.",
        "",
        "| reading_id | ksm_idx | title | xls_len | detail_len | delta | truncated |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for item in rows:
        lines.append(
            f"| {item['reading_id']} | {item['ksm_idx']} | {item['title']} | "
            f"{item['xls_len']} | {item['detail_len']} | {item['delta']} | "
            f"{'yes' if item['truncated_in_xls'] else 'no'} |"
        )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--readings",
        type=Path,
        default=Path("data/processed_layer/readings.csv"),
    )
    parser.add_argument(
        "--xls",
        type=Path,
        default=Path("data/금석문 사용자 검색_2026-07-03.xls"),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/truncation_spotcheck_2026-07-03.json"),
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=Path("reports/truncation_spotcheck_2026-07-03.md"),
    )
    args = parser.parse_args()
    df = run_spotcheck(args.readings, args.xls, args.report_json, args.report_md)
    print(df.to_string(index=False))
    print(f"\nWrote {args.report_json} and {args.report_md}")


if __name__ == "__main__":
    main()
