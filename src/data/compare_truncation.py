"""Compare XLS reading length vs NRICH detail-page full text."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.nrich.detail_client import NrichDetailClient, pick_best_reading_match
from src.nrich.xls_parser import load_reading_rows


DEFAULT_SAMPLES = [
    {"ksm_idx": "3", "title_ko": "이계신도비", "reader": "『가평군 금석문집』"},
    {"ksm_idx": "3", "title_ko": "이계신도비", "reader": "김혜완"},
    {"ksm_idx": "5", "title_ko": "이소한신도비", "reader": "김은미"},
    {"ksm_idx": "1739", "title_ko": "가락국 수로왕릉비", "reader": ""},
    {"ksm_idx": "8722", "title_ko": "가락국 태조릉숭선전비", "reader": ""},
    {"ksm_idx": "9216", "title_ko": "가상부귀명 사유팔금문 경", "reader": ""},
    {"ksm_idx": "9426", "title_ko": "가갸·거겨명 철사 죽문 병", "reader": ""},
    {"ksm_idx": "2866", "title_ko": "경주 황룡사 구층목탑 금동찰주본기", "reader": "정병삼"},
]


def _find_xls_row(readings, title_ko: str, reader: str):
    for row in readings:
        if row.title_ko != title_ko:
            continue
        if reader and row.reader != reader:
            continue
        return row
    for row in readings:
        if row.title_ko == title_ko:
            return row
    return None


def run_compare(
    xls_path: Path,
    samples: list[dict],
    report_json: Path,
    report_md: Path,
) -> pd.DataFrame:
    xls_readings = load_reading_rows(xls_path)
    client = NrichDetailClient()
    rows: list[dict] = []

    for sample in samples:
        ksm_idx = str(sample["ksm_idx"])
        title_ko = sample["title_ko"]
        reader = sample.get("reader", "")
        xls_row = _find_xls_row(xls_readings, title_ko, reader)
        detail_readings = client.fetch_pandok_readings(ksm_idx)
        detail_match = None
        if xls_row:
            detail_match = pick_best_reading_match(
                xls_row.reading_text,
                xls_row.reader,
                detail_readings,
            )
        elif detail_readings:
            detail_match = max(detail_readings, key=lambda item: len(item.text_raw))

        xls_len = len(xls_row.reading_text) if xls_row else 0
        detail_len = len(detail_match.text_raw) if detail_match else 0
        delta = detail_len - xls_len
        truncated = delta > 0

        rows.append(
            {
                "ksm_idx": ksm_idx,
                "title_ko": title_ko,
                "reader": reader or (xls_row.reader if xls_row else ""),
                "xls_len": xls_len,
                "detail_len": detail_len,
                "delta": delta,
                "truncated_in_xls": truncated,
                "detail_pan_idx": detail_match.pan_idx if detail_match else "",
                "detail_reader": detail_match.reader if detail_match else "",
                "detail_url": client.detail_url(ksm_idx),
            }
        )

    df = pd.DataFrame(rows)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "xls_path": str(xls_path),
        "sample_count": len(rows),
        "truncated_count": int(df["truncated_in_xls"].sum()),
        "rows": rows,
    }
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# XLS vs detail-page truncation report",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- XLS: `{xls_path}`",
        f"- Samples: {payload['sample_count']}",
        f"- Truncated in XLS: {payload['truncated_count']}",
        "",
        "| ksm_idx | title | reader | xls_len | detail_len | delta | truncated |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['ksm_idx']} | {row['title_ko']} | {row['reader']} | "
            f"{row['xls_len']} | {row['detail_len']} | {row['delta']} | "
            f"{'yes' if row['truncated_in_xls'] else 'no'} |"
        )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xls",
        type=Path,
        default=Path("data/금석문 사용자 검색_2026-07-03.xls"),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/xls_detail_truncation_compare.json"),
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=Path("reports/xls_detail_truncation_compare.md"),
    )
    args = parser.parse_args()
    df = run_compare(args.xls, DEFAULT_SAMPLES, args.report_json, args.report_md)
    print(df.to_string(index=False))
    print(f"\nWrote {args.report_json} and {args.report_md}")


if __name__ == "__main__":
    main()
