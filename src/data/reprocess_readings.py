"""Re-apply preprocess rules to an existing readings.csv (text_raw unchanged)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.preprocess.gsm_v4 import (
    PREPROCESS_RULE_VERSION,
    preprocess_reading_text,
    strip_preprocess_notes,
)

PROCESSED_COLUMNS = [
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


def reprocess_readings_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    marker_stats: dict[str, int] = {}
    hangul_replaced_total = 0
    out = df.copy()
    rows: list[dict] = []

    for _, row in df.iterrows():
        source_note = strip_preprocess_notes(str(row.get("notes", "") or ""))
        processed = preprocess_reading_text(str(row["text_raw"]), source_note=source_note)
        hangul_replaced_total += processed.hangul_replaced
        for marker in processed.observed_markers:
            marker_stats[marker] = marker_stats.get(marker, 0) + 1
        rows.append(
            {
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

    for col in PROCESSED_COLUMNS:
        out[col] = [row[col] for row in rows]

    inventory = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "preprocess_rule": PREPROCESS_RULE_VERSION,
        "normalized_defect_symbol": "□",
        "normalized_hangul_symbol": "◇",
        "records_processed": len(out),
        "reprocessed_from_csv": True,
        "observed_markers": sorted(marker_stats.keys()),
        "normalized_defect_total": int(out["defect_count"].sum()),
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
    return out, inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed_layer/readings.csv"),
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
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out, inventory = reprocess_readings_dataframe(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8")
    args.inventory.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Reprocessed {len(out)} readings -> {args.output}")
    print(f"Training texts: {int(out['is_training_text'].sum())}")
    print(f"Real defects: {int(out['has_real_defect'].sum())}")
    print(f"Inventory -> {args.inventory}")


if __name__ == "__main__":
    main()
