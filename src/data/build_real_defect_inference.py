"""Build real defect inference dataset from readings with □ markers."""

from __future__ import annotations

import argparse
import csv
import random
from datetime import date
from pathlib import Path

import pandas as pd

from src.preprocess.gsm_v4 import NORMALIZED_DEFECT_SYMBOL, NORMALIZED_HANGUL_SYMBOL

CASE_COLUMNS = [
    "case_id",
    "reading_id",
    "source",
    "source_record_id",
    "inscription_id",
    "artifact_id",
    "title",
    "era",
    "text_with_defect",
    "model_input",
    "defect_marker_raw",
    "defect_position",
    "left_context",
    "right_context",
    "has_hangul_context",
    "gold_label",
    "usage",
    "notes",
]

CONTEXT_WINDOW = 64
HANGUL_MARKER_NOTE = "[hangul->"


def _first_defect_marker(defect_markers_raw: str) -> str:
    for part in str(defect_markers_raw or "□").split(";"):
        part = part.strip()
        if part and not part.startswith(HANGUL_MARKER_NOTE):
            return part
    return "□"


def _defect_positions(text: str) -> list[int]:
    return [idx for idx, ch in enumerate(text) if ch == NORMALIZED_DEFECT_SYMBOL]


def _strip_hangul_markers(text: str) -> str:
    return text.replace(NORMALIZED_HANGUL_SYMBOL, "")


def _map_defect_position(text: str, pos: int) -> int:
    """Map a □ index in text to the index after removing ◇ markers."""
    return pos - text[:pos].count(NORMALIZED_HANGUL_SYMBOL)


def _eligible_defect_readings(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["has_real_defect"] == True) & (df["defect_count"] > 0)].copy()  # noqa: E712


def _case_row(reading: pd.Series, pos: int, case_num: int) -> dict:
    text = str(reading["text_normalized"])
    marker = _first_defect_marker(str(reading.get("defect_markers_raw", "") or "□"))
    has_hangul_context = NORMALIZED_HANGUL_SYMBOL in text
    stripped = _strip_hangul_markers(text)
    stripped_pos = _map_defect_position(text, pos)
    left = stripped[max(0, stripped_pos - CONTEXT_WINDOW) : stripped_pos]
    right = stripped[stripped_pos + 1 : stripped_pos + 1 + CONTEXT_WINDOW]
    notes = "real defect □ only; no gold label"
    if has_hangul_context:
        notes += "; hangul ◇ stripped from model_input/context"
    return {
        "case_id": f"real_{case_num:06d}",
        "reading_id": reading["reading_id"],
        "source": reading["source"],
        "source_record_id": reading["source_record_id"],
        "inscription_id": reading["inscription_id"],
        "artifact_id": reading.get("artifact_id", ""),
        "title": reading.get("title", ""),
        "era": reading.get("era", ""),
        "text_with_defect": text,
        "model_input": stripped[:stripped_pos] + "[MASK]" + stripped[stripped_pos + 1 :],
        "defect_marker_raw": marker,
        "defect_position": pos,
        "left_context": left,
        "right_context": right,
        "has_hangul_context": has_hangul_context,
        "gold_label": "",
        "usage": "demo_only",
        "notes": notes,
    }


def build_cases_streaming(
    readings_csv: Path,
    output_csv: Path,
    *,
    max_cases: int | None = 5000,
    seed: int = 42,
    max_per_reading: int = 2,
) -> int:
    """Stream cases to CSV to avoid loading all defect positions in memory."""
    df = pd.read_csv(readings_csv)
    defect_df = _eligible_defect_readings(df)
    readings = [row for _, row in defect_df.iterrows()]
    rng = random.Random(seed)
    rng.shuffle(readings)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    case_num = 1
    written = 0
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CASE_COLUMNS)
        writer.writeheader()
        for reading in readings:
            text = str(reading["text_normalized"])
            positions = _defect_positions(text)
            if not positions:
                continue
            if len(positions) > max_per_reading:
                positions = rng.sample(positions, max_per_reading)
            for pos in positions:
                writer.writerow(_case_row(reading, pos, case_num))
                case_num += 1
                written += 1
                if max_cases is not None and written >= max_cases:
                    return written
    return written


def build_cases(
    readings_csv: Path,
    *,
    max_cases: int | None = 5000,
    seed: int = 42,
    max_per_reading: int = 2,
) -> pd.DataFrame:
    rows: list[dict] = []
    df = pd.read_csv(readings_csv)
    defect_df = _eligible_defect_readings(df)
    readings = [row for _, row in defect_df.iterrows()]
    rng = random.Random(seed)
    rng.shuffle(readings)
    case_num = 1
    for reading in readings:
        text = str(reading["text_normalized"])
        positions = _defect_positions(text)
        if not positions:
            continue
        if len(positions) > max_per_reading:
            positions = rng.sample(positions, max_per_reading)
        for pos in positions:
            rows.append(_case_row(reading, pos, case_num))
            case_num += 1
            if max_cases is not None and len(rows) >= max_cases:
                return pd.DataFrame(rows, columns=CASE_COLUMNS)
    return pd.DataFrame(rows, columns=CASE_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--readings",
        type=Path,
        default=Path("data/processed_layer/readings.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/model_inputs/real_defect_inference.csv"),
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=5000,
        help="Cap exported cases (default 5000 pool; 0 = no cap, streamed).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-reading", type=int, default=2)
    args = parser.parse_args()

    max_cases = None if args.max_cases == 0 else args.max_cases
    if max_cases is None or max_cases > 10000:
        count = build_cases_streaming(
            args.readings,
            args.output,
            max_cases=max_cases,
            seed=args.seed,
            max_per_reading=args.max_per_reading,
        )
    else:
        cases = build_cases(
            args.readings,
            max_cases=max_cases,
            seed=args.seed,
            max_per_reading=args.max_per_reading,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cases.to_csv(args.output, index=False, encoding="utf-8")
        count = len(cases)
    print(f"Wrote {count} cases to {args.output} ({date.today().isoformat()})")


if __name__ == "__main__":
    main()
