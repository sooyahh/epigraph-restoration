"""Aggregate per-experiment evaluation summaries into a single baseline table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

SUMMARY_COLUMNS = [
    "experiment_id",
    "experiment_name",
    "model_name",
    "dataset_name",
    "split",
    "num_samples",
    "exact_at_1",
    "exact_at_5",
    "exact_at_10",
    "group_at_1",
    "group_at_5",
    "group_at_10",
    "created_at",
    "notes",
]

DEFAULT_INPUTS = [
    Path("reports/evaluation/exp003_freq_test_summary.csv"),
    Path("reports/evaluation/exp004_ngram_test_summary.csv"),
    Path("reports/evaluation/exp006_bert_test_summary.csv"),
    Path("reports/evaluation/exp007_sillokbert_test_summary.csv"),
]


def _overall_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if "[" not in row.get("experiment_name", "")]


def aggregate_summaries(
    inputs: list[Path],
    output_csv: Path,
) -> int:
    merged: list[dict] = []
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(f"Missing summary CSV: {path}")
        overall = _overall_rows(path)
        if len(overall) != 1:
            raise RuntimeError(
                f"Expected exactly one overall row in {path}, found {len(overall)}"
            )
        merged.append(overall[0])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(merged)
    return len(merged)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="*",
        type=Path,
        default=DEFAULT_INPUTS,
        help="Per-experiment summary CSV paths (overall row only).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/evaluation/baseline_test_summary.csv"),
    )
    args = parser.parse_args()
    count = aggregate_summaries(args.inputs, args.output)
    print(f"Wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
