"""Generate artificial single-char mask dataset from defect-free readings."""

from __future__ import annotations

import argparse
import random
from datetime import date
from pathlib import Path

import pandas as pd

from src.nrich.text_utils import is_cjk_ideograph
from src.data.split_by_inscription import assign_inscription_splits

MASK_COLUMNS = [
    "sample_id",
    "reading_id",
    "source",
    "source_record_id",
    "inscription_id",
    "artifact_id",
    "title",
    "era",
    "original_text",
    "masked_input",
    "label_char",
    "mask_position",
    "left_context",
    "right_context",
    "context_window",
    "mask_type",
    "split",
    "created_at",
    "generation_rule",
    "notes",
]

GENERATION_RULE = "single_hanja_mask_v1"
CONTEXT_WINDOW = 64


def _eligible_positions(text: str) -> list[int]:
    return [idx for idx, ch in enumerate(text) if is_cjk_ideograph(ch)]


def _masks_for_reading(
    row: pd.Series,
    split_value: str,
    rng: random.Random,
    max_masks_per_reading: int,
) -> list[dict]:
    text = row["text_normalized"]
    positions = _eligible_positions(text)
    if not positions:
        return []

    budget = min(
        max_masks_per_reading,
        max(1, len(positions) // 40),
    )
    chosen = rng.sample(positions, k=min(budget, len(positions)))
    samples: list[dict] = []
    for position in sorted(chosen):
        label = text[position]
        masked = text[:position] + "[MASK]" + text[position + 1 :]
        left = text[max(0, position - CONTEXT_WINDOW) : position]
        right = text[position + 1 : position + 1 + CONTEXT_WINDOW]
        samples.append(
            {
                "reading_id": row["reading_id"],
                "source": row["source"],
                "source_record_id": row["source_record_id"],
                "inscription_id": row["inscription_id"],
                "artifact_id": row.get("artifact_id", ""),
                "title": row.get("title", ""),
                "era": row.get("era", ""),
                "original_text": text,
                "masked_input": masked,
                "label_char": label,
                "mask_position": position,
                "left_context": left,
                "right_context": right,
                "context_window": CONTEXT_WINDOW,
                "mask_type": "single_char",
                "split": split_value,
                "created_at": date.today().isoformat(),
                "generation_rule": GENERATION_RULE,
                "notes": "",
            }
        )
    return samples


def build_artificial_mask_dataframe(
    readings: pd.DataFrame,
    splits: pd.DataFrame,
    seed: int = 42,
    max_masks_per_reading: int = 32,
) -> pd.DataFrame:
    rng = random.Random(seed)
    split_map = splits.set_index("split_unit_id")["split"].to_dict()
    rows: list[dict] = []
    sample_counter = 1

    training = readings[readings["is_training_text"] == True]  # noqa: E712
    for _, reading in training.iterrows():
        text = reading["text_normalized"]
        if not isinstance(text, str) or not text.strip():
            continue
        split_value = split_map.get(reading["inscription_id"], "train")
        for sample in _masks_for_reading(reading, split_value, rng, max_masks_per_reading):
            sample["sample_id"] = f"mask_{sample_counter:08d}"
            sample_counter += 1
            rows.append(sample)

    return pd.DataFrame(rows, columns=MASK_COLUMNS)


def validate_no_label_leak(df: pd.DataFrame) -> None:
    for _, row in df.iterrows():
        pos = int(row["mask_position"])
        text = row["original_text"]
        label = row["label_char"]
        masked = row["masked_input"]
        if text[pos] != label:
            raise ValueError(f"Label mismatch at {row['sample_id']}")
        reconstructed = masked.replace("[MASK]", label, 1)
        if reconstructed != text:
            raise ValueError(f"Mask reconstruction failed at {row['sample_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--readings",
        type=Path,
        default=Path("data/processed_layer/readings.csv"),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("data/model_inputs/splits.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/model_inputs/artificial_mask_dataset.csv"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-masks-per-reading", type=int, default=32)
    args = parser.parse_args()

    readings = pd.read_csv(args.readings)
    if not args.splits.exists():
        splits = assign_inscription_splits(readings, seed=args.seed)
        args.splits.parent.mkdir(parents=True, exist_ok=True)
        splits.to_csv(args.splits, index=False, encoding="utf-8")
    else:
        splits = pd.read_csv(args.splits)

    df = build_artificial_mask_dataframe(
        readings,
        splits,
        seed=args.seed,
        max_masks_per_reading=args.max_masks_per_reading,
    )
    validate_no_label_leak(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} mask samples -> {args.output}")


if __name__ == "__main__":
    main()
