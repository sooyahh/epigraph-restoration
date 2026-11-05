"""Inscription-level train/valid/test split."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

SPLIT_COLUMNS = [
    "split_unit_id",
    "split_unit_type",
    "split",
    "source",
    "era",
    "num_readings",
    "num_mask_samples",
    "seed",
    "split_rule",
    "created_at",
    "notes",
]

SPLIT_RULE = "group_split_by_inscription_v1"


def assign_inscription_splits(
    readings: pd.DataFrame,
    seed: int = 42,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
) -> pd.DataFrame:
    eligible = readings[readings["is_training_text"] == True]  # noqa: E712
    inscription_ids = sorted(eligible["inscription_id"].dropna().unique())
    if not inscription_ids:
        raise RuntimeError("No is_training_text inscriptions available for split.")

    rng = pd.Series(inscription_ids).sample(frac=1.0, random_state=seed).tolist()
    n = len(rng)
    train_cut = int(n * train_ratio)
    valid_cut = train_cut + int(n * valid_ratio)

    split_map: dict[str, str] = {}
    for idx, inscription_id in enumerate(rng):
        if idx < train_cut:
            split_map[inscription_id] = "train"
        elif idx < valid_cut:
            split_map[inscription_id] = "valid"
        else:
            split_map[inscription_id] = "test"

    rows: list[dict] = []
    for inscription_id in inscription_ids:
        group = readings[readings["inscription_id"] == inscription_id]
        rows.append(
            {
                "split_unit_id": inscription_id,
                "split_unit_type": "inscription_id",
                "split": split_map[inscription_id],
                "source": group["source"].iloc[0] if len(group) else "",
                "era": group["era"].iloc[0] if len(group) else "",
                "num_readings": int(len(group)),
                "num_mask_samples": "",
                "seed": seed,
                "split_rule": SPLIT_RULE,
                "created_at": date.today().isoformat(),
                "notes": "",
            }
        )
    return pd.DataFrame(rows, columns=SPLIT_COLUMNS)


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
        default=Path("data/model_inputs/splits.csv"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    readings = pd.read_csv(args.readings)
    df = assign_inscription_splits(readings, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(df["split"].value_counts().to_string())
    print(f"Wrote {len(df)} split rows -> {args.output}")


if __name__ == "__main__":
    main()
