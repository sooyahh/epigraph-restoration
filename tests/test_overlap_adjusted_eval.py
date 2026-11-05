"""Tests for overlap-adjusted evaluation."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.audit_ngram_leakage import build_per_sample_overlap_flags
from scripts.overlap_adjusted_eval import evaluate_predictions_with_overlap


class OverlapAdjustedEvalTests(unittest.TestCase):
    def test_evaluate_subsets(self) -> None:
        flags = pd.DataFrame(
            [
                {"sample_id": "s1", "has_train_2gram_match": True, "label_char": "天"},
                {"sample_id": "s2", "has_train_2gram_match": False, "label_char": "地"},
                {"sample_id": "s3", "has_train_2gram_match": False, "label_char": "人"},
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            pred_path = Path(tmp) / "pred.csv"
            with pred_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["sample_id", "label_char", "top10_chars"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"sample_id": "s1", "label_char": "天", "top10_chars": "天;地"},
                        {"sample_id": "s2", "label_char": "地", "top10_chars": "人;天"},
                        {"sample_id": "s3", "label_char": "人", "top10_chars": "人;天"},
                    ]
                )
            rows = evaluate_predictions_with_overlap(
                pred_path,
                model_label="test",
                flags_df=flags,
            )
        by_subset = {row["subset"]: row for row in rows}
        self.assertEqual(by_subset["all"]["num_samples"], 3)
        self.assertEqual(by_subset["overlap"]["num_samples"], 1)
        self.assertEqual(by_subset["no_overlap"]["num_samples"], 2)
        self.assertAlmostEqual(by_subset["all"]["exact_at_1"], 2 / 3)
        self.assertAlmostEqual(by_subset["overlap"]["exact_at_1"], 1.0)
        self.assertAlmostEqual(by_subset["no_overlap"]["exact_at_1"], 0.5)

    def test_build_flags_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mask_path = Path(tmp) / "mask.csv"
            pd.DataFrame(
                [
                    {
                        "sample_id": "t1",
                        "split": "train",
                        "original_text": "天地人",
                        "mask_position": 1,
                        "label_char": "地",
                    },
                    {
                        "sample_id": "e1",
                        "split": "test",
                        "original_text": "天地人",
                        "mask_position": 1,
                        "label_char": "地",
                    },
                ]
            ).to_csv(mask_path, index=False)
            flags = build_per_sample_overlap_flags(mask_path, split="test")
            self.assertEqual(len(flags), 1)
            self.assertIn("has_train_2gram_match", flags.columns)


if __name__ == "__main__":
    unittest.main()
