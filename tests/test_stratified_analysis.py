"""Tests for stratified analysis."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.eval.stratified_analysis import build_stratified_report


class StratifiedAnalysisTests(unittest.TestCase):
    def test_build_stratified_report_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask_csv = root / "mask.csv"
            pred_csv = root / "pred.csv"
            mask_rows = [
                {
                    "sample_id": "mask_1",
                    "split": "train",
                    "label_char": "日",
                    "era": "조선",
                    "original_text": "嘉慶二十五年庚辰四月日",
                    "mask_position": 10,
                },
                {
                    "sample_id": "mask_2",
                    "split": "test",
                    "label_char": "日",
                    "era": "조선",
                    "original_text": "嘉慶二十五年庚辰四月日",
                    "mask_position": 10,
                },
                {
                    "sample_id": "mask_3",
                    "split": "test",
                    "label_char": "Rare",
                    "era": "고려",
                    "original_text": "ABCRareD",
                    "mask_position": 3,
                },
            ]
            pd.DataFrame(mask_rows).to_csv(mask_csv, index=False)
            pred_rows = [
                {
                    "sample_id": "mask_2",
                    "top1_char": "日",
                    "top10_chars": "日;月",
                },
                {
                    "sample_id": "mask_3",
                    "top1_char": "X",
                    "top10_chars": "X;Y",
                },
            ]
            with pred_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=pred_rows[0].keys())
                writer.writeheader()
                writer.writerows(pred_rows)

            report = build_stratified_report(
                pred_csv,
                mask_csv,
                model_key="test_model",
                split="test",
            )["test_model"]
            self.assertEqual(report["overall"]["num_samples"], 2)
            self.assertEqual(report["overall"]["exact_at_1"], 0.5)
            self.assertIn("very_common (100+)", report["by_freq_bucket"])


if __name__ == "__main__":
    unittest.main()
